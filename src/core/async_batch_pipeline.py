"""
Optimized Async Batch Processing Pipeline for FLUX Klein 4B

Features:
- Async image loading and preprocessing
- Dynamic batch sizing based on VRAM availability
- Pipeline warmup and reuse
- CUDA streams for overlapping operations
- Prefetch queue with configurable depth
- Progress callbacks and ETA estimation
"""

import asyncio
import os
import time
import contextlib
import inspect
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Any
from collections import deque
import threading

import torch
from PIL import Image
import numpy as np

from src.runtime_policies import should_enable_autocast

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_batch_size: int = 4
    prefetch_depth: int = 3
    num_workers: int = 4
    enable_cuda_streams: bool = True
    warmup_steps: int = 1
    dynamic_batching: bool = True
    vram_threshold_mb: int = 1024  # Reserve 1GB for safety


@dataclass
class ProcessingStats:
    """Statistics for batch processing."""
    total_images: int = 0
    processed_images: int = 0
    failed_images: int = 0
    start_time: float = 0.0
    total_time: float = 0.0
    avg_time_per_image: float = 0.0

    def eta_seconds(self) -> float:
        """Calculate estimated time remaining in seconds."""
        if self.processed_images == 0:
            return 0.0
        remaining = self.total_images - self.processed_images
        return self.avg_time_per_image * remaining

    def update(self, processing_time: float):
        """Update stats with new processing time."""
        self.processed_images += 1
        if self.processed_images == 1:
            self.avg_time_per_image = processing_time
        else:
            # Exponential moving average
            alpha = 0.3
            self.avg_time_per_image = (
                alpha * processing_time +
                (1 - alpha) * self.avg_time_per_image
            )


class AsyncImageLoader:
    """Async image loader with prefetching."""

    def __init__(self, num_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def load_image(self, path: str) -> Optional[Image.Image]:
        """Load single image synchronously."""
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    async def load_image_async(self, path: str) -> Tuple[str, Optional[Image.Image]]:
        """Load single image asynchronously."""
        # ``get_event_loop`` is deprecated when there is no running loop
        # (Python 3.10+); inside ``async def`` we always have one so the
        # documented replacement is ``get_running_loop``.
        loop = asyncio.get_running_loop()
        img = await loop.run_in_executor(self.executor, self.load_image, path)
        return path, img

    async def load_batch_async(
        self,
        paths: List[str]
    ) -> List[Tuple[str, Optional[Image.Image]]]:
        """Load multiple images concurrently."""
        tasks = [self.load_image_async(p) for p in paths]
        return await asyncio.gather(*tasks)

    def shutdown(self):
        """Shutdown executor."""
        self.executor.shutdown(wait=True)


class PrefetchQueue:
    """Queue with async prefetching."""

    def __init__(
        self,
        image_paths: List[Any],
        loader: AsyncImageLoader,
        batch_size: int,
        prefetch_depth: int
    ):
        self.image_paths = image_paths
        self.loader = loader
        self.batch_size = batch_size
        self.prefetch_depth = prefetch_depth
        self.current_idx = 0
        self.current_bucket_idx = 0
        self.current_idx_in_bucket = 0
        self.queue = deque(maxlen=prefetch_depth)
        self.lock = threading.Lock()
        self._prefetch_tasks: set = set()
        self._closed = False

    def get_next_batch_paths(self) -> List[str]:
        """Get paths for next batch, respecting bucket boundaries if provided."""
        with self.lock:
            if not self.image_paths:
                return []

            # Check if we have a list of buckets (List[List[str]])
            if isinstance(self.image_paths[0], list):
                while self.current_bucket_idx < len(self.image_paths):
                    bucket = self.image_paths[self.current_bucket_idx]
                    if not bucket:
                        self.current_bucket_idx += 1
                        continue

                    start = self.current_idx_in_bucket
                    end = min(start + self.batch_size, len(bucket))
                    batch_paths = bucket[start:end]

                    self.current_idx_in_bucket = end
                    if self.current_idx_in_bucket >= len(bucket):
                        self.current_bucket_idx += 1
                        self.current_idx_in_bucket = 0

                    return batch_paths
                return []
            else:
                # Standard flat list behavior
                start = self.current_idx
                end = min(start + self.batch_size, len(self.image_paths))
                batch_paths = self.image_paths[start:end]
                self.current_idx = end
                return batch_paths

    async def prefetch(self):
        """Prefetch images into queue."""
        if self._closed:
            return
        while len(self.queue) < self.prefetch_depth:
            batch_paths = self.get_next_batch_paths()
            if not batch_paths:
                break
            batch_data = await self.loader.load_batch_async(batch_paths)
            if batch_data:
                self.queue.append(batch_data)

    def _start_prefetch_task(self):
        if self._closed:
            return
        task = asyncio.create_task(self.prefetch())
        self._prefetch_tasks.add(task)
        task.add_done_callback(self._prefetch_tasks.discard)

    async def get_batch(self) -> Optional[List[Tuple[str, Optional[Image.Image]]]]:
        """Get next batch from queue."""
        if not self.queue and self.has_more():
            await self.prefetch()

        # Trigger prefetch for next batches in background
        self._start_prefetch_task()

        if self.queue:
            return self.queue.popleft()
        return None

    async def close(self):
        """Cancel and await background prefetch tasks."""
        self._closed = True
        if not self._prefetch_tasks:
            return
        tasks = list(self._prefetch_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._prefetch_tasks.clear()

    def has_more(self) -> bool:
        """Check if more batches available."""
        with self.lock:
            if not self.image_paths:
                return len(self.queue) > 0

            if isinstance(self.image_paths[0], list):
                has_remaining_buckets = self.current_bucket_idx < len(self.image_paths)
                return has_remaining_buckets or len(self.queue) > 0

            return self.current_idx < len(self.image_paths) or len(self.queue) > 0


class CUDAStreamManager:
    """Manager for CUDA streams to overlap operations."""

    def __init__(self, num_streams: int = 2):
        self.num_streams = num_streams
        self.streams = []
        if torch.cuda.is_available():
            self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.current_stream_idx = 0

    def get_next_stream(self) -> Optional[torch.cuda.Stream]:
        """Get next stream in round-robin fashion."""
        if not self.streams:
            return None
        stream = self.streams[self.current_stream_idx]
        self.current_stream_idx = (self.current_stream_idx + 1) % self.num_streams
        return stream

    def synchronize_all(self):
        """Wait for all streams to complete."""
        for stream in self.streams:
            stream.synchronize()


class DynamicBatchSizer:
    """Dynamic batch sizing based on available VRAM."""

    def __init__(
        self,
        initial_batch_size: int,
        min_batch_size: int = 1,
        max_batch_size: int = 8,
        vram_threshold_mb: int = 1024
    ):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.vram_threshold_mb = vram_threshold_mb
        self.oom_count = 0

    def get_available_vram_mb(self) -> float:
        """Get available VRAM in MB.

        Returns ``float('inf')`` on CPU-only hosts so the VRAM-based adjuster
        in :meth:`check_and_adjust` becomes a no-op (CPU has no VRAM, so a
        VRAM threshold is meaningless there and shouldn't force batch size to
        the minimum).
        """
        if torch.cuda.is_available():
            # Use mem_get_info for accurate free memory (handles fragmentation and other apps)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            return free_bytes / (1024 ** 2)
        elif torch.backends.mps.is_available():
            # MPS doesn't expose memory stats, use conservative estimate
            return 2048  # Assume 2GB available
        return float('inf')

    def check_and_adjust(self) -> int:
        """Check VRAM and adjust batch size if needed."""
        available_mb = self.get_available_vram_mb()

        if available_mb < self.vram_threshold_mb and self.current_batch_size > self.min_batch_size:
            # Reduce batch size
            self.current_batch_size = max(
                self.min_batch_size,
                self.current_batch_size // 2
            )
            print(f"  Reduced batch size to {self.current_batch_size} (VRAM: {available_mb:.0f}MB)")
        elif available_mb > self.vram_threshold_mb * 3 and self.current_batch_size < self.max_batch_size:
            # Increase batch size
            self.current_batch_size = min(
                self.max_batch_size,
                self.current_batch_size + 1
            )
            print(f"  Increased batch size to {self.current_batch_size} (VRAM: {available_mb:.0f}MB)")

        return self.current_batch_size

    def handle_oom(self):
        """Handle OOM error by reducing batch size."""
        self.oom_count += 1
        if self.current_batch_size > self.min_batch_size:
            self.current_batch_size = max(
                self.min_batch_size,
                self.current_batch_size // 2
            )
            print(f"  OOM detected! Reduced batch size to {self.current_batch_size}")


class AsyncBatchPipeline:
    """Async batch processing pipeline for FLUX Klein."""

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.loader = AsyncImageLoader(num_workers=self.config.num_workers)
        self.stream_manager = None
        if self.config.enable_cuda_streams and torch.cuda.is_available():
            self.stream_manager = CUDAStreamManager(num_streams=2)
        self.stats = ProcessingStats()
        self.pipeline_warmed_up = False
        # Periodic allocator reset (every 10 batches by default, not every batch)
        self.batches_since_reset = 0
        self.reset_interval_batches = 10

    def warmup_pipeline(
        self,
        pipe: Any,
        device: str,
        sample_image: Image.Image,
        height: int,
        width: int,
        steps: int
    ):
        """Warm up the pipeline with dummy runs."""
        if self.pipeline_warmed_up or pipe is None:
            self.pipeline_warmed_up = True # Mark as warmed up even if pipe is None (mocking)
            return

        print(f"  Warming up pipeline ({self.config.warmup_steps} steps)...")

        for i in range(self.config.warmup_steps):
            try:
                with torch.inference_mode():
                    warmup_kwargs = {
                        "prompt": "warmup",
                        "height": height,
                        "width": width,
                        "num_inference_steps": steps,
                        "guidance_scale": 0.0,
                    }
                    try:
                        supported_params = inspect.signature(pipe.__call__).parameters
                    except (TypeError, ValueError):
                        supported_params = {}

                    if "image" in supported_params:
                        warmup_kwargs["image"] = sample_image.resize((width, height))

                    _ = pipe(**warmup_kwargs)

            except Exception as e:
                print(f"  Warmup step {i+1} failed: {e}")

        # One-time cache clear after warmup (not per step)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()

        self.pipeline_warmed_up = True
        print("  Pipeline warmup complete")

    async def process_batch_async(
        self,
        image_paths: List[Any],
        pipe: Any,
        device: str,
        process_fn: Callable,
        output_folder: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        **process_kwargs
    ) -> ProcessingStats:
        """
        Process images in async batches.

        Args:
            image_paths: List of image file paths or List of buckets (List[List[str]])
            pipe: Pipeline object
            device: Device to use ('cuda', 'mps', 'cpu')
            process_fn: Function to process each image
            output_folder: Output directory
            progress_callback: Progress callback function
            **process_kwargs: Additional kwargs for process_fn
        """
        if not image_paths:
            return self.stats

        # Calculate total images correctly for both flat and bucketed input
        if isinstance(image_paths[0], list):
            self.stats.total_images = sum(len(bucket) for bucket in image_paths)
        else:
            self.stats.total_images = len(image_paths)

        # Reset statistics for this run
        self.stats.processed_images = 0
        self.stats.failed_images = 0
        self.stats.start_time = time.time()
        self.stats.total_time = 0.0
        self.stats.avg_time_per_image = 0.0

        # Dynamic batch sizer
        batch_sizer = DynamicBatchSizer(
            initial_batch_size=self.config.max_batch_size,
            min_batch_size=1,
            max_batch_size=self.config.max_batch_size,
            vram_threshold_mb=self.config.vram_threshold_mb
        )

        # Create prefetch queue
        current_batch_size = batch_sizer.current_batch_size
        prefetch_queue = PrefetchQueue(
            image_paths=image_paths,
            loader=self.loader,
            batch_size=current_batch_size,
            prefetch_depth=self.config.prefetch_depth
        )

        # Warmup with first image
        if not self.pipeline_warmed_up and image_paths:
            warmup_path = image_paths[0][0] if isinstance(image_paths[0], list) else image_paths[0]
            first_img = self.loader.load_image(warmup_path)
            if first_img:
                self.warmup_pipeline(
                    pipe=pipe,
                    device=device,
                    sample_image=first_img,
                    height=process_kwargs.get('height', 1024),
                    width=process_kwargs.get('width', 1024),
                    steps=process_kwargs.get('steps', 4)
                )

        processed_count = 0

        try:
            # Process batches
            while prefetch_queue.has_more():
                # Adjust batch size dynamically
                if self.config.dynamic_batching:
                    new_batch_size = batch_sizer.check_and_adjust()
                    if new_batch_size != current_batch_size:
                        current_batch_size = new_batch_size
                        prefetch_queue.batch_size = current_batch_size

                # Get next batch
                batch_data = await prefetch_queue.get_batch()
                if not batch_data:
                    break

                # Filter out failed loads
                valid_batch = [(path, img) for path, img in batch_data if img is not None]
                failed_count = len(batch_data) - len(valid_batch)
                if failed_count > 0:
                    self.stats.failed_images += failed_count
                if not valid_batch:
                    continue

                # Unpack batch
                batch_paths = [p for p, i in valid_batch]
                batch_images = [i for p, i in valid_batch]

                start_time = time.time()

                try:
                    # Use CUDA stream if available (for whole batch)
                    stream = None
                    if self.stream_manager:
                        stream = self.stream_manager.get_next_stream()

                    # Optimized context for RTX 3070: Use BFloat16 if supported, else Float16.
                    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
                    def autocast_factory():
                        # ``should_enable_autocast`` matches ``model_key``
                        # against known SDNQ/FLUX tags; falling back to
                        # ``str(type(pipe))`` never matches and only added
                        # noise. When neither ``model_choice`` nor
                        # ``model_key`` is supplied, let the helper use its
                        # pipe-based fallback by passing ``None`` explicitly.
                        model_key = (
                            process_kwargs.get("model_choice")
                            or process_kwargs.get("model_key")
                            or None
                        )
                        if device == "cuda" and should_enable_autocast(device, model_key, pipe):
                            return torch.autocast(device_type="cuda", dtype=dtype)
                        return contextlib.nullcontext()

                    if stream:
                        with torch.cuda.stream(stream), autocast_factory():
                            results = await process_fn(
                                paths=batch_paths,
                                images=batch_images,
                                pipe=pipe,
                                device=device,
                                output_folder=output_folder,
                                **process_kwargs
                            )
                    else:
                        with autocast_factory():
                            results = await process_fn(
                                paths=batch_paths,
                                images=batch_images,
                                pipe=pipe,
                                device=device,
                                output_folder=output_folder,
                                **process_kwargs
                            )
                
                    # Update stats
                    processing_time = time.time() - start_time
                    # Average time per image in this batch
                    avg_time = processing_time / len(valid_batch)
                    
                    for _ in valid_batch:
                        self.stats.update(avg_time)
                    
                    processed_count += len(valid_batch)

                    # Progress callback
                    if progress_callback:
                        progress = processed_count / self.stats.total_images
                        eta = self.stats.eta_seconds()
                        eta_str = f"ETA: {int(eta//60)}m {int(eta%60)}s"
                        progress_callback(
                            progress,
                            f"Processed {processed_count}/{self.stats.total_images} - {eta_str}"
                        )

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        batch_sizer.handle_oom()
                        if hasattr(pipe, "enable_vae_slicing"):
                            try:
                                pipe.enable_vae_slicing()
                                pipe._ufig_memory_policy_dirty = True
                                print("  OOM fallback: enabled VAE slicing for retry.")
                            except Exception:
                                pass
                        elif hasattr(getattr(pipe, "vae", None), "enable_slicing"):
                            try:
                                pipe.vae.enable_slicing()
                                print("  OOM fallback: enabled VAE slicing for retry.")
                            except Exception:
                                pass
                        # Clear cache and retry with smaller batch
                        if device == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        elif device == "mps":
                            torch.mps.empty_cache()
                            torch.mps.synchronize()

                        # Deterministic single retry: split batch in half and retry serially.
                        if len(valid_batch) > 1:
                            midpoint = max(1, len(valid_batch) // 2)
                            retry_chunks = [valid_batch[:midpoint], valid_batch[midpoint:]]
                        else:
                            retry_chunks = [valid_batch]

                        retried_processed = 0
                        failed_chunks = []
                        for chunk in retry_chunks:
                            if not chunk:
                                continue
                            chunk_paths = [p for p, _ in chunk]
                            chunk_images = [i for _, i in chunk]
                            try:
                                with autocast_factory():
                                    _ = await process_fn(
                                        paths=chunk_paths,
                                        images=chunk_images,
                                        pipe=pipe,
                                        device=device,
                                        output_folder=output_folder,
                                        **process_kwargs
                                    )
                                retried_processed += len(chunk)
                            except RuntimeError as retry_err:
                                if "out of memory" in str(retry_err).lower():
                                    failed_chunks.append(chunk)
                                else:
                                    raise retry_err
                            except Exception:
                                failed_chunks.append(chunk)

                        if failed_chunks and hasattr(pipe, "enable_attention_slicing"):
                            try:
                                pipe.enable_attention_slicing()
                                pipe._ufig_memory_policy_dirty = True
                                print("  OOM fallback: enabled attention slicing for final retry.")
                            except Exception:
                                pass
                            if device == "cuda":
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            elif device == "mps":
                                torch.mps.empty_cache()
                                torch.mps.synchronize()

                            remaining_failed_chunks = []
                            for chunk in failed_chunks:
                                if not chunk:
                                    continue
                                chunk_paths = [p for p, _ in chunk]
                                chunk_images = [i for _, i in chunk]
                                try:
                                    with autocast_factory():
                                        _ = await process_fn(
                                            paths=chunk_paths,
                                            images=chunk_images,
                                            pipe=pipe,
                                            device=device,
                                            output_folder=output_folder,
                                            **process_kwargs
                                        )
                                    retried_processed += len(chunk)
                                except RuntimeError as retry_err:
                                    if "out of memory" in str(retry_err).lower():
                                        remaining_failed_chunks.append(chunk)
                                    else:
                                        raise retry_err
                                except Exception:
                                    remaining_failed_chunks.append(chunk)
                            failed_chunks = remaining_failed_chunks

                        retried_failed = sum(len(chunk) for chunk in failed_chunks)

                        if retried_processed > 0:
                            # Keep moving-average accounting stable with the same batch timer.
                            processing_time = max(1e-6, time.time() - start_time)
                            avg_time = processing_time / max(1, retried_processed)
                            for _ in range(retried_processed):
                                self.stats.update(avg_time)
                            processed_count += retried_processed
                        if retried_failed > 0:
                            self.stats.failed_images += retried_failed
                            print(f"  OOM retry failed for {retried_failed} image(s); reduced batch size.")
                    elif "cancelled" in str(e).lower() or "stopped" in str(e).lower():
                        # Re-raise cancellation to stop the entire pipeline
                        raise e
                    else:
                        print(f"  Error processing batch: {e}")
                        import traceback
                        traceback.print_exc()
                        self.stats.failed_images += len(valid_batch)

                except Exception as e:
                    print(f"  Error processing batch: {e}")
                    self.stats.failed_images += len(valid_batch)

                # Periodic allocator reset (every reset_interval_batches, not every batch)
                # Keeps fragmentation in check without killing allocator reuse
                self.batches_since_reset += 1
                if self.batches_since_reset >= self.reset_interval_batches:
                    import gc
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    elif device == "mps":
                        torch.mps.empty_cache()
                    self.batches_since_reset = 0
        finally:
            await prefetch_queue.close()


        self.stats.total_time = time.time() - self.stats.start_time
        return self.stats

    def shutdown(self):
        """Cleanup resources."""
        self.loader.shutdown()


def create_optimized_batch_processor(
    max_batch_size: int = 4,
    prefetch_depth: int = 3,
    num_workers: int = 4,
    enable_cuda_streams: bool = True,
    warmup_steps: int = 1,
    dynamic_batching: bool = True,
    vram_threshold_mb: int = 1024
) -> AsyncBatchPipeline:
    """
    Create an optimized async batch processor.

    Args:
        max_batch_size: Maximum batch size
        prefetch_depth: Number of batches to prefetch
        num_workers: Number of worker threads for image loading
        enable_cuda_streams: Enable CUDA streams for overlapping
        warmup_steps: Number of warmup iterations
        dynamic_batching: Enable dynamic batch size adjustment
        vram_threshold_mb: VRAM threshold for batch size adjustment (MB)

    Returns:
        AsyncBatchPipeline instance
    """
    config = BatchConfig(
        max_batch_size=max_batch_size,
        prefetch_depth=prefetch_depth,
        num_workers=num_workers,
        enable_cuda_streams=enable_cuda_streams,
        warmup_steps=warmup_steps,
        dynamic_batching=dynamic_batching,
        vram_threshold_mb=vram_threshold_mb
    )
    return AsyncBatchPipeline(config)
