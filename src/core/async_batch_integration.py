"""
Integration module for async batch pipeline in app.py.

This module provides wrapper functions to integrate the async batch pipeline
with the existing batch_process_folder function.
"""

import asyncio
import os
import contextlib
from typing import Any, Optional, Callable, List, Dict, Tuple
from PIL import Image
import torch
from concurrent.futures import ThreadPoolExecutor

# Thread pool for blocking I/O operations
_io_executor = ThreadPoolExecutor(max_workers=4)

# Cache for batch processors
_batch_processor_cache: Dict[str, Any] = {}
_pipe_call_params_cache: Dict[type, Any] = {}

# Ampere optimization flag
_ampere_optimizations_enabled = False

from src.runtime_policies import (
    apply_global_cuda_speed_knobs,
    resolution_preset_to_long_edge,
    resolve_generation_guidance,
    should_enable_autocast,
)
from src.core.async_batch_pipeline import (
    create_optimized_batch_processor,
    ProcessingStats
)
from src.image.vlm_prompt_upsampler import upsample_prompt_from_image


def _get_processor_cache_key(
    max_batch_size: int,
    prefetch_depth: int,
    device: str,
    **kwargs
) -> str:
    """Generate a cache key for the batch processor configuration."""
    key_parts = [
        f"batch_size={max_batch_size}",
        f"prefetch={prefetch_depth}",
        f"device={device}",
    ]
    # Add relevant configuration parameters
    for k, v in sorted(kwargs.items()):
        if k in ['enable_cuda_streams', 'warmup_steps', 'dynamic_batching', 'vram_threshold_mb']:
            key_parts.append(f"{k}={v}")
    return "|".join(key_parts)


def clear_batch_processor_cache():
    """Clear the batch processor cache. Call when model/device changes."""
    global _batch_processor_cache
    for processor in _batch_processor_cache.values():
        try:
            processor.shutdown()
        except Exception:
            pass
    _batch_processor_cache.clear()
    print("[Cache] Batch processor cache cleared")


def _gpu_total_vram_gb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
    return 0.0


def enable_ampere_optimizations() -> None:
    """
    Enable Ampere-specific optimizations for RTX 30xx/40xx GPUs.
    Call this once before batch processing starts.

    Enables:
    - TF32 for matmul/conv (10%+ speedup on Ampere)
    - cuDNN benchmark/autotune for fixed shapes
    - cuDNN deterministic=False for faster kernels
    - Maximize fragmentation cache for better allocator reuse
    """
    global _ampere_optimizations_enabled
    if _ampere_optimizations_enabled:
        return
    
    if not torch.cuda.is_available():
        return
    
    apply_global_cuda_speed_knobs(torch)
    _ampere_optimizations_enabled = True


def calculate_dimensions_from_ratio(width: int, height: int, target_resolution: str) -> tuple:
    """Calculate output dimensions maintaining aspect ratio for target resolution."""
    target_size = resolution_preset_to_long_edge(target_resolution)

    aspect_ratio = width / height

    if aspect_ratio >= 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)

    # Round to nearest multiple of 64 for model compatibility
    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64

    new_width = max(256, min(2048, new_width))
    new_height = max(256, min(2048, new_height))

    return new_width, new_height


def apply_scale_to_dimensions(width: int, height: int, downscale_factor: Any) -> tuple:
    """Apply scale factor to dimensions."""
    try:
        if isinstance(downscale_factor, str):
            factor = float(downscale_factor.lower().replace("x", ""))
        else:
            factor = float(downscale_factor)
    except (ValueError, TypeError):
        factor = 1.0

    if factor <= 1.0:
        # FLUX and Z-Image VAEs require dimensions divisible by 16.
        # Align even when no downscale is applied to avoid pipeline crashes.
        new_width = (width // 16) * 16 or 16
        new_height = (height // 16) * 16 or 16
        return max(256, new_width), max(256, new_height)

    # FLUX VAE requires dimensions divisible by vae_scale_factor*2 = 16.
    # Round down to nearest 16 to avoid latent shape mismatches.
    new_width = max(256, int(width / factor))
    new_height = max(256, int(height / factor))

    new_width = (new_width // 16) * 16
    new_height = (new_height // 16) * 16

    new_width = max(256, min(2048, new_width))
    new_height = max(256, min(2048, new_height))

    return new_width, new_height


import inspect

def build_safe_kwargs(pipe: Any, **kwargs):
    """Build kwargs for pipeline call, only including supported parameters."""
    try:
        # Always allow num_images_per_prompt to ensure we don't get duplicates
        allowed_overrides = {"num_images_per_prompt"}

        pipe_class = pipe.__class__
        supported_params = _pipe_call_params_cache.get(pipe_class)
        if supported_params is None:
            sig = inspect.signature(pipe.__call__)
            supported_params = sig.parameters
            _pipe_call_params_cache[pipe_class] = supported_params
        return {k: v for k, v in kwargs.items() if k in supported_params or k in allowed_overrides}
    except (TypeError, ValueError):
        # Fallback to a minimal set if signature cannot be inspected
        return {"prompt": kwargs.get("prompt", ""), "num_images_per_prompt": 1}

def create_batch_processor_func(
    pipe: Any,
    device: str = "cuda",
    autocast_ctx: Any = None,
    **fixed_kwargs
) -> Callable:
    """
    Create a processor function for a batch of images.
    """
    # Extract fixed parameters that don't change per batch
    prompt = fixed_kwargs.get("prompt", "")
    negative_prompt = fixed_kwargs.get("negative_prompt", "")
    steps = fixed_kwargs.get("steps", 4)
    guidance = fixed_kwargs.get("guidance", 3.5)
    model_choice = fixed_kwargs.get("model_choice")
    seed = fixed_kwargs.get("seed", -1)
    input_folder = fixed_kwargs.get("input_folder", "")
    preset = fixed_kwargs.get("preset", "~1024px")
    downscale_factor = fixed_kwargs.get("downscale_factor", 1.0)
    img2img_strength = fixed_kwargs.get("img2img_strength", 1.0)

    async def process_batch(
        paths: List[str],
        images: List[Image.Image],
        pipe: Any,
        device: str,
        output_folder: str,
        **kwargs
    ) -> List[Image.Image]:
        """Process a batch of images."""
        batch_size = len(images)
        if batch_size == 0:
            return []

        # Determine if we should override guidance scale for distilled models
        # (SDNQ, INT8, etc. work best with 0.0 to avoid warnings)
        final_guidance = resolve_generation_guidance(model_choice, guidance)

        # 1. Preprocess all images (Resize & Dimensions)
        resized_images = []
        target_sizes = []
        
        for img in images:
            new_w, new_h = calculate_dimensions_from_ratio(
                img.width, img.height, preset
            )
            new_w, new_h = apply_scale_to_dimensions(new_w, new_h, downscale_factor)
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized_images.append(resized)
            target_sizes.append((new_w, new_h))

        # 2. Prepare Prompts & Seeds
        batch_prompts = []
        batch_negatives = []
        seeds = []
        
        # Base seed
        base_seed_val = int(seed) if seed != -1 else torch.randint(0, 2**32, (1,)).item()
        batch_indices = kwargs.get('indices')
        if not batch_indices:
            batch_indices = list(range(batch_size))

        for i in range(batch_size):
            # Seed
            current_seed = base_seed_val + int(batch_indices[i])
            seeds.append(current_seed)
            
            # Prompts
            current_prompt = prompt
            current_negative = negative_prompt
            batch_prompts.append(current_prompt)
            batch_negatives.append(current_negative)

        # 3. Prepare Generators
        generators = []
        if device == "cuda":
            gen_device = "cuda"
        elif device == "mps":
            gen_device = "mps"
        else:
            gen_device = None
            
        for s in seeds:
            g = torch.Generator(gen_device).manual_seed(s) if gen_device else torch.Generator().manual_seed(s)
            generators.append(g)

        # 4. Run Inference (Batch)
        # Disable autocast for SDNQ models to avoid "Unexpected floating ScalarType" error
        autocast_context = (
            autocast_ctx
            if (autocast_ctx and should_enable_autocast(device, str(type(pipe)), pipe))
            else contextlib.nullcontext()
        )

        results = []

        try:
            with torch.inference_mode(), autocast_context:
                # Check sizes
                first_size = target_sizes[0]
                all_same_size = all(s == first_size for s in target_sizes)
                
                if all_same_size:
                    kwargs = build_safe_kwargs(
                        pipe,
                        prompt=batch_prompts,
                        negative_prompt=batch_negatives,
                        image=resized_images,
                        strength=float(img2img_strength),
                        height=first_size[1],
                        width=first_size[0],
                        num_inference_steps=int(steps),
                        guidance_scale=final_guidance,
                        generator=generators,
                        num_images_per_prompt=1,
                    )
                    print(f"  Running batch inference: {len(batch_prompts)} prompts...")
                    batch_results = pipe(**kwargs).images
                    print(f"  Inference complete. Got {len(batch_results)} images.")
                    results = batch_results
                else:
                    # Fallback to serial for this batch if sizes differ
                    print("  Warning: Batch images have different sizes. Processing serially.")
                    for i in range(batch_size):
                        kwargs = build_safe_kwargs(
                            pipe,
                            prompt=batch_prompts[i],
                            negative_prompt=batch_negatives[i],
                            image=resized_images[i],
                            strength=float(img2img_strength),
                            height=target_sizes[i][1],
                            width=target_sizes[i][0],
                            num_inference_steps=int(steps),
                            guidance_scale=final_guidance,
                            generator=generators[i],
                            num_images_per_prompt=1,
                        )
                        res = pipe(**kwargs).images[0]
                        results.append(res)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise e
            print(f"  Batch inference error: {e}")
            return []

        # 5. Post-processing & Saving
        final_results = []
        # Ensure we don't process more results than we have input paths
        process_limit = min(len(results), len(paths))
        if len(results) != len(paths):
            print(f"  Warning: Batch results mismatch. Paths: {len(paths)}, Results: {len(results)}. Using limit: {process_limit}")

        for i in range(process_limit):
            res = results[i]
            path = paths[i]

            # Save
            base_name = os.path.splitext(os.path.basename(path))[0]

            # Normalize paths for Windows compatibility
            norm_path = os.path.abspath(os.path.normpath(path))
            norm_input = os.path.abspath(os.path.normpath(input_folder))

            try:
                rel_dir = os.path.relpath(os.path.dirname(norm_path), norm_input)
            except ValueError:
                # Paths on different drives or other relpath error
                rel_dir = "."

            output_dir = (
                output_folder if rel_dir == "."
                else os.path.join(output_folder, rel_dir)
            )
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{base_name}_out.png")
            # Post-processing: apply library-driven face swap when the user
            # has enabled auto-swap in the Face Swap tab. A no-op otherwise.
            # Must run before final_results.append so the returned list matches
            # the saved files (mirrors single-image path in image_gen.py).
            try:
                from src.image import faceswap_config
                res = faceswap_config.post_process_with_library(res, device=device)
            except Exception as _swap_exc:
                print(f"  [face-swap] batch library post-process skipped: {_swap_exc}")
            final_results.append(res)
            await asyncio.to_thread(res.save, output_path)
            print(f"  [SAVED] {output_path}")

        return final_results

    return process_batch



async def process_batch_folder_async(
    image_paths: List[str],
    pipe: Any,
    device: str,
    output_folder: str,
    progress_callback: Optional[Callable] = None,
    max_batch_size: int = 4,
    prefetch_depth: int = 3,
    **process_kwargs
) -> ProcessingStats:
    """
    Process a folder of images asynchronously with optimizations.

    Args:
        image_paths: List of image file paths
        pipe: Pipeline object
        device: Device to use
        output_folder: Output directory
        progress_callback: Gradio progress callback
        max_batch_size: Maximum batch size
        prefetch_depth: Prefetch queue depth
        **process_kwargs: Additional processing parameters

    Returns:
        ProcessingStats with processing statistics
    """
    # Enable Ampere optimizations once per session
    if device == "cuda":
        enable_ampere_optimizations()

    # Determine optimal batch size based on device
    if device == "cuda":
        # Check VRAM
        if torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # 3070 has 8GB; bias toward stability.
            if total_vram_gb <= 8:
                max_batch_size = min(max_batch_size, 2)
            elif total_vram_gb <= 12:
                max_batch_size = min(max_batch_size, 3)
            else:
                max_batch_size = min(max_batch_size, 4)
    elif device == "mps":
        max_batch_size = 2  # Conservative for MPS
    else:
        max_batch_size = 1  # CPU

    # Bucket images by target size to keep true batching
    preset = process_kwargs.get("preset", "~1024px")
    downscale_factor = process_kwargs.get("downscale_factor", 1.0)
    buckets = {}
    fallback = []
    
    def get_image_dimensions(path):
        """Get image dimensions synchronously (called via asyncio.to_thread)."""
        with Image.open(path) as img:
            new_w, new_h = calculate_dimensions_from_ratio(img.width, img.height, preset)
            new_w, new_h = apply_scale_to_dimensions(new_w, new_h, downscale_factor)
        return new_w, new_h
    
    semaphore = asyncio.Semaphore(64)

    async def get_image_dimensions_limited(path):
        async with semaphore:
            try:
                return path, await asyncio.to_thread(get_image_dimensions, path), None
            except Exception as exc:
                return path, None, exc

    for path, size, exc in await asyncio.gather(*(get_image_dimensions_limited(path) for path in image_paths)):
        if exc is None and size is not None:
            buckets.setdefault(size, []).append(path)
        else:
            fallback.append(path)

    if buckets:
        bucket_keys = sorted(buckets.keys())
        process_kwargs.setdefault("width", bucket_keys[0][0])
        process_kwargs.setdefault("height", bucket_keys[0][1])
        bucket_list = [buckets[key] for key in bucket_keys]
        if fallback:
            bucket_list.append(fallback)
        processing_paths = bucket_list
    else:
        processing_paths = image_paths

    # Create or reuse processor
    cache_key = _get_processor_cache_key(
        max_batch_size=max_batch_size,
        prefetch_depth=prefetch_depth,
        device=device,
        enable_cuda_streams=(device == "cuda"),
        warmup_steps=1,
        dynamic_batching=True,
        vram_threshold_mb=1536 if device == "cuda" else 1024
    )
    
    if cache_key in _batch_processor_cache:
        processor = _batch_processor_cache[cache_key]
        print(f"[Cache] Reusing batch processor for config: {cache_key}")
    else:
        processor = create_optimized_batch_processor(
            max_batch_size=max_batch_size,
            prefetch_depth=prefetch_depth,
            num_workers=4,
            enable_cuda_streams=(device == "cuda"),
            warmup_steps=1,
            dynamic_batching=True,
            vram_threshold_mb=1536 if device == "cuda" else 1024
        )
        _batch_processor_cache[cache_key] = processor
        print(f"[Cache] Created new batch processor for config: {cache_key}")


    # Create process function
    process_fn = create_batch_processor_func(
        pipe=pipe,
        device=device,
        **process_kwargs
    )

    # Add index to kwargs for seed calculation
    indexed_paths = {p: i for i, p in enumerate(image_paths)}

    # Wrapper to inject index for batch
    async def process_with_index(paths, images, pipe, device, output_folder, **kwargs):
        batch_indices = [indexed_paths.get(p, 0) for p in paths]

        return await process_fn(
            paths=paths,
            images=images,
            pipe=pipe,
            device=device,
            output_folder=output_folder,
            indices=batch_indices,
            **kwargs
        )

    # Cached processors stay alive across runs and are released only when
    # clear_batch_processor_cache() is called for a model/device change.
    stats = await processor.process_batch_async(
        image_paths=processing_paths,
        pipe=pipe,
        device=device,
        process_fn=process_with_index,
        output_folder=output_folder,
        progress_callback=progress_callback,
        **process_kwargs
    )

    return stats


def run_async_batch_processing(
    image_paths: List[str],
    pipe: Any,
    device: str,
    output_folder: str,
    progress_callback: Optional[Callable] = None,
    **process_kwargs
) -> ProcessingStats:
    """
    Synchronous wrapper for async batch processing.

    This function can be called from the existing synchronous code.
    """
    # ``get_event_loop`` is deprecated when no loop is running (Python
    # 3.10+) and will be removed in a future version.  Probe explicitly
    # for a *running* loop instead: if there is one we must hop to a
    # worker thread (asyncio.run refuses to nest), otherwise we create a
    # fresh loop via ``asyncio.run``.
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False

    if running:
        # Create a new loop in a worker thread so we don't collide with
        # the caller's running loop.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                process_batch_folder_async(
                    image_paths=image_paths,
                    pipe=pipe,
                    device=device,
                    output_folder=output_folder,
                    progress_callback=progress_callback,
                    **process_kwargs
                )
            )
            return future.result()

    return asyncio.run(
        process_batch_folder_async(
            image_paths=image_paths,
            pipe=pipe,
            device=device,
            output_folder=output_folder,
            progress_callback=progress_callback,
            **process_kwargs
        )
    )
