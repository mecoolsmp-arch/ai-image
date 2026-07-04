import os
import traceback
import torch
import contextlib
import re
import inspect
from PIL import Image as PILImage
from typing import Optional, List, Any, Callable

from src.runtime_policies import (
    is_flux_model,
    resolve_default_inference_steps,
    resolve_generation_guidance,
    should_enable_autocast,
    should_use_attention_slicing,
    should_use_vae_slicing,
    should_use_vae_tiling,
)
from src.utils.device_utils import get_device_vram_gb

class ImageGenerator:
    """Core image generation engine decoupled from UI."""

    def __init__(self, pipeline_manager):
        self.pm = pipeline_manager
        self.stop_requested = False
        self._cached_pipe_params = None  # Cache inspect.signature result
        self._cached_pipe_class = None   # Track which pipe class was cached
        self._last_config_signature = None  # Skip redundant configure_optimization_policy calls
        self._pipe_signature_cache = {}

    def request_stop(self):
        self.stop_requested = True

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        steps: int = 20,
        seed: int = -1,
        guidance: float = 3.5,
        device: str = "cuda",
        model_choice: str = "Z-Image Turbo (Int8 - 8GB Safe)",
        input_images: Optional[List[Any]] = None,
        downscale_factor: float = 1.0,
        img2img_strength: float = 0.8,
        lora_file: Optional[str] = None,
        lora_strength: float = 1.0,
        enable_klein_anatomy_fix: bool = False,
        optimization_profile: Optional[str] = None,
        enable_windows_compile_probe: Optional[bool] = None,
        enable_cuda_graphs: Optional[bool] = None,
        enable_optional_accelerators: Optional[bool] = None,
        # ---- Preservation (pose + expression from a reference image) ----
        enable_preservation: bool = False,
        preservation_input: Optional[Any] = None,
        preservation_detector: str = "dwpose",
        preservation_mode: str = "body_face",
        # ---- Klein Face Expression Transfer LoRA (quality path) --------
        # Complement to DWPose-based preservation: loads the Civitai
        # v2658175 LoRA and prepends its dual-image trigger word. Works
        # stand-alone (without enable_preservation) or stacked on top of
        # the DWPose skeleton path.
        enable_expression_transfer: bool = False,
        # ---- Text preservation (OCR the source, repaint on output) -----
        # Target use case: manga / comic / caption-heavy source images
        # where the diffusion model reliably garbles or deletes text.
        # We OCR the source at pipeline start (while the model loads),
        # then repaint the text on the final image so the
        # preserved text ends up at the intended resolution.
        enable_text_preservation: bool = False,
        text_preservation_source: Optional[Any] = None,
        text_preservation_languages: Optional[List[str]] = None,
        text_preservation_min_confidence: float = 0.3,
        # Internal callers can opt out of the library-driven face swap
        # post-process.
        skip_face_swap: bool = False,
        progress_callback: Optional[Callable] = None,
    ):
        self.stop_requested = False
        height = self._safe_int(height, 512)
        width = self._safe_int(width, 512)
        # Both FLUX and Z-Image VAEs require dimensions divisible by
        # vae_scale_factor*2=16. Align down to avoid pipeline crashes.
        height = (height // 16) * 16 or 16
        width = (width // 16) * 16 or 16
        steps = self._safe_int(steps, 20)
        seed = self._safe_int(seed, -1)
        downscale_factor = self._normalize_downscale_factor(downscale_factor)
        # Skip redundant configure_optimization_policy calls when parameters haven't changed
        config_signature = (device, optimization_profile, enable_windows_compile_probe, enable_cuda_graphs, enable_optional_accelerators)
        if config_signature != self._last_config_signature:
            self.pm.configure_optimization_policy(
                device=device,
                profile=optimization_profile,
                enable_windows_compile_probe=enable_windows_compile_probe,
                enable_optional_accelerators=enable_optional_accelerators,
            )
            self._last_config_signature = config_signature

        # 1. Model & Pipeline Preparation
        try:
            self.pm.ensure_models_downloaded(
                model_choice,
                enable_klein_anatomy_fix=enable_klein_anatomy_fix,
                progress=progress_callback
            )
        except Exception as e:
            return None, f"Model download failed: {e}", None

        pipe = self.pm.load_pipeline(model_choice, device)
        current_model = self.pm.current_model
        # Capture fallback reason immediately so later operations do not
        # overwrite the user's real model fallback context.
        fallback_reason = self.pm.last_model_fallback_reason

        # Clamp step count for distilled models. FLUX.2 [klein] and Z-Image
        # Turbo are 4-step-distilled: quality plateaus at ~4 steps and any
        # additional steps are wasted wall-clock on RTX 3070. Users who pick
        # 20+ steps in the UI unknowingly 5× their latency with no visible
        # improvement; clamp silently here instead of surprising them.
        resolved_profile = getattr(self.pm, "optimization_profile", "balanced")
        effective_steps = resolve_default_inference_steps(
            current_model, steps, optimization_profile=resolved_profile
        )
        if effective_steps != steps:
            print(
                f"  Steps clamped {steps}→{effective_steps} for distilled model "
                f"'{current_model}' (4-step recipe, quality plateaus above ~8)."
            )
            steps = effective_steps

        if self.stop_requested:
            return None, "Cancelled by user.", None

        # Capture the user's original first input reference BEFORE the
        # preservation block mutates ``input_images`` (it prepends a pose
        # skeleton at index 0). Text preservation's source-image fallback
        # at the end of the pipeline must see the original manga/comic
        # panel, not the black-background stick-figure skeleton —
        # otherwise OCR runs on the skeleton and extracts nothing.
        _original_first_input = input_images[0] if input_images else None

        # Preservation: extract pose + facial landmarks from preservation_input
        # and prepend the skeleton to input_images so FLUX.2-klein conditions
        # the generation on it. Also appends a short pose directive to the
        # prompt. Silent no-op when disabled or when preservation_input is
        # empty; silent degrade (generate without pose) when extraction fails.
        preservation_status: Optional[str] = None
        if enable_preservation and preservation_input is not None:
            try:
                from src.image.preservation import (
                    augment_prompt_for_preservation,
                    build_preservation_inputs,
                )
                preservation_pil = preservation_input
                # Gallery items arrive as (image, caption) tuples — unwrap.
                if isinstance(preservation_pil, (tuple, list)):
                    preservation_pil = preservation_pil[0]
                existing_refs = list(input_images) if input_images else None
                augmented, _skeleton, preservation_status = build_preservation_inputs(
                    preservation_pil,
                    existing_input_images=existing_refs,
                    device=device,
                    detector=preservation_detector,
                    mode=preservation_mode,
                )
                # Only augment the prompt when a skeleton was actually
                # extracted. ``build_preservation_inputs`` returns the
                # original (possibly non-empty) ``existing_input_images``
                # list even when extraction fails, so checking
                # ``augmented is not None`` is insufficient — we'd promise
                # the model a pose skeleton that isn't in the refs and
                # confuse it. Gate on ``_skeleton`` directly.
                if _skeleton is not None:
                    input_images = augmented
                    prompt = augment_prompt_for_preservation(prompt)
                print(f"  [preservation] {preservation_status}")
            except Exception as _pres_exc:
                print(f"  [preservation] skipped: {_pres_exc}")

        # LoRA Loading
        if lora_file:
            self.pm.load_lora(lora_file, lora_strength, device)

        if enable_klein_anatomy_fix and "flux2-klein" in current_model:
            if os.path.exists(self.pm.klein_anatomy_lora_path):
                self.pm.load_lora(self.pm.klein_anatomy_lora_path, 0.8, device)

        # Klein Face Expression Transfer LoRA — quality complement to the
        # DWPose preservation path. Auto-downloads on first use, loads at
        # the Civitai-recommended 1.0 strength, and prepends its dual-
        # image trigger phrase so the model knows which reference slot
        # holds the expression to copy. Silent no-op on non-Klein models
        # (the LoRA weights don't map onto Z-Image / SDXL layers).
        # Automation: when enabled without manual preservation_input, automatically
        # use the first input image as reference for pose/expression extraction.
        if enable_expression_transfer and "flux2-klein" in current_model:
            # Automatic mode: use first input image as reference if no manual input provided
            if preservation_input is None and input_images and len(input_images) > 0:
                preservation_input = input_images[0]
                print(f"  [expression-transfer] automatic mode: using first input image as reference")
            
            # Enable preservation to extract pose/expression from reference image
            if preservation_input is not None and not enable_preservation:
                enable_preservation = True
                print(f"  [expression-transfer] auto-enabled preservation for expression extraction")
            
            try:
                from src.constants import (
                    KLEIN_EXPRESSION_LORA_STRENGTH,
                    KLEIN_EXPRESSION_LORA_TRIGGER,
                )

                expr_path = self.pm.ensure_builtin_lora_downloaded(
                    "klein_expression", progress_callback
                )
                if expr_path and os.path.exists(expr_path):
                    # PipelineManager.load_lora only holds one LoRA at
                    # a time — a subsequent call unloads whatever was
                    # loaded before. Warn the user when the expression
                    # transfer LoRA is about to displace a previously
                    # loaded adapter (their custom upload, or the
                    # Anatomy Quality Fixer) so they aren't surprised
                    # that the prior LoRA's effect disappeared.  A
                    # proper multi-adapter stack is tracked as a
                    # follow-up in load_lora itself.
                    prev = getattr(self.pm, "current_lora_path", None)
                    if prev:
                        print(
                            "  [expression-transfer] warning: replacing "
                            f"previously loaded LoRA '{os.path.basename(prev)}'. "
                            "PipelineManager only supports one LoRA at a time; "
                            "disable Klein Anatomy Fix or your custom LoRA if "
                            "you want to keep it active alongside expression "
                            "transfer."
                        )
                    self.pm.load_lora(expr_path, KLEIN_EXPRESSION_LORA_STRENGTH, device)
                    # Prepend the trigger phrase only once — users may
                    # already have authored it into their prompt.
                    if KLEIN_EXPRESSION_LORA_TRIGGER.lower() not in (prompt or "").lower():
                        prompt = f"{KLEIN_EXPRESSION_LORA_TRIGGER}. {prompt}"
            except Exception as _expr_exc:
                print(f"  [expression-transfer] skipped: {_expr_exc}")

        # 5b. Optional Accelerators (Pruna FORA, qkv_diffusers, etc.)
        # Apply BEFORE memory policy / CPU offload because FORA breaks
        # when components are moved between CPU and GPU each step.
        if current_model == "flux2-klein-sdnq":
            pipe, _ = self.pm.prepare_flux_sdnq_optional_accelerators(
                pipe,
                device=device,
                steps=steps,
                enable_optional_accelerators=enable_optional_accelerators,
                mode="single",
                has_lora=self.pm.current_lora_path is not None,
                has_pulid=False,
                has_faceswap=False,
                has_pose_control=enable_preservation,
                has_cpu_offload=self.pm.should_enable_cpu_offload(
                    current_model, enable_preservation, device
                ),
            )
        elif current_model == "flux2-klein-int8":
            pipe, _ = self.pm.prepare_flux_int8_optional_accelerators(
                pipe,
                device=device,
                steps=steps,
                enable_optional_accelerators=enable_optional_accelerators,
                mode="single",
                has_lora=self.pm.current_lora_path is not None,
                has_pulid=False,
                has_faceswap=False,
                has_pose_control=enable_preservation,
                has_cpu_offload=self.pm.should_enable_cpu_offload(
                    current_model, enable_preservation, device
                ),
            )

        # 6. Generation Parameters
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()

        generator = torch.Generator(device if device != "cpu" else None).manual_seed(int(seed))

        vram_gb = get_device_vram_gb(device)
        use_attention_slicing = should_use_attention_slicing(
            device=device,
            model_key=current_model,
            pipe=pipe,
            vram_gb=vram_gb,
            optimization_profile=resolved_profile,
        )
        if use_attention_slicing:
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
        elif hasattr(pipe, "disable_attention_slicing"):
            try:
                pipe.disable_attention_slicing()
            except Exception:
                pass

        if should_use_vae_slicing(device=device, vram_gb=vram_gb, optimization_profile=resolved_profile):
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            elif hasattr(getattr(pipe, "vae", None), "enable_slicing"):
                try:
                    pipe.vae.enable_slicing()
                except Exception:
                    pass
        if should_use_vae_tiling(
            device=device,
            model_key=current_model,
            pipe=pipe,
            vram_gb=vram_gb,
            width=int(width),
            height=int(height),
            optimization_profile=resolved_profile,
        ):
            if hasattr(pipe, "enable_vae_tiling"):
                try:
                    pipe.enable_vae_tiling()
                except ValueError:
                    pass
            elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
                try:
                    pipe.vae.enable_tiling()
                except ValueError:
                    pass
        if hasattr(pipe, "enable_model_cpu_offload") and self.pm.should_enable_cpu_offload(
            current_model,
            enable_preservation,
            device,
        ):
            pipe.enable_model_cpu_offload()
            print("  CPU offload enabled (model components will move between CPU/GPU per step).")
            print("  Note: This significantly slows generation. Disable pose preservation for faster inference.")

        final_guidance = resolve_generation_guidance(current_model, guidance)

        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if should_enable_autocast(device, current_model, pipe)
            else contextlib.nullcontext()
        )

        # 7. Core Generation Logic
        try:
            with torch.inference_mode(), autocast_ctx:
                # Decide generation mode
                has_input = input_images is not None and len(input_images) > 0
                is_flux = is_flux_model(current_model)

                # Helper to build common kwargs
                def get_pipe_kwargs(target_pipe):
                    kwargs = {"prompt": prompt}
                    # Cache inspect.signature result per pipe class to avoid
                    # repeated introspection on every generation call.
                    supported_params = self._get_pipe_call_params(target_pipe)

                    if "prompt_2" in supported_params:
                        kwargs["prompt_2"] = prompt
                    if negative_prompt and "negative_prompt" in supported_params:
                        kwargs["negative_prompt"] = negative_prompt
                    if negative_prompt and "negative_prompt_2" in supported_params:
                        kwargs["negative_prompt_2"] = negative_prompt
                    return kwargs

                image = None
                mode_str = "txt2img"

                # Img2Img Path
                if image is None and has_input:
                    img_w, img_h = self._scale_dims(width, height, downscale_factor)
                    processed_images = []
                    for img_data in input_images[:6]:
                        img = img_data[0] if isinstance(img_data, tuple) else img_data
                        resized = img.copy().resize((img_w, img_h), PILImage.Resampling.LANCZOS).convert("RGB")
                        processed_images.append(resized)

                    input_arg = processed_images if len(processed_images) > 1 else processed_images[0]

                    if is_flux:
                        # Flux2KleinPipeline auto-aligns image dims to multiples
                        # of vae_scale_factor*2=16 internally. Pass height/width
                        # only when they match the downscaled image to avoid
                        # latent shape mismatches between noise and image latents.
                        flux_kwargs = get_pipe_kwargs(pipe)
                        flux_kwargs["image"] = input_arg
                        flux_kwargs["height"] = img_h
                        flux_kwargs["width"] = img_w
                        flux_kwargs["num_inference_steps"] = int(steps)
                        flux_kwargs["guidance_scale"] = final_guidance
                        flux_kwargs["generator"] = generator
                        flux_kwargs["num_images_per_prompt"] = 1
                        # Forward ``img2img_strength`` to the FLUX img2img
                        # pipeline. Without this the Klein Hi-Res LoRA upscale
                        # (which passes 0.35 for a light refine) silently
                        # runs at the pipeline default (~0.8–1.0) and heavily
                        # regenerates the image instead of preserving content.
                        # Guard by signature because not every FLUX-family
                        # pipeline exposes ``strength``.
                        flux_params = self._get_pipe_call_params(pipe)
                        if "strength" in flux_params:
                            flux_kwargs["strength"] = float(img2img_strength)
                        # Only pass height/width if they match the image dims
                        # after pipeline's internal alignment (multiples of 16)
                        aligned_w = (img_w // 16) * 16
                        aligned_h = (img_h // 16) * 16
                        if aligned_w != img_w or aligned_h != img_h:
                            # Dimensions weren't aligned — let pipeline derive
                            # height/width from the image itself
                            flux_kwargs.pop("height", None)
                            flux_kwargs.pop("width", None)
                        image = pipe(**flux_kwargs).images[0]
                    elif "zimage" in current_model:
                        zimage_img2img_pipe = self.pm.get_zimage_img2img_pipeline(
                            device=device,
                            use_full_model=False,
                        )
                        image = zimage_img2img_pipe(
                            **get_pipe_kwargs(zimage_img2img_pipe),
                            image=input_arg,
                            strength=float(img2img_strength),
                            height=img_h, width=img_w,
                            num_inference_steps=int(steps),
                            guidance_scale=0.0,
                            generator=generator,
                            num_images_per_prompt=1,
                        ).images[0]
                    mode_str = f"img2img ({len(processed_images)} ref)"

                # C. Txt2Img Path (Fallback)
                if image is None:
                    image = pipe(
                        **get_pipe_kwargs(pipe),
                        height=int(height), width=int(width),
                        num_inference_steps=int(steps),
                        guidance_scale=final_guidance,
                        generator=generator,
                        num_images_per_prompt=1,
                    ).images[0]
                    mode_str = "txt2img"

        except Exception as e:
            traceback.print_exc()
            return None, f"Generation failed: {e}"

        if self.stop_requested: return None, "Cancelled."

        # Cleanup & Return
        self._cleanup(device)

        # Post-processing: library-driven face swap. If the user has enabled
        # "Auto-swap saved characters" in the Face Swap tab, every generated
        # image is fed through inswapper so saved characters stay consistent
        # across Single / Batch / Video without manual per-output assignments.
        if skip_face_swap:
            # Internal caller (e.g. Klein Hi-Res LoRA refine) opted out —
            # the outer generation already ran library auto-swap, so running
            # it again would double-swap the same faces and visibly degrade
            # identity + add ~1 s/face. Standalone Upscale-tab invocations
            # likewise never asked for a swap.
            print("  [face-swap] library post-process skipped (internal caller opted out).")
        else:
            try:
                from src.image import faceswap_config
                image = faceswap_config.post_process_with_library(image, device=device)
            except Exception as _swap_exc:
                # Library auto-swap is best-effort — never fails generation.
                print(f"  [face-swap] library post-process skipped: {_swap_exc}")

        # Text preservation: runs LAST so the repainted text lands on
        # the final post-face-swap image at full output
        # resolution. If no explicit source was passed, fall back to
        # the first input/reference image — for manga-to-realistic the
        # img2img source IS the text source. Independent of
        # ``skip_face_swap``: recursive callers (e.g. the Klein Hi-Res
        # LoRA refine) don't set ``enable_text_preservation=True`` on
        # the inner call so the feature naturally doesn't re-trigger.
        text_status: Optional[str] = None
        if enable_text_preservation:
            try:
                from src.image.text_preservation import preserve_text
                tp_source = text_preservation_source
                if tp_source is None:
                    # Use the pre-preservation reference so OCR runs on
                    # the original manga/comic panel, never on the pose
                    # skeleton that the preservation step prepended.
                    tp_source = _original_first_input
                tp_langs = tuple(text_preservation_languages or ("en",))
                image, text_status = preserve_text(
                    tp_source,
                    image,
                    device=device,
                    languages=tp_langs,
                    min_confidence=float(text_preservation_min_confidence),
                )
                print(f"  [text-preservation] {text_status}")
            except Exception as _text_exc:
                # Best-effort — never fail the generation because OCR
                # misbehaved.
                print(f"  [text-preservation] skipped: {_text_exc}")

        fallback_info = f" | Note: {fallback_reason}" if fallback_reason else ""
        status = f"Seed: {seed} | Mode: {mode_str} | Model: {current_model} | Device: {device}{fallback_info}"
        if preservation_status:
            status += f" | {preservation_status}"
        if text_status:
            status += f" | {text_status}"
        return image, status

    def _get_pipe_call_params(self, target_pipe):
        pipe_class = target_pipe.__class__
        cached = self._pipe_signature_cache.get(pipe_class)
        if cached is not None:
            return cached
        try:
            supported_params = inspect.signature(target_pipe.__call__).parameters
        except (TypeError, ValueError):
            supported_params = {}
        self._pipe_signature_cache[pipe_class] = supported_params
        self._cached_pipe_class = pipe_class
        self._cached_pipe_params = supported_params
        return supported_params

    def _scale_dims(self, w, h, factor):
        base_w = self._safe_int(w, 512)
        base_h = self._safe_int(h, 512)
        scale = self._parse_downscale_factor(factor)
        if scale <= 1.0:
            # Still align to 16 even without downscale — Z-Image VAE
            # requires dimensions divisible by vae_scale_factor*2 = 16.
            return max(256, (base_w // 16) * 16), max(256, (base_h // 16) * 16)
        # FLUX VAE requires dimensions divisible by vae_scale_factor*2 = 16.
        # Round down to nearest 16 to avoid latent shape mismatches.
        align = 16
        new_w = int((base_w / scale) // align) * align
        new_h = int((base_h / scale) // align) * align
        return max(256, new_w), max(256, new_h)

    def _safe_int(self, value, default: int) -> int:
        try:
            if value is None:
                return default
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(value)
            text = str(value).strip()
            if not text:
                return default
            return int(float(text))
        except (TypeError, ValueError):
            return default

    def _parse_downscale_factor(self, value) -> float:
        if value is None:
            return 1.0
        if isinstance(value, (int, float)):
            factor = float(value)
        else:
            text = str(value).strip().lower()
            if not text:
                return 1.0
            if text.endswith("x"):
                text = text[:-1].strip()
            match = re.search(r"\d+(?:\.\d+)?", text)
            if match:
                text = match.group(0)
            try:
                factor = float(text)
            except ValueError:
                return 1.0
        return factor if factor > 0 else 1.0

    def _normalize_downscale_factor(self, value) -> str:
        factor = self._parse_downscale_factor(value)
        if factor <= 1.0:
            return "1x"
        return f"{factor:g}x"

    def _cleanup(self, device):
        # Lightweight cleanup: avoid gc.collect() and empty_cache() after every
        # generation as they force CUDA to release cached blocks, causing
        # re-allocation overhead on the next generation. Heavy cleanup is
        # handled during model switches in PipelineManager.load_pipeline().
        pass
