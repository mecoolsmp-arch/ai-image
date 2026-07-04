import os
import sys
import json
import copy
import torch
import threading
import contextlib
import inspect
import shutil
import tempfile
import atexit
from typing import Optional, Dict, Any, List, Union, Callable, Tuple
from huggingface_hub import snapshot_download, try_to_load_from_cache, hf_hub_download
from safetensors.torch import load_file

from src.runtime_policies import (
    default_enable_windows_compile_probe,
    is_cuda13_runtime,
    is_flux_model,
    is_sdnq_model,
    is_sdnq_or_quantized,
    record_torch_compile_probe_result,
    resolve_optimization_profile,
    resolve_model_choice_for_device,
    should_probe_torch_compile,
    should_use_attention_slicing,
    should_use_vae_slicing,
    should_use_vae_tiling,
)
from src.utils.device_utils import get_memory_usage, print_memory, get_device_vram_gb
from src.utils.json_compat import json_dtype_patch_context

ZIMAGE_INT8_REPO_ID = "Disty0/Z-Image-Turbo-SDNQ-int8"

class PipelineManager:
    """Manages AI pipelines, models, and LoRA state."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.pipe = None
        self.current_device = None
        self.current_model = None  # Internal key like "zimage-int8"
        self.current_lora_path = None
        self.current_lora_network = None
        self.compiled_models = {}
        self.controlnet_union = None
        self.zimage_img2img_pipes = {}
        self.flux2_small_decoder_vaes: Dict[str, Any] = {}
        self.runtime_memory_policy_cache: Dict[tuple, Dict[str, Any]] = {}
        self.active_runtime_memory_policy: Dict[str, Any] = {}
        self._active_runtime_policy_signature = None
        self.optional_accelerator_pipes: Dict[tuple, Any] = {}
        self.active_optional_accelerator_status: Dict[str, Any] = {}
        self.last_model_fallback_reason = None
        self.optimization_profile = "balanced"
        self.enable_windows_compile_probe = False
        self.enable_optional_accelerators = False

        self.models_dir = os.path.join(base_dir, "models")
        self.loras_dir = os.path.join(base_dir, "loras")
        self.state_dir = os.path.join(base_dir, "user_state")
        self.cache_dir = os.path.join(base_dir, "cache")

        # Klein Anatomy Quality Fixer — 4B variant.  The Civitai URL takes
        # a *version* ID (2617474 is the 4B; the bare model ID 2324991 used
        # to resolve to the 9B build which silently mismatched 4B users).
        # The on-disk filename is suffixed ``_4b`` so existing installs that
        # already downloaded the old 9B weights at the un-suffixed path
        # don't keep short-circuiting the download guard and silently
        # running the wrong LoRA.  A best-effort cleanup of the stale
        # 9B file runs below so disk isn't wasted.
        self.klein_anatomy_lora_url = "https://civitai.com/api/download/models/2617474"
        self.klein_anatomy_lora_path = os.path.join(
            self.loras_dir, "kleinSliderAnatomy_4b.safetensors"
        )
        _legacy_anatomy_path = os.path.join(
            self.loras_dir, "kleinSliderAnatomy.safetensors"
        )
        if os.path.exists(_legacy_anatomy_path):
            try:
                os.remove(_legacy_anatomy_path)
                print(
                    "  [migration] removed legacy Klein Anatomy 9B LoRA at "
                    f"{_legacy_anatomy_path}; 4B build will download on next use."
                )
            except OSError:
                # Best-effort: leaving it in place wastes ~11 MB but never
                # breaks generation because the 4B path is independent.
                pass

        # Klein High-Resolution LoRA — 4B build.  Used as the quality-first
        # upscaler path (trigger "High resolution", strength 0.9, ~88 MB).
        # Runs a full FLUX.2 img2img pass; slower than ESRGAN but preserves
        # content much more faithfully.
        # Klein Face Expression Transfer LoRA — 4B build (trigger "transfer
        # character face expression in image1 with character face expression
        # in image2", strength 1.0, ~88 MB).  Quality complement to DWPose
        # preservation — skips the pose-extraction pre-pass entirely.
        # This is the 4B version (model 2363566).
        self.klein_expression_lora_url = "https://civitai.com/api/download/models/2363566"
        self.klein_expression_lora_path = os.path.join(self.loras_dir, "kleinFaceExpressionTransfer.safetensors")

        # Built-in anime-to-photoreal LoRAs
        # Realistic Snapshot v5 for Z-Image Turbo: adds raw texture, realistic skin,
        # natural lighting. Recommended LoRA strength 0.60-0.70.
        self.zimage_realistic_lora_url = "https://civitai.com/api/download/models/2748431"
        self.zimage_realistic_lora_path = os.path.join(self.loras_dir, "realistic_snapshot_v5.safetensors")
        # Ultra Real - Amateur Selfies for FLUX.2 Klein 4B: photorealistic smartphone-style selfies.
        # Specifically optimized for Flux.2 Klein 4B (88MB). Recommended strength 1.0-1.25.
        self.flux_anime2real_lora_url = "https://civitai.com/api/download/models/2666766"
        self.flux_anime2real_lora_path = os.path.join(self.loras_dir, "ultra_real_amateur_selfies_klein4b.safetensors")

        self._verified_model_repos: set = set()

    def configure_optimization_policy(
        self,
        device: Optional[str] = None,
        profile: Optional[str] = None,
        enable_windows_compile_probe: Optional[bool] = None,
        enable_optional_accelerators: Optional[bool] = None,
    ) -> None:
        target_device = device or self.current_device or "cuda"
        cuda_runtime = getattr(torch.version, "cuda", None)
        self.optimization_profile = resolve_optimization_profile(
            profile,
            device=target_device,
            cuda_runtime=cuda_runtime,
        )
        if enable_windows_compile_probe is None:
            self.enable_windows_compile_probe = default_enable_windows_compile_probe(
                target_device,
                cuda_runtime=cuda_runtime,
            )
        else:
            self.enable_windows_compile_probe = bool(enable_windows_compile_probe)
        if enable_optional_accelerators is not None:
            self.enable_optional_accelerators = bool(enable_optional_accelerators)

    def get_flux2_pipeline_class(self):
        """Resolve Flux2 pipeline class across diffusers versions."""
        try:
            from diffusers import Flux2KleinPipeline
            return Flux2KleinPipeline
        except Exception:
            from diffusers import Flux2Pipeline
            return Flux2Pipeline

    def _build_asymmetric_flux2_small_decoder_vae(self, dtype):
        """Build the FLUX.2 small decoder locally when diffusers cannot instantiate it natively."""
        from diffusers import AutoencoderKLFlux2
        from diffusers.models.autoencoders.autoencoder_kl_flux2 import Decoder

        repo_id = "black-forest-labs/FLUX.2-small-decoder"
        config_path = hf_hub_download(repo_id, "config.json")
        weights_path = hf_hub_download(repo_id, "diffusion_pytorch_model.safetensors")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        vae = AutoencoderKLFlux2(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            down_block_types=tuple(config["down_block_types"]),
            up_block_types=tuple(config["up_block_types"]),
            block_out_channels=tuple(config["block_out_channels"]),
            layers_per_block=config["layers_per_block"],
            act_fn=config["act_fn"],
            latent_channels=config["latent_channels"],
            norm_num_groups=config["norm_num_groups"],
            sample_size=config["sample_size"],
            force_upcast=config["force_upcast"],
            use_quant_conv=config["use_quant_conv"],
            use_post_quant_conv=config["use_post_quant_conv"],
            mid_block_add_attention=config["mid_block_add_attention"],
            batch_norm_eps=config["batch_norm_eps"],
            batch_norm_momentum=config["batch_norm_momentum"],
            patch_size=tuple(config["patch_size"]),
        )
        vae.decoder = Decoder(
            in_channels=config["latent_channels"],
            out_channels=config["out_channels"],
            up_block_types=tuple(config["up_block_types"]),
            block_out_channels=tuple(config["decoder_block_out_channels"]),
            layers_per_block=config["layers_per_block"],
            norm_num_groups=config["norm_num_groups"],
            act_fn=config["act_fn"],
            mid_block_add_attention=config["mid_block_add_attention"],
        )
        vae.register_to_config(decoder_block_out_channels=tuple(config["decoder_block_out_channels"]))
        vae.load_state_dict(load_file(weights_path), strict=True)
        return vae.to(dtype=dtype)

    def get_flux2_small_decoder_vae(self, dtype):
        """Load and cache the FLUX.2 small decoder VAE for compatible 4B models."""
        cache_key = str(dtype)
        if cache_key in self.flux2_small_decoder_vaes:
            return self.flux2_small_decoder_vaes[cache_key]

        from diffusers import AutoencoderKLFlux2
        supports_asymmetric_decoder = (
            "decoder_block_out_channels" in inspect.signature(AutoencoderKLFlux2.__init__).parameters
        )

        if supports_asymmetric_decoder:
            try:
                vae = AutoencoderKLFlux2.from_pretrained(
                    "black-forest-labs/FLUX.2-small-decoder",
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
            except Exception as exc:
                print(
                    "  Warning: Native FLUX.2 small decoder loading failed; "
                    f"rebuilding the same small-decoder weights with the local compatibility loader: {exc}"
                )
                vae = self._build_asymmetric_flux2_small_decoder_vae(dtype)
        else:
            vae = self._build_asymmetric_flux2_small_decoder_vae(dtype)

        self.flux2_small_decoder_vaes[cache_key] = vae
        return vae

    @contextlib.contextmanager
    def _lenient_flux2_transformer_load(self):
        """Allow the SDNQ Flux2 transformer to load checkpoints that omit guidance weights."""
        from diffusers import Flux2Transformer2DModel

        had_custom_load_state_dict = hasattr(Flux2Transformer2DModel, "load_state_dict")
        original_load_state_dict = getattr(Flux2Transformer2DModel, "load_state_dict", torch.nn.Module.load_state_dict)

        def patched_load_state_dict(model_self, state_dict, strict=True, assign=False):
            return original_load_state_dict(model_self, state_dict, strict=False, assign=assign)

        Flux2Transformer2DModel.load_state_dict = patched_load_state_dict
        try:
            yield
        finally:
            if had_custom_load_state_dict:
                Flux2Transformer2DModel.load_state_dict = original_load_state_dict
            else:
                delattr(Flux2Transformer2DModel, "load_state_dict")

    def _zero_flux2_guidance_weights(self, transformer, dtype, device):
        """Materialize the missing Flux2 guidance branch as zeros for guidance-free checkpoints."""
        if bool(getattr(getattr(transformer, "config", None), "guidance_embeds", True)):
            return

        reference_device = torch.device(device)
        reference_dtype = dtype
        for param in transformer.parameters():
            if not getattr(param, "is_meta", False):
                reference_device = param.device
                reference_dtype = param.dtype
                break

        guidance_paths = [
            ("time_guidance_embed", "guidance_embedder", "linear_1", "weight"),
            ("time_guidance_embed", "guidance_embedder", "linear_2", "weight"),
        ]

        for path in guidance_paths:
            module = transformer
            for attr in path[:-1]:
                module = getattr(module, attr, None)
                if module is None:
                    break
            if module is None:
                continue

            param_name = path[-1]
            param = getattr(module, param_name, None)
            if param is None:
                continue

            zeros = torch.zeros(param.shape, device=reference_device, dtype=reference_dtype)
            setattr(module, param_name, torch.nn.Parameter(zeros, requires_grad=False))

    def _load_flux2_klein_sdnq_transformer(self, model_path: str, dtype, device: str):
        from diffusers import Flux2Transformer2DModel
        from sdnq.loader import load_sdnq_model

        with self._lenient_flux2_transformer_load():
            transformer = load_sdnq_model(
                os.path.join(model_path, "transformer"),
                model_cls=Flux2Transformer2DModel,
                dtype=dtype,
                device=device,
            )

        self._zero_flux2_guidance_weights(transformer, dtype=dtype, device=device)
        return transformer

    def _normalize_flux2_chat_template_input(self, value):
        if isinstance(value, list):
            if all(isinstance(item, dict) and "role" in item for item in value):
                normalized_messages = []
                for message in value:
                    normalized_message = dict(message)
                    content = normalized_message.get("content")
                    if isinstance(content, list) and all(
                        isinstance(part, dict) and part.get("type") == "text" for part in content
                    ):
                        normalized_message["content"] = "".join(part.get("text", "") for part in content)
                    normalized_messages.append(normalized_message)
                return normalized_messages
            return [self._normalize_flux2_chat_template_input(item) for item in value]
        return value

    def _wrap_flux2_chat_template_tokenizer(self, tokenizer):
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if apply_chat_template is None or getattr(tokenizer, "_ufig_flux2_chat_template_wrapped", False):
            return tokenizer

        def wrapped_apply_chat_template(messages, *args, **kwargs):
            normalized_messages = self._normalize_flux2_chat_template_input(messages)
            return apply_chat_template(normalized_messages, *args, **kwargs)

        tokenizer.apply_chat_template = wrapped_apply_chat_template
        tokenizer._ufig_flux2_chat_template_wrapped = True
        return tokenizer

    def _load_flux2_klein_sdnq_text_encoder(self, model_path: str, dtype, device: str):
        from accelerate import init_empty_weights
        from sdnq.loader import (
            apply_sdnq_options_to_model,
            load_files,
            load_sdnq_model,
            post_process_model,
        )
        from sdnq.quantizer import get_quant_args_from_config, sdnq_post_load_quant
        from transformers import AutoConfig, Qwen3ForCausalLM

        text_encoder_path = os.path.join(model_path, "text_encoder")

        try:
            return load_sdnq_model(
                text_encoder_path,
                model_cls=Qwen3ForCausalLM,
                dtype=dtype,
                device=device,
            )
        except UnboundLocalError as exc:
            if "transformers" not in str(exc):
                raise

        with open(os.path.join(text_encoder_path, "quantization_config.json"), "r", encoding="utf-8") as f:
            quantization_config = json.load(f)

        config = AutoConfig.from_pretrained(text_encoder_path, trust_remote_code=True)
        with init_empty_weights():
            text_encoder = Qwen3ForCausalLM(config)
            text_encoder = sdnq_post_load_quant(
                text_encoder,
                torch_dtype=dtype,
                add_skip_keys=False,
                use_dynamic_quantization=False,
                **get_quant_args_from_config(quantization_config),
            )

        files = sorted(
            os.path.join(text_encoder_path, name)
            for name in os.listdir(text_encoder_path)
            if name.endswith(".safetensors")
        )
        state_dict = load_files(
            files,
            key_mapping=getattr(text_encoder, "_checkpoint_conversion_mapping", None),
            device=device,
            method="safetensors",
        )

        tied_keys = getattr(text_encoder, "_tied_weights_keys", None)
        if isinstance(tied_keys, dict):
            for key, value in tied_keys.items():
                if value in state_dict and key not in state_dict:
                    state_dict[key] = state_dict[value]
        elif "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

        text_encoder.load_state_dict(state_dict, assign=True)
        del state_dict

        text_encoder = post_process_model(text_encoder)
        text_encoder = apply_sdnq_options_to_model(text_encoder, dtype=dtype)
        text_encoder.eval()
        return text_encoder

    def is_flux2_small_decoder_active(self, pipe) -> bool:
        vae = getattr(pipe, "vae", None)
        config = getattr(vae, "config", None)
        decoder_block_out_channels = getattr(config, "decoder_block_out_channels", None)
        if decoder_block_out_channels is None:
            return False
        return tuple(decoder_block_out_channels) == (96, 192, 384, 384)

    def _build_flux_pruna_smash_config(
        self,
        device: str,
        optimization_profile: str = "balanced",
        model_key: str = "flux2-klein-sdnq",
    ):
        from pruna import SmashConfig

        # Aggressive caching for max_speed; conservative for stability.
        if optimization_profile == "max_speed":
            fora_interval = 4
            fora_start_step = 2
        elif optimization_profile == "stability":
            fora_interval = 2
            fora_start_step = 4
        else:
            fora_interval = 3
            fora_start_step = 3

        # FLUX.2-klein-4B is a distilled model — it never uses CFG, so the
        # transformer is invoked once per step regardless of int8 / SDNQ
        # quantization format.  fora_backbone_calls_per_step=2 would double
        # latency with no quality benefit.
        backbone_calls = 1

        smash_config = SmashConfig(
            {
                "fora": {
                    "fora_interval": fora_interval,
                    "fora_start_step": fora_start_step,
                    "fora_backbone_calls_per_step": backbone_calls,
                },
                # qkv_diffusers factorizer batches QKV projections into one
                # fused matmul — HuggingFace docs cite up to 4.2x on FLUX.
                "qkv_diffusers": {},
            },
            device=device,
        )
        if hasattr(smash_config, "disable_saving"):
            smash_config.disable_saving()
        return smash_config

    def _prepare_flux_optional_accelerators(
        self,
        pipe,
        *,
        device: str,
        steps: int,
        enable_optional_accelerators: bool,
        mode: str = "single",
        has_lora: bool = False,
        has_pulid: bool = False,
        has_faceswap: bool = False,
        has_pose_control: bool = False,
        has_cpu_offload: bool = False,
        allowed_models: Tuple[str, ...] = ("flux2-klein-sdnq",),
        cache_version: str = "v1",
    ) -> Tuple[Any, Dict[str, Any]]:
        status: Dict[str, Any] = {
            "requested": bool(enable_optional_accelerators),
            "enabled": False,
            "backend": "none",
            "pruna_fora": False,
            "skip_reason": None,
            "small_decoder": self.is_flux2_small_decoder_active(pipe),
        }

        def skip(reason: str) -> Tuple[Any, Dict[str, Any]]:
            status["skip_reason"] = reason
            self.active_optional_accelerator_status = dict(status)
            if status["requested"]:
                print(f"  [Optional Accelerators] Pruna FORA skipped: {reason}")
            return pipe, dict(status)

        if not status["requested"]:
            self.active_optional_accelerator_status = dict(status)
            return pipe, dict(status)
        if self.current_model not in allowed_models:
            return skip(f"unsupported model: {self.current_model}")
        if device != "cuda":
            return skip("requires CUDA")
        if int(steps) < 4:
            return skip("requires at least 4 inference steps")
        if mode != "single":
            return skip(f"disabled for {mode} workflow")
        if has_pose_control:
            return skip("disabled for pose/controlnet workflows")
        if has_cpu_offload:
            return skip("disabled when CPU offload would be enabled")
        if has_lora:
            return skip("disabled when LoRA adapters are active")
        if has_pulid:
            return skip("disabled when PuLID patching is active")
        if has_faceswap:
            return skip("disabled when FaceSwap post-processing is active")

        try:
            from pruna import smash
        except Exception as exc:
            return skip(f"pruna unavailable: {exc}")

        cache_key = (
            id(pipe),
            device,
            f"{self.current_model}-pruna-fora-{cache_version}",
        )
        cached = self.optional_accelerator_pipes.get(cache_key)
        if cached is not None:
            status["enabled"] = True
            status["backend"] = "pruna-fora"
            status["pruna_fora"] = True
            self.active_optional_accelerator_status = dict(status)
            print(
                f"  [Optional Accelerators] Using cached Pruna FORA pipeline for {self.current_model}."
            )
            return cached, dict(status)

        try:
            smash_config = self._build_flux_pruna_smash_config(
                device=device,
                optimization_profile=getattr(self, "optimization_profile", "balanced"),
                model_key=self.current_model,
            )
            # Triton autotuner may write dtype objects via json.dumps during
            # compilation; scope the encoder patch to this block only.
            with json_dtype_patch_context():
                accelerated_pipe = smash(model=copy.deepcopy(pipe), smash_config=smash_config)
        except Exception as exc:
            return skip(f"pruna smash failed: {exc}")

        self.optional_accelerator_pipes[cache_key] = accelerated_pipe
        status["enabled"] = True
        status["backend"] = "pruna-fora"
        status["pruna_fora"] = True
        self.active_optional_accelerator_status = dict(status)
        print(
            f"  [Optional Accelerators] Enabled Pruna FORA for {self.current_model}."
        )
        return accelerated_pipe, dict(status)

    def prepare_flux_sdnq_optional_accelerators(
        self,
        pipe,
        *,
        device: str,
        steps: int,
        enable_optional_accelerators: bool,
        mode: str = "single",
        has_lora: bool = False,
        has_pulid: bool = False,
        has_faceswap: bool = False,
        has_pose_control: bool = False,
        has_cpu_offload: bool = False,
    ) -> Tuple[Any, Dict[str, Any]]:
        return self._prepare_flux_optional_accelerators(
            pipe,
            device=device,
            steps=steps,
            enable_optional_accelerators=enable_optional_accelerators,
            mode=mode,
            has_lora=has_lora,
            has_pulid=has_pulid,
            has_faceswap=has_faceswap,
            has_pose_control=has_pose_control,
            has_cpu_offload=has_cpu_offload,
            allowed_models=("flux2-klein-sdnq",),
            cache_version="v1",
        )

    def prepare_flux_int8_optional_accelerators(
        self,
        pipe,
        *,
        device: str,
        steps: int,
        enable_optional_accelerators: bool,
        mode: str = "single",
        has_lora: bool = False,
        has_pulid: bool = False,
        has_faceswap: bool = False,
        has_pose_control: bool = False,
        has_cpu_offload: bool = False,
    ) -> Tuple[Any, Dict[str, Any]]:
        return self._prepare_flux_optional_accelerators(
            pipe,
            device=device,
            steps=steps,
            enable_optional_accelerators=enable_optional_accelerators,
            mode=mode,
            has_lora=has_lora,
            has_pulid=has_pulid,
            has_faceswap=has_faceswap,
            has_pose_control=has_pose_control,
            has_cpu_offload=has_cpu_offload,
            allowed_models=("flux2-klein-int8",),
            cache_version="v1",
        )

    def compile_pipeline_components(self, pipe, device, cache_key=None, model_key: Optional[str] = None):
        """Apply torch.compile to pipeline components for RTX 3070 Ampere optimization."""
        if device != "cuda" or not torch.cuda.is_available():
            return False

        # Defense-in-depth: never torch.compile SDNQ models.
        # The inductor backend cannot find CUDA kernels for oneDNN
        # quantized ops (onednn.qlinear_prepack), causing:
        #   NotImplementedError: could not find kernel for
        #     onednn.qlinear_prepack.default at dispatch key CUDA
        # optimum-quanto int8 does NOT use oneDNN, so we allow it to try.
        if is_sdnq_model(model_key, pipe):
            return False

        capability = torch.cuda.get_device_capability(0)
        if not capability or capability[0] < 8:
            return False

        key = cache_key or f"{pipe.__class__.__name__}:{device}"
        if self.compiled_models.get(key):
            return False
        if not should_probe_torch_compile(
            cache_key=key,
            device=device,
            model_key=model_key or key,
            pipe=pipe,
            optimization_profile=self.optimization_profile,
            enable_windows_compile_probe=self.enable_windows_compile_probe,
            cuda_runtime=getattr(torch.version, "cuda", None),
        ):
            return False

        try:
            # Triton autotuner may serialize dtype objects with json.dumps
            # during kernel compilation; scope the encoder patch to the
            # compile phase only.
            with json_dtype_patch_context():
                compile_mode = "reduce-overhead" if os.name == "nt" else "max-autotune"
                fullgraph = False if os.name == "nt" else True
                # Prefer regional compilation (compile_repeated_blocks) for dramatically
                # faster compile latency (~7x faster cold start) while keeping the same
                # runtime speedup. Fall back to full torch.compile if unavailable.
                compiled_any = False
                for component_name in ("transformer", "unet"):
                    component = getattr(pipe, component_name, None)
                    if component is None:
                        continue
                    if hasattr(component, "compile_repeated_blocks"):
                        component.compile_repeated_blocks(fullgraph=fullgraph)
                        compiled_any = True
                    else:
                        setattr(pipe, component_name, torch.compile(component, mode=compile_mode, fullgraph=fullgraph))
                        compiled_any = True
                if not compiled_any:
                    return False
                self.compiled_models[key] = True
                record_torch_compile_probe_result(key, True)
                return True
        except Exception as exc:
            print(f"  torch.compile skipped: {exc}")
            record_torch_compile_probe_result(key, False)
            return False

    def _apply_pipeline_memory_policy(self, pipe, model_key: str, device: str):
        self.apply_runtime_memory_policy(
            pipe,
            model_key=model_key,
            device=device,
        )

    def _prepare_component_for_inference(self, component) -> None:
        if component is None:
            return
        if hasattr(component, "eval"):
            try:
                component.eval()
            except Exception:
                pass
        parameters = getattr(component, "parameters", None)
        if not callable(parameters):
            return
        try:
            for param in parameters():
                if getattr(param, "requires_grad", False):
                    param.requires_grad_(False)
        except Exception:
            pass

    def _set_attention_slicing(self, pipe, enabled: bool) -> None:
        if enabled:
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            return
        if hasattr(pipe, "disable_attention_slicing"):
            try:
                pipe.disable_attention_slicing()
            except Exception:
                pass

    def _set_vae_slicing(self, pipe, enabled: bool) -> None:
        if enabled:
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            elif hasattr(getattr(pipe, "vae", None), "enable_slicing"):
                try:
                    pipe.vae.enable_slicing()
                except Exception:
                    pass
            return
        if hasattr(pipe, "disable_vae_slicing"):
            try:
                pipe.disable_vae_slicing()
                return
            except Exception:
                pass
        if hasattr(getattr(pipe, "vae", None), "disable_slicing"):
            try:
                pipe.vae.disable_slicing()
            except Exception:
                pass

    def _set_vae_tiling(self, pipe, enabled: bool) -> None:
        if enabled:
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
            return
        if hasattr(pipe, "disable_vae_tiling"):
            try:
                pipe.disable_vae_tiling()
                return
            except Exception:
                pass
        if hasattr(getattr(pipe, "vae", None), "disable_tiling"):
            try:
                pipe.vae.disable_tiling()
            except Exception:
                pass

    def _configure_zimage_attention_backend(self, pipe) -> Optional[str]:
        transformer = getattr(pipe, "transformer", None)
        set_attention_backend = getattr(transformer, "set_attention_backend", None)
        if not callable(set_attention_backend):
            return None

        for backend in ("_flash_3", "flash"):
            try:
                set_attention_backend(backend)
                print(f"  Z-Image attention backend: {backend}")
                return backend
            except Exception:
                continue
        return None

    def _optimize_quantized_zimage_pipeline(self, pipe) -> None:
        try:
            from sdnq.loader import apply_sdnq_options_to_model
            from sdnq.common import use_triton_mm

            # use_triton_mm is True when SDNQ_USE_TRITON_MM=1 is set at app
            # startup AND the sdnq.triton_mm module is importable. This uses
            # Triton-based INT8 matmul (int_mm) instead of torch._int_mm,
            # avoiding onednn.qlinear_prepack CUDA dispatch errors.
            # Note: use_torch_compile is False on Windows (SDNQ_USE_TORCH_COMPILE=0)
            # to prevent torch.compile/inductor from hanging during Triton kernel
            # compilation. Quantized matmul still works without torch.compile.
            if use_triton_mm and (torch.cuda.is_available() or torch.xpu.is_available()):
                print("  Z-Image INT8: applying quantized matmul options (Triton-based, no torch.compile)...")
                if getattr(pipe, "transformer", None) is not None:
                    print("    - Optimizing transformer...")
                    pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
                if getattr(pipe, "text_encoder", None) is not None:
                    print("    - Optimizing text_encoder...")
                    pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
                print("  Z-Image INT8 quantized matmul enabled (Triton-based).")
            else:
                # Triton MM not available — use fallback without quantized matmul
                print("  Z-Image INT8: Triton int_mm unavailable, using dequantize fallback.")
                try:
                    if getattr(pipe, "transformer", None) is not None:
                        pipe.transformer = apply_sdnq_options_to_model(pipe.transformer)
                    if getattr(pipe, "text_encoder", None) is not None:
                        pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder)
                except Exception:
                    pass
        except Exception as exc:
            print(f"  Z-Image INT8 optimization skipped: {exc}")

    def _should_compile_flux_sdnq_vae(self, device: str) -> bool:
        """Avoid blocking SDNQ pipeline load on expensive Windows/CUDA 13 VAE compiles."""
        if str(device).lower() != "cuda" or not torch.cuda.is_available():
            return False

        override = os.environ.get("UFIG_COMPILE_FLUX_SDNQ_VAE")
        if override is not None:
            return override.strip().lower() in {"1", "true", "yes", "on"}

        # On Windows/CUDA 13, torch.compile of the VAE encode/decode path can
        # dominate (or hang) first Image Editing load. SDNQ already uses Triton
        # quantized matmul + attention backends for the expensive transformer.
        if os.name == "nt" and is_cuda13_runtime(getattr(torch.version, "cuda", None)):
            return False

        return self.optimization_profile != "stability"

    def get_cached_runtime_memory_policy(
        self,
        model_key: Optional[str],
        mode: str,
        width: Optional[int],
        height: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        key = (str(model_key or ""), str(mode or ""), int(width or 0), int(height or 0))
        cached = self.runtime_memory_policy_cache.get(key)
        return dict(cached) if cached else None

    def cache_runtime_memory_policy(
        self,
        model_key: Optional[str],
        mode: str,
        width: Optional[int],
        height: Optional[int],
        policy: Dict[str, Any],
    ) -> None:
        key = (str(model_key or ""), str(mode or ""), int(width or 0), int(height or 0))
        self.runtime_memory_policy_cache[key] = dict(policy)

    def apply_runtime_memory_policy(
        self,
        pipe,
        model_key: Optional[str],
        device: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        attention_slicing: Optional[bool] = None,
        vae_slicing: Optional[bool] = None,
        vae_tiling: Optional[bool] = None,
        oom_retry: bool = False,
    ) -> Dict[str, Any]:
        vram_gb = get_device_vram_gb(device)
        enable_attn_slicing = should_use_attention_slicing(
            device=device,
            model_key=model_key,
            pipe=pipe,
            vram_gb=vram_gb,
            optimization_profile=self.optimization_profile,
            oom_retry=oom_retry,
        ) if attention_slicing is None else bool(attention_slicing)
        enable_vae_slicing = should_use_vae_slicing(
            device=device,
            vram_gb=vram_gb,
            optimization_profile=self.optimization_profile,
            oom_retry=oom_retry,
        ) if vae_slicing is None else bool(vae_slicing)
        enable_vae_tiling = should_use_vae_tiling(
            device=device,
            model_key=model_key,
            pipe=pipe,
            vram_gb=vram_gb,
            width=width,
            height=height,
            optimization_profile=self.optimization_profile,
        ) if vae_tiling is None else bool(vae_tiling)

        next_policy = {
            "attention_slicing": enable_attn_slicing,
            "vae_slicing": enable_vae_slicing,
            "vae_tiling": enable_vae_tiling,
            "width": int(width or 0),
            "height": int(height or 0),
            "oom_retry": bool(oom_retry),
        }
        next_signature = (
            id(pipe),
            str(model_key or ""),
            str(device or ""),
            next_policy["attention_slicing"],
            next_policy["vae_slicing"],
            next_policy["vae_tiling"],
            next_policy["width"],
            next_policy["height"],
            next_policy["oom_retry"],
        )
        force_apply = bool(getattr(pipe, "_ufig_memory_policy_dirty", False))
        if force_apply or self._active_runtime_policy_signature != next_signature:
            self._set_attention_slicing(pipe, enable_attn_slicing)
            self._set_vae_slicing(pipe, enable_vae_slicing)
            self._set_vae_tiling(pipe, enable_vae_tiling)
            self._active_runtime_policy_signature = next_signature
            if force_apply:
                try:
                    pipe._ufig_memory_policy_dirty = False
                except Exception:
                    pass

        self.active_runtime_memory_policy = next_policy
        return dict(self.active_runtime_memory_policy)

    def load_zimage_pipeline(self, device="mps", use_full_model=False):
        """Load the RTX 3070-safe Z-Image INT8 pipeline."""
        from diffusers import ZImagePipeline, ZImageTransformer2DModel, FlowMatchEulerDiscreteScheduler
        from sdnq.loader import load_sdnq_model
        from transformers import AutoTokenizer, Qwen3ForCausalLM

        dtype = torch.bfloat16 if device in ["cuda", "mps"] else torch.float32

        print(f"Loading Z-Image-Turbo (int8) on {device}...")
        # Load SDNQ components manually to bypass diffusers' auto-quantization
        # detection which doesn't recognize the "sdnq" quant_method.
        print("  Step 1/5: Downloading/resolving model files...")
        model_path = snapshot_download(ZIMAGE_INT8_REPO_ID)

        print("  Step 2/5: Loading SDNQ transformer (this may take a minute)...")
        transformer = load_sdnq_model(
            os.path.join(model_path, "transformer"),
            model_cls=ZImageTransformer2DModel,
            dtype=dtype,
            device=device,
        )
        print("    Transformer loaded.")

        print("  Step 3/5: Loading SDNQ text_encoder (this may take a minute)...")
        try:
            text_encoder = load_sdnq_model(
                os.path.join(model_path, "text_encoder"),
                model_cls=Qwen3ForCausalLM,
                dtype=dtype,
                device=device,
            )
            print("    Text encoder loaded.")
        except UnboundLocalError as exc:
            # Fallback: load_sdnq_model may fail for some transformers versions.
            # Use the same manual SDNQ loading path as FLUX text_encoder.
            if "transformers" not in str(exc):
                raise
            print("    Primary loader failed, using fallback manual SDNQ loading...")
            from sdnq.loader import apply_sdnq_options_to_model, load_files, post_process_model
            from sdnq.quantizer import get_quant_args_from_config, sdnq_post_load_quant
            from accelerate import init_empty_weights
            from transformers import AutoConfig

            text_encoder_path = os.path.join(model_path, "text_encoder")
            with open(os.path.join(text_encoder_path, "quantization_config.json"), "r", encoding="utf-8") as f:
                quantization_config = json.load(f)
            config = AutoConfig.from_pretrained(text_encoder_path, trust_remote_code=True)
            with init_empty_weights():
                text_encoder = Qwen3ForCausalLM(config)
                text_encoder = sdnq_post_load_quant(
                    text_encoder,
                    torch_dtype=dtype,
                    add_skip_keys=False,
                    use_dynamic_quantization=False,
                    **get_quant_args_from_config(quantization_config),
                )
            files = sorted(
                os.path.join(text_encoder_path, name)
                for name in os.listdir(text_encoder_path)
                if name.endswith(".safetensors")
            )
            state_dict = load_files(files, device=device, method="safetensors")
            tied_keys = getattr(text_encoder, "_tied_weights_keys", None)
            if isinstance(tied_keys, dict):
                for key, value in tied_keys.items():
                    if value in state_dict and key not in state_dict:
                        state_dict[key] = state_dict[value]
            elif "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
                state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
            text_encoder.load_state_dict(state_dict, assign=True)
            del state_dict
            text_encoder = post_process_model(text_encoder)
            text_encoder = apply_sdnq_options_to_model(text_encoder, dtype=dtype)
            text_encoder.eval()
            print("    Text encoder loaded (fallback).")

        print("  Step 4/5: Assembling pipeline...")
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_path, "tokenizer"),
            use_fast=False,
        )

        # Load the pipeline skeleton without the SDNQ components,
        # then inject our pre-loaded ones.
        pipe = ZImagePipeline.from_pretrained(
            ZIMAGE_INT8_REPO_ID,
            transformer=None,
            text_encoder=None,
            tokenizer=None,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        pipe.transformer = transformer
        pipe.text_encoder = text_encoder
        pipe.tokenizer = tokenizer
        self._prepare_component_for_inference(pipe.transformer)
        self._prepare_component_for_inference(pipe.text_encoder)
        self._prepare_component_for_inference(pipe.vae)

        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            use_beta_sigmas=True,
        )

        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None

        self._configure_zimage_attention_backend(pipe)
        print("    Pipeline assembled. Moving to device...")
        pipe.to(device)
        print("  Step 5/5: Applying SDNQ optimizations...")
        self._optimize_quantized_zimage_pipeline(pipe)

        if device == "cuda" and hasattr(pipe, "vae") and pipe.vae is not None:
            try:
                pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
            except Exception:
                pass

        for component in (getattr(pipe, "vae", None),):
            if component is not None and hasattr(component, "fuse_qkv_projections"):
                try:
                    component.fuse_qkv_projections()
                except Exception:
                    pass
        # Skip fuse_qkv_projections for SDNQ quantized transformer.
        # Fusing QKV on quantized weights may interfere with SDNQ's
        # custom weight layout and could trigger oneDNN operations.

        compile_key = "zimage-int8"
        self._apply_pipeline_memory_policy(pipe, model_key=compile_key, device=device)

        self.compile_pipeline_components(pipe, device, compile_key, model_key=compile_key)

        # Attention backend cascade (SageAttention → xFormers → SDPA). Same
        # treatment as the FLUX.2-klein SDNQ loader — this is the single
        # biggest speedup for Z-Image Turbo on RTX 3070 (Ampere) because the
        # int8 transformer spends most of its wall-clock inside attention,
        # and the default math SDPA kernel leaves ~2x on the table.
        self._install_attention_backend(pipe, device)

        # Compile VAE encoder for the Z-Image image-to-image code path. The
        # decoder compile already happens via compile_pipeline_components for
        # the transformer; the encoder is only hit on img2img so we compile it
        # lazily here to avoid slowing down the first text-to-image call.
        if device == "cuda" and torch.cuda.is_available() and hasattr(pipe, "vae") and pipe.vae is not None:
            try:
                with json_dtype_patch_context():
                    compile_mode = "reduce-overhead" if os.name == "nt" else "max-autotune"
                    fullgraph = False if os.name == "nt" else True
                    vae_encode_key = f"zimage-int8-vae-encode:{device}"
                    if not self.compiled_models.get(vae_encode_key) and hasattr(pipe.vae, "encode"):
                        pipe.vae.encode = torch.compile(pipe.vae.encode, mode=compile_mode, fullgraph=fullgraph)
                        self.compiled_models[vae_encode_key] = True
                        print("  Z-Image VAE encoder compiled (image-edit path).")
                    vae_decode_key = f"zimage-int8-vae-decode:{device}"
                    if not self.compiled_models.get(vae_decode_key) and hasattr(pipe.vae, "decode"):
                        pipe.vae.decode = torch.compile(pipe.vae.decode, mode=compile_mode, fullgraph=fullgraph)
                        self.compiled_models[vae_decode_key] = True
                        print("  Z-Image VAE decoder compiled.")
            except Exception as exc:
                print(f"  Z-Image VAE compile skipped: {exc}")

        print("  Z-Image-Turbo (int8) load complete.")
        return pipe

    def get_zimage_img2img_pipeline(self, device="mps", use_full_model=False):
        """Return a cached Z-Image img2img wrapper sharing the active base pipeline."""
        from diffusers import AutoPipelineForImage2Image, FlowMatchEulerDiscreteScheduler

        model_key = "zimage-int8"
        cache_key = f"{model_key}:{device}"
        cached = self.zimage_img2img_pipes.get(cache_key)
        if cached is not None:
            return cached

        if self.pipe is None or self.current_model != model_key or self.current_device != device:
            self.pipe = self.load_zimage_pipeline(device=device, use_full_model=use_full_model)
            self.current_device = device
            self.current_model = model_key

        img2img_pipe = AutoPipelineForImage2Image.from_pipe(self.pipe)
        img2img_pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            img2img_pipe.scheduler.config,
            use_beta_sigmas=True,
        )

        if hasattr(img2img_pipe, "safety_checker"):
            img2img_pipe.safety_checker = None

        self._apply_pipeline_memory_policy(img2img_pipe, model_key=model_key, device=device)

        self.zimage_img2img_pipes[cache_key] = img2img_pipe
        return img2img_pipe

    def should_enable_cpu_offload(
        self,
        model_key: Optional[str],
        enable_pose_preservation: bool,
        device: str,
    ) -> bool:
        if device != "cuda":
            return False
        if not enable_pose_preservation:
            return False
        if not is_flux_model(model_key):
            return False
        if is_sdnq_or_quantized(model_key, None):
            return False
        # Quantized models (SDNQ 4-bit, int8) are small enough to fit in GPU
        # without offloading. CPU offloading moves model components between
        # CPU/GPU each step, which is catastrophically slow (~5+ min generation).
        # Only offload when VRAM is genuinely insufficient.
        vram_gb = get_device_vram_gb(device)
        if vram_gb is not None:
            return vram_gb < 12
        return True

    def load_flux2_klein_pipeline(self, device="mps"):
        """Load FLUX.2-klein-4B with int8 quantized transformer and text encoder."""
        import gc
        from transformers import Qwen3ForCausalLM, AutoTokenizer, AutoConfig
        from optimum.quanto import requantize
        from accelerate import init_empty_weights
        from src.image.quantized_flux2 import QuantizedFlux2Transformer2DModel

        print(f"Loading FLUX.2-klein-4B (int8 quantized) on {device}...")

        # Defensive memory clearing before loading large quantized models
        gc.collect()
        if torch.cuda.is_available() and device == "cuda":
            torch.cuda.empty_cache()

        print("  Step 1/6: Downloading/resolving model files...")
        model_path = snapshot_download("aydin99/FLUX.2-klein-4B-int8")

        dtype = torch.bfloat16 if device in ["cuda", "mps"] else torch.float32

        # For quanto-requantized models, only move to device. Do NOT call
        # .to(dtype=...) — quanto replaces layers with QLinear which may
        # segfault when asked to recast their quantized weights.
        def _safe_move(model):
            # Detect quanto-wrapped models (QuantizedDiffusersModel or Qwen3ForCausalLM
            # after requantize) and skip dtype conversion entirely.
            # Quanto models use QLinear layers that segfault when recast via .to(dtype=...).
            is_quanto_model = (
                hasattr(model, "_wrapped")
                or hasattr(model, "qconfig")
                or getattr(model, "is_quantized", False)
            )
            if is_quanto_model:
                try:
                    model.to(device=device)
                except Exception as exc2:
                    print(f"    Warning: Could not move model to {device}: {exc2}")
                return

            try:
                model.to(device=device, dtype=dtype)
                return
            except TypeError:
                # Combined device+dtype failed (typical for quanto-wrapped models)
                # — move device only and skip dtype.
                try:
                    model.to(device=device)
                except Exception as exc2:
                    print(f"    Warning: Could not move model to {device}: {exc2}")
                    return
            except Exception:
                # Any other failure: try device-only as a last resort.
                try:
                    model.to(device=device)
                except Exception as exc2:
                    print(f"    Warning: Could not move model to {device}: {exc2}")
                    return

        print("  Step 2/6: Loading int8 quantized transformer...")
        qtransformer = QuantizedFlux2Transformer2DModel.from_pretrained(model_path)
        _safe_move(qtransformer)
        print("    Transformer loaded.")

        # Clear cache between large component loads to minimize peak VRAM
        gc.collect()
        if torch.cuda.is_available() and device == "cuda":
            torch.cuda.empty_cache()

        print("  Step 3/6: Loading int8 quantized text encoder...")
        config = AutoConfig.from_pretrained(f"{model_path}/text_encoder", trust_remote_code=True)
        with init_empty_weights():
            text_encoder = Qwen3ForCausalLM(config)

        with open(f"{model_path}/text_encoder/quanto_qmap.json", "r", encoding="utf-8") as f:
            qmap = json.load(f)
        state_dict = load_file(f"{model_path}/text_encoder/model.safetensors")
        requantize(text_encoder, state_dict=state_dict, quantization_map=qmap)
        text_encoder.eval()
        _safe_move(text_encoder)
        print("    Text encoder loaded.")

        # Clear cache before loading VAE to keep peak VRAM low
        gc.collect()
        if torch.cuda.is_available() and device == "cuda":
            torch.cuda.empty_cache()

        print("  Step 4/6: Loading tokenizer...")
        tokenizer = self._wrap_flux2_chat_template_tokenizer(
            AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")
        )
        print("    Tokenizer loaded.")

        print("  Step 5/6: Loading small decoder VAE...")
        vae = self.get_flux2_small_decoder_vae(dtype)
        print(f"    FLUX VAE: Using small decoder VAE (cached={str(dtype) in self.flux2_small_decoder_vaes})")
        Flux2PipelineClass = self.get_flux2_pipeline_class()
        pipe_kwargs = {
            "transformer": None,
            "text_encoder": None,
            "tokenizer": None,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        pipe_kwargs["vae"] = vae

        print("  Step 5b/6: Assembling pipeline skeleton...")
        pipe = Flux2PipelineClass.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            **pipe_kwargs,
        )

        # Set custom components BEFORE moving to device to avoid
        # transferring the default (unused) components to GPU first.
        pipe.transformer = qtransformer._wrapped
        pipe.text_encoder = text_encoder
        pipe.tokenizer = tokenizer
        self._prepare_component_for_inference(pipe.transformer)
        self._prepare_component_for_inference(pipe.text_encoder)
        self._prepare_component_for_inference(pipe.vae)
        print("  Step 5c/6: Moving pipeline to device...")
        pipe.to(device)

        # Apply channels_last memory format to VAE for faster conv operations
        if device == "cuda" and hasattr(pipe, "vae") and pipe.vae is not None:
            try:
                pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
            except Exception:
                pass

        # Fuse QKV projection matrices for faster attention (single matmul instead of 3)
        for component in (getattr(pipe, "transformer", None), getattr(pipe, "vae", None)):
            if component is not None and hasattr(component, "fuse_qkv_projections"):
                try:
                    component.fuse_qkv_projections()
                except Exception:
                    pass

        print("  Step 6a/6: Applying memory policy...")
        self._apply_pipeline_memory_policy(pipe, model_key="flux2-klein-int8", device=device)

        print("  Step 6b/6: Compiling pipeline components...")
        self.compile_pipeline_components(
            pipe,
            device,
            "flux2-klein-int8",
            model_key="flux2-klein-int8",
        )

        print("  Step 6c/6: Installing attention backend...")
        # Attention backend cascade (SageAttention → xFormers → SDPA).
        self._install_attention_backend(pipe, device)
        print("  FLUX.2-klein-4B (int8) loaded successfully.")
        return pipe

    def load_flux2_klein_sdnq_pipeline(self, device="mps", model_id="Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic"):
        """Load FLUX.2-klein with SDNQ 4-bit quantization."""
        from sdnq.loader import apply_sdnq_options_to_model
        from transformers import AutoTokenizer
        print(f"Loading {model_id} on {device}...")

        model_path = snapshot_download(model_id)

        dtype = torch.bfloat16 if device in ["cuda", "mps"] else torch.float32
        vae = self.get_flux2_small_decoder_vae(dtype)
        Flux2PipelineClass = self.get_flux2_pipeline_class()
        tokenizer = self._wrap_flux2_chat_template_tokenizer(
            AutoTokenizer.from_pretrained(
                f"{model_path}/tokenizer",
                use_fast=False,
            )
        )

        transformer = self._load_flux2_klein_sdnq_transformer(model_path, dtype=dtype, device=device)
        text_encoder = self._load_flux2_klein_sdnq_text_encoder(model_path, dtype=dtype, device=device)

        pipe = Flux2PipelineClass.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            transformer=None,
            text_encoder=None,
            tokenizer=None,
            torch_dtype=dtype,
            vae=vae,
            low_cpu_mem_usage=True,
        )

        # Set custom components BEFORE moving to device to avoid
        # transferring the default (unused) components to GPU first.
        pipe.transformer = transformer
        pipe.text_encoder = text_encoder
        pipe.tokenizer = tokenizer
        self._prepare_component_for_inference(pipe.transformer)
        self._prepare_component_for_inference(pipe.text_encoder)
        self._prepare_component_for_inference(pipe.vae)
        pipe.to(device)

        try:
            from sdnq.common import use_triton_mm

            # use_triton_mm is True when SDNQ_USE_TRITON_MM=1 is set at app
            # startup AND the sdnq.triton_mm module is importable. This uses
            # Triton-based INT8 matmul (int_mm) instead of torch._int_mm,
            # avoiding onednn.qlinear_prepack CUDA dispatch errors.
            # Note: use_torch_compile is False on Windows (SDNQ_USE_TORCH_COMPILE=0)
            # to prevent torch.compile/inductor from hanging during Triton kernel
            # compilation. Quantized matmul still works without torch.compile.
            if use_triton_mm and (torch.cuda.is_available() or torch.xpu.is_available()):
                print("  FLUX SDNQ: applying quantized matmul options (Triton-based, no torch.compile)...")
                pipe.transformer = apply_sdnq_options_to_model(
                    pipe.transformer,
                    use_quantized_matmul=True,
                )
                pipe.text_encoder = apply_sdnq_options_to_model(
                    pipe.text_encoder,
                    use_quantized_matmul=True,
                )
                print("  FLUX SDNQ quantized matmul enabled (Triton-based).")
            else:
                print("  FLUX SDNQ: Triton int_mm unavailable, using dequantize fallback.")
                try:
                    pipe.transformer = apply_sdnq_options_to_model(pipe.transformer)
                    pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder)
                except Exception:
                    pass
        except Exception:
            pass

        # Apply channels_last memory format to VAE for faster conv operations
        if device == "cuda" and hasattr(pipe, "vae") and pipe.vae is not None:
            try:
                pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
            except Exception:
                pass

        # Fuse QKV projection matrices for faster attention (single matmul instead of 3)
        # Skip transformer for SDNQ quantized models — fusing QKV on quantized
        # weights may interfere with SDNQ's custom weight layout.
        for component in (getattr(pipe, "vae", None),):
            if component is not None and hasattr(component, "fuse_qkv_projections"):
                try:
                    component.fuse_qkv_projections()
                except Exception:
                    pass

        self._apply_pipeline_memory_policy(pipe, model_key="flux2-klein-sdnq", device=device)

        # Attention backend: prefer SageAttention (2-5x faster via INT8 QK + FP16
        # PV), fall back to xFormers (memory-efficient attention, Ampere-friendly),
        # else leave the default SDPA kernel in place. Single dispatch point so
        # we never silently run on the slow math SDPA kernel.
        self._install_attention_backend(pipe, device)

        # Keep SDNQ cold load responsive. VAE compile is opt-in on
        # Windows/CUDA 13 because first-run inductor compilation can dominate
        # or hang the Image Editing load path.
        if self._should_compile_flux_sdnq_vae(device) and hasattr(pipe, "vae") and pipe.vae is not None:
            try:
                with json_dtype_patch_context():
                    compile_mode = "reduce-overhead" if os.name == "nt" else "max-autotune"
                    fullgraph = False if os.name == "nt" else True
                    vae_decode_key = f"flux2-sdnq-vae-decode:{device}"
                    if not self.compiled_models.get(vae_decode_key):
                        pipe.vae.decode = torch.compile(pipe.vae.decode, mode=compile_mode, fullgraph=fullgraph)
                        self.compiled_models[vae_decode_key] = True
                        print("  FLUX SDNQ VAE decoder compiled.")
                    vae_encode_key = f"flux2-sdnq-vae-encode:{device}"
                    if not self.compiled_models.get(vae_encode_key) and hasattr(pipe.vae, "encode"):
                        pipe.vae.encode = torch.compile(pipe.vae.encode, mode=compile_mode, fullgraph=fullgraph)
                        self.compiled_models[vae_encode_key] = True
                        print("  FLUX SDNQ VAE encoder compiled (image-edit path).")
            except Exception as exc:
                print(f"  FLUX SDNQ VAE compile skipped: {exc}")

        return pipe

    def load_controlnet_union(self, device="cuda"):
        """Load FLUX ControlNet Union."""
        if self.controlnet_union is not None:
            return self.controlnet_union

        try:
            from diffusers import FluxControlNetModel
            print(f"Loading FLUX ControlNet Union on {device}...")
            dtype = torch.bfloat16 if device in ["cuda", "mps"] else torch.float32
            self.controlnet_union = FluxControlNetModel.from_pretrained(
                "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            self.controlnet_union.to(device)
            return self.controlnet_union
        except Exception as e:
            print(f"  Warning: Failed to load ControlNet Union: {e}")
            return None

    def load_pipeline(self, model_choice: str, device: str = "mps"):
        """Main entry point to load or switch pipelines."""
        resolved_model_choice, fallback_reason = resolve_model_choice_for_device(
            model_choice,
            device,
            vram_gb=get_device_vram_gb(device),
        )
        self.last_model_fallback_reason = fallback_reason
        model_choice = resolved_model_choice

        if "Z-Image" in model_choice:
            model_type = "zimage-int8"
        elif "Int8" in model_choice:
            model_type = "flux2-klein-int8"
        elif "4bit SDNQ" in model_choice:
            model_type = "flux2-klein-sdnq"
        elif "FLUX" in model_choice:
            model_type = "flux2-klein-int8"
        else:
            model_type = "zimage-int8"

        if self.pipe is not None and self.current_device == device and self.current_model == model_type:
            return self.pipe

        if self.pipe is not None:
            print(f"Switching from {self.current_model} to {model_type}...")
            self.cleanup_auxiliary_models()
            del self.pipe
            self.zimage_img2img_pipes.clear()
            # Preserve VAE cache when switching between flux2 models (they share the same VAE)
            old_is_flux = is_flux_model(self.current_model)
            new_is_flux = is_flux_model(model_type)
            if not (old_is_flux and new_is_flux):
                self.flux2_small_decoder_vaes.clear()
            # Invalidate stale torch.compile cache entries. The transformer
            # inside the old pipe has just been destroyed, so any compile
            # keys guarding transformer compilation must be dropped or the
            # fresh pipeline silently ships with an uncompiled transformer.
            # VAE compile keys are dropped whenever the matching VAE cache
            # was cleared (non-flux transitions drop all VAE keys; flux
            # -> flux keeps flux2-sdnq-vae-* keys because the VAE object
            # is preserved via self.flux2_small_decoder_vaes).
            _preserve_flux_vae_keys = old_is_flux and new_is_flux
            for _key in list(self.compiled_models.keys()):
                if _preserve_flux_vae_keys and "flux2-sdnq-vae-" in _key:
                    continue
                self.compiled_models.pop(_key, None)
            self.optional_accelerator_pipes.clear()
            self.active_runtime_memory_policy = {}
            self._active_runtime_policy_signature = None
            self.active_optional_accelerator_status = {}
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()

        if model_type == "flux2-klein-int8":
            try:
                self.pipe = self.load_flux2_klein_pipeline(device)
            except Exception as exc:
                print(f"\n  [CRITICAL] FLUX int8 pipeline failed to load: {exc}")
                print("  Falling back to SDNQ 4-bit pipeline (slower but more stable on this environment).\n")
                model_type = "flux2-klein-sdnq"
                self.last_model_fallback_reason = (
                    f"FLUX.2-klein-4B (int8) loading failed ({exc}); fell back to SDNQ 4-bit. "
                    "This usually means optimum.quanto is incompatible with your PyTorch build."
                )
                self.pipe = self.load_flux2_klein_sdnq_pipeline(device)
        elif model_type == "flux2-klein-sdnq":
            self.pipe = self.load_flux2_klein_sdnq_pipeline(device)
        elif model_type == "zimage-int8":
            self.pipe = self.load_zimage_pipeline(device, use_full_model=False)
        else:
            self.pipe = self.load_zimage_pipeline(device, use_full_model=False)

        self.current_device = device
        self.current_model = model_type
        return self.pipe

    def load_lora(self, lora_file, lora_strength: float, device: str):
        """Load or update LoRA adapter."""
        from src.image import lora_zimage

        is_quantized = any(q in str(self.current_model) for q in ["sdnq", "int8"])
        if not is_quantized:
            return f"LoRA only supported with FLUX quantized models and Z-Image Turbo (Int8) (current: {self.current_model})"

        if lora_file is None or lora_file == "":
            if self.current_lora_path is not None:
                if self.current_lora_network is not None:
                    self.current_lora_network.remove()
                    self.current_lora_network = None
                elif not is_quantized:
                    # Only call unload_lora_weights for native (non-quantized) LoRA.
                    # Quantized models use custom lora_zimage which doesn't
                    # register with diffusers' LoRA system.
                    self.pipe.unload_lora_weights()
                self.current_lora_path = None
            return "No LoRA loaded"

        lora_path = lora_file if isinstance(lora_file, str) else lora_file.name
        if not os.path.exists(lora_path):
            return f"LoRA file not found: {lora_path}"

        if not lora_path.endswith(".safetensors"):
            return "Please select a .safetensors file"

        if self.current_lora_path == lora_path:
            if self.current_lora_network is not None:
                self.current_lora_network.multiplier = lora_strength
            else:
                self.pipe.set_adapters(["default"], adapter_weights=[lora_strength])
            return f"Updated LoRA strength to {lora_strength}"

        if self.current_lora_path is not None:
            if self.current_lora_network is not None:
                self.current_lora_network.remove()
                self.current_lora_network = None
            elif not is_quantized:
                # Only call unload_lora_weights for native (non-quantized) LoRA.
                self.pipe.unload_lora_weights()

        try:
            if is_quantized:
                self.current_lora_network = lora_zimage.load_lora_for_pipeline(
                    self.pipe, lora_path, multiplier=lora_strength, device=device
                )
            else:
                self.pipe.load_lora_weights(lora_path, adapter_name="default")
                self.pipe.set_adapters(["default"], adapter_weights=[lora_strength])
            self.current_lora_path = lora_path
            return f"Loaded LoRA: {os.path.basename(lora_path)} (strength={lora_strength})"
        except Exception as e:
            print(f"Error loading LoRA: {e}")
            self.current_lora_path = None
            self.current_lora_network = None
            return f"Error loading LoRA: {e}"

    def unload_controlnet_union(self):
        if self.controlnet_union is not None:
            del self.controlnet_union
            self.controlnet_union = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def unload_lora(self):
        """Unload LoRA adapter to free memory."""
        if self.current_lora_network is not None:
            try:
                self.current_lora_network.remove()
            except Exception:
                pass
            self.current_lora_network = None
        self.current_lora_path = None
        if self.pipe is not None and hasattr(self.pipe, 'unload_lora_weights'):
            try:
                self.pipe.unload_lora_weights()
            except Exception:
                pass

    def cleanup_auxiliary_models(self):
        """Unload auxiliary models (ControlNet Union, LoRA adapters, etc.) to free VRAM."""
        self.unload_controlnet_union()
        self.unload_lora()
        self._restore_original_sdpa()
        self.active_optional_accelerator_status = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _install_sage_attention_patch(self):
        """Install SageAttention as a module-level monkey-patch of F.scaled_dot_product_attention.

        Uses a module-level guard so the true original is captured exactly once even
        if multiple PipelineManager instances call this concurrently. Re-entrant
        calls are a no-op.
        """
        from sageattention import sageattn
        import torch.nn.functional as F

        if getattr(F.scaled_dot_product_attention, "_sage_patched", False):
            return

        # Stash the true original on the module itself, not on self, so every
        # instance sees the same reference and restoration is idempotent.
        original_sdpa = getattr(F, "_sage_original_sdpa", None) or F.scaled_dot_product_attention
        F._sage_original_sdpa = original_sdpa

        def sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *args, **kwargs):
            # Fall back to original SDPA if shapes are incompatible
            if attn_mask is not None or dropout_p > 0.0:
                return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, *args, **kwargs)
            try:
                return sageattn(query, key, value, is_causal=is_causal, tensor_layout="HND")
            except Exception:
                return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, *args, **kwargs)

        sage_sdpa._sage_patched = True
        F.scaled_dot_product_attention = sage_sdpa
        print("  SageAttention enabled (INT8 QK + FP16 PV attention).")

    def _restore_original_sdpa(self):
        """Restore original F.scaled_dot_product_attention if SageAttention was patched."""
        import torch.nn.functional as F
        original = getattr(F, "_sage_original_sdpa", None)
        if original is not None and getattr(F.scaled_dot_product_attention, "_sage_patched", False):
            F.scaled_dot_product_attention = original
            F._sage_original_sdpa = None
            print("  SageAttention monkey-patch removed (restoring original SDPA).")

    def _install_xformers_fallback(self, pipe) -> bool:
        """
        Enable xFormers memory-efficient attention as a VRAM-saving fallback.

        Used when SageAttention is unavailable (e.g. no CUDA 12+ / no
        sageattention wheel installed). xFormers' fused attention kernel is
        well-supported on Ampere (RTX 3070 sm_86) and routinely saves
        ~15-25% peak VRAM during the transformer forward pass on FLUX.2,
        which is the difference between "fits on an 8 GB card" and "OOM" at
        1024² resolution.

        Returns True if the fallback was enabled, False otherwise.
        """
        try:
            import xformers  # noqa: F401
        except ImportError:
            return False
        if pipe is None or not hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            return False
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("  xFormers memory-efficient attention enabled (Ampere-friendly fallback).")
            return True
        except Exception as exc:
            # Rare: xformers installed but incompatible with the pipeline's
            # processor class. Fail quietly — SDPA math kernel still works.
            print(f"  xFormers fallback skipped: {exc}")
            return False

    def _install_attention_backend(self, pipe, device: str) -> None:
        """
        Attempt SageAttention first, then xFormers, then native SDPA.

        Keeping this sequence explicit makes the flux2-klein loaders
        idempotent and avoids silently running with the slow math SDPA
        kernel on 3070-class hardware where SageAttention isn't installed.
        """
        if device != "cuda" or not torch.cuda.is_available():
            return
        try:
            self._install_sage_attention_patch()
            return
        except ImportError:
            pass
        except Exception as exc:
            print(f"  SageAttention skipped: {exc}")
        self._install_xformers_fallback(pipe)

    def check_model_exists(self, model_repo_id: str) -> bool:
        """Check if model exists in HuggingFace cache."""
        try:
            cached_path = try_to_load_from_cache(model_repo_id, filename="model_index.json")
            if cached_path is None:
                cached_path = try_to_load_from_cache(model_repo_id, filename="config.json")
            return cached_path is not None
        except Exception:
            return False

    def get_model_repos_for_choice(self, model_choice: str) -> list:
        """Get model repositories for a given choice."""
        if any(token in model_choice for token in ["9B", "NVFP4"]):
            return [ZIMAGE_INT8_REPO_ID]
        if "Z-Image" in model_choice:
            return [ZIMAGE_INT8_REPO_ID]
        elif "Int8" in model_choice:
            return ["aydin99/FLUX.2-klein-4B-int8", "black-forest-labs/FLUX.2-small-decoder"]
        return [
            "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
            "black-forest-labs/FLUX.2-klein-4B",
            "black-forest-labs/FLUX.2-small-decoder",
        ]

    def download_file(self, url, path, description="File", progress=None):
        """Download a file with progress tracking."""
        import requests
        from src.config import CIVITAI_API_TOKEN

        print(f"Downloading {description} from {url} to {path}")
        try:
            headers = {}
            # Add Civitai API token if URL is from Civitai and token is available
            if "civitai.com" in url and CIVITAI_API_TOKEN:
                headers["Authorization"] = f"Bearer {CIVITAI_API_TOKEN}"

            response = requests.get(url, stream=True, timeout=30, headers=headers)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                downloaded = 0
                for data in response.iter_content(1024 * 1024):
                    f.write(data)
                    downloaded += len(data)
                    if total_size > 0 and progress:
                        progress(downloaded / total_size, desc=f"Downloading {description}...")
            return True
        except Exception as e:
            print(f"ERROR: Failed to download {description}: {e}")
            if os.path.exists(path): os.remove(path)
            raise

    def ensure_models_downloaded(self, model_choice: str, **kwargs):
        """Ensure all required models for a choice are downloaded."""
        model_repos = self.get_model_repos_for_choice(model_choice)
        progress = kwargs.get("progress")

        if kwargs.get("enable_klein_anatomy_fix") and not os.path.exists(self.klein_anatomy_lora_path):
            self.download_file(self.klein_anatomy_lora_url, self.klein_anatomy_lora_path, "Klein Anatomy Fix", progress)

        if kwargs.get("enable_multi_character"): model_repos.append("guozinan/PuLID")
        if kwargs.get("enable_pose_preservation"):
            model_repos.append("Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0")
            model_repos.append("lllyasviel/Annotators")

        for repo_id in model_repos:
            if repo_id in self._verified_model_repos:
                continue
            if not self.check_model_exists(repo_id):
                print(f"Downloading model: {repo_id}")
                if progress: progress(0.5, desc=f"Downloading {repo_id}...")
                snapshot_download(repo_id)
            self._verified_model_repos.add(repo_id)

        swapper_path = os.path.join(self.models_dir, "insightface", "inswapper_128.onnx")
        if kwargs.get("enable_faceswap") and not os.path.exists(swapper_path):
            insightface_dir = os.path.join(self.models_dir, "insightface")
            os.makedirs(insightface_dir, exist_ok=True)
            hf_hub_download(repo_id="ezioruan/inswapper_128.onnx", filename="inswapper_128.onnx", local_dir=insightface_dir)

        # Built-in anime-to-photoreal LoRA downloads
        os.makedirs(self.loras_dir, exist_ok=True)
        if kwargs.get("enable_zimage_realistic_lora") and not os.path.exists(self.zimage_realistic_lora_path):
            self.download_file(self.zimage_realistic_lora_url, self.zimage_realistic_lora_path, "Realistic Snapshot LoRA (Z-Image)", progress)
        if kwargs.get("enable_flux_anime2real_lora") and not os.path.exists(self.flux_anime2real_lora_path):
            self.download_file(self.flux_anime2real_lora_url, self.flux_anime2real_lora_path, "Ultra Real Amateur Selfies LoRA (FLUX 4B)", progress)

    def resolve_builtin_lora_path(self, builtin_lora_key: str) -> Optional[str]:
        """Resolve a built-in LoRA key to its file path."""
        if builtin_lora_key is None or builtin_lora_key == "":
            return None

        path_map = {
            "klein_anatomy": self.klein_anatomy_lora_path,
            "klein_expression": self.klein_expression_lora_path,
            "zimage_realistic": self.zimage_realistic_lora_path,
            "flux_anime2real": self.flux_anime2real_lora_path,
        }
        return path_map.get(builtin_lora_key)

    def ensure_builtin_lora_downloaded(self, builtin_lora_key: str, progress=None) -> Optional[str]:
        """Ensure a built-in LoRA is downloaded and return its path."""
        if builtin_lora_key is None or builtin_lora_key == "":
            return None

        path_map = {
            "klein_anatomy": (self.klein_anatomy_lora_path, self.klein_anatomy_lora_url, "Klein Anatomy Fix"),
            "klein_expression": (self.klein_expression_lora_path, self.klein_expression_lora_url, "Klein Face Expression Transfer LoRA"),
            "zimage_realistic": (self.zimage_realistic_lora_path, self.zimage_realistic_lora_url, "Realistic Snapshot LoRA (Z-Image)"),
            "flux_anime2real": (self.flux_anime2real_lora_path, self.flux_anime2real_lora_url, "Ultra Real Amateur Selfies LoRA (FLUX 4B)"),
        }

        if builtin_lora_key not in path_map:
            print(f"Warning: Unknown built-in LoRA key: {builtin_lora_key}")
            return None

        lora_path, lora_url, description = path_map[builtin_lora_key]

        if not os.path.exists(lora_path):
            print(f"Downloading built-in LoRA: {description}")
            try:
                self.download_file(lora_url, lora_path, description, progress)
                print(f"  Downloaded: {os.path.basename(lora_path)}")
            except Exception as e:
                print(f"  Error downloading LoRA: {e}")
                return None

        return lora_path if os.path.exists(lora_path) else None
