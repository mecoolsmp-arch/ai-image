"""UI state dataclass for type-safe state management."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class UIState:
    """Dataclass for UI state management."""
    model_choice: str
    prompt: str = ""
    negative_prompt: str = ""
    preset_choice: str = "None"
    resolution_preset: str = "~768px"
    batch_input_folder: str = ""
    batch_output_folder: str = ""
    batch_resolution_preset: str = "~768px"
    downscale_factor: str = "1x"
    img2img_strength: float = 0.6
    height: int = 512
    width: int = 512
    steps: int = 4
    seed: int = -1
    guidance_scale: float = 0.0
    enable_klein_anatomy_fix: bool = False
    enable_expression_transfer: bool = False
    preservation_enable: bool = False
    preservation_detector: str = "DWPose"
    preservation_mode: str = "Pose"
    device: str = "cuda"
    lora_file: Optional[str] = None
    builtin_lora: Optional[str] = None
    lora_strength: float = 1.0
    optimization_profile: str = "balanced"
    enable_windows_compile_probe: bool = False
    enable_cuda_graphs: bool = False
    enable_optional_accelerators: bool = False
    input_image_names: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_choice": self.model_choice,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "preset_choice": self.preset_choice,
            "resolution_preset": self.resolution_preset,
            "batch_input_folder": self.batch_input_folder,
            "batch_output_folder": self.batch_output_folder,
            "batch_resolution_preset": self.batch_resolution_preset,
            "downscale_factor": self.downscale_factor,
            "img2img_strength": self.img2img_strength,
            "height": self.height,
            "width": self.width,
            "steps": self.steps,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "enable_klein_anatomy_fix": self.enable_klein_anatomy_fix,
            "enable_expression_transfer": self.enable_expression_transfer,
            "preservation_enable": self.preservation_enable,
            "preservation_detector": self.preservation_detector,
            "preservation_mode": self.preservation_mode,
            "device": self.device,
            "lora_file": self.lora_file,
            "builtin_lora": self.builtin_lora,
            "lora_strength": self.lora_strength,
            "optimization_profile": self.optimization_profile,
            "enable_windows_compile_probe": self.enable_windows_compile_probe,
            "enable_cuda_graphs": self.enable_cuda_graphs,
            "enable_optional_accelerators": self.enable_optional_accelerators,
            "input_image_names": self.input_image_names,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UIState":
        """Create from dictionary for JSON deserialization."""
        return cls(
            model_choice=data.get("model_choice", ""),
            prompt=data.get("prompt", ""),
            negative_prompt=data.get("negative_prompt", ""),
            preset_choice=data.get("preset_choice", "None"),
            resolution_preset=data.get("resolution_preset", "~768px"),
            batch_input_folder=data.get("batch_input_folder", ""),
            batch_output_folder=data.get("batch_output_folder", ""),
            batch_resolution_preset=data.get("batch_resolution_preset", "~768px"),
            downscale_factor=data.get("downscale_factor", "1x"),
            img2img_strength=data.get("img2img_strength", 0.6),
            height=data.get("height", 512),
            width=data.get("width", 512),
            steps=data.get("steps", 4),
            seed=data.get("seed", -1),
            guidance_scale=data.get("guidance_scale", 0.0),
            enable_klein_anatomy_fix=data.get("enable_klein_anatomy_fix", False),
            enable_expression_transfer=data.get("enable_expression_transfer", False),
            preservation_enable=data.get("preservation_enable", False),
            preservation_detector=data.get("preservation_detector", "DWPose"),
            preservation_mode=data.get("preservation_mode", "Pose"),
            device=data.get("device", "cuda"),
            lora_file=data.get("lora_file", None),
            builtin_lora=data.get("builtin_lora", None),
            lora_strength=data.get("lora_strength", 1.0),
            optimization_profile=data.get("optimization_profile", "balanced"),
            enable_windows_compile_probe=data.get("enable_windows_compile_probe", False),
            enable_cuda_graphs=data.get("enable_cuda_graphs", False),
            enable_optional_accelerators=data.get("enable_optional_accelerators", False),
            input_image_names=data.get("input_image_names", []),
        )

    @classmethod
    def load_payload(cls, data: Dict[str, Any]) -> "UIState":
        """Create an instance from a persisted payload."""
        return cls.from_dict(data or {})
