"""Canvas helpers for the Gradio Image Editing workspace."""

from __future__ import annotations

import os
from typing import Any

from PIL import Image, ImageChops, ImageFilter

from src.image.background_removal import (
    RMBG_14_LABEL,
    SIMPLE_BACKGROUND_LABEL,
    remove_background_ai,
)
from src.security import clamp_int


EDIT_CANVAS_FILL_COLORS = {
    "Transparent": (242, 242, 242, 0),
    "Soft gray": (242, 242, 242, 255),
    "White": (255, 255, 255, 255),
    "Black": (0, 0, 0, 255),
}


PHOTOSHOP_EDIT_TOOLS = ("Edit Region", "Extend Image", "Harmonize")
EXTEND_ASPECT_RATIO_PRESETS = {
    "Original": None,
    "1:1 Square": (1, 1),
    "4:5 Portrait": (4, 5),
    "3:4 Portrait": (3, 4),
    "9:16 Story": (9, 16),
    "16:9 Wide": (16, 9),
    "2:1 Wide": (2, 1),
    "21:9 Cinematic": (21, 9),
    "Custom": None,
}


def _safe_int_value(value: Any, default: int) -> int:
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


def _safe_float_value(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def resolve_fast_edit_dimensions(
    requested_width: Any,
    requested_height: Any,
    source_size: tuple[int, int] | None = None,
    max_long_edge: int = 1024,
):
    """Resolve 3070-safe edit dimensions, preserving source aspect and 16px alignment."""
    req_w = max(16, _safe_int_value(requested_width, 1024))
    req_h = max(16, _safe_int_value(requested_height, 1024))
    if source_size:
        src_w, src_h = source_size
        if src_w > 0 and src_h > 0:
            scale = min(float(max_long_edge) / float(max(src_w, src_h)), 1.0)
            req_w = int(src_w * scale)
            req_h = int(src_h * scale)
    else:
        scale = min(float(max_long_edge) / float(max(req_w, req_h)), 1.0)
        req_w = int(req_w * scale)
        req_h = int(req_h * scale)

    req_w = max(16, (req_w // 16) * 16)
    req_h = max(16, (req_h // 16) * 16)
    return req_w, req_h


def build_photoshop_edit_prompt(
    prompt: str | None,
    edit_tool: str = "Edit Region",
    edit_region_action: str = "Replace",
    harmonize_strength: Any = 0.55,
):
    """Add concise Photoshop-style feature instructions to the user's prompt."""
    user_prompt = (prompt or "").strip()
    tool = edit_tool if edit_tool in PHOTOSHOP_EDIT_TOOLS else "Edit Region"
    action = str(edit_region_action or "Replace").strip()
    strength = max(0.0, min(1.0, _safe_float_value(harmonize_strength, 0.55)))

    if tool == "Extend Image":
        instruction = (
            "Generative Expand: extend the image naturally into the new canvas. "
            "Use the existing image edges, perspective, lighting, texture, and subject scale as context."
        )
    elif tool == "Harmonize":
        instruction = (
            f"Harmonize the selected region with the surrounding image at {strength:.2f} strength. "
            "Match color temperature, exposure, shadows, grain, and lighting direction."
        )
    else:
        instruction = (
            f"Edit only the selected region using {action.lower()} behavior. "
            "Use the surrounding image as context so the result remains realistic and consistent."
        )

    return f"{instruction} {user_prompt}".strip()


def coerce_editor_image(value: Any):
    """Return the visible PIL image from Gradio Image/ImageEditor values."""
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, dict):
        for key in ("composite", "background", "image"):
            image = coerce_editor_image(value.get(key))
            if image is not None:
                return image
        layers = value.get("layers") or []
        for layer in reversed(layers):
            image = coerce_editor_image(layer)
            if image is not None:
                return image
        return None
    if isinstance(value, (list, tuple)) and value:
        return coerce_editor_image(value[0])
    if isinstance(value, str) and os.path.exists(value):
        with Image.open(value) as image:
            return image.copy()
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return Image.fromarray(value)
    except Exception:
        pass
    return None


def normalize_image_editor_upload(value: Any):
    """Ensure an uploaded image appears correctly in Gradio ImageEditor.

    Gradio ImageEditor occasionally fails to display uploaded images
    automatically (background set but composite missing, or image lands
    in layers instead of background).  This helper normalises the
    EditorValue dict so the frontend always has a valid composite.
    """
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        bg = value.get("background")
        layers = list(value.get("layers") or [])
        composite = value.get("composite")

        bg_is_empty = bg is None
        if not bg_is_empty and hasattr(bg, "mode") and "A" in bg.mode:
            try:
                alpha = bg.getchannel("A")
                bg_is_empty = alpha.getextrema() == (0, 0)
            except Exception:
                pass

        if bg_is_empty and layers:
            new_layers = []
            moved_bg = None
            for layer in layers:
                layer_img = coerce_editor_image(layer)
                if layer_img is not None and moved_bg is None:
                    moved_bg = layer_img.copy()
                else:
                    new_layers.append(layer)
            if moved_bg is not None:
                bg = moved_bg
                layers = new_layers

        if composite is None and bg is not None:
            composite = bg.copy()
            return {
                "background": bg,
                "layers": layers,
                "composite": composite,
            }

        return value

    img = coerce_editor_image(value)
    if img is not None:
        return img
    return value


def coerce_editor_context_image(value: Any):
    """Return the image context without treating brush-mask layers as pixels."""
    if isinstance(value, dict):
        layers = value.get("layers") or []
        if layers:
            image = coerce_editor_image(value.get("background"))
            if image is not None:
                return image
        for key in ("composite", "background", "image"):
            image = coerce_editor_image(value.get(key))
            if image is not None:
                return image
        return None
    return coerce_editor_image(value)


def extract_editor_brush_mask(value: Any):
    """Build an L-mode mask from ImageEditor brush layers."""
    if not isinstance(value, dict):
        return None
    layers = value.get("layers") or []
    if not layers:
        return None

    context = coerce_editor_context_image(value)
    if context is None:
        return None

    mask = Image.new("L", context.size, 0)
    for layer in layers:
        layer_image = coerce_editor_image(layer)
        if layer_image is None:
            continue
        layer_alpha = layer_image.convert("RGBA").getchannel("A")
        if layer_alpha.size != mask.size:
            layer_alpha = layer_alpha.resize(mask.size, Image.Resampling.LANCZOS)
        mask = ImageChops.lighter(mask, layer_alpha)
    return mask if mask.getbbox() else None


def remove_harmonize_reference_background(
    reference_image: Any,
    remove_background: bool = True,
    background_model: str = SIMPLE_BACKGROUND_LABEL,
    device: str | None = None,
):
    """Return an RGBA drop-in image, optionally removing a flat corner-colored background."""
    reference = coerce_editor_image(reference_image)
    if reference is None:
        return None

    rgba = reference.convert("RGBA")
    if not remove_background:
        return rgba

    if background_model == RMBG_14_LABEL:
        ai_cutout = remove_background_ai(rgba, device=device)
        if ai_cutout is not None:
            return ai_cutout.convert("RGBA")

    original_alpha = rgba.getchannel("A")
    if original_alpha.getextrema()[0] < 255:
        return rgba

    width, height = rgba.size
    if width < 2 or height < 2:
        return rgba

    corners = [
        rgba.getpixel((0, 0))[:3],
        rgba.getpixel((width - 1, 0))[:3],
        rgba.getpixel((0, height - 1))[:3],
        rgba.getpixel((width - 1, height - 1))[:3],
    ]
    background = tuple(int(sum(channel) / len(corners)) for channel in zip(*corners))
    background_image = Image.new("RGB", rgba.size, background)
    difference = ImageChops.difference(rgba.convert("RGB"), background_image).convert("L")

    transparent_below = 24
    opaque_above = 80

    def _alpha_from_difference(value: int) -> int:
        if value <= transparent_below:
            return 0
        if value >= opaque_above:
            return 255
        return int(((value - transparent_below) / (opaque_above - transparent_below)) * 255)

    new_alpha = difference.point(_alpha_from_difference)
    if new_alpha.getbbox() is None:
        return rgba
    rgba.putalpha(new_alpha)
    return rgba


def compose_harmonize_reference_on_context(
    context_image: Any,
    reference_image: Any,
    remove_background: bool = True,
    background_model: str = SIMPLE_BACKGROUND_LABEL,
    device: str | None = None,
    max_reference_long_edge_ratio: float = 0.65,
):
    """Center a Harmonize drop-in image on the edit context and return composite + mask."""
    context = coerce_editor_context_image(context_image)
    reference = remove_harmonize_reference_background(
        reference_image,
        remove_background=remove_background,
        background_model=background_model,
        device=device,
    )
    if context is None or reference is None:
        return None, None

    context_rgba = context.convert("RGBA")
    reference_rgba = reference.convert("RGBA")
    max_ratio = max(0.05, min(1.0, _safe_float_value(max_reference_long_edge_ratio, 0.65)))
    max_long_edge = int(max(context_rgba.size) * max_ratio)
    if max(reference_rgba.size) > max_long_edge > 0:
        scale = float(max_long_edge) / float(max(reference_rgba.size))
        new_size = (
            max(1, int(reference_rgba.width * scale)),
            max(1, int(reference_rgba.height * scale)),
        )
        reference_rgba = reference_rgba.resize(new_size, Image.Resampling.LANCZOS)

    paste_x = max(0, (context_rgba.width - reference_rgba.width) // 2)
    paste_y = max(0, (context_rgba.height - reference_rgba.height) // 2)
    if reference_rgba.width > context_rgba.width or reference_rgba.height > context_rgba.height:
        crop_box = (
            0,
            0,
            min(reference_rgba.width, context_rgba.width),
            min(reference_rgba.height, context_rgba.height),
        )
        reference_rgba = reference_rgba.crop(crop_box)
        paste_x = 0
        paste_y = 0

    composite = context_rgba.copy()
    composite.alpha_composite(reference_rgba, dest=(paste_x, paste_y))
    mask = Image.new("L", context_rgba.size, 0)
    mask.paste(reference_rgba.getchannel("A"), (paste_x, paste_y))
    return composite.convert("RGB"), mask if mask.getbbox() else None


def combine_edit_masks(*masks: Image.Image | None):
    """Merge edit masks by keeping the most editable value per pixel."""
    combined = None
    for mask in masks:
        if mask is None:
            continue
        mask_l = mask.convert("L")
        if combined is None:
            combined = mask_l
            continue
        if mask_l.size != combined.size:
            mask_l = mask_l.resize(combined.size, Image.Resampling.LANCZOS)
        combined = ImageChops.lighter(combined, mask_l)
    return combined if combined is not None and combined.getbbox() else None


def _canvas_fill_rgba(fill_mode: Any):
    return EDIT_CANVAS_FILL_COLORS.get(
        str(fill_mode or "").strip(),
        EDIT_CANVAS_FILL_COLORS["Soft gray"],
    )


def _flatten_canvas_alpha(image: Image.Image, fill_mode: Any):
    rgba = image.convert("RGBA")
    if str(fill_mode or "").strip() == "Transparent":
        return rgba
    background = Image.new("RGBA", rgba.size, _canvas_fill_rgba(fill_mode))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def _paste_edge_repeat(canvas: Image.Image, source: Image.Image, left: int, right: int, top: int, bottom: int):
    width, height = source.size
    if left:
        canvas.paste(source.crop((0, 0, 1, height)).resize((left, height)), (0, top))
    if right:
        canvas.paste(
            source.crop((width - 1, 0, width, height)).resize((right, height)),
            (left + width, top),
        )
    if top:
        canvas.paste(source.crop((0, 0, width, 1)).resize((width, top)), (left, 0))
    if bottom:
        canvas.paste(
            source.crop((0, height - 1, width, height)).resize((width, bottom)),
            (left, top + height),
        )
    if left and top:
        canvas.paste(source.crop((0, 0, 1, 1)).resize((left, top)), (0, 0))
    if right and top:
        canvas.paste(
            source.crop((width - 1, 0, width, 1)).resize((right, top)),
            (left + width, 0),
        )
    if left and bottom:
        canvas.paste(
            source.crop((0, height - 1, 1, height)).resize((left, bottom)),
            (0, top + height),
        )
    if right and bottom:
        canvas.paste(
            source.crop((width - 1, height - 1, width, height)).resize((right, bottom)),
            (left + width, top + height),
        )


def _normalized_extend_edges(left: Any, right: Any, top: Any, bottom: Any):
    return (
        clamp_int(_safe_int_value(left, 0), 0, 2048, 0),
        clamp_int(_safe_int_value(right, 0), 0, 2048, 0),
        clamp_int(_safe_int_value(top, 0), 0, 2048, 0),
        clamp_int(_safe_int_value(bottom, 0), 0, 2048, 0),
    )


def _parse_extend_aspect_ratio(aspect_ratio: Any, custom_ratio: Any = ""):
    import re

    label = str(aspect_ratio or "Original").strip()
    if label == "Custom":
        label = str(custom_ratio or "").strip()
    preset = EXTEND_ASPECT_RATIO_PRESETS.get(label)
    if preset:
        width, height = preset
        return float(width) / float(height)
    match = re.search(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", label)
    if not match:
        return None
    width = float(match.group(1))
    height = float(match.group(2))
    if width <= 0 or height <= 0:
        return None
    return width / height


def resolve_extend_edges_for_aspect_ratio(
    source_size: tuple[int, int],
    edit_extend_aspect_ratio: Any = "Original",
    edit_extend_anchor: Any = "Center",
    edit_extend_custom_ratio: Any = "",
):
    """Return left/right/top/bottom canvas growth for Photoshop-style aspect presets."""
    width, height = source_size
    if width <= 0 or height <= 0:
        return 0, 0, 0, 0

    target_ratio = _parse_extend_aspect_ratio(edit_extend_aspect_ratio, edit_extend_custom_ratio)
    if target_ratio is None:
        return 0, 0, 0, 0

    current_ratio = float(width) / float(height)
    target_width = width
    target_height = height
    if current_ratio < target_ratio:
        target_width = max(width, int(round(height * target_ratio)))
    elif current_ratio > target_ratio:
        target_height = max(height, int(round(width / target_ratio)))

    extra_width = clamp_int(target_width - width, 0, 2048, 0)
    extra_height = clamp_int(target_height - height, 0, 2048, 0)
    anchor = str(edit_extend_anchor or "Center").strip()

    if anchor == "Extend Left":
        left, right = extra_width, 0
    elif anchor == "Extend Right":
        left, right = 0, extra_width
    else:
        left = extra_width // 2
        right = extra_width - left

    if anchor == "Extend Up":
        top, bottom = extra_height, 0
    elif anchor == "Extend Down":
        top, bottom = 0, extra_height
    else:
        top = extra_height // 2
        bottom = extra_height - top

    return left, right, top, bottom


def prepare_edit_canvas_for_aspect_extend(
    edit_base_image: Any,
    edit_extend_aspect_ratio: Any = "Original",
    edit_extend_anchor: Any = "Center",
    edit_canvas_fill: str = "Transparent",
    edit_extend_custom_ratio: Any = "",
):
    """Expand the canvas to a target aspect ratio, like Photoshop's Crop/Generative Expand flow."""
    source = coerce_editor_context_image(edit_base_image)
    if source is None:
        return None
    left, right, top, bottom = resolve_extend_edges_for_aspect_ratio(
        source.size,
        edit_extend_aspect_ratio,
        edit_extend_anchor,
        edit_extend_custom_ratio,
    )
    return prepare_edit_canvas_for_outpaint(
        edit_base_image,
        edit_extend_left=left,
        edit_extend_right=right,
        edit_extend_top=top,
        edit_extend_bottom=bottom,
        edit_canvas_fill=edit_canvas_fill,
    )


def prepare_edit_mask_for_aspect_extend(
    edit_base_image: Any,
    edit_extend_aspect_ratio: Any = "Original",
    edit_extend_anchor: Any = "Center",
    edit_extend_custom_ratio: Any = "",
):
    """Build an edit mask for the aspect-ratio canvas growth."""
    source = coerce_editor_context_image(edit_base_image)
    if source is None:
        return None
    left, right, top, bottom = resolve_extend_edges_for_aspect_ratio(
        source.size,
        edit_extend_aspect_ratio,
        edit_extend_anchor,
        edit_extend_custom_ratio,
    )
    return prepare_edit_mask_for_outpaint(
        edit_base_image,
        edit_extend_left=left,
        edit_extend_right=right,
        edit_extend_top=top,
        edit_extend_bottom=bottom,
    )


def prepare_edit_canvas_for_outpaint(
    edit_base_image: Any,
    edit_extend_left: Any = 0,
    edit_extend_right: Any = 0,
    edit_extend_top: Any = 0,
    edit_extend_bottom: Any = 0,
    edit_canvas_fill: str = "Edge repeat",
):
    """Expand the edit canvas for outpainting and return a flattened RGB PIL image."""
    source = coerce_editor_context_image(edit_base_image)
    if source is None:
        return None

    left, right, top, bottom = _normalized_extend_edges(
        edit_extend_left,
        edit_extend_right,
        edit_extend_top,
        edit_extend_bottom,
    )
    fill_mode = edit_canvas_fill or "Edge repeat"

    source_rgba = source.convert("RGBA")
    if not any([left, right, top, bottom]):
        return _flatten_canvas_alpha(source_rgba, fill_mode)

    width, height = source_rgba.size
    new_size = (width + left + right, height + top + bottom)
    base_fill = (
        EDIT_CANVAS_FILL_COLORS["Soft gray"]
        if fill_mode == "Edge repeat"
        else _canvas_fill_rgba(fill_mode)
    )
    canvas = Image.new("RGBA", new_size, base_fill)
    if fill_mode == "Edge repeat":
        _paste_edge_repeat(canvas, source_rgba, left, right, top, bottom)
    canvas.paste(source_rgba, (left, top), source_rgba)
    return _flatten_canvas_alpha(canvas, fill_mode)


def prepare_edit_mask_for_outpaint(
    edit_base_image: Any,
    edit_extend_left: Any = 0,
    edit_extend_right: Any = 0,
    edit_extend_top: Any = 0,
    edit_extend_bottom: Any = 0,
):
    """Return the brush/edit mask, marking extended canvas pixels as editable."""
    source = coerce_editor_context_image(edit_base_image)
    if source is None:
        return None

    left, right, top, bottom = _normalized_extend_edges(
        edit_extend_left,
        edit_extend_right,
        edit_extend_top,
        edit_extend_bottom,
    )
    brush_mask = extract_editor_brush_mask(edit_base_image)
    if brush_mask is None:
        brush_mask = Image.new("L", source.size, 0)
    elif brush_mask.size != source.size:
        brush_mask = brush_mask.resize(source.size, Image.Resampling.LANCZOS)

    source_alpha = source.convert("RGBA").getchannel("A")
    if source_alpha.getextrema()[0] < 255:
        transparent_area = source_alpha.point(lambda value: 255 - value)
        brush_mask = ImageChops.lighter(brush_mask, transparent_area)

    if not any([left, right, top, bottom]):
        return brush_mask if brush_mask.getbbox() else None

    width, height = source.size
    expanded_mask = Image.new("L", (width + left + right, height + top + bottom), 255)
    expanded_mask.paste(brush_mask, (left, top))
    return expanded_mask


def apply_brush_mask_to_generated_image(
    original_context: Image.Image,
    generated_image: Image.Image,
    mask: Image.Image | None,
    feather: Any = 16,
):
    """Composite generated pixels only through the brush/extend mask."""
    if mask is None:
        return generated_image

    original = original_context.convert("RGB")
    generated = generated_image.convert("RGB")
    if generated.size != original.size:
        generated = generated.resize(original.size, Image.Resampling.LANCZOS)
    mask_l = mask.convert("L")
    if mask_l.size != original.size:
        mask_l = mask_l.resize(original.size, Image.Resampling.LANCZOS)
    feather_px = clamp_int(_safe_int_value(feather, 16), 0, 128, 16)
    if feather_px:
        mask_l = mask_l.filter(ImageFilter.GaussianBlur(radius=feather_px))
    return Image.composite(generated, original, mask_l)


def harmonize_region_with_context(
    original_context: Image.Image,
    generated_image: Image.Image,
    mask: Image.Image | None,
    amount: Any = 0.55,
):
    """Match selected generated pixels toward the surrounding context color."""
    if mask is None:
        return generated_image
    strength = max(0.0, min(1.0, _safe_float_value(amount, 0.55)))
    if strength <= 0:
        return generated_image

    try:
        import numpy as np
    except Exception:
        return generated_image

    original = original_context.convert("RGB")
    generated = generated_image.convert("RGB")
    if generated.size != original.size:
        generated = generated.resize(original.size, Image.Resampling.LANCZOS)
    mask_l = mask.convert("L")
    if mask_l.size != original.size:
        mask_l = mask_l.resize(original.size, Image.Resampling.LANCZOS)

    gen_arr = np.asarray(generated).astype("float32")
    ctx_arr = np.asarray(original).astype("float32")
    mask_arr = np.asarray(mask_l)
    selected = mask_arr > 0
    context = mask_arr < 16
    if not selected.any() or not context.any():
        return generated

    selected_mean = gen_arr[selected].mean(axis=0)
    context_mean = ctx_arr[context].mean(axis=0)
    shift = (context_mean - selected_mean) * strength
    adjusted = gen_arr.copy()
    adjusted[selected] = np.clip(adjusted[selected] + shift, 0, 255)
    return Image.fromarray(adjusted.astype("uint8"), mode="RGB")
