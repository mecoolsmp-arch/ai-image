import unittest
from unittest import mock

from PIL import Image

from src.image.edit_canvas import (
    apply_brush_mask_to_generated_image,
    build_photoshop_edit_prompt,
    compose_harmonize_reference_on_context,
    extract_editor_brush_mask,
    harmonize_region_with_context,
    normalize_image_editor_upload,
    prepare_edit_canvas_for_aspect_extend,
    prepare_edit_mask_for_outpaint,
    prepare_edit_mask_for_aspect_extend,
    prepare_edit_canvas_for_outpaint,
    remove_harmonize_reference_background,
    resolve_extend_edges_for_aspect_ratio,
    resolve_fast_edit_dimensions,
)


class EditCanvasTests(unittest.TestCase):
    def test_prepare_edit_canvas_expands_edges_and_centres_source(self):
        source = Image.new("RGB", (4, 3), (10, 20, 30))

        result = prepare_edit_canvas_for_outpaint(
            source,
            edit_extend_left=2,
            edit_extend_right=6,
            edit_extend_top=1,
            edit_extend_bottom=3,
            edit_canvas_fill="White",
        )

        self.assertEqual(result.size, (12, 7))
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((2, 1)), (10, 20, 30))
        self.assertEqual(result.getpixel((0, 0)), (255, 255, 255))

    def test_prepare_edit_canvas_accepts_gradio_image_editor_composite(self):
        composite = Image.new("RGBA", (5, 5), (80, 90, 100, 255))
        editor_value = {
            "background": Image.new("RGBA", (5, 5), (0, 0, 0, 0)),
            "layers": [],
            "composite": composite,
        }

        result = prepare_edit_canvas_for_outpaint(editor_value)

        self.assertEqual(result.size, (5, 5))
        self.assertEqual(result.getpixel((0, 0)), (80, 90, 100))

    def test_prepare_edit_canvas_uses_background_not_brush_layer_as_context(self):
        background = Image.new("RGBA", (4, 4), (10, 20, 30, 255))
        brush_layer = Image.new("RGBA", (4, 4), (255, 0, 0, 0))
        brush_layer.putpixel((1, 1), (255, 0, 0, 255))
        editor_value = {
            "background": background,
            "layers": [brush_layer],
            "composite": Image.alpha_composite(background, brush_layer),
        }

        result = prepare_edit_canvas_for_outpaint(editor_value)

        self.assertEqual(result.getpixel((1, 1)), (10, 20, 30))

    def test_extract_editor_brush_mask_combines_layer_alpha(self):
        layer = Image.new("RGBA", (4, 4), (255, 255, 255, 0))
        layer.putpixel((2, 1), (255, 255, 255, 255))
        editor_value = {
            "background": Image.new("RGBA", (4, 4), (0, 0, 0, 255)),
            "layers": [layer],
            "composite": Image.new("RGBA", (4, 4), (0, 0, 0, 255)),
        }

        mask = extract_editor_brush_mask(editor_value)

        self.assertEqual(mask.mode, "L")
        self.assertEqual(mask.getpixel((2, 1)), 255)
        self.assertEqual(mask.getpixel((0, 0)), 0)

    def test_apply_brush_mask_keeps_unbrushed_context_unchanged(self):
        original = Image.new("RGB", (4, 4), (10, 20, 30))
        generated = Image.new("RGB", (4, 4), (200, 210, 220))
        mask = Image.new("L", (4, 4), 0)
        mask.putpixel((1, 1), 255)

        result = apply_brush_mask_to_generated_image(original, generated, mask, feather=0)

        self.assertEqual(result.getpixel((1, 1)), (200, 210, 220))
        self.assertEqual(result.getpixel((0, 0)), (10, 20, 30))

    def test_prepare_edit_mask_for_outpaint_marks_extended_area(self):
        source = Image.new("RGB", (4, 4), (10, 20, 30))

        mask = prepare_edit_mask_for_outpaint(
            source,
            edit_extend_left=2,
            edit_extend_right=0,
            edit_extend_top=0,
            edit_extend_bottom=0,
        )

        self.assertEqual(mask.size, (6, 4))
        self.assertEqual(mask.getpixel((0, 0)), 255)
        self.assertEqual(mask.getpixel((2, 0)), 0)

    def test_resolve_extend_edges_for_aspect_ratio_extends_left_for_wide_canvas(self):
        edges = resolve_extend_edges_for_aspect_ratio((4, 3), "2:1 Wide", "Extend Left")

        self.assertEqual(edges, (2, 0, 0, 0))

    def test_prepare_edit_canvas_for_aspect_extend_shows_transparent_new_canvas(self):
        source = Image.new("RGBA", (4, 3), (10, 20, 30, 255))

        result = prepare_edit_canvas_for_aspect_extend(
            source,
            edit_extend_aspect_ratio="2:1 Wide",
            edit_extend_anchor="Extend Left",
            edit_canvas_fill="Transparent",
        )

        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.size, (6, 3))
        self.assertEqual(result.getpixel((0, 0))[3], 0)
        self.assertEqual(result.getpixel((2, 0)), (10, 20, 30, 255))

    def test_prepare_edit_mask_for_aspect_extend_marks_new_canvas(self):
        source = Image.new("RGB", (4, 3), (10, 20, 30))

        mask = prepare_edit_mask_for_aspect_extend(
            source,
            edit_extend_aspect_ratio="2:1 Wide",
            edit_extend_anchor="Extend Left",
        )

        self.assertEqual(mask.size, (6, 3))
        self.assertEqual(mask.getpixel((0, 0)), 255)
        self.assertEqual(mask.getpixel((2, 0)), 0)

    def test_prepare_edit_mask_for_aspect_extend_keeps_prepared_transparent_area_editable(self):
        prepared = Image.new("RGBA", (6, 3), (242, 242, 242, 0))
        prepared.paste(Image.new("RGBA", (4, 3), (10, 20, 30, 255)), (2, 0))

        mask = prepare_edit_mask_for_aspect_extend(
            prepared,
            edit_extend_aspect_ratio="2:1 Wide",
            edit_extend_anchor="Extend Left",
        )

        self.assertEqual(mask.size, (6, 3))
        self.assertEqual(mask.getpixel((0, 0)), 255)
        self.assertEqual(mask.getpixel((2, 0)), 0)

    def test_harmonize_region_shifts_selection_toward_context_color(self):
        original = Image.new("RGB", (5, 5), (20, 40, 60))
        generated = Image.new("RGB", (5, 5), (220, 220, 220))
        mask = Image.new("L", (5, 5), 0)
        mask.putpixel((2, 2), 255)

        result = harmonize_region_with_context(original, generated, mask, amount=0.75)

        self.assertLess(result.getpixel((2, 2))[0], 160)
        self.assertEqual(result.getpixel((0, 0)), (220, 220, 220))

    def test_remove_harmonize_reference_background_can_keep_full_image(self):
        reference = Image.new("RGB", (4, 4), (10, 200, 30))

        result = remove_harmonize_reference_background(reference, remove_background=False)

        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.getchannel("A").getpixel((0, 0)), 255)

    def test_remove_harmonize_reference_background_clears_flat_corner_background(self):
        reference = Image.new("RGB", (5, 5), (250, 250, 250))
        reference.putpixel((2, 2), (20, 180, 40))

        result = remove_harmonize_reference_background(reference, remove_background=True)

        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.getchannel("A").getpixel((0, 0)), 0)
        self.assertEqual(result.getchannel("A").getpixel((2, 2)), 255)

    def test_remove_harmonize_reference_background_uses_rmbg_ai_cutout_when_selected(self):
        reference = Image.new("RGB", (4, 4), (10, 20, 30))
        ai_cutout = Image.new("RGBA", (4, 4), (10, 20, 30, 128))

        with mock.patch("src.image.edit_canvas.remove_background_ai", return_value=ai_cutout) as remover:
            result = remove_harmonize_reference_background(
                reference,
                remove_background=True,
                background_model="Fast AI Cutout (RMBG-1.4)",
                device="cuda",
            )

        self.assertEqual(result.getchannel("A").getpixel((0, 0)), 128)
        remover.assert_called_once()

    def test_compose_harmonize_reference_on_context_centers_drop_in_and_returns_mask(self):
        context = Image.new("RGB", (8, 8), (10, 20, 30))
        reference = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        reference.putpixel((1, 1), (220, 40, 30, 255))
        reference.putpixel((2, 1), (220, 40, 30, 255))
        reference.putpixel((1, 2), (220, 40, 30, 255))
        reference.putpixel((2, 2), (220, 40, 30, 255))

        composite, mask = compose_harmonize_reference_on_context(
            context,
            reference,
            remove_background=False,
        )

        self.assertEqual(composite.size, context.size)
        self.assertEqual(mask.size, context.size)
        self.assertEqual(composite.getpixel((3, 3)), (220, 40, 30))
        self.assertEqual(mask.getpixel((3, 3)), 255)
        self.assertEqual(mask.getpixel((0, 0)), 0)

    def test_build_photoshop_edit_prompt_adds_feature_instructions(self):
        prompt = build_photoshop_edit_prompt(
            "add warm window light",
            edit_tool="Edit Region",
            edit_region_action="Replace",
            harmonize_strength=0.55,
        )

        self.assertIn("Edit only the selected region", prompt)
        self.assertIn("Use the surrounding image as context", prompt)
        self.assertIn("add warm window light", prompt)

    def test_resolve_fast_edit_dimensions_caps_to_3070_safe_long_edge(self):
        width, height = resolve_fast_edit_dimensions(
            requested_width=2048,
            requested_height=2048,
            source_size=(1800, 900),
            max_long_edge=1024,
        )

        self.assertEqual((width, height), (1024, 512))
        self.assertEqual(width % 16, 0)
        self.assertEqual(height % 16, 0)

    def test_normalize_upload_sets_composite_when_missing(self):
        bg = Image.new("RGB", (4, 4), (10, 20, 30))
        value = {"background": bg, "layers": [], "composite": None}

        result = normalize_image_editor_upload(value)

        self.assertIsNotNone(result["composite"])
        self.assertEqual(result["composite"].size, bg.size)

    def test_normalize_upload_moves_layer_to_empty_background(self):
        layer = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
        value = {
            "background": Image.new("RGBA", (4, 4), (0, 0, 0, 0)),
            "layers": [layer],
            "composite": None,
        }

        result = normalize_image_editor_upload(value)

        self.assertEqual(result["background"].size, (4, 4))
        self.assertEqual(result["background"].getpixel((0, 0)), (255, 0, 0, 255))
        self.assertEqual(result["layers"], [])
        self.assertIsNotNone(result["composite"])

    def test_normalize_upload_returns_pil_image_unchanged(self):
        img = Image.new("RGB", (4, 4), (10, 20, 30))

        result = normalize_image_editor_upload(img)

        self.assertIs(result, img)

    def test_normalize_upload_is_idempotent(self):
        bg = Image.new("RGB", (4, 4), (10, 20, 30))
        value = {"background": bg, "layers": [], "composite": bg.copy()}

        result = normalize_image_editor_upload(value)

        self.assertIs(result, value)


if __name__ == "__main__":
    unittest.main()
