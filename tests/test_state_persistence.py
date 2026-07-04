from pathlib import Path
import unittest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
CONSTANTS_PATH = Path(__file__).resolve().parents[1] / "src" / "constants.py"


class StatePersistenceTests(unittest.TestCase):
    def test_img2img_strength_default_is_present_in_initial_state_loader(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn('coerce_float(state.get("img2img_strength"), 0.6', source)
        self.assertIn('"img2img_strength": img2img_strength', source)

    def test_img2img_strength_is_persisted(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("def persist_ui_state(", source)
        self.assertIn("img2img_strength", source)
        self.assertIn('"img2img_strength": float(img2img_strength)', source)

    def test_tensorrt_state_is_not_persisted(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertNotIn('"enable_tensorrt"', source)
        self.assertNotIn("enable_tensorrt,", source)

    def test_fast_flux_migration_flag_is_present(self):
        source = APP_PATH.read_text(encoding="utf-8")
        constants_source = CONSTANTS_PATH.read_text(encoding="utf-8")
        self.assertIn('FAST_FLUX_STATE_MIGRATION_KEY = "fast_flux_default_migrated_v1"', constants_source)
        self.assertIn("FAST_FLUX_STATE_MIGRATION_KEY", source)
        self.assertIn('FAST_FLUX_STATE_MIGRATION_KEY: bool(existing_state.get(FAST_FLUX_STATE_MIGRATION_KEY, False))', source)
        self.assertIn('FAST_FLUX_STATE_MIGRATION_KEY: fast_flux_state_migrated', source)

    def test_fast_resolution_preset_is_exposed(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn('SINGLE_RESOLUTION_PRESETS = [FAST_RESOLUTION_PRESET, "~1024px", "~1280px", "~1536px (32GB+)"]', source)
        self.assertIn('BATCH_RESOLUTION_PRESETS = [', source)
        self.assertIn('value=SINGLE_RESOLUTION_PRESETS[0]', source)


if __name__ == "__main__":
    unittest.main()
