import json
import unittest


class JsonCompatTests(unittest.TestCase):
    def test_patch_serializes_third_party_dtype_like_objects(self):
        from src.utils.json_compat import (
            patch_json_dtype_serialization,
            unpatch_json_dtype_serialization,
        )

        patch_json_dtype_serialization()
        try:
            import torch

            self.assertEqual(
                json.loads(json.dumps({"dtype": torch.float16})),
                {"dtype": "torch.float16"},
            )
        finally:
            unpatch_json_dtype_serialization()

    def test_unpatch_restores_original_encoder(self):
        from src.utils.json_compat import (
            patch_json_dtype_serialization,
            unpatch_json_dtype_serialization,
        )

        original = json.JSONEncoder.default
        patch_json_dtype_serialization()
        self.assertIsNot(json.JSONEncoder.default, original)
        unpatch_json_dtype_serialization()
        self.assertIs(json.JSONEncoder.default, original)

    def test_context_manager_is_reversible(self):
        from src.utils.json_compat import (
            json_dtype_patch_context,
            patch_json_dtype_serialization,
            unpatch_json_dtype_serialization,
        )

        # Ensure clean state
        unpatch_json_dtype_serialization()
        original = json.JSONEncoder.default

        with json_dtype_patch_context():
            self.assertIsNot(json.JSONEncoder.default, original)

        self.assertIs(json.JSONEncoder.default, original)


if __name__ == "__main__":
    unittest.main()
