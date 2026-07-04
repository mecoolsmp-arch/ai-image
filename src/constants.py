"""Small constants used by the image-only app."""

FAST_FLUX_STATE_MIGRATION_KEY = "fast_flux_default_migrated_v1"
CHARACTER_MANAGER_STATE_FILENAME = "character_manager.json"
LEGACY_CHARACTER_MANAGER_STATE_FILENAME = "character_manager.pkl"

GRADIO_DELETE_CACHE = 7200

KLEIN_ANATOMY_LORA_URL = "https://civitai.com/api/download/models/2617474"

KLEIN_EXPRESSION_LORA_URL = "https://civitai.com/api/download/models/2363566"
KLEIN_EXPRESSION_LORA_TRIGGER = (
    "transfer character face expression in image1 "
    "with character face expression in image2"
)
KLEIN_EXPRESSION_LORA_STRENGTH = 1.0
