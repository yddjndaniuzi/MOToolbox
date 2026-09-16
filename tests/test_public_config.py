import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pressconf.config_store import MODEL_PRESETS, resolve_model
from pressconf.model_client import normalize_provider


class PublicConfigTests(unittest.TestCase):
    def test_built_in_presets_only_contain_public_endpoints(self):
        serialized = json.dumps(MODEL_PRESETS, ensure_ascii=False).lower()
        self.assertNotIn(".local", serialized)
        self.assertNotIn(".internal", serialized)
        self.assertEqual(MODEL_PRESETS["custom_openai"]["base_url"], "")
        self.assertEqual(MODEL_PRESETS["custom_anthropic"]["base_url"], "")
        self.assertEqual(MODEL_PRESETS["custom_openai"]["model"], "")
        self.assertEqual(MODEL_PRESETS["custom_anthropic"]["model"], "")

    def test_custom_gateway_keeps_environment_based_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            config_dir = base_dir / "pressconf" / "config"
            config_dir.mkdir(parents=True)
            (config_dir / "models.json").write_text(
                json.dumps(
                    {
                        "default_by_use": {"writing": "private-model"},
                        "models": [
                            {
                                "id": "private-model",
                                "name": "Private model",
                                "provider": "legacy-gateway",
                                "base_url": "",
                                "model": "vendor/model",
                                "secret_ref": "legacy_gateway_api_key",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            env = {
                "LEGACY_GATEWAY_API_KEY": "unit-test-value",
                "LEGACY_GATEWAY_BASE_URL": "https://gateway.example.com/v1",
            }
            with patch.dict(os.environ, env, clear=False):
                model = resolve_model(base_dir, "writing")

        self.assertEqual(model["api_key"], "unit-test-value")
        self.assertEqual(model["base_url"], "https://gateway.example.com/v1")
        self.assertEqual(normalize_provider(model["provider"]), "openai-compatible")

    def test_custom_anthropic_suffix_remains_supported(self):
        self.assertEqual(normalize_provider("legacy-gateway-anthropic"), "anthropic")


if __name__ == "__main__":
    unittest.main()
