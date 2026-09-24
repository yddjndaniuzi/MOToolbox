import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pressconf.config_store import (
    MODEL_PRESETS,
    delete_model_config,
    load_models_config,
    resolve_model,
    resolve_model_candidates,
    save_model_config,
    save_model_routing,
    strength_for_tier,
)
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

    def test_legacy_single_model_routes_all_strengths_without_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            config_dir = base_dir / "pressconf" / "config"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "models.json"
            original = {
                "default_by_use": {"brief_refine": "only", "writing": "only"},
                "models": [{"id": "only", "model": "vendor/only", "provider": "anthropic"}],
            }
            config_path.write_text(json.dumps(original), encoding="utf-8")
            loaded = load_models_config(base_dir)

            self.assertEqual(config_path.read_text(encoding="utf-8"), json.dumps(original))
            for strength in ("light", "standard", "heavy"):
                self.assertEqual(loaded["routing_by_strength"][strength]["primary"], "only")
                self.assertEqual(resolve_model(base_dir, "writing", strength)["id"], "only")

    def test_model_pool_upsert_and_strength_routing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            common = {
                "provider_tab": "custom_anthropic",
                "provider": "anthropic",
                "custom_protocol": "anthropic",
                "base_url": "https://models.example.com",
                "use_brief_refine": "on",
                "use_writing": "on",
            }
            save_model_config(base_dir, common | {
                "id": "fast", "name": "Fast", "model": "vendor/fast",
                "secret_ref": "shared_key", "api_key": "secret",
            })
            save_model_config(base_dir, common | {
                "id": "strong", "name": "Strong", "model": "vendor/strong",
                "secret_ref": "shared_key",
            })
            save_model_routing(base_dir, {
                "route_light_primary": "fast",
                "route_light_fallback_1": "strong",
                "route_standard_primary": "fast",
                "route_standard_fallback_1": "strong",
                "route_heavy_primary": "strong",
                "route_heavy_fallback_1": "fast",
            })

            self.assertEqual([item["id"] for item in load_models_config(base_dir)["models"]], ["fast", "strong"])
            self.assertEqual(
                [item["id"] for item in resolve_model_candidates(base_dir, "brief_refine", "heavy")],
                ["strong", "fast"],
            )
            selected = resolve_model(base_dir, "writing", "light")
            self.assertEqual(selected["id"], "fast")
            self.assertEqual([item["id"] for item in selected["fallbacks"]], ["strong"])
            self.assertEqual(selected["api_key"], "secret")

            delete_model_config(base_dir, "strong")
            routes = load_models_config(base_dir)["routing_by_strength"]
            self.assertTrue(all(route["primary"] == "fast" for route in routes.values()))

    def test_adding_same_model_category_does_not_replace_existing_pool_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            common = {
                "provider_tab": "custom_anthropic",
                "provider": "anthropic",
                "custom_protocol": "anthropic",
                "id": "custom-anthropic-model",
                "base_url": "https://models.example.com",
                "secret_ref": "shared_key",
                "use_writing": "on",
            }
            first_id = save_model_config(base_dir, common | {
                "name": "First", "model": "vendor/first", "api_key": "secret",
            })
            second_id = save_model_config(base_dir, common | {
                "name": "Second", "model": "vendor/second",
            })
            third_id = save_model_config(base_dir, common | {
                "name": "Third", "model": "vendor/third",
            })

            models = load_models_config(base_dir)["models"]
            self.assertEqual(
                [first_id, second_id, third_id],
                ["custom-anthropic-model", "custom-anthropic-model-2", "custom-anthropic-model-3"],
            )
            self.assertEqual(
                [item["id"] for item in models],
                ["custom-anthropic-model", "custom-anthropic-model-2", "custom-anthropic-model-3"],
            )
            self.assertEqual(
                [item["model"] for item in models],
                ["vendor/first", "vendor/second", "vendor/third"],
            )

    def test_editing_model_updates_only_the_selected_pool_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            common = {
                "provider_tab": "custom_openai",
                "provider": "openai-compatible",
                "custom_protocol": "openai-compatible",
                "base_url": "https://models.example.com/v1",
                "secret_ref": "shared_key",
                "use_writing": "on",
            }
            save_model_config(base_dir, common | {
                "id": "first", "name": "First", "model": "vendor/first", "api_key": "secret",
            })
            save_model_config(base_dir, common | {
                "id": "second", "name": "Second", "model": "vendor/second",
            })
            save_model_config(base_dir, common | {
                "original_id": "first", "id": "first", "name": "First updated", "model": "vendor/first-v2",
            })

            models = load_models_config(base_dir)["models"]
            self.assertEqual([item["id"] for item in models], ["first", "second"])
            self.assertEqual(models[0]["name"], "First updated")
            self.assertEqual(models[0]["model"], "vendor/first-v2")

    def test_report_tiers_map_to_model_strengths(self):
        self.assertEqual(strength_for_tier("nano"), "light")
        self.assertEqual(strength_for_tier("lite"), "standard")
        self.assertEqual(strength_for_tier("full"), "heavy")


if __name__ == "__main__":
    unittest.main()
