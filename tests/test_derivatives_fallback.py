import unittest
from unittest.mock import patch

from pressconf.derivatives import call_writing_model_streaming
from pressconf.model_client import IncompleteGenerationError


class WritingModelFallbackTests(unittest.TestCase):
    def test_length_stop_discards_partial_output_and_uses_fallback(self):
        calls = []

        def fake_stream_chat_model(**kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "primary":
                kwargs["on_delta"]("partial")
                raise IncompleteGenerationError("模型输出未完整结束（停止原因：length）。")
            kwargs["on_delta"]("complete")
            return "complete"

        progress = []
        selected_model = {}
        with patch("pressconf.derivatives.stream_chat_model", side_effect=fake_stream_chat_model):
            result = call_writing_model_streaming(
                api_key="one",
                base_url="https://example.invalid",
                model="primary",
                provider="openai-compatible",
                prompt="test",
                max_tokens=100,
                stage="test",
                progress_callback=progress.append,
                fallback_models=[
                    {
                        "api_key": "two",
                        "base_url": "https://example.invalid",
                        "model": "fallback",
                        "provider": "anthropic",
                    }
                ],
                selected_model=selected_model,
            )

        self.assertEqual(result, "complete")
        self.assertEqual(calls, ["primary", "fallback"])
        self.assertEqual(selected_model["model"], "fallback")
        self.assertEqual(selected_model["provider"], "anthropic")
        self.assertTrue(any("达到输出上限" in item.get("message", "") for item in progress))

    def test_non_retryable_error_does_not_hide_configuration_problem(self):
        with patch(
            "pressconf.derivatives.stream_chat_model",
            side_effect=RuntimeError("OpenAI-compatible API 返回错误：400 invalid model"),
        ) as stream:
            with self.assertRaisesRegex(RuntimeError, "invalid model"):
                call_writing_model_streaming(
                    api_key="one",
                    base_url="https://example.invalid",
                    model="primary",
                    provider="openai-compatible",
                    prompt="test",
                    max_tokens=100,
                    stage="test",
                    fallback_models=[
                        {
                            "api_key": "two",
                            "base_url": "https://example.invalid",
                            "model": "fallback",
                            "provider": "anthropic",
                        }
                    ],
                )

        self.assertEqual(stream.call_count, 1)


if __name__ == "__main__":
    unittest.main()
