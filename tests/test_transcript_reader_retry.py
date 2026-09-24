from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from pressconf.brief import Segment
from pressconf.model_client import IncompleteGenerationError
from pressconf.transcript_reader import polish_segments


MODEL_CONFIG = {
    "api_key": "test-key",
    "base_url": "https://example.invalid/v1",
    "model": "gpt-test",
    "provider": "openai-compatible",
    "fallbacks": [],
}


class TranscriptReaderRetryTests(unittest.TestCase):
    def test_length_limited_excerpt_is_split_and_reassembled(self) -> None:
        source = "甲" * 509
        calls: list[dict] = []

        def fake_generation(**kwargs):
            calls.append(kwargs)
            excerpt = kwargs["prompt"].split("【ASR 原文】\n", 1)[1]
            if len(excerpt) > 400:
                raise IncompleteGenerationError("模型输出未完整结束（停止原因：length）。")
            return excerpt

        with patch("pressconf.transcript_reader.resolve_model", return_value=MODEL_CONFIG), patch(
            "pressconf.transcript_reader.call_writing_model_streaming", side_effect=fake_generation
        ):
            polished, meta = polish_segments(
                base_dir=Path("."),
                segments=[Segment(start=0, end=75, text=source)],
            )

        self.assertEqual(source, polished[0])
        self.assertEqual(3, len(calls))
        self.assertTrue(all(call["max_tokens"] == 3600 for call in calls))
        self.assertEqual("gpt-test", meta["model"])

    def test_split_prefers_sentence_boundary_and_preserves_every_character(self) -> None:
        from pressconf.transcript_reader import split_polish_text

        source = "前文。" + "中间内容，" * 60 + "后文。"
        left, right = split_polish_text(source)
        self.assertTrue(left.endswith(("。", "，")))
        self.assertEqual(source, left + right)

    def test_unrecoverable_length_keeps_source_text_and_does_not_block_reader(self) -> None:
        source = "原始转写" * 20
        with patch("pressconf.transcript_reader.resolve_model", return_value=MODEL_CONFIG), patch(
            "pressconf.transcript_reader.call_writing_model_streaming",
            side_effect=IncompleteGenerationError("模型输出未完整结束（停止原因：length）。"),
        ):
            polished, meta = polish_segments(
                base_dir=Path("."),
                segments=[Segment(start=0, end=75, text=source)],
            )

        self.assertEqual(source, polished[0])
        self.assertEqual([1], meta["polish_fallback_segments"])
        self.assertIn("已保留原始转写", meta["warnings"][0])


if __name__ == "__main__":
    unittest.main()
