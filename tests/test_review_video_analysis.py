from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pressconf.media_feedback import (
    REVIEW_VIDEO_CHUNK_LIMIT,
    REVIEW_VIDEO_FINAL_SECTIONS,
    generate_review_video_analysis,
)
from pressconf.model_client import IncompleteGenerationError


MODEL_CONFIG = {
    "api_key": "test-key",
    "base_url": "https://example.invalid/v1",
    "model": "gpt-test",
    "provider": "openai-compatible",
    "fallbacks": [],
}


def completed_stage(**kwargs):
    selected = kwargs.get("selected_model")
    if selected is not None:
        selected.update({"model": "gpt-test", "provider": "openai-compatible"})
    stage = str(kwargs["stage"])
    if stage.startswith("逐字稿理解"):
        return """## 片段情绪
正面。

## 正面观点
- 观点：体验稳定。原文：“体验稳定”。

## 负面/争议观点
- 无明确负面观点。

## 可引用金句
- “体验稳定”｜片段线索。"""

    section_number = int(stage.split()[-1].split("/")[0]) - 1
    headings = REVIEW_VIDEO_FINAL_SECTIONS[section_number]["headings"]
    return "\n\n".join(f"{heading}\n完整内容。" for heading in headings)


class ReviewVideoAnalysisTests(unittest.TestCase):
    def test_long_transcript_is_split_and_final_report_is_generated_by_sections(self) -> None:
        transcript = "甲" * (REVIEW_VIDEO_CHUNK_LIMIT * 2 + 1_500)
        calls: list[dict] = []

        def capture(**kwargs):
            calls.append(kwargs)
            return completed_stage(**kwargs)

        with tempfile.TemporaryDirectory() as temp, patch(
            "pressconf.media_feedback.resolve_model", return_value=MODEL_CONFIG
        ), patch("pressconf.media_feedback.call_writing_model_streaming", side_effect=capture):
            result, meta = generate_review_video_analysis(
                base_dir=Path(temp),
                product_name="产品 A",
                media_name="媒体 A",
                video_title="长视频",
                transcript_text=transcript,
                work_dir=Path(temp),
            )

        chunk_calls = [call for call in calls if str(call["stage"]).startswith("逐字稿理解")]
        final_calls = [call for call in calls if str(call["stage"]).startswith("评测视频分析")]
        self.assertEqual(3, len(chunk_calls))
        self.assertEqual(len(REVIEW_VIDEO_FINAL_SECTIONS), len(final_calls))
        self.assertEqual(3, meta["chunk_count"])
        self.assertEqual(2, meta["pipeline_version"])
        self.assertIn("# 媒体评测视频分析", result)
        self.assertIn("## 附：证据摘要", result)
        self.assertTrue(all("个中文字符以内" in call["prompt"] for call in calls))

    def test_completed_parts_are_reused_after_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            work_dir = Path(temp)
            with patch("pressconf.media_feedback.resolve_model", return_value=MODEL_CONFIG), patch(
                "pressconf.media_feedback.call_writing_model_streaming", side_effect=completed_stage
            ) as first_call:
                first_result, _ = generate_review_video_analysis(
                    base_dir=work_dir,
                    product_name="产品 A",
                    media_name="媒体 A",
                    video_title="视频",
                    transcript_text="这是完整逐字稿。",
                    work_dir=work_dir,
                )
            self.assertGreater(first_call.call_count, 0)

            with patch("pressconf.media_feedback.resolve_model", return_value=MODEL_CONFIG), patch(
                "pressconf.media_feedback.call_writing_model_streaming",
                side_effect=AssertionError("缓存命中时不应再次调用模型"),
            ) as second_call:
                second_result, _ = generate_review_video_analysis(
                    base_dir=work_dir,
                    product_name="产品 A",
                    media_name="媒体 A",
                    video_title="视频",
                    transcript_text="这是完整逐字稿。",
                    work_dir=work_dir,
                )
            self.assertEqual(0, second_call.call_count)
            self.assertEqual(first_result, second_result)

    def test_length_failure_splits_the_current_chunk_and_continues(self) -> None:
        attempts = 0

        def fail_large_chunk(**kwargs):
            nonlocal attempts
            if kwargs["stage"] == "逐字稿理解 1/1":
                attempts += 1
                raise IncompleteGenerationError("模型输出未完整结束（停止原因：length）。")
            return completed_stage(**kwargs)

        with tempfile.TemporaryDirectory() as temp, patch(
            "pressconf.media_feedback.resolve_model", return_value=MODEL_CONFIG
        ), patch("pressconf.media_feedback.call_writing_model_streaming", side_effect=fail_large_chunk):
            result, meta = generate_review_video_analysis(
                base_dir=Path(temp),
                product_name="产品 A",
                media_name="媒体 A",
                video_title="视频",
                transcript_text="甲" * 9_000,
                work_dir=Path(temp),
            )

        self.assertEqual(2, attempts)
        self.assertEqual(1, meta["chunk_count"])
        self.assertIn("## 主要正面观点", result)


if __name__ == "__main__":
    unittest.main()
