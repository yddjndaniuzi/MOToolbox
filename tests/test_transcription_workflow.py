from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pressconf.asr_hotword_loader import asr_hotword_prompt, build_asr_hotwords
from pressconf.coverage import build_coverage_report, transcript_evidence
from pressconf.refine import build_prompt
from pressconf.web import LiveRecorder, create_app
from pressconf.domains import domain_from_manifest, resolve_domain
from pressconf.fact_ledger import build_fact_ledger
from pressconf.brief import compose_brief_base
from pressconf.media_feedback import build_review_video_chunk_prompt
from pressconf.transcript import (
    DEFAULT_MLX_ASR_MODEL,
    asr_model_candidates,
    configured_mlx_model,
    looks_like_asr_repetition,
    run_mlx_whisper,
    sanitize_asr_replacement_chars,
)


def make_srt(rows: list[tuple[int, str]]) -> str:
    blocks = []
    for index, (second, text) in enumerate(rows, start=1):
        blocks.append(
            f"{index}\n00:00:{second:02d},000 --> 00:00:{second + 1:02d},000\n{text}\n"
        )
    return "\n".join(blocks)


class EvidenceTests(unittest.TestCase):
    def test_mlx_turbo_is_the_default_asr_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(configured_mlx_model(), DEFAULT_MLX_ASR_MODEL)

    def test_mlx_asr_uses_fast_defaults_and_writes_timing_quality(self) -> None:
        captured: dict = {}

        def fake_transcribe(audio, **kwargs):
            captured["samples"] = len(audio)
            captured.update(kwargs)
            return {
                "language": "zh",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "小鹏 MONA L03 正式发布",
                        "avg_logprob": -0.1,
                        "no_speech_prob": 0.01,
                        "compression_ratio": 1.1,
                    },
                    {
                        "start": 0.9,
                        "end": 2.0,
                        "text": "宇宙" * 60,
                        "avg_logprob": -3.0,
                        "no_speech_prob": 0.0,
                        "compression_ratio": 20.0,
                    },
                ],
            }

        fake_module = SimpleNamespace(transcribe=fake_transcribe)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_path = root / "audio.wav"
            with wave.open(str(audio_path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(struct.pack("<16000h", *([0] * 16000)))
            with patch.dict(sys.modules, {"mlx_whisper": fake_module}), patch.dict(os.environ, {}, clear=True), patch(
                "pressconf.transcript.detect_speech_regions", return_value=None
            ):
                target = run_mlx_whisper(audio_path, root, {})
            self.assertIn("小鹏 MONA L03", target.read_text(encoding="utf-8"))
            self.assertNotIn("ASR 重复失真", target.read_text(encoding="utf-8"))
            self.assertNotIn("beam_size", captured)
            self.assertFalse(captured["word_timestamps"])
            quality = json.loads((root / "quality.json").read_text(encoding="utf-8"))
            self.assertEqual(quality["engine"], "mlx-whisper")
            self.assertEqual(quality["device"], "metal")
            self.assertEqual(quality["settings"]["decoder"], "greedy")
            self.assertEqual(quality["discarded_tail_segments"], 1)
            self.assertIn("elapsed_sec", quality)
            self.assertIn("realtime_factor", quality)

    def test_mlx_vad_skips_silent_audio_before_decoder(self) -> None:
        fake_module = SimpleNamespace(transcribe=lambda *_args, **_kwargs: self.fail("decoder received silence"))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_path = root / "audio.wav"
            with wave.open(str(audio_path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\x00\x00" * (60 * 16000))
            with patch.dict(sys.modules, {"mlx_whisper": fake_module}), patch(
                "pressconf.transcript.detect_speech_regions", return_value=[]
            ):
                target = run_mlx_whisper(audio_path, root, {})
            self.assertIn("未检测到可转写的人声", target.read_text(encoding="utf-8"))
            quality = json.loads((root / "quality.json").read_text(encoding="utf-8"))
            self.assertTrue(quality["vad_predecode"])
            self.assertTrue(quality["vad_no_speech"])
            self.assertEqual(quality["vad_decoded_windows"], 0)

    def test_asr_runaway_repetition_is_marked_not_silently_accepted(self) -> None:
        broken = "宇宙" * 60 + "�"
        self.assertTrue(looks_like_asr_repetition(broken))
        self.assertEqual(sanitize_asr_replacement_chars(broken), "[ASR 重复失真，待核实]")

    def test_isolated_replacement_character_does_not_reject_whole_transcript(self) -> None:
        repaired = sanitize_asr_replacement_chars("这里有一个听不清的词�，其余内容正常")
        self.assertEqual(repaired, "这里有一个听不清的词[听不清]，其余内容正常")

    def test_hub_rate_limit_skips_uncached_primary_and_uses_local_model(self) -> None:
        states = {"medium": "missing", "small": "complete", "base": "complete"}
        with patch("pressconf.transcript.huggingface_rate_limited", return_value=True), patch(
            "pressconf.transcript.faster_whisper_cache_state", side_effect=lambda model: states[model]
        ):
            self.assertEqual(asr_model_candidates("medium"), ["small", "base"])
    def test_full_transcript_is_injected_into_refine_prompt(self) -> None:
        srt = make_srt([(1, "起售价是2699元"), (3, "顶配版本是3599元")])
        evidence, meta = transcript_evidence(srt)
        prompt = build_prompt("S 系列", "基础稿没有价格", {}, transcript_evidence_text=evidence)
        self.assertEqual(meta["strategy"], "full")
        self.assertIn("2699元", prompt)
        self.assertIn("3599元", prompt)
        self.assertIn("原始转写证据", prompt)

    def test_long_transcript_keeps_price_facts_and_timeline_coverage(self) -> None:
        rows = [(index, "普通介绍内容") for index in range(1, 50)]
        rows[24] = (25, "价格公布为7999元")
        srt = make_srt(rows)
        evidence, meta = transcript_evidence(srt, max_chars=500)
        self.assertEqual(meta["strategy"], "fact-priority-balanced")
        self.assertIn("7999元", evidence)
        self.assertIn("0:00:01", evidence)
        self.assertIn("0:00:49", evidence)

    def test_coverage_report_ignores_srt_sequence_and_timestamps(self) -> None:
        srt = make_srt([(1, "起售价2699元"), (3, "电池容量6500mAh")])
        with tempfile.TemporaryDirectory() as temp:
            report = build_coverage_report(
                result_dir=Path(temp), transcript=srt, base_text="起售价2699元", refined_text="起售价2699元"
            )
            self.assertEqual(report["facts"]["source_count"], 2)
            self.assertIn("6500mAh", report["facts"]["base_missing"])
            saved = json.loads((Path(temp) / "coverage_report.json").read_text())
            self.assertEqual(saved["facts"]["source_count"], 2)


class HotwordTests(unittest.TestCase):
    def test_hotwords_only_load_relevant_categories(self) -> None:
        context = build_asr_hotwords({"event_name": "vivo X Fold 折叠屏影像发布会"})
        self.assertIn("foldables", context["categories"])
        self.assertIn("imaging", context["categories"])
        self.assertNotIn("ai_os", context["categories"])


class RecordingTests(unittest.TestCase):
    def test_concat_failure_preserves_every_segment_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            segments_dir = root / "live_segments"
            segments_dir.mkdir()
            segments = []
            for index in range(2):
                path = segments_dir / f"part_{index}.ts"
                path.write_bytes(b"x" * 100)
                segments.append(path)
            recorder = LiveRecorder.__new__(LiveRecorder)
            recorder.segments = segments
            recorder.segments_dir = segments_dir
            recorder.output_path = root / "live_recording.mp4"
            recorder.duration_seconds = 60
            recorder._started_at = 0.0
            failed = SimpleNamespace(returncode=1, stderr="concat failed")
            with patch("pressconf.web.subprocess.run", return_value=failed):
                with self.assertRaisesRegex(RuntimeError, "2 个分段"):
                    recorder._merge_segments()
            self.assertTrue(all(path.exists() for path in segments))
            meta = json.loads((root / "recording_meta.json").read_text())
            self.assertFalse(meta["complete"])
            self.assertEqual(meta["segment_count"], 2)


class DomainPlatformTests(unittest.TestCase):
    def test_capture_ui_exposes_domain_and_task_selectors(self) -> None:
        app = create_app()
        response = app.test_client().get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('name="domain_id"', html)
        self.assertIn('value="automotive"', html)
        self.assertIn('value="semiconductor"', html)
        self.assertIn('name="task_type"', html)
        self.assertIn('value="faithful_transcript"', html)

        review_html = app.test_client().get("/review-video").get_data(as_text=True)
        self.assertIn('id="domainId"', review_html)
        self.assertIn('value="robotics"', review_html)
        self.assertIn('value="semiconductor"', review_html)

    def test_automotive_domain_detection(self) -> None:
        result = resolve_domain(
            {"event_name": "全新车型发布会"},
            "新车采用 800V 平台，CLTC 续航 820 公里，支持城市 NOA 和激光雷达。",
        )
        self.assertEqual(result["resolved"], "automotive")
        self.assertGreaterEqual(result["confidence"], 0.45)

    def test_chatgpt_title_routes_to_model_domain_before_asr(self) -> None:
        result = resolve_domain({"event_name": "ChatGPT-5.6 发布会完整版【中字】"})
        self.assertEqual(result["resolved"], "foundation_model")
        context = build_asr_hotwords({"event_name": "ChatGPT-5.6 发布会完整版【中字】"})
        self.assertNotIn("电池", context["terms"])
        self.assertIn("Token", context["terms"])

    def test_semiconductor_domain_detection_and_pre_asr_hotwords(self) -> None:
        result = resolve_domain(
            {"event_name": "NVIDIA GTC Blackwell Ultra 芯片发布会"},
            "新 GPU 采用 Chiplet 和 HBM3E，支持 NVLink，FP8 算力达到 20 PFLOPS。",
        )
        self.assertEqual(result["resolved"], "semiconductor")
        self.assertGreaterEqual(result["confidence"], 0.45)
        context = build_asr_hotwords({"event_name": "NVIDIA GTC Blackwell Ultra 芯片发布会"})
        self.assertEqual(context["domain"], "semiconductor")
        self.assertIn("HBM3E", context["terms"])
        self.assertIn("NVLink", context["terms"])
        self.assertNotIn("电池", context["terms"])

    def test_manual_domain_overrides_detection(self) -> None:
        result = resolve_domain(
            {"domain": {"requested": "robotics"}},
            "CLTC 续航和汽车智驾",
        )
        self.assertEqual(result["resolved"], "robotics")
        self.assertEqual(result["confidence"], 1.0)

    def test_fact_ledger_classifies_domain_metric_with_evidence(self) -> None:
        srt = make_srt([(1, "新车 CLTC 续航达到820公里，预售价为299900元")])
        manifest = {
            "event_name": "新车发布会",
            "domain": {"requested": "automotive", "resolved": "automotive", "confidence": 1.0},
        }
        with tempfile.TemporaryDirectory() as temp:
            ledger = build_fact_ledger(result_dir=Path(temp), manifest=manifest, transcript=srt)
            types = {fact["type"] for fact in ledger["facts"]}
            self.assertIn("range", types)
            self.assertIn("price", types)
            self.assertTrue(all(fact["source_text"] for fact in ledger["facts"]))

    def test_semiconductor_fact_ledger_preserves_units_and_conditions(self) -> None:
        srt = make_srt([
            (1, "芯片采用3nm制程，包含2080亿晶体管，拥有128核CPU，最高频率5.7GHz"),
            (2, "配备192GB HBM3E，内存带宽8TB/s，FP8峰值算力20PFLOPS，整卡功耗1000W"),
            (3, "同功耗下性能提升30%"),
            (4, "每瓦能效提升40%"),
        ])
        manifest = {
            "event_name": "AI 芯片发布会",
            "domain": {"requested": "semiconductor", "resolved": "semiconductor", "confidence": 1.0},
        }
        with tempfile.TemporaryDirectory() as temp:
            ledger = build_fact_ledger(result_dir=Path(temp), manifest=manifest, transcript=srt)
            types = {fact["type"] for fact in ledger["facts"]}
            self.assertTrue({
                "process_node", "transistor_count", "core_count", "frequency",
                "memory_capacity", "bandwidth", "compute", "tdp", "performance_gain", "power_efficiency",
            }.issubset(types))
            self.assertTrue(any(fact["unit"].lower() == "pflops" for fact in ledger["facts"]))

    def test_non_phone_base_uses_domain_sections(self) -> None:
        manifest = {
            "event_name": "机器人发布会",
            "domain": {"requested": "robotics", "resolved": "robotics", "confidence": 1.0, "signals": ["人工指定"]},
            "task_type": "launch_notes",
            "stats": {"duration_sec": 60},
            "keyframes": [],
        }
        brief = compose_brief_base(manifest, make_srt([(1, "机器人拥有40自由度")]), "robot")
        self.assertIn("## 发布会概述", brief)
        self.assertIn("产品总结", brief)
        self.assertIn("传播总结", brief)
        self.assertIn("## 本体参数", brief)
        self.assertIn("## 现场演示可信度", brief)
        self.assertNotIn("## 参数表：", brief)

    def test_semiconductor_brief_uses_total_detail_structure_and_chip_topics(self) -> None:
        manifest = {
            "event_name": "AI 芯片发布会",
            "domain": {"requested": "semiconductor", "resolved": "semiconductor", "confidence": 1.0},
            "task_type": "launch_notes",
            "stats": {"duration_sec": 60},
            "keyframes": [],
        }
        transcript = make_srt([
            (1, "新架构采用3nm制程和Chiplet设计"),
            (2, "HBM3E内存带宽达到8TB/s，并通过NVLink互连"),
        ])
        brief = compose_brief_base(manifest, transcript, "chip")
        self.assertIn("## 发布会概述", brief)
        self.assertIn("产品总结", brief)
        self.assertIn("传播总结", brief)
        self.assertIn("## 架构与制程", brief)
        self.assertIn("## 存储、互连与 I/O", brief)
        self.assertIn("存储、带宽与互连", brief)

    def test_english_segment_titles_do_not_match_substrings_as_product_tiers(self) -> None:
        manifest = {
            "event_name": "ChatGPT 发布会",
            "domain": {"requested": "foundation_model", "resolved": "foundation_model", "confidence": 1.0},
            "task_type": "business_review",
            "stats": {"duration_sec": 60},
            "keyframes": [],
        }
        transcript = make_srt([(1, "We use this experience to present AI research professionally.")])
        brief = compose_brief_base(manifest, transcript, "model")
        self.assertNotIn("AI / SE / Pro", brief)
        self.assertNotIn("se / pro", brief.lower())

    def test_foundation_model_segment_titles_use_domain_topics(self) -> None:
        manifest = {
            "event_name": "ChatGPT 发布会",
            "domain": {"requested": "foundation_model", "resolved": "foundation_model", "confidence": 1.0},
            "task_type": "business_review",
            "stats": {"duration_sec": 60},
            "keyframes": [],
        }
        transcript = make_srt([(1, "ChatGPT Work connects to Slack."), (2, "The desktop app can use local files and computer use.")])
        brief = compose_brief_base(manifest, transcript, "model")
        self.assertIn("ChatGPT Work", brief)
        self.assertIn("Desktop App 与 Computer Use", brief)

    def test_phone_business_review_keeps_legacy_structure(self) -> None:
        manifest = {
            "event_name": "手机发布会",
            "domain": {"requested": "smartphone", "resolved": "smartphone", "confidence": 1.0},
            "task_type": "business_review",
            "stats": {"duration_sec": 60},
            "keyframes": [],
        }
        brief = compose_brief_base(manifest, make_srt([(1, "手机售价2699元")]), "phone")
        self.assertIn("## 参数表：", brief)
        self.assertIn("产品：[待补", brief)

    def test_faithful_transcript_uses_transcript_sections(self) -> None:
        manifest = {
            "event_name": "模型发布会",
            "domain": {"requested": "foundation_model", "resolved": "foundation_model", "confidence": 1.0},
            "task_type": "faithful_transcript",
            "stats": {"duration_sec": 60},
            "keyframes": [],
        }
        brief = compose_brief_base(manifest, make_srt([(1, "模型上下文是一百万Token")]), "model")
        self.assertIn("## 逐字稿正文", brief)
        self.assertIn("模型上下文是一百万Token", brief)
        self.assertNotIn("## 商业化判断", brief)

    def test_automotive_prompt_does_not_apply_phone_judgment_template(self) -> None:
        domain, _ = domain_from_manifest({"domain": {"requested": "automotive", "resolved": "automotive"}})
        prompt = build_prompt(
            "新车发布会",
            "## 车型与版本矩阵",
            {},
            transcript_evidence_text="CLTC 续航 820 公里",
            domain=domain,
            domain_resolution={"confidence": 1.0, "signals": ["人工指定"]},
            task_type="business_review",
        )
        self.assertIn("严格区分 CLTC/WLTC", prompt)
        self.assertIn("车型与版本矩阵", prompt)
        self.assertNotIn("大年", prompt)
        self.assertNotIn("影像", prompt)

    def test_semiconductor_prompt_requires_metric_boundaries(self) -> None:
        domain, _ = domain_from_manifest({"domain": {"requested": "semiconductor", "resolved": "semiconductor"}})
        prompt = build_prompt(
            "芯片发布会",
            "## 计算性能与测试条件",
            {},
            transcript_evidence_text="FP8 峰值算力 20 PFLOPS，整卡功耗 1000W",
            domain=domain,
            domain_resolution={"confidence": 1.0, "signals": ["人工指定"]},
            task_type="business_review",
        )
        self.assertIn("理论峰值", prompt)
        self.assertIn("稠密/稀疏", prompt)
        self.assertIn("TDP、TBP", prompt)
        self.assertNotIn("影像", prompt)

    def test_domain_hotwords_are_injected_before_generic_terms(self) -> None:
        context = build_asr_hotwords({
            "event_name": "AI 模型发布",
            "domain": {"requested": "foundation_model", "resolved": "foundation_model"},
        })
        self.assertEqual(context["domain"], "foundation_model")
        self.assertIn("Token", context["terms"])
        self.assertIn("Benchmark", context["terms"])
        self.assertEqual(context["categories"], [])
        self.assertNotIn("电池", context["terms"])
        self.assertIn("大模型/AI发布会", asr_hotword_prompt(context))
        self.assertNotIn("消费电子发布会专名", asr_hotword_prompt(context))

    def test_review_video_help_uses_domain_guidance(self) -> None:
        domain, _ = domain_from_manifest({"domain": {"requested": "robotics", "resolved": "robotics"}})
        prompt = build_review_video_chunk_prompt(
            product_name="机器人 A",
            media_name="媒体",
            video_title="实测",
            chunk_index=1,
            chunk_count=1,
            transcript_chunk="演示中有遥操作介入",
            domain=domain,
        )
        self.assertIn("机器人", prompt)
        self.assertIn("自主执行与遥操作", prompt)


if __name__ == "__main__":
    unittest.main()
