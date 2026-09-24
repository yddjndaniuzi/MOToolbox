import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pressconf.asr_segments import issues, transcribe_chunks, VERSION
from pressconf.transcript import asr_cache_usable, read_existing_transcript, sanitize_asr_artifact_phrases, sanitize_cached_asr
from pressconf.refine import load_raw_transcript


class AudioSegmentationTests(unittest.TestCase):
    def test_repeated_short_cues_and_nan_are_rejected(self):
        self.assertIn("repeated-cues", issues([{"text": "I'm out."}] * 5, 120))
        self.assertIn("decoder-quality", issues([{"text": "真实内容", "avg_logprob": float("nan")}], 1))

    def test_known_artifact_and_long_cue_without_speech_are_retried(self):
        fake = {"start": 5, "end": 35, "text": "优优独播剧场——YoYo Television Series Exclusive"}
        self.assertIn("known-hallucination", issues([fake], 60))
        ordinary = {"start": 5, "end": 35, "text": "今天发布新产品"}
        self.assertIn("long-cue-no-speech", issues([ordinary], 60, lambda *_: False))
        self.assertNotIn("long-cue-no-speech", issues([ordinary], 60, lambda *_: True))
        result = transcribe_chunks(
            np.zeros(60 * 16000),
            lambda *_: {"segments": [fake]},
        )
        self.assertTrue(result["unresolved_intervals"])
        self.assertNotIn("优优", " ".join(s["text"] for s in result["segments"]))
        self.assertNotIn("known-hallucination", issues([{"text": "Thank you.", "start": 0, "end": 2}], 2))
        self.assertIn("known-hallucination", issues([{"text": "Thank you.", "start": 0, "end": 29}], 29))
        repeated = [{"text": "星轨银和月轮运", "start": i * 30, "end": (i + 1) * 30} for i in range(2)]
        self.assertIn("repeated-long-cues", issues(repeated, 60))
        retried = transcribe_chunks(np.zeros(60 * 16000),
                                    lambda *_: {"segments": repeated},
                                    has_speech=lambda *_: True)
        self.assertTrue(retried["unresolved_intervals"])
        self.assertNotIn("星轨银和月轮运", " ".join(s["text"] for s in retried["segments"]))

    def test_cached_asr_masks_only_confirmed_artifact_cues(self):
        source = ("1\n00:00:01,000 --> 00:00:30,000\n优优独播剧场——YoYo Television Series Exclusive\n\n"
                  "2\n00:00:31,000 --> 00:00:33,000\n悠悠助手正式发布\n")
        cleaned = sanitize_cached_asr(source)
        self.assertIn("转写异常", cleaned)
        self.assertNotIn("优优独播剧场", cleaned)
        self.assertIn("悠悠助手正式发布", cleaned)
        brief = sanitize_asr_artifact_phrases("优优独播剧场——YoYo Television Series Exclusive 女士们先生们欢迎新品")
        self.assertNotIn("独播剧场", brief)
        self.assertIn("女士们先生们欢迎新品", brief)

    def test_cached_asr_masks_long_and_repeated_cues_but_keeps_short_thanks(self):
        rows = [(0, 29, "Thank you."), (30, 59, "Thank you."),
                (60, 62, "Thank you."), (63, 93, "欢迎收看订阅的频道"),
                (94, 124, "星轨银和月轮运"), (124, 154, "星轨银和月轮运"),
                (155, 157, "产品发布")]
        def stamp(value):
            return f"00:{value // 60:02}:{value % 60:02},000"
        source = "\n\n".join(f"{i}\n{stamp(a)} --> {stamp(b)}\n{text}"
                              for i, (a, b, text) in enumerate(rows, 1))
        cleaned = sanitize_cached_asr(source)
        self.assertEqual(cleaned.count("Thank you."), 1)
        self.assertNotIn("欢迎收看订阅的频道", cleaned)
        self.assertNotIn("星轨银和月轮运", cleaned)
        self.assertIn("产品发布", cleaned)
        self.assertIn("疑似 ASR 幻觉", sanitize_asr_artifact_phrases("请不吝点赞、订阅、转发、打赏支持明镜与点点栏目"))

    def test_retry_recovers_tail_and_keeps_absolute_timestamps(self):
        audio = np.zeros(361 * 16000, dtype=np.float32)
        calls = []
        def decode(samples, retry):
            calls.append((len(samples) / 16000, retry))
            duration = len(samples) / 16000
            if not retry and len(calls) == 2:
                return {"segments": [{"start": i * 30, "end": (i + 1) * 30, "text": "I'm out."} for i in range(6)]}
            return {"language": "zh", "segments": [{"start": i, "end": min(i + 1, duration), "text": f"真实产品介绍和合作信息{i}"} for i in range(int(duration))]}
        result = transcribe_chunks(audio, decode)
        self.assertEqual(len([c for c in calls if c[1]]), 3)
        self.assertFalse(result["unresolved_intervals"])
        self.assertEqual(result["segments"][-1]["end"], 361)
        self.assertTrue(any(s["start"] >= 240 for s in result["segments"]))
        self.assertLessEqual(max(c[0] for c in calls), 186)

    def test_failed_retry_is_visible_instead_of_no_content(self):
        result = transcribe_chunks(np.zeros(65 * 16000), lambda *_: {"segments": []})
        self.assertEqual(len(result["unresolved_intervals"]), 2)
        self.assertIn("转写异常", result["segments"][0]["text"])
        self.assertEqual(result["segments"][-1]["end"], 65)

    def test_predecode_vad_decodes_only_voiced_windows_with_original_times(self):
        calls = []
        def decode(samples, retry):
            calls.append((len(samples) / 16000, retry))
            return {"language": "zh", "segments": [{"start": 5, "end": 10, "text": "真实产品发言"}]}
        result = transcribe_chunks(np.zeros(120 * 16000), decode,
                                   speech_regions=[(30, 50), (90, 110)])
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(length < 30 and not retry for length, retry in calls))
        self.assertEqual([round(s["start"]) for s in result["segments"]], [32, 92])
        self.assertTrue(result["vad_predecode"])
        self.assertEqual(result["vad_decoded_windows"], 2)
        self.assertEqual(result["vad_skipped_seconds"], 80)
        silent = transcribe_chunks(np.zeros(30 * 16000),
                                   lambda *_: self.fail("decoder received silence"), speech_regions=[])
        self.assertTrue(silent["vad_no_speech"])
        self.assertIn("未检测到可转写的人声", silent["segments"][0]["text"])

    def test_old_long_asr_cache_expires_but_unresolved_current_cache_survives(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            manifest = {"stats": {"duration_sec": 7500}}
            self.assertFalse(asr_cache_usable(p / "asr.srt", manifest))
            self.assertTrue(asr_cache_usable(p / "source.srt", manifest))
            (p / "quality.json").write_text(json.dumps({"asr_pipeline_version": VERSION}))
            self.assertTrue(asr_cache_usable(p / "asr.srt", manifest))
            (p / "quality.json").write_text(json.dumps({"asr_pipeline_version": 1}))
            self.assertTrue(asr_cache_usable(p / "asr.srt", manifest))
            (p / "quality.json").write_text(json.dumps({"asr_pipeline_version": 2}))
            self.assertTrue(asr_cache_usable(p / "asr.srt", manifest))
            (p / "quality.json").write_text(json.dumps({"asr_pipeline_version": VERSION, "unresolved_intervals": [1]}))
            self.assertTrue(asr_cache_usable(p / "asr.srt", manifest))

    def test_refine_reads_saved_transcript_with_quality_gaps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            transcript_dir = root / "transcript"
            transcript_dir.mkdir()
            (transcript_dir / "asr.srt").write_text(
                "1\n00:00:00,000 --> 00:00:01,000\n产品发布\n\n"
                "2\n00:00:01,000 --> 00:00:30,000\n优优独播剧场——YoYo Television Series Exclusive\n",
                encoding="utf-8",
            )
            (transcript_dir / "meta.json").write_text(json.dumps({"method": "asr", "path": "transcript/asr.srt"}))
            (transcript_dir / "quality.json").write_text(json.dumps({"unresolved_intervals": [{"start": 20}]}))
            content, meta = read_existing_transcript(root)
            self.assertIn("产品发布", content)
            self.assertNotIn("优优独播剧场", content)
            self.assertNotIn("优优独播剧场", load_raw_transcript(root, meta))
            self.assertEqual(meta["method"], "asr")
            self.assertEqual(len(meta["quality"]["unresolved_intervals"]), 1)
