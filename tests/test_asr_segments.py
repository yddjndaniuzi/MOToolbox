import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pressconf.asr_segments import issues, transcribe_chunks, VERSION
from pressconf.transcript import asr_cache_usable


class AudioSegmentationTests(unittest.TestCase):
    def test_repeated_short_cues_and_nan_are_rejected(self):
        self.assertIn("repeated-cues", issues([{"text": "I'm out."}] * 5, 120))
        self.assertIn("decoder-quality", issues([{"text": "真实内容", "avg_logprob": float("nan")}], 1))

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

    def test_old_long_asr_cache_expires_but_official_subtitles_survive(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            manifest = {"stats": {"duration_sec": 7500}}
            self.assertFalse(asr_cache_usable(p / "asr.srt", manifest))
            self.assertTrue(asr_cache_usable(p / "source.srt", manifest))
            (p / "quality.json").write_text(json.dumps({"asr_pipeline_version": VERSION}))
            self.assertTrue(asr_cache_usable(p / "asr.srt", manifest))
            (p / "quality.json").write_text(json.dumps({"asr_pipeline_version": VERSION, "unresolved_intervals": [1]}))
            self.assertFalse(asr_cache_usable(p / "asr.srt", manifest))
