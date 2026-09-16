"""Bounded audio decoding with local recovery and explicit uncertain intervals."""
from __future__ import annotations

import math
import re
from collections import Counter

VERSION = 1
CHUNK_SECONDS = 180
OVERLAP_SECONDS = 3


def issues(segments, duration):
    texts = [re.sub(r"[\W_]", "", s.get("text", "")).lower() for s in segments]
    texts = [t for t in texts if t]
    reasons = []
    if not texts:
        return ["empty"]
    if any(n >= 4 for n in Counter(texts).values()):
        reasons.append("repeated-cues")
    if duration >= 30 and sum(map(len, texts)) < duration * 0.3:
        reasons.append("sparse-text")
    for s in segments:
        values = [float(s.get(k, 0) or 0) for k in ("avg_logprob", "compression_ratio", "no_speech_prob")]
        if not all(math.isfinite(v) for v in values) or values[0] < -1.5 or values[1] > 3:
            reasons.append("decoder-quality")
            break
    return reasons


def transcribe_chunks(audio, decode, progress=None):
    duration = len(audio) / 16000
    merged, reports, languages = [], [], []

    def run(start, end, retry=False):
        left, right = max(0, start - OVERLAP_SECONDS), min(duration, end + OVERLAP_SECONDS)
        try:
            result = decode(audio[round(left * 16000):round(right * 16000)], retry)
        except Exception:
            if retry:
                raise
            result = {"segments": []}
        language = result.get("language", "unknown")
        languages.append(language)
        selected = []
        for raw in result.get("segments", []):
            s = dict(raw)
            a, b = float(s.get("start", 0)) + left, float(s.get("end", 0)) + left
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            # The overlap provides speech context; each cue belongs to one core.
            if start <= (a + b) / 2 < end and b > a:
                s.update(start=max(start, a), end=min(end, b))
                selected.append(s)
        bad = issues(selected, end - start)
        reports.append(dict(start=start, end=end, retry=retry, language=language, issues=bad))
        if bad and not retry:
            recovered = []
            for sub in range(round(start), math.ceil(end), 60):
                recovered.extend(run(sub, min(sub + 60, end), True))
            return recovered
        if bad:
            # A failed transcript is not evidence that the speaker had nothing to say.
            usable = [s for s in selected if not issues([s], 0)] if "repeated-cues" not in bad else []
            marker = {"start": start, "end": end, "text": "[转写异常，待核实音频；不能据此判断无内容或发布会结束]"}
            return [marker, *usable]
        return selected

    for index, start in enumerate(range(0, math.ceil(duration), CHUNK_SECONDS), 1):
        merged.extend(run(start, min(start + CHUNK_SECONDS, duration)))
        if progress:
            progress(index, math.ceil(duration / CHUNK_SECONDS))
    return {"segments": merged, "language": Counter(languages).most_common(1)[0][0] if languages else "unknown",
            "audio_chunks": reports, "asr_pipeline_version": VERSION,
            "unresolved_intervals": [r for r in reports if r["retry"] and r["issues"]]}
