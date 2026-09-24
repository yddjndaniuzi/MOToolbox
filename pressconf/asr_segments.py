"""Bounded audio decoding with local recovery and explicit uncertain intervals."""
from __future__ import annotations

import math
import re
from collections import Counter

VERSION = 4
CHUNK_SECONDS = 180
OVERLAP_SECONDS = 3
SILENCE_SPLIT_SECONDS = 12
UNCERTAIN_AUDIO = "[转写异常，待核实音频；不能据此判断无内容或发布会结束]"
NO_SPEECH_AUDIO = "[未检测到可转写的人声，请核对音频源]"
KNOWN_HALLUCINATIONS = (
    "优优独播剧场",
    "yoyotelevisionseriesexclusive",
    "中文字幕志愿者李宗盛",
    "明镜与点点栏目",
    "欢迎收看订阅的频道",
)
LONG_HALLUCINATIONS = {
    "thankyou",
    "iknowyou",
    "andthemomentohcomealive",
    "thatthatthat",
    "阿莱拥有电影感",
}


def compact_text(text):
    return re.sub(r"[\W_]", "", str(text or "")).lower()


def is_known_hallucination(text, duration=0):
    compact = compact_text(text)
    return (any(phrase in compact for phrase in KNOWN_HALLUCINATIONS)
            or duration >= 15 and (compact in LONG_HALLUCINATIONS or "字幕志愿者" in compact))


def repeated_long_keys(segments):
    ordered = sorted(segments, key=lambda s: float(s.get("start", 0)))
    result = set()
    for previous, current in zip(ordered, ordered[1:]):
        key = compact_text(current.get("text", ""))
        if (key and key == compact_text(previous.get("text", ""))
                and float(previous.get("end", 0)) - float(previous.get("start", 0)) >= 15
                and float(current.get("end", 0)) - float(current.get("start", 0)) >= 15
                and float(current.get("start", 0)) <= float(previous.get("end", 0)) + 5):
            result.add(key)
    return result


def issues(segments, duration, has_speech=None):
    texts = [compact_text(s.get("text", "")) for s in segments]
    texts = [t for t in texts if t]
    reasons = []
    if not texts:
        return ["empty"]
    if any(n >= 4 for n in Counter(texts).values()):
        reasons.append("repeated-cues")
    if repeated_long_keys(segments):
        reasons.append("repeated-long-cues")
    if any(is_known_hallucination(s.get("text", ""),
                                  float(s.get("end", 0)) - float(s.get("start", 0))) for s in segments):
        reasons.append("known-hallucination")
    if has_speech is not None and any(
        float(s.get("end", 0)) - float(s.get("start", 0)) >= 15
        and not has_speech(float(s.get("start", 0)), float(s.get("end", 0)))
        for s in segments
    ):
        reasons.append("long-cue-no-speech")
    if duration >= 30 and sum(map(len, texts)) < duration * 0.3:
        reasons.append("sparse-text")
    for s in segments:
        values = [float(s.get(k, 0) or 0) for k in ("avg_logprob", "compression_ratio", "no_speech_prob")]
        if not all(math.isfinite(v) for v in values) or values[0] < -1.5 or values[1] > 3:
            reasons.append("decoder-quality")
            break
    return reasons


def speech_windows(regions, start, end):
    """Keep short pauses as context, but exclude long intervals without voice."""
    windows = []
    for left, right in sorted(regions):
        left, right = max(start, float(left)), min(end, float(right))
        if right <= left:
            continue
        if windows and left - windows[-1][1] <= SILENCE_SPLIT_SECONDS:
            windows[-1] = (windows[-1][0], max(windows[-1][1], right))
        else:
            windows.append((left, right))
    return windows


def transcribe_chunks(audio, decode, progress=None, has_speech=None, speech_regions=None):
    duration = len(audio) / 16000
    merged, reports, languages = [], [], []
    decoded_windows = 0
    skipped_seconds = 0.0

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
        bad = issues(selected, end - start, has_speech)
        reports.append(dict(start=start, end=end, retry=retry, language=language, issues=bad))
        if bad and not retry:
            recovered = []
            sub = start
            while sub < end:
                recovered.extend(run(sub, min(sub + 60, end), True))
                sub += 60
            return recovered
        if bad:
            # A failed transcript is not evidence that the speaker had nothing to say.
            counts = Counter(compact_text(s.get("text", "")) for s in selected)
            repeated_long = repeated_long_keys(selected)
            usable = [s for s in selected if counts[compact_text(s.get("text", ""))] < 4
                      and compact_text(s.get("text", "")) not in repeated_long
                      and not issues([s], 0, has_speech)]
            marker = {"start": start, "end": end, "text": UNCERTAIN_AUDIO}
            return [marker, *usable]
        return selected

    for index, start in enumerate(range(0, math.ceil(duration), CHUNK_SECONDS), 1):
        end = min(start + CHUNK_SECONDS, duration)
        windows = [(start, end)] if speech_regions is None else speech_windows(speech_regions, start, end)
        skipped_seconds += end - start - sum(right - left for left, right in windows)
        for left, right in windows:
            decoded_windows += 1
            merged.extend(run(left, right))
        if progress:
            progress(index, math.ceil(duration / CHUNK_SECONDS))
    if speech_regions is not None and not decoded_windows and duration > 0:
        merged.append({"start": 0, "end": duration, "text": NO_SPEECH_AUDIO})
    return {"segments": merged, "language": Counter(languages).most_common(1)[0][0] if languages else "unknown",
            "audio_chunks": reports, "asr_pipeline_version": VERSION,
            "vad_predecode": speech_regions is not None, "vad_decoded_windows": decoded_windows,
            "vad_skipped_seconds": round(max(0.0, skipped_seconds), 3),
            "vad_no_speech": speech_regions is not None and not decoded_windows,
            "unresolved_intervals": [r for r in reports if r["retry"] and r["issues"]]}
