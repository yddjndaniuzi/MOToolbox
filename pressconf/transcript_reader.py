from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from pressconf.brief import Segment, parse_transcript, segment_cues
from pressconf.config_store import resolve_model
from pressconf.derivatives import call_writing_model_streaming
from pressconf.model_client import is_model_length_error


TRANSCRIPT_READER_SYSTEM = (
    "你是消费电子评测视频逐字稿精校助手。你的任务是把 ASR 原始转写整理得可读，"
    "但不能补写原文没有的信息，不能总结、删改观点、添加说话人或改变表达立场。"
    "遇到听不清、术语不确定或 ASR 明显破碎的地方，保留原文最稳妥的表达，不要猜测。"
)
SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?，。；：！？、])")
TIMESTAMP_PATTERN = re.compile(r"^\s*(?:\[[^\]]+\]\s*)+")
POLISH_MAX_TOKENS = 3600
POLISH_MIN_SPLIT_CHARS = 120
POLISH_MAX_SPLIT_DEPTH = 6


def build_transcript_reader(
    *,
    base_dir: Path,
    transcript_text: str,
    transcript_meta: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cues = parse_transcript(transcript_text)
    segments = segment_cues(cues, window_seconds=75)
    if not segments:
        raise RuntimeError("没有可用于逐字稿查阅的时间段。")

    polish_asr = transcript_meta.get("method") == "asr" and has_timeline(segments)
    polished: list[str] = []
    model_meta: dict[str, Any] = {}
    if polish_asr:
        polished, model_meta = polish_segments(
            base_dir=base_dir,
            segments=segments,
            progress_callback=progress_callback,
        )
    else:
        polished = [readable_text(segment.text) for segment in segments]

    reader_segments = [
        {
            "index": index,
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": text or readable_text(segment.text),
            "source_text": readable_text(segment.text),
        }
        for index, (segment, text) in enumerate(zip(segments, polished), start=1)
    ]
    meta = {
        "segment_count": len(reader_segments),
        "polished": polish_asr,
        "source_method": transcript_meta.get("method", ""),
        **model_meta,
    }
    return reader_segments, meta


def write_transcript_reader(result_dir: Path, segments: list[dict[str, Any]], meta: dict[str, Any]) -> Path:
    path = result_dir / "transcript_reader.json"
    path.write_text(json.dumps({"segments": segments, "meta": meta}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_transcript_reader(result_dir: Path) -> dict[str, Any]:
    path = result_dir / "transcript_reader.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def has_timeline(segments: list[Segment]) -> bool:
    return any(segment.end > segment.start for segment in segments)


def polish_segments(
    *,
    base_dir: Path,
    segments: list[Segment],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    model_config = resolve_model(base_dir, "writing", "light")
    api_key = model_config["api_key"].strip()
    if not api_key:
        raise RuntimeError(f"没有配置 {model_config.get('name', model_config.get('model', '模型'))} 的 API Key。")

    polished: list[str] = []
    fallback_segments: list[int] = []
    for index, segment in enumerate(segments, start=1):
        try:
            result = polish_segment_with_recovery(
                text=segment.text,
                index=index,
                count=len(segments),
                model_config=model_config,
                progress_callback=progress_callback,
            )
        except RuntimeError as exc:
            if not is_model_length_error(exc):
                raise
            # Transcript polishing is a readability pass over a preserved ASR
            # source. If a tiny subpart still reaches the model cap, keep its
            # original wording and let the main analysis continue with evidence.
            result = segment.text
            fallback_segments.append(index)
        polished.append(clean_polished_text(result) or readable_text(segment.text))
    meta = {
        "model": model_config["model"].strip(),
        "provider": model_config["provider"].strip(),
    }
    if fallback_segments:
        meta["polish_fallback_segments"] = fallback_segments
        meta["warnings"] = [
            f"逐字稿第 {index} 段精校达到模型输出上限，已保留原始转写并继续分析。"
            for index in fallback_segments
        ]
    return polished, meta


def polish_segment_with_recovery(
    *,
    text: str,
    index: int,
    count: int,
    model_config: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    depth: int = 0,
) -> str:
    stage = f"逐字稿精校 {index}/{count}"
    try:
        return call_writing_model_streaming(
            api_key=model_config["api_key"].strip(),
            base_url=model_config["base_url"].rstrip("/"),
            model=model_config["model"].strip(),
            provider=model_config["provider"].strip(),
            prompt=polish_segment_prompt(text),
            # Small transcript excerpts can still exhaust a reasoning model's
            # budget before it finishes the visible text. Give each request a
            # stable output allowance, then reduce input size if it still hits it.
            max_tokens=POLISH_MAX_TOKENS,
            stage=stage,
            system_prompt=TRANSCRIPT_READER_SYSTEM,
            progress_callback=progress_callback,
            fallback_models=model_config.get("fallbacks"),
        )
    except RuntimeError as exc:
        if (not is_model_length_error(exc) or depth >= POLISH_MAX_SPLIT_DEPTH
                or len(text) <= POLISH_MIN_SPLIT_CHARS * 2):
            raise
        left, right = split_polish_text(text)
        if not left.strip() or not right.strip():
            raise
        if progress_callback:
            progress_callback({
                "stage": stage,
                "message": f"逐字稿第 {index}/{count} 段达到输出上限，已按原文边界拆成两段重试",
            })
        left_result = polish_segment_with_recovery(
            text=left, index=index, count=count, model_config=model_config,
            progress_callback=progress_callback, depth=depth + 1,
        )
        right_result = polish_segment_with_recovery(
            text=right, index=index, count=count, model_config=model_config,
            progress_callback=progress_callback, depth=depth + 1,
        )
        return join_polished_text(left_result, right_result)


def split_polish_text(text: str) -> tuple[str, str]:
    midpoint = len(text) // 2
    lower = max(POLISH_MIN_SPLIT_CHARS, int(len(text) * 0.35))
    upper = min(len(text) - POLISH_MIN_SPLIT_CHARS, int(len(text) * 0.65))
    if lower > upper:
        return text[:midpoint], text[midpoint:]

    strong = "。！？!?；;"
    weak = "，,、：: "
    for punctuation in (strong, weak):
        candidates = [position + 1 for position in range(lower, upper)
                      if text[position] in punctuation]
        if candidates:
            split_at = min(candidates, key=lambda position: abs(position - midpoint))
            return text[:split_at], text[split_at:]
    return text[:midpoint], text[midpoint:]


def join_polished_text(left: str, right: str) -> str:
    left, right = left.rstrip(), right.lstrip()
    if left and right and left[-1].isascii() and right[0].isascii() \
            and left[-1].isalnum() and right[0].isalnum():
        return left + " " + right
    return left + right


def polish_segment_prompt(text: str) -> str:
    return f"""请精校下面这段 ASR 逐字稿，输出便于逐字查阅的正文。

要求：
1. 修正断句、标点、明显重复和明显识别错字，让口语内容可读。
2. 保留原话的信息密度、语气、观点顺序和专有名词；不做摘要，不补事实，不翻译。
3. 不输出标题、时间码、说明、列表编号或代码块，只输出精校后的正文。
4. 如果原文不足以确认某个术语，保留原始表达，不要猜。

【ASR 原文】
{text.strip()}
"""


def clean_polished_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^\s*```(?:text|markdown|md)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = TIMESTAMP_PATTERN.sub("", text)
    text = re.sub(r"^(?:精校(?:后)?(?:的)?逐字稿|正文)\s*[:：]\s*", "", text)
    return readable_text(text)


def readable_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
