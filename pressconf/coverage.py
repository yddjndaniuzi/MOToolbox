from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pressconf.brief import Cue, parse_transcript
from pressconf.keyframes import format_timestamp


FACT_PATTERN = re.compile(
    r"(?:[￥$¥]\s*)?\d+(?:[.,]\d+)*(?:\s*(?:元|万元|万|亿|%|GB|TB|MB|W|mAh|Hz|英寸|克|mm|nm|fps|款|色|倍))?",
    re.IGNORECASE,
)
ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9+.-]*\s*){1,4}(?:Pro|Ultra|Max|Mini|Air|Fold|Flip|OS|AI|芯片|系列)?\b"
)
PRICE_HINT = re.compile(r"(售价|价格|起售|首销|首发|国补|优惠|赠送|权益|SKU|版本|配置|元|￥|¥|\$)", re.IGNORECASE)


def transcript_evidence(content: str, max_chars: int = 100_000) -> tuple[str, dict[str, Any]]:
    """Build time-coded evidence for refinement without silently keeping only the beginning."""
    cues = parse_transcript(content)
    rows = [format_cue(cue) for cue in cues if cue.text.strip()]
    total_chars = sum(len(row) for row in rows)
    if total_chars <= max_chars:
        selected = rows
        strategy = "full"
    else:
        selected = select_balanced_rows(cues, max_chars)
        strategy = "fact-priority-balanced"
    evidence = "\n".join(selected)
    return evidence, {
        "strategy": strategy,
        "cue_count": len(cues),
        "selected_cue_count": len(selected),
        "source_chars": total_chars,
        "evidence_chars": len(evidence),
        "coverage_ratio": round(len(selected) / max(len(cues), 1), 4),
    }


def format_cue(cue: Cue) -> str:
    if cue.end > 0:
        return f"[{format_timestamp(cue.start)}-{format_timestamp(cue.end)}] {cue.text.strip()}"
    return cue.text.strip()


def select_balanced_rows(cues: list[Cue], max_chars: int) -> list[str]:
    if not cues:
        return []
    priority = {index for index, cue in enumerate(cues) if FACT_PATTERN.search(cue.text) or PRICE_HINT.search(cue.text)}
    selected = set(priority)
    used = sum(len(format_cue(cues[index])) + 1 for index in selected)

    # Fill remaining capacity evenly across the complete timeline, not from the start.
    remaining = [index for index in range(len(cues)) if index not in selected]
    if remaining and used < max_chars:
        average = max(1, sum(len(format_cue(cues[index])) + 1 for index in remaining) // len(remaining))
        slots = max(1, (max_chars - used) // average)
        if slots >= len(remaining):
            selected.update(remaining)
        else:
            for slot in range(slots):
                pos = round(slot * (len(remaining) - 1) / max(slots - 1, 1))
                selected.add(remaining[pos])

    result: list[str] = []
    used = 0
    for index in sorted(selected):
        row = format_cue(cues[index])
        if used + len(row) + 1 > max_chars:
            continue
        result.append(row)
        used += len(row) + 1
    return result


def extract_facts(text: str) -> list[str]:
    facts: list[str] = []
    seen: set[str] = set()
    for match in FACT_PATTERN.finditer(text):
        value = re.sub(r"[\s,，]+", "", match.group(0)).strip(".。")
        if not value or value in seen:
            continue
        seen.add(value)
        facts.append(value)
    return facts


def build_coverage_report(
    *,
    result_dir: Path,
    transcript: str,
    base_text: str,
    refined_text: str = "",
    evidence_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clean_transcript = "\n".join(cue.text for cue in parse_transcript(transcript))
    stages = {"transcript": clean_transcript, "brief_base": base_text, "brief_refined": refined_text}
    stage_facts = {name: set(extract_facts(text)) for name, text in stages.items()}
    source_facts = extract_facts(clean_transcript)
    report = {
        "schema_version": 1,
        "characters": {name: len(text) for name, text in stages.items()},
        "facts": {
            "source_count": len(source_facts),
            "base_retained": [fact for fact in source_facts if fact in stage_facts["brief_base"]],
            "base_missing": [fact for fact in source_facts if fact not in stage_facts["brief_base"]],
            "refined_retained": [fact for fact in source_facts if fact in stage_facts["brief_refined"]],
            "refined_missing": [fact for fact in source_facts if fact not in stage_facts["brief_refined"]],
        },
        "evidence": evidence_meta or {},
    }
    report["facts"]["base_retention_ratio"] = round(
        len(report["facts"]["base_retained"]) / max(len(source_facts), 1), 4
    )
    report["facts"]["refined_retention_ratio"] = round(
        len(report["facts"]["refined_retained"]) / max(len(source_facts), 1), 4
    )
    (result_dir / "coverage_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
