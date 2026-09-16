from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


DOMAIN_ROOT = Path(__file__).resolve().parent / "domains"
DEFAULT_DOMAIN = "generic"
DOMAIN_IDS = ("generic", "smartphone", "automotive", "foundation_model", "robotics", "semiconductor")
TASK_TYPES = {
    "faithful_transcript": "忠实转写",
    "launch_notes": "发布会速记",
    "business_review": "业务帮看",
    "fact_extract": "专项事实抽取",
}


@dataclass(frozen=True)
class DomainProfile:
    id: str
    name: str
    description: str
    keywords: tuple[str, ...]
    strong_keywords: tuple[str, ...]
    sections: tuple[str, ...]
    fact_types: tuple[str, ...]
    hotwords: tuple[str, ...]
    analyst_role: str
    guidance: str


@lru_cache(maxsize=16)
def load_domain(domain_id: str) -> DomainProfile:
    domain_id = domain_id if domain_id in DOMAIN_IDS else DEFAULT_DOMAIN
    path = DOMAIN_ROOT / domain_id / "domain.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return DomainProfile(
        id=domain_id,
        name=str(data["name"]),
        description=str(data.get("description") or ""),
        keywords=tuple(data.get("keywords") or ()),
        strong_keywords=tuple(data.get("strong_keywords") or ()),
        sections=tuple(data.get("sections") or ()),
        fact_types=tuple(data.get("fact_types") or ()),
        hotwords=tuple(data.get("hotwords") or ()),
        analyst_role=str(data.get("analyst_role") or "科技行业分析师"),
        guidance=str(data.get("guidance") or ""),
    )


def list_domains() -> list[DomainProfile]:
    return [load_domain(domain_id) for domain_id in DOMAIN_IDS]


def requested_domain(manifest: dict[str, Any]) -> str:
    domain = manifest.get("domain") or {}
    if isinstance(domain, str):
        return domain if domain in DOMAIN_IDS else "auto"
    value = str(domain.get("requested") or manifest.get("domain_id") or "auto")
    return value if value in DOMAIN_IDS else "auto"


def resolve_domain(manifest: dict[str, Any], transcript: str = "") -> dict[str, Any]:
    requested = requested_domain(manifest)
    if requested != "auto":
        return {"requested": requested, "resolved": requested, "confidence": 1.0, "signals": ["人工指定"]}

    source = manifest.get("source") or {}
    text = " ".join(
        str(value or "") for value in (
            manifest.get("event_name"), source.get("title"), source.get("url"), transcript[:20_000]
        )
    ).lower()
    scores: dict[str, float] = {}
    signals: dict[str, list[str]] = {}
    for domain_id in DOMAIN_IDS:
        if domain_id == DEFAULT_DOMAIN:
            continue
        profile = load_domain(domain_id)
        hits: list[str] = []
        score = 0.0
        for keyword in profile.keywords:
            count = len(re.findall(re.escape(keyword.lower()), text))
            if count:
                score += min(count, 4)
                hits.append(keyword)
        for keyword in profile.strong_keywords:
            count = len(re.findall(re.escape(keyword.lower()), text))
            if count:
                score += min(count, 3) * 3
                hits.append(keyword)
        scores[domain_id] = score
        signals[domain_id] = list(dict.fromkeys(hits))[:8]

    winner = max(scores, key=scores.get) if scores else DEFAULT_DOMAIN
    best = scores.get(winner, 0.0)
    total = sum(scores.values())
    confidence = best / max(total, 1.0)
    if best < 3 or confidence < 0.45:
        return {"requested": "auto", "resolved": DEFAULT_DOMAIN, "confidence": round(confidence, 3), "signals": []}
    return {
        "requested": "auto",
        "resolved": winner,
        "confidence": round(confidence, 3),
        "signals": signals[winner],
    }


def domain_from_manifest(manifest: dict[str, Any], transcript: str = "") -> tuple[DomainProfile, dict[str, Any]]:
    existing = manifest.get("domain") or {}
    if isinstance(existing, dict) and existing.get("resolved") and not transcript:
        resolution = existing
    else:
        resolution = resolve_domain(manifest, transcript)
    return load_domain(str(resolution.get("resolved") or DEFAULT_DOMAIN)), resolution


def task_type_from_manifest(manifest: dict[str, Any]) -> str:
    value = str(manifest.get("task_type") or "business_review")
    return value if value in TASK_TYPES else "business_review"


def sections_for_task(profile: DomainProfile, task_type: str) -> tuple[str, ...]:
    if task_type == "faithful_transcript":
        return ("视频信息", "校订说明", "逐字稿正文", "待核实词")
    if task_type == "fact_extract":
        return ("事实摘要", "实体与版本", "数字与指标", "价格与时间", "证据索引", "冲突与待核实项")
    sections = profile.sections
    if task_type in {"business_review", "launch_notes"} and not any("概述" in item for item in sections):
        return ("发布会概述",) + sections
    return sections
