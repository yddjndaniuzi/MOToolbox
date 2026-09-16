from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pressconf.domains import domain_from_manifest


HOTWORD_ROOT = Path(__file__).resolve().parent / "asr_hotwords"
ASR_PROMPT_MAX_CHARS = 360

BRAND_ALIASES: dict[str, tuple[str, ...]] = {
    "apple": ("apple", "苹果", "iphone", "ipad", "mac", "wwdc", "airpods", "vision pro"),
    "honor": ("honor", "荣耀", "magic"),
    "huawei": ("huawei", "华为", "mate", "pura", "nova", "pocket", "鸿蒙", "harmonyos"),
    "oppo": ("oppo", "oneplus", "一加", "find", "reno", "coloros"),
    "samsung": ("samsung", "三星", "galaxy", "one ui"),
    "vivo": ("vivo", "iqoo", "originos", "x fold", "x flip"),
    "xiaomi": ("xiaomi", "小米", "redmi", "红米", "hyperos", "澎湃"),
}

CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "ai_os": ("ai", "系统", "os", "开发者", "鸿蒙", "harmony", "coloros", "originos", "hyperos", "magicos"),
    "chips_components": ("芯片", "性能", "soc", "骁龙", "天玑", "处理器"),
    "foldables": ("折叠", "fold", "flip", "pocket"),
    "imaging": ("影像", "相机", "拍照", "视频", "人像", "长焦"),
    "pricing_sales": ("价格", "售价", "首销", "发布会", "发布", "新品"),
}


def build_asr_hotwords(manifest: dict[str, Any], max_terms: int = 240) -> dict[str, Any]:
    domain, domain_resolution = domain_from_manifest(manifest)
    # common.txt is historically a smartphone vocabulary (battery, telephoto,
    # satellite communication...), not a domain-neutral list.
    selected_files = [HOTWORD_ROOT / "common.txt"] if domain.id == "smartphone" else []

    brands = detect_brands(manifest) if domain.id in {"generic", "smartphone"} else []
    for brand in brands:
        path = HOTWORD_ROOT / "brands" / f"{brand}.txt"
        if path.exists():
            selected_files.append(path)
    haystack = manifest_text(manifest).lower()
    categories = [
        category for category, aliases in CATEGORY_ALIASES.items()
        if any(alias.lower() in haystack for alias in aliases)
    ] if domain.id in {"generic", "smartphone"} else []
    # Pricing vocabulary is small and important in virtually every hardware launch.
    if (
        domain.id in {"generic", "smartphone"}
        and "pricing_sales" not in categories
        and not any(key in haystack for key in ("wwdc", "开发者大会", "系统"))
    ):
        categories.append("pricing_sales")
    for category in categories:
        path = HOTWORD_ROOT / "categories" / f"{category}.txt"
        if path.exists():
            selected_files.append(path)

    terms = dedupe_terms([
        *domain.hotwords,
        *(term for path in selected_files for term in read_terms(path)),
    ])
    limited_terms = terms[:max_terms]
    return {
        "terms": limited_terms,
        "term_count": len(limited_terms),
        "available_term_count": len(terms),
        "brands": brands,
        "categories": categories,
        "domain": domain.id,
        "domain_name": domain.name,
        "domain_resolution": domain_resolution,
        "files": [str(path.relative_to(HOTWORD_ROOT)) for path in selected_files if path.exists()],
    }


def detect_brands(manifest: dict[str, Any]) -> list[str]:
    haystack = manifest_text(manifest).lower()
    brands = []
    for brand, aliases in BRAND_ALIASES.items():
        if any(alias.lower() in haystack for alias in aliases):
            brands.append(brand)
    return brands


def manifest_text(manifest: dict[str, Any]) -> str:
    source = manifest.get("source") or {}
    parts = [
        str(manifest.get("event_name") or ""),
        str(source.get("title") or ""),
        str(source.get("url") or ""),
        Path(str(manifest.get("video") or "")).stem if manifest.get("video") else "",
    ]
    return " ".join(parts)


def read_terms(path: Path) -> list[str]:
    if not path.exists():
        return []
    terms = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        term = line.strip()
        if not term or term.startswith("#"):
            continue
        terms.append(normalize_term(term))
    return [term for term in terms if term]


def normalize_term(term: str) -> str:
    term = re.sub(r"\s+", " ", term).strip()
    return term


def dedupe_terms(terms: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in terms:
        term = normalize_term(str(raw))
        key = term.lower()
        if not term or key in seen:
            continue
        seen.add(key)
        result.append(term)
    return result


def asr_hotword_prompt(context: dict[str, Any], max_chars: int = ASR_PROMPT_MAX_CHARS) -> str:
    terms = [str(term) for term in context.get("terms", []) if str(term).strip()]
    if not terms:
        return ""
    domain_name = str(context.get("domain_name") or "科技").strip()
    prefix = f"本场{domain_name}发布会可能出现的专名，仅用于辅助拼写，不代表音频中一定出现："
    selected: list[str] = []
    for term in terms:
        candidate = prefix + "、".join([*selected, term])
        if len(candidate) > max_chars:
            break
        selected.append(term)
    return prefix + "、".join(selected) if selected else ""
