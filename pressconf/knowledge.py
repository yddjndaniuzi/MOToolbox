from __future__ import annotations

import re
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pressconf.config_store import index_root, load_knowledge_config, write_json, read_json


def index_path(base_dir: Path) -> Path:
    return index_root(base_dir) / "obsidian_pressconf_index.json"


def build_knowledge_index(base_dir: Path) -> dict[str, Any]:
    config = load_knowledge_config(base_dir)
    root = Path(config["vault_path"]).expanduser()
    scan_root = root / config.get("include_subdir", "") if config.get("include_subdir") else root
    documents = []
    if scan_root.exists():
        for path in iter_markdown_files(scan_root):
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = path.read_text(encoding="utf-8", errors="ignore")
            documents.append(extract_document(path, root, content))

    index = {
        "config": config,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "document_count": len(documents),
        "documents": documents,
    }
    write_json(index_path(base_dir), index)
    return index


def iter_markdown_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = [item for item in dirnames if not item.startswith(".")]
        for filename in filenames:
            if filename.startswith(".") or not filename.endswith(".md"):
                continue
            path = Path(current) / filename
            try:
                if path.stat().st_size > 2_000_000:
                    continue
            except OSError:
                continue
            paths.append(path)
    return sorted(paths)


def load_knowledge_index(base_dir: Path) -> dict[str, Any]:
    return read_json(index_path(base_dir), {"document_count": 0, "documents": []})


def extract_document(path: Path, vault_root: Path, content: str) -> dict[str, Any]:
    title = path.stem
    overview = extract_section(content, "概述")
    price = extract_section(content, "价格")
    product = extract_section(content, "产品概要")
    params = extract_section(content, "参数表")
    text_for_keywords = " ".join([title, overview, price, product])
    return {
        "title": title,
        "path": str(path),
        "relative_path": str(path.relative_to(vault_root)) if vault_root in path.parents else str(path),
        "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "overview": truncate(overview, 2600),
        "price": truncate(price, 1400),
        "product": truncate(product, 900),
        "params": truncate(params, 900),
        "keywords": extract_keywords(text_for_keywords),
    }


def extract_section(content: str, heading: str) -> str:
    pattern = re.compile(rf"^##\s*{re.escape(heading)}[：:]?\s*$", re.MULTILINE)
    match = pattern.search(content)
    if not match:
        return ""
    start = match.end()
    next_heading = re.search(r"^##\s+", content[start:], flags=re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(content)
    return content[start:end].strip()


def extract_keywords(text: str) -> list[str]:
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9+_-]{1,}|[\u4e00-\u9fff]{2,}", text)
    stopwords = {"发布会", "简报", "信息", "提要", "系列", "产品", "概述", "价格", "参数", "官方", "升级"}
    seen = set()
    result = []
    for item in candidates:
        token = item.strip()
        if not token or token in stopwords or token.lower() in stopwords:
            continue
        key = token.lower()
        if key not in seen:
            seen.add(key)
            result.append(token)
    return result[:80]


def search_knowledge(base_dir: Path, query: str, limit: int = 8) -> list[dict[str, Any]]:
    query = query.strip()
    if not query:
        return []
    index = load_knowledge_index(base_dir)
    tokens = extract_keywords(query)
    if not tokens:
        tokens = [query]
    scored = []
    for doc in index.get("documents", []):
        score = score_document(doc, tokens)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [with_score(doc, score) for score, doc in scored[:limit]]


def score_document(doc: dict[str, Any], tokens: list[str]) -> int:
    haystacks = {
        "title": str(doc.get("title", "")),
        "keywords": " ".join(doc.get("keywords", [])),
        "overview": str(doc.get("overview", "")),
        "price": str(doc.get("price", "")),
    }
    score = 0
    for token in tokens:
        lower = token.lower()
        if lower in haystacks["title"].lower():
            score += 12
        if lower in haystacks["keywords"].lower():
            score += 7
        if lower in haystacks["overview"].lower():
            score += 4
        if lower in haystacks["price"].lower():
            score += 2
    return score


def with_score(doc: dict[str, Any], score: int) -> dict[str, Any]:
    result = dict(doc)
    result["score"] = score
    result["overview_excerpt"] = truncate(clean_markdown(str(doc.get("overview", ""))), 420)
    return result


def clean_markdown(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def truncate(value: str, limit: int) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"
