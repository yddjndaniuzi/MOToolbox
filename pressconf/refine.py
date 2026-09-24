from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from pressconf.config_store import resolve_model, strength_for_tier
from pressconf.coverage import build_coverage_report, transcript_evidence
from pressconf.domains import DomainProfile, TASK_TYPES, domain_from_manifest, sections_for_task, task_type_from_manifest
from pressconf.model_client import call_chat_model, is_model_capacity_error, stream_chat_model
from pressconf.segmented_refine import refine_segmented, section_ids, validate_sections
from pressconf.transcript import sanitize_asr_artifact_phrases, sanitize_cached_asr

def refine_brief(
    result_dir: Path,
    display_name: str,
    base_dir: Path,
    user_instruction: str = "",
    tier_override: str = "",
    format_override: str = "",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    model_config: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    brief_path = result_dir / "brief_base.md"
    if not brief_path.exists():
        raise RuntimeError("还没有生成简报基础稿，请先完成转写和基础稿生成。")

    source = sanitize_asr_artifact_phrases(brief_path.read_text(encoding="utf-8"))
    manifest = load_json_file(result_dir / "manifest.json")
    domain, domain_resolution = domain_from_manifest(manifest)
    task_type = task_type_from_manifest(manifest)
    ledger_context = load_fact_ledger_context(result_dir)
    transcript_meta = load_transcript_meta(result_dir)
    transcript_text = load_raw_transcript(result_dir, transcript_meta)
    evidence, evidence_meta = transcript_evidence(transcript_text)
    volume = assess_information_volume(
        result_dir, source, transcript_meta, display_name, tier_override, format_override,
        domain_id=domain.id,
    )
    model_config = model_config or resolve_model(
        base_dir,
        "brief_refine",
        strength_for_tier(str(volume.get("tier") or "")),
    )
    api_key = model_config["api_key"].strip()
    base_url = model_config["base_url"].rstrip("/")
    model = model_config["model"].strip()
    provider = model_config["provider"].strip()
    if not api_key:
        raise RuntimeError(f"没有配置 {model_config.get('name', model)} 的 API Key。")
    partial_path = result_dir / "brief_refined.partial.md"
    if partial_path.exists():
        partial_path.unlink()
    generation_meta: dict[str, Any] = {}
    if len(section_ids(source)) > 4 or len(transcript_text) > 24_000:
        # Full evidence is consumed batch by batch; the old sampled evidence is
        # deliberately not passed to this path.
        instructions = build_prompt(
            display_name, "", transcript_meta, user_instruction, volume,
            domain=domain, domain_resolution=domain_resolution,
            task_type=task_type, ledger_context=ledger_context,
        )
        refined, generation_meta = refine_segmented(
            result_dir=result_dir, source=source, transcript=transcript_text,
            instructions=instructions, model_config=model_config,
            task_type=task_type, progress_callback=progress_callback,
        )
        evidence_meta = {
            "strategy": "full-transcript-segmented", "coverage_ratio": 1.0,
            "coverage_scope": "available_transcript_only",
            "source_chars": generation_meta["source_chars"],
            "batch_count": generation_meta["batch_count"],
        }
    else:
        candidates = [model_config, *(model_config.get("fallbacks") or [])]
        for candidate_index, candidate in enumerate(candidates):
            api_key = candidate["api_key"].strip()
            base_url = candidate["base_url"].rstrip("/")
            model = candidate["model"].strip()
            provider = candidate["provider"].strip()
            try:
                refined = call_deepseek_streaming(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    provider=provider,
                    display_name=display_name,
                    source=source,
                    transcript_evidence_text=evidence,
                    domain=domain,
                    domain_resolution=domain_resolution,
                    task_type=task_type,
                    ledger_context=ledger_context,
                    transcript_meta=transcript_meta,
                    user_instruction=user_instruction,
                    volume=volume,
                    partial_path=partial_path,
                    progress_callback=progress_callback,
                    generation_meta=generation_meta,
                )
                break
            except RuntimeError as exc:
                if not is_model_capacity_error(exc) or candidate_index + 1 >= len(candidates):
                    raise
                if partial_path.exists():
                    partial_path.unlink()
                if progress_callback:
                    progress_callback({"message": f"{model} 限流，正在切换备用模型"})
        else:
            raise RuntimeError("所有候选模型均不可用。")
    validate_sections(source, refined)
    refined = restore_image_blocks(source, refined)
    unresolved = (transcript_meta.get("quality") or {}).get("unresolved_intervals") or []
    if unresolved:
        refined = (f"> 转写完整性提示：{len(unresolved)} 个音频时段重试后仍待核实。"
                   "以下报告不能视为完整音频记录；文本覆盖率不代表音频识别完整率。\n\n" + refined)
    output_path = result_dir / "brief_refined.md"
    completed_path = result_dir / "brief_refined.complete.tmp"
    completed_path.write_text(refined.strip() + "\n", encoding="utf-8")
    completed_path.replace(output_path)
    coverage = build_coverage_report(
        result_dir=result_dir,
        transcript=transcript_text,
        base_text=source,
        refined_text=refined,
        evidence_meta=evidence_meta,
    )
    if partial_path.exists():
        partial_path.unlink()
    meta = {
        "model": model,
        "provider": provider,
        "base_url": base_url,
        "source": "brief_base.md",
        "output": "brief_refined.md",
        "source_language": transcript_meta.get("language", "unknown"),
        "output_language": transcript_meta.get("output_language", "zh-CN"),
        "user_instruction": user_instruction,
        "info_tier": volume.get("tier"),
        "info_tier_label": volume.get("label"),
        "info_signals": volume.get("signals"),
        "target_chars": volume.get("target_chars"),
        "event_format": volume.get("event_format"),
        "evidence": evidence_meta,
        "generation": generation_meta,
        "fact_retention": {
            "base": coverage["facts"]["base_retention_ratio"],
            "refined": coverage["facts"]["refined_retention_ratio"],
        },
        "domain": domain_resolution,
        "task_type": task_type,
    }
    (result_dir / "refine_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path, meta

def call_deepseek(api_key: str, base_url: str, model: str, display_name: str, source: str, provider: str = "openai-compatible") -> str:
    return call_chat_model(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是头部手机品牌市场/竞情团队的内部分析师。你的读者懂手机行业，"
                    "需要判断、节奏和信息密度，不需要基础科普。输出中文 Markdown，保留飞书 "
                    "<text color=\"...\"> 标签。不要编造价格、参数、竞品对比；没有依据就保留待补/待核实。"
                ),
            },
            {
                "role": "user",
                "content": build_prompt(display_name, source, {}),
            },
        ],
        temperature=0.35,
        max_tokens=12000,
        timeout=240,
    )


def call_deepseek_streaming(
    *,
    api_key: str,
    base_url: str,
    model: str,
    provider: str,
    display_name: str,
    source: str,
    transcript_evidence_text: str,
    domain: DomainProfile,
    domain_resolution: dict[str, Any],
    task_type: str,
    ledger_context: str,
    transcript_meta: dict[str, Any],
    user_instruction: str,
    volume: dict[str, Any] | None = None,
    partial_path: Path,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    generation_meta: dict[str, Any] | None = None,
) -> str:
    messages = [
            {
                "role": "system",
                "content": (
                    f"你是{domain.analyst_role}。读者是对应业务团队，需要准确证据、判断和信息密度。"
                    "输出中文 Markdown，保留飞书 "
                    "<text color=\"...\"> 标签。不要编造价格、参数、竞品对比；没有依据就保留待补/待核实。"
                    "如果源发布会是英文或中英混合，你要理解原文信息后转写成自然中文市场简报，"
                    "保留产品名、功能名、芯片名、技术名等英文专有名词原文，不要生硬逐句翻译。"
                ),
            },
            {
                "role": "user",
                "content": build_prompt(
                    display_name, source, transcript_meta, user_instruction, volume,
                    transcript_evidence_text=transcript_evidence_text,
                    domain=domain,
                    domain_resolution=domain_resolution,
                    task_type=task_type,
                    ledger_context=ledger_context,
                ),
            },
    ]
    chunks: list[str] = []

    def handle_delta(delta: str) -> None:
        chunks.append(delta)
        current = "".join(chunks)
        partial_path.write_text(current, encoding="utf-8")
        if progress_callback and len(current) % 80 < len(delta):
            progress_callback(
                {
                    "generated_chars": len(current),
                    "preview": tail_preview(current),
                }
            )

    result = stream_chat_model(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=messages,
        temperature=0.35,
        max_tokens=16000 if (volume or {}).get("tier") == "full" else 12000,
        timeout=300,
        on_delta=handle_delta,
        on_metadata=generation_meta.update if generation_meta is not None else None,
    )
    if not result:
        raise RuntimeError("模型没有返回内容。")
    if progress_callback:
        progress_callback({"generated_chars": len(result), "preview": tail_preview(result)})
    return result


def tail_preview(text: str, limit: int = 900) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def load_transcript_meta(result_dir: Path) -> dict[str, Any]:
    path = result_dir / "transcript" / "meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_fact_ledger_context(result_dir: Path, max_chars: int = 24_000) -> str:
    ledger = load_json_file(result_dir / "fact_ledger.json")
    rows: list[str] = []
    for fact in ledger.get("facts") or []:
        timestamp = fact.get("timestamp") or [0, 0]
        row = (
            f"[{timestamp[0]}-{timestamp[1]}] {fact.get('type')}: "
            f"{fact.get('value')}{fact.get('unit') or ''}；原文：{fact.get('source_text') or ''}"
        )
        if sum(len(item) + 1 for item in rows) + len(row) > max_chars:
            break
        rows.append(row)
    return "\n".join(rows)


# 产品线重要程度关键词，按需增删。命中 FULL 的是旗舰/战略产品线（含系统/OS 发布会和
# 苹果全部活动），命中 LITE 的是副线/中端/半代迭代；都未命中时退回按内容信息量估档。
FULL_LINE_PATTERN = re.compile(
    r"(mate|pura\s|pura\d|华为\s?p\d|x\s?fold|x\s?flip|find\s?[xn]|vivo\s?x\d|mix|magic\s?[v\d]"
    r"|iphone|galaxy\s?[sz]\d|小米\s?\d{2}|xiaomi\s?\d{2}|一加\s?\d{2}|pocket|mate\s?xt|折叠|ultra"
    r"|coloros|harmonyos|magicos|originos|hyperos|鸿蒙|系统|开发者大会|\bodc\b|\bmdc\b"
    r"|apple|苹果|wwdc|ipad|\bmac\b|macbook|vision\s?pro|airpods)",
    re.IGNORECASE,
)
LITE_LINE_PATTERN = re.compile(
    r"(reno|nova|civi|vivo\s?s\d|vivo\s?y\d|oppo\s?a\d|redmi|红米|iqoo|neo\d?\b|真我|realme"
    r"|荣耀\s?\d{2,3}|畅享|麦芒|note\s?\d|gt\d?\b|平板|pad|watch|手表)",
    re.IGNORECASE,
)
NANO_HINT_PATTERN = re.compile(r"(配件|耳机|手环|音箱|充电|global|海外|国际版)", re.IGNORECASE)

# 转写有效字数低于该值时，无论产品线多重要都按 nano 处理（如海外发布只有短素材）。
NANO_CONTENT_FLOOR = 3000

# 纯软件/系统发布会判定：名称命中以下关键词，且转写中价格话术极少时，
# 简报去掉价格、参数表模块。
SOFTWARE_EVENT_PATTERN = re.compile(
    r"(wwdc|开发者大会|\bodc\b|\bmdc\b|coloros|harmonyos|magicos|originos|hyperos|鸿蒙|系统发布|os\s?\d+)",
    re.IGNORECASE,
)
PRICE_MENTION_PATTERN = re.compile(
    r"(\d[\d,]{2,}\s*元|售价|起售|定价|首销|首发价|国补|价格为|\$\s?\d{2,}|￥\s?\d)",
)


def assess_information_volume(
    result_dir: Path,
    source: str,
    transcript_meta: dict[str, Any],
    display_name: str = "",
    tier_override: str = "",
    format_override: str = "",
    domain_id: str = "smartphone",
) -> dict[str, Any]:
    tier_override = (tier_override or "").strip().lower()
    format_override = (format_override or "").strip().lower()
    transcript_text = load_transcript_text(result_dir, transcript_meta)
    transcript_chars = effective_transcript_chars(transcript_text)
    if domain_id != "smartphone":
        event_format, format_reason = "domain", f"按 {domain_id} 领域模板输出"
    elif format_override in {"hardware", "software"}:
        event_format, format_reason = format_override, "用户手动指定发布形态"
    else:
        event_format, format_reason = detect_event_format(display_name, transcript_text)
    duration_min = load_duration_minutes(result_dir)
    section_count = len(re.findall(r"^\*\*\d+\.", source, flags=re.MULTILINE))

    line_tier = ""
    line_reason = ""
    name = display_name or ""
    if domain_id != "smartphone":
        line_tier, line_reason = "", ""
    elif FULL_LINE_PATTERN.search(name):
        line_tier, line_reason = "full", f"发布会名称命中旗舰/战略线关键词（{FULL_LINE_PATTERN.search(name).group(0)}）"
    elif NANO_HINT_PATTERN.search(name):
        line_tier, line_reason = "nano", f"发布会名称命中配件/海外类关键词（{NANO_HINT_PATTERN.search(name).group(0)}）"
    elif LITE_LINE_PATTERN.search(name):
        line_tier, line_reason = "lite", f"发布会名称命中副线/中端产品线关键词（{LITE_LINE_PATTERN.search(name).group(0)}）"

    if tier_override in {"full", "lite", "nano"}:
        tier = tier_override
        basis = "用户手动指定档位"
    elif line_tier:
        tier = line_tier
        basis = f"产品线重要程度判定：{line_reason}"
    elif transcript_chars >= 12000 or duration_min >= 50 or (transcript_chars >= 8000 and section_count >= 8):
        tier = "full"
        basis = "产品线未识别，按内容信息量估为充足"
    elif transcript_chars >= 4000 or duration_min >= 18 or section_count >= 5:
        tier = "lite"
        basis = "产品线未识别，按内容信息量估为中等"
    else:
        tier = "nano"
        basis = "产品线未识别，按内容信息量估为稀少"

    # 内容量只向下兜底：素材太薄时再重要的产品线也撑不起 full/lite。手动指定档位时不降档。
    if tier != "nano" and transcript_chars < NANO_CONTENT_FLOOR and basis != "用户手动指定档位":
        tier = "nano"
        basis += f"；但有效转写不足 {NANO_CONTENT_FLOOR} 字，降为 nano"

    if tier == "full":
        label = "完整业务报告" if domain_id != "smartphone" else "简报（旗舰/战略发布）"
        target_chars = min(max(transcript_chars // 5, 3500), 9000)
    elif tier == "lite":
        label = "业务信息提要" if domain_id != "smartphone" else "信息提要 lite（副线/中端发布）"
        target_chars = min(max(transcript_chars // 8, 1800), 3500)
    else:
        label = "业务快报" if domain_id != "smartphone" else "nano（配件/海外/信息量稀少）"
        target_chars = 1200

    content_stats = f"有效转写约 {transcript_chars} 字"
    if duration_min:
        content_stats += f"，视频时长约 {duration_min:.0f} 分钟"
    content_stats += f"，自动切分 {section_count} 个内容段落"
    signals = f"{basis}；{content_stats}"
    if event_format == "software":
        signals += f"；发布形态判定为纯软件/系统发布（{format_reason}）"
    return {
        "tier": tier,
        "label": label,
        "target_chars": target_chars,
        "signals": signals,
        "event_format": event_format,
    }


def load_transcript_text(result_dir: Path, transcript_meta: dict[str, Any]) -> str:
    text = load_raw_transcript(result_dir, transcript_meta)
    text = re.sub(r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}", " ", text)
    return re.sub(r"<[^>]+>", " ", text)


def load_raw_transcript(result_dir: Path, transcript_meta: dict[str, Any]) -> str:
    relative = str(transcript_meta.get("path") or "")
    text = ""
    if relative:
        path = result_dir / relative
        if path.exists():
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if path.name.startswith("asr."):
                    text = sanitize_cached_asr(text)
            except OSError:
                text = ""
    return text


def effective_transcript_chars(transcript_text: str) -> int:
    cjk_chars = len(re.findall(r"[一-鿿]", transcript_text))
    latin_words = len(re.findall(r"\b[A-Za-z][A-Za-z'-]{1,}\b", transcript_text))
    return cjk_chars + latin_words * 2


def detect_event_format(display_name: str, transcript_text: str) -> tuple[str, str]:
    name_match = SOFTWARE_EVENT_PATTERN.search(display_name or "")
    price_hits = len(PRICE_MENTION_PATTERN.findall(transcript_text))
    if name_match and price_hits <= 2:
        return "software", f"名称命中系统/开发者大会关键词（{name_match.group(0)}）且转写中几乎没有价格话术（{price_hits} 处）"
    return "hardware", ""


def load_duration_minutes(result_dir: Path) -> float:
    path = result_dir / "manifest.json"
    if not path.exists():
        return 0.0
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0.0
    return float(manifest.get("stats", {}).get("duration_sec") or 0) / 60


def event_format_prompt(volume: dict[str, Any] | None) -> str:
    if not volume or volume.get("event_format") != "software":
        return ""
    return (
        "\n本场判定为纯软件/系统发布会，结构相应调整：\n"
        "- 删除「价格」和「参数表」两个模块，不要输出「无 SKU」「无硬件参数」这类空模块占位。\n"
        "- 概述不写「价格」段，可改为「生态/开放/升级策略」判断段，或直接省略。\n"
        "- 产品概要保留，产品名称写系统/版本名（如 iOS 26、ColorOS 16）。\n"
        "- 如确有付费服务/订阅定价出现，在概述或详情对应章节自然带过，不单独成模块。"
    )


def volume_prompt(volume: dict[str, Any] | None) -> str:
    if not volume:
        return "档位未判定：按基础稿信息量自行把握，原则是宁可保留细节，不要为精炼丢信息。"
    tier = volume.get("tier")
    label = volume.get("label", tier)
    signals = volume.get("signals", "")
    target_chars = int(volume.get("target_chars") or 0)
    format_block = event_format_prompt(volume)
    header = (
        f"系统已按产品线重要程度（优先）和内容信息量（兜底）判定档位（依据：{signals}），"
        f"本次按【{label}】输出，不要自行更改档位；如用户补充要求中明确指定了档位，以用户要求为准：\n"
    )
    if tier == "full":
        return header + (
            f"- 全模块完整输出，全文目标不少于 {target_chars} 字（不含图片链接和飞书标签）。\n"
            "- 发布会详情逐章节展开速记：保留基础稿和原文线索中的产品点、数据、价格话术和现场细节，不得为了精炼把事实细节压缩掉；每个章节至少 3-5 条要点。\n"
            "- 概述各段充分展开，多机型时逐机型写判断。\n"
            "- 如个别位置信息确实不足，保留 [待补] 占位，但不能以信息不足为由整体缩短。"
        ) + format_block
    if tier == "lite":
        return header + (
            f"- 重点写满概述、产品概要、价格、参数表；全文目标 {target_chars} 字左右。\n"
            "- 发布会详情压缩为每章节 1-3 条要点，但必须保留每个章节的 **N. 标题** 行（图片按章节对位，删掉标题行会丢图）。"
        ) + format_block
    return header + (
        f"- 概述给 2-3 个最有信息量的要点即可；全文控制在 {target_chars} 字以内，不要为凑长度铺陈。\n"
        "- 发布会详情每章节一句话速记，但必须保留每个章节的 **N. 标题** 行（图片按章节对位，删掉标题行会丢图）。"
    ) + format_block


def build_prompt(
    display_name: str,
    source: str,
    transcript_meta: dict[str, Any] | None = None,
    user_instruction: str = "",
    volume: dict[str, Any] | None = None,
    transcript_evidence_text: str = "",
    domain: DomainProfile | None = None,
    domain_resolution: dict[str, Any] | None = None,
    task_type: str = "business_review",
    ledger_context: str = "",
) -> str:
    transcript_meta = transcript_meta or {}
    domain = domain or domain_from_manifest({"domain": {"requested": "smartphone", "resolved": "smartphone"}})[0]
    if domain.id != "smartphone" or task_type != "business_review":
        return build_domain_prompt(
            display_name=display_name,
            source=source,
            transcript_meta=transcript_meta,
            user_instruction=user_instruction,
            transcript_evidence_text=transcript_evidence_text,
            domain=domain,
            domain_resolution=domain_resolution or {},
            task_type=task_type,
            ledger_context=ledger_context,
        )
    source_language = transcript_meta.get("language_label") or transcript_meta.get("language") or "未判定"
    language_instruction = language_prompt(transcript_meta)
    volume_instruction = volume_prompt(volume)
    user_instruction = user_instruction.strip()
    user_instruction_block = f"""

用户本次补充要求：
{user_instruction}
""".rstrip() if user_instruction else ""
    evidence_block = f"""

原始转写证据（带时间码，事实依据优先级高于基础稿；基础稿没有提到但这里存在的关键产品点、数字、价格、权益和现场细节必须补回）：

<transcript_evidence>
{transcript_evidence_text}
</transcript_evidence>
""" if transcript_evidence_text else ""
    return f"""
请基于下面这份自动生成的发布会简报基础稿，做一次「MO 可继续编辑」级别的精加工。

发布会：{display_name}
源语言：{source_language}
语言处理：{language_instruction}
信息量档位：{volume_instruction}
{user_instruction_block}

要求：
1. 保持文档结构：概述、产品概要、价格、参数表、发布会详情；若上方信息量档位说明判定本场为纯软件/系统发布，按其要求删除价格、参数表模块。
2. 概述是整篇的灵魂，必须独立成立——读者只看概述就能拿到完整判断：
   - 格式必须干脆：每段 = 一行蓝色加粗标题 + 下挂子弹点列表。标题行顶格写 <text color="blue">**…**</text>，行首不加「-」符号；标题下的信息点每条独立成行、行首用「-」，一行只说一件事，禁止把多个信息点糅成大段落。
   - 段内有从属关系时必须用缩进体现层级：一级信息点行首「-」顶格，二级缩进 2 空格、三级缩进 4 空格（如「时间分配」总点下挂各章节分点、机型总判断下挂各配置分点），不要把有层级的内容拍平成同级列表。
   - 单产品线发布会保持 产品 / 讲述 / 价格 三段；多条产品线同台发布时，可改为每条产品线独立成段，价格、发布会判断殿后。
   - 每段蓝色加粗标题本身就要是浓缩判断（如「三杯并两杯，例行升级感强，电池是最大短板」），不是「产品升级情况」这类分类标签；多机型时每个机型用加粗子标题再展开，让「只读加粗」的读者也能拿到完整结论链。
   - 产品段：本代最核心的技术/设计突破并给真实性判断（确有其事还是宣称存疑）、与上代相比的升级/原地踏步/缩水、这是「大年」还是「维护版」。
   - 讲述段：品牌从什么切入、用哪个卖点作主线、贯穿全场反复出现的口径关键词（尤其连续多年「念经」的概念）、各章节时间分配（能算就量化为时长+占全场百分比）、对比口径实指谁、主讲人表现与站台嘉宾、「发布会讲的」和「产品实际是什么」之间的 gap（混讲）。
   - 价格段：起步价+封顶价、逐 SKU 说清哪些涨哪些平（不要只说起步价）、与竞品和同门产品的卡位、锚点 SKU/版本捆绑/首销优惠/赠品对冲/国补影响、配件定价。
   - 能判断的地方直接写判断，不能确认的地方保留 [待补] 或 <text color="gray">// 待核实</text>。
3. 发布会详情要从机械转写改成速记：跳过铺垫和废话，保留产品点、讲述动作、价格/权益信息；有意思的现场细节值得保留（口误、嘉宾站台、演示翻车、官方玩梗——往往能反映品牌心态）；价格发布尽量单独成节，记录现场价格锚点话术和赠品/搭售信息；IoT 与配件也要覆盖，但篇幅克制。
4. 不要虚构发布会没有出现的参数、价格、竞品关系。数据是脚注，判断才是正文；不写「总体而言亮点颇多」这类空话，不做没有判断的中立信息汇总。
5. 保留已插入的图片 Markdown，不要删除、不要新增、不要移动图片。图片由系统按时间轴另行校准，你只负责文字精加工。
   发布会详情各章节标题必须保持基础稿的 **N. 标题** 行格式（加粗行、编号逐一对应），不要改成 ## / ### 标题层级，不要合并或跳过章节编号——图片按章节编号对位，改格式会丢图。章节标题文字本身可以重新提炼。
6. 如果用户本次补充要求要求展开某个部分，请在不改变整体结构的前提下优先展开对应段落。
7. 信息量大的发布会（多机型/多 SKU），在标题下方补一个「快问快答」速查块：用「什么芯片？」「多少钱？」等 1-3 个最高频问题 + 按机型/SKU 分行的极简答案，让读者 10 秒拿到最常被问的结论；信息量少时省略。
8. 输出完整 Markdown，不要解释你做了什么。
9. 完稿前逐项检查原始转写证据中的金额、SKU、参数数字、产品/功能名；不得因为基础稿遗漏而继续遗漏。无法判断的 ASR 疑似错字要标记待核实，不要擅自修成另一个数字。

基础稿如下：

{source}
{evidence_block}
""".strip()


def build_domain_prompt(
    *,
    display_name: str,
    source: str,
    transcript_meta: dict[str, Any],
    user_instruction: str,
    transcript_evidence_text: str,
    domain: DomainProfile,
    domain_resolution: dict[str, Any],
    task_type: str,
    ledger_context: str,
) -> str:
    task_label = TASK_TYPES.get(task_type, task_type)
    task_guidance = {
        "faithful_transcript": "忠实整理原话，修正标点和高置信度同音错误，不删事实、不增加行业判断；保留时间码。",
        "launch_notes": "按议题生成完整速记，去掉口头重复，但保留事实、数字、限定条件、演示动作和现场细节。",
        "business_review": "在完整事实基础上给出业务判断、竞争含义、可信度和风险；事实、官方宣称、演示和分析判断必须分开。",
        "fact_extract": "以结构化事实、参数、价格、版本和证据索引为核心；不为叙事流畅而省略限定条件。",
    }.get(task_type, "按证据生成完整发布会报告。")
    sections = "、".join(sections_for_task(domain, task_type))
    user_block = f"\n用户补充要求：\n{user_instruction.strip()}\n" if user_instruction.strip() else ""
    ledger_block = f"\n<fact_ledger>\n{ledger_context}\n</fact_ledger>\n" if ledger_context else ""
    evidence_block = f"\n<transcript_evidence>\n{transcript_evidence_text}\n</transcript_evidence>\n"
    return f"""
请将基础稿加工为【{domain.name} · {task_label}】成稿。

发布会：{display_name}
领域识别：{domain.name}，置信度 {float(domain_resolution.get('confidence') or 0):.0%}，依据：{'、'.join(domain_resolution.get('signals') or []) or '人工指定/通用回退'}
领域要求：{domain.guidance}
任务要求：{task_guidance}
源语言处理：{language_prompt(transcript_meta)}
{user_block}

工程约束：
1. 全文必须遵循“总—分”结构。推荐章节为：{sections}。正文开头必须先写“## 发布会概述”，之后再进入行业分项；不得省略、后置或用“模型定位”等分项代替总述。
   - 概述至少包含两项独立判断：“产品总结”回答发布了什么、核心升级/价值、目标用户与可信度；“传播总结”回答整场如何讲、主线与节奏、重点是否讲清、Demo 和传播得失。
   - 概述要让读者不看后文也能理解整场发布会，不得只是后文章节目录或事实罗列。“分”的章节可按行业调整，也可删除确无内容的空章节，但不得把其他行业模板强套进来。
2. 每个数字、价格、版本、能力边界和比较结论都要能回溯到事实账本或原始转写；疑似 ASR 错误标记“待核实”。
3. 明确区分“发布会事实”“官方宣称”“现场 Demo”“分析判断”，不得把宣称写成已验证事实。
4. 保留所有图片 Markdown，不新增、不移动。发布会详情的 **N. 标题** 编号和格式必须逐一保留；标题文字必须改写成该时间段的明确议题或事件，不得沿用关键词堆砌、单个缩写或“AI / SE / Pro”一类不可理解标题。
5. 不虚构竞品关系、参数、价格、测试条件、交付或量产状态。
6. 输出完整中文 Markdown，不解释处理过程。

基础稿：
<brief_base>
{source}
</brief_base>
{ledger_block}
原始转写证据：
{evidence_block}
""".strip()


def language_prompt(transcript_meta: dict[str, Any]) -> str:
    language = str(transcript_meta.get("language") or "unknown")
    if language == "en":
        return "源转写主要为英文。请用中文输出简报；理解英文发布内容后做市场语言转译；Apple Intelligence、ProMotion、Ceramic Shield、A 系列芯片、产品名、功能名等专有名词保留英文原文，必要时用中文解释其市场含义。"
    if language == "mixed":
        return "源转写为中英混合。请统一用中文输出，保留英文产品名、功能名、技术名和 slogan，不要把专有名词翻成生硬中文。"
    if language == "zh":
        return "源转写主要为中文。按中文发布会处理，必要时保留英文产品名、技术名和官方 slogan。"
    return "源语言未可靠判定。默认用中文输出，遇到英文专有名词保留原文。"


def restore_image_blocks(source: str, refined: str) -> str:
    source_groups = extract_section_images(source)
    if not source_groups:
        return refined

    lines = refined.splitlines()
    section_ranges: list[tuple[int, int, int]] = []
    for index, line in enumerate(lines):
        section = section_number(line)
        if section is not None:
            if section_ranges:
                prev_section, start, _ = section_ranges[-1]
                section_ranges[-1] = (prev_section, start, index)
            section_ranges.append((section, index, len(lines)))

    if not section_ranges:
        return refined

    new_lines: list[str] = []
    cursor = 0
    for section, start, end in section_ranges:
        new_lines.extend(lines[cursor:start])
        block = lines[start:end]
        images = source_groups.get(section, [])
        new_lines.extend(rebuild_section_block(block, images))
        cursor = end
    new_lines.extend(lines[cursor:])
    return "\n".join(new_lines).rstrip() + "\n"


def extract_section_images(markdown: str) -> dict[int, list[str]]:
    groups: dict[int, list[str]] = {}
    current: int | None = None
    for line in markdown.splitlines():
        section = section_number(line)
        if section is not None:
            current = section
            groups.setdefault(current, [])
            continue
        if current is not None and line.strip().startswith("![]("):
            groups.setdefault(current, []).append(line.strip())
    return groups


def section_number(line: str) -> int | None:
    match = re.match(r"\*\*(\d+)\.", line.strip())
    if not match:
        return None
    return int(match.group(1))


def rebuild_section_block(block: list[str], images: list[str]) -> list[str]:
    without_images = [line for line in block if not line.strip().startswith("![](")]
    if not images:
        return without_images

    indented_images = [f"  {image}" for image in images]
    for index, line in enumerate(without_images):
        if "关于图片" in line:
            return without_images[: index + 1] + indented_images + without_images[index + 1 :]
    return without_images + ["- **关于图片**："] + indented_images
