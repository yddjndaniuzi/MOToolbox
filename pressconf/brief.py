from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pressconf.keyframes import format_timestamp
from pressconf.domains import TASK_TYPES, domain_from_manifest, sections_for_task, task_type_from_manifest


@dataclass
class Cue:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str


def parse_transcript(content: str) -> list[Cue]:
    content = content.strip()
    if not content:
        return []
    if "-->" in content:
        return parse_srt(content)
    return [Cue(start=0.0, end=0.0, text=normalize_text(content))]


def parse_srt(content: str) -> list[Cue]:
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").replace("\r", "\n").strip())
    cues: list[Cue] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        time_index = next((index for index, line in enumerate(lines) if "-->" in line), -1)
        if time_index < 0:
            continue
        start_raw, end_raw = [item.strip() for item in lines[time_index].split("-->", 1)]
        text = normalize_text(" ".join(lines[time_index + 1 :]))
        if not text:
            continue
        cues.append(Cue(start=parse_srt_time(start_raw), end=parse_srt_time(end_raw), text=text))
    return cues


def parse_srt_time(value: str) -> float:
    match = re.search(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})", value)
    if not match:
        return 0.0
    hours, minutes, seconds, millis = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis.ljust(3, "0")) / 1000


def normalize_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def segment_cues(cues: list[Cue], window_seconds: int = 180) -> list[Segment]:
    if not cues:
        return []
    if len(cues) == 1 and cues[0].end == 0:
        return [Segment(start=0.0, end=0.0, text=cues[0].text)]

    segments: list[Segment] = []
    current_start = cues[0].start
    current_end = cues[0].end
    texts: list[str] = []

    for cue in cues:
        gap = cue.start - current_end
        over_window = cue.start - current_start >= window_seconds
        if texts and (gap > 25 or over_window):
            segments.append(Segment(start=current_start, end=current_end, text=normalize_text(" ".join(texts))))
            current_start = cue.start
            texts = []
        current_end = cue.end
        texts.append(cue.text)

    if texts:
        segments.append(Segment(start=current_start, end=current_end, text=normalize_text(" ".join(texts))))
    return segments


def compose_brief_base(manifest: dict[str, Any], transcript: str, slug: str) -> str:
    event_name = manifest.get("event_name") or slug
    transcript_meta = manifest.get("transcript_meta") or {}
    source_language = str(transcript_meta.get("language") or "unknown")
    source_language_label = str(transcript_meta.get("language_label") or "语言未判定")
    language_note = str(transcript_meta.get("language_note") or "")
    cues = parse_transcript(transcript)
    segments = segment_cues(cues)
    keyframes = manifest.get("keyframes", [])
    total_seconds = float(manifest.get("stats", {}).get("duration_sec") or 0)
    domain, resolution = domain_from_manifest(manifest, transcript)
    task_type = task_type_from_manifest(manifest)
    if domain.id != "smartphone" or task_type != "business_review":
        return compose_domain_base(
            manifest=manifest,
            slug=slug,
            domain=domain,
            resolution=resolution,
            task_type=task_type,
            segments=segments,
            keyframes=keyframes,
            total_seconds=total_seconds,
            source_language_label=source_language_label,
            language_note=language_note,
        )

    lines = [
        f"# {event_name} 发布会简报【AI 基础稿】",
        "",
        "<text color=\"gray\">// 本稿由 ASR/SRT 与关键帧自动对齐生成，是简报写作底稿；概述、价格、参数表需要结合官网/产品信息二次核实。</text>",
        f"<text color=\"gray\">// 转写语言：{source_language_label}；输出语言：中文。{language_note}</text>",
        "<text color=\"gray\">// 信息量大的发布会可在此补「快问快答」速查：1-3 个最高频问题（什么芯片？多少钱？）+ 按机型/SKU 分行的极简答案。</text>",
        "",
        "## 概述：",
        "",
        "<text color=\"blue\">**产品：[待补：一句话核心产品判断]**</text>",
        "- [待补：本代最核心技术/设计突破，以及是真升级还是官方宣称]",
        "- [待补：对比上代/同代竞品的关键变化]",
        "",
        f"<text color=\"blue\">**讲述：[待补：一句话发布会叙事策略判断]**</text>",
        f"- 本场自动切分为 {len(segments)} 个内容段落，视频总时长约 {format_timestamp(total_seconds) if total_seconds else '待确认'}<text color=\"gray\"> // 章节时长由转写时间轴粗分，需人工合并/校正</text>",
        "- [待补：品牌从什么切入、用哪个卖点作主线、哪里讲得聪明/哪里混讲]",
        "",
        "<text color=\"blue\">**价格：[待补：一句话定价判断]**</text>",
        "- [待补：起步价/封顶价、上代涨跌、竞品对位、首销权益]",
        "",
        "<text color=\"gray\">// 本篇由 AI 基于转写稿生成，关键判断供参考，建议对照原始视频核实</text>",
        "",
        "## 产品概要：",
        "**产品名称：[待补：品牌 + 完整型号]**  ",
        "**发布时间：[待补：YYYY.MM.DD]**  ",
        "**产品 Slogan：[待补：官方宣传语原文]**",
        "",
        "## 价格：",
        "**[待补：产品名]**",
        "",
        "<text color=\"gray\">// 待补 SKU 矩阵、上代对比、首销权益/国补信息。</text>",
        "",
        "## 参数表：",
        "",
        "<text color=\"gray\">// 待根据官网参数页/发布会参数页/官方海报补全；不要把宣传话术直接写进参数表。</text>",
        "",
        "## 发布会详情",
        "",
    ]

    if not segments:
        lines.extend(["<text color=\"gray\">// 暂未识别到可用转写内容</text>", ""])
        return "\n".join(lines)

    for index, segment in enumerate(segments, start=1):
        title = segment_title(index, segment)
        summary = rough_summary(segment.text)
        excerpt = short_excerpt(segment.text)
        matched_frames = match_frames_for_segment(keyframes, segment, max_frames=3)

        duration = segment.end - segment.start if segment.end > segment.start else 0
        share = f"，约占全场 {duration / total_seconds:.0%}" if duration and total_seconds else ""
        lines.extend(
            [
                f"**{title}**",
                f"- {summary}",
                f"- <text color=\"gray\">// 时间段：{segment_time_range(segment)}{share}；原文线索：{excerpt}</text>",
            ]
        )
        for frame in matched_frames:
            lines.append(f"  ![]({frame.get('path')})")
        lines.append("")

    return "\n".join(lines)


def compose_domain_base(
    *,
    manifest: dict[str, Any],
    slug: str,
    domain: Any,
    resolution: dict[str, Any],
    task_type: str,
    segments: list[Segment],
    keyframes: list[dict[str, Any]],
    total_seconds: float,
    source_language_label: str,
    language_note: str,
) -> str:
    event_name = manifest.get("event_name") or slug
    task_label = TASK_TYPES.get(task_type, task_type)
    lines = [
        f"# {event_name}【{domain.name} · {task_label}基础稿】",
        "",
        f'<text color="gray">// 领域：{domain.name}；识别置信度：{float(resolution.get("confidence") or 0):.0%}；依据：{"、".join(resolution.get("signals") or []) or "通用回退"}</text>',
        f'<text color="gray">// 转写语言：{source_language_label}；输出语言：中文。{language_note}</text>',
        f'<text color="gray">// 自动切分 {len(segments)} 段；视频时长约 {format_timestamp(total_seconds) if total_seconds else "待确认"}；所有判断需回溯时间码证据。</text>',
        "",
    ]
    sections = sections_for_task(domain, task_type)
    detail_heading = "逐字稿正文" if task_type == "faithful_transcript" else "发布会详情"
    for section in sections:
        if section == detail_heading:
            continue
        lines.extend([f"## {section}", ""])
        if section == "发布会概述":
            lines.extend([
                '<text color="blue">**产品总结：[待补：本场发布了什么、核心升级与业务价值]**</text>',
                "- [待补：产品/能力主线、关键变化、目标用户及可信度判断]",
                "",
                '<text color="blue">**传播总结：[待补：整场发布会如何讲、重点是否讲清]**</text>',
                "- [待补：叙事主线、篇幅节奏、核心口径、Demo 与传播得失]",
                "",
            ])
        else:
            lines.extend([f"- [待补：按{domain.name}领域要求整理{section}]", ""])
    lines.extend([f"## {detail_heading}", ""])

    if not segments:
        lines.append('<text color="gray">// 暂未识别到可用转写内容</text>')
        return "\n".join(lines)

    excerpt_limit = 100_000 if task_type == "faithful_transcript" else (1600 if task_type == "launch_notes" else 1000)
    for index, segment in enumerate(segments, start=1):
        matched_frames = match_frames_for_segment(keyframes, segment, max_frames=3)
        lines.extend([
            f"**{segment_title(index, segment, domain.id)}**",
            f"- <text color=\"gray\">// 时间段：{segment_time_range(segment)}</text>",
            f"- {short_excerpt(segment.text, excerpt_limit)}",
        ])
        for frame in matched_frames:
            lines.append(f"  ![]({frame.get('path')})")
        lines.append("")
    return "\n".join(lines)


def segment_title(index: int, segment: Segment, domain_id: str = "smartphone") -> str:
    if segment.end > 0:
        keywords = extract_topic_labels(segment.text, domain_id)
        label = " / ".join(keywords[:2]) if keywords else f"内容段（{format_timestamp(segment.start)}）"
        return f"{index}. {label}"
    return f"{index}. 全文转写"


def segment_time_range(segment: Segment) -> str:
    if segment.end > 0:
        return f"{format_timestamp(segment.start)} - {format_timestamp(segment.end)}"
    return "全文"


def rough_summary(text: str, limit: int = 90) -> str:
    sentences = [item.strip() for item in re.split(r"[。！？!?]", text) if item.strip()]
    if not sentences:
        return truncate(text, limit)
    useful = [sentence for sentence in sentences if len(sentence) >= 8]
    summary = "；".join(useful[:2] or sentences[:2])
    return truncate(summary, limit)


def short_excerpt(text: str, limit: int = 720) -> str:
    return truncate(text, limit)


def truncate(text: str, limit: int) -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def extract_keywords(text: str) -> list[str]:
    chinese = re.findall(
        r"(影像|拍照|视频|人像|长焦|屏幕|性能|芯片|散热|续航|电池|充电|设计|轻薄|系统|游戏|价格|配色|发布|新品|折叠|旗舰|标准版)",
        text,
        flags=re.IGNORECASE,
    )
    # ASCII product tiers must be standalone tokens. The previous unbounded
    # regex matched "se" in ordinary English words such as "use" and
    # "experience", producing headings like "AI / SE / Pro".
    ascii_terms = re.findall(r"(?<![A-Za-z0-9])(AI|Pro|Ultra|Mini|SE)(?![A-Za-z0-9])", text, re.IGNORECASE)
    candidates = chinese + ascii_terms
    seen: set[str] = set()
    keywords: list[str] = []
    for candidate in candidates:
        normalized = candidate.upper() if candidate.lower() == "ai" else candidate
        if normalized not in seen:
            seen.add(normalized)
            keywords.append(normalized)
    return keywords


DOMAIN_TOPIC_PATTERNS: dict[str, tuple[tuple[str, str], ...]] = {
    "foundation_model": (
        (r"chat\s?gpt work", "ChatGPT Work"),
        (r"desktop app|local files|computer use", "Desktop App 与 Computer Use"),
        (r"hosted sites|\bsites\b|interactive (?:website|visualization)", "Sites 与交互式生成"),
        (r"pre-?training|post-?training|reinforcement learning", "模型训练与能力演进"),
        (r"benchmark|state.of.the.art|frontier eval|terminal bench", "Benchmark 与效率"),
        (r"safety|red team|cybersecurity|vulnerabilit|project daybreak", "安全与网络安全"),
        (r"codex|researcher|experiment", "Codex 与研发提效"),
        (r"translat|japan|farm|farmer", "实时翻译与农业案例"),
    ),
    "automotive": (
        (r"智能驾驶|智驾|NOA|辅助驾驶", "智能驾驶"),
        (r"座舱|车机|智能座舱", "智能座舱"),
        (r"续航|电池|补能|充电", "续航与补能"),
        (r"价格|售价|预售", "价格与上市"),
    ),
    "robotics": (
        (r"本体|自由度|关节|执行器", "本体与运动能力"),
        (r"感知|视觉|传感器", "感知系统"),
        (r"演示|demo|任务", "现场演示"),
        (r"量产|交付|价格", "量产与商业化"),
    ),
    "semiconductor": (
        (r"架构|微架构|制程|工艺节点|晶体管|chiplet", "架构、制程与晶体管"),
        (r"算力|flops|tops|benchmark|基准测试|吞吐", "计算性能与测试条件"),
        (r"功耗|能效|tdp|tbp|每瓦", "功耗与能效"),
        (r"hbm|gddr|缓存|内存|带宽|nvlink|pcie|cxl|互连", "存储、带宽与互连"),
        (r"封装|cowos|soic|foveros|emib|晶圆|良率|代工", "封装、制造与供应链"),
        (r"cuda|rocm|编译器|软件栈|开发者", "软件栈与生态"),
        (r"量产|出货|送样|客户|上市|价格", "量产、客户与商业化"),
    ),
}


def extract_topic_labels(text: str, domain_id: str) -> list[str]:
    patterns = DOMAIN_TOPIC_PATTERNS.get(domain_id, ())
    labels = [label for pattern, label in patterns if re.search(pattern, text, re.IGNORECASE)]
    if labels:
        return labels
    return extract_keywords(text)


def match_frames_for_segment(keyframes: list[dict[str, Any]], segment: Segment, max_frames: int = 3) -> list[dict[str, Any]]:
    if not keyframes:
        return []
    if segment.end <= 0:
        return keyframes[:max_frames]

    in_window = [
        frame
        for frame in keyframes
        if segment.start <= float(frame.get("timestamp_sec", 0)) <= segment.end
    ]
    if not in_window:
        midpoint = (segment.start + segment.end) / 2
        return sorted(keyframes, key=lambda frame: abs(float(frame.get("timestamp_sec", 0)) - midpoint))[:1]

    return pick_representative_frames(in_window, segment, max_frames)


def pick_representative_frames(frames: list[dict[str, Any]], segment: Segment, max_frames: int) -> list[dict[str, Any]]:
    if len(frames) <= max_frames:
        return frames

    duration = max(segment.end - segment.start, 1)
    targets = [0.16, 0.5, 0.82][:max_frames]
    selected: list[dict[str, Any]] = []

    for target in targets:
        ranked = sorted(
            frames,
            key=lambda frame: representative_score(frame, segment.start, duration, target, selected),
        )
        for frame in ranked:
            if frame not in selected:
                selected.append(frame)
                break

    return sorted(selected, key=lambda frame: float(frame.get("timestamp_sec", 0)))


def representative_score(
    frame: dict[str, Any],
    segment_start: float,
    duration: float,
    target: float,
    selected: list[dict[str, Any]],
) -> float:
    timestamp = float(frame.get("timestamp_sec", 0))
    position = min(max((timestamp - segment_start) / duration, 0), 1)
    reason_bonus = {
        "strong-change": -0.10,
        "visual-change": -0.07,
        "time-coverage": -0.03,
        "first": 0.04,
    }.get(str(frame.get("reason", "")), 0)
    diff_bonus = -min(float(frame.get("diff_score", 0)), 0.2) * 0.25
    duplicate_penalty = 0.0
    for item in selected:
        if abs(timestamp - float(item.get("timestamp_sec", 0))) < duration * 0.18:
            duplicate_penalty += 0.35
    return abs(position - target) + reason_bonus + diff_bonus + duplicate_penalty


def spread_frames(frames: list[dict[str, Any]], max_frames: int) -> list[dict[str, Any]]:
    if len(frames) <= max_frames:
        return frames
    frames = sorted(frames, key=lambda frame: float(frame.get("timestamp_sec", 0)))
    if max_frames == 1:
        return [frames[len(frames) // 2]]
    step = (len(frames) - 1) / (max_frames - 1)
    return [frames[round(index * step)] for index in range(max_frames)]


def write_brief_base(result_dir: Path, manifest: dict[str, Any], transcript: str, slug: str) -> Path:
    content = compose_brief_base(manifest, transcript, slug)
    output_path = result_dir / "brief_base.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path
