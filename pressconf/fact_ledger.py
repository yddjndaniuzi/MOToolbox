from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pressconf.brief import Cue, parse_transcript
from pressconf.domains import DomainProfile, domain_from_manifest


VALUE_PATTERN = re.compile(
    r"(?:[￥$¥]\s*)?(\d+(?:[.,]\d+)*)\s*"
    r"(万元|亿元|亿个?晶体管|百万个?晶体管|元|万|亿|%|TB/s|GB/s|Gb/s|GT/s|MT/s|TFLOPS|PFLOPS|EFLOPS|GFLOPS|TOPS|GHz|MHz|GB|TB|MB|kW|kWh|mAh|Hz|km|公里|毫米|mm²|mm2|mm|nm|英寸|秒|ms|倍|自由度|核|cores?|tokens?|tok/s|V|W)?",
    re.IGNORECASE,
)

TYPE_RULES: dict[str, tuple[str, ...]] = {
    "price": ("价格", "售价", "起售", "预售", "元", "￥", "权益"),
    "range": ("续航", "cltc", "wltc", "公里", "km"),
    "charging": ("充电", "补能", "快充", "800v", "kw"),
    "power": ("功率", "马力", "扭矩", "kw"),
    "dimensions": ("尺寸", "轴距", "车长", "车宽", "车高", "mm", "毫米"),
    "adas": ("智驾", "noa", "激光雷达", "tops", "辅助驾驶"),
    "benchmark": ("benchmark", "评测", "得分", "准确率", "榜单"),
    "context_window": ("上下文", "context", "token"),
    "latency": ("延迟", "latency", "首token", "秒", "ms"),
    "throughput": ("吞吐", "tok/s", "每秒token"),
    "api_price": ("api", "输入价格", "输出价格", "百万token"),
    "dof": ("自由度", "dof", "关节"),
    "payload": ("负载", "载荷", "公斤", "kg"),
    "runtime": ("续航", "运行时间", "小时", "分钟"),
    "precision": ("精度", "重复定位"),
    "production": ("产能", "量产", "台"),
    "delivery": ("交付", "上市", "开售", "发货"),
    "battery": ("电池", "mah", "kwh", "容量"),
    "imaging": ("影像", "相机", "像素", "光圈", "长焦"),
    "display": ("屏幕", "亮度", "刷新率", "hz", "英寸"),
    "process_node": ("制程", "工艺", "节点", "nm", "finfet", "gaa"),
    "transistor_count": ("晶体管",),
    "die_size": ("die", "裸片", "芯片面积", "mm²", "mm2"),
    "core_count": ("核心", "核", "core", "cu", "sm"),
    "frequency": ("频率", "主频", "加速频率", "boost", "ghz", "mhz"),
    "compute": ("算力", "flops", "tops", "fp64", "fp32", "tf32", "fp16", "bf16", "fp8", "fp4", "int8", "int4"),
    "performance_gain": ("性能提升", "性能提高", "单线程性能", "多线程性能", "ipc提升", "吞吐提升"),
    "memory_capacity": ("显存", "内存", "缓存", "hbm", "gddr", "容量"),
    "memory_speed": ("显存速率", "内存速率", "mt/s", "gt/s"),
    "bandwidth": ("带宽", "gb/s", "tb/s", "gb/s"),
    "tdp": ("tdp", "tbp", "功耗", "热设计功耗", "整卡功耗"),
    "power_efficiency": ("能效", "每瓦", "性能功耗比"),
    "yield": ("良率",),
}


def classify_fact(text: str, unit: str, profile: DomainProfile) -> str:
    haystack = f"{text} {unit}".lower()
    candidates = set(profile.fact_types)
    normalized_unit = unit.lower()
    if normalized_unit in {"元", "万元", "亿元"}:
        return "price" if "price" in candidates else "number"
    if normalized_unit in {"km", "公里"} and "range" in candidates:
        return "range"
    if normalized_unit in {"mah", "kwh"} and "battery" in candidates:
        return "battery"
    if normalized_unit in {"v", "kw", "w"} and "charging" in candidates and any(
        keyword in haystack for keyword in ("充电", "补能", "平台", "800v")
    ):
        return "charging"
    if normalized_unit in {"tokens", "token", "tok/s"}:
        if "throughput" in candidates and normalized_unit == "tok/s":
            return "throughput"
        if "context_window" in candidates:
            return "context_window"
    if normalized_unit == "自由度" and "dof" in candidates:
        return "dof"
    if normalized_unit == "nm" and "process_node" in candidates:
        return "process_node"
    if "晶体管" in normalized_unit and "transistor_count" in candidates:
        return "transistor_count"
    if normalized_unit in {"mm²", "mm2"} and "die_size" in candidates:
        return "die_size"
    if normalized_unit in {"核", "core", "cores"} and "core_count" in candidates:
        return "core_count"
    if normalized_unit in {"ghz", "mhz"} and "frequency" in candidates:
        return "frequency"
    if normalized_unit in {"tflops", "pflops", "eflops", "gflops", "tops"} and "compute" in candidates:
        return "compute"
    if normalized_unit in {"tb/s", "gb/s", "gb/s"} and "bandwidth" in candidates:
        return "bandwidth"
    if normalized_unit in {"mt/s", "gt/s"} and "memory_speed" in candidates:
        return "memory_speed"
    if normalized_unit in {"gb", "tb", "mb"} and "memory_capacity" in candidates and any(
        keyword in haystack for keyword in ("显存", "内存", "缓存", "hbm", "gddr", "容量")
    ):
        return "memory_capacity"
    if normalized_unit == "%" and "power_efficiency" in candidates and any(
        keyword in haystack for keyword in ("能效", "每瓦", "性能功耗比")
    ):
        return "power_efficiency"
    if normalized_unit == "%" and "performance_gain" in candidates and any(
        keyword in haystack for keyword in ("性能提升", "性能提高", "ipc提升", "吞吐提升")
    ):
        return "performance_gain"
    if normalized_unit == "w" and "tdp" in candidates and any(
        keyword in haystack for keyword in ("tdp", "tbp", "功耗", "整卡", "热设计")
    ):
        return "tdp"
    for fact_type, keywords in TYPE_RULES.items():
        if fact_type in candidates and any(keyword.lower() in haystack for keyword in keywords):
            return fact_type
    if any(symbol in haystack for symbol in ("元", "￥", "¥", "$")):
        return "price" if "price" in candidates else "number"
    return "number"


def cue_confidence(cue: Cue, quality: dict[str, Any]) -> float:
    suspicious = quality.get("suspicious_segments") or []
    for item in suspicious:
        start = float(item.get("start") or 0)
        end = float(item.get("end") or 0)
        if cue.start <= end and cue.end >= start:
            return 0.45
    return 0.8 if quality else 0.65


def build_fact_ledger(
    *,
    result_dir: Path,
    manifest: dict[str, Any],
    transcript: str,
    transcript_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile, resolution = domain_from_manifest(manifest, transcript)
    quality = (transcript_meta or {}).get("quality") or {}
    facts: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float]] = set()
    seen_terms: set[str] = set()
    for cue in parse_transcript(transcript):
        for match in VALUE_PATTERN.finditer(cue.text):
            value = match.group(1).replace(",", "")
            unit = match.group(2) or ""
            key = (value, unit.lower(), round(cue.start, 1))
            if key in seen:
                continue
            seen.add(key)
            facts.append({
                "id": f"fact-{len(facts) + 1:05d}",
                "domain": profile.id,
                "type": classify_fact(cue.text, unit, profile),
                "value": value,
                "unit": unit,
                "timestamp": [round(cue.start, 3), round(cue.end, 3)],
                "source_text": cue.text,
                "confidence": cue_confidence(cue, quality),
                "status": "extracted",
            })
        lower_text = cue.text.lower()
        for term in profile.hotwords:
            normalized = term.strip()
            if not normalized or normalized.lower() in seen_terms or normalized.lower() not in lower_text:
                continue
            seen_terms.add(normalized.lower())
            facts.append({
                "id": f"fact-{len(facts) + 1:05d}",
                "domain": profile.id,
                "type": "entity",
                "value": normalized,
                "unit": "",
                "timestamp": [round(cue.start, 3), round(cue.end, 3)],
                "source_text": cue.text,
                "confidence": cue_confidence(cue, quality),
                "status": "extracted",
            })
    ledger = {
        "schema_version": 1,
        "domain": resolution,
        "fact_types": list(profile.fact_types),
        "fact_count": len(facts),
        "facts": facts,
    }
    (result_dir / "fact_ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ledger
