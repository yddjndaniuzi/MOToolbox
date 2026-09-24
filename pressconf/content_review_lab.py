from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from pressconf.config_store import resolve_model
from pressconf.derivatives import call_writing_model_streaming
from pressconf.media_feedback import sanitize_portable_markdown, split_transcript_chunks


CONTENT_REVIEW_SYSTEM = (
    "你是消费电子品牌市场团队的媒体内容审核专家。你审的是媒体/KOL发来待确认的稿件、脚本或逐字稿，"
    "目标是帮助品牌团队识别事实、口径、负面表达、歧义、舆情和传播风险，并给出具体修改建议。"
    "你不能把媒体合理观点一律改成品牌广告口吻；必须区分媒体可保留的独立评价、需要确认的事实，"
    "以及确实建议修改或升级人工复核的风险表达。所有判断必须来自输入内容和用户给出的审核参考，"
    "不得编造产品参数、价格、时间、媒体身份、测试结果或官方口径。"
)

INTERNAL_CODENAME_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:[PQON]\d{1,4})(?![A-Za-z0-9])")


def generate_content_review_lab(
    *,
    base_dir: Path,
    product_name: str,
    media_name: str,
    content_title: str,
    source_label: str,
    source_text: str,
    review_reference: str = "",
    additional_instruction: str = "",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    model_config = resolve_model(
        base_dir,
        "writing",
        "heavy" if len(source_text) > 24_000 else "standard",
    )
    api_key = str(model_config["api_key"]).strip()
    if not api_key:
        raise RuntimeError(f"没有配置 {model_config.get('name', model_config.get('model', '模型'))} 的 API Key。")

    chunks = split_transcript_chunks(source_text, 24000)
    if not chunks:
        raise RuntimeError("没有读取到可审核的媒体正文。")

    evidence_notes: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        stage = f"风险初筛 {index}/{len(chunks)}"
        note = call_writing_model_streaming(
            api_key=api_key,
            base_url=str(model_config["base_url"]).rstrip("/"),
            model=str(model_config["model"]).strip(),
            provider=str(model_config["provider"]).strip(),
            prompt=build_chunk_prompt(
                product_name=product_name,
                media_name=media_name,
                content_title=content_title,
                source_label=source_label,
                chunk_index=index,
                chunk_count=len(chunks),
                chunk=chunk,
                review_reference=review_reference,
                additional_instruction=additional_instruction,
            ),
            max_tokens=7000,
            stage=stage,
            system_prompt=CONTENT_REVIEW_SYSTEM,
            progress_callback=progress_callback,
            fallback_models=model_config.get("fallbacks"),
        )
        evidence_notes.append(f"## 片段 {index}/{len(chunks)}\n{note.strip()}")

    result = call_writing_model_streaming(
        api_key=api_key,
        base_url=str(model_config["base_url"]).rstrip("/"),
        model=str(model_config["model"]).strip(),
        provider=str(model_config["provider"]).strip(),
        prompt=build_final_prompt(
            product_name=product_name,
            media_name=media_name,
            content_title=content_title,
            source_label=source_label,
            review_reference=review_reference,
            additional_instruction=additional_instruction,
            evidence_notes="\n\n".join(evidence_notes),
        ),
        max_tokens=18000,
        stage="媒体内容审核报告",
        system_prompt=CONTENT_REVIEW_SYSTEM,
        progress_callback=progress_callback,
        fallback_models=model_config.get("fallbacks"),
    )
    result = sanitize_portable_markdown(result)
    warnings = sorted(set(INTERNAL_CODENAME_PATTERN.findall(result)))
    if warnings:
        result += "\n\n> 自检提醒：输出中仍包含疑似内部代号：" + "、".join(warnings)
    return result.strip(), {
        "model": str(model_config["model"]).strip(),
        "provider": str(model_config["provider"]).strip(),
        "chunk_count": len(chunks),
        "warnings": warnings,
        "product_name": product_name,
        "media_name": media_name,
        "title": content_title,
        "source_label": source_label,
    }


def build_chunk_prompt(
    *,
    product_name: str,
    media_name: str,
    content_title: str,
    source_label: str,
    chunk_index: int,
    chunk_count: int,
    chunk: str,
    review_reference: str,
    additional_instruction: str = "",
) -> str:
    reference_block = build_reference_block(review_reference)
    instruction_block = f"\n【用户补充要求】\n{additional_instruction.strip()}\n" if additional_instruction.strip() else ""
    return f"""请初筛下面这段待审核媒体内容，只做证据提取，不输出最终审核报告。

产品/项目：{product_name or "未填写"}
媒体/账号：{media_name or "未填写"}
内容标题：{content_title or "未填写"}
内容来源：{source_label or "未填写"}
片段：{chunk_index}/{chunk_count}
{reference_block}
{instruction_block}

初筛维度：
1. 可能导致负面传播或误读的表达：翻车、拉踩、极端否定、带节奏标题、结论先行但证据不足。
2. 不明确或待确认的信息：参数、价格、SKU、解禁/上市时间、OTA、功能边界、实验条件、数据来源。
3. 与审核参考明显冲突的表述；如没有参考，只写“待确认”，不要自行判错。
4. 竞品攻击、绝对化、保密/内部代号、样机/试产/系统版本边界不清。
5. 值得保留的媒体独立观点或亮点表达，避免后续误删。

输出格式：
## 片段判断
- 风险密度：高 / 中 / 低
- 一句话说明：

## 风险证据
| 原文短摘 | 风险类型 | 风险说明 | 建议动作 |
| --- | --- | --- | --- |

## 待确认事实
| 原文短摘 | 需要确认什么 | 为什么 |
| --- | --- | --- |

## 可保留表达
- 原文短摘：...
  保留理由：...

要求：
- 原文只摘必要短句，保持原意，不要大段复刻。
- 如文本中没有明显问题，明确写“未发现明显风险表达”。
- 不要替媒体写完整改稿，不要输出最终审核结论。

【待审核媒体内容片段】
{chunk.strip()}
"""


def build_final_prompt(
    *,
    product_name: str,
    media_name: str,
    content_title: str,
    source_label: str,
    review_reference: str,
    additional_instruction: str,
    evidence_notes: str,
) -> str:
    reference_block = build_reference_block(review_reference)
    instruction_block = f"\n【用户补充要求】\n{additional_instruction.strip()}\n" if additional_instruction.strip() else ""
    return f"""请根据【风险初筛笔记】生成一份实验版《媒体待审内容审核报告》Markdown，供市场/PR团队审核媒体发来的稿件、脚本或逐字稿。

产品/项目：{product_name or "未填写"}
媒体/账号：{media_name or "未填写"}
内容标题：{content_title or "未填写"}
内容来源：{source_label or "未填写"}
{reference_block}
{instruction_block}

审核原则：
1. 审核结论必须落到四档：可继续流转 / 建议修改后流转 / 风险较高需人工复核 / 暂不建议放行。
2. 重点新增“可能有问题的表达”和“修改意见”，覆盖负面表达、不明确信息、事实/口径/保密/传播风险。
3. 对事实问题要区分“与参考冲突”和“参考不足待确认”，没有证据时不能武断判错。
4. 对媒体合理的独立评价要允许保留，只在风险和边界上提修改，不把全文改成品牌广告。
5. 修改建议必须具体：说明原文短摘、问题、风险等级、建议动作；能给示例时给一个短示例。
6. 如输入是视频逐字稿，提醒人工复核最终视频标题、画面字幕和口播上下文。

请输出：
# 媒体待审内容审核报告（实验版）

## 审核摘要
- 审核结论：
- 总体判断：
- 最需要先处理的 3 个点：
- 当前审核依据边界：

## 内容信息
- 产品/项目：
- 媒体/账号：
- 标题：
- 来源：

## 可能有问题的表达
用表格输出，列为：
| 优先级 | 原文短摘 | 问题类型 | 为什么有问题 | 建议动作 |
| --- | --- | --- | --- | --- |

问题类型至少覆盖命中的类别：负面/歧义、事实待确认、口径冲突、强表达/合规、保密/内部信息、传播风险。

## 修改意见
按优先级写 3-10 条。每条都包含：
- 原文短摘
- 修改目标
- 建议改法
- 可选改写示例
- 需要谁确认（市场/产品/PR/法务/媒体）

## 待确认信息清单
列出审核参考不足、必须人工核对后才能放行的参数、价格、时间、功能边界或测试条件。

## 可保留的媒体观点与亮点
列出不建议过度干预的观点或表达，并说明为什么可保留。

## 流转建议
说明下一步是直接回媒体、先问产品、补官方口径，还是需升级复核。

## 附：审核证据摘要
保留片段级证据线索，便于回到原稿定位；不要完整复刻原文。

格式要求：
- 输出纯中文 Markdown，不要 HTML/XML/飞书专用标签，不要把全文包进代码块。
- 表述要像内部审核报告，先给风险判断，再给可操作建议。
- 高风险问题必须在摘要和表格中都出现。

【风险初筛笔记】
{evidence_notes.strip()}
"""


def build_reference_block(review_reference: str) -> str:
    if not review_reference.strip():
        return "\n【审核参考】\n未提供官方口径/资料。涉及事实真伪时只标“待确认”，不要判定错误。\n"
    return f"""
【审核参考】
以下是用户提供的官方口径、产品资料、评测注意事项或审核关注点。只能按它判断冲突和边界，不得补充材料外事实。
{review_reference.strip()}
"""
