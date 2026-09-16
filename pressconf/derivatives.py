from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from pressconf.config_store import load_lark_config, resolve_model
from pressconf.lark_export import parse_json_object, post_json
from pressconf.model_client import stream_chat_model


DEFAULT_WRITING_SYSTEM = (
    "你是消费电子品牌市场团队的资深内容策略专家。你擅长从市场白皮书和产品简介中，"
    "提炼可公开传播的卖点资产，并生成媒体评测指南与正式新闻稿。输出中文 Markdown。"
    "必须尊重输入事实，不得编造参数、价格、SKU、时间、认证、首发、唯一、最强等信息。"
    "所有外发物料不得出现内部开发代号、项目代号或工程代号，例如 Project-A、Device-X、Alpha-2 这类仅供内部识别的名称；"
    "如输入材料出现这类名称，必须替换为产品标准名称或删除。"
    "排版遵循 W3C 中文排版相关规范：中文与英文、阿拉伯数字、单位和技术名混排时，按规范补足必要空格。"
    "涉及“国内”“中国”“回国”等可能引发地域、国别或立场误读的表述时，必须谨慎审阅；"
    "注意“中国”不等于“中国大陆”，产品可能已在港澳台发售，禁止笼统使用“回国”“中国首发”“国内首发”，"
    "优先使用“中国大陆市场首发”“面向中国大陆市场发布”等带限定词的表述；无法确认事实边界时，提示做事实核查或改成中性表达。"
)


REVIEW_GUIDE_SYSTEM = (
    "你是消费电子品牌市场团队中专门负责媒体评测指南的资深产品市场专家。"
    "评测指南是直接外发给媒体/KOL 的成稿文档，以品牌方第一人称（「我们」）写作，"
    "告诉媒体这台产品应该被理解成什么、我们为什么这样做、值得怎么体验和呈现。"
    "它读起来必须像一篇自信、克制、有判断的产品叙事成稿——不是内部分析报告，不是合规审查记录，不是要点清单式 brief。"
    "固定骨架（不增加任何额外 meta 结构，如「产品主叙事」「视觉素材锚点」「生成说明」「质检说明」）："
    "特别注意事项（开篇，固定模板，编号列表 5-8 条，只写标准外发控制信息）→ 产品概览（可选）→ 3-5 个 Part 正文 → 参数表/附录。"
    "Part 标题 = 卖点 + 传播判断；小节标题就是卖点短语本身；小节正文以自然段叙事为主："
    "先讲立项判断或行业痛点，再讲体验与场景，技术参数织入叙事来解释体验从何而来。"
    "禁止出现「体验结论」「用户感知」「技术/数据支撑」「媒体可拍点」「建议拍摄内容」「建议测试内容」「表达边界」这类模板化小标题；"
    "拍摄/测试建议如确有必要，以小字注「体验建议：……」附在相关小节末尾，每个 Part 至多一处。"
    "默认评测指南外发那一刻，所有不确定信息已在品牌侧解决：全文（含特别注意事项）不得出现 ⚠️、「待确认」「待补充」「以实测为准」等字样，"
    "材料中未决的内容直接略写或不写，不要罗列给媒体要求规避。"
    "必须尊重输入事实，不得编造参数、价格、SKU、时间、认证、首发、唯一、最强等信息。"
    "所有外发物料不得出现内部开发代号、项目代号或工程代号，例如 Project-A、Device-X、Alpha-2 这类仅供内部识别的名称；"
    "如输入材料出现这类名称，必须替换为产品标准名称或删除。"
    "排版遵循 W3C 中文排版相关规范：中文与英文、阿拉伯数字、单位和技术名混排时，按规范补足必要空格。"
    "涉及“国内”“中国”“回国”等可能引发地域、国别或立场误读的表述时，必须谨慎审阅；无法确认事实边界时，提示做事实核查或改成中性表达。"
)

INTERNAL_CODENAME_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:[PQON]\d{1,4})(?![A-Za-z0-9])")


def fetch_lark_doc_markdown(base_dir: Path, doc_url: str) -> str:
    config = load_lark_config(base_dir)
    url = str(config.get("mcp_url") or "").strip()
    if not url:
        raise RuntimeError("请先在后台配置飞书 MCP URL。")

    headers = parse_json_object(str(config.get("mcp_headers") or "{}"), "Headers JSON")
    payload = {
        "jsonrpc": "2.0",
        "id": "motoolbox-fetch-doc",
        "method": "tools/call",
        "params": {
            "name": "fetch-doc",
            "arguments": {
                "doc_id": doc_url,
                "need_url": False,
                "skip_task_detail": True,
            },
        },
    }
    response = post_json(url, payload, headers)
    markdown = extract_markdown(response)
    if not markdown.strip():
        raise RuntimeError(f"飞书 MCP 没有返回可用正文：{doc_url}")
    return markdown.strip()


def generate_derivatives(
    *,
    base_dir: Path,
    product_name: str,
    product_position: str,
    launch_info: str,
    price_info: str,
    whitepaper: str,
    intro: str,
    asset_table: str,
    additional_instruction: str = "",
    existing_asset: str = "",
    existing_review: str = "",
    existing_press: str = "",
    existing_qa: str = "",
    requested_outputs: list[str] | tuple[str, ...] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    model_config = resolve_model(base_dir, "writing")
    api_key = model_config["api_key"].strip()
    base_url = model_config["base_url"].rstrip("/")
    model = model_config["model"].strip()
    provider = model_config["provider"].strip()
    if not api_key:
        raise RuntimeError(f"没有配置 {model_config.get('name', model)} 的 API Key。")

    context = {
        "product_name": product_name,
        "product_position": product_position,
        "launch_info": launch_info,
        "price_info": price_info,
        "whitepaper": whitepaper,
        "intro": intro,
        "asset_table": asset_table,
        "additional_instruction": additional_instruction,
        "existing_asset": existing_asset,
        "existing_review": existing_review,
        "existing_press": existing_press,
        "existing_qa": existing_qa,
    }
    requested = normalize_requested_outputs(requested_outputs)
    if not requested:
        raise RuntimeError("请至少选择一份本次要生成的材料。")

    sections: dict[str, str] = {}
    asset_reference = existing_asset.strip() or asset_table.strip()
    if "asset" in requested:
        sections["asset"] = call_writing_model_streaming(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider=provider,
            prompt=build_asset_table_prompt(**context),
            max_tokens=8000,
            stage="卖点资产表",
            progress_callback=progress_callback,
        )
        asset_reference = sections["asset"]

    if "review" in requested:
        sections["review"] = call_writing_model_streaming(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider=provider,
            prompt=build_review_guide_prompt(**context, generated_asset_table=asset_reference),
            max_tokens=22000,
            stage="评测指南",
            system_prompt=REVIEW_GUIDE_SYSTEM,
            progress_callback=progress_callback,
        )

    if "press" in requested:
        sections["press"] = call_writing_model_streaming(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider=provider,
            prompt=build_press_release_prompt(**context, generated_asset_table=asset_reference),
            max_tokens=12000,
            stage="新闻稿",
            progress_callback=progress_callback,
        )

    if "qa" in requested:
        review_for_qa = sections.get("review", "").strip() or existing_review.strip()
        press_for_qa = sections.get("press", "").strip() or existing_press.strip()
        if not review_for_qa and not press_for_qa:
            raise RuntimeError("质检需要本次生成或当前已编辑的评测指南/新闻稿。")
        sections["qa"] = call_writing_model_streaming(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider=provider,
            prompt=build_qa_prompt(
                product_name=product_name,
                generated_asset_table=asset_reference,
                review_guide=review_for_qa,
                press_release=press_for_qa,
                additional_instruction=additional_instruction,
            ),
            max_tokens=5000,
            stage="自检与待确认",
            progress_callback=progress_callback,
        )

    result = sections_to_result(sections)
    codename_warnings = find_internal_codenames(result)
    if codename_warnings:
        warning_text = build_codename_warning(codename_warnings)
        if "qa" in sections:
            sections["qa"] = (sections["qa"].strip() + "\n\n" + warning_text).strip()
            result = sections_to_result(sections)
    return result, {
        "model": model,
        "provider": provider,
        "base_url": base_url,
        "sections": sections,
        "requested_outputs": requested,
        "warnings": [build_codename_warning(codename_warnings)] if codename_warnings else [],
    }


def call_writing_model_streaming(
    *,
    api_key: str,
    base_url: str,
    model: str,
    provider: str,
    prompt: str,
    max_tokens: int,
    stage: str,
    system_prompt: str = DEFAULT_WRITING_SYSTEM,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
    ]
    chunks: list[str] = []

    def handle_delta(delta: str) -> None:
        chunks.append(delta)
        current = "".join(chunks)
        if progress_callback and len(current) % 120 < len(delta):
            progress_callback({"stage": stage, "generated_chars": len(current), "preview": tail_preview(current)})

    result = stream_chat_model(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=messages,
        temperature=0.35,
        max_tokens=max_tokens,
        timeout=420,
        on_delta=handle_delta,
    )
    if not result:
        raise RuntimeError("模型没有返回内容。")
    if progress_callback:
        progress_callback({"stage": stage, "generated_chars": len(result), "preview": tail_preview(result)})
    return result


def build_common_context(
    *,
    product_name: str,
    product_position: str,
    launch_info: str,
    price_info: str,
    whitepaper: str,
    intro: str,
    asset_table: str,
    additional_instruction: str = "",
    existing_asset: str = "",
    existing_review: str = "",
    existing_press: str = "",
    existing_qa: str = "",
) -> str:
    refinement_block = ""
    if additional_instruction.strip():
        refinement_block = f"""

【用户二次加工要求】
{additional_instruction.strip()}

执行要求：
- 本次生成必须优先响应上述方向性要求。
- 如果用户要求与事实边界冲突，以事实边界为准，并在质检中说明。
"""
    existing_block = ""
    if any(item.strip() for item in (existing_asset, existing_review, existing_press, existing_qa)):
        existing_block = f"""

【当前已编辑产出，仅供二次加工参考】
卖点资产表：
{existing_asset.strip() or "无"}

评测指南：
{existing_review.strip() or "无"}

新闻稿：
{existing_press.strip() or "无"}

质检：
{existing_qa.strip() or "无"}
"""
    return f"""
产品名称：{product_name or "待补充"}
产品定位：{product_position or "待补充"}
发布信息：{launch_info or "待补充"}
价格 / SKU：{price_info or "待补充"}

{refinement_block}
{existing_block}

【工作台卖点资产草稿，可参考但不要盲从】
{asset_table or "无"}

【市场白皮书】
{whitepaper}

【产品简介】
{intro}
""".strip()


def build_asset_table_prompt(**context: str) -> str:
    return f"""
请基于已提供的【市场白皮书】和/或【产品简介】，生成一份可复用的“卖点资产表”。

要求：
1. 产品简介决定主线和卖点优先级；市场白皮书补充事实、数据、技术依据、竞对参照和边界。
2. 不得编造输入材料中不存在的信息；缺失处写“待补充”。
3. TBD、内部备注、灰字、风险、未确认数据、竞品攻击话术不得进入推荐表达。
4. 所有“首发、唯一、最高、最强、行业第一”等强表达，必须标明依据或标为待核实。
5. 所有外发物料不得出现内部开发代号、项目代号或工程代号，例如 Project-A、Device-X、Alpha-2；如材料中出现，进入“注意边界”，不得进入“推荐表达”。

请输出 Markdown 表格，字段包括：
优先级、卖点名称、用户收益、技术/产品支撑、关键数据、竞对参照、推荐表达、注意边界、适合评测展开方式、适合新闻稿表达方式。

{build_common_context(**context)}
""".strip()


def build_review_guide_prompt(*, generated_asset_table: str, **context: str) -> str:
    return f"""
请基于已提供的【卖点资产表】【市场白皮书】和/或【产品简介】，生成一份“评测指南”。

这不是媒体 brief，不是新闻稿摘要，也不是要点清单。它是一份可直接外发给媒体老师的评测指南成稿，以品牌方第一人称（「我们」）写作：有节奏、有标题、有体验叙事，告诉媒体这台产品应该被理解成什么、我们为什么这样做、值得怎么体验和呈现。

固定骨架（严格按此结构输出，不增加任何额外 meta 结构，如「产品主叙事」「视觉素材锚点」「生成说明」「质检说明」）：
1. 开头必须是“特别注意事项”，严格套用以下固定模板，保持条目顺序和句式，只替换方括号内容；输入材料没给的信息保留方括号占位，交付前由内部补齐，不要写成“待确认/待补充”：

# 特别注意事项
1. 产品标准名称：[产品标准名称]
2. 主要卖点：
[卖点短语 1]｜[卖点短语 2]｜[卖点短语 3]
[卖点短语 4]｜[卖点短语 5]｜[卖点短语 6]
3. 媒体机为试产样机，并非最终市售版本，评测中如遇到功能与本文档描述不一致的情况，请与 [品牌] 媒介同事联系沟通
4. 评测前登录 [品牌账号]，并将系统升至最新版本进行评测
5. 所有评测内容（包括不限于图赏、评测、视频），解禁时间统一为 <text color="red">**[X 月 X 日 X:XX]**</text>，请各位媒体老师留意。

主要卖点写成 1-2 行卖点短语，用「｜」分隔，取自各 Part 的核心卖点。个别产品确需补充条目（如配件适配、网络制式说明）可在第 5 条后追加，总数不超过 8 条。不要把输入材料里的待确认/未决信息罗列在这里告诉媒体规避——默认指南外发那一刻，这些问题已在品牌侧解决。
2. “产品概览”（可选，材料足够时写）：3-5 行列出各 Part 标题和一句话定位。
3. 正文 3-5 个 Part，每个 Part 对应一个核心卖点方向。
4. 每个 Part 的写法：
   - Part 标题 = 卖点 + 传播判断，例如“设计与工艺：开合皆精巧”“大外屏：初见『全面』，上手『全能』”“体验：彻底告别『美丽小废物』”。
   - 每个 Part 下分 3-6 个小节，小节标题就是卖点短语本身，例如“合上超小巧，展开超轻薄”“大容量高密度电池，轻薄不打折”。
   - 小节正文以自然段叙事为主：先讲立项判断或行业痛点（为什么做），再讲体验与场景（用起来什么样），技术参数织入叙事来解释体验从何而来。确需并列的规格可用短列表，但列表不能成为主文体。
   - 禁止出现「体验结论」「用户感知」「技术/数据支撑」「媒体可拍点」「建议拍摄内容」「建议测试内容」「表达边界」这类模板化小标题。拍摄/测试建议如确有必要，以小字注“体验建议：……”附在相关小节末尾，每个 Part 至多一处。
5. 最后是“参数表 / 附录”：表格形式，只作正文补充，不替代正文。

竞品对比：
- 鼓励有事实支撑的具名横向对比（如“对比 iPhone 15 Pro Max 轻 19g”），必要时可用对比表格。这是评测指南的价值所在，不要回避。
- “最强”“第一”“唯一”“首创”必须有输入材料中明确的适用范围支撑，按材料给出的范围表述；无支撑则不用。
- 禁止贬损性、攻击性语言；对比陈述事实即可。

事实与风险收口：
- 输入材料中标注为待确认、目标值、方案未定的内容：默认外发前已在品牌侧解决，正文直接略写或不写，也不要罗列进“特别注意事项”。全文不得出现 ⚠️、“待确认”“待补充”“以实测为准”等字样；未决事实留给内部质检环节列出，不进入外发稿。
- 实验室数据所在小节末尾统一加一条灰色注：“* 以上数据为实验室测试数据，依据行业内测量方式不同，实际结果可能略有差异。”

行文要求：
- 要有产品市场文档的“成稿感”：句子可以有气口，有判断，有承接；自信、克制。
- 多用场景化表达：上手、点亮、展开、合上、夜景、长焦、续航、弱网、游戏、自拍、社交分享等。
- 技术表达必须落到体验，不要写成工程规格堆砌。
- 不要过短。目标长度：至少 5000 中文字；如果材料丰富，写到 8000-12000 字。
- 不要为了凑长空泛铺陈。每段都必须有来自材料的事实支撑或明确的评测表达价值。
- 全文不得出现内部开发代号、项目代号或工程代号，例如 Project-A、Device-X、Alpha-2；只能使用产品标准名称、系列名称或公开传播名称。
- 排版遵循 W3C 中文排版相关规范：中文与英文、阿拉伯数字、单位、技术名混排时，补足必要空格，例如产品名、芯片名、数值和单位前后不要黏连。
- 涉及“国内”“中国”“回国”等表述时，必须逐句审阅是否有事实依据和立场风险；如材料未明确支持，改写为更中性的地域/市场表述，或放入待确认与免责声明提示事实核查。

行文范例（只学习以下节选的标题方式、第一人称口吻、段落节奏和“技术为体验服务”的叙事方式；尖括号内容是结构占位，禁止当作事实复用）：

# Part 1. 设计与工艺：开合皆精巧

## 合上超小巧，展开超轻薄
无论开合皆有出色手感，是我们在 <产品标准名称> 立项之初确定的目标。

合上时，它是一个掌心大小的灵动宝盒，可以征服一切「小废包」；前后全等深微曲玻璃，营造出极佳的手感。

展开后，它在保持轻薄机身的同时带来更大的可视面积。与材料中选定的对标产品相比，<产品标准名称> 在重量和握持宽度上更有优势，同时保留了大屏体验。

# Part 4. 体验：彻底告别「美丽小废物」

## 续航：大容量高密度电池，久用无忧
在行业中，「小折叠」常常意味着「小电池」。特殊形态和精密堆叠给电池设计带来挑战，因此我们从材料体系与空间利用率入手，在有限机身内提升电池容量和能量密度，让轻巧形态与全天续航不再彼此妥协。

材料创新不能只停留在参数表上。正文还要继续解释输入材料中给出的技术方案如何改善传输效率、充电速度或长期使用体验，把工程指标转化为用户能感知的收益。

（范例结束）

【卖点资产表】
{generated_asset_table or "未提供卖点资产表，请直接从原始材料提炼评测主线。"}

{build_common_context(**context)}
""".strip()


def build_press_release_prompt(*, generated_asset_table: str, **context: str) -> str:
    return f"""
请基于已提供的【卖点资产表】【市场白皮书】和/或【产品简介】，生成正式新品新闻稿。

核心原则：
1. 产品简介决定主线和卖点排序，市场白皮书仅提供事实支撑。新闻稿不是白皮书摘要，不是评测指南，而是围绕发布事件的公共叙事。
2. 动笔前先提炼一句话新闻主旨——这款产品发布的意义（某系列转型？某品类突破？某代产品全面升级？某项体验建立新标准？），让这句话贯穿整篇叙事逻辑，不必单独输出。
3. 面向公众、媒体、渠道传播，正式、清晰、可信；只写确定、可公开、可传播的信息。
4. 技术点必须转成用户收益（每个技术点对应“所以用户能……”），不堆工程语言。
5. 不得编造价格、SKU、时间、认证、首发、唯一、最高等信息；缺失写待补充。
6. 不写内部备注、TBD、风险、灰字、竞品攻击、未确认数据、供应商/成本等内部商业信息，不出现“据白皮书”“内部评估”等来源痕迹，不出现“建议媒体体验”“可以测试”等评测语气。
7. 全文不得出现内部开发代号、项目代号或工程代号，例如 Project-A、Device-X、Alpha-2；只能使用产品标准名称、系列名称或公开传播名称。
8. 输出必须是一篇可直接发布的新闻稿成稿，不是写作提纲、结构模板、卖点摘要或资料附录。
9. 排版遵循 W3C 中文排版相关规范：中文与英文、阿拉伯数字、单位、技术名混排时，补足必要空格，例如产品名、芯片名、数值和单位前后不要黏连。

地域表述审查（强制）：
- 核心风险：“中国”不等于“中国大陆”。产品可能已在港澳台市场发布售卖，笼统写“中国首发”“回国”会暗示港澳台不是中国的一部分，属于严重政治错误。
- 禁止使用：“回国”“回归中国”“中国首发”“首次回到中国”“国内首发”。
- 优先使用：“面向中国大陆市场发布”“中国大陆市场首发”“首次登陆中国大陆”“首次与大陆用户见面”。
- 无法从材料确认是否已在港澳台发售时，改用中性表述，并在编辑备注中提示“请确认是否已在港澳台发售，如是则需改为‘中国大陆市场首发’”。

成稿格式：
- 开头先给一个正式新闻标题，紧跟 3-5 个备选标题（有序列表，方便编辑挑选）。标题包含产品名和核心卖点/定位，已知价格时可含价格，不使用未证实的“唯一/第一/最强”。
- 标题后直接进入新闻正文。首段自然交代发布信息、产品标准名称、产品定位、核心卖点和已确认的价格/开售信息；不要写“导语：”标签。
- 后续用自然段完成整体升级、重点卖点、技术带来的体验收益、价格版本与开售信息。可以按传播节奏分段，必要时可少量使用不带编号的关键词式段间标题（如“设计：数字系列一脉相承的精致好手感”），但不能用“一、二、三”“1. 2. 3.”这类章节提示，章节之间不加分隔线。
- 收束段融入最后一个章节末尾，回到产品定位、用户价值和品类意义，不单独加“结语”标题。
- 正文不输出提纲说明、字段标签、Markdown 表格、参数表、SKU 表或待确认表。参数、版本、价格、开售信息如需出现，必须自然写进正文段落。
- 不把正文写成逐项罗列的 bullet list；卖点写成连贯段落，用分号、逗号或短句串联，让信息沿新闻叙事自然成段。
- 如有待确认项或地域表述需核实项，在正文结束后另起“编辑备注”部分集中列出，不混入新闻稿正文。

【卖点资产表】
{generated_asset_table or "未提供卖点资产表，请直接从原始材料提炼新闻稿主线。"}

{build_common_context(**context)}
""".strip()


def build_qa_prompt(
    *,
    product_name: str,
    generated_asset_table: str,
    review_guide: str,
    press_release: str,
    additional_instruction: str = "",
) -> str:
    instruction_block = ""
    if additional_instruction.strip():
        instruction_block = f"""

【用户二次加工要求】
{additional_instruction.strip()}

检查时必须优先确认上述要求是否已在评测指南、新闻稿或卖点资产中落实；未落实的地方要列入高优先级问题，并给出可直接补写/改写的建议。
"""
    return f"""
请检查以下副产物是否存在事实、结构、风格和风险问题。

产品名称：{product_name or "待补充"}
{instruction_block}

重点检查：
1. 产品名称、价格、SKU、时间、参数是否前后不一致或疑似编造。
2. 是否出现 TBD、内部备注、灰字风险、竞品攻击话术、未确认数据。
3. 是否出现内部开发代号、项目代号或工程代号，例如 Project-A、Device-X、Alpha-2；任何外发物料出现这类名称都必须标为高风险并建议替换/删除。
4. 评测指南是否符合固定骨架：特别注意事项（是否套用固定模板：产品标准名称、主要卖点短语、样机说明与沟通方式、评测前升级要求、红色加粗解禁时间，总数不超过 8 条）→ 产品概览（可选）→ 3-5 个 Part 正文 → 参数表/附录；是否混入了「产品主叙事」「视觉素材锚点」「生成说明」「质检说明」等额外 meta 结构；特别注意事项是否被写成了材料中未决信息的罗列、要求媒体规避（这属于必须修正的问题——未决事实只在本质检的“风险问题”中向内部列出，不进入外发稿）。
5. 评测指南 Part 写法是否达标：
   - Part 标题是否是“卖点 + 传播判断”，小节标题是否是卖点短语本身。
   - 小节正文是否以自然段叙事为主（立项判断/行业痛点 → 体验场景 → 技术织入），列表是否喧宾夺主。
   - 是否出现「体验结论」「用户感知」「技术/数据支撑」「媒体可拍点」「建议拍摄内容」「建议测试内容」「表达边界」等模板化小标题；拍摄/测试建议是否超出“每个 Part 至多一处小字注”的限制。
   - 全文（含特别注意事项）是否出现 ⚠️、“待确认”“待补充”“以实测为准”等不应出现在外发稿中的字样。
   - 实验室数据所在小节是否带灰色免责注。
6. 评测指南是否仍然过短、过干、像媒体 brief 或内部分析报告，而不是第一人称、可直接外发的产品叙事成稿。
7. 新闻稿是否像白皮书摘要，是否没有把技术转成用户收益。
8. 新闻稿是否仍是提纲/模板格式：除开头备选标题外，正文出现“一、二、三”“1. 2. 3.”章节提示、“导语：”等字段标签、Markdown 表格、参数/SKU/待确认附录，而不是可直接发布的自然段成稿（正文之后单独的“编辑备注”允许存在，但其内容不得混入正文）。
9. 评测指南和新闻稿是否符合 W3C 中文排版相关规范：中文与英文、阿拉伯数字、单位、技术名混排时是否缺少必要空格；如发现黏连，给出可直接替换的修订。
10. 地域表述审查（高优先级）：是否出现“回国”“回归中国”“中国首发”“国内首发”等表述。“中国”不等于“中国大陆”——产品可能已在港澳台发售，这类表述属于严重政治风险，必须改为“中国大陆市场首发”“面向中国大陆市场发布”等带“大陆”限定词的表述；无法确认港澳台发售情况的，提示事实核查。其他“国内”“中国”相关表述逐条判断是否有输入材料事实依据。
11. 强表达是否有依据。

请按以下格式输出：
- 结构缺口：缺少哪个必要模块，为什么影响评测指南可用性
- 内容缺口：缺少哪些可拍、可测、可写、可体验信息
- 风险问题：事实、强表达、内部信息、内部开发代号、敏感地域/国别/立场表述、待确认项
- 排版问题：列出中英/数字/单位混排缺少空格的位置，并给出改写
- 推荐补写：给出可直接补进评测指南的改写段落或小节标题
- 新闻稿问题：仅列新闻稿相关问题

【卖点资产表】
{generated_asset_table}

【评测指南】
{review_guide}

【新闻稿】
{press_release}
""".strip()


def extract_markdown(value: Any) -> str:
    candidates: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key in ("markdown", "content", "text"):
                value = item.get(key)
                if isinstance(value, str):
                    parsed = try_json(value)
                    if parsed is not None:
                        visit(parsed)
                    else:
                        candidates.append(value)
            for child in item.values():
                if not isinstance(child, str):
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    candidates = [item.strip() for item in candidates if item and item.strip()]
    if not candidates:
        return ""
    return max(candidates, key=len)


def try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def tail_preview(text: str, limit: int = 1200) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def parse_sections(text: str) -> dict[str, str]:
    markers = {
        "asset": "<<<ASSET_TABLE>>>",
        "review": "<<<REVIEW_GUIDE>>>",
        "press": "<<<PRESS_RELEASE>>>",
        "qa": "<<<QA_CHECK>>>",
    }
    positions = [(key, text.find(marker), marker) for key, marker in markers.items()]
    found = [(key, pos, marker) for key, pos, marker in positions if pos >= 0]
    if not found:
        return {}
    found.sort(key=lambda item: item[1])
    sections: dict[str, str] = {}
    for index, (key, pos, marker) in enumerate(found):
        start = pos + len(marker)
        end = found[index + 1][1] if index + 1 < len(found) else len(text)
        sections[key] = text[start:end].strip()
    return sections


def normalize_requested_outputs(value: list[str] | tuple[str, ...] | None) -> list[str]:
    allowed = ("asset", "review", "press", "qa")
    requested = {str(item).strip() for item in (value or [])}
    return [item for item in allowed if item in requested]


def sections_to_result(sections: dict[str, str]) -> str:
    markers = {
        "asset": "<<<ASSET_TABLE>>>",
        "review": "<<<REVIEW_GUIDE>>>",
        "press": "<<<PRESS_RELEASE>>>",
        "qa": "<<<QA_CHECK>>>",
    }
    return "\n\n".join(
        f"{markers[key]}\n{sections[key].strip()}"
        for key in markers
        if sections.get(key, "").strip()
    )


def find_internal_codenames(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in INTERNAL_CODENAME_PATTERN.findall(text):
        seen.setdefault(match, None)
    return list(seen.keys())


def build_codename_warning(codenames: list[str]) -> str:
    joined = "、".join(codenames)
    return f"""
## 高风险：内部开发代号泄露

检测到输出中仍包含疑似内部开发代号：{joined}

处理要求：
- 所有外发物料不得出现内部开发代号、项目代号或工程代号。
- 请将上述代号替换为产品标准名称、系列名称或公开传播名称。
- 如果无法确认公开名称，请删除相关表述，或标记为“待补充产品标准名称”。
""".strip()
