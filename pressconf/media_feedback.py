from __future__ import annotations

import re
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree as ET

from pypdf import PdfReader

from pressconf.config_store import resolve_model
from pressconf.derivatives import call_writing_model_streaming
from pressconf.domains import DomainProfile, load_domain


FEEDBACK_SYSTEM = (
    "你是消费电子品牌市场团队的资深媒体反馈分析师。你擅长把媒体沟通会后的问卷原始表单，"
    "整理成一份详实、简练、有信息量、可供市场/产品/传播团队决策的中文媒体反馈文档。"
    "你的判断必须来自输入材料，不能编造媒体名称、价格、比例、投票结果、产品参数或未出现的观点。"
    "如果需要估算频次或比例，必须明确基于可识别有效样本数，并说明口径。"
)

REVIEW_VIDEO_SYSTEM = (
    "你是消费电子品牌市场团队的资深媒体评测分析师。你擅长阅读媒体/KOL评测视频逐字稿，"
    "判断视频整体情感倾向、提炼正负面观点、识别可供老板汇报和传播引用的媒体原话。"
    "所有判断必须来自逐字稿或已给定的视频信息，不得编造媒体名称、产品参数、测试结果、报价或原文不存在的观点。"
    "引用原文时只做短摘，必须保持原意，并标注可追溯的时间线索或片段线索。"
)

INTERNAL_CODENAME_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:[PQON]\d{1,4})(?![A-Za-z0-9])")
MARKDOWN_FENCE_PATTERN = re.compile(r"^\s*```(?:markdown|md)?\s*\n(?P<body>.*)\n\s*```\s*$", re.IGNORECASE | re.DOTALL)


@dataclass
class PriceRecord:
    media: str
    product: str
    raw_value: str
    parsed_value: int | None
    note: str = ""


@dataclass
class PriceAnalysis:
    records: list[PriceRecord]

    @property
    def has_records(self) -> bool:
        return any(record.parsed_value is not None for record in self.records)

    def to_markdown(self) -> str:
        if not self.records:
            return "## 价格预期确定性计算\n\n未在原始材料中识别到可计算的媒体价格预期字段。"

        lines = ["## 价格预期确定性计算", ""]
        lines.append("> 说明：本节由程序从原始问卷中确定性抽取和计算，不由模型推理生成。只有单一明确的绝对价格会计入统计；多价格表达（如“5499 或 5699”“5899-6299”）和相对价格（如“与上代持平”“低 500”）均标记为相对/待判断口径，不计入均值、中位数、最高/最低。原始表达保留在对照表中，便于二次加工时人工判断。")
        lines.append("")
        lines.extend(self.summary_markdown())
        lines.append("")
        lines.extend(self.table_markdown())
        return "\n".join(lines).strip()

    def summary_markdown(self) -> list[str]:
        lines = ["### 价格统计摘要", "", "| 产品/版本 | 有效样本数 | 平均数 | 中位数 | 最低 | 最高 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for product in sorted({record.product for record in self.records}):
            values = [record.parsed_value for record in self.records if record.product == product and record.parsed_value is not None]
            if not values:
                lines.append(f"| {escape_table_cell(product)} | 0 | - | - | - | - |")
                continue
            mean_value = round(statistics.fmean(values))
            median_value = round(statistics.median(values))
            lines.append(
                f"| {escape_table_cell(product)} | {len(values)} | {mean_value} | {median_value} | {min(values)} | {max(values)} |"
            )
        return lines

    def table_markdown(self) -> list[str]:
        lines = ["### 媒体价格预期对照表", "", "| 媒体 | 产品/版本 | 原始价格预期 | 计入统计值 | 备注 |", "| --- | --- | --- | ---: | --- |"]
        for record in self.records:
            value = str(record.parsed_value) if record.parsed_value is not None else "-"
            lines.append(
                "| "
                + " | ".join(
                    [
                        escape_table_cell(record.media),
                        escape_table_cell(record.product),
                        escape_table_cell(record.raw_value),
                        value,
                        escape_table_cell(record.note),
                    ]
                )
                + " |"
            )
        return lines


def extract_feedback_source(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    if suffix in {".xlsx", ".xlsm"}:
        return xlsx_to_markdown(path)
    if suffix == ".docx":
        return docx_to_markdown(path)
    if suffix == ".pdf":
        return pdf_to_markdown(path)
    raise ValueError(f"暂不支持的文件类型：{path.suffix}")


def extract_price_analysis(path: Path, extracted_text: str = "") -> PriceAnalysis:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        try:
            return extract_xlsx_price_analysis(path)
        except Exception:
            return extract_text_price_analysis(extracted_text or extract_feedback_source(path))
    return extract_text_price_analysis(extracted_text or extract_feedback_source(path))


def extract_xlsx_price_analysis(path: Path) -> PriceAnalysis:
    rows = read_xlsx_rows(path)
    if not rows:
        return PriceAnalysis([])
    header_source, body = split_xlsx_header(rows)
    header = normalize_header(header_source)
    media_index = find_media_column(header)
    price_indexes = find_price_columns(header)
    records: list[PriceRecord] = []
    for row in body:
        padded = row[: len(header)] + [""] * max(0, len(header) - len(row))
        if not any(cell.strip() for cell in padded):
            continue
        media = clean_text(padded[media_index]) if media_index is not None and media_index < len(padded) else ""
        media = media or first_nonempty_cell(padded[:4]) or "未标注媒体"
        for index in price_indexes:
            raw_value = clean_text(padded[index]) if index < len(padded) else ""
            if not raw_value:
                continue
            parsed_value, note = parse_price_value(raw_value)
            records.append(
                PriceRecord(
                    media=media,
                    product=price_product_name(header[index]),
                    raw_value=raw_value,
                    parsed_value=parsed_value,
                    note=note,
                )
            )
    return PriceAnalysis(records)


def extract_text_price_analysis(text: str) -> PriceAnalysis:
    records: list[PriceRecord] = []
    for line in str(text or "").splitlines():
        clean = clean_text(re.sub(r"<br\s*/?>", " ", line, flags=re.IGNORECASE))
        if not clean or not re.search(r"(价格|定价|起售价|售价)", clean):
            continue
        parsed_value, note = parse_price_value(clean)
        if parsed_value is None and "持平" not in clean and "贵" not in clean and "便宜" not in clean:
            continue
        media = infer_markdown_row_media(clean)
        product = infer_text_price_product(clean)
        records.append(PriceRecord(media=media, product=product, raw_value=clean[:500], parsed_value=parsed_value, note=note))
    return PriceAnalysis(records)


def find_media_column(header: list[str]) -> int | None:
    for index, name in enumerate(header):
        if re.search(r"(媒体名称|媒体|账号|机构)", name):
            return index
    return None


def find_price_columns(header: list[str]) -> list[int]:
    result = []
    for index, name in enumerate(header):
        compact = re.sub(r"\s+", "", name)
        if re.search(r"(价格|定价|售价|心里预期|心理预期)", compact) and not re.search(r"(争议|疑虑|原因|倾向|购买|信心)", compact):
            result.append(index)
    return result


def price_product_name(header: str) -> str:
    name = clean_text(header)
    if "/" in name:
        name = name.split("/")[-1]
    name = re.sub(r"^\d+[、.．]?", "", name)
    name = re.sub(r".*?基于您了解的产品力.*?/", "", name)
    name = re.sub(r".*?心理?预期价格.*?/", "", name)
    name = re.sub(r"(价格|定价|售价|起|（起）|\\(起\\)|心里预期|心理预期)", "", name)
    return clean_text(name.strip(" /-:：")) or "未标注产品"


def first_nonempty_cell(cells: list[str]) -> str:
    for cell in cells:
        value = clean_text(cell)
        if value:
            return value
    return ""


def parse_price_value(raw_value: str) -> tuple[int | None, str]:
    text = normalize_price_text(raw_value)
    if is_relative_price_expression(text):
        return None, "相对价格/待判断口径，未计入统计"
    values = extract_price_numbers(text)
    if not values:
        if re.search(r"(持平|一样|不变|贵|便宜|低|高)", text):
            return None, "相对价格/待判断口径，未计入统计"
        return None, "未识别到有效绝对价格"
    if len(values) == 1:
        return values[0], ""
    return None, f"多价格表达/待判断口径，识别到 {values}，未计入统计"


def is_relative_price_expression(text: str) -> bool:
    if re.search(r"(持平|一样|不变)", text):
        return True
    if re.search(r"(比|较|相对|相比|对比).{0,20}\d{2,5}.{0,8}(贵|便宜|低|高|多|少|溢价)", text):
        return True
    if re.search(r"(贵|便宜|低|高|加价|溢价|多付|多出|少)\s*\d{2,5}", text) and not re.search(r"(最高|高达|低至|低于|高于)\s*\d{3,5}", text):
        return True
    return False


def normalize_price_text(value: str) -> str:
    return (
        clean_text(value)
        .replace("－", "-")
        .replace("—", "-")
        .replace("–", "-")
        .replace("～", "-")
        .replace("~", "-")
        .replace("k", "K")
    )


def extract_price_numbers(text: str) -> list[int]:
    values: list[int] = []
    used_spans: list[tuple[int, int]] = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*万", text):
        values.append(round(float(match.group(1)) * 10000))
        used_spans.append(match.span())
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*K", text):
        values.append(round(float(match.group(1)) * 1000))
        used_spans.append(match.span())
    for match in re.finditer(r"\d{3,5}", text):
        if any(start <= match.start() < end for start, end in used_spans):
            continue
        number = int(match.group(0))
        if 500 <= number <= 50000:
            values.append(number)
    deduped: list[int] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def infer_markdown_row_media(line: str) -> str:
    cells = [clean_text(cell.replace("\\|", "|")) for cell in line.strip().strip("|").split("|")]
    if len(cells) >= 4:
        for index in (3, 2, 1):
            if index < len(cells) and cells[index] and not re.fullmatch(r"\d+", cells[index]):
                return cells[index]
    return "未标注媒体"


def infer_text_price_product(line: str) -> str:
    cells = [clean_text(cell.replace("\\|", "|")) for cell in line.strip().strip("|").split("|")]
    for cell in cells:
        if re.search(
            r"(Ultra|Pro(?:\s*Max)?|Pad|Fold|Flip|Phone|手机|平板|眼镜|系列|第\s*\d+\s*代|[A-Z][A-Za-z]+\s*\d{1,3})",
            cell,
            flags=re.IGNORECASE,
        ):
            return truncate_cell(re.sub(r"(价格|定价|售价|起售价).*", "", cell), 80) or "未标注产品"
    return "未标注产品"


def xlsx_to_markdown(path: Path, max_cell_chars: int = 2200) -> str:
    rows = read_xlsx_rows(path)
    if not rows:
        return ""
    header_source, body = split_xlsx_header(rows)
    header = normalize_header(header_source)
    lines = [f"# 原始问卷：{path.name}", "", f"- 有效行数：{sum(1 for row in body if any(cell.strip() for cell in row))}", ""]
    lines.append("| " + " | ".join(escape_table_cell(cell or f"列{i + 1}") for i, cell in enumerate(header)) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in body:
        if not any(cell.strip() for cell in row):
            continue
        padded = row[: len(header)] + [""] * max(0, len(header) - len(row))
        values = [truncate_cell(cell, max_cell_chars) for cell in padded[: len(header)]]
        lines.append("| " + " | ".join(escape_table_cell(value) for value in values) + " |")
    return "\n".join(lines).strip()


def split_xlsx_header(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    if len(rows) < 2:
        return rows[0], rows[1:]
    first, second = rows[0], rows[1]
    width = max(len(first), len(second))
    first = first + [""] * (width - len(first))
    second = second + [""] * (width - len(second))
    second_has_subheaders = sum(1 for cell in second if cell.strip()) >= max(3, width // 4)
    first_has_blanks = sum(1 for cell in first if not cell.strip()) >= max(2, width // 5)
    if second_has_subheaders and first_has_blanks:
        merged = []
        for top, sub in zip(first, second):
            if top and sub:
                merged.append(f"{top} / {sub}")
            else:
                merged.append(top or sub)
        return merged, rows[2:]
    return rows[0], rows[1:]


def read_xlsx_rows(path: Path) -> list[list[str]]:
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared_strings = read_shared_strings(archive)
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_by_id = {item.attrib["Id"]: item.attrib["Target"] for item in rels}
        sheets = workbook.findall(".//main:sheets/main:sheet", ns)
        if not sheets:
            return []
        rel_id = sheets[0].attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id", "")
        sheet_target = rel_by_id.get(rel_id, "worksheets/sheet1.xml")
        sheet_path = "xl/" + sheet_target.lstrip("/")
        root = ET.fromstring(archive.read(sheet_path))

    rows: list[list[str]] = []
    for row in root.findall(".//main:sheetData/main:row", ns):
        values_by_col: dict[int, str] = {}
        max_col = 0
        for cell in row.findall("main:c", ns):
            ref = cell.attrib.get("r", "")
            col = column_number(ref) if ref else max_col + 1
            max_col = max(max_col, col)
            values_by_col[col] = xlsx_cell_text(cell, shared_strings, ns)
        rows.append([values_by_col.get(col, "") for col in range(1, max_col + 1)])
    return trim_empty_edges(rows)


def read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    return ["".join(t.text or "" for t in item.findall(".//main:t", ns)) for item in root.findall("main:si", ns)]


def xlsx_cell_text(cell: ET.Element, shared_strings: list[str], ns: dict[str, str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return clean_text("".join(t.text or "" for t in cell.findall(".//main:t", ns)))
    value = cell.find("main:v", ns)
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        try:
            return clean_text(shared_strings[int(value.text)])
        except (ValueError, IndexError):
            return clean_text(value.text)
    return clean_text(value.text)


def column_number(ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", ref.upper())
    number = 0
    for letter in letters:
        number = number * 26 + ord(letter) - 64
    return number or 1


def docx_to_markdown(path: Path) -> str:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    lines = [f"# 原始反馈：{path.name}", ""]
    for table in root.findall(".//w:tbl", ns):
        rows = []
        for tr in table.findall("w:tr", ns):
            cells = []
            for tc in tr.findall("w:tc", ns):
                text = clean_text("\n".join(paragraph_text(p, ns) for p in tc.findall(".//w:p", ns)))
                cells.append(text)
            if any(cells):
                rows.append(cells)
        if rows:
            width = max(len(row) for row in rows)
            header = rows[0] + [""] * (width - len(rows[0]))
            lines.append("| " + " | ".join(escape_table_cell(cell or f"列{i + 1}") for i, cell in enumerate(header)) + " |")
            lines.append("| " + " | ".join("---" for _ in header) + " |")
            for row in rows[1:]:
                padded = row + [""] * (width - len(row))
                lines.append("| " + " | ".join(escape_table_cell(truncate_cell(cell, 2400)) for cell in padded) + " |")
            lines.append("")
    paragraphs = [clean_text(paragraph_text(p, ns)) for p in root.findall(".//w:body/w:p", ns)]
    paragraphs = [item for item in paragraphs if item]
    if paragraphs:
        lines.extend(["## 文档段落", ""])
        lines.extend(paragraphs)
    return "\n".join(lines).strip()


def pdf_to_markdown(path: Path) -> str:
    reader = PdfReader(path)
    lines = [f"# PDF 原文：{path.name}", ""]
    extracted_pages = 0
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue
        extracted_pages += 1
        lines.extend([f"## 第 {page_number} 页", "", text, ""])
    if not extracted_pages:
        raise ValueError("PDF 未提取到可用文本，可能是扫描件或纯图片 PDF。")
    return "\n".join(lines).strip()


def paragraph_text(paragraph: ET.Element, ns: dict[str, str]) -> str:
    return "".join(t.text or "" for t in paragraph.findall(".//w:t", ns))


def normalize_header(row: list[str]) -> list[str]:
    headers = [clean_text(cell) or f"列{i + 1}" for i, cell in enumerate(row)]
    seen: dict[str, int] = {}
    result = []
    for header in headers:
        count = seen.get(header, 0) + 1
        seen[header] = count
        result.append(header if count == 1 else f"{header}_{count}")
    return result


def trim_empty_edges(rows: list[list[str]]) -> list[list[str]]:
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        return []
    max_cols = max(len(row) for row in rows)
    keep_cols = [
        idx
        for idx in range(max_cols)
        if any(idx < len(row) and row[idx].strip() for row in rows)
    ]
    return [[row[idx] if idx < len(row) else "" for idx in keep_cols] for row in rows]


def clean_text(value: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", str(value or "").replace("\u3000", " ")).strip()


def truncate_cell(value: str, limit: int) -> str:
    value = clean_text(value)
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def escape_table_cell(value: str) -> str:
    return clean_text(value).replace("|", "\\|").replace("\n", "<br>")


def generate_media_feedback(
    *,
    base_dir: Path,
    product_name: str,
    meeting_context: str,
    questionnaire_text: str,
    price_analysis_markdown: str = "",
    reference_feedback: str = "",
    additional_instruction: str = "",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    model_config = resolve_model(base_dir, "writing")
    api_key = model_config["api_key"].strip()
    if not api_key:
        raise RuntimeError(f"没有配置 {model_config.get('name', model_config.get('model', '模型'))} 的 API Key。")
    prompt = build_media_feedback_prompt(
        product_name=product_name,
        meeting_context=meeting_context,
        questionnaire_text=questionnaire_text,
        price_analysis_markdown=price_analysis_markdown,
        reference_feedback=reference_feedback,
        additional_instruction=additional_instruction,
    )
    result = call_writing_model_streaming(
        api_key=api_key,
        base_url=model_config["base_url"].rstrip("/"),
        model=model_config["model"].strip(),
        provider=model_config["provider"].strip(),
        prompt=prompt,
        max_tokens=22000,
        stage="媒体反馈",
        system_prompt=FEEDBACK_SYSTEM,
        progress_callback=progress_callback,
    )
    result = sanitize_portable_markdown(result)
    model = model_config["model"].strip()
    warnings = sorted(set(INTERNAL_CODENAME_PATTERN.findall(result)))
    if warnings:
        result += "\n\n> 自检提醒：输出中仍包含疑似内部代号：" + "、".join(warnings)
    if price_analysis_markdown.strip():
        price_warnings = find_unmatched_price_mentions(result, price_analysis_markdown)
        if price_warnings:
            result += "\n\n> 价格口径自检提醒：正文中出现了未在确定性价格计算结果中匹配到的疑似价格数字，请二次加工时核对或删除：" + "、".join(price_warnings[:20])
            warnings.extend(f"price:{item}" for item in price_warnings)
        result = append_price_analysis(result, price_analysis_markdown)
    return result.strip(), {"model": model, "provider": model_config["provider"].strip(), "warnings": warnings}


def generate_review_video_analysis(
    *,
    base_dir: Path,
    product_name: str,
    media_name: str,
    video_title: str,
    video_url: str = "",
    transcript_text: str,
    additional_instruction: str = "",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    domain: DomainProfile | None = None,
) -> tuple[str, dict[str, Any]]:
    domain = domain or load_domain("generic")
    review_system = (
        f"你是{domain.analyst_role}，负责分析媒体/KOL视频逐字稿。"
        "判断整体倾向、提炼正负面观点、验证事实边界，并识别可供业务汇报和传播引用的原话。"
        f"领域要求：{domain.guidance}"
        "所有判断必须来自逐字稿，不得编造参数、测试结果或观点；短摘必须保持原意并可追溯。"
    )
    model_config = resolve_model(base_dir, "writing")
    api_key = model_config["api_key"].strip()
    if not api_key:
        raise RuntimeError(f"没有配置 {model_config.get('name', model_config.get('model', '模型'))} 的 API Key。")

    chunks = split_transcript_chunks(transcript_text, 24000)
    if not chunks:
        raise RuntimeError("没有获取到有效逐字稿。")
    chunk_notes: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        stage = f"逐字稿理解 {index}/{len(chunks)}"
        note = call_writing_model_streaming(
            api_key=api_key,
            base_url=model_config["base_url"].rstrip("/"),
            model=model_config["model"].strip(),
            provider=model_config["provider"].strip(),
            prompt=build_review_video_chunk_prompt(
                product_name=product_name,
                media_name=media_name,
                video_title=video_title,
                chunk_index=index,
                chunk_count=len(chunks),
                transcript_chunk=chunk,
                additional_instruction=additional_instruction,
                domain=domain,
            ),
            max_tokens=6000,
            stage=stage,
            system_prompt=review_system,
            progress_callback=progress_callback,
        )
        chunk_notes.append(f"## 片段 {index}/{len(chunks)}\n{note.strip()}")

    result = call_writing_model_streaming(
        api_key=api_key,
        base_url=model_config["base_url"].rstrip("/"),
        model=model_config["model"].strip(),
        provider=model_config["provider"].strip(),
        prompt=build_review_video_final_prompt(
            product_name=product_name,
            media_name=media_name,
            video_title=video_title,
            video_url=video_url,
            chunk_notes="\n\n".join(chunk_notes),
            additional_instruction=additional_instruction,
            domain=domain,
        ),
        max_tokens=14000,
        stage="评测视频分析",
        system_prompt=review_system,
        progress_callback=progress_callback,
    )
    result = sanitize_portable_markdown(result)
    warnings = sorted(set(INTERNAL_CODENAME_PATTERN.findall(result)))
    if warnings:
        result += "\n\n> 自检提醒：输出中仍包含疑似内部代号：" + "、".join(warnings)
    return result.strip(), {
        "model": model_config["model"].strip(),
        "provider": model_config["provider"].strip(),
        "chunk_count": len(chunks),
        "warnings": warnings,
        "domain": domain.id,
    }


def sanitize_portable_markdown(markdown: str) -> str:
    """Keep generated reports portable for Obsidian and plain Markdown editors."""
    text = str(markdown or "").strip()
    match = MARKDOWN_FENCE_PATTERN.match(text)
    if match:
        text = match.group("body").strip()

    text = re.sub(r"^\s*</?callout\b[^>]*>\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"<text\b[^>]*>(.*?)</text>", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?text\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_transcript_chunks(text: str, limit: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    blocks = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        block_len = len(block) + 2
        if current and current_len + block_len > limit:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        if block_len > limit:
            for start in range(0, len(block), limit):
                chunks.append(block[start : start + limit])
            continue
        current.append(block)
        current_len += block_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text[:limit]]


def build_media_feedback_prompt(
    *,
    product_name: str,
    meeting_context: str,
    questionnaire_text: str,
    price_analysis_markdown: str = "",
    reference_feedback: str = "",
    additional_instruction: str = "",
) -> str:
    reference_block = ""
    if reference_feedback.strip():
        reference_block = f"""

【参考反馈文档风格】
以下是用户认可的历史媒体反馈产物。只学习它们的信息架构、归纳密度、引用方式和判断口径，不要照抄具体结论。
{reference_feedback.strip()}
"""
    price_block = ""
    if price_analysis_markdown.strip():
        price_block = f"""

【价格预期确定性计算结果】
以下价格统计由程序从原始问卷中抽取并计算。你必须以这份结果为全文唯一价格口径，不要自行重算、改写样本数或创造新的均值/中位数/最高最低价。
全文所有出现价格统计的地方，包括概述、价格预期章节、风险提示、传播建议和附录，必须与这份结果保持一致。
正文可以解释价格含义，但不得出现与计算结果冲突的样本数、平均数、中位数、最低价、最高价、价位区间或价格结论。
多价格表达和相对价格已在对照表中标记为“待判断口径”，不得把它们私自折算进均值/中位数。
文档末尾必须保留“价格预期确定性计算”附录。
{price_analysis_markdown.strip()}
"""
    instruction_block = f"\n【用户补充要求】\n{additional_instruction.strip()}\n" if additional_instruction.strip() else ""
    return f"""请把【媒体问卷原始材料】整理成一份可直接给市场、产品、传播团队阅读的《媒体反馈》Markdown 文档。

产品/项目：{product_name or "未填写"}
沟通会背景：{meeting_context or "未填写"}
{instruction_block}
写作目标：
1. 详实但简练：只保留有判断价值的信息，删除重复口水话。
2. 有信息量：每个结论都尽量说明“多少媒体/哪类媒体/哪些代表媒体/为什么重要”。
3. 有证据：关键结论后附 2-5 条媒体原话，格式为“媒体名：原话摘录”。
4. 有判断：区分好评、槽点、风险、价格预期、传播建议，不把原始回答机械堆叠。
5. 有口径：价格、购买倾向、功能排序等能量化就量化；不能量化则说明原因。
6. 有风险意识：少数但高价值的尖锐意见不能因频次低被淹没，需进入“风险提示/待跟进”。
7. 尊重事实：不得编造媒体、样本数、价格均值、百分比、产品参数或原文不存在的观点。
8. 价格口径一致：如提供【价格预期确定性计算结果】，全文所有价格统计必须以计算结果为准；不允许正文和附录出现不同的均值、中位数、最低/最高、样本数。

请优先输出以下结构，可根据问卷主题增删产品分节：
## 概述
- 整体评价：一句话判断媒体情绪与核心分歧。
- 👍 好评点：3-5 条，按重要性排序。
- 👎 吐槽点：3-5 条，按风险和频次排序。
- 💰 价格预期：均价/众数/区间/分布，注明有效样本口径。
- ⚠️ 风险提示：传播、产品、舆情、定价或版本策略风险。

# 媒体评价详情
## 好评点
每个观点用“加粗观点句 + 解释 + 代表原话”的方式写。观点句要像结论，不要只是关键词。

## 槽点/争议
把高频吐槽和低频高风险意见分开写；需要说明它们可能影响的是口碑、评测风向、价格接受度还是产品定位。

## 价格预期和购买倾向
如原表有价格字段，必须优先使用【价格预期确定性计算结果】里的有效样本数、平均数、中位数、最低/最高和媒体对照表；不要让模型自行推理价格数字。若没有确定性计算结果，再说明“未识别到可计算价格字段”。
写完后自检：正文中每一个价格统计数字都必须能在【价格预期确定性计算结果】中找到对应口径；找不到就删除或改成“待人工判断”。

## 传播建议/产品待跟进
把媒体建议转成可执行项：传播怎么讲、评测怎么引导、哪些功能需要补证据、哪些争议需要 FAQ。

# 附：媒体反馈原文摘要
不要完整复刻所有原文，只保留可追溯摘要。按媒体列出最有代表性的反馈。

格式要求：
- 输出纯中文 Markdown，面向 Obsidian、GitHub、Typora 等通用 Markdown 工具。
- 不要把全文包在 ```markdown 或任何代码块里。
- 禁止输出任何 HTML/XML/飞书专用标签，包括但不限于 <callout>、</callout>、<text>、</text>、<br>、<font>、<span>、<div>。
- 如需强调，请只使用 Markdown 原生语法：# 标题、- 列表、**加粗**、> 引用、表格。
- 媒体原话要短摘，避免大段搬运。
- 如果提供了【价格预期确定性计算结果】，最终文档最后必须附上同名附录，包含统计摘要和“媒体价格预期对照表”。
- 如果问卷含多个产品，分别成章，但开头概述必须做总判断。
- 如果某些问题样本不足，写“样本不足，不建议下结论”。
{reference_block}
{price_block}

【媒体问卷原始材料】
{questionnaire_text.strip()}
"""


def append_price_analysis(result: str, price_analysis_markdown: str) -> str:
    result = sanitize_portable_markdown(result)
    price_analysis_markdown = price_analysis_markdown.strip()
    if not price_analysis_markdown:
        return result
    result = re.sub(r"\n*## 价格预期确定性计算[\s\S]*$", "", result).strip()
    return result + "\n\n" + price_analysis_markdown


def find_unmatched_price_mentions(result: str, price_analysis_markdown: str) -> list[str]:
    allowed = {str(value) for value in extract_price_numbers(price_analysis_markdown)}
    allowed.update(re.findall(r"(?<!\d)\d{1,3}(?!\d)", price_analysis_markdown))
    body = re.sub(r"\n*## 价格预期确定性计算[\s\S]*$", "", result).strip()
    mentions: list[str] = []
    price_patterns = [
        r"(?<!\d)(\d{3,5})(?!\d)\s*元",
        r"(?:均价|平均数|中位数|最低|最高|众数|价格|定价|售价|起售价)[^\n]{0,12}(?<!\d)(\d{3,5})(?!\d)",
        r"(?<!\d)(\d+(?:\.\d+)?)\s*[Kk]",
        r"(?<!\d)(\d+(?:\.\d+)?)\s*万",
    ]
    for pattern in price_patterns:
        for match in re.finditer(pattern, body):
            raw = match.group(1)
            normalized_values = extract_price_numbers(match.group(0))
            normalized = [str(value) for value in normalized_values] or [raw]
            for value in normalized:
                if value not in allowed and value not in mentions:
                    mentions.append(value)
    return mentions


def build_review_video_chunk_prompt(
    *,
    product_name: str,
    media_name: str,
    video_title: str,
    chunk_index: int,
    chunk_count: int,
    transcript_chunk: str,
    additional_instruction: str = "",
    domain: DomainProfile | None = None,
) -> str:
    domain = domain or load_domain("generic")
    instruction_block = f"\n【用户补充要求】\n{additional_instruction.strip()}\n" if additional_instruction.strip() else ""
    return f"""请阅读下面这段评测视频逐字稿片段，提取可用于最终汇总的事实证据。

产品/项目：{product_name or "未填写"}
媒体/账号：{media_name or "未填写"}
视频标题：{video_title or "未填写"}
业务领域：{domain.name}
领域关注：{domain.guidance}
片段：{chunk_index}/{chunk_count}
{instruction_block}

输出要求：
1. 只基于本片段，不要借外部知识补全。
2. 判断本片段对产品的情绪：正面 / 负面 / 中性 / 混合，并说明理由。
3. 提炼本片段出现的正面观点、负面观点、争议观点，每条都附原文短摘。
4. 提取可传播引用的“金句”：必须是媒体原话短摘，不要改写成品牌话术。
5. 如果逐字稿有 SRT 时间码，请尽量保留时间线索；没有时间码则写“片段 {chunk_index}”。
6. 不要输出最终报告，只输出本片段结构化笔记。

建议格式：
## 片段情绪

## 正面观点
- 观点：...
  原文：...

## 负面/争议观点
- 观点：...
  原文：...

## 可引用金句
- “...”｜时间线索：...

【逐字稿片段】
{transcript_chunk.strip()}
"""


def build_review_video_final_prompt(
    *,
    product_name: str,
    media_name: str,
    video_title: str,
    video_url: str,
    chunk_notes: str,
    additional_instruction: str = "",
    domain: DomainProfile | None = None,
) -> str:
    domain = domain or load_domain("generic")
    instruction_block = f"\n【用户补充要求】\n{additional_instruction.strip()}\n" if additional_instruction.strip() else ""
    return f"""请基于【逐字稿分段理解笔记】，生成一份《媒体评测视频分析》Markdown 文档，供老板快速了解媒体评测情况，并供后续传播引用媒体内容。

产品/项目：{product_name or "未填写"}
媒体/账号：{media_name or "未填写"}
视频标题：{video_title or "未填写"}
视频链接：{video_url or "未填写"}
业务领域：{domain.name}
领域关注：{domain.guidance}
{instruction_block}

写作目标：
1. 明确判断该条评测视频整体情感倾向：正面 / 偏正面 / 中性 / 偏负面 / 负面，并给出一句话总判断。
2. 说明判断依据：哪些核心观点、哪些用词、哪些测试/体验描述支撑这个倾向。
3. 分别提炼主要正面观点和主要负面/争议观点，按对传播与业务的重要性排序。
4. 原文摘录媒体评测金句，用于老板呈报和传播引用。金句必须短、准、可追溯，不要大段搬运。
5. 给出传播可用性建议：哪些话可以引用，哪些需要谨慎，哪些负面点需要 FAQ 或后续沟通。
6. 不得编造逐字稿中没有的观点、参数、测试结论、媒体身份或外部背景。

请输出以下结构：
# 媒体评测视频分析

## 概述
- 整体倾向：
- 一句话判断：
- 老板可看要点：
- 传播可用性：

## 视频信息
- 产品/项目：
- 媒体/账号：
- 视频标题：
- 视频链接：

## 情感倾向判断
用 2-4 段说明该视频为什么是正面或负面。需要区分“口播语气好”和“结论真正推荐”的差别。

## 主要正面观点
每条包含：观点结论、为什么重要、原文短摘。

## 主要负面/争议观点
每条包含：问题结论、可能影响、原文短摘、建议回应方式。

## 可传播引用金句
用表格输出：金句原文、推荐用途、时间线索、引用风险。只放最值得用的 5-10 条。

## 对外传播建议
把可引用内容转成传播动作建议；如有负面内容，说明是否需要避免主动扩散。

## 附：证据摘要
保留可追溯的片段级摘要，不要完整复刻逐字稿。

格式要求：
- 输出纯中文 Markdown，面向 Obsidian、GitHub、Typora 等通用 Markdown 工具。
- 不要把全文包在 ```markdown 或任何代码块里。
- 禁止输出任何 HTML/XML/飞书专用标签，包括但不限于 <callout>、</callout>、<text>、</text>、<br>、<font>、<span>、<div>。
- 如需强调，请只使用 Markdown 原生语法：# 标题、- 列表、**加粗**、> 引用、表格。

【逐字稿分段理解笔记】
{chunk_notes.strip()}
"""
