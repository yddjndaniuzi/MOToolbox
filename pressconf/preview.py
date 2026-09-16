from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PreviewBlock:
    index: int
    kind: str
    html: str
    raw: str
    start_line: int
    end_line: int


def render_markdown_preview(markdown: str, slug: str) -> list[PreviewBlock]:
    blocks: list[PreviewBlock] = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and set(lines[index + 1].strip()) <= {"|", "-", ":", " "}:
            table_lines = [line]
            index += 1
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            raw = "\n".join(table_lines)
            blocks.append(PreviewBlock(len(blocks), "table", render_table(table_lines), raw, index - len(table_lines), index - 1))
            continue

        if stripped.startswith("#"):
            level = min(len(stripped) - len(stripped.lstrip("#")), 3)
            text = stripped[level:].strip()
            blocks.append(PreviewBlock(len(blocks), "heading", f"<h{level}>{inline_markdown(text, slug)}</h{level}>", line, index, index))
            index += 1
            continue

        if stripped.startswith("![]("):
            images = []
            raw_lines = []
            while index < len(lines) and lines[index].strip().startswith("![]("):
                raw_lines.append(lines[index])
                images.append(render_image(lines[index].strip(), slug))
                index += 1
            blocks.append(PreviewBlock(len(blocks), "images", "<div class=\"doc-images\">" + "".join(images) + "</div>", "\n".join(raw_lines), index - len(raw_lines), index - 1))
            continue

        if stripped.startswith("- "):
            items = []
            raw_lines = []
            while index < len(lines) and lines[index].strip().startswith("- "):
                raw_line = lines[index]
                raw_lines.append(raw_line)
                text = raw_line.strip()[2:].strip()
                index += 1
                image_lines = []
                while index < len(lines) and lines[index].strip().startswith("![]("):
                    image_lines.append(lines[index].strip())
                    raw_lines.append(lines[index])
                    index += 1
                image_html = "<div class=\"doc-images inline-images\">" + "".join(render_image(item, slug) for item in image_lines) + "</div>" if image_lines else ""
                items.append(f"<li>{inline_markdown(text, slug)}{image_html}</li>")
            blocks.append(PreviewBlock(len(blocks), "list", "<ul>" + "".join(items) + "</ul>", "\n".join(raw_lines), index - len(raw_lines), index - 1))
            continue

        paragraph = [line]
        index += 1
        while index < len(lines) and lines[index].strip() and not starts_new_block(lines[index]):
            paragraph.append(lines[index])
            index += 1
        raw = "\n".join(paragraph)
        html_text = "<br>".join(inline_markdown(item.strip(), slug) for item in paragraph)
        blocks.append(PreviewBlock(len(blocks), "paragraph", f"<p>{html_text}</p>", raw, index - len(paragraph), index - 1))

    return blocks


def starts_new_block(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(("#", "- ", "![](", "|"))


def inline_markdown(text: str, slug: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"&lt;text color=&quot;([^&]+)&quot;&gt;(.*?)&lt;/text&gt;", r"<span class=\"doc-color doc-color-\1\">\2</span>", text)
    text = re.sub(r"&lt;br&gt;", "<br>", text)
    text = re.sub(r"!\[\]\(([^)]+)\)", lambda match: render_image(match.group(0), slug), text)
    return text


def render_image(markdown_image: str, slug: str) -> str:
    match = re.search(r"!\[\]\(([^)]+)\)", markdown_image)
    if not match:
        return ""
    path = match.group(1).strip()
    if path.startswith(("http://", "https://", "/")):
        src = path
    else:
        src = f"/raw/{slug}/{path}"
    name = html.escape(Path(path).name)
    escaped_path = html.escape(path)
    return f"<button class=\"doc-image\" type=\"button\" draggable=\"true\" data-image-path=\"{escaped_path}\"><img src=\"{html.escape(src)}\" alt=\"{name}\"><span>{name}</span><b>删除</b></button>"


def render_table(lines: list[str]) -> str:
    rows = []
    for line_index, line in enumerate(lines):
        if line_index == 1:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        tag = "th" if line_index == 0 else "td"
        rows.append("<tr>" + "".join(f"<{tag}>{inline_markdown(cell, '')}</{tag}>" for cell in cells) + "</tr>")
    return "<div class=\"doc-table-wrap\"><table>" + "".join(rows) + "</table></div>"
