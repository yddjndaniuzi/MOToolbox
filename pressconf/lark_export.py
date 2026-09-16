from __future__ import annotations

import json
import http.client
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import base64
from html import escape
from datetime import datetime
from pathlib import Path
from typing import Any

from pressconf.config_store import load_lark_config, write_json
from pressconf.runtime import certifi_ssl_context


LOCAL_IMAGE_RE = re.compile(r"^\s*!\[[^\]]*\]\((frames/[^)]+)\)\s*$")
MCP_TASK_POLL_INTERVAL_SECONDS = 3
MCP_TASK_POLL_TIMEOUT_SECONDS = 120


def lark_cli_path() -> str:
    return (
        shutil.which("lark-cli")
        or "/Users/mi/.npm-global/bin/lark-cli"
    )


def lark_status(base_dir: Path) -> dict[str, Any]:
    config = load_lark_config(base_dir)
    if config.get("transport") == "mcp_http":
        url = str(config.get("mcp_url") or "").strip()
        return {
            "transport": "mcp_http",
            "configured": bool(url),
            "message": "MCP URL 已配置" if url else "等待配置 MCP URL",
            "config": config,
        }

    cli = lark_cli_path()
    cli_exists = Path(cli).exists()
    configured = False
    message = "未找到 lark-cli"
    if cli_exists:
        message = "lark-cli 已安装，尚未验证授权"
        try:
            result = subprocess.run(
                [
                    cli,
                    "docs",
                    "+create",
                    "--api-version",
                    "v2",
                    "--as",
                    str(config.get("identity") or "user"),
                    "--doc-format",
                    "markdown",
                    "--content",
                    "# MOtoolbox 状态检查",
                    "--dry-run",
                ],
                cwd=base_dir,
                capture_output=True,
                text=True,
                timeout=4,
            )
            status_payload = parse_lark_output(result.stdout or result.stderr)
            configured = result.returncode == 0 and status_payload.get("ok", True) is not False
            message = "飞书授权可用" if configured else parse_status_message(result.stderr or result.stdout)
        except subprocess.TimeoutExpired:
            message = "飞书授权检查超时"
    return {
        "cli_path": cli,
        "cli_exists": cli_exists,
        "configured": configured,
        "message": message,
        "config": config,
    }


def export_brief_to_lark(
    *,
    base_dir: Path,
    result_dir: Path,
    display_name: str,
    variant: str = "refined",
    dry_run: bool = False,
) -> dict[str, Any]:
    source_path = brief_source_path(result_dir, variant)
    if not source_path.exists():
        raise RuntimeError("还没有可写入飞书的简报 Markdown。")

    return export_markdown_to_lark(
        base_dir=base_dir,
        result_dir=result_dir,
        source_path=source_path,
        display_name=display_name,
        metadata={"variant": "refined" if source_path.name in {"brief_refined.md", "brief_current.md"} else "base"},
        dry_run=dry_run,
    )


def export_markdown_to_lark(
    *,
    base_dir: Path,
    result_dir: Path,
    source_path: Path,
    display_name: str,
    export_stem: str = "lark_export",
    metadata: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create a Lark document from an arbitrary local Markdown file."""
    config = load_lark_config(base_dir)
    if not config.get("enabled", True):
        raise RuntimeError("飞书写入未启用，请先在后台开启绑定。")
    if not source_path.exists():
        raise RuntimeError("还没有可写入飞书的 Markdown。")

    markdown = source_path.read_text(encoding="utf-8")
    image_urls = resolve_image_urls(markdown, result_dir, config, dry_run=dry_run)
    export_markdown = prepare_lark_markdown(
        markdown,
        display_name,
        result_slug=result_dir.name,
        image_url_base=str(config.get("image_url_base") or ""),
        image_urls=image_urls,
        image_grid_columns=parse_grid_columns(config.get("image_grid_columns")),
    )
    export_path = result_dir / f"{export_stem}.md"
    export_path.write_text(export_markdown, encoding="utf-8")

    if config.get("transport") == "mcp_http":
        meta = export_via_mcp_http(
            base_dir=base_dir,
            result_dir=result_dir,
            display_name=display_name,
            source_path=source_path,
            export_path=export_path,
            markdown=export_markdown,
            config=config,
            dry_run=dry_run,
        )
        meta.update(metadata or {})
        if not dry_run:
            write_json(result_dir / f"{export_stem}.json", meta)
        return meta

    cli = lark_cli_path()
    if not Path(cli).exists():
        raise RuntimeError("没有找到 lark-cli，请先完成飞书 MCP/CLI 环境安装。")

    command = [
        cli,
        "docs",
        "+create",
        "--api-version",
        "v2",
        "--as",
        str(config.get("identity") or "user"),
        "--doc-format",
        "markdown",
        "--content",
        f"@{export_path}",
    ]
    command.extend(target_args(config))
    if dry_run:
        command.append("--dry-run")

    result = subprocess.run(command, cwd=base_dir, capture_output=True, text=True)
    if result.returncode != 0:
        error_text = result.stderr or result.stdout or "飞书文档创建失败"
        if "not configured" in error_text:
            raise RuntimeError("飞书 CLI 尚未授权：请先在本机运行 `lark-cli config init --new`，按提示打开验证链接完成绑定。")
        raise RuntimeError(error_text.strip())

    payload = parse_lark_output(result.stdout)
    normalized = normalize_mcp_response(payload)
    meta = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "title": display_name,
        "source": source_path.name,
        "export_markdown": export_path.name,
        "command": redact_command(command),
        "lark": payload,
        "doc_id": find_key(normalized, {"doc_id", "document_id", "token"}),
        "url": find_key(normalized, {"doc_url", "url", "document_url", "share_url"}) or find_url(normalized, result.stdout),
        "raw_stdout": result.stdout.strip(),
    }
    meta.update(metadata or {})
    if not dry_run:
        write_json(result_dir / f"{export_stem}.json", meta)
    return meta


def brief_source_path(result_dir: Path, variant: str) -> Path:
    current_path = result_dir / "brief_current.md"
    if variant in {"refined", "current"} and current_path.exists():
        return current_path
    if variant == "refined" and (result_dir / "brief_refined.md").exists():
        return result_dir / "brief_refined.md"
    return result_dir / "brief_base.md"


def export_via_mcp_http(
    *,
    base_dir: Path,
    result_dir: Path,
    display_name: str,
    source_path: Path,
    export_path: Path,
    markdown: str,
    config: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    url = str(config.get("mcp_url") or "").strip()
    if not url:
        raise RuntimeError("请先在后台填写飞书 MCP URL。")

    headers = parse_json_object(str(config.get("mcp_headers") or "{}"), "Headers JSON")
    arguments_template = parse_json_object(str(config.get("mcp_arguments") or "{}"), "参数 JSON")
    arguments = replace_placeholders(
        arguments_template,
        {
            "title": display_name,
            "markdown": markdown,
            "source": source_path.name,
            "slug": result_dir.name,
        },
    )
    replacements = {
        "title": display_name,
        "markdown": markdown,
        "source": source_path.name,
        "slug": result_dir.name,
        "tool": str(config.get("mcp_tool") or "docs_create").strip(),
        "method": str(config.get("mcp_method") or "tools/call").strip(),
        "arguments": json.dumps(arguments, ensure_ascii=False),
    }
    payload_template = normalize_payload_template(str(config.get("mcp_payload") or "").strip())
    if payload_template:
        payload = parse_json_object(replace_placeholders(payload_template, replacements), "完整 Payload JSON")
    else:
        payload = {
            "jsonrpc": "2.0",
            "id": "motoolbox-lark-export",
            "method": replacements["method"],
            "params": {
                "name": replacements["tool"],
                "arguments": arguments,
            },
        }
    meta = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "title": display_name,
        "variant": "refined" if source_path.name == "brief_refined.md" else "base",
        "source": source_path.name,
        "export_markdown": export_path.name,
        "transport": "mcp_http",
        "mcp_url": url,
        "mcp_method": payload.get("method", ""),
        "mcp_tool": str(config.get("mcp_tool") or "docs_create").strip(),
        "request_payload": payload,
    }
    if dry_run:
        return meta

    response = post_json(url, payload, headers)
    if response.get("error"):
        raise RuntimeError(f"MCP 写入失败：{json.dumps(response['error'], ensure_ascii=False)}")
    response, attempts = resolve_mcp_task_result(url, payload, headers, response)
    normalized = normalize_mcp_response(response)
    fields = extract_lark_export_fields(normalized)
    meta["lark"] = response
    meta["mcp_attempts"] = attempts
    meta["doc_id"] = fields["doc_id"]
    meta["url"] = fields["url"] or find_url(normalized, json.dumps(normalized, ensure_ascii=False))
    if not meta["url"] and fields["task_id"] and fields["status"] in {"running", "pending", "processing"}:
        raise RuntimeError("飞书文档仍在后台生成中，暂未返回文档链接，请稍后重试写入。")
    return meta


def prepare_lark_markdown(
    markdown: str,
    display_name: str,
    *,
    result_slug: str = "",
    image_url_base: str = "",
    image_urls: dict[str, str] | None = None,
    image_grid_columns: int = 4,
) -> str:
    image_urls = image_urls or {}
    lines = []
    pending_images: list[str] = []

    def flush_images() -> None:
        if not pending_images:
            return
        lines.extend(render_image_grid(pending_images, image_grid_columns))
        pending_images.clear()

    for line in markdown.splitlines():
        if "关于图片" in line:
            continue
        match = LOCAL_IMAGE_RE.match(line)
        if match:
            image_path = match.group(1)
            image_url = image_urls.get(image_path) or public_image_url(image_url_base, result_slug, image_path)
            if image_url:
                pending_images.append(image_url)
        else:
            flush_images()
            lines.append(line)
    flush_images()

    content = "\n".join(lines).strip()
    if not content.startswith("# "):
        content = f"# {display_name}\n\n{content}"
    return content.rstrip() + "\n"


def parse_grid_columns(value: Any) -> int:
    try:
        columns = int(value or 4)
    except (TypeError, ValueError):
        columns = 4
    return min(4, max(1, columns))


def render_image_grid(image_urls: list[str], columns: int) -> list[str]:
    if columns <= 1 or len(image_urls) == 1:
        return [f'<image url="{escape(url, quote=True)}" align="center"/>' for url in image_urls]

    output: list[str] = []
    for index in range(0, len(image_urls), columns):
        row = image_urls[index : index + columns]
        if len(row) == 1:
            output.append(f'<image url="{escape(row[0], quote=True)}" align="center"/>')
            continue
        output.append(f'<grid cols="{len(row)}">')
        for url in row:
            output.append("<column>")
            output.append("")
            output.append(f'<image url="{escape(url, quote=True)}" align="center"/>')
            output.append("")
            output.append("</column>")
        output.append("</grid>")
    return output


def resolve_image_urls(markdown: str, result_dir: Path, config: dict[str, Any], dry_run: bool = False) -> dict[str, str]:
    image_paths = collect_image_paths(markdown)
    if not image_paths:
        return {}
    if str(config.get("image_url_base") or "").strip():
        return {}
    provider = str(config.get("image_upload_provider") or "none").strip()
    if provider != "imgbb":
        raise RuntimeError("简报里包含配图，但尚未配置图片写入方式。请在 http://127.0.0.1:5058/admin 选择 ImgBB 并填写 API Key，或填写图片公共 URL 前缀。")
    api_key = str(config.get("imgbb_api_key") or "").strip()
    if not api_key:
        raise RuntimeError("已选择 ImgBB 图床，但后台没有填写 ImgBB API Key。")
    if dry_run:
        return {path: f"https://example.invalid/{result_dir.name}/{path}" for path in image_paths}

    cache_path = result_dir / "image_uploads.json"
    cache = read_json_file(cache_path, {"uploads": {}})
    uploads = cache.setdefault("uploads", {})
    resolved: dict[str, str] = {}
    for image_path in image_paths:
        cached = uploads.get(image_path) or {}
        if cached.get("provider") == "imgbb" and cached.get("url"):
            resolved[image_path] = cached["url"]
            continue
        local_path = result_dir / image_path
        if not local_path.exists():
            continue
        try:
            url = upload_imgbb(local_path, api_key=api_key, expiration=str(config.get("imgbb_expiration") or "604800"))
        except Exception as exc:
            raise RuntimeError(f"写入飞书前上传图片失败（{image_path}）：{exc}。已成功上传的图片已保存，可稍后重试。") from exc
        uploads[image_path] = {
            "provider": "imgbb",
            "url": url,
            "uploaded_at": datetime.now().isoformat(timespec="seconds"),
        }
        resolved[image_path] = url
        write_json(cache_path, cache)
    return resolved


def collect_image_paths(markdown: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for line in markdown.splitlines():
        match = LOCAL_IMAGE_RE.match(line)
        if match and match.group(1) not in seen:
            paths.append(match.group(1))
            seen.add(match.group(1))
    return paths


def upload_imgbb(path: Path, *, api_key: str, expiration: str = "604800") -> str:
    image_data = base64.b64encode(path.read_bytes()).decode("ascii")
    form = {
        "image": image_data,
        "name": path.stem,
    }
    query = {"key": api_key}
    if expiration:
        query["expiration"] = expiration
    url = "https://api.imgbb.com/1/upload?" + urllib.parse.urlencode(query)
    data = urllib.parse.urlencode(form).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60, context=certifi_ssl_context()) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            break
        except urllib.error.HTTPError as exc:
            # Authentication and other permanent HTTP failures must not be retried.
            if exc.code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"ImgBB 图片上传失败 HTTP {exc.code}") from exc
            if attempt == 2:
                raise RuntimeError(f"ImgBB 图片上传失败 HTTP {exc.code}，已尝试 3 次") from exc
        except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException) as exc:
            if attempt == 2:
                raise RuntimeError("连接 ImgBB 图床时中断或超时，已尝试 3 次，请检查网络或代理") from exc
        time.sleep(2 ** attempt)
    if not payload.get("success"):
        raise RuntimeError(f"ImgBB 上传失败：{json.dumps(payload, ensure_ascii=False)}")
    data_node = payload.get("data") or {}
    image_url = data_node.get("display_url") or data_node.get("url")
    if not image_url:
        raise RuntimeError(f"ImgBB 未返回图片 URL：{json.dumps(payload, ensure_ascii=False)}")
    return str(image_url)


def read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def public_image_url(image_url_base: str, result_slug: str, image_path: str) -> str:
    base = image_url_base.strip().rstrip("/")
    if not base:
        return ""
    if base.endswith(f"/{result_slug}"):
        return f"{base}/{image_path}"
    return f"{base}/{result_slug}/{image_path}"


def target_args(config: dict[str, Any]) -> list[str]:
    target_type = config.get("target_type")
    if target_type == "folder" and config.get("folder_token"):
        return ["--parent-token", str(config["folder_token"])]
    if target_type == "wiki":
        if config.get("wiki_node"):
            return ["--parent-token", str(config["wiki_node"])]
    return []


def parse_lark_output(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {}
    if text.startswith("event:") or "\ndata:" in text:
        data_lines = []
        for line in text.splitlines():
            if line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        if data_lines:
            text = "\n".join(data_lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    candidates = [line for line in text.splitlines() if line.strip().startswith("{")]
    for candidate in reversed(candidates or [text]):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return {"text": text}


def normalize_mcp_response(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    extracted: list[Any] = []
    for text in iter_text_values(payload):
        parsed = try_parse_json(text)
        if isinstance(parsed, dict):
            extracted.append(parsed)
    if extracted:
        normalized["_extracted"] = extracted
    return normalized


def iter_text_values(value: Any):
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            yield value["text"]
        for item in value.values():
            yield from iter_text_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_text_values(item)


def try_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_json_object(text: str, label: str) -> dict[str, Any]:
    try:
        data = json.loads(text or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} 不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{label} 必须是 JSON object。")
    return data


def normalize_payload_template(text: str) -> str:
    if not text:
        return ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(data, dict) and "mcpServers" in data:
        return ""
    return text


def replace_placeholders(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for key, replacement in replacements.items():
            value = value.replace("{{" + key + "}}", replacement)
        return value
    if isinstance(value, list):
        return [replace_placeholders(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: replace_placeholders(item, replacements) for key, item in value.items()}
    return value


def post_json(url: str, payload: dict[str, Any], headers: dict[str, Any]) -> dict[str, Any]:
    request_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": "2024-11-05",
    }
    request_headers.update({str(key): str(value) for key, value in headers.items()})
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=90, context=certifi_ssl_context()) as response:
            text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MCP HTTP {exc.code}：{text}") from exc
    except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException) as exc:
        raise RuntimeError("连接飞书 MCP 时中断或超时，未能确认写入结果。请先检查飞书中是否已生成文档，避免重复创建。") from exc
    return parse_lark_output(text)


def resolve_mcp_task_result(
    url: str,
    original_payload: dict[str, Any],
    headers: dict[str, Any],
    first_response: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    response = first_response
    attempts = [response]
    deadline = time.monotonic() + MCP_TASK_POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        normalized = normalize_mcp_response(response)
        fields = extract_lark_export_fields(normalized)
        if fields["url"] or not fields["task_id"] or fields["status"] not in {"running", "pending", "processing"}:
            return response, attempts
        time.sleep(MCP_TASK_POLL_INTERVAL_SECONDS)
        response = post_json(url, mcp_task_poll_payload(original_payload, fields["task_id"]), headers)
        attempts.append(response)
        if response.get("error"):
            raise RuntimeError(f"MCP 写入失败：{json.dumps(response['error'], ensure_ascii=False)}")
    return response, attempts


def mcp_task_poll_payload(original_payload: dict[str, Any], task_id: str) -> dict[str, Any]:
    params = original_payload.get("params") if isinstance(original_payload.get("params"), dict) else {}
    return {
        "jsonrpc": str(original_payload.get("jsonrpc") or "2.0"),
        "id": f"{original_payload.get('id') or 'motoolbox-lark-export'}-poll",
        "method": str(original_payload.get("method") or "tools/call"),
        "params": {
            "name": str(params.get("name") or ""),
            "arguments": {"task_id": task_id},
        },
    }


def extract_lark_export_fields(normalized: dict[str, Any]) -> dict[str, str]:
    return {
        "doc_id": find_key(normalized, {"doc_id", "document_id", "token"}),
        "url": find_key(normalized, {"doc_url", "url", "document_url", "share_url"})
        or find_url(normalized, json.dumps(normalized, ensure_ascii=False)),
        "task_id": find_key(normalized, {"task_id"}),
        "status": find_key(normalized, {"status"}).lower(),
    }


def parse_status_message(text: str) -> str:
    payload = parse_lark_output(text)
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        hint = str(error.get("hint") or "").strip()
        if message == "not configured":
            return "尚未完成飞书授权"
        return hint or message or "飞书授权不可用"
    return text.strip()[:160] or "飞书授权不可用"


def find_url(payload: dict[str, Any], stdout: str) -> str:
    stack: list[Any] = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if key in {"url", "document_url", "share_url"} and isinstance(value, str) and value.startswith("http"):
                    return value
                stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    match = re.search(r"https?://\S+", stdout)
    if not match:
        return ""
    return match.group(0).rstrip('"}],')


def find_key(payload: Any, keys: set[str]) -> str:
    stack: list[Any] = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if key in keys and isinstance(value, str) and value:
                    return value
                stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return ""


def redact_command(command: list[str]) -> list[str]:
    return list(command)
