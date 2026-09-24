from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable

from pressconf.runtime import certifi_ssl_context


OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openai-compatible", "deepseek"}
ANTHROPIC_VERSION = "2023-06-01"


class IncompleteGenerationError(RuntimeError):
    """The provider did not confirm a complete text response."""


def is_model_capacity_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "too many tokens per day",
            "rate_limit_error",
            "rate limit",
            "status code: 429",
            "返回错误：429",
            "overloaded_error",
        )
    )


def is_model_length_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "停止原因：length",
            "停止原因：max_tokens",
            "stop_reason: length",
            "stop_reason: max_tokens",
        )
    )


def validate_completion(reason: str | None, allowed: set[str]) -> None:
    if reason not in allowed:
        raise IncompleteGenerationError(
            f"模型输出未完整结束（停止原因：{reason or '连接提前结束/缺少停止原因'}）。"
            "草稿已保留，请重试；长内容应分段生成。"
        )


def call_chat_model(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.35,
    max_tokens: int = 12000,
    timeout: int = 300,
) -> str:
    provider = normalize_provider(provider)
    ensure_endpoint(base_url)
    if provider == "anthropic":
        return call_anthropic(
            api_key=api_key,
            base_url=base_url,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    return call_openai_compatible(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def stream_chat_model(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.35,
    max_tokens: int = 12000,
    timeout: int = 300,
    on_delta: Callable[[str], None] | None = None,
    on_metadata: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    provider = normalize_provider(provider)
    ensure_endpoint(base_url)
    if provider == "anthropic":
        return stream_anthropic(
            api_key=api_key,
            base_url=base_url,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            on_delta=on_delta,
            on_metadata=on_metadata,
        )
    return stream_openai_compatible(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        on_delta=on_delta,
        on_metadata=on_metadata,
    )


def normalize_provider(provider: str) -> str:
    provider = (provider or "openai-compatible").strip().lower()
    if provider in {"claude", "anthropic"} or provider.endswith("-anthropic"):
        return "anthropic"
    if provider in OPENAI_COMPATIBLE_PROVIDERS or provider.endswith(("-openai", "-openai-compatible")):
        return "openai-compatible"
    # Custom gateways are OpenAI-compatible unless their provider name
    # explicitly opts into the Anthropic protocol above.
    return "openai-compatible"


def ensure_endpoint(base_url: str) -> None:
    if not str(base_url or "").strip():
        raise RuntimeError("没有配置模型 Base URL。")


def call_openai_compatible(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    payload = openai_compatible_payload(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    data = post_json(
        f"{base_url.rstrip('/')}/chat/completions",
        payload,
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout,
        "OpenAI-compatible",
    )
    try:
        choices = data["choices"]
        if not choices:
            raise ValueError("empty choices")
        validate_completion(choices[0].get("finish_reason"), {"stop"})
        return str((choices[0].get("message") or {}).get("content") or "")
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise RuntimeError(f"OpenAI-compatible API 返回格式异常：{data}") from exc


def stream_openai_compatible(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
    on_delta: Callable[[str], None] | None,
    on_metadata: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    payload = openai_compatible_payload(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    chunks: list[str] = []
    reason = None
    usage: dict[str, Any] = {}
    terminal_seen = False
    malformed_event = False
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=certifi_ssl_context()) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    terminal_seen = True
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    malformed_event = True
                    continue
                maybe_error = payload.get("error")
                if maybe_error:
                    raise RuntimeError(f"OpenAI-compatible API 返回错误：{maybe_error}")
                usage.update(payload.get("usage") or {})
                choices = payload.get("choices") or []
                if not choices:
                    continue
                reason = choices[0].get("finish_reason") or reason
                delta = (choices[0].get("delta") or {}).get("content", "")
                if not delta:
                    continue
                chunks.append(str(delta))
                if on_delta:
                    on_delta(str(delta))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI-compatible API 返回错误：{exc.code} {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 OpenAI-compatible API：{exc.reason}") from exc
    if on_metadata:
        on_metadata({"stop_reason": reason, "usage": usage, "terminal_seen": terminal_seen})
    if malformed_event:
        raise IncompleteGenerationError("OpenAI-compatible 流包含无法解析的数据帧，输出可能不完整；草稿已保留，请重试。")
    validate_completion(reason, {"stop"})
    if not terminal_seen:
        raise IncompleteGenerationError("OpenAI-compatible 流缺少 [DONE] 终止帧，输出可能不完整；草稿已保留，请重试。")
    return "".join(chunks).strip()


def openai_compatible_payload(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    # Reasoning models exposed through OpenAI-compatible gateways commonly
    # reject non-default temperature values. Since temperature is optional,
    # omitting it preserves the provider/model default and works for both
    # reasoning and conventional chat models.
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }


def call_anthropic(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    payload = anthropic_payload(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    data = post_json(
        f"{base_url.rstrip('/')}/v1/messages",
        payload,
        anthropic_headers(api_key),
        timeout,
        "Anthropic",
    )
    validate_completion(data.get("stop_reason"), {"end_turn"})
    return extract_anthropic_text(data)


def stream_anthropic(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
    on_delta: Callable[[str], None] | None,
    on_metadata: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    payload = anthropic_payload(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/messages",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=anthropic_headers(api_key),
        method="POST",
    )
    chunks: list[str] = []
    reason = None
    usage: dict[str, Any] = {}
    terminal_seen = False
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=certifi_ssl_context()) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "error" or event.get("error"):
                    raise RuntimeError(f"Anthropic API 返回错误：{event.get('error') or event}")
                if event.get("type") == "message_start":
                    usage.update((event.get("message") or {}).get("usage") or {})
                usage.update(event.get("usage") or {})
                if event.get("type") == "message_stop":
                    terminal_seen = True
                delta = event.get("delta", {})
                reason = delta.get("stop_reason") or reason
                if delta.get("type") != "text_delta":
                    continue
                text = str(delta.get("text") or "")
                if not text:
                    continue
                chunks.append(text)
                if on_delta:
                    on_delta(text)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Anthropic API 返回错误：{exc.code} {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 Anthropic API：{exc.reason}") from exc
    if on_metadata:
        on_metadata({"stop_reason": reason, "usage": usage, "terminal_seen": terminal_seen})
    validate_completion(reason, {"end_turn"})
    if not terminal_seen:
        validate_completion(None, {"end_turn"})
    return "".join(chunks).strip()


def anthropic_payload(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    system_parts = [item["content"] for item in messages if item.get("role") == "system" and item.get("content")]
    user_messages = [
        {"role": "assistant" if item.get("role") == "assistant" else "user", "content": item.get("content", "")}
        for item in messages
        if item.get("role") != "system"
    ]
    payload: dict[str, Any] = {
        "model": model,
        "messages": user_messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    # Some Anthropic models exposed through Bedrock-compatible gateways reject
    # temperature entirely. It is optional in the Messages API, so omitting it
    # is the most portable behavior across native and proxied Anthropic models.
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    return payload


def anthropic_headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "Content-Type": "application/json",
    }


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int, label: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=certifi_ssl_context()) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"{label} API 返回错误：{exc.code} {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 {label} API：{exc.reason}") from exc


def extract_anthropic_text(data: dict[str, Any]) -> str:
    try:
        blocks = data["content"]
    except KeyError as exc:
        raise RuntimeError(f"Anthropic API 返回格式异常：{data}") from exc
    chunks = [str(block.get("text", "")) for block in blocks if block.get("type") == "text"]
    result = "".join(chunks).strip()
    if not result:
        raise RuntimeError(f"Anthropic API 没有返回文本内容：{data}")
    return result
