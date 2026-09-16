from __future__ import annotations

import json
import os
import re
import secrets
from pathlib import Path
from typing import Any

from pressconf.runtime import embedded_vault_path

DEFAULT_VAULT_PATH = ""
MODEL_PRESETS = {
    "deepseek": {
        "id": "deepseek-v4-pro",
        "name": "DeepSeek V4 Pro",
        "provider": "openai-compatible",
        "provider_tab": "deepseek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-pro",
        "secret_ref": "deepseek_api_key",
    },
    "custom_openai": {
        "id": "custom-openai-model",
        "name": "自定义 OpenAI 兼容模型",
        "provider": "openai-compatible",
        "provider_tab": "custom_openai",
        "base_url": "",
        "model": "",
        "secret_ref": "custom_openai_api_key",
    },
    "custom_anthropic": {
        "id": "custom-anthropic-model",
        "name": "自定义 Anthropic 兼容模型",
        "provider": "anthropic",
        "provider_tab": "custom_anthropic",
        "base_url": "",
        "model": "",
        "secret_ref": "custom_anthropic_api_key",
    },
}


def config_root(base_dir: Path) -> Path:
    root = base_dir / "pressconf" / "config"
    root.mkdir(parents=True, exist_ok=True)
    return root


def index_root(base_dir: Path) -> Path:
    root = base_dir / "pressconf" / "index"
    root.mkdir(parents=True, exist_ok=True)
    return root


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if private:
        path.chmod(0o600)


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def load_models_config(base_dir: Path) -> dict[str, Any]:
    config_path = config_root(base_dir) / "models.json"
    config = read_json(config_path, {})
    if config:
        return config

    load_env(base_dir / ".env")
    model_id = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return {
        "default_by_use": {"brief_refine": model_id, "writing": model_id},
        "models": [
            {
                "id": model_id,
                "name": "DeepSeek V4 Pro",
                "provider": "openai-compatible",
                "provider_tab": "deepseek",
                "base_url": base_url,
                "model": model_id,
                "secret_ref": "deepseek_api_key",
                "uses": ["brief_refine", "writing"],
                "enabled": True,
            }
        ],
    }


def load_secrets(base_dir: Path) -> dict[str, str]:
    secrets_path = config_root(base_dir) / "secrets.json"
    secrets = read_json(secrets_path, {})
    if secrets:
        return secrets
    load_env(base_dir / ".env")
    env_map = {
        "deepseek_api_key": "DEEPSEEK_API_KEY",
        "openai_api_key": "OPENAI_API_KEY",
        "anthropic_api_key": "ANTHROPIC_API_KEY",
    }
    loaded = {secret_ref: os.environ[env_name] for secret_ref, env_name in env_map.items() if os.environ.get(env_name)}
    for env_name, value in os.environ.items():
        if env_name.endswith("_API_KEY") and value:
            loaded.setdefault(env_name.lower(), value)
    return loaded


def save_model_config(base_dir: Path, form: dict[str, Any]) -> None:
    provider_tab = str(form.get("provider_tab") or form.get("preset") or "deepseek").strip()
    preset = MODEL_PRESETS.get(provider_tab, {})
    provider = str(form.get("provider") or preset.get("provider") or "openai-compatible").strip()
    if provider_tab.startswith("custom_"):
        provider = str(form.get("custom_protocol") or provider).strip()
    model_id = str(form.get("id") or "").strip() or str(form.get("model") or "").strip()
    if not model_id:
        model_id = str(preset.get("id") or "custom-model")
    secret_ref = str(form.get("secret_ref") or preset.get("secret_ref") or f"{model_id}_api_key").strip()
    uses = [item for item in ("brief_refine", "writing", "image_match", "knowledge_rag") if form.get(f"use_{item}")]
    config = {
        "default_by_use": {
            "brief_refine": model_id,
            "writing": model_id,
        },
        "models": [
            {
                "id": model_id,
                "name": str(form.get("name") or preset.get("name") or model_id).strip(),
                "provider": provider,
                "provider_tab": provider_tab,
                "base_url": str(form.get("base_url") or preset.get("base_url") or "").strip().rstrip("/"),
                "model": str(form.get("model") or model_id).strip(),
                "secret_ref": secret_ref,
                "uses": uses or ["brief_refine"],
                "enabled": True,
            }
        ],
    }
    write_json(config_root(base_dir) / "models.json", config)

    api_key = str(form.get("api_key") or "").strip()
    if api_key:
        secrets = load_secrets(base_dir)
        secrets[secret_ref] = api_key
        write_json(config_root(base_dir) / "secrets.json", secrets, private=True)


def resolve_model(base_dir: Path, use: str = "brief_refine") -> dict[str, str]:
    config = load_models_config(base_dir)
    secrets = load_secrets(base_dir)
    model_id = (config.get("default_by_use") or {}).get(use)
    models = config.get("models") or []
    model = next((item for item in models if item.get("id") == model_id), models[0] if models else {})
    secret_ref = model.get("secret_ref", "")
    api_key = secrets.get(secret_ref, "")
    provider = str(model.get("provider") or "openai-compatible").strip()
    if not api_key:
        load_env(base_dir / ".env")
        fallback_env_name = {
            "openai": "OPENAI_API_KEY",
            "openai-compatible": "DEEPSEEK_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }.get(provider, "DEEPSEEK_API_KEY")
        env_name = secret_ref.upper() if re.fullmatch(r"[A-Za-z0-9_]+", secret_ref) else fallback_env_name
        api_key = os.environ.get(env_name, "")
    return {
        "api_key": api_key,
        "base_url": str(model.get("base_url") or default_base_url(provider)).rstrip("/"),
        "model": str(model.get("model") or model.get("id") or "deepseek-v4-pro"),
        "name": str(model.get("name") or model.get("model") or "DeepSeek"),
        "provider": provider,
    }


def default_base_url(provider: str) -> str:
    provider = str(provider or "openai-compatible").strip().lower()
    env_prefix = re.sub(r"[^A-Z0-9]+", "_", provider.upper()).strip("_")
    provider_base_url = os.environ.get(f"{env_prefix}_BASE_URL", "") if env_prefix else ""
    family_prefix = env_prefix.split("_", 1)[0] if env_prefix else ""
    family_base_url = os.environ.get(f"{family_prefix}_BASE_URL", "") if family_prefix else ""
    if provider_base_url or family_base_url:
        return provider_base_url or family_base_url
    if provider in {"anthropic", "claude"} or provider.endswith("-anthropic"):
        return "https://api.anthropic.com"
    if provider == "openai":
        return "https://api.openai.com/v1"
    return "https://api.deepseek.com"


def masked_secret(base_dir: Path, secret_ref: str) -> str:
    value = load_secrets(base_dir).get(secret_ref, "")
    if not value:
        return "未配置"
    if len(value) <= 8:
        return "已配置"
    return f"{value[:3]}...{value[-4:]}"


def load_knowledge_config(base_dir: Path) -> dict[str, Any]:
    embedded_vault = embedded_vault_path()
    return read_json(
        config_root(base_dir) / "knowledge.json",
        {
            "name": "竞品发布会简报库",
            "type": "obsidian",
            "vault_path": str(embedded_vault) if embedded_vault else DEFAULT_VAULT_PATH,
            "include_subdir": "",
        },
    )


def save_knowledge_config(base_dir: Path, form: dict[str, Any]) -> dict[str, Any]:
    config = {
        "name": str(form.get("name") or "竞品发布会简报库").strip(),
        "type": "obsidian",
        "vault_path": str(form.get("vault_path") or "").strip(),
        "include_subdir": str(form.get("include_subdir") or "").strip(),
    }
    write_json(config_root(base_dir) / "knowledge.json", config)
    return config


def load_lark_config(base_dir: Path) -> dict[str, Any]:
    config = read_json(
        config_root(base_dir) / "lark.json",
        {
            "enabled": True,
            "transport": "mcp_http",
            "mcp_url": "",
            "mcp_headers": "{}",
            "mcp_method": "tools/call",
            "mcp_tool": "create-doc",
            "mcp_arguments": "{\n  \"title\": \"{{title}}\",\n  \"markdown\": \"{{markdown}}\"\n}",
            "mcp_payload": "",
            "image_url_base": "",
            "image_host_token": secrets.token_urlsafe(18),
            "image_upload_provider": "none",
            "imgbb_api_key": "",
            "imgbb_expiration": "604800",
            "image_grid_columns": "4",
            "identity": "user",
            "target_type": "default",
            "folder_token": "",
            "wiki_space": "",
            "wiki_node": "",
        },
    )
    config.setdefault("transport", "mcp_http")
    config.setdefault("mcp_url", "")
    config.setdefault("mcp_headers", "{}")
    config.setdefault("mcp_method", "tools/call")
    if config.get("mcp_tool") == "docs_create":
        config["mcp_tool"] = "create-doc"
    config.setdefault("mcp_tool", "create-doc")
    config.setdefault("mcp_arguments", "{\n  \"title\": \"{{title}}\",\n  \"markdown\": \"{{markdown}}\"\n}")
    config.setdefault("mcp_payload", "")
    config.setdefault("image_url_base", "")
    config.setdefault("image_upload_provider", "none")
    config.setdefault("imgbb_api_key", "")
    config.setdefault("imgbb_expiration", "604800")
    config.setdefault("image_grid_columns", "4")
    if not config.get("image_host_token"):
        config["image_host_token"] = secrets.token_urlsafe(18)
        write_json(config_root(base_dir) / "lark.json", config)
    try:
        payload = json.loads(str(config.get("mcp_payload") or "{}"))
        if isinstance(payload, dict) and "mcpServers" in payload:
            config["mcp_payload"] = ""
    except json.JSONDecodeError:
        pass
    return config


def save_lark_config(base_dir: Path, form: dict[str, Any]) -> dict[str, Any]:
    target_type = str(form.get("target_type") or "default").strip()
    if target_type not in {"default", "folder", "wiki"}:
        target_type = "default"
    identity = str(form.get("identity") or "user").strip()
    if identity not in {"user", "bot"}:
        identity = "user"
    transport = str(form.get("transport") or "mcp_http").strip()
    if transport not in {"mcp_http", "lark_cli"}:
        transport = "mcp_http"
    config = {
        "enabled": bool(form.get("enabled")),
        "transport": transport,
        "mcp_url": str(form.get("mcp_url") or "").strip(),
        "mcp_headers": str(form.get("mcp_headers") or "{}").strip() or "{}",
        "mcp_method": str(form.get("mcp_method") or "tools/call").strip(),
        "mcp_tool": str(form.get("mcp_tool") or "create-doc").strip(),
        "mcp_arguments": str(form.get("mcp_arguments") or "").strip()
        or "{\n  \"title\": \"{{title}}\",\n  \"markdown\": \"{{markdown}}\"\n}",
        "mcp_payload": str(form.get("mcp_payload") or "").strip(),
        "image_url_base": str(form.get("image_url_base") or "").strip().rstrip("/"),
        "image_host_token": str(form.get("image_host_token") or load_lark_config(base_dir).get("image_host_token") or secrets.token_urlsafe(18)).strip(),
        "image_upload_provider": str(form.get("image_upload_provider") or "none").strip(),
        "imgbb_api_key": str(form.get("imgbb_api_key") or load_lark_config(base_dir).get("imgbb_api_key") or "").strip(),
        "imgbb_expiration": str(form.get("imgbb_expiration") or "604800").strip(),
        "image_grid_columns": str(form.get("image_grid_columns") or "4").strip(),
        "identity": identity,
        "target_type": target_type,
        "folder_token": str(form.get("folder_token") or "").strip(),
        "wiki_space": str(form.get("wiki_space") or "").strip(),
        "wiki_node": str(form.get("wiki_node") or "").strip(),
    }
    write_json(config_root(base_dir) / "lark.json", config, private=True)
    return config
