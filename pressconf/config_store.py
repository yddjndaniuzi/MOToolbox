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

MODEL_STRENGTHS = {
    "light": {
        "label": "轻量任务",
        "description": "短文本、逐字稿精校与低成本批处理",
    },
    "standard": {
        "label": "标准任务",
        "description": "常规写作、扫描汇总与中等长度分析",
    },
    "heavy": {
        "label": "高强度任务",
        "description": "长发布会精加工、长报告与复杂综合判断",
    },
}

DEFAULT_STRENGTH_BY_USE = {
    "brief_refine": "heavy",
    "writing": "standard",
    "image_match": "light",
    "knowledge_rag": "light",
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
        return normalize_models_config(config)

    load_env(base_dir / ".env")
    model_id = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return normalize_models_config({
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
    })


def normalize_models_config(config: dict[str, Any]) -> dict[str, Any]:
    """Add strength routing to legacy single-model configs without rewriting them."""
    normalized = dict(config or {})
    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_model in normalized.get("models") or []:
        if not isinstance(raw_model, dict):
            continue
        model = dict(raw_model)
        model_id = str(model.get("id") or model.get("model") or "").strip()
        if not model_id or model_id in seen:
            continue
        model["id"] = model_id
        model.setdefault("enabled", True)
        models.append(model)
        seen.add(model_id)
    normalized["models"] = models

    enabled_ids = [str(item["id"]) for item in models if item.get("enabled", True)]
    all_ids = [str(item["id"]) for item in models]
    available_ids = enabled_ids or all_ids
    first_id = available_ids[0] if available_ids else ""

    defaults = dict(normalized.get("default_by_use") or {})
    for use in DEFAULT_STRENGTH_BY_USE:
        if defaults.get(use) not in available_ids and first_id:
            defaults[use] = first_id
    normalized["default_by_use"] = defaults

    raw_routes = normalized.get("routing_by_strength") or {}
    routes: dict[str, dict[str, Any]] = {}
    for strength in MODEL_STRENGTHS:
        raw_route = raw_routes.get(strength) or {}
        if isinstance(raw_route, str):
            raw_route = {"primary": raw_route}
        preferred_use = "brief_refine" if strength == "heavy" else "writing"
        primary = str(raw_route.get("primary") or defaults.get(preferred_use) or first_id).strip()
        if primary not in available_ids:
            primary = first_id
        fallbacks: list[str] = []
        for model_id in raw_route.get("fallbacks") or []:
            model_id = str(model_id).strip()
            if model_id in available_ids and model_id != primary and model_id not in fallbacks:
                fallbacks.append(model_id)
        routes[strength] = {"primary": primary, "fallbacks": fallbacks}
    normalized["routing_by_strength"] = routes
    return normalized


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


def unique_model_id(requested_id: str, existing_ids: set[str]) -> str:
    """Return a stable, human-readable ID without replacing another pool item."""
    requested_id = requested_id.strip() or "custom-model"
    if requested_id not in existing_ids:
        return requested_id
    index = 2
    while f"{requested_id}-{index}" in existing_ids:
        index += 1
    return f"{requested_id}-{index}"


def save_model_config(base_dir: Path, form: dict[str, Any]) -> str:
    provider_tab = str(form.get("provider_tab") or form.get("preset") or "deepseek").strip()
    preset = MODEL_PRESETS.get(provider_tab, {})
    provider = str(form.get("provider") or preset.get("provider") or "openai-compatible").strip()
    if provider_tab.startswith("custom_"):
        provider = str(form.get("custom_protocol") or provider).strip()
    requested_model_id = str(form.get("id") or "").strip() or str(form.get("model") or "").strip()
    if not requested_model_id:
        requested_model_id = str(preset.get("id") or "custom-model")
    uses = [item for item in ("brief_refine", "writing", "image_match", "knowledge_rag") if form.get(f"use_{item}")]
    models_path = config_root(base_dir) / "models.json"
    config = load_models_config(base_dir) if models_path.exists() else {
        "default_by_use": {},
        "routing_by_strength": {},
        "models": [],
    }
    original_id = str(form.get("original_id") or "").strip()
    models = [dict(item) for item in config.get("models") or []]
    existing_ids = {str(item.get("id") or "").strip() for item in models}
    if original_id:
        if original_id not in existing_ids:
            raise ValueError("要编辑的模型配置不存在，请刷新页面后重试。")
        if requested_model_id != original_id and requested_model_id in existing_ids:
            raise ValueError("配置 ID 已被其他模型使用，请换一个 ID。")
        model_id = requested_model_id
    else:
        model_id = unique_model_id(requested_model_id, existing_ids)
    secret_ref = str(form.get("secret_ref") or preset.get("secret_ref") or f"{model_id}_api_key").strip()
    saved_model = {
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
    replaced = False
    if original_id:
        for index, item in enumerate(models):
            if item.get("id") == original_id:
                models[index] = saved_model
                replaced = True
                break
    if not replaced:
        models.append(saved_model)
    config["models"] = models

    if original_id != model_id:
        defaults = config.get("default_by_use") or {}
        for use, selected_id in list(defaults.items()):
            if selected_id == original_id:
                defaults[use] = model_id
        for route in (config.get("routing_by_strength") or {}).values():
            if route.get("primary") == original_id:
                route["primary"] = model_id
            route["fallbacks"] = [model_id if item == original_id else item for item in route.get("fallbacks") or []]

    if len(models) == 1:
        config["default_by_use"] = {use: model_id for use in DEFAULT_STRENGTH_BY_USE}
        config["routing_by_strength"] = {
            strength: {"primary": model_id, "fallbacks": []}
            for strength in MODEL_STRENGTHS
        }
    else:
        for use in saved_model["uses"]:
            config.setdefault("default_by_use", {}).setdefault(use, model_id)
    config = normalize_models_config(config)
    write_json(config_root(base_dir) / "models.json", config)

    api_key = str(form.get("api_key") or "").strip()
    if api_key:
        secrets = load_secrets(base_dir)
        secrets[secret_ref] = api_key
        write_json(config_root(base_dir) / "secrets.json", secrets, private=True)
    return model_id


def save_model_routing(base_dir: Path, form: Any) -> dict[str, Any]:
    config = load_models_config(base_dir)
    available = {str(item.get("id")) for item in config.get("models") or [] if item.get("enabled", True)}
    routes: dict[str, dict[str, Any]] = {}
    for strength in MODEL_STRENGTHS:
        primary = str(form.get(f"route_{strength}_primary") or "").strip()
        if primary not in available:
            raise ValueError(f"{MODEL_STRENGTHS[strength]['label']}没有选择有效的主模型。")
        fallbacks: list[str] = []
        for index in (1, 2):
            model_id = str(form.get(f"route_{strength}_fallback_{index}") or "").strip()
            if model_id in available and model_id != primary and model_id not in fallbacks:
                fallbacks.append(model_id)
        routes[strength] = {"primary": primary, "fallbacks": fallbacks}
    config["routing_by_strength"] = routes
    config["default_by_use"] = {
        "brief_refine": routes["heavy"]["primary"],
        "writing": routes["standard"]["primary"],
        "image_match": routes["light"]["primary"],
        "knowledge_rag": routes["light"]["primary"],
    }
    config = normalize_models_config(config)
    write_json(config_root(base_dir) / "models.json", config)
    return config


def delete_model_config(base_dir: Path, model_id: str) -> dict[str, Any]:
    config = load_models_config(base_dir)
    models = [dict(item) for item in config.get("models") or [] if item.get("id") != model_id]
    if len(models) == len(config.get("models") or []):
        return config
    if not models:
        raise ValueError("至少保留一个模型配置。")
    config["models"] = models
    config = normalize_models_config(config)
    write_json(config_root(base_dir) / "models.json", config)
    return config


def resolve_model_candidates(base_dir: Path, use: str = "brief_refine", strength: str | None = None) -> list[dict[str, Any]]:
    config = load_models_config(base_dir)
    strength = strength if strength in MODEL_STRENGTHS else DEFAULT_STRENGTH_BY_USE.get(use, "standard")
    route = (config.get("routing_by_strength") or {}).get(strength) or {}
    routed_ids = [route.get("primary"), *(route.get("fallbacks") or [])]
    legacy_id = (config.get("default_by_use") or {}).get(use)
    if legacy_id and not routed_ids[0]:
        routed_ids.insert(0, legacy_id)
    models = config.get("models") or []
    candidates: list[dict[str, Any]] = []
    for model_id in routed_ids:
        model = next((item for item in models if item.get("id") == model_id and item.get("enabled", True)), None)
        if model is None:
            continue
        resolved = resolve_model_entry(base_dir, model)
        resolved["strength"] = strength
        resolved["use"] = use
        if not any(item["id"] == resolved["id"] for item in candidates):
            candidates.append(resolved)
    if not candidates and models:
        candidates.append(resolve_model_entry(base_dir, models[0]) | {"strength": strength, "use": use})
    return candidates


def resolve_model(base_dir: Path, use: str = "brief_refine", strength: str | None = None) -> dict[str, Any]:
    candidates = resolve_model_candidates(base_dir, use, strength)
    if candidates:
        model = dict(candidates[0])
        model["fallbacks"] = candidates[1:]
        return model
    return resolve_model_entry(base_dir, {})


def resolve_model_entry(base_dir: Path, model: dict[str, Any]) -> dict[str, str]:
    secrets = load_secrets(base_dir)
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
        "id": str(model.get("id") or model.get("model") or ""),
        "api_key": api_key,
        "base_url": str(model.get("base_url") or default_base_url(provider)).rstrip("/"),
        "model": str(model.get("model") or model.get("id") or "deepseek-v4-pro"),
        "name": str(model.get("name") or model.get("model") or "DeepSeek"),
        "provider": provider,
    }


def strength_for_tier(tier: str) -> str:
    return {"nano": "light", "lite": "standard", "full": "heavy"}.get(str(tier or "").strip().lower(), "standard")


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
