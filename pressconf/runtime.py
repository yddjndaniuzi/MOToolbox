from __future__ import annotations

import os
import ssl
import sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import certifi


APP_NAME = "MOtoolbox"
DESKTOP_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
)
BROWSER_COOKIE_PATHS = (
    ("chrome", Path.home() / "Library" / "Application Support" / "Google" / "Chrome"),
    ("edge", Path.home() / "Library" / "Application Support" / "Microsoft Edge"),
    ("firefox", Path.home() / "Library" / "Application Support" / "Firefox" / "Profiles"),
)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resource_root() -> Path:
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS).resolve()
    return source_root()


def data_root() -> Path:
    configured = os.environ.get("MOTOOLBOX_DATA_DIR", "").strip()
    if configured:
        root = Path(configured).expanduser()
    elif is_frozen():
        root = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        root = source_root()
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def bundled_python() -> Path:
    venv_python = source_root() / ".venv" / "bin" / "python"
    if not is_frozen() and venv_python.exists():
        return venv_python
    return Path(sys.executable)


def ytdlp_command(*args: str) -> list[str]:
    managed = data_root() / "tools" / "yt-dlp"
    if managed.is_file() and os.access(managed, os.X_OK):
        return [str(managed), *args]
    if is_frozen():
        return [sys.executable, "--motoolbox-ytdlp", *args]
    return [str(bundled_python()), "-m", "yt_dlp", *args]


def ytdlp_site_args(url: str, cookie_browser: str = "") -> list[str]:
    host = (urlparse(url).hostname or "").lower()
    if host == "bilibili.com" or host.endswith(".bilibili.com") or host == "b23.tv" or host.endswith(".b23.tv"):
        args = [
            "--user-agent",
            DESKTOP_USER_AGENT,
            "--referer",
            "https://www.bilibili.com/",
            "--add-header",
            "Accept-Language:zh-CN,zh;q=0.9,en;q=0.8",
        ]
        if cookie_browser:
            args.extend(["--cookies-from-browser", cookie_browser])
        return args
    return []


def ytdlp_site_arg_variants(url: str) -> list[list[str]]:
    base_args = ytdlp_site_args(url)
    if not base_args:
        return [[]]
    return [base_args, *(ytdlp_site_args(url, browser) for browser in ytdlp_cookie_browsers())]


def ytdlp_cookie_browsers() -> list[str]:
    configured = os.environ.get("MOTOOLBOX_YTDLP_COOKIE_BROWSER", "").strip()
    if configured:
        return [configured]
    return [name for name, path in BROWSER_COOKIE_PATHS if path.exists()]


def embedded_vault_path() -> Path | None:
    path = resource_root() / "pressconf" / "embedded_vault"
    return path if path.exists() else None


@lru_cache(maxsize=1)
def certifi_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    # Bundled Python runtimes may not discover macOS CA paths on their own.
    context.load_verify_locations(cafile=certifi.where())
    return context
