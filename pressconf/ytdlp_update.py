"""Install verified official yt-dlp releases outside the signed app bundle."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

from pressconf.runtime import certifi_ssl_context, data_root, ytdlp_command

_LOCK = threading.Lock()
_STATE = {"running": False, "message": "", "error": ""}


def managed_binary() -> Path:
    return data_root() / "tools" / "yt-dlp"


def version(command=None) -> str:
    result = subprocess.run(command or ytdlp_command("--ignore-config", "--version"), capture_output=True, text=True, timeout=30)
    value = result.stdout.strip()
    if result.returncode or not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}(?:[.\w+-]*)", value):
        raise RuntimeError("无法读取 yt-dlp 版本：" + (result.stderr or value)[-500:])
    return value


def status() -> dict:
    with _LOCK:
        state = dict(_STATE)
    try:
        state["version"] = version()
    except Exception as exc:
        state["version"] = "未知"
        state["version_error"] = str(exc)
    state["supported"] = sys.platform == "darwin"
    return state


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "MOtoolbox", "Accept": "application/vnd.github+json" if url.startswith("https://api.github.com/") else "application/octet-stream"})
    with urllib.request.urlopen(req, context=certifi_ssl_context(), timeout=60) as response:
        return response.read()


def install_latest() -> str:
    if sys.platform != "darwin":
        raise RuntimeError("一键更新目前支持 macOS")
    release = json.loads(fetch("https://api.github.com/repos/yt-dlp/yt-dlp/releases/latest"))
    tag = release["tag_name"]
    if not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", tag):
        raise RuntimeError("官方版本号格式异常")
    base = f"https://github.com/yt-dlp/yt-dlp/releases/download/{tag}/"
    checksums = fetch(base + "SHA2-256SUMS").decode()
    expected = next((line.split()[0] for line in checksums.splitlines() if line.split()[-1:] == ["yt-dlp_macos"]), None)
    if not expected:
        raise RuntimeError("官方发布缺少 SHA256 校验值")
    payload = fetch(base + "yt-dlp_macos")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeError("下载校验失败，已保留原版本，请重试")
    target = managed_binary()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".yt-dlp-", dir=target.parent)
    candidate = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        candidate.chmod(0o755)
        actual = version([str(candidate), "--ignore-config", "--version"])
        if actual != tag:
            raise RuntimeError("下载版本验证失败，已保留原版本")
        os.replace(candidate, target)
        return actual
    finally:
        candidate.unlink(missing_ok=True)


def _worker():
    try:
        installed = install_latest()
        with _LOCK:
            _STATE.update(message=f"已更新至 {installed}，下一次采集立即生效。", error="")
    except Exception as exc:
        with _LOCK:
            _STATE.update(message="更新失败，原版本仍可使用。", error=str(exc))
    finally:
        with _LOCK:
            _STATE["running"] = False


def start_update() -> bool:
    with _LOCK:
        if _STATE["running"]:
            return False
        _STATE.update(running=True, message="正在下载并验证官方最新版本…", error="")
    threading.Thread(target=_worker, daemon=True).start()
    return True
