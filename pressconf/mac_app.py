from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import fcntl

from pressconf.runtime import data_root


HOST = "127.0.0.1"
PREFERRED_PORT = 5058
STATE_FILE = "server.json"
LOCK_FILE = "server.lock"


def acquire_instance_lock(root: Path):
    lock_path = root / LOCK_FILE
    lock_file = lock_path.open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return None
    return lock_file


def state_path(root: Path) -> Path:
    return root / STATE_FILE


def read_server_state(root: Path) -> dict:
    path = state_path(root)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_server_state(root: Path, port: int) -> None:
    state_path(root).write_text(
        json.dumps({"host": HOST, "port": port, "pid": os.getpid()}, ensure_ascii=False),
        encoding="utf-8",
    )


def is_server_healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/healthz", timeout=0.5) as response:
            return response.status == 200
    except Exception:
        return False


def notify_existing_instance() -> None:
    if sys.platform != "darwin":
        return
    script = (
        'display notification "已有一个实例在运行中，请使用已打开的 MOtoolbox 页面。" '
        'with title "MOtoolbox"'
    )
    try:
        subprocess.run(["osascript", "-e", script], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def resolve_port() -> int:
    configured = os.environ.get("MOTOOLBOX_PORT", "").strip()
    if configured:
        return int(configured)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex((HOST, PREFERRED_PORT)) != 0:
            return PREFERRED_PORT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def open_browser_later(url: str) -> None:
    time.sleep(1.2)
    webbrowser.open(url)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--motoolbox-ytdlp":
        import yt_dlp

        yt_dlp.main(sys.argv[2:])
        return

    root = data_root()
    lock_file = acquire_instance_lock(root)
    if lock_file is None:
        state = read_server_state(root)
        existing_port = int(state.get("port") or 0)
        if existing_port and is_server_healthy(existing_port):
            notify_existing_instance()
            return
        notify_existing_instance()
        return

    if not os.environ.get("MOTOOLBOX_PORT", "").strip() and is_server_healthy(PREFERRED_PORT):
        lock_file.close()
        notify_existing_instance()
        return

    from pressconf.web import app

    port = resolve_port()
    write_server_state(root, port)
    url = f"http://{HOST}:{port}"
    if os.environ.get("MOTOOLBOX_NO_BROWSER", "").strip() != "1":
        threading.Thread(target=open_browser_later, args=(url,), daemon=True).start()
    try:
        app.run(host=HOST, port=port, debug=False, use_reloader=False)
    finally:
        lock_file.close()


if __name__ == "__main__":
    main()
