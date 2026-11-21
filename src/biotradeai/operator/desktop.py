"""Desktop launcher: the operator app in its own window.

1. starts the agent host in the background if it isn't running (unless --no-host);
2. serves the app on 127.0.0.1 (random free port) with a fresh session token;
3. opens a native window (pywebview, Edge WebView2 on Windows). If that is not
   available it falls back to an Edge "app" window, then to the default browser.

Closing the window stops the web server only; the agent host keeps running.
"""

from __future__ import annotations

import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

from biotradeai.operator.db import OperatorDB
from biotradeai.operator.host import host_status, start_detached

TASK_NAME = "BioTradeAI agent host"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def _serve(token: str, port: int) -> threading.Thread:
    import uvicorn

    from biotradeai.operator.api import create_app

    config = uvicorn.Config(create_app(token), host="127.0.0.1", port=port,
                            log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, name="operator-ui", daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return t


def _edge() -> str | None:
    for c in (shutil.which("msedge"),
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if c and Path(c).exists():
            return c
    return None


def launch(*, browser: bool = False, port: int = 0, start_host: bool = True) -> int:
    if start_host and not host_status(OperatorDB()).running:
        start_detached()
    token = secrets.token_urlsafe(32)
    port = port or _free_port()
    server = _serve(token, port)
    url = f"http://127.0.0.1:{port}/auth?t={token}"
    if not browser:
        try:
            import webview  # type: ignore[import-not-found,unused-ignore]

            webview.create_window("BioTradeAI Operator", url, width=1320, height=860,
                                  min_size=(1024, 680), background_color="#F7F7F5")
            webview.start()
            return 0
        except Exception as exc:
            print(f"(native window unavailable: {exc}; using a browser window)")
        edge = _edge()
        if edge:
            subprocess.Popen([edge, f"--app={url}", "--window-size=1320,860"])  # noqa: S603
            print("Operator app opened. Keep this terminal open; Ctrl+C to close the app.")
            _wait(server)
            return 0
    webbrowser.open(url)
    print(f"Operator app: http://127.0.0.1:{port}  (Ctrl+C to stop)")
    _wait(server)
    return 0


def _wait(server: threading.Thread) -> None:
    try:
        while server.is_alive():
            server.join(1.0)
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------- start at login


def _schtasks() -> str:
    exe = shutil.which("schtasks")
    if exe is None:
        raise RuntimeError("Windows Task Scheduler (schtasks) is not available")
    return exe


def autostart_enabled() -> bool | None:
    """True/False on Windows; None where it doesn't apply."""
    if os.name != "nt":
        return None
    try:
        r = subprocess.run([_schtasks(), "/Query", "/TN", TASK_NAME],  # noqa: S603
                           capture_output=True, check=False, timeout=10)
    except (RuntimeError, OSError, subprocess.TimeoutExpired):
        return None
    return r.returncode == 0


def set_autostart(on: bool) -> None:
    if os.name != "nt":
        raise RuntimeError("start-at-login is only set up for Windows; use cron or systemd")
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe")
    runner = pyw if pyw.exists() else exe
    from biotradeai.paths import REPO_ROOT

    if on:
        # Task Scheduler runs this command line at logon; cd into the repo for .env/data.
        cmd = f'cmd /c cd /d "{REPO_ROOT}" && "{runner}" -m biotradeai host'
        args = [_schtasks(), "/Create", "/F", "/TN", TASK_NAME, "/SC", "ONLOGON", "/RL",
                "LIMITED", "/TR", cmd]
    else:
        args = [_schtasks(), "/Delete", "/F", "/TN", TASK_NAME]
    r = subprocess.run(args, capture_output=True, text=True, check=False,  # noqa: S603
                       timeout=20)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "schtasks failed").strip()[:300])
