#!/usr/bin/env python3
"""app.py — Launcher for llama.cpp / ik_llama."""

import socket
import sys
import threading
import time

import uvicorn
import webview

from launcher import app


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def run_server(port: int):
    uvicorn.run(app, host='127.0.0.1', port=port, log_level='warning')


def wait_for_server(port: int, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=0.1):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def main():
    port = find_free_port()
    print(f"[app] Server in http://127.0.0.1:{port}")
    threading.Thread(target=run_server, args=(port,), daemon=True).start()

    if not wait_for_server(port):
        print("[app] server did not respond in time")
        return

    time.sleep(0.3)   # small extra margin for uvicorn to finish starting up

    webview.create_window(
        'Llama.cpp / ik_llama Launcher',
        f'http://127.0.0.1:{port}',
        width=1400,
        height=900,
        min_size=(1000, 600),
    )
    # gtk on Linux, qt on Windows (PyGObject/GTK is not packaged for Windows)
    gui = 'qt' if sys.platform.startswith('win') else 'gtk'
    webview.start(gui=gui, private_mode=False)


if __name__ == '__main__':
    main()
