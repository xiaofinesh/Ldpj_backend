"""Interactive command controller for the main loop.

Supports single-key commands for runtime control:
    s  - start / resume processing
    e  - stop / pause processing
    w  - toggle watchdog (auto-restart on fault)
    t  - run a single test cycle (manual trigger)
    h  - run health check and print report
    d  - print diagnostic information
    q  - quit the application
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class CommandController:
    """Non-blocking keyboard listener that dispatches single-key commands."""

    HELP_TEXT = (
        "\n--- Ldpj_backend 命令 ---\n"
        "  s  启动/恢复处理\n"
        "  e  停止/暂停处理\n"
        "  w  切换看门狗 (自动重启)\n"
        "  h  执行健康检查\n"
        "  d  打印诊断信息\n"
        "  q  退出程序\n"
        "------------------------\n"
    )

    def __init__(self):
        self._handlers: Dict[str, Callable[[], None]] = {}
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def register(self, key: str, handler: Callable[[], None]) -> None:
        self._handlers[key.lower()] = handler

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True, name="cmd-ctrl")
        self._thread.start()
        print(self.HELP_TEXT)

    def stop(self) -> None:
        self._running = False

    def _listen(self) -> None:
        while self._running:
            try:
                line = sys.stdin.readline().strip().lower()
                if not line:
                    continue
                key = line[0]
                handler = self._handlers.get(key)
                if handler:
                    handler()
                else:
                    print(f"未知命令: '{key}'  输入有效命令 (s/e/w/h/d/q)")
            except EOFError:
                break
            except Exception as exc:
                logger.error("Command handler error: %s", exc)
