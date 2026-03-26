"""Interactive command controller with startup banner and mode selection.

v2.5: Startup shows system status banner, then mode selection (normal/debug),
      then command menu. System auto-starts in running state.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


def print_banner(mode: str, plc_connected: bool, model_loaded: bool,
                 model_version: str, cabin_count: int, active_range: str) -> None:
    """Print system status banner at startup."""
    plc_str = "已连接" if plc_connected else "未连接"
    model_str = f"已加载 ({model_version})" if model_loaded else "未加载"
    mode_str = "S7 (真实PLC)" if mode == "s7" else "Mock (模拟数据)"

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║          Ldpj_backend v2.5 — 漏液检测系统           ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  运行模式:  {mode_str:<40s}║")
    print(f"║  PLC状态:   {plc_str:<40s}║")
    print(f"║  AI模型:    {model_str:<40s}║")
    print(f"║  活跃舱室:  {active_range:<40s}║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def select_display_mode() -> str:
    """Let user choose display mode at startup."""
    print("  请选择显示模式:")
    print("    [1] 正常模式 — 仅显示警告和错误, 每30秒报告状态")
    print("    [2] 调试模式 — 显示所有日志 (INFO/DEBUG)")
    print()

    try:
        choice = input("  请选择 (1/2, 默认1): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    if choice == "2":
        print("  → 调试模式\n")
        return "debug"
    else:
        print("  → 正常模式\n")
        return "normal"


class CommandController:
    """Non-blocking keyboard listener that dispatches single-key commands."""

    MENU_TEXT = (
        "──────────── 操作命令 ────────────\n"
        "  s  启动/恢复 采集与推理\n"
        "  e  暂停 采集与推理\n"
        "  w  切换看门狗\n"
        "  x  导出数据到 CSV\n"
        "  m  切换显示模式 (正常/调试)\n"
        "  h  执行健康检查\n"
        "  d  打印诊断信息\n"
        "  q  退出程序\n"
        "──────────────────────────────────\n"
    )

    def __init__(self):
        self._handlers: Dict[str, Callable[[], None]] = {}
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def register(self, key: str, handler: Callable[[], None]) -> None:
        self._handlers[key.lower()] = handler

    def start(self) -> None:
        self._running = True
        print(self.MENU_TEXT)
        print("  系统已自动启动, 等待命令...\n")
        self._thread = threading.Thread(target=self._listen, daemon=True, name="cmd-ctrl")
        self._thread.start()

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
                    print(f"  未知命令: '{key}'")
                    print(self.MENU_TEXT)
            except EOFError:
                break
            except Exception as exc:
                logger.error("Command handler error: %s", exc)
