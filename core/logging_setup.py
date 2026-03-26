"""Logging setup with dynamic console level switching and periodic status report.

v2.5: Supports switching console output between NORMAL (WARNING+) and DEBUG (all).
      File handler always logs everything (INFO level).
      Periodic status reporter prints system summary every N seconds.
"""

from __future__ import annotations

import logging
import threading
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class ConsoleLevelFilter(logging.Filter):
    """Dynamically filters console output by level."""

    def __init__(self):
        super().__init__()
        self._min_level = logging.WARNING  # Default: normal mode

    @property
    def min_level(self) -> int:
        return self._min_level

    def set_level(self, level: int) -> None:
        self._min_level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self._min_level


# Module-level reference for mode switching
_console_filter: Optional[ConsoleLevelFilter] = None
_console_handler: Optional[logging.StreamHandler] = None


def setup_logging(cfg: Dict[str, Any]) -> logging.Logger:
    """Configure root logger with file (all) + console (filtered) handlers."""
    global _console_filter, _console_handler

    log_file = cfg.get("file", "ldpj_backend.log")
    fmt = cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    rotate = cfg.get("rotate", {})
    max_bytes = rotate.get("max_bytes", 5_242_880)
    backup_count = rotate.get("backup_count", 5)

    formatter = logging.Formatter(fmt)

    # File handler: always logs INFO and above
    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler: filtered by ConsoleLevelFilter (default WARNING)
    _console_filter = ConsoleLevelFilter()
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(formatter)
    _console_handler.addFilter(_console_filter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Root accepts everything; handlers filter
    root.handlers = [file_handler, _console_handler]

    return root


def set_console_mode(mode: str) -> str:
    """Switch console output mode.

    Parameters
    ----------
    mode : str
        "normal" → WARNING+ only, "debug" → all (DEBUG+)

    Returns
    -------
    str
        The new mode name.
    """
    global _console_filter
    if _console_filter is None:
        return "unknown"

    if mode == "debug":
        _console_filter.set_level(logging.DEBUG)
        return "debug"
    elif mode == "silent":
        _console_filter.set_level(logging.CRITICAL + 1)
        return "silent"
    else:
        _console_filter.set_level(logging.WARNING)
        return "normal"


def get_console_mode() -> str:
    if _console_filter is None:
        return "unknown"
    return "debug" if _console_filter.min_level <= logging.DEBUG else "normal"


class StatusReporter:
    """Periodically prints a one-line system status summary to console.

    Only active in normal mode (not debug mode, which already shows everything).
    """

    def __init__(self, interval_s: float = 30.0):
        self._interval = interval_s
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._get_status: Optional[Callable] = None

    def set_status_fn(self, fn: Callable[[], Dict[str, Any]]) -> None:
        self._get_status = fn

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="status-reporter")
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            time.sleep(self._interval)
            if not self._running:
                break
            # Only print in normal mode
            if get_console_mode() != "normal":
                continue
            if self._get_status:
                try:
                    self._report()
                except Exception:
                    pass

    def _report(self) -> None:
        if not self._get_status:
            return
        st = self._get_status()
        ts = time.strftime("%H:%M:%S")

        running = "运行中" if st.get("running") else "已暂停"
        model = st.get("model_version", "none")
        polls = st.get("poller_stats", {}).get("total_polls", 0)
        errors = st.get("poller_stats", {}).get("errors", 0)
        db_records = st.get("db_records", "?")

        # Count cabin states
        cabin_states = st.get("cabin_states", {})
        idle = sum(1 for v in cabin_states.values() if v.get("state") == "IDLE")
        collecting = sum(1 for v in cabin_states.values() if v.get("state") == "COLLECTING")
        processing = sum(1 for v in cabin_states.values() if v.get("state") == "PROCESSING")

        print(f"\r[{ts}] {running} | 模型={model} | "
              f"轮询={polls} 错误={errors} | "
              f"舱: IDLE={idle} COLL={collecting} PROC={processing} | "
              f"记录={db_records}")
