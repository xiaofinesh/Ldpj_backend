"""Tiny rate-limited warning helper (v2.6).

Keeps the ``ldpj_backend.log`` from drowning in the same warning every
cycle when an edge case (short section, clamped C_d, etc.) fires
continuously. Always logs the first occurrence; afterwards logs at most
once per ``every_n`` calls or once per ``min_interval_s`` seconds,
whichever fires first.

State is per-process and per-key. The key is just an arbitrary string
chosen by the caller (typically the call site's identifier).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Tuple

_state: Dict[str, Tuple[int, float]] = {}
_lock = threading.Lock()
_logger = logging.getLogger(__name__)


def warn_throttled(
    key: str,
    message: str,
    *args,
    first_n: int = 1,
    every_n: int = 100,
    min_interval_s: float = 60.0,
) -> None:
    """Log .warning at most once every `every_n` calls or `min_interval_s`,
    whichever fires first; always log the first `first_n` occurrences.

    The total occurrence count is prepended to the message so postmortems
    can tell how often the path actually fired between logged lines.
    """
    with _lock:
        count, last_ts = _state.get(key, (0, 0.0))
        count += 1
        now = time.monotonic()
        should_log = (
            count <= first_n
            or count % every_n == 0
            or (now - last_ts) >= min_interval_s
        )
        if should_log:
            last_ts = now
        _state[key] = (count, last_ts)

    if should_log:
        _logger.warning("[%s #%d] " + message, key, count, *args)


def reset_state() -> None:
    """Reset all per-key counters. Used by tests; not thread-safe vs. live
    callers, so do not call from production code."""
    with _lock:
        _state.clear()
