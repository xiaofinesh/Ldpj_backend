"""Logging setup helper with rotating file handler."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Any


def setup_logging(cfg: Dict[str, Any]) -> logging.Logger:
    level = cfg.get("level", "INFO")
    log_file = cfg.get("file", "ldpj_backend.log")
    fmt = cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    rotate = cfg.get("rotate", {})
    max_bytes = rotate.get("max_bytes", 5_242_880)
    backup_count = rotate.get("backup_count", 5)

    formatter = logging.Formatter(fmt)

    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers = [file_handler, console_handler]

    return root
