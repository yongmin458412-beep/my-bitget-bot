"""Structured logging setup."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from .utils import ensure_directory


class JsonFormatter(logging.Formatter):
    """Format log records as one-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if hasattr(record, "extra_data"):
            payload["extra"] = getattr(record, "extra_data")
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(log_dir: str | Path, level: str = "INFO") -> None:
    """Configure root logging once."""

    ensure_directory(log_dir)
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(level.upper())
    formatter = JsonFormatter()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        Path(log_dir) / "bot.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""

    return logging.getLogger(name)


def bind_extra(logger: logging.Logger, **extra_data: Any) -> logging.LoggerAdapter:
    """Return a logger adapter with structured context."""

    return logging.LoggerAdapter(logger, {"extra_data": extra_data})

