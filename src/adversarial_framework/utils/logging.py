"""Structured logging setup using structlog.

Call ``setup_logging()`` once at application startup (CLI or API).
All modules use ``structlog.get_logger(__name__)`` for structured output.
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
) -> None:
    """Configure structlog + stdlib logging for the entire application.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR).
        json_output: If True, emit JSON lines; otherwise, coloured console.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # ── stdlib root logger ───────────────────────────────────────────────────
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=log_level,
        force=True,
    )

    # ── structlog processors ─────────────────────────────────────────────────
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Apply structlog formatting to stdlib handlers
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    root = logging.getLogger()
    for handler in root.handlers:
        handler.setFormatter(formatter)
