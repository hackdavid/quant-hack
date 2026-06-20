"""Structured logging setup using structlog.

All logs are JSON with:
- ts: ISO-8601 timestamp
- level: debug/info/warning/error
- event: structured event name
- ... custom fields
"""

import logging
import sys
from pathlib import Path

import structlog


def setup_logging(
    log_level: str = "info",
    log_file: Path | None = None,
    console: bool = True,
) -> structlog.BoundLogger:
    """Configure structured logging.

    Args:
        log_level: Minimum log level (debug, info, warning, error)
        log_file: Optional file path for JSON logs
        console: Whether to log to console (human-readable format)

    Returns:
        Configured logger
    """
    level = getattr(logging, log_level.upper())

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=level,
        handlers=[],
    )

    # Console handler (human-readable)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        logging.root.addHandler(console_handler)

    # File handler (JSON)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        logging.root.addHandler(file_handler)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure formatters
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
        ],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            (
                structlog.dev.ConsoleRenderer(colors=True)
                if console
                else structlog.processors.JSONRenderer()
            ),
        ],
    )

    # Apply formatter to handlers
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)

    return structlog.get_logger()


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Bound logger
    """
    return structlog.get_logger(name)
