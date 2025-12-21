"""
Logging configuration using loguru.
Provides structured logging with request tracing.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional
import uuid
from contextvars import ContextVar

from app.config import settings


# Context variable for request ID tracking
request_id_ctx: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


def setup_logging():
    """
    Configure loguru logging with file rotation and formatting.
    Sets up separate log files for different severity levels.
    """
    # Remove default handler
    logger.remove()

    # Console handler with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )

    # Ensure log directory exists
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Info and above to file with rotation
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {extra[request_id]} | {message}",
        level="INFO",
        rotation="00:00",  # Rotate at midnight
        retention="30 days",
        compression="zip",
        enqueue=True,
    )

    # Error logs to separate file
    logger.add(
        log_dir / "error_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {extra[request_id]} | {message}",
        level="ERROR",
        rotation="00:00",
        retention="90 days",
        compression="zip",
        enqueue=True,
    )

    # Debug logs in debug mode
    if settings.debug:
        logger.add(
            log_dir / "debug_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {extra[request_id]} | {message}",
            level="DEBUG",
            rotation="100 MB",
            retention="7 days",
            enqueue=True,
        )

    logger.info(f"Logging initialized - Level: {settings.log_level}")


def get_logger(name: str):
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance configured with request ID tracking
    """
    return logger.bind(name=name, request_id=lambda: request_id_ctx.get() or "no-request")


def set_request_id(request_id: Optional[str] = None):
    """
    Set the request ID for the current context.

    Args:
        request_id: Optional request ID. If None, generates a new UUID.
    """
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    request_id_ctx.set(request_id)
    return request_id


def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return request_id_ctx.get()
