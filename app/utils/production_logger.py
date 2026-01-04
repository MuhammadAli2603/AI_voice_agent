"""
Production-grade logging and monitoring configuration

Features:
- Structured JSON logging for production
- Call/Request ID tracking
- Performance metrics
- Error tracking
- Multiple log handlers (console, file, error, audit)
- Log rotation and retention
"""
import sys
import json
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from loguru import logger
from contextvars import ContextVar

# Context variables for tracking
call_context: ContextVar[Optional[str]] = ContextVar('call_context', default=None)
request_context: ContextVar[Optional[str]] = ContextVar('request_context', default=None)


class ProductionLogger:
    """Production-grade logger with structured logging and monitoring"""

    def __init__(
        self,
        app_name: str = "ai_voice_agent",
        environment: str = "production",
        log_dir: str = "logs"
    ):
        self.app_name = app_name
        self.environment = environment
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create subdirectories for different log types
        (self.log_dir / "app").mkdir(exist_ok=True)
        (self.log_dir / "errors").mkdir(exist_ok=True)
        (self.log_dir / "audit").mkdir(exist_ok=True)
        (self.log_dir / "performance").mkdir(exist_ok=True)

        self._setup_logger()

    def _format_record(self, record: dict) -> str:
        """Format log record as JSON for production"""

        # Extract context
        call_id = call_context.get()
        request_id = request_context.get()

        # Build structured log
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record["level"].name,
            "logger": record["name"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
            "app": self.app_name,
            "environment": self.environment,
        }

        # Add context IDs if available
        if call_id:
            log_entry["call_id"] = call_id
        if request_id:
            log_entry["request_id"] = request_id

        # Add exception info if present
        if record["exception"]:
            log_entry["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback
            }

        # Add extra fields
        if record.get("extra"):
            log_entry["extra"] = record["extra"]

        return json.dumps(log_entry)

    def _human_format(self, record: dict) -> str:
        """Human-readable format for development"""
        call_id = call_context.get()
        call_str = f"[Call: {call_id[:8]}] " if call_id else ""

        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            f"{call_str}"
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    def _setup_logger(self):
        """Configure all log handlers"""

        # Remove default handler
        logger.remove()

        is_production = self.environment == "production"

        # 1. CONSOLE HANDLER
        if is_production:
            # JSON format for production (easier to parse)
            logger.add(
                sys.stdout,
                format=lambda record: self._format_record(record) + "\n",
                level="INFO",
                serialize=False
            )
        else:
            # Human-readable for development
            logger.add(
                sys.stdout,
                format=self._human_format,
                level="DEBUG",
                colorize=True
            )

        # 2. APPLICATION LOG (all logs)
        logger.add(
            self.log_dir / "app" / "app_{time:YYYY-MM-DD}.log",
            format=lambda record: self._format_record(record) + "\n",
            level="DEBUG",
            rotation="00:00",  # Rotate at midnight
            retention="30 days",
            compression="zip",
            enqueue=True  # Async logging
        )

        # 3. ERROR LOG (errors and critical only)
        logger.add(
            self.log_dir / "errors" / "error_{time:YYYY-MM-DD}.log",
            format=lambda record: self._format_record(record) + "\n",
            level="ERROR",
            rotation="10 MB",
            retention="90 days",  # Keep errors longer
            compression="zip",
            enqueue=True
        )

        # 4. AUDIT LOG (important business events)
        logger.add(
            self.log_dir / "audit" / "audit_{time:YYYY-MM-DD}.log",
            format=lambda record: self._format_record(record) + "\n",
            level="INFO",
            rotation="00:00",
            retention="365 days",  # Keep audit logs for 1 year
            compression="zip",
            enqueue=True,
            filter=lambda record: record["extra"].get("audit", False)
        )

        # 5. PERFORMANCE LOG
        logger.add(
            self.log_dir / "performance" / "performance_{time:YYYY-MM-DD}.log",
            format=lambda record: self._format_record(record) + "\n",
            level="INFO",
            rotation="00:00",
            retention="30 days",
            compression="zip",
            enqueue=True,
            filter=lambda record: record["extra"].get("performance", False)
        )

    def set_call_context(self, call_id: str):
        """Set call ID for context tracking"""
        call_context.set(call_id)

    def set_request_context(self, request_id: str):
        """Set request ID for context tracking"""
        request_context.set(request_id)

    def clear_context(self):
        """Clear all context"""
        call_context.set(None)
        request_context.set(None)

    def audit(self, message: str, **kwargs):
        """Log audit event (important business events)"""
        logger.bind(audit=True).info(message, **kwargs)

    def performance(self, operation: str, duration_ms: float, **kwargs):
        """Log performance metric"""
        logger.bind(performance=True).info(
            f"Performance: {operation}",
            extra={
                "operation": operation,
                "duration_ms": duration_ms,
                **kwargs
            }
        )


class PerformanceTimer:
    """Context manager for measuring operation performance"""

    def __init__(self, operation: str, log_func=None, **extra):
        self.operation = operation
        self.log_func = log_func or prod_logger.performance
        self.extra = extra
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000

        if exc_type:
            logger.error(
                f"Operation '{self.operation}' failed after {duration_ms:.2f}ms",
                extra=self.extra
            )
        else:
            self.log_func(self.operation, duration_ms, **self.extra)


# Initialize production logger
def get_production_logger(
    app_name: str = "ai_voice_agent",
    environment: str = None
) -> ProductionLogger:
    """Get or create production logger instance"""

    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")

    return ProductionLogger(
        app_name=app_name,
        environment=environment
    )


# Global instance
prod_logger = get_production_logger()
log = logger  # Alias for backward compatibility


# Convenience functions
def set_call_id(call_id: str):
    """Set call ID for tracking"""
    prod_logger.set_call_context(call_id)


def set_request_id(request_id: str):
    """Set request ID for tracking"""
    prod_logger.set_request_context(request_id)


def clear_context():
    """Clear tracking context"""
    prod_logger.clear_context()


def audit_log(message: str, **kwargs):
    """Log audit event"""
    prod_logger.audit(message, **kwargs)


def perf_timer(operation: str, **extra):
    """Create performance timer context manager"""
    return PerformanceTimer(operation, **extra)
