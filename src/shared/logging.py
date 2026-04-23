import logging
import re
from typing import ClassVar


class SensitiveDataFilter(logging.Filter):
    """Filter that redacts sensitive data from log messages."""

    # Patterns to redact (database URLs, passwords, etc.)
    SENSITIVE_PATTERNS: ClassVar[list[tuple[str, str]]] = [
        # PostgreSQL connection strings
        (r"postgresql://[^@]+@", "postgresql://***:***@"),
        # Generic password patterns
        (r"password[=:]\s*['\"]?[^'\"&\s]+", "password=***"),
        (r"POSTGRES_PASSWORD[=:]\s*['\"]?[^'\"&\s]+", "POSTGRES_PASSWORD=***"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive data from log message."""
        if record.msg:
            msg = str(record.msg)
            for pattern, replacement in self.SENSITIVE_PATTERNS:
                msg = re.sub(pattern, replacement, msg, flags=re.IGNORECASE)
            record.msg = msg
        return True


def get_logger(name: str) -> logging.Logger:
    """Get a logger with sensitive data filtering.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Only add filter if not already present
    if not any(isinstance(f, SensitiveDataFilter) for f in logger.filters):
        logger.addFilter(SensitiveDataFilter())

    return logger


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application-wide logging.

    Args:
        level: Logging level (default: INFO).
    """
    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add sensitive data filter to root logger
    root_logger = logging.getLogger()
    if not any(isinstance(f, SensitiveDataFilter) for f in root_logger.filters):
        root_logger.addFilter(SensitiveDataFilter())
