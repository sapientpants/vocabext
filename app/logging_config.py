"""Logging configuration module with optional file output."""

import logging
import sys
from logging.handlers import RotatingFileHandler

from app.config import settings

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FORMAT_DETAILED = (
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)


def setup_logging() -> None:
    """Configure application logging based on settings.

    Sets up console logging (always) and file logging (if enabled).
    File logging uses rotation to prevent unbounded growth.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level))

    # Remove existing handlers to avoid duplicates on reload
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if settings.log_file_enabled:
        log_path = settings.resolved_log_file_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=log_path,
            maxBytes=settings.log_file_max_bytes,
            backupCount=settings.log_file_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT_DETAILED))
        root_logger.addHandler(file_handler)
