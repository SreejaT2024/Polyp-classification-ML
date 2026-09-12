"""Logging Utility Module.

Provides a custom logger setup that outputs formatted logs to both standard stream
and log files with rotating file handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from config.config import LogConfig, PathConfig


def get_logger(
    name: str = "polyp_pipeline",
    log_config: Optional[LogConfig] = None,
    path_config: Optional[PathConfig] = None,
) -> logging.Logger:
    """Configures and returns a logger instance.

    Args:
        name: Name of the logger module.
        log_config: Optional LogConfig dataclass instance.
        path_config: Optional PathConfig dataclass instance.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if already configured
    if logger.hasHandlers():
        return logger

    if log_config is None:
        log_config = LogConfig()
    if path_config is None:
        path_config = PathConfig()

    level = getattr(logging, log_config.level.upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt=log_config.log_format,
        datefmt=log_config.date_format,
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    log_file_path = path_config.logs_dir / log_config.log_file_name
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
