"""Logging utilities."""
from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path

from config import LOG_FILE, RESULTS_DIR
from utils.helpers import create_directories


def setup_logger(name: str, log_file: str | Path = LOG_FILE) -> Logger:
    """Configure and return a logger."""
    create_directories()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

        file_handler = logging.FileHandler(Path(log_file), encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


__all__ = ["setup_logger"]

