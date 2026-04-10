"""
utils/logger.py — Observability Layer
Provides structured, layer-aware logging across the entire system.
Architecture principle: Every layer is observable (latency, status, failures).
"""

import logging
import sys
from typing import Optional


def get_layer_logger(layer_name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured Logger for a named pipeline layer.
    Each layer gets its own namespace with consistent formatting.
    Prevents duplicate handlers on repeated calls.
    """
    logger = logging.getLogger(f"mis.{layer_name}")   # mis = meeting intelligence system

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(level)
    logger.propagate = False  # Don't bubble to root logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(name)s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
