"""Logging middleware for TurboServe."""

import logging
import time

logger = logging.getLogger("turboserve")


class RequestLogger:
    """Simple request/response logger."""

    @staticmethod
    def log_request(method: str, path: str, status: int, duration_ms: float):
        logger.info(f"{method} {path} -> {status} ({duration_ms:.0f}ms)")

    @staticmethod
    def log_error(method: str, path: str, error: str):
        logger.error(f"{method} {path} -> ERROR: {error}")
