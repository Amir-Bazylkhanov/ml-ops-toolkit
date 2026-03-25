"""Structured JSON logging middleware for FastAPI / Starlette.

Each HTTP request receives a UUID correlation ID that is:

* Carried through the application logger as a ``correlation_id`` field in
  every JSON log record emitted during the request lifetime.
* Returned to the caller in the ``X-Correlation-ID`` response header.

Usage::

    from fastapi import FastAPI
    from app.middleware.logging_middleware import LoggingMiddleware

    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

Log records are written to stdout in JSON format via ``python-json-logger``.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from pythonjsonlogger import jsonlogger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

def _build_json_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that emits structured JSON to stdout.

    The formatter includes the standard ``asctime``, ``levelname``, ``name``,
    and ``message`` fields plus any *extra* keys injected at call sites.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        level: Minimum log level.  Defaults to ``logging.INFO``.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


_logger = _build_json_logger(__name__)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class LoggingMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that logs every HTTP request as a structured JSON record.

    Each request is assigned a UUID v4 ``correlation_id`` which is:

    * Logged alongside ``method``, ``path``, ``status_code``, and
      ``duration_ms`` at the INFO level upon response completion.
    * Written back to the client in the ``X-Correlation-ID`` response header.

    Args:
        app: The downstream ASGI application.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process a single request/response cycle.

        Generates a correlation ID, invokes the downstream handler, and
        emits a structured log record with timing information.

        Args:
            request: Incoming Starlette/FastAPI request object.
            call_next: Callable that passes the request to the next handler
                in the middleware chain.

        Returns:
            The response from the downstream application, augmented with the
            ``X-Correlation-ID`` header.
        """
        correlation_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        response: Response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1_000

        _logger.info(
            "request completed",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 3),
            },
        )

        response.headers["X-Correlation-ID"] = correlation_id
        return response
