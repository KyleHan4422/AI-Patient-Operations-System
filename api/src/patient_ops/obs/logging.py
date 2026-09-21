"""Structured logging bootstrap.

Every log line is a dict, not a sentence: `jq`-queryable in dev, ingestible by
a log platform in prod. Phase 12 extends this module with trace and cost
instrumentation; this is the minimum needed to make every later phase
debuggable.
"""

from __future__ import annotations

import logging
import sys
import uuid

import structlog
from starlette.requests import Request
from starlette.types import ASGIApp

from patient_ops.config import Settings

REQUEST_ID_HEADER = "X-Request-ID"


def configure_logging(settings: Settings) -> None:
    """Idempotent: safe to call from both the API and the worker entrypoint."""
    level = getattr(logging, settings.log_level, logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level, force=True)

    shared = [
        # merge_contextvars is what makes request_id appear on every log line
        # emitted anywhere downstream, without threading it through call args.
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer = (
        structlog.dev.ConsoleRenderer(colors=True)  # humans read dev logs
        if settings.is_dev
        else structlog.processors.JSONRenderer()  # machines read the rest
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


class RequestIdMiddleware:
    """Assigns a correlation id to every request and binds it to the log context.

    Uses contextvars rather than thread-locals: under asyncio many requests
    share one thread, so a thread-local would leak one request's id into
    another's log lines.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        async def send_wrapper(message) -> None:  # noqa: ANN001
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((REQUEST_ID_HEADER.lower().encode(), request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            structlog.contextvars.clear_contextvars()


__all__ = [
    "REQUEST_ID_HEADER",
    "RequestIdMiddleware",
    "configure_logging",
    "get_logger",
]
