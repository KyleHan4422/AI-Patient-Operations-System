"""ARQ worker: the non-critical path, enforced by a process boundary.

Nothing here can block an HTTP response. Confirmation sends are slow, failable
and irrelevant to whether a booking succeeded -- so they run in a different
process, reached through a Redis queue.

The distinction between critical and non-critical work is not documented in a
comment; it is enforced by which process the code runs in.

Phase 9 replaces `ping` with the real send_confirmation job, its retry policy
and its dead-letter queue.
"""

from __future__ import annotations

from typing import Any

from arq.connections import RedisSettings

from patient_ops import __version__
from patient_ops.config import get_settings
from patient_ops.obs.logging import configure_logging, get_logger

log = get_logger(__name__)


async def ping(ctx: dict[str, Any]) -> str:
    """Placeholder job. Proves the queue round-trips before Phase 9 needs it."""
    log.info("job_ping", job_id=ctx.get("job_id"), job_try=ctx.get("job_try"))
    return "pong"


async def on_startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(settings)
    log.info("worker_startup", version=__version__, app_env=settings.app_env)


async def on_shutdown(ctx: dict[str, Any]) -> None:
    log.info("worker_shutdown")


def _redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(get_settings().redis_url)


class WorkerSettings:
    """Discovered by `arq patient_ops.worker.WorkerSettings`."""

    functions = [ping]
    on_startup = staticmethod(on_startup)
    on_shutdown = staticmethod(on_shutdown)
    redis_settings = _redis_settings()
    max_tries = 3  # Phase 9 relies on this for the confirmation retry budget
    job_timeout = 30
