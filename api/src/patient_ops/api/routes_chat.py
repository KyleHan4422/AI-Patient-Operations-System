"""Chat over HTTP: one request per turn, the reply streamed back as SSE.

    POST /api/chat/turn                          send one message
    GET  /api/chat/threads/{thread_id}/messages  the transcript so far

The turn endpoint streams Server-Sent Events over POST (a browser's EventSource
only does GET, so the web client reads the stream with fetch). Every stream is
exactly:

    event: meta    {"thread_id", "request_id"}        first, always
    event: stage   {"intent"}                         once, when the branch is known
    event: token   {"text"}                           zero or more; provisional
    event: done    {"text", "thread_id", "degraded"}  success; `text` is authoritative
                   (+ "guardrail": {"id", "category"} when a guardrail answered)
      -- or --
    event: error   {"code", "message", "request_id"}  failure; `code` is an ErrorCode

`done.text` is the reply of record. Clients show tokens as they arrive, then
replace them with it. Most replies never stream a token at all: an answer about
the clinic is checked against its evidence before it is said, and a booking
confirmation is filled from the row the database holds. `stage` is
what a client shows meanwhile -- it says which branch the turn took, from the
classifier's own output rather than from a guess.

`done.degraded` lists the fallbacks the turn took because an optional
dependency was down (e.g. ["rate_limit"] when Redis is unreachable). Usually
empty. The reply is still the reply; this is so a degraded system is never a
silent one.

Errors found before the stream starts get a real status code (422 bad input,
429 too many turns, 503 no model configured). Once the 200 has been sent it cannot be taken back,
so a failure mid-turn is an `error` event.

Except for an emergency. G0 (guardrails/emergency.py) reads the message before
anything else is resolved; when it matches, the rate limit, the model and the
tools are not consulted at all, and the reply is fixed text: stage
`emergency`, then `done` with `guardrail`. A patient whose face is swelling
toward their eye is not told "too many messages", or "no model configured".

This module translates and nothing more; what a turn *is* lives in graph/turn.py,
so the voice channel (Phase 13) can reuse it.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from typing import Annotated, Literal

import psycopg
import sqlalchemy.exc
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, StringConstraints

from patient_ops.adapters.calendar.faults import build_calendar
from patient_ops.adapters.llm.client import classify_llm_error
from patient_ops.adapters.llm.embeddings import embedding_model_name, min_score_for
from patient_ops.db import repo
from patient_ops.degradation import DegradedModes
from patient_ops.domain.availability import SchedulingPolicy
from patient_ops.errors import ErrorCode, ToolError
from patient_ops.graph.context import GraphContext
from patient_ops.graph.turn import Final, Stage, Token, emergency_turn, run_turn
from patient_ops.guardrails.emergency import EmergencyMatch, screen
from patient_ops.obs.logging import get_logger
from patient_ops.redis_layer.holds import SlotHolds
from patient_ops.redis_layer.idempotency import InFlightDedup
from patient_ops.tools.booking import BookingDesk
from patient_ops.tools.registry import ReadOnlyToolset

log = get_logger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Bounds what one message can cost: every character is re-sent to the model
# on later turns, until it falls out of the history window.
MAX_MESSAGE_CHARS = 2000


class TurnRequest(BaseModel):
    thread_id: uuid.UUID | None = None  # omitted on a first turn: the server mints one
    message: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_CHARS)
    ]
    channel: Literal["web", "voice"] = "web"


class TranscriptEntry(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    # "G0" on a reply a guardrail gave instead of the graph, so a reloaded
    # conversation shows it as it was first shown.
    guardrail: str | None = None


# What the user sees. The code, the request id and the exception go to the log.
# Read through user_message(), which falls back rather than raising: a missing
# entry must not turn a handled failure into an unhandled one.
USER_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.TRANSIENT: "Sorry, I couldn't answer just now. Please try again in a moment.",
    ErrorCode.PERMANENT: "Sorry, I can't answer right now. Please contact the clinic directly.",
    ErrorCode.UNKNOWN: "Sorry, something went wrong on our side. Please try again.",
}


def user_message(code: ErrorCode) -> str:
    return USER_MESSAGES.get(code, USER_MESSAGES[ErrorCode.UNKNOWN])


_DATABASE_UNAVAILABLE = (
    psycopg.OperationalError,  # includes psycopg_pool.PoolTimeout
    sqlalchemy.exc.OperationalError,
    sqlalchemy.exc.TimeoutError,
)


def classify_turn_error(exc: BaseException) -> ErrorCode:
    # A ToolError has already been classified by the layer that raised it --
    # "the corpus was never ingested" is permanent, and re-deriving that here
    # from the exception type would lose it.
    if isinstance(exc, ToolError):
        return exc.code
    code = classify_llm_error(exc)
    if code is not None:
        return code
    if isinstance(exc, _DATABASE_UNAVAILABLE):
        return ErrorCode.TRANSIENT
    return ErrorCode.UNKNOWN


def screen_emergency(body: TurnRequest) -> EmergencyMatch | None:
    """G0, first. Every other dependency of a turn takes this one and stands
    aside when it matched, so nothing that can fail runs ahead of it."""
    return screen(body.message)


Emergency = Annotated[EmergencyMatch | None, Depends(screen_emergency)]


def require_chat_model(request: Request, emergency: Emergency) -> BaseChatModel | None:
    """Resolved before the stream opens, so "not configured" is a real 503."""
    if emergency is not None:
        return None
    model: BaseChatModel | None = request.app.state.chat_model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": ErrorCode.PERMANENT,
                "message": "No language model is configured. "
                "Set OPENAI_API_KEY, or LLM_PROVIDER=fake to run without one.",
            },
        )
    return model


def turn_degradation() -> DegradedModes:
    """This turn's record of fallbacks. One instance per request: FastAPI
    caches a dependency within a request, so the rate limiter and the graph
    context below receive the same one."""
    return DegradedModes()


async def enforce_rate_limit(
    request: Request,
    degraded: Annotated[DegradedModes, Depends(turn_degradation)],
    emergency: Emergency,
) -> None:
    """R5: a token bucket per client, checked before the stream opens -- so a
    client over its budget gets a real 429 and a Retry-After, and no model is
    called on its behalf. Fails open: with Redis down the turn goes ahead and
    records `rate_limit` among its degraded modes.

    An emergency is neither limited nor counted: it costs no model call, and
    someone who has just sent twenty panicked messages is the last person to
    tell to wait.
    """
    settings = request.app.state.settings
    if not settings.rate_limit_enabled or emergency is not None:
        return
    # The socket peer. Behind a reverse proxy this is the proxy, and the key
    # must come from the X-Forwarded-For entry the proxy itself appended --
    # never from the header as the client sent it, which anyone can forge.
    client = request.client.host if request.client else "unknown"
    decision = await request.app.state.rate_limiter.take("chat", client, degraded)
    if not decision.allowed:
        log.info("rate_limited", client=client, retry_after_s=decision.retry_after_s)
        raise HTTPException(
            status_code=429,
            detail={
                "code": ErrorCode.TRANSIENT,
                "message": "Too many messages in a short time. Please wait a moment.",
            },
            headers={"Retry-After": str(decision.retry_after_s)},
        )


def require_toolset(request: Request, emergency: Emergency) -> ReadOnlyToolset | None:
    """The turn's read-only tools, built fresh: it records what this turn used.

    Resolved before the stream opens, like the model, so an embedder that is
    not configured and a model with no calibrated threshold are both a 503 with
    a reason -- not an error event three seconds into an answer.
    """
    if emergency is not None:
        return None
    state = request.app.state
    settings = state.settings
    if state.embeddings is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": ErrorCode.PERMANENT,
                "message": "No embedding model is configured, so the clinic's documents "
                "cannot be searched. Set OPENAI_API_KEY, or LLM_PROVIDER=fake.",
            },
        )
    model = embedding_model_name(settings)
    try:
        min_score = min_score_for(model, settings.rag_min_score)
    except ToolError as exc:
        raise HTTPException(
            status_code=503, detail={"code": exc.code, "message": exc.detail}
        ) from exc
    return ReadOnlyToolset(
        session_factory=state.session_factory,
        embeddings=state.embeddings,
        embedding_model=model,
        min_score=min_score,
        top_k=settings.rag_top_k,
        tz=settings.clinic_tz,
    )


def booking_desk(
    request: Request,
    degraded: Annotated[DegradedModes, Depends(turn_degradation)],
    emergency: Emergency,
) -> BookingDesk | None:
    """What the booking path may reach this turn, and nothing an agent can.

    Built per turn around long-lived parts: the breaker and the Redis
    connection are the process's, while the calendar wrapper, the holds and
    the dedup carry this turn's DegradedModes, so a fallback taken anywhere on
    the booking path shows up in this turn's `done.degraded`.
    """
    if emergency is not None:
        return None
    state = request.app.state
    settings = state.settings
    return BookingDesk(
        session_factory=state.session_factory,
        calendar=build_calendar(
            settings,
            state.session_factory,
            breaker=state.calendar_breaker,
            degraded=degraded,
            injector=state.fault_injector,
        ),
        holds=SlotHolds(
            state.coordinator,
            ttl=timedelta(seconds=settings.hold_ttl_s),
            step=timedelta(minutes=settings.slot_step_min),
            degraded=degraded,
        ),
        dedup=InFlightDedup(
            state.coordinator,
            inflight_ttl_s=settings.idempotency_inflight_ttl_s,
            wait_s=settings.idempotency_wait_s,
            degraded=degraded,
        ),
        tz=settings.clinic_tz,
        policy=SchedulingPolicy(
            step=timedelta(minutes=settings.slot_step_min),
            min_lead=timedelta(minutes=settings.booking_min_lead_min),
            max_horizon=timedelta(days=settings.booking_max_horizon_days),
        ),
    )


@router.post(
    "/turn", response_class=EventSourceResponse, dependencies=[Depends(enforce_rate_limit)]
)
async def turn(
    body: TurnRequest,
    request: Request,
    degraded: Annotated[DegradedModes, Depends(turn_degradation)],
    emergency: Emergency,
    chat_model: Annotated[BaseChatModel | None, Depends(require_chat_model)],
    toolset: Annotated[ReadOnlyToolset | None, Depends(require_toolset)],
    booking: Annotated[BookingDesk | None, Depends(booking_desk)],
) -> AsyncIterator[ServerSentEvent]:
    state = request.app.state
    thread_id = body.thread_id or uuid.uuid4()
    request_id = structlog.contextvars.get_contextvars().get("request_id", "")
    structlog.contextvars.bind_contextvars(thread_id=str(thread_id))

    yield ServerSentEvent(event="meta", data={"thread_id": thread_id, "request_id": request_id})

    if emergency is not None:
        # G0 matched before any model saw the message: fixed text, no graph.
        events = emergency_turn(
            state.graph,
            state.transcript,
            match=emergency,
            text=body.message,
            thread_id=thread_id,
            channel=body.channel,
            request_id=request_id,
            clinic_phone=state.settings.clinic_phone,
        )
        async for event in events:
            if isinstance(event, Stage):
                yield ServerSentEvent(event="stage", data={"intent": event.intent})
            else:
                yield _done(event, thread_id)
        return

    assert chat_model is not None and toolset is not None, "resolved when G0 did not match"
    context = GraphContext(
        thread_id=thread_id,
        request_id=request_id,
        chat_model=chat_model,
        recorder=state.transcript,
        channel=body.channel,
        history_limit=state.settings.chat_history_max_messages,
        toolset=toolset,
        booking=booking,
        max_tool_rounds=state.settings.agent_max_tool_rounds,
        degraded=degraded,
    )
    started = time.perf_counter()
    try:
        async for event in run_turn(state.graph, text=body.message, context=context):
            if isinstance(event, Stage):
                yield ServerSentEvent(event="stage", data={"intent": event.intent})
            elif isinstance(event, Token):
                yield ServerSentEvent(event="token", data={"text": event.text})
            else:
                log.info(
                    "turn_complete",
                    latency_ms=round((time.perf_counter() - started) * 1000),
                    degraded=list(event.degraded),
                )
                yield _done(event, thread_id)
    except Exception as exc:
        # Not BaseException: a client disconnect cancels the turn, and that
        # cancellation must propagate rather than be answered.
        code = classify_turn_error(exc)
        log.error("turn_failed", code=code, error_type=type(exc).__name__, exc_info=True)
        yield ServerSentEvent(
            event="error",
            data={"code": code, "message": user_message(code), "request_id": request_id},
        )


def _done(event: Final, thread_id: uuid.UUID) -> ServerSentEvent:
    data: dict = {"text": event.text, "thread_id": thread_id, "degraded": list(event.degraded)}
    if event.guardrail is not None:
        data["guardrail"] = event.guardrail
    return ServerSentEvent(event="done", data=data)


@router.get("/threads/{thread_id}/messages", response_model=list[TranscriptEntry])
async def thread_messages(thread_id: uuid.UUID, request: Request) -> list[TranscriptEntry]:
    async with request.app.state.session_factory() as session:
        messages = await repo.get_transcript(session, thread_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="No conversation with this thread_id.")
    return [
        TranscriptEntry(
            role=m.role,  # type: ignore[arg-type]
            content=m.content,
            created_at=m.created_at,
            guardrail=(m.meta or {}).get("guardrail") if m.role == "assistant" else None,
        )
        for m in messages
    ]
