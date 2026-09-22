"""Chat over HTTP: one request per turn, the reply streamed back as SSE.

    POST /api/chat/turn                          send one message
    GET  /api/chat/threads/{thread_id}/messages  the transcript so far

The turn endpoint streams Server-Sent Events over POST (a browser's EventSource
only does GET, so the web client reads the stream with fetch). Every stream is
exactly:

    event: meta    {"thread_id", "request_id"}        first, always
    event: token   {"text"}                           zero or more; provisional
    event: done    {"text", "thread_id"}              success; `text` is authoritative
      -- or --
    event: error   {"code", "message", "request_id"}  failure; `code` is an ErrorCode

`done.text` is the reply of record. Clients show tokens as they arrive, then
replace them with it: some replies (a booking confirmation, from Phase 6) are
filled from a template and never stream at all.

Errors found before the stream starts get a real status code (422 bad input,
503 no model configured). Once the 200 has been sent it cannot be taken back,
so a failure mid-turn is an `error` event.

This module translates and nothing more; what a turn *is* lives in graph/turn.py,
so the voice channel (Phase 13) can reuse it.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Literal

import psycopg
import sqlalchemy.exc
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, StringConstraints

from patient_ops.adapters.llm.client import classify_llm_error
from patient_ops.db import repo
from patient_ops.errors import ErrorCode
from patient_ops.graph.context import GraphContext
from patient_ops.graph.turn import Token, run_turn
from patient_ops.obs.logging import get_logger

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


# What the user sees. The code, the request id and the exception go to the log.
USER_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.TRANSIENT: "Sorry, I couldn't answer just now. Please try again in a moment.",
    ErrorCode.PERMANENT: "Sorry, I can't answer right now. Please contact the clinic directly.",
    ErrorCode.UNKNOWN: "Sorry, something went wrong on our side. Please try again.",
}

_DATABASE_UNAVAILABLE = (
    psycopg.OperationalError,  # includes psycopg_pool.PoolTimeout
    sqlalchemy.exc.OperationalError,
    sqlalchemy.exc.TimeoutError,
)


def classify_turn_error(exc: BaseException) -> ErrorCode:
    code = classify_llm_error(exc)
    if code is not None:
        return code
    if isinstance(exc, _DATABASE_UNAVAILABLE):
        return ErrorCode.TRANSIENT
    return ErrorCode.UNKNOWN


def require_chat_model(request: Request) -> BaseChatModel:
    """Resolved before the stream opens, so "not configured" is a real 503."""
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


@router.post("/turn", response_class=EventSourceResponse)
async def turn(
    body: TurnRequest,
    request: Request,
    chat_model: Annotated[BaseChatModel, Depends(require_chat_model)],
) -> AsyncIterator[ServerSentEvent]:
    state = request.app.state
    thread_id = body.thread_id or uuid.uuid4()
    request_id = structlog.contextvars.get_contextvars().get("request_id", "")
    structlog.contextvars.bind_contextvars(thread_id=str(thread_id))

    yield ServerSentEvent(event="meta", data={"thread_id": thread_id, "request_id": request_id})

    # Phase 7: the G0 emergency filter goes here -- before any model sees the
    # message, so an emergency never depends on a classifier.

    context = GraphContext(
        thread_id=thread_id,
        request_id=request_id,
        chat_model=chat_model,
        recorder=state.transcript,
        channel=body.channel,
        history_limit=state.settings.chat_history_max_messages,
    )
    started = time.perf_counter()
    try:
        async for event in run_turn(state.graph, text=body.message, context=context):
            if isinstance(event, Token):
                yield ServerSentEvent(event="token", data={"text": event.text})
            else:
                log.info("turn_complete", latency_ms=round((time.perf_counter() - started) * 1000))
                yield ServerSentEvent(
                    event="done", data={"text": event.text, "thread_id": thread_id}
                )
    except Exception as exc:
        # Not BaseException: a client disconnect cancels the turn, and that
        # cancellation must propagate rather than be answered.
        code = classify_turn_error(exc)
        log.error("turn_failed", code=code, error_type=type(exc).__name__, exc_info=True)
        yield ServerSentEvent(
            event="error",
            data={"code": code, "message": USER_MESSAGES[code], "request_id": request_id},
        )


@router.get("/threads/{thread_id}/messages", response_model=list[TranscriptEntry])
async def thread_messages(thread_id: uuid.UUID, request: Request) -> list[TranscriptEntry]:
    async with request.app.state.session_factory() as session:
        messages = await repo.get_transcript(session, thread_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="No conversation with this thread_id.")
    return [
        TranscriptEntry(role=m.role, content=m.content, created_at=m.created_at)  # type: ignore[arg-type]
        for m in messages
    ]
