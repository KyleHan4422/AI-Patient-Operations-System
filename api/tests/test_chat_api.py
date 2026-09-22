"""The chat endpoints over real HTTP semantics, a real Postgres and the echo model.

The app is driven through its lifespan -- httpx's ASGITransport does not run
it -- so these tests exercise exactly what `make dev` builds: the checkpointer
pool, the compiled graph, the transcript writer.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from patient_ops.api.routes_chat import MAX_MESSAGE_CHARS
from patient_ops.config import Settings
from patient_ops.main import create_app
from tests.fakes import TimingOutModel


@dataclass(frozen=True)
class Event:
    name: str
    data: dict


def parse_sse(body: str) -> list[Event]:
    events = []
    for block in body.split("\n\n"):
        name, data = None, []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data.append(line.removeprefix("data:").strip())
            # lines starting with ":" are keep-alive comments
        if name:
            events.append(Event(name, json.loads("\n".join(data))))
    return events


@asynccontextmanager
async def running_app(
    settings: Settings, chat_model: BaseChatModel | None = None
) -> AsyncIterator[AsyncClient]:
    """Start the app as uvicorn would -- lifespan and all -- and stop it on exit."""
    app = create_app(settings, chat_model=chat_model)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client


async def say(client: AsyncClient, message: str, thread_id: str | None = None) -> list[Event]:
    body = {"message": message} | ({"thread_id": thread_id} if thread_id else {})
    response = await client.post("/api/chat/turn", json=body)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


def done_text(events: list[Event]) -> str:
    assert events[-1].name == "done", events
    return events[-1].data["text"]


@pytest.fixture
def chat_settings(test_settings: Settings, engine) -> Settings:
    # `engine` is requested for its teardown: every table, the checkpoints
    # included, is emptied after each test.
    return test_settings.model_copy(update={"llm_provider": "fake"})


# ---------------------------------------------------------------------------
# The stream
# ---------------------------------------------------------------------------
async def test_a_turn_streams_meta_then_tokens_then_done(chat_settings):
    async with running_app(chat_settings) as client:
        events = await say(client, "hello there")

    names = [e.name for e in events]
    assert names[0] == "meta" and names[-1] == "done"
    assert set(names[1:-1]) == {"token"} and len(names) > 3
    assert "".join(e.data["text"] for e in events[1:-1]) == done_text(events)
    assert events[0].data["request_id"], "every stream carries its request id"


async def test_first_turn_mints_a_thread_id_that_later_turns_reuse(chat_settings):
    async with running_app(chat_settings) as client:
        first = await say(client, "My name is Kyle")
        thread_id = first[0].data["thread_id"]
        uuid.UUID(thread_id)  # a real UUID, minted by the server
        second = await say(client, "What is my name?", thread_id)

    assert second[0].data["thread_id"] == thread_id
    assert '"My name is Kyle"' in done_text(second)


@pytest.mark.parametrize(
    "body",
    [
        {"message": ""},
        {"message": "   \n\t "},
        {"message": "x" * (MAX_MESSAGE_CHARS + 1)},
        {"message": "hi", "thread_id": "not-a-uuid"},
        {"message": "hi", "channel": "sms"},
        {},
    ],
    ids=["empty", "whitespace", "too-long", "bad-thread-id", "bad-channel", "missing"],
)
async def test_bad_input_is_a_422_before_any_stream(chat_settings, body):
    async with running_app(chat_settings) as client:
        response = await client.post("/api/chat/turn", json=body)
    assert response.status_code == 422


async def test_no_model_configured_is_a_real_503(chat_settings):
    settings = chat_settings.model_copy(
        update={"llm_provider": "openai", "openai_api_key": SecretStr("")}
    )
    async with running_app(settings) as client:
        response = await client.post("/api/chat/turn", json={"message": "hi"})
    assert response.status_code == 503
    assert "LLM_PROVIDER=fake" in response.json()["detail"]["message"]


async def test_a_model_failure_mid_stream_is_an_error_event(chat_settings):
    async with running_app(chat_settings, chat_model=TimingOutModel()) as client:
        events = await say(client, "hello")
        thread_id = events[0].data["thread_id"]
        history = await client.get(f"/api/chat/threads/{thread_id}/messages")

    assert [e.name for e in events] == ["meta", "error"]
    assert events[-1].data["code"] == "transient"
    assert "try again" in events[-1].data["message"]
    assert "Timeout" not in events[-1].data["message"], "internals stay in the log"
    assert history.status_code == 404, "a failed turn is not recorded"


# ---------------------------------------------------------------------------
# The transcript
# ---------------------------------------------------------------------------
async def test_history_of_an_unknown_thread_is_404(chat_settings):
    async with running_app(chat_settings) as client:
        response = await client.get(f"/api/chat/threads/{uuid.uuid4()}/messages")
    assert response.status_code == 404


async def test_history_lists_every_message_in_order(chat_settings):
    async with running_app(chat_settings) as client:
        first = await say(client, "one")
        thread_id = first[0].data["thread_id"]
        second = await say(client, "two", thread_id)
        history = (await client.get(f"/api/chat/threads/{thread_id}/messages")).json()

    assert [(m["role"], m["content"]) for m in history] == [
        ("user", "one"),
        ("assistant", done_text(first)),
        ("user", "two"),
        ("assistant", done_text(second)),
    ]


# ---------------------------------------------------------------------------
# ★ The Phase 2 acceptance criterion
# ---------------------------------------------------------------------------
async def test_a_restarted_app_remembers_the_conversation(chat_settings):
    """Three turns, a full shutdown, a brand-new app, a fourth turn.

    The second app shares nothing in memory with the first -- new process
    state, new connection pools, new compiled graph. If turn four still sees
    turn one, the memory can only have come from Postgres.
    """
    async with running_app(chat_settings) as client:
        first = await say(client, "My name is Kyle")
        thread_id = first[0].data["thread_id"]
        await say(client, "I have a toothache", thread_id)
        await say(client, "I prefer mornings", thread_id)
    # -- the first app is fully shut down here: lifespan exited, pools closed --

    async with running_app(chat_settings) as client:
        fourth = await say(client, "What did I say first?", thread_id)
        history = (await client.get(f"/api/chat/threads/{thread_id}/messages")).json()

    reply = done_text(fourth)
    assert "4 message(s)" in reply
    assert '"My name is Kyle"' in reply
    assert len(history) == 8
