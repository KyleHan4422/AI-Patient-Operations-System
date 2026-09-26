"""G0 at the route: an emergency is answered whatever else is broken.

The filter itself is tested in test_emergency.py. What is tested here is where
it sits -- ahead of the rate limit, the model, the tools and the graph -- and
what an emergency turn leaves behind: a transcript row for staff, a checkpoint
the next turn continues from, and no booking still waiting for a "yes".
"""

from __future__ import annotations

import uuid

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from patient_ops.db.transcript import TurnRecord
from patient_ops.graph.build import build_graph
from patient_ops.graph.turn import UNRECORDED, Final, Stage, emergency_turn, turn_config
from patient_ops.guardrails.emergency import screen
from tests.factories import booked_count
from tests.fakes import ListRecorder, PinnedIntent, TimingOutModel
from tests.test_booking_flow import BookingChat, offer_cleanings
from tests.test_chat_api import chat_settings, parse_sse, running_app, say  # noqa: F401
from tests.test_degradation import degraded_settings, limited_settings  # noqa: F401

EMERGENCY = "my face is swollen up to my eye and I can't swallow"


def assert_emergency_stream(events, category: str = "er") -> None:
    assert [e.name for e in events] == ["meta", "stage", "done"], events
    assert events[1].data["intent"] == "emergency"
    done = events[-1].data
    assert done["guardrail"] == {"id": "G0", "category": category}
    assert "911" in done["text"].splitlines()[0]


async def test_an_emergency_is_answered_without_the_model(chat_settings):  # noqa: F811
    """TimingOutModel fails every call: had anything asked it, this would be
    an error event."""
    async with running_app(chat_settings, TimingOutModel()) as client:
        events = await say(client, EMERGENCY)
        thread_id = events[0].data["thread_id"]
        history = (await client.get(f"/api/chat/threads/{thread_id}/messages")).json()

    assert_emergency_stream(events)
    assert events[-1].data["degraded"] == []
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == EMERGENCY
    assert history[1]["guardrail"] == "G0", "the transcript says which reply was fixed text"


async def test_an_emergency_is_answered_with_no_model_configured(chat_settings):  # noqa: F811
    settings = chat_settings.model_copy(
        update={"llm_provider": "openai", "openai_api_key": SecretStr("")}
    )
    async with running_app(settings) as client:
        ordinary = await client.post("/api/chat/turn", json={"message": "hi"})
        events = await say(client, EMERGENCY)
    assert ordinary.status_code == 503, "the contrast: anything else is refused"
    assert_emergency_stream(events)


async def test_an_emergency_is_answered_over_the_rate_limit(limited_settings):  # noqa: F811
    async with running_app(limited_settings, PinnedIntent()) as client:
        for _ in range(2):
            await client.post("/api/chat/turn", json={"message": "hi"})
        refused = await client.post("/api/chat/turn", json={"message": "hi"})
        events = await say(client, EMERGENCY)
        still_refused = await client.post("/api/chat/turn", json={"message": "hi"})

    assert refused.status_code == 429
    assert_emergency_stream(events)
    assert still_refused.status_code == 429, "an emergency does not refill the bucket"


async def test_an_emergency_is_answered_with_redis_down(degraded_settings):  # noqa: F811
    async with running_app(degraded_settings, TimingOutModel()) as client:
        events = await say(client, EMERGENCY)
    assert_emergency_stream(events)
    assert events[-1].data["degraded"] == [], "the rate limiter was never asked"


async def test_the_next_turn_remembers_the_emergency(chat_settings):  # noqa: F811
    async with running_app(chat_settings, PinnedIntent()) as client:
        first = await say(client, EMERGENCY)
        thread_id = first[0].data["thread_id"]
        second = await say(client, "ok thanks", thread_id)
        history = (await client.get(f"/api/chat/threads/{thread_id}/messages")).json()

    assert second[-1].name == "done"
    assert [m["role"] for m in history] == ["user", "assistant"] * 2
    assert history[1]["content"] == first[-1].data["text"]


# ---------------------------------------------------------------------------
# The turn itself, without HTTP
# ---------------------------------------------------------------------------
class FailingRecorder:
    async def record_turn(self, turn: TurnRecord) -> None:
        raise OSError("postgres is down")


async def _run(graph, recorder, text: str, thread_id: uuid.UUID) -> list[Stage | Final]:
    match = screen(text)
    assert match is not None
    return [
        event
        async for event in emergency_turn(
            graph,
            recorder,
            match=match,
            text=text,
            thread_id=thread_id,
            channel="web",
            request_id="req-test",
            clinic_phone="(212) 555-0199",
        )
    ]


async def test_a_failed_write_still_answers_and_says_it_was_not_recorded():
    graph = build_graph(InMemorySaver())
    events = await _run(graph, FailingRecorder(), EMERGENCY, uuid.uuid4())

    final = events[-1]
    assert isinstance(final, Final)
    assert "911" in final.text
    assert final.degraded == (UNRECORDED,)


async def test_the_reply_and_the_record_agree():
    graph, recorder, thread_id = build_graph(InMemorySaver()), ListRecorder(), uuid.uuid4()
    events = await _run(graph, recorder, "I knocked out my front tooth", thread_id)

    final = events[-1]
    assert isinstance(final, Final) and final.guardrail == {
        "id": "G0",
        "category": "urgent_dental",
    }
    (turn,) = recorder.turns
    assert turn.assistant_text == final.text
    assert turn.meta["rule"] == "knocked_out_tooth"
    values = (await graph.aget_state(turn_config(thread_id))).values
    assert [m.text for m in values["messages"]] == [turn.user_text, final.text]
    assert values["intent"] == "emergency"


async def test_an_emergency_ends_a_booking_so_a_later_yes_writes_nothing(
    session_factory, clinic, calendar, coordinator
):
    """At the read-back, the patient reports an emergency. The "yes" that
    follows answers the emergency reply, not "Shall I book it?"."""
    chat = BookingChat(session_factory, calendar, coordinator)
    await offer_cleanings(chat)
    assert (await chat.say("the second one")).endswith("Shall I book it?")

    await _run(chat.graph, chat.recorder, EMERGENCY, chat.thread_id)
    assert (await chat.state())["booking"] is None

    # Labelled small talk, which is exactly what continues an open booking --
    # so had the draft survived, this "yes" would have written it.
    reply = await chat.say("yes", model=PinnedIntent())
    assert await booked_count(session_factory) == 0
    assert "booked" not in reply.lower()
