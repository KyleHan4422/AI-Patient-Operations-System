"""A booking, from "please book me a cleaning" to the appointment row.

A real Postgres, a real Redis, the real calendar adapter behind a real breaker
-- and the offline model, which reads each message by keyword. What is
asserted here does not depend on the model's judgement: that nothing is
written without a read-back and a yes, that only an offered time can be
chosen, that "you're booked" is only said about a row that was read back, and
that every failure on the way is told to the patient as what it is.

Clock: FIXED_NOW is Thursday 2026-10-01 08:00 in New York. The hygienist works
Mon-Fri 09-17, the minimum lead is two hours, so the first three cleanings
offered -- earliest per day -- are Thu 10:00, Fri 09:00 and Mon 09:00.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import date, datetime, timedelta

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from patient_ops.adapters.calendar.base import BookingRequest, CalendarProvider, Slot
from patient_ops.adapters.calendar.breaker import BreakerCalendar
from patient_ops.adapters.calendar.faults import FaultyCalendar
from patient_ops.adapters.llm.fake import EchoChatModel
from patient_ops.db.models import Appointment, Patient
from patient_ops.db.session import build_session_factory
from patient_ops.degradation import DegradedModes
from patient_ops.faults import FaultInjector
from patient_ops.graph import replies
from patient_ops.graph.build import build_graph
from patient_ops.graph.context import GraphContext
from patient_ops.graph.turn import Final, run_turn, turn_config
from patient_ops.redis_layer.breaker import BreakerEvent, FailoverBreaker, build_breaker
from patient_ops.redis_layer.client import Coordinator
from patient_ops.redis_layer.holds import SlotHolds
from patient_ops.redis_layer.idempotency import InFlightDedup
from patient_ops.tools.booking import BookingDesk, idempotency_key
from tests.factories import FIXED_NOW, TZ, booked_count, local
from tests.fakes import ListRecorder, PinnedIntent, is_intent_call
from tests.test_chat_api import done_text, running_app, say

PAT_PHONE = "(212) 555-0100"  # Pat Test, tests/factories.py
THU, FRI, MON = date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 5)


class BookingChat:
    """One conversation over the booking path, one fresh desk per turn -- as
    api/routes_chat.py builds one per request."""

    def __init__(
        self,
        session_factory,
        calendar: CalendarProvider,
        coordinator: Coordinator,
        *,
        breaker: FailoverBreaker | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.calendar = calendar
        self.coordinator = coordinator
        self.breaker = breaker or build_breaker(
            coordinator, "calendar", failure_threshold=5, cooldown_s=30
        )
        self.graph = build_graph(InMemorySaver())
        self.thread_id = uuid.uuid4()
        self.recorder = ListRecorder()
        self.now = FIXED_NOW
        self.degraded: list[str] = []

    def desk(self, degraded: DegradedModes) -> BookingDesk:
        return BookingDesk(
            session_factory=self.session_factory,
            calendar=BreakerCalendar(self.calendar, self.breaker, degraded=degraded),
            holds=SlotHolds(
                self.coordinator,
                ttl=timedelta(seconds=120),
                step=timedelta(minutes=30),
                degraded=degraded,
            ),
            dedup=InFlightDedup(self.coordinator, inflight_ttl_s=30, wait_s=1, degraded=degraded),
            tz=TZ,
            now=lambda: self.now,
        )

    async def say(self, text: str, model: BaseChatModel | None = None) -> str:
        degraded = DegradedModes()
        context = GraphContext(
            thread_id=self.thread_id,
            request_id="req-test",
            chat_model=model or EchoChatModel(),
            recorder=self.recorder,
            booking=self.desk(degraded),
            degraded=degraded,
        )
        final = None
        async for event in run_turn(self.graph, text=text, context=context):
            if isinstance(event, Final):
                final = event
        assert final is not None
        self.degraded = list(final.degraded)
        return final.text

    async def state(self) -> dict:
        return (await self.graph.aget_state(turn_config(self.thread_id))).values

    async def kind(self) -> str:
        return (await self.state())["answer_kind"]


@pytest.fixture
def chat(session_factory, clinic, calendar, coordinator) -> BookingChat:
    return BookingChat(session_factory, calendar, coordinator)


async def offer_cleanings(chat: BookingChat) -> str:
    assert await chat.say("Please book me a cleaning") == replies.ASK_PHONE
    return await chat.say(PAT_PHONE)


async def appointments(session_factory) -> list[Appointment]:
    async with session_factory() as session:
        return list((await session.scalars(select(Appointment))).all())


# ---------------------------------------------------------------------------
# The whole path
# ---------------------------------------------------------------------------
async def test_a_booking_from_the_first_message_to_the_row(chat, session_factory, clinic):
    offer = await offer_cleanings(chat)
    assert "1. Thu 01 Oct at 10:00 with Hy Gienist, RDH" in offer
    assert "2. Fri 02 Oct at 09:00" in offer and "3. Mon 05 Oct at 09:00" in offer
    assert await chat.kind() == "booking_offer"

    read_back = await chat.say("the second one please")
    assert read_back == (
        "Just to confirm: Cleaning with Hy Gienist, RDH on Fri 02 Oct at 09:00. Shall I book it?"
    )
    assert await booked_count(session_factory) == 0, "choosing is not booking"

    confirmation = await chat.say("yes")
    [row] = await appointments(session_factory)
    assert (row.patient_id, row.provider_id, row.procedure_code) == (
        clinic.patient_id,
        clinic.hygienist_id,
        "CLEANING",
    )
    assert row.start_at == local(FRI, 9) and row.source == "web"
    assert confirmation == (
        "You're booked: Cleaning with Hy Gienist, RDH on Fri 02 Oct at 09:00. "
        f"Your reference is {row.external_ref}."
    )

    state = await chat.state()
    assert state["answer_kind"] == "booking_confirmed"
    assert state["booking"] is None, "a finished booking is not carried into the next turn"
    assert state["appointment_id"] == row.id
    last = chat.recorder.turns[-1]
    assert last.meta["appointment_id"] == row.id
    assert [c.name for c in last.tool_calls] == ["book_appointment", "get_appointment"]


async def test_the_offer_holds_its_slots_and_the_booking_releases_the_rest(chat, redis_client):
    await offer_cleanings(chat)
    owner = str(chat.thread_id)
    assert len(await redis_client.smembers(f"holds:owner:{owner}")) == 6  # 3 hour-long slots

    await chat.say("2")
    await chat.say("yes")
    held_by_us = [
        k for k in await redis_client.keys("hold:*") if await redis_client.get(k) == owner
    ]
    assert len(held_by_us) == 2, "only the booked slot's own cells are left to expire"


async def test_details_given_up_front_are_not_asked_for_again(chat):
    offer = await chat.say(f"Can you book me a cleaning on 2026-10-05? My number is {PAT_PHONE}")
    assert "Mon 05 Oct at 09:00" in offer
    assert "Thu 01 Oct" not in offer, "the patient asked for Monday"


# ---------------------------------------------------------------------------
# Who
# ---------------------------------------------------------------------------
async def test_two_patients_with_one_name_are_told_apart_by_date_of_birth(chat, session_factory):
    async with session_factory() as session, session.begin():
        session.add_all(
            [
                Patient(full_name="Maria Garcia", phone="+12125550101", dob=date(1985, 4, 12)),
                Patient(full_name="Maria Garcia", phone="+12125550102", dob=date(1992, 9, 30)),
            ]
        )
    assert await chat.say("Book me a cleaning, my name is Maria Garcia") == replies.ASK_DOB
    await chat.say("1992-09-30")
    await chat.say("the first one")
    await chat.say("yes")

    async with session_factory() as session:
        second = await session.scalar(select(Patient).where(Patient.phone == "+12125550102"))
    [row] = await appointments(session_factory)
    assert row.patient_id == second.id


async def test_an_unknown_patient_is_sent_to_the_front_desk(chat, session_factory):
    await chat.say("Please book me a cleaning")
    assert await chat.say("(212) 555-0199") == replies.PATIENT_NOT_FOUND
    assert (await chat.state())["booking"] is None
    async with session_factory() as session:
        assert len((await session.scalars(select(Patient))).all()) == 1, "no patient is created"


# ---------------------------------------------------------------------------
# Only an offered time, only after a read-back, only after a yes
# ---------------------------------------------------------------------------
async def test_a_yes_before_any_read_back_books_nothing(chat, session_factory):
    await offer_cleanings(chat)
    assert (await chat.say("yes")).startswith("Here are the next free times")
    assert await booked_count(session_factory) == 0


async def test_a_choice_outside_the_offers_is_not_booked(chat, session_factory):
    await offer_cleanings(chat)
    await chat.say("the third one")
    reply = await chat.say("yes", model=ProposalModel(proposal={"choice": 7, "confirmed": "yes"}))
    assert reply.startswith("You're booked"), "an out-of-range number changes nothing"
    [row] = await appointments(session_factory)
    assert row.start_at == local(MON, 9), "the slot read back, not one the model named"


async def test_no_after_the_read_back_offers_the_times_again(chat, session_factory):
    await offer_cleanings(chat)
    await chat.say("the first one")
    assert (await chat.say("no")).startswith("No problem. Here are the next free times")
    assert await booked_count(session_factory) == 0


async def test_small_talk_in_the_middle_of_a_booking_stays_in_the_booking(chat):
    await offer_cleanings(chat)
    await chat.say("the first one")
    reply = await chat.say("thanks!", model=PinnedIntent(intent="smalltalk"))
    assert reply.endswith("Shall I book it?"), "the question is still open, so it is asked again"


async def test_change_and_cancel_go_to_the_front_desk(chat):
    assert await chat.say("I need to cancel my appointment") == replies.BOOKING_CHANGES_NOT_YET
    assert await chat.say("Can I move my appointment?") == replies.BOOKING_CHANGES_NOT_YET


async def test_an_offer_left_too_long_is_replaced(chat):
    await offer_cleanings(chat)
    chat.now = FIXED_NOW + timedelta(minutes=20)
    reply = await chat.say("the first one")
    assert reply.startswith(replies.OFFERS_STALE), "not read back from a stale offer"
    assert await chat.kind() == "booking_offer"


# ---------------------------------------------------------------------------
# Someone else got there first
# ---------------------------------------------------------------------------
async def test_a_slot_held_by_another_conversation_is_not_read_back(
    chat, coordinator, redis_client
):
    await offer_cleanings(chat)
    first = (await chat.state())["booking"]["offered"][0]
    # Our hold lapses and another conversation is offered the same time.
    await redis_client.flushdb()
    other = SlotHolds(coordinator, ttl=timedelta(seconds=120), step=timedelta(minutes=30))
    await other.place(_slot(first), "another-thread")

    reply = await chat.say("the first one")
    assert reply.startswith(replies.SLOT_TAKEN)
    assert "Thu 01 Oct at 10:00" not in reply


async def test_a_conflict_at_the_write_offers_fresh_times(chat, calendar, clinic, session_factory):
    await offer_cleanings(chat)
    await chat.say("the second one")
    # Between the read-back and the yes, the front desk books that very slot.
    await calendar.book(
        BookingRequest(
            patient_id=clinic.patient_id,
            provider_id=clinic.hygienist_id,
            procedure_code="CLEANING",
            start_at=local(FRI, 9),
            end_at=local(FRI, 10),
            idempotency_key="front-desk-1",
            source="staff",
        )
    )
    reply = await chat.say("yes")
    assert reply.startswith(replies.SLOT_TAKEN)
    assert "Fri 02 Oct at 09:00" not in reply
    assert await booked_count(session_factory) == 1, "only the front desk's booking"
    assert await chat.kind() == "booking_offer"


# ---------------------------------------------------------------------------
# The calendar misbehaves
# ---------------------------------------------------------------------------
async def test_an_open_breaker_books_nothing_and_a_later_yes_retries(chat, session_factory):
    chat.breaker = build_breaker(chat.coordinator, "calendar", failure_threshold=1, cooldown_s=30)
    await offer_cleanings(chat)
    await chat.say("the first one")
    await chat.breaker.record(await chat.breaker.allow(), BreakerEvent.FAILURE)

    assert await chat.say("yes") == replies.CALENDAR_PAUSED
    assert await booked_count(session_factory) == 0
    assert (await chat.state())["booking"]["stage"] == "confirming"

    # The calendar recovers (a breaker that has closed again).
    chat.breaker = build_breaker(chat.coordinator, "fresh", failure_threshold=1, cooldown_s=30)
    assert (await chat.say("yes")).startswith("You're booked")
    assert await booked_count(session_factory) == 1


async def test_a_write_that_timed_out_after_committing_is_confirmed_from_the_row(
    session_factory, clinic, calendar, coordinator
):
    faulty = FaultyCalendar(
        calendar, FaultInjector.from_string("book_appointment:timeout_after_write@1")
    )
    chat = BookingChat(session_factory, faulty, coordinator)
    await offer_cleanings(chat)
    await chat.say("the first one")
    reply = await chat.say("yes")

    [row] = await appointments(session_factory)
    assert reply.startswith("You're booked") and row.external_ref in reply


async def test_a_write_that_timed_out_before_committing_can_be_retried(
    session_factory, clinic, calendar, coordinator
):
    faulty = FaultyCalendar(calendar, FaultInjector.from_string("book_appointment:timeout@1"))
    chat = BookingChat(session_factory, faulty, coordinator)
    await offer_cleanings(chat)
    await chat.say("the first one")

    assert await chat.say("yes") == replies.CALENDAR_NO_ANSWER
    assert await booked_count(session_factory) == 0
    assert (await chat.say("yes")).startswith("You're booked")
    assert await booked_count(session_factory) == 1


async def test_without_redis_it_still_books_and_says_what_it_did_without(
    session_factory, clinic, calendar, redis_client
):
    down = Coordinator(redis_client, FaultInjector.from_string("redis:unavailable"))
    chat = BookingChat(session_factory, calendar, down)
    offer = await offer_cleanings(chat)
    assert "Thu 01 Oct at 10:00" in offer
    assert set(chat.degraded) >= {"holds", "breaker"}
    await chat.say("the first one")
    assert (await chat.say("yes")).startswith("You're booked")
    assert "idempotency" in chat.degraded
    assert await booked_count(session_factory) == 1


# ---------------------------------------------------------------------------
# One request, one appointment
# ---------------------------------------------------------------------------
def test_the_idempotency_key_is_the_request_not_the_attempt():
    slot = _slot({"provider_id": 2, "start_at": local(FRI, 9), "end_at": local(FRI, 10)})
    later = _slot({"provider_id": 2, "start_at": local(FRI, 10), "end_at": local(FRI, 11)})
    assert idempotency_key("t", 1, slot) == idempotency_key("t", 1, slot)
    assert idempotency_key("t", 1, slot) != idempotency_key("t", 1, later)
    assert idempotency_key("t", 1, slot) != idempotency_key("u", 1, slot)


async def test_two_identical_writes_at_once_make_one_appointment(chat, clinic, session_factory):
    desk = chat.desk(DegradedModes())
    slot = _slot(
        {"provider_id": clinic.hygienist_id, "start_at": local(FRI, 9), "end_at": local(FRI, 10)}
    )
    key = idempotency_key(str(chat.thread_id), clinic.patient_id, slot)
    request = BookingRequest(
        patient_id=clinic.patient_id,
        provider_id=slot.provider_id,
        procedure_code=slot.procedure_code,
        start_at=slot.start_at,
        end_at=slot.end_at,
        idempotency_key=key,
    )
    first, second = await asyncio.gather(desk.book(request), desk.book(request))
    assert first.appointment_id == second.appointment_id
    assert await booked_count(session_factory) == 1


# ---------------------------------------------------------------------------
class ProposalModel(PinnedIntent):
    """Routes to booking, and reads every message as the same proposal -- a
    model that says something the state machine must not act on."""

    intent: str = "booking"
    proposal: dict = {}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if is_intent_call(kwargs):
            return super()._generate(messages, stop, run_manager, **kwargs)
        call = {"name": "BookingProposal", "args": self.proposal, "id": "call_booking"}
        return ChatResult(generations=[ChatGeneration(message=AIMessage("", tool_calls=[call]))])


def _slot(offered: dict) -> Slot:
    start = offered["start_at"]
    end = offered["end_at"]
    if isinstance(start, str):
        start, end = datetime.fromisoformat(start), datetime.fromisoformat(end)
    return Slot(offered["provider_id"], "CLEANING", start, end)


# ---------------------------------------------------------------------------
# Over HTTP, as `make dev` wires it
# ---------------------------------------------------------------------------
async def test_a_booking_over_http_with_the_app_wiring(test_settings, engine, clinic, redis_client):
    """The route builds the desk: calendar, breaker, holds, dedup, fault injector.

    The app's calendar runs on the real clock, so this books whatever is
    first free rather than a fixed time -- what it checks is the wiring.
    """
    settings = test_settings.model_copy(update={"llm_provider": "fake", "clinic_timezone": TZ.key})
    async with running_app(settings) as client:
        first = await say(client, "Please book me a cleaning")
        thread_id = first[0].data["thread_id"]
        assert first[1].data["intent"] == "booking"
        assert done_text(first) == replies.ASK_PHONE
        assert "1. " in done_text(await say(client, PAT_PHONE, thread_id))
        assert done_text(await say(client, "the first one", thread_id)).endswith("Shall I book it?")
        booked = await say(client, "yes", thread_id)

    assert done_text(booked).startswith("You're booked")
    assert booked[-1].data["degraded"] == []
    async with build_session_factory(engine)() as session:
        [row] = (await session.scalars(select(Appointment))).all()
    assert row.external_ref in done_text(booked)


# ---------------------------------------------------------------------------
# Regressions found in review
# ---------------------------------------------------------------------------
class DyingCalendar(CalendarProvider):
    """Delegates, except that book() can be made to crash the turn outright --
    a worker killed between the yes and the verification."""

    def __init__(self, inner: CalendarProvider) -> None:
        self.inner, self.armed = inner, False

    async def find_slots(self, *args, **kwargs):
        return await self.inner.find_slots(*args, **kwargs)

    async def get_booking(self, **kwargs):
        return await self.inner.get_booking(**kwargs)

    async def book(self, request):
        if self.armed:
            raise RuntimeError("worker died mid-write")
        return await self.inner.book(request)


async def test_a_turn_that_died_mid_write_does_not_book_on_the_next_message(
    session_factory, clinic, calendar, coordinator
):
    dying = DyingCalendar(calendar)
    chat = BookingChat(session_factory, dying, coordinator)
    await offer_cleanings(chat)
    await chat.say("the first one")
    dying.armed = True
    with pytest.raises(RuntimeError):
        await chat.say("yes")
    dying.armed = False

    reply = await chat.say("hello?", model=PinnedIntent(intent="smalltalk"))
    assert reply.endswith("Shall I book it?"), "only a yes given this turn reaches the write"
    assert await booked_count(session_factory) == 0
    assert (await chat.say("yes")).startswith("You're booked")


async def test_a_read_back_refused_by_the_breaker_keeps_the_same_key_retry(
    session_factory, clinic, calendar, coordinator
):
    """The timeout that is being verified opens the breaker, which then refuses
    the verification. Unknown is not lost: a yes retries under the same key."""
    faulty = FaultyCalendar(calendar, FaultInjector.from_string("book_appointment:timeout@1"))
    chat = BookingChat(
        session_factory,
        faulty,
        coordinator,
        breaker=build_breaker(coordinator, "calendar", failure_threshold=1, cooldown_s=30),
    )
    await offer_cleanings(chat)
    await chat.say("the first one")

    assert await chat.say("yes") == replies.BOOKING_UNVERIFIED
    assert (await chat.state())["booking"]["stage"] == "confirming"
    chat.breaker = build_breaker(coordinator, "recovered", failure_threshold=1, cooldown_s=30)
    assert (await chat.say("yes")).startswith("You're booked")
    assert await booked_count(session_factory) == 1


async def test_the_retry_asked_for_is_not_turned_into_a_stale_offer(chat, session_factory):
    chat.breaker = build_breaker(chat.coordinator, "calendar", failure_threshold=1, cooldown_s=30)
    await offer_cleanings(chat)
    chat.now = FIXED_NOW + timedelta(minutes=12)
    await chat.say("the first one")
    await chat.breaker.record(await chat.breaker.allow(), BreakerEvent.FAILURE)
    assert await chat.say("yes") == replies.CALENDAR_PAUSED

    chat.now = FIXED_NOW + timedelta(minutes=16)  # past OFFER_STALE_AFTER from the offer
    chat.breaker = build_breaker(chat.coordinator, "recovered", failure_threshold=1, cooldown_s=30)
    assert (await chat.say("yes")).startswith("You're booked")


async def test_an_unreadable_label_mid_booking_continues_the_booking(chat):
    await offer_cleanings(chat)
    reply = await chat.say("2", model=PinnedIntent(intent="not-an-intent"))
    assert reply.endswith("Shall I book it?")


async def test_a_natural_opening_is_not_mistaken_for_a_name(chat):
    assert await chat.say("Hi, I'm looking to book a cleaning") == replies.ASK_PHONE


async def test_a_time_of_day_is_not_a_choice(chat):
    await offer_cleanings(chat)
    assert (await chat.say("do you have anything at 3?")).startswith("Here are the next free")
