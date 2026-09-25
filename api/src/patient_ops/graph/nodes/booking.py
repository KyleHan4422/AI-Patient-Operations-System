"""The booking path: plan -> execute -> verify. Deterministic from end to end.

    booking_agent   the only model call: what did the patient just say?
    plan_booking    a state machine over the booking in progress -- who, what,
                    when, which time, "yes" -- deciding the next step
    execute_booking the write, and nothing else
    verify_booking  reads the appointment back and decides what is true

A booking spans several turns, so it lives in `state["booking"]` (a
BookingDraft) and each turn picks up where the last one stopped:

    collecting --(patient + treatment known)--> offered --(a number)--> confirming
         ^                                          ^                       |
         +---- a new treatment or date range -------+------- "no" ----------+
                                                                            | "yes"
                                                                        executing
                                                                            |
                                  booked (draft cleared) <-- verify <-- execute

Three rules the whole module is built around:

  Only offered times.  The reader picks an offered slot by its number. A time
                       it names itself is never booked: it was never held and
                       never checked.
  Only after a yes.    execute runs only from `confirming`, which is entered
                       only by reading the slot back to the patient.
  Only from the row.   "You're booked" is written from the appointment read
                       back after the write. A write that cannot be read back
                       is reported as uncertain, never as booked.

Redis is advice throughout (holds, dedup, the shared breaker); the no_overlap
EXCLUDE constraint and the UNIQUE idempotency key are the guarantees.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Literal

from langchain_core.messages import trim_messages
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from patient_ops.adapters.calendar.base import Booking, BookingRequest, Slot
from patient_ops.adapters.calendar.breaker import CIRCUIT_OPEN
from patient_ops.agents.booking import BookingProposal, read_booking_request
from patient_ops.errors import ErrorCode, ToolError
from patient_ops.graph import replies
from patient_ops.graph.context import GraphContext
from patient_ops.graph.state import AgentState
from patient_ops.obs.logging import get_logger
from patient_ops.redis_layer.holds import HoldResult, HoldStatus
from patient_ops.tools.booking import BookingDesk, idempotency_key

log = get_logger(__name__)

OFFER_COUNT = 3
DEFAULT_SEARCH_DAYS = 7  # when the patient names no dates
MAX_SEARCH_DAYS = 14  # a window wider than this is narrowed, not refused
# How many candidate slots are tried for a hold before giving up on filling
# all three offers. Bounds the Redis calls one offer can make.
MAX_HOLD_ATTEMPTS = 12
# Offers older than this are replaced before anything is booked from them: a
# time shown a while ago may since have moved inside the minimum lead time.
OFFER_STALE_AFTER = timedelta(minutes=15)
# A booking left unfinished this long is forgotten, and a later "yes" or
# "thanks" is no longer read as an answer to it.
DRAFT_ABANDONED_AFTER = timedelta(minutes=30)

Stage = Literal["collecting", "offered", "confirming", "executing"]
Outcome = Literal["written", "conflict", "invalid", "circuit_open", "no_answer", "unknown"]


class OfferedSlot(BaseModel):
    provider_id: int
    provider_name: str
    procedure_code: str
    start_at: datetime
    end_at: datetime

    def slot(self) -> Slot:
        return Slot(self.provider_id, self.procedure_code, self.start_at, self.end_at)


class BookingDraft(BaseModel):
    """The booking in progress, as the checkpoint stores it (a plain dict)."""

    stage: Stage = "collecting"
    updated_at: datetime
    # Who, while it is still being worked out. Dropped once a patient is found.
    phone: str | None = None
    full_name: str | None = None
    dob: date | None = None
    patient_id: int | None = None
    procedure_code: str | None = None
    procedure_name: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    offered: list[OfferedSlot] = Field(default_factory=list)
    offered_at: datetime | None = None
    chosen: int | None = None  # 1-based, into `offered`
    idempotency_key: str | None = None
    outcome: Outcome | None = None  # what execute saw; verify decides what it means

    @property
    def chosen_slot(self) -> OfferedSlot:
        assert self.chosen is not None
        return self.offered[self.chosen - 1]

    def forget_offers(self) -> None:
        self.offered, self.offered_at, self.chosen = [], None, None
        self.stage = "collecting"

    def dump(self) -> dict[str, Any]:
        # JSON types only: the checkpoint must be able to load it back after
        # this model gains a field.
        return self.model_dump(mode="json")


def active_draft(state: AgentState, now: datetime) -> BookingDraft | None:
    raw = state.get("booking")
    if not raw:
        return None
    draft = BookingDraft.model_validate(raw)
    return None if now - draft.updated_at > DRAFT_ABANDONED_AFTER else draft


def _say(
    draft: BookingDraft | None, text: str, kind: str, now: datetime, **extra: Any
) -> dict[str, Any]:
    if draft is not None:
        draft.updated_at = now
    return {
        "draft": text,
        "answer_kind": kind,
        "booking": draft.dump() if draft is not None else None,
        **extra,
    }


def _offered_lines(draft: BookingDraft, desk: BookingDesk) -> list[str]:
    """The offers as the reader sees them: numbered, with the date spelled out."""
    return [
        f"{o.start_at.astimezone(desk.tz):%A %Y-%m-%d %H:%M} with {o.provider_name}"
        for o in draft.offered
    ]


def _view(offered: OfferedSlot, desk: BookingDesk) -> replies.SlotView:
    return replies.SlotView(offered.provider_name, offered.start_at.astimezone(desk.tz))


# ---------------------------------------------------------------------------
# booking_agent -- the one model call
# ---------------------------------------------------------------------------
async def booking_agent(state: AgentState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    desk = ctx.booking
    if desk is None:  # no calendar here: plan_booking says so, and no model is asked
        return {"booking_proposal": None}

    draft = active_draft(state, desk.now())
    history = trim_messages(
        state["messages"],
        strategy="last",
        max_tokens=min(ctx.history_limit, 8),  # the question it answers is recent
        token_counter=len,
        start_on="human",
    )
    proposal = await read_booking_request(
        history,
        model=ctx.chat_model,
        today=desk.today(),
        catalogue=await desk.procedure_catalogue(),
        offered=_offered_lines(draft, desk) if draft and draft.offered else (),
    )
    log.info("booking_proposal", **proposal.model_dump(exclude_none=True, mode="json"))
    return {"booking_proposal": proposal.model_dump(mode="json")}


# ---------------------------------------------------------------------------
# plan_booking -- the state machine
# ---------------------------------------------------------------------------
async def plan_booking(state: AgentState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    desk = ctx.booking
    if desk is None:
        return {"draft": replies.BOOKING_NOT_YET, "answer_kind": "booking_deferred"}

    now = desk.now()
    owner = str(ctx.thread_id)
    raw = state.get("booking_proposal") or {"action": "unclear"}
    proposal = BookingProposal.model_validate(raw)
    draft = active_draft(state, now) or BookingDraft(updated_at=now)
    if draft.stage == "executing":
        # A turn that died between the yes and the verification (a
        # disconnect, a crash) left this behind. Only a yes given *this* turn
        # may reach the write, so step back to the read-back: if the write
        # did happen, the next yes replays it under the same key.
        log.warning("booking_resumed_after_interrupted_write", key=draft.idempotency_key)
        draft.stage, draft.outcome = "confirming", None

    if proposal.action in ("change", "cancel"):
        await desk.release_holds(owner)
        return _say(None, replies.BOOKING_CHANGES_NOT_YET, "booking_changes_not_yet", now)

    # -- what and when: kept whenever they are said, even before we know who
    # is asking -- "book me a cleaning next Tuesday" is not asked twice.
    # A new treatment or window withdraws the offers made for the old one.
    catalogue = dict(await desk.procedure_catalogue())
    code = (proposal.procedure_code or "").strip().upper()
    if code in catalogue and code != draft.procedure_code:
        draft.procedure_code, draft.procedure_name = code, catalogue[code]
        await _withdraw(desk, draft, owner)
    if proposal.date_from or proposal.date_to:
        window = _window(proposal, desk.today())
        if window != (draft.date_from, draft.date_to):
            draft.date_from, draft.date_to = window
            await _withdraw(desk, draft, owner)

    # -- who ----------------------------------------------------------------
    if draft.patient_id is None:
        stop = await _identify(desk, draft, proposal, now)
        if stop is not None:
            return stop
    if draft.procedure_code is None:
        return _say(draft, replies.ask_procedure(list(catalogue.values())), "booking_ask", now)

    if draft.offered_at is not None and now - draft.offered_at > OFFER_STALE_AFTER:
        await _withdraw(desk, draft, owner)
        return await _offer(desk, draft, owner, now, lead=replies.OFFERS_STALE)

    # -- which one, and yes ---------------------------------------------------
    choice = proposal.choice
    if draft.offered and choice is not None and 1 <= choice <= len(draft.offered):
        if choice != draft.chosen:
            return await _choose(desk, draft, owner, choice, now)
    if draft.stage == "confirming":
        if proposal.confirmed == "yes":
            return await _confirm(desk, draft, owner, now)
        if proposal.confirmed == "no":
            draft.chosen, draft.stage = None, "offered"
            return _relist(desk, draft, now, lead="No problem.")
        return _say(
            draft,
            replies.read_back_sentence(draft.procedure_name or "", _view(draft.chosen_slot, desk)),
            "booking_read_back",
            now,
        )
    if draft.offered:
        return _relist(desk, draft, now)
    return await _offer(desk, draft, owner, now)


async def _identify(
    desk: BookingDesk, draft: BookingDraft, proposal: BookingProposal, now: datetime
) -> dict[str, Any] | None:
    """Find exactly one patient, or say what is needed next. None means found."""
    draft.phone = proposal.phone or draft.phone
    draft.full_name = proposal.full_name or draft.full_name
    draft.dob = proposal.date_of_birth or draft.dob
    if not draft.phone and not draft.full_name:
        return _say(draft, replies.ASK_PHONE, "booking_ask", now)

    if draft.phone:
        try:
            matches = await desk.find_patients(phone=draft.phone)
        except ValueError:
            draft.phone = None
            return _say(draft, replies.ASK_PHONE_AGAIN, "booking_ask", now)
    else:
        matches = await desk.find_patients(name=draft.full_name, dob=draft.dob)
        if len(matches) > 1 and draft.dob is None:
            # Two patients share this name. A guess here books someone
            # else's appointment, so the only move is to ask.
            return _say(draft, replies.ASK_DOB, "booking_ask", now)

    if len(matches) != 1:
        # Nobody, or still more than one after the date of birth: the front
        # desk can sort out which record is theirs. No patient is created here.
        return _say(None, replies.PATIENT_NOT_FOUND, "booking_not_found", now)

    draft.patient_id = matches[0].patient_id
    draft.phone = draft.full_name = draft.dob = None  # found: stop carrying it around
    log.info("booking_patient_identified", patient_id=draft.patient_id)
    return None


def _window(proposal: BookingProposal, today: date) -> tuple[date, date]:
    start = max(proposal.date_from or proposal.date_to or today, today)
    end = proposal.date_to or start + timedelta(days=DEFAULT_SEARCH_DAYS - 1)
    end = min(max(end, start), start + timedelta(days=MAX_SEARCH_DAYS - 1))
    return start, end


async def _withdraw(desk: BookingDesk, draft: BookingDraft, owner: str) -> None:
    """Take back whatever was offered: the holds go back to everyone else."""
    if draft.offered:
        await desk.release_holds(owner)
    draft.forget_offers()


def _spread(slots: list[Slot], tz: Any) -> list[Slot]:
    """Earliest time on each day first, then the rest in order.

    Three offers at 09:00, 09:30 and 10:00 on one morning are one offer three
    times; a patient who cannot do Monday wants to see Tuesday too.
    """
    ordered = sorted(slots, key=lambda s: s.start_at)
    seen: set[date] = set()
    first, rest = [], []
    for slot in ordered:
        day = slot.start_at.astimezone(tz).date()
        (rest if day in seen else first).append(slot)
        seen.add(day)
    return first + rest


async def _offer(
    desk: BookingDesk, draft: BookingDraft, owner: str, now: datetime, *, lead: str = ""
) -> dict[str, Any]:
    assert draft.procedure_code is not None
    today = desk.today()
    date_from = draft.date_from or today
    date_to = draft.date_to or today + timedelta(days=DEFAULT_SEARCH_DAYS - 1)
    try:
        slots = await desk.find_slots(draft.procedure_code, date_from, date_to)
    except ToolError as exc:
        if exc.code is not ErrorCode.INVALID:
            return _say(draft, replies.CALENDAR_UNAVAILABLE, "booking_unavailable", now)
        slots = []  # e.g. a window beyond the booking horizon: nothing to offer

    # Earlier offers were already given back by whoever called this (_withdraw).
    picked: list[Slot] = []
    for slot in _spread(slots, desk.tz)[:MAX_HOLD_ATTEMPTS]:
        # TAKEN: offered to another conversation in the last couple of minutes.
        # DEGRADED: Redis is not there -- offer it anyway; EXCLUDE decides.
        if await desk.hold(slot, owner) is HoldResult.TAKEN:
            continue
        picked.append(slot)
        if len(picked) == OFFER_COUNT:
            break

    name = draft.procedure_name or draft.procedure_code
    if not picked:
        draft.forget_offers()
        text = replies.no_slots(name, date_from, date_to)
        return _say(draft, f"{lead} {text}".strip(), "booking_no_slots", now)

    names = await desk.provider_names({s.provider_id for s in picked})
    draft.offered = [
        OfferedSlot(
            provider_id=s.provider_id,
            provider_name=names.get(s.provider_id, "our team"),
            procedure_code=s.procedure_code,
            start_at=s.start_at,
            end_at=s.end_at,
        )
        for s in sorted(picked, key=lambda s: s.start_at)
    ]
    draft.offered_at, draft.chosen, draft.stage = now, None, "offered"
    views = [_view(o, desk) for o in draft.offered]
    return _say(draft, replies.offer_sentence(name, views, lead=lead), "booking_offer", now)


def _relist(desk: BookingDesk, draft: BookingDraft, now: datetime, *, lead: str = "") -> dict:
    views = [_view(o, desk) for o in draft.offered]
    name = draft.procedure_name or draft.procedure_code or ""
    return _say(draft, replies.offer_sentence(name, views, lead=lead), "booking_offer", now)


async def _still_ours(desk: BookingDesk, slot: Slot, owner: str) -> bool:
    """False if another conversation has been offered this slot since we held it.

    Our hold may simply have expired (GONE): then it is taken again now, so
    the slot is not offered to anyone else while the patient says yes.
    """
    status = await desk.hold_status(slot, owner)
    if status is HoldStatus.OTHERS:
        return False
    if status is HoldStatus.GONE:
        return await desk.hold(slot, owner) is not HoldResult.TAKEN
    return True  # MINE, or DEGRADED: the constraint will decide


async def _choose(
    desk: BookingDesk, draft: BookingDraft, owner: str, choice: int, now: datetime
) -> dict[str, Any]:
    offered = draft.offered[choice - 1]
    if not await _still_ours(desk, offered.slot(), owner):
        await _withdraw(desk, draft, owner)
        return await _offer(desk, draft, owner, now, lead=replies.SLOT_TAKEN)
    draft.chosen, draft.stage = choice, "confirming"
    return _say(
        draft,
        replies.read_back_sentence(draft.procedure_name or "", _view(offered, desk)),
        "booking_read_back",
        now,
    )


async def _confirm(
    desk: BookingDesk, draft: BookingDraft, owner: str, now: datetime
) -> dict[str, Any]:
    if not await _still_ours(desk, draft.chosen_slot.slot(), owner):
        await _withdraw(desk, draft, owner)
        return await _offer(desk, draft, owner, now, lead=replies.SLOT_TAKEN)
    draft.stage, draft.outcome, draft.updated_at = "executing", None, now
    # No reply yet: nothing has happened, and verify_booking will say what did.
    return {"booking": draft.dump()}


def route_after_plan(state: AgentState) -> str:
    booking = state.get("booking") or {}
    return "execute_booking" if booking.get("stage") == "executing" else "respond"


# ---------------------------------------------------------------------------
# execute_booking -- the write, and only the write
# ---------------------------------------------------------------------------
async def execute_booking(state: AgentState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    desk = ctx.booking
    assert desk is not None, "plan_booking only routes here with a booking desk"
    draft = BookingDraft.model_validate(state["booking"])
    if draft.stage != "executing" or draft.patient_id is None or draft.chosen is None:
        raise RuntimeError(f"execute_booking reached from stage {draft.stage!r}")

    slot = draft.chosen_slot.slot()
    key = idempotency_key(str(ctx.thread_id), draft.patient_id, slot)
    draft.idempotency_key = key
    request = BookingRequest(
        patient_id=draft.patient_id,
        provider_id=slot.provider_id,
        procedure_code=slot.procedure_code,
        start_at=slot.start_at,
        end_at=slot.end_at,
        idempotency_key=key,
        source=ctx.channel,
    )
    try:
        await desk.book(request)
        draft.outcome = "written"
    except ToolError as exc:
        draft.outcome = _outcome_of(exc)
        log.warning("booking_write_failed", code=exc.code, cause=exc.cause, outcome=draft.outcome)
    return {"booking": draft.dump()}


def _outcome_of(exc: ToolError) -> Outcome:
    if exc.code is ErrorCode.CONFLICT:
        return "conflict"
    if exc.code is ErrorCode.INVALID:
        return "invalid"
    if exc.code is ErrorCode.TRANSIENT and exc.cause == CIRCUIT_OPEN:
        return "circuit_open"  # refused before the call: certainly not written
    if exc.code is ErrorCode.TRANSIENT:
        return "no_answer"  # a timeout may or may not have written -- read it back
    return "unknown"


# ---------------------------------------------------------------------------
# verify_booking -- what is true, and what the patient is told
# ---------------------------------------------------------------------------
async def verify_booking(state: AgentState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    desk = ctx.booking
    assert desk is not None
    now = desk.now()
    owner = str(ctx.thread_id)
    draft = BookingDraft.model_validate(state["booking"])
    outcome, key = draft.outcome, draft.idempotency_key
    assert key is not None

    if outcome in ("conflict", "invalid"):
        # Not written. Someone else has the slot (the EXCLUDE constraint said
        # so), or the calendar refused it: offer fresh times.
        lead = replies.SLOT_TAKEN if outcome == "conflict" else replies.SLOT_UNAVAILABLE
        await _withdraw(desk, draft, owner)
        return await _offer(desk, draft, owner, now, lead=lead)
    if outcome == "circuit_open":
        return _await_retry(draft, replies.CALENDAR_PAUSED, now)

    # Written, timed out, or unclassified: the row is the only witness.
    try:
        row = await desk.read_back(key)
    except ToolError as exc:
        # Could not look -- often the breaker, opened by the very failure we
        # are verifying. Whether it was written is unknown, and that is safe
        # to leave open: a retry sends the same key, which replays a booking
        # that exists and writes one that does not. Never two.
        log.error("booking_unverified", key=key, code=exc.code, outcome=outcome)
        return _await_retry(draft, replies.BOOKING_UNVERIFIED, now)

    if row is None:
        if outcome == "no_answer":
            # The calendar did not answer and holds no row for this key.
            return _await_retry(draft, replies.CALENDAR_NO_ANSWER, now)
        log.error("booking_missing_after_write", key=key, outcome=outcome)
        await desk.release_holds(owner)
        return _say(None, replies.BOOKING_UNCERTAIN, "booking_uncertain", now)

    if not _is_what_was_asked(row, draft):
        log.error("booking_mismatch", key=key, appointment_id=row.appointment_id)
        await desk.release_holds(owner)
        return _say(None, replies.BOOKING_UNCERTAIN, "booking_uncertain", now)

    # Booked. The other offers go back to everyone; this slot's own hold is
    # harmless and expires by itself.
    await desk.release_holds(owner, keep=draft.chosen_slot.slot())
    procedures = dict(await desk.procedure_catalogue())
    providers = await desk.provider_names({row.provider_id})
    text = replies.booking_confirmation(
        procedure_name=procedures.get(row.procedure_code, row.procedure_code),
        provider_name=providers.get(row.provider_id, "our team"),
        start_at=row.start_at.astimezone(desk.tz),
        reference=row.external_ref,
    )
    log.info("booking_confirmed", appointment_id=row.appointment_id)
    return _say(None, text, "booking_confirmed", now, appointment_id=row.appointment_id)


def _await_retry(draft: BookingDraft, text: str, now: datetime) -> dict[str, Any]:
    """Nothing is known to be written; the chosen time stays, and a yes retries.

    The offer is treated as fresh again: the patient has just been asked to
    say yes, and that yes must not be turned into a stale re-offer.
    """
    draft.stage, draft.outcome, draft.offered_at = "confirming", None, now
    return _say(draft, text, "booking_unavailable", now)


def _is_what_was_asked(row: Booking, draft: BookingDraft) -> bool:
    chosen = draft.chosen_slot
    return (
        row.status == "booked"
        and row.patient_id == draft.patient_id
        and row.provider_id == chosen.provider_id
        and row.procedure_code == chosen.procedure_code
        and row.start_at == chosen.start_at
        and row.end_at == chosen.end_at
    )
