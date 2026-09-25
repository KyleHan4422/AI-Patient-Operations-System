"""The booking reader: what the patient just said, as a typed proposal.

The second model in the system that sees clinic context, and it has even less
reach than the first: no tools at all. It reads the conversation and fills in
a BookingProposal -- a phone number, a treatment, "the second one", "yes" --
and that is the whole of its job. Looking the patient up, finding free times,
holding them, writing the appointment and reading it back are deterministic
code in graph/nodes/booking.py, behind a boundary this module cannot import
across (scripts/check_invariants.py).

Two things it deliberately cannot say:

  a time    It picks an offered slot by its number, never by naming a time.
            A model that can write "Tuesday 10:30" can write one that was
            never offered, never held, and never checked.
  "booked"  It reports that the patient said yes. Whether that turns into a
            booking is decided by the state machine, which only accepts a yes
            to a read-back it actually made.

A proposal that will not parse is an empty proposal: the turn asks its
question again rather than guessing. A provider that is down still fails the
turn loudly, as in classify_intent.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from patient_ops.obs.logging import get_logger

log = get_logger(__name__)


class BookingProposal(BaseModel):
    """What the patient's latest message says about the booking. Leave out anything
    they did not say."""

    action: Literal["book", "change", "cancel", "unclear"] = Field(
        default="book",
        description=(
            "book: a new appointment, or answering a booking question. change: move an "
            "existing appointment. cancel: cancel one. unclear: none of these."
        ),
    )
    procedure_code: str | None = Field(
        default=None, description="A treatment code from the list on file, e.g. 'CLEANING'."
    )
    phone: str | None = Field(default=None, description="The phone number, exactly as written.")
    full_name: str | None = Field(default=None, description="The patient's full name.")
    date_of_birth: date | None = Field(default=None, description="As YYYY-MM-DD.")
    date_from: date | None = Field(
        default=None, description="First day they could come in, as YYYY-MM-DD."
    )
    date_to: date | None = Field(
        default=None, description="Last day they could come in, as YYYY-MM-DD."
    )
    choice: int | None = Field(
        default=None,
        description="Which of the offered times they picked, by its number (1, 2, 3).",
    )
    confirmed: Literal["yes", "no"] | None = Field(
        default=None,
        description=(
            "Their answer when they were asked whether to book (or try booking again) "
            "the time read back to them. Otherwise empty."
        ),
    )


SYSTEM_PROMPT = """\
You read a dental clinic's booking conversation and record what the patient's
LATEST message says. Earlier messages are context only: do not repeat details
from them. Leave every field empty that the latest message does not state.

- Dates as YYYY-MM-DD, worked out from today's date below. "Next week" is
  Monday to Friday of next week; "Tuesday" is the next Tuesday.
- procedure_code must be one of the treatment codes listed below.
- choice is the NUMBER of an offered time the patient picked. Never invent a
  time. If they describe one ("the 10 o'clock"), give its number from the list.
- confirmed is only for an answer to the assistant asking whether to book --
  or try booking again -- the one time it read back.
- action is change or cancel only for an appointment they already have."""


def system_prompt(
    *, today: date, catalogue: Sequence[tuple[str, str]], offered: Sequence[str]
) -> str:
    parts = [SYSTEM_PROMPT, f"Today is {today:%A} {today.isoformat()}."]
    if catalogue:
        parts.append("Treatments on file: " + ", ".join(f"{c} ({n})" for c, n in catalogue) + ".")
    if offered:
        parts.append("Offered times: " + "; ".join(f"{n}) {t}" for n, t in enumerate(offered, 1)))
    return "\n\n".join(parts)


async def read_booking_request(
    messages: Sequence[AnyMessage],
    *,
    model: BaseChatModel,
    today: date,
    catalogue: Sequence[tuple[str, str]],
    offered: Sequence[str] = (),
) -> BookingProposal:
    prompt = system_prompt(today=today, catalogue=catalogue, offered=offered)
    try:
        raw = await model.with_structured_output(BookingProposal).ainvoke(
            [SystemMessage(prompt), *messages]
        )
        return BookingProposal.model_validate(raw)
    except (OutputParserException, ValidationError, ValueError) as exc:
        log.warning("booking_proposal_unparseable", error=str(exc))
        return BookingProposal(action="unclear")
