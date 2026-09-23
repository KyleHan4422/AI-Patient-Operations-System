"""The clinic's voice: every sentence that states a fact, written by code.

A language model is good at wording and bad at the difference between "we have
no record of that plan" and "we don't take that plan". So the exact facts -- an
insurance plan, a price, the opening hours -- are rendered here, from the row,
by a function with no model in it. What the patient hears about a record is a
property of this module, not of a prompt.

The prose answers are the other half: those are written by the model from
passages it cites, and checked against them by graph/grounding.py before they
ever reach here.

Everything in this file is a pure function of a fact. That makes the sentences
testable -- "an unknown plan never produces a sentence containing 'we don't
accept'" is an assertion, not a hope -- and it is the same shape Phase 6 needs
when a booking confirmation is filled from the verified appointment row.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, time

from patient_ops.tools.facts import (
    Fact,
    HoursFact,
    InsuranceFact,
    OpeningWindow,
    PriceFact,
    price_range,
)

SHORT_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# One wording for every abstention, whatever caused it. The patient does not
# benefit from knowing whether the corpus was thin or the answer failed its
# grounding check; the difference is recorded in kb_gaps, where it is
# actionable.
ABSTENTION = (
    "I don't have that in the clinic's records, and I'd rather not guess at it. "
    "The front desk will be able to tell you -- they have the details I don't."
)

BOOKING_NOT_YET = (
    "I can't book, change or cancel appointments yet, so please call the clinic for that. "
    "I can help with opening hours, prices, which insurance plans we take, "
    "and the clinic's policies."
)


@dataclass(frozen=True)
class Citation:
    """Where a sentence came from, in a form a patient can go and check."""

    chunk_id: int
    heading_path: str
    source_path: str
    effective_date: date


def _clock(value: time) -> str:
    return value.strftime("%H:%M")


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------
def insurance_sentence(fact: InsuranceFact) -> str:
    match fact.status:
        case "unknown":
            return (
                f"I don't have a plan called \"{fact.asked_for}\" on file, so I can't say "
                "either way -- that isn't a no. The front desk can check it against your card."
            )
        case "in_network":
            body = f"Yes -- we accept {fact.plan_name}, and we're in network with them."
        case "out_of_network":
            body = (
                f"We do accept {fact.plan_name}, but we're out of network with it, "
                "so your share of the cost may be higher."
            )
        case _:
            body = f"We don't accept {fact.plan_name}."
    return f"{body} {fact.notes}" if fact.notes else body


def price_sentence(fact: PriceFact) -> str:
    match fact.status:
        case "unknown_procedure":
            return (
                f'I don\'t have a treatment called "{fact.asked_for}" on our list, '
                "so I can't look a price up for it. The front desk can."
            )
        case "no_price_on_file":
            return (
                f"We don't have a price on file for {fact.name}, so I'd rather not guess at "
                "one. The front desk can quote it for you."
            )
        case _:
            return f"The price we have on file for {fact.name} is {price_range(fact)}."


def _grouped_days(windows: Sequence[OpeningWindow]) -> list[str]:
    """Opening hours as a clinic writes them on its door.

    Consecutive days with identical hours collapse into one line, so a full
    week reads "Mon-Fri 09:00-17:00; Sat 09:00-12:00".
    """
    by_day: dict[int, tuple[tuple[time, time], ...]] = {}
    for window in sorted(windows):
        by_day[window.weekday] = (*by_day.get(window.weekday, ()), (window.start, window.end))

    lines: list[str] = []
    run: list[int] = []

    def flush() -> None:
        if not run:
            return
        hours = ", ".join(f"{_clock(s)}-{_clock(e)}" for s, e in by_day[run[0]])
        days = (
            SHORT_DAYS[run[0]] if len(run) == 1 else f"{SHORT_DAYS[run[0]]}-{SHORT_DAYS[run[-1]]}"
        )
        lines.append(f"{days} {hours}")
        run.clear()

    for day in sorted(by_day):
        if run and day == run[-1] + 1 and by_day[day] == by_day[run[0]]:
            run.append(day)
        else:
            flush()
            run.append(day)
    flush()
    return lines


def hours_sentence(fact: HoursFact) -> str:
    if not fact.windows:
        return (
            "I don't have the opening hours on file, so I'd rather not guess. "
            "The front desk can tell you."
        )
    # Said plainly, because it is the confusion this answer invites: the doors
    # being open is not the same as a slot being free, and this assistant
    # cannot see slots at all until Phase 6.
    sentence = "We're open " + "; ".join(_grouped_days(fact.windows)) + "."
    if fact.closures:
        shut = ", ".join(f"{c.day:%a %d %b} ({c.reason})" for c in fact.closures)
        sentence += f" We're closed on {shut}."
    return sentence + " That's when the clinic is open -- I can't see free appointment times yet."


def fact_sentence(fact: Fact) -> str:
    match fact:
        case InsuranceFact():
            return insurance_sentence(fact)
        case PriceFact():
            return price_sentence(fact)
        case HoursFact():
            return hours_sentence(fact)


def facts_block(facts: Sequence[Fact]) -> str:
    """Every fact looked up this turn, one paragraph each, in the order asked."""
    return "\n\n".join(fact_sentence(f) for f in facts)


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------
def sources_block(citations: Sequence[Citation]) -> str:
    """Where the prose came from.

    Part of the reply text, not a separate field on the HTTP response: the
    transcript, the voice channel (Phase 13) and the /ops view all get the
    provenance for free, and a policy answer without a date is a rumour.
    """
    if not citations:
        return ""
    if len(citations) == 1:
        one = citations[0]
        return f"Source: {one.heading_path} (as of {one.effective_date})"
    listed = "\n".join(f"- {c.heading_path} (as of {c.effective_date})" for c in citations)
    return f"Sources:\n{listed}"
