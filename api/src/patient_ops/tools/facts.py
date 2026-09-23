"""What an exact lookup comes back with.

Three kinds of fact -- insurance, price, opening hours -- and one distinction
runs through all of them:

    definite   the clinic's records answer the question
    unknown    the records have nothing to say about it

"Not on file" is never "no". A plan the clinic has never heard of and a plan
the clinic refuses are different answers to a patient, and only one of them is
safe to state. Keeping that difference in the type, rather than in a nullable
field every caller has to remember to check, is why these exist at all.

Each fact is rendered twice, for two audiences that must not share wording:

    for_model()          a compact line the agent reads while it reasons
    graph/replies.py     the sentence the patient is told, written by code

The second one is the guarantee. The wording of an exact fact is never left to
a language model, so "we don't have that plan on file" cannot turn into "we
don't accept that plan" somewhere between the row and the patient.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from decimal import Decimal
from typing import Literal

# ---------------------------------------------------------------------------
# Insurance
# ---------------------------------------------------------------------------
# in_network and out_of_network are both "accepted"; the difference is what the
# patient pays, so it is not collapsed. unknown is the one that matters most.
InsuranceStatus = Literal["in_network", "out_of_network", "not_accepted", "unknown"]


@dataclass(frozen=True)
class InsuranceFact:
    asked_for: str  # what the patient called it, kept for the reply and the log
    plan_name: str | None  # the name as the clinic has it on file
    status: InsuranceStatus
    notes: str | None = None

    @property
    def definite(self) -> bool:
        return self.status != "unknown"


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
# A treatment with no price on file and a treatment the clinic does not list
# are both "we can't tell you", but they are not the same thing to the front
# desk, and the reply differs.
PriceStatus = Literal["priced", "no_price_on_file", "unknown_procedure"]


@dataclass(frozen=True)
class PriceFact:
    asked_for: str
    code: str | None
    name: str | None
    price_min: Decimal | None
    price_max: Decimal | None
    status: PriceStatus

    @property
    def definite(self) -> bool:
        return self.status == "priced"


# ---------------------------------------------------------------------------
# Opening hours
# ---------------------------------------------------------------------------
@dataclass(frozen=True, order=True)
class OpeningWindow:
    """A clinic-local wall-clock window on one weekday. Monday = 0."""

    weekday: int
    start: time
    end: time


@dataclass(frozen=True)
class ClosedDay:
    day: date
    reason: str


@dataclass(frozen=True)
class HoursFact:
    """When the clinic is open at all -- the union of every provider's schedule.

    Not "when can I be seen": that is availability, it depends on the treatment
    and on what is already booked, and it is the booking path's question
    (Phase 6). This answers the one a patient asks before they phone.
    """

    windows: tuple[OpeningWindow, ...]
    closures: tuple[ClosedDay, ...]
    horizon_days: int

    @property
    def definite(self) -> bool:
        return bool(self.windows)


Fact = InsuranceFact | PriceFact | HoursFact

WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _clock(value: time) -> str:
    return value.strftime("%H:%M")


def _money(amount: Decimal) -> str:
    return f"${amount:,.0f}" if amount == amount.to_integral_value() else f"${amount:,.2f}"


def for_model(fact: Fact) -> str:
    """The one-line form the agent reads.

    Deliberately terse and slightly clinical: it is evidence, not prose to be
    copied. The agent is told (in its prompt) that the patient's reply to a
    lookup is written from the record, so it has no reason to paraphrase this.
    """
    match fact:
        case InsuranceFact(status="unknown"):
            return (
                f'insurance plan "{fact.asked_for}": NOT ON FILE. '
                "This means the clinic has no record of that plan, not that it is refused."
            )
        case InsuranceFact():
            return f'insurance plan "{fact.plan_name}": {fact.status}' + (
                f" ({fact.notes})" if fact.notes else ""
            )
        case PriceFact(status="unknown_procedure"):
            return f'treatment "{fact.asked_for}": not on the clinic\'s list of treatments'
        case PriceFact(status="no_price_on_file"):
            return f"{fact.code} ({fact.name}): NO PRICE ON FILE -- do not estimate one"
        case PriceFact():
            return f"{fact.code} ({fact.name}): {price_range(fact)}"
        case HoursFact(windows=()):
            return "opening hours: no schedule on file"
        case HoursFact():
            days = "; ".join(
                f"{WEEKDAY_NAMES[w.weekday]} {_clock(w.start)}-{_clock(w.end)}"
                for w in fact.windows
            )
            closed = (
                "; closed " + ", ".join(f"{c.day} ({c.reason})" for c in fact.closures)
                if fact.closures
                else ""
            )
            return f"opening hours: {days}{closed}"


def price_range(fact: PriceFact) -> str:
    """A price as the patient sees it -- $180, or $900-$1,400 for a range."""
    assert fact.price_min is not None and fact.price_max is not None
    if fact.price_min == fact.price_max:
        return _money(fact.price_min)
    return f"{_money(fact.price_min)}-{_money(fact.price_max)}"
