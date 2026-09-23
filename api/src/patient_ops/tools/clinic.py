"""The exact lookups: insurance, prices, opening hours.

These read the seven domain tables Phase 1 built -- the system of record -- with
exact queries. Nothing here is a similarity search, and that split is the whole
design: which plans are accepted and what a crown costs are facts with one
correct answer, and an embedding model asked for one will happily return the
neighbouring plan instead.

Every function turns rows into the facts of tools/facts.py, so a caller cannot
see a NULL and decide for itself what it meant.
"""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from patient_ops.db import repo
from patient_ops.tools.facts import (
    ClosedDay,
    HoursFact,
    InsuranceFact,
    OpeningWindow,
    PriceFact,
)

# How far ahead a closure is worth mentioning. Long enough to catch the
# holiday a patient is asking around, short enough that the reply stays short.
CLOSURE_HORIZON_DAYS = 45


async def lookup_insurance_plan(session: AsyncSession, plan_name: str) -> InsuranceFact:
    """Whether the clinic takes a plan.

    The three answers a patient can get are "yes", "no" and "I have no record
    of that plan", and the third is the one that must survive all the way to
    the reply. Aetna is deliberately absent from the demo data for exactly this
    case: an assistant that turns a missing row into "no, we don't take Aetna"
    has told a patient something the clinic never said.
    """
    plan = await repo.find_insurance_plan(session, plan_name)
    if plan is None:
        return InsuranceFact(asked_for=plan_name, plan_name=None, status="unknown")
    if not plan.accepted:
        status = "not_accepted"
    else:
        status = "in_network" if plan.in_network else "out_of_network"
    return InsuranceFact(
        asked_for=plan_name, plan_name=plan.plan_name, status=status, notes=plan.notes
    )


async def lookup_price(session: AsyncSession, procedure: str) -> PriceFact:
    """What a treatment costs, when the clinic has published a price for it.

    A missing price is not a zero and not a guess: procedures.price_min is
    nullable precisely so the clinic can list a treatment it prices case by
    case (whitening, in the demo data). Both bounds have to be present -- half
    a range is not a range.
    """
    found = await repo.find_procedure(session, procedure)
    if found is None:
        return PriceFact(
            asked_for=procedure,
            code=None,
            name=None,
            price_min=None,
            price_max=None,
            status="unknown_procedure",
        )
    priced = found.price_min is not None and found.price_max is not None
    return PriceFact(
        asked_for=procedure,
        code=found.code,
        name=found.name,
        price_min=found.price_min,
        price_max=found.price_max,
        status="priced" if priced else "no_price_on_file",
    )


def _merge(windows: list[OpeningWindow]) -> list[OpeningWindow]:
    """Union overlapping or touching windows within each weekday.

    Two providers working 09-12 and 09-17 is one open day of 09-17, not two
    lines in the reply. Same rule as domain/availability._merge, on wall-clock
    times rather than instants -- opening hours are a weekly pattern, so they
    never cross a DST boundary the way a concrete appointment does.
    """
    merged: list[OpeningWindow] = []
    for window in sorted(windows):
        last = merged[-1] if merged else None
        if last and last.weekday == window.weekday and window.start <= last.end:
            if window.end > last.end:
                merged[-1] = OpeningWindow(last.weekday, last.start, window.end)
        else:
            merged.append(window)
    return merged


async def opening_hours(
    session: AsyncSession, *, today: date, horizon_days: int = CLOSURE_HORIZON_DAYS
) -> HoursFact:
    """When the clinic's doors are open, plus the days it is shut.

    The union of every provider's schedule, which is a different question from
    "when can I be seen for a crown" -- that one depends on the treatment and
    on what is already booked, and it belongs to the booking path (Phase 6).
    Answering it from these windows would be the classic confident wrong
    answer: open on Saturday, but only the hygienist works it.
    """
    schedules = await repo.all_schedules(session)
    closures = await repo.closures_with_reasons(
        session, today, today + timedelta(days=horizon_days)
    )
    windows = _merge([OpeningWindow(s.weekday, s.start, s.end) for s in schedules])
    return HoursFact(
        windows=tuple(windows),
        closures=tuple(ClosedDay(day=c.closed_on, reason=c.reason) for c in closures),
        horizon_days=horizon_days,
    )
