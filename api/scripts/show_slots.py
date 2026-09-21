#!/usr/bin/env python3
"""Print bookable slots for a procedure -- Phase 1 has no HTTP API, so this is
how to look at availability by eye.

Goes through build_calendar, so it sees exactly what the app will: the clinic
timezone, the scheduling policy and, if FAULT_INJECT is set, the injected faults.

Usage:  python scripts/show_slots.py CLEANING [--days 7] [--from 2026-11-23]
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import date, datetime, timedelta

from sqlalchemy import select

from patient_ops.adapters.calendar.faults import build_calendar
from patient_ops.config import get_settings
from patient_ops.db.models import ClinicClosure, Provider
from patient_ops.db.session import build_engine, build_session_factory
from patient_ops.errors import ToolError


async def main(code: str, first: date, days: int) -> int:
    settings = get_settings()
    tz = settings.clinic_tz
    engine = build_engine(settings)
    sessions = build_session_factory(engine)
    last = first + timedelta(days=days - 1)
    try:
        async with sessions() as session:
            names = dict((await session.execute(select(Provider.id, Provider.name))).all())
            closed = dict(
                (
                    await session.execute(
                        select(ClinicClosure.closed_on, ClinicClosure.reason).where(
                            ClinicClosure.closed_on.between(first, last)
                        )
                    )
                ).all()
            )
        try:
            slots = await build_calendar(settings, sessions).find_slots(code, first, last)
        except ToolError as err:
            print(f"error: {err}")
            return 1
    finally:
        await engine.dispose()

    by_day: dict[date, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for slot in slots:
        local = slot.start_at.astimezone(tz)
        by_day[local.date()][slot.provider_id].append(local.strftime("%H:%M"))

    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M %Z")
    print(
        f"{code} -- {len(slots)} slots, {first} .. {last}  (now {now}, {settings.clinic_timezone})"
    )
    for offset in range(days):
        day = first + timedelta(days=offset)
        label = day.strftime("%a %Y-%m-%d")
        if day in closed:
            print(f"\n{label}  closed: {closed[day]}")
        elif not by_day[day]:
            print(f"\n{label}  no slots")
        else:
            print(f"\n{label}")
            for provider_id, starts in sorted(by_day[day].items()):
                print(f"  {names[provider_id]:<20} {' '.join(starts)}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("code", help="procedure code, e.g. CLEANING, EXAM, CROWN")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--from", dest="first", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    first = args.first or datetime.now(get_settings().clinic_tz).date()
    raise SystemExit(asyncio.run(main(args.code.upper(), first, args.days)))
