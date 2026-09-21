#!/usr/bin/env python3
"""Seed the demo clinic. Safe to run any number of times.

Every row is upserted on its natural key (INSERT ... ON CONFLICT ... DO UPDATE):
re-running changes nothing, and editing a value here then re-running updates it.
It is additive -- a row deleted from this file stays in the database; use
`make db-reset` for a clean slate.

The data is test design, not filler. Each oddity below exists for a later
evaluation case:

  - Delta Dental PPO is accepted; Cigna DPPO, its near neighbour in embedding
    space, is not. (Phase 4: an exact lookup beats fuzzy retrieval.)
  - Aetna is deliberately absent. (Phase 4: "not on file" must come back as
    unknown, and the agent must abstain rather than guess.)
  - Two patients share a name. (Phase 6: patient search returns "multiple" and
    the agent must ask a follow-up question.)
  - WHITENING has no price on file. (Phase 4: the price tool must say so.)
  - Only Dr. Chen works Saturday mornings -- "Saturdays by appointment".

Phone numbers use 555-01xx, reserved for fictional use in North America.

Usage:  python scripts/seed.py
"""

from __future__ import annotations

import asyncio
from datetime import date, time
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from patient_ops.config import get_settings
from patient_ops.db.models import (
    Base,
    ClinicClosure,
    InsurancePlan,
    Patient,
    Procedure,
    Provider,
    ProviderSchedule,
)
from patient_ops.db.session import build_engine, build_session_factory
from patient_ops.domain.phone import normalize_phone

MON, TUE, WED, THU, FRI, SAT = range(6)
WEEKDAYS = (MON, TUE, WED, THU, FRI)

PROVIDERS = [
    ("Dr. Emily Chen", "general"),
    ("Dr. Marcus Reed", "general"),
    ("Sofia Alvarez, RDH", "hygienist"),
]

TUE_TO_FRI = (TUE, WED, THU, FRI)

# provider name -> [(weekdays, start, end)]; a lunch break is simply two windows.
SCHEDULES = {
    "Dr. Emily Chen": [
        (WEEKDAYS, time(9), time(12)),
        (WEEKDAYS, time(13), time(17)),
        ((SAT,), time(9), time(13)),
    ],
    "Dr. Marcus Reed": [(TUE_TO_FRI, time(10), time(13)), (TUE_TO_FRI, time(14), time(18))],
    "Sofia Alvarez, RDH": [(WEEKDAYS, time(8), time(12)), (WEEKDAYS, time(13), time(16))],
}

CLOSURES = [
    (date(2026, 11, 26), "Thanksgiving"),
    (date(2026, 12, 25), "Christmas Day"),
    (date(2027, 1, 1), "New Year's Day"),
]

# code, name, minutes, price range (None = not on file), specialty
PROCEDURES = [
    ("EXAM", "Comprehensive exam", 30, ("95", "150"), "general"),
    ("CLEANING", "Adult cleaning", 60, ("120", "200"), "hygienist"),
    ("FILLING", "Composite filling", 60, ("150", "300"), "general"),
    ("CROWN", "Porcelain crown", 90, ("1200", "1800"), "general"),
    ("ROOT_CANAL", "Root canal (molar)", 90, ("1000", "1500"), "general"),
    ("EXTRACTION", "Simple extraction", 45, ("150", "350"), "general"),
    ("WHITENING", "In-office whitening", 60, None, "general"),
]

# plan name, accepted, in network, notes
INSURANCE_PLANS = [
    ("Delta Dental PPO", True, True, "In network. Preventive care typically covered in full."),
    ("Cigna DPPO", False, False, "Not accepted. Patients may self-pay and file their own claim."),
    ("MetLife PDP Plus", True, False, "Accepted as out-of-network; patient pays the difference."),
    ("Guardian DentalGuard Preferred", True, True, "In network."),
]

PATIENTS = [
    ("Maria Garcia", "(212) 555-0101", date(1985, 4, 12)),
    ("Maria Garcia", "(212) 555-0102", date(1992, 9, 30)),  # same name, different person
    ("James Wilson", "(212) 555-0103", date(1978, 1, 5)),
    ("Wei Zhang", "(212) 555-0104", date(1990, 6, 18)),
    ("Aisha Khan", "(212) 555-0105", date(2001, 11, 2)),
]


def _upsert(model, rows: list[dict], conflict_on: list, update: list[str]):
    stmt = pg_insert(model).values(rows)
    return stmt.on_conflict_do_update(
        index_elements=conflict_on, set_={col: stmt.excluded[col] for col in update}
    )


async def seed(session_factory: async_sessionmaker[AsyncSession]) -> dict[str, int]:
    """Upsert the demo clinic in one transaction; return row counts per table."""
    async with session_factory() as session, session.begin():
        await session.execute(
            _upsert(
                Provider,
                [{"name": n, "specialty": s} for n, s in PROVIDERS],
                [Provider.name],
                ["specialty"],
            )
        )
        provider_ids = dict((await session.execute(select(Provider.name, Provider.id))).all())

        await session.execute(
            _upsert(
                ProviderSchedule,
                [
                    {
                        "provider_id": provider_ids[name],
                        "weekday": d,
                        "start_time": s,
                        "end_time": e,
                    }
                    for name, blocks in SCHEDULES.items()
                    for days, s, e in blocks
                    for d in days
                ],
                [
                    ProviderSchedule.provider_id,
                    ProviderSchedule.weekday,
                    ProviderSchedule.start_time,
                ],
                ["end_time"],
            )
        )
        await session.execute(
            _upsert(
                ClinicClosure,
                [{"closed_on": d, "reason": r} for d, r in CLOSURES],
                [ClinicClosure.closed_on],
                ["reason"],
            )
        )
        await session.execute(
            _upsert(
                Procedure,
                [
                    {
                        "code": code,
                        "name": name,
                        "duration_min": minutes,
                        "price_min": Decimal(price[0]) if price else None,
                        "price_max": Decimal(price[1]) if price else None,
                        "specialty": specialty,
                    }
                    for code, name, minutes, price, specialty in PROCEDURES
                ],
                [Procedure.code],
                ["name", "duration_min", "price_min", "price_max", "specialty"],
            )
        )
        await session.execute(
            _upsert(
                InsurancePlan,
                [
                    {"plan_name": n, "accepted": a, "in_network": i, "notes": notes}
                    for n, a, i, notes in INSURANCE_PLANS
                ],
                # Matches the expression index uq_insurance_plans_lower_plan_name.
                [func.lower(InsurancePlan.plan_name)],
                ["plan_name", "accepted", "in_network", "notes"],
            )
        )
        await session.execute(
            _upsert(
                Patient,
                [
                    {"full_name": n, "phone": normalize_phone(p), "dob": dob}
                    for n, p, dob in PATIENTS
                ],
                [Patient.phone],
                ["full_name", "dob"],
            )
        )

    async with session_factory() as session:
        return {
            table.name: await session.scalar(select(func.count()).select_from(table))
            for table in Base.metadata.sorted_tables
        }


async def main() -> None:
    engine = build_engine(get_settings())
    try:
        counts = await seed(build_session_factory(engine))
    finally:
        await engine.dispose()
    width = max(map(len, counts))
    print(f"seeded {engine.url.database}:")
    for table, n in counts.items():
        print(f"  {table:<{width}}  {n}")


if __name__ == "__main__":
    asyncio.run(main())
