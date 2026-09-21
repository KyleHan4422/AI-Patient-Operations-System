"""Small, explicit test data.

Tests build the few rows they need instead of relying on scripts/seed.py: demo
data changes for demo reasons, and a test must not break because of that.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from types import ModuleType
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from patient_ops.db.models import ClinicClosure, Patient, Procedure, Provider, ProviderSchedule

TZ = ZoneInfo("America/New_York")
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

OPEN_MONDAY = date(2026, 10, 5)  # ordinary working day
CLOSED_MONDAY = date(2026, 10, 12)  # in clinic_closures
FIXED_NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)  # "now" for every calendar test: a Thursday


def local(day: date, hour: int, minute: int = 0) -> datetime:
    """An aware datetime at clinic-local wall-clock time."""
    return datetime.combine(day, time(hour, minute), tzinfo=TZ)


async def booked_count(session_factory: async_sessionmaker[AsyncSession]) -> int:
    async with session_factory() as session:
        return await session.scalar(
            text("SELECT count(*) FROM appointments WHERE status = 'booked'")
        )


@dataclass(frozen=True)
class Clinic:
    dentist_id: int  # general; Mon-Fri 09-12 and 13-17 (lunch break)
    hygienist_id: int  # hygienist; Mon-Fri 09-17
    patient_id: int


async def build_minimal_clinic(session_factory: async_sessionmaker[AsyncSession]) -> Clinic:
    async with session_factory() as session, session.begin():
        dentist = Provider(name="Dr. Dentist", specialty="general")
        hygienist = Provider(name="Hy Gienist, RDH", specialty="hygienist")
        patient = Patient(full_name="Pat Test", phone="+12125550100")
        session.add_all(
            [
                dentist,
                hygienist,
                patient,
                Procedure(code="EXAM", name="Exam", duration_min=30, specialty="general"),
                Procedure(code="CROWN", name="Crown", duration_min=90, specialty="general"),
                Procedure(code="CLEANING", name="Cleaning", duration_min=60, specialty="hygienist"),
                ClinicClosure(closed_on=CLOSED_MONDAY, reason="Staff training"),
            ]
        )
        await session.flush()  # assigns the ids used below

        for weekday in range(5):  # Monday..Friday
            session.add_all(
                [
                    ProviderSchedule(
                        provider_id=dentist.id,
                        weekday=weekday,
                        start_time=time(9),
                        end_time=time(12),
                    ),
                    ProviderSchedule(
                        provider_id=dentist.id,
                        weekday=weekday,
                        start_time=time(13),
                        end_time=time(17),
                    ),
                    ProviderSchedule(
                        provider_id=hygienist.id,
                        weekday=weekday,
                        start_time=time(9),
                        end_time=time(17),
                    ),
                ]
            )
        return Clinic(dentist_id=dentist.id, hygienist_id=hygienist.id, patient_id=patient.id)


def load_script(name: str) -> ModuleType:
    """Import api/scripts/<name>.py -- scripts are not a package, but are tested like one."""
    spec = importlib.util.spec_from_file_location(f"scripts.{name}", SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while it executes
    spec.loader.exec_module(module)
    return module
