"""Tests for the architectural checker itself.

A guard nobody tests may have been letting everything through since the day it
was written. Each case is a small agent module written to a temp file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.factories import load_script

checker = load_script("check_invariants")
AGENTS_PACKAGE = "patient_ops.agents"


def scan(tmp_path: Path, source: str) -> list[str]:
    path = tmp_path / "agent_under_test.py"
    path.write_text(source)
    found = [*checker.scan_imports(path, AGENTS_PACKAGE), *checker.scan_file(path)]
    return [v.rule for v in found]


@pytest.mark.parametrize(
    "source",
    [
        "from patient_ops.db import repo",
        "from patient_ops import db",  # the imported *name* is the module
        "import patient_ops.adapters.calendar.fake",
        "from ..adapters.calendar import fake",  # relative, resolved from agents/
        "from .. import redis_layer",
        "import sqlalchemy.orm",
        "from redis.asyncio import Redis",
        "from patient_ops.rag.retrieve import search_knowledge_base",  # tools/ only
        "import patient_ops.rag.ingest",
        "def later():\n    from patient_ops.db.repo import find_patients\n",  # function-local
        # type-only imports count: agents should not even type against the ORM
        "if TYPE_CHECKING:\n    from patient_ops.db.models import Patient\n",
    ],
)
def test_write_capable_imports_are_caught(tmp_path: Path, source: str):
    rules = scan(tmp_path, source)
    assert rules and all(r.startswith("import of write-capable") for r in rules)


@pytest.mark.parametrize(
    "source",
    [
        "from patient_ops.tools import knowledge",  # the sanctioned way to read
        "from . import base",
        "from patient_ops.domain.availability import compute_slots",  # pure logic
        "import datetime\nfrom pydantic import BaseModel",
    ],
)
def test_read_only_imports_are_allowed(tmp_path: Path, source: str):
    assert scan(tmp_path, source) == []


def test_reaching_a_denied_module_without_importing_it_is_caught(tmp_path: Path):
    """`import patient_ops` is harmless on its own; walking to .db from it is not."""
    source = "import patient_ops\n\n\ndef f():\n    return patient_ops.db.repo.find_patients\n"
    assert scan(tmp_path, source) == ["reference to write-capable patient_ops.db"]


def test_a_local_name_that_shadows_a_library_is_not_flagged(tmp_path: Path):
    assert scan(tmp_path, "def f(redis):\n    return redis.get('x')\n") == []


def test_write_calls_are_still_caught(tmp_path: Path):
    assert scan(tmp_path, "session.commit()") == ["database transaction commit"]
    assert scan(tmp_path, 'sql = "INSERT INTO appointments VALUES (1)"') == ["raw SQL INSERT"]


def test_dynamic_imports_are_caught(tmp_path: Path):
    rules = scan(tmp_path, "import importlib\nm = importlib.import_module('patient_ops.db')")
    assert rules == ["dynamic import -- defeats the import rule"]


def test_prose_about_writes_is_not_flagged(tmp_path: Path):
    """Phase 0 behaviour: describing a write in a docstring or comment is fine."""
    source = '"""Agents never call book_appointment() or session.commit()."""\n# no .commit( here\n'
    assert scan(tmp_path, source) == []


def test_reviewed_exception_marker(tmp_path: Path):
    source = "from patient_ops.db.models import SPECIALTIES  # noqa: invariant -- constant only"
    assert scan(tmp_path, source) == []


def test_the_real_agents_directory_is_clean():
    files = sorted(checker.GUARDED_DIR.rglob("*.py"))
    assert files, "the guarded directory must exist and be scanned"
    for f in files:
        assert checker.scan_imports(f, checker.package_of(f)) == []
        assert checker.scan_file(f) == []
