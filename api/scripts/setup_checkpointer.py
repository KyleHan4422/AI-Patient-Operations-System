#!/usr/bin/env python3
"""Create or upgrade LangGraph's checkpoint tables. Safe to run any number of times.

`make migrate` runs this right after `alembic upgrade head`: Alembic owns our
tables, LangGraph owns its own, and both are brought up to date together.

Usage:  python scripts/setup_checkpointer.py
"""

from __future__ import annotations

from patient_ops.config import get_settings
from patient_ops.db.models import LANGGRAPH_TABLES
from patient_ops.graph.checkpointer import setup_checkpointer


def main() -> None:
    setup_checkpointer(get_settings().libpq_url)
    print(f"✓ checkpointer tables up to date ({', '.join(sorted(LANGGRAPH_TABLES))})")


if __name__ == "__main__":
    main()
