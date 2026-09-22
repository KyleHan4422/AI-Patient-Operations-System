#!/usr/bin/env python3
"""Load knowledge_base/*.md into Postgres. Safe to run any number of times.

Only documents whose file, chunking rules or embedding model changed are
re-embedded, so re-running costs nothing and changes nothing.

Usage:  python scripts/ingest.py
"""

from __future__ import annotations

import asyncio
import sys

from patient_ops.adapters.llm.embeddings import build_embeddings, embedding_model_name
from patient_ops.config import get_settings
from patient_ops.db.session import build_engine, build_session_factory
from patient_ops.errors import ToolError
from patient_ops.rag.ingest import ingest


async def main() -> int:
    settings = get_settings()
    embeddings = build_embeddings(settings)
    if embeddings is None:
        print(
            "✗ no embedding model configured: set OPENAI_API_KEY, "
            "or LLM_PROVIDER=fake to use the offline embedder",
            file=sys.stderr,
        )
        return 1

    engine = build_engine(settings)
    try:
        report = await ingest(
            build_session_factory(engine),
            embeddings,
            kb_dir=settings.kb_dir,
            model_name=embedding_model_name(settings),
        )
    except (ToolError, ValueError) as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1
    finally:
        await engine.dispose()

    mark = "✓" if report.changed else "·"
    print(f"{mark} {report.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
