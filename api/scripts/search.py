#!/usr/bin/env python3
"""Search the knowledge base from the terminal -- Phase 3 has no HTTP route yet.

Prints what the assistant would be given as evidence, and, when nothing
qualifies, the score it would have needed. Looking at these numbers by eye is
how a threshold stops being an abstraction.

Usage:  python scripts/search.py "how do I cancel" [--k 4] [--min-score 0.35]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import textwrap

from patient_ops.adapters.llm.embeddings import (
    build_embeddings,
    embedding_model_name,
    min_score_for,
)
from patient_ops.config import get_settings
from patient_ops.db.session import build_engine, build_session_factory
from patient_ops.errors import ToolError
from patient_ops.rag.retrieve import search_knowledge_base


async def main(query: str, k: int, min_score: float | None) -> int:
    settings = get_settings()
    embeddings = build_embeddings(settings)
    if embeddings is None:
        print(
            "✗ no embedding model configured (set OPENAI_API_KEY or LLM_PROVIDER=fake)",
            file=sys.stderr,
        )
        return 1

    model = embedding_model_name(settings)
    engine = build_engine(settings)
    try:
        threshold = min_score_for(
            model, min_score if min_score is not None else settings.rag_min_score
        )
        async with build_session_factory(engine)() as session:
            found = await search_knowledge_base(
                session, embeddings, query, model=model, k=k, min_score=threshold
            )
    except ToolError as exc:
        print(f"✗ [{exc.code}] {exc.detail}", file=sys.stderr)
        return 1
    finally:
        await engine.dispose()

    print(f"query: {query!r}   model: {found.model}   threshold: {found.threshold:.3f}\n")
    for chunk in found.chunks:
        print(f"  {chunk.score:.3f}  {chunk.heading_path}")
        print(f"         {chunk.source_path} · as of {chunk.effective_date}")
        print(textwrap.indent(textwrap.shorten(chunk.content, 220), "         "), "\n")

    if not found.chunks:
        print(
            f"  no passage reached the threshold (best {found.best_score:.3f} < "
            f"{found.threshold:.3f}) -- the assistant would abstain and offer a callback"
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=get_settings().rag_top_k)
    parser.add_argument(
        "--min-score", type=float, default=None, help="override the calibrated threshold"
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.query, args.k, args.min_score)))
