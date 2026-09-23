"""Similarity search over the knowledge base. Reads only.

The contract that matters is what an empty result means:

    no chunks == the knowledge base has nothing relevant to this question.

Exactly one meaning. Anything else that could produce no rows -- the corpus was
never ingested, or was ingested with a different embedding model -- raises
instead. If those returned an empty list too, forgetting `make ingest` would
disguise itself as an honest abstention: the assistant would say "I don't have
that on file" to every question, correctly formatted and completely wrong, with
nothing in any log to say why.

Scores are cosine similarity, 1.0 for identical direction. Passages below the
model's calibrated threshold are dropped here rather than passed upward, so a
caller cannot accidentally reason over evidence that did not qualify. The
highest raw score is still reported, because "how close did we get" is what
tells a KB_GAP report which document the clinic is missing -- reported
alongside the passage it belongs to, which is what makes it actionable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from langchain_core.embeddings import Embeddings
from sqlalchemy import Float, cast, select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.ext.asyncio import AsyncSession

from patient_ops.adapters.llm.client import classify_llm_error
from patient_ops.db.errors import classify_db_error
from patient_ops.db.models import KbChunk, KbDocument
from patient_ops.errors import ErrorCode, ToolError


@dataclass(frozen=True)
class RetrievedChunk:
    """One passage, with everything needed to cite it."""

    chunk_id: int
    document_title: str
    source_path: str
    heading_path: str
    content: str
    effective_date: date
    score: float


@dataclass(frozen=True)
class Retrieval:
    chunks: list[RetrievedChunk]
    best_score: float  # before the threshold was applied
    threshold: float
    model: str
    # The closest passage whether or not it qualified. An abstention's KB_GAP
    # record names it: "0.31 against Insurance & payment > What we accept" is
    # what tells the clinic which document is thin, and a bare score is not.
    nearest: RetrievedChunk | None = None

    @property
    def found_something(self) -> bool:
        return bool(self.chunks)


async def search_knowledge_base(
    session: AsyncSession,
    embeddings: Embeddings,
    query: str,
    *,
    model: str,
    k: int = 4,
    min_score: float,
) -> Retrieval:
    """The `k` passages closest to `query`, keeping only those at or above `min_score`."""
    if not query.strip():
        raise ValueError("query must not be blank")
    if k < 1:
        raise ValueError("k must be at least 1")

    try:
        vector = await embeddings.aembed_query(query)
    except Exception as exc:  # noqa: BLE001 -- classified, then re-raised as ToolError
        code = classify_llm_error(exc) or ErrorCode.UNKNOWN
        raise ToolError(
            code, f"embedding the query failed: {exc}", cause=type(exc).__name__
        ) from exc

    # Order by distance ascending, not by score descending: that is the form an
    # index could serve if this corpus ever grows enough to need one.
    distance = KbChunk.embedding.cosine_distance(vector)
    statement = (
        select(
            KbChunk.id,
            KbDocument.title,
            KbDocument.source_path,
            KbChunk.heading_path,
            KbChunk.content,
            KbDocument.effective_date,
            cast(1 - distance, Float).label("score"),
        )
        .join(KbDocument, KbDocument.id == KbChunk.document_id)
        # Vectors from another model are not comparable to this query's, so
        # they are not "worse matches" -- they are a different space entirely.
        .where(KbDocument.embedding_model == model)
        .order_by(distance)
        .limit(k)
    )
    try:
        rows = (await session.execute(statement)).all()
    except (DBAPIError, PoolTimeoutError) as exc:
        raise classify_db_error(exc) from exc

    if not rows:
        raise ToolError(
            ErrorCode.PERMANENT,
            f"the knowledge base holds no passages embedded with {model!r}: run `make ingest`",
        )

    found = [
        RetrievedChunk(
            chunk_id=chunk_id,
            document_title=title,
            source_path=source_path,
            heading_path=heading_path,
            content=content,
            effective_date=effective_date,
            score=float(score),
        )
        for chunk_id, title, source_path, heading_path, content, effective_date, score in rows
    ]
    return Retrieval(
        chunks=[c for c in found if c.score >= min_score],
        best_score=found[0].score,
        threshold=min_score,
        model=model,
        nearest=found[0],
    )
