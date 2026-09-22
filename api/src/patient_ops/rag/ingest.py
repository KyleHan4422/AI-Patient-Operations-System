"""Reading the corpus into the database. The only module here that writes.

Same split as db/repo.py (reads) and db/transcript.py (writes): retrieval and
ingestion live in different files so that "what can change the knowledge base"
is a question about imports, not about method names.

One run has four phases, and their order is the design:

  1. parse    every file, in memory. A broken frontmatter fails here -- before
              a single embedding is paid for and before a row is touched.
  2. compare  against what is already stored, to decide what actually changed.
  3. embed    only the changed chunks. Network calls, deliberately outside any
              transaction: a transaction held open across a slow API call
              holds a connection and its locks with it.
  4. commit   everything in one transaction. Readers see the old corpus or the
              new one, never a half-replaced one.

A document is unchanged only when its file hash, the chunker version and the
embedding model all match what is stored. (The chunker version is read through
its module rather than imported by name, so the rule is one a test can change.)
The file hash alone is the trap: change the chunking rules or the model and
every file still hashes the same, so ingestion would skip them all and leave
stale vectors behind, silently.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from langchain_core.embeddings import Embeddings
from sqlalchemy import delete, func, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from patient_ops.adapters.llm.client import classify_llm_error
from patient_ops.db.models import EMBEDDING_DIM, KbChunk, KbDocument
from patient_ops.errors import ErrorCode, ToolError
from patient_ops.rag import chunk as chunk_rules
from patient_ops.rag.chunk import Chunk, chunk_document, parse_document


@dataclass(frozen=True)
class PreparedDocument:
    """One file, parsed and chunked, not yet embedded."""

    source_path: str  # relative to the corpus directory, so it is machine-independent
    title: str
    category: str
    effective_date: date
    content_hash: str
    chunks: list[Chunk]


@dataclass(frozen=True)
class IngestReport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks_embedded: int = 0
    chunks_total: int = 0
    model: str = ""

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)

    def summary(self) -> str:
        return (
            f"{self.added} added, {self.updated} updated, {self.unchanged} unchanged, "
            f"{self.removed} removed · {self.chunks_embedded} chunks embedded · "
            f"{self.chunks_total} chunks in the knowledge base · model {self.model}"
        )


def read_corpus(kb_dir: Path) -> list[PreparedDocument]:
    """Every document under `kb_dir`, parsed and chunked. Pure except for reading files.

    Files whose name starts with "_" are skipped, so notes for authors can sit
    beside the corpus without being retrievable.
    """
    if not kb_dir.is_dir():
        raise ValueError(f"knowledge base directory not found: {kb_dir}")
    paths = sorted(p for p in kb_dir.glob("*.md") if not p.name.startswith("_"))
    if not paths:
        # Otherwise a mistyped KB_DIR would look like "the author deleted
        # everything" and ingestion would faithfully empty the knowledge base.
        raise ValueError(f"no documents found in {kb_dir}")

    prepared: list[PreparedDocument] = []
    for path in paths:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        try:
            frontmatter = parse_document(text).frontmatter
            chunks = chunk_document(text)
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from None
        if not chunks:
            raise ValueError(f"{path.name}: no text to retrieve")
        prepared.append(
            PreparedDocument(
                source_path=path.name,
                title=frontmatter.title,
                category=frontmatter.category,
                effective_date=frontmatter.effective_date,
                # Hash the bytes, not the parsed text: whitespace-only edits
                # change the chunks, so they must count as a change.
                content_hash=hashlib.sha256(raw).hexdigest(),
                chunks=chunks,
            )
        )
    return prepared


async def _stored_state(session: AsyncSession) -> dict[str, tuple[str, int, str]]:
    rows = await session.execute(
        select(
            KbDocument.source_path,
            KbDocument.content_hash,
            KbDocument.chunker_version,
            KbDocument.embedding_model,
        )
    )
    return {path: (content_hash, version, model) for path, content_hash, version, model in rows}


async def _embed(embeddings: Embeddings, texts: Sequence[str]) -> list[list[float]]:
    """Vectors for `texts`, with provider failures mapped onto ErrorCode."""
    if not texts:
        return []
    try:
        vectors = await embeddings.aembed_documents(list(texts))
    except Exception as exc:  # noqa: BLE001 -- classified, then re-raised as ToolError
        code = classify_llm_error(exc) or ErrorCode.UNKNOWN
        raise ToolError(code, f"embedding failed: {exc}", cause=type(exc).__name__) from exc
    bad = next((v for v in vectors if len(v) != EMBEDDING_DIM), None)
    if bad is not None:
        # Caught here rather than by Postgres, one row into the transaction.
        raise ToolError(
            ErrorCode.PERMANENT,
            f"embedding model returned {len(bad)} dimensions, but kb_chunks.embedding "
            f"is vector({EMBEDDING_DIM})",
        )
    return vectors


async def ingest(
    session_factory: async_sessionmaker[AsyncSession],
    embeddings: Embeddings,
    *,
    kb_dir: Path,
    model_name: str,
) -> IngestReport:
    """Bring the stored knowledge base in line with the files. Safe to re-run."""
    prepared = read_corpus(kb_dir)

    async with session_factory() as session:
        stored = await _stored_state(session)

    fingerprint = (chunk_rules.CHUNKER_VERSION, model_name)
    work = [
        doc for doc in prepared if stored.get(doc.source_path) != (doc.content_hash, *fingerprint)
    ]
    on_disk = {doc.source_path for doc in prepared}
    removed = sorted(set(stored) - on_disk)
    added = sum(1 for doc in work if doc.source_path not in stored)

    vectors = await _embed(embeddings, [c.embedding_text for doc in work for c in doc.chunks])

    async with session_factory() as session, session.begin():
        offset = 0
        for doc in work:
            document_id = await session.scalar(
                pg_insert(KbDocument)
                .values(
                    source_path=doc.source_path,
                    title=doc.title,
                    category=doc.category,
                    effective_date=doc.effective_date,
                    content_hash=doc.content_hash,
                    chunker_version=chunk_rules.CHUNKER_VERSION,
                    embedding_model=model_name,
                    ingested_at=func.now(),
                )
                .on_conflict_do_update(
                    index_elements=[KbDocument.source_path],
                    set_={
                        "title": doc.title,
                        "category": doc.category,
                        "effective_date": doc.effective_date,
                        "content_hash": doc.content_hash,
                        "chunker_version": chunk_rules.CHUNKER_VERSION,
                        "embedding_model": model_name,
                        "ingested_at": func.now(),
                    },
                )
                .returning(KbDocument.id)
            )
            # Replace, never merge: the number of chunks changes with the text,
            # and stale passages are worse than missing ones.
            await session.execute(delete(KbChunk).where(KbChunk.document_id == document_id))
            await session.execute(
                insert(KbChunk),
                [
                    {
                        "document_id": document_id,
                        "ordinal": chunk.ordinal,
                        "heading_path": chunk.heading_path,
                        "content": chunk.content,
                        "approx_tokens": chunk.approx_tokens,
                        "embedding": vectors[offset + i],
                    }
                    for i, chunk in enumerate(doc.chunks)
                ],
            )
            offset += len(doc.chunks)

        if removed:
            # The FK is ON DELETE CASCADE, so the chunks go with them.
            await session.execute(delete(KbDocument).where(KbDocument.source_path.in_(removed)))

        chunks_total = await session.scalar(select(func.count()).select_from(KbChunk)) or 0

    return IngestReport(
        added=added,
        updated=len(work) - added,
        unchanged=len(prepared) - len(work),
        removed=len(removed),
        chunks_embedded=len(vectors),
        chunks_total=chunks_total,
        model=model_name,
    )
