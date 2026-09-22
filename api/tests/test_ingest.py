"""Ingestion: what it writes, what it refuses to write, and what it skips.

Most of these run against a corpus built in a temp directory, so a test can
edit a document and re-ingest it. The last one uses the real knowledge base,
because a corpus nobody ingests in CI is a corpus that breaks unnoticed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import func, select

from patient_ops.adapters.llm.embeddings import FAKE_EMBEDDING_MODEL, HashingEmbeddings
from patient_ops.config import get_settings
from patient_ops.db.models import KbChunk, KbDocument
from patient_ops.errors import ErrorCode, ToolError
from patient_ops.rag import chunk as chunk_module
from patient_ops.rag.ingest import ingest, read_corpus
from tests.fakes import CountingEmbeddings, FailingEmbeddings, WrongWidthEmbeddings

MODEL = FAKE_EMBEDDING_MODEL


def write_doc(kb_dir: Path, name: str, *, title: str, body: str, category: str = "policy") -> Path:
    path = kb_dir / name
    path.write_text(
        f"---\ntitle: {title}\ncategory: {category}\neffective_date: 2026-01-15\n"
        f"authority: official\n---\n\n{body}",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def kb_dir(tmp_path: Path) -> Path:
    corpus = tmp_path / "kb"
    corpus.mkdir()
    write_doc(
        corpus,
        "cancellations.md",
        title="Cancellations",
        body="## Notice\n\nWe ask for 24 hours notice.\n\n## Late\n\nCall if you are late.\n",
    )
    write_doc(
        corpus,
        "parking.md",
        title="Parking",
        body="## Garage\n\nThere is a garage one block north.\n",
        category="general",
    )
    return corpus


async def run(session_factory, kb_dir: Path, embeddings=None, model: str = MODEL):
    return await ingest(
        session_factory, embeddings or HashingEmbeddings(), kb_dir=kb_dir, model_name=model
    )


async def counts(session_factory) -> tuple[int, int]:
    async with session_factory() as session:
        documents = await session.scalar(select(func.count()).select_from(KbDocument))
        chunks = await session.scalar(select(func.count()).select_from(KbChunk))
    return documents, chunks


# ---------------------------------------------------------------------------
# A first run
# ---------------------------------------------------------------------------
async def test_first_run_stores_documents_and_chunks(session_factory, kb_dir: Path):
    report = await run(session_factory, kb_dir)

    assert (report.added, report.updated, report.unchanged, report.removed) == (2, 0, 0, 0)
    assert report.chunks_embedded == report.chunks_total == 3
    assert await counts(session_factory) == (2, 3)

    async with session_factory() as session:
        document = await session.scalar(
            select(KbDocument).where(KbDocument.source_path == "parking.md")
        )
        assert document.title == "Parking"
        assert document.category == "general"
        assert document.embedding_model == MODEL
        assert document.chunker_version == chunk_module.CHUNKER_VERSION
        chunk = await session.scalar(select(KbChunk).where(KbChunk.document_id == document.id))
        assert chunk.heading_path == "Parking > Garage"
        assert chunk.content == "There is a garage one block north."
        assert len(chunk.embedding) == 1536


# ---------------------------------------------------------------------------
# Re-running: the three things that make a document stale
# ---------------------------------------------------------------------------
async def test_re_running_embeds_nothing(session_factory, kb_dir: Path):
    """Idempotence is measured in API calls, because that is what it costs."""
    await run(session_factory, kb_dir)
    counting = CountingEmbeddings()

    report = await run(session_factory, kb_dir, counting)

    assert (report.added, report.updated, report.unchanged) == (0, 0, 2)
    assert counting.calls == 0
    assert await counts(session_factory) == (2, 3)


async def test_an_edited_document_replaces_its_own_chunks_only(session_factory, kb_dir: Path):
    await run(session_factory, kb_dir)
    async with session_factory() as session:
        untouched = (await session.scalars(select(KbChunk.id))).all()
    write_doc(
        kb_dir,
        "parking.md",
        title="Parking",
        body="## Garage\n\nThe garage is now two blocks south.\n\n## Meters\n\nMetered parking.\n",
        category="general",
    )
    counting = CountingEmbeddings()

    report = await run(session_factory, kb_dir, counting)

    assert (report.added, report.updated, report.unchanged) == (0, 1, 1)
    assert counting.calls == 2, "only the changed document is re-embedded"
    async with session_factory() as session:
        contents = set((await session.scalars(select(KbChunk.content))).all())
        surviving = set((await session.scalars(select(KbChunk.id))).all())
    assert "The garage is now two blocks south." in contents
    assert "There is a garage one block north." not in contents, "stale passages must not linger"
    assert len(surviving & set(untouched)) == 2, "the untouched document keeps its chunks"


async def test_new_chunking_rules_re_embed_everything(session_factory, kb_dir: Path, monkeypatch):
    """The trap the file hash alone cannot catch: same bytes, different chunks."""
    await run(session_factory, kb_dir)
    monkeypatch.setattr(chunk_module, "CHUNKER_VERSION", chunk_module.CHUNKER_VERSION + 1)
    counting = CountingEmbeddings()

    report = await run(session_factory, kb_dir, counting)

    assert (report.added, report.updated, report.unchanged) == (0, 2, 0)
    assert counting.calls == 3


async def test_a_new_embedding_model_re_embeds_everything(session_factory, kb_dir: Path):
    """Vectors from two models are not comparable, so the old ones are waste."""
    await run(session_factory, kb_dir)
    counting = CountingEmbeddings()

    report = await run(session_factory, kb_dir, counting, model="some-other-model")

    assert (report.added, report.updated, report.unchanged) == (0, 2, 0)
    assert counting.calls == 3
    async with session_factory() as session:
        models = set((await session.scalars(select(KbDocument.embedding_model))).all())
    assert models == {"some-other-model"}


async def test_a_deleted_file_removes_its_document_and_chunks(session_factory, kb_dir: Path):
    await run(session_factory, kb_dir)

    (kb_dir / "parking.md").unlink()
    report = await run(session_factory, kb_dir)

    assert (report.added, report.updated, report.unchanged, report.removed) == (0, 0, 1, 1)
    assert await counts(session_factory) == (1, 2), "the FK cascade takes the chunks"


# ---------------------------------------------------------------------------
# Failures leave nothing behind
# ---------------------------------------------------------------------------
async def test_a_broken_document_is_rejected_before_anything_is_written(
    session_factory, kb_dir: Path
):
    (kb_dir / "broken.md").write_text("---\ntitle: Broken\ncategory: nonsense\n---\nbody\n")
    counting = CountingEmbeddings()

    with pytest.raises(ValueError, match="broken.md: invalid frontmatter"):
        await run(session_factory, kb_dir, counting)

    assert counting.calls == 0, "parsing must fail before any embedding is paid for"
    assert await counts(session_factory) == (0, 0)


async def test_an_embedding_failure_leaves_the_corpus_as_it_was(session_factory, kb_dir: Path):
    await run(session_factory, kb_dir)
    before = await counts(session_factory)
    write_doc(kb_dir, "parking.md", title="Parking", body="## G\n\nNew text.\n", category="general")

    with pytest.raises(ToolError) as caught:
        await run(session_factory, kb_dir, FailingEmbeddings())

    assert caught.value.code is ErrorCode.TRANSIENT
    assert caught.value.cause == "APITimeoutError"
    assert await counts(session_factory) == before
    async with session_factory() as session:
        contents = set((await session.scalars(select(KbChunk.content))).all())
    assert "New text." not in contents


async def test_vectors_of_the_wrong_width_are_caught_before_the_transaction(
    session_factory, kb_dir: Path
):
    with pytest.raises(ToolError, match="768 dimensions") as caught:
        await run(session_factory, kb_dir, WrongWidthEmbeddings())

    assert caught.value.code is ErrorCode.PERMANENT
    assert await counts(session_factory) == (0, 0)


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        (lambda d: d / "missing", "directory not found"),
        (lambda d: (d / "empty").mkdir() or (d / "empty"), "no documents found"),
    ],
)
def test_a_corpus_that_is_not_there_is_an_error(tmp_path: Path, make, expected: str):
    """A mistyped path must not read as "the author deleted every document"."""
    with pytest.raises(ValueError, match=expected):
        read_corpus(make(tmp_path))


# ---------------------------------------------------------------------------
# The real corpus
# ---------------------------------------------------------------------------
async def test_the_clinic_corpus_ingests(session_factory):
    report = await run(session_factory, get_settings().kb_dir)

    assert report.added >= 8
    assert report.chunks_total >= 30, "suspiciously few passages for this corpus"
    async with session_factory() as session:
        paths = set((await session.scalars(select(KbDocument.source_path))).all())
    assert "aftercare.md" in paths
