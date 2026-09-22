"""Retrieval, against the real corpus.

The offline embedder is deterministic, so these can assert *which* passage
comes back -- an assertion no real model would let anyone write. That is the
whole reason it exists.
"""

from __future__ import annotations

import pytest

from patient_ops.adapters.llm.embeddings import FAKE_EMBEDDING_MODEL, HashingEmbeddings
from patient_ops.config import get_settings
from patient_ops.errors import ErrorCode, ToolError
from patient_ops.rag.ingest import ingest
from patient_ops.rag.retrieve import search_knowledge_base
from tests.fakes import FailingEmbeddings

MODEL = FAKE_EMBEDDING_MODEL
embeddings = HashingEmbeddings()


@pytest.fixture
async def corpus(session_factory):
    """The clinic's own documents, ingested with the offline embedder."""
    await ingest(session_factory, embeddings, kb_dir=get_settings().kb_dir, model_name=MODEL)
    return session_factory


async def search(session_factory, query: str, *, model: str = MODEL, k: int = 4, min_score=0.0):
    async with session_factory() as session:
        return await search_knowledge_base(
            session, embeddings, query, model=model, k=k, min_score=min_score
        )


# ---------------------------------------------------------------------------
# Finding the right passage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("how much notice do I need to cancel", "appointments-and-cancellations.md"),
        ("where do I park my car", "visiting-the-clinic.md"),
        ("can I eat right after a filling", "aftercare.md"),
        ("my tooth was knocked out playing football", "dental-emergencies.md"),
        ("what should I bring to my first appointment", "new-patients.md"),
        ("do you offer payment plans", "insurance-and-payment.md"),
    ],
)
async def test_a_question_retrieves_the_document_that_answers_it(corpus, query: str, expected: str):
    """hit@k: the answering document is among the passages the agent will see.

    Not "is ranked first", because the offline embedder ranks by word overlap
    alone: "what should I bring to my first appointment" puts any section about
    appointments near the top, whatever it says. A real embedding model knows
    the difference; the stand-in cannot, and a test that pretended otherwise
    would be tuned to the stand-in rather than to retrieval.
    """
    found = await search(corpus, query)
    hits = [c.source_path for c in found.chunks]
    assert expected in hits, f"top hits were {hits} (best {found.best_score:.3f})"


async def test_a_distinctive_question_ranks_its_own_document_first(corpus):
    found = await search(corpus, "where do I park my car")
    assert found.chunks[0].source_path == "visiting-the-clinic.md"
    assert found.chunks[0].heading_path.endswith("Parking")


async def test_results_are_ordered_and_bounded(corpus):
    found = await search(corpus, "what happens after an extraction", k=3)
    assert len(found.chunks) == 3
    assert [c.score for c in found.chunks] == sorted((c.score for c in found.chunks), reverse=True)
    assert found.best_score == found.chunks[0].score


async def test_every_hit_can_be_cited(corpus):
    """Phase 4 shows the heading and the date; a hit without them is unusable."""
    hit = (await search(corpus, "how do I get my records sent to another dentist")).chunks[0]
    assert hit.document_title and hit.heading_path.startswith(hit.document_title)
    assert hit.source_path.endswith(".md")
    assert hit.effective_date.year >= 2025
    assert hit.content.strip()
    assert 0.0 <= hit.score <= 1.0


# ---------------------------------------------------------------------------
# The threshold
# ---------------------------------------------------------------------------
async def test_weak_matches_are_dropped_but_the_best_score_is_still_reported(corpus):
    found = await search(corpus, "how much notice do I need to cancel", min_score=0.99)
    assert found.chunks == []
    assert found.best_score > 0.2, "the passage was found; it just did not qualify"
    assert found.threshold == 0.99
    assert not found.found_something


async def test_an_off_topic_question_scores_far_below_a_real_one(corpus):
    off_topic = await search(corpus, "who won the world series last night")
    on_topic = await search(corpus, "how much notice do I need to cancel")
    assert off_topic.best_score < on_topic.best_score / 2


# ---------------------------------------------------------------------------
# Empty is never ambiguous
# ---------------------------------------------------------------------------
async def test_an_uningested_knowledge_base_raises_rather_than_looking_empty(session_factory):
    """Forgetting `make ingest` must not be indistinguishable from abstaining."""
    with pytest.raises(ToolError) as caught:
        await search(session_factory, "how do I cancel")
    assert caught.value.code is ErrorCode.PERMANENT
    assert "make ingest" in caught.value.detail


async def test_another_models_vectors_are_not_searched(corpus):
    """Two models are two vector spaces; mixing them yields confident nonsense."""
    with pytest.raises(ToolError, match="no passages embedded"):
        await search(corpus, "how do I cancel", model="text-embedding-3-small")


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("query", "k"), [("", 4), ("   ", 4), ("valid", 0), ("valid", -1)])
async def test_nonsense_arguments_are_rejected(corpus, query: str, k: int):
    with pytest.raises(ValueError):
        await search(corpus, query, k=k)


async def test_an_embedding_timeout_is_classified_not_swallowed(corpus):
    async with corpus() as session:
        with pytest.raises(ToolError) as caught:
            await search_knowledge_base(
                session, FailingEmbeddings(), "how do I cancel", model=MODEL, min_score=0.0
            )
    assert caught.value.code is ErrorCode.TRANSIENT
    assert caught.value.cause == "APITimeoutError"
