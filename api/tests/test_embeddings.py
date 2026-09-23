"""The embedding door: which model, which threshold, and the offline stand-in.

No network anywhere in this file. Constructing the real embedder does not call
the API, and every similarity assertion is about the deterministic stand-in.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from patient_ops.adapters.llm.embeddings import (
    CALIBRATED_MIN_SCORE,
    FAKE_EMBEDDING_MODEL,
    HashingEmbeddings,
    build_embeddings,
    embedding_model_name,
    min_score_for,
)
from patient_ops.config import Settings
from patient_ops.db.models import EMBEDDING_DIM
from patient_ops.errors import ErrorCode, ToolError

embedder = HashingEmbeddings()


def similarity(a: str, b: str) -> float:
    """Cosine similarity -- a dot product, because the vectors are normalised."""
    return sum(x * y for x, y in zip(embedder.embed_query(a), embedder.embed_query(b), strict=True))


# ---------------------------------------------------------------------------
# Choosing a model
# ---------------------------------------------------------------------------
def test_fake_provider_gets_the_offline_embedder():
    settings = Settings(llm_provider="fake")
    assert isinstance(build_embeddings(settings), HashingEmbeddings)
    assert embedding_model_name(settings) == FAKE_EMBEDDING_MODEL


def test_openai_without_a_key_has_no_embedder():
    """Boot must succeed without a key; the failure belongs at the call site."""
    assert build_embeddings(Settings(llm_provider="openai", openai_api_key=SecretStr(""))) is None


def test_openai_with_a_key_is_configured_from_settings():
    settings = Settings(
        llm_provider="openai",
        openai_api_key=SecretStr("sk-test"),
        embedding_model="text-embedding-3-small",
    )
    built = build_embeddings(settings)
    assert isinstance(built, OpenAIEmbeddings)
    assert built.model == "text-embedding-3-small"
    # The column is vector(1536); asking for that width keeps a wider model usable.
    assert built.dimensions == EMBEDDING_DIM
    assert embedding_model_name(settings) == "text-embedding-3-small"


# ---------------------------------------------------------------------------
# The threshold belongs to the model
# ---------------------------------------------------------------------------
def test_an_uncalibrated_model_is_refused_not_defaulted():
    with pytest.raises(ToolError) as caught:
        min_score_for("some-new-model-nobody-measured")
    assert caught.value.code is ErrorCode.PERMANENT
    assert "make calibrate" in caught.value.detail


def test_an_explicit_override_wins():
    assert min_score_for("some-new-model-nobody-measured", 0.31) == 0.31


def test_calibrated_models_have_a_threshold():
    """Both models the project ships with must have been measured."""
    for model in (FAKE_EMBEDDING_MODEL, "text-embedding-3-small"):
        assert 0.0 < min_score_for(model) < 1.0
    assert set(CALIBRATED_MIN_SCORE) >= {FAKE_EMBEDDING_MODEL, "text-embedding-3-small"}


# ---------------------------------------------------------------------------
# The stand-in
# ---------------------------------------------------------------------------
def test_vectors_have_the_column_width_and_unit_length():
    vector = embedder.embed_query("bite gently on the gauze")
    assert len(vector) == EMBEDDING_DIM
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


def test_text_without_content_words_yields_a_zero_vector():
    """It scores 0 against everything, rather than matching everything."""
    assert not any(embedder.embed_query("the and of it"))


def test_embed_documents_matches_embed_query():
    assert embedder.embed_documents(["a tooth"]) == [embedder.embed_query("a tooth")]


def test_related_text_scores_above_unrelated_text():
    question = "how much notice do I need to cancel an appointment"
    related = "We ask for at least 24 hours notice to change or cancel an appointment."
    unrelated = "A root canal treats infection inside a tooth."
    assert similarity(question, related) > 0.2
    assert similarity(question, related) > 5 * similarity(question, unrelated)


def test_word_forms_are_brought_together():
    """ "Where do I park" has to find a section titled "Parking"."""
    assert similarity("where do I park my car", "Parking is available in the garage") > 0.3
    assert similarity("booking an appointment", "appointments are booked by phone") > 0.3


def test_the_same_text_embeds_identically_in_a_different_process():
    """Guards against Python's per-process hash salt.

    With the built-in hash(), `make ingest` and `make search` would write and
    read two different vector spaces, and every score would be near zero for a
    reason no log would explain.
    """
    code = (
        "import json;"
        "from patient_ops.adapters.llm.embeddings import HashingEmbeddings;"
        "print(json.dumps(HashingEmbeddings().embed_query('parking garage')[:16]))"
    )
    runs = [
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("0", "12345")
    ]
    assert json.loads(runs[0]) == json.loads(runs[1]) == embedder.embed_query("parking garage")[:16]
