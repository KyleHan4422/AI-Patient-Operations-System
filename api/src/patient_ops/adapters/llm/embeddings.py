"""The one door to an embedding model -- and to the threshold that goes with it.

Same rule as client.py: nothing else in the codebase names a provider. The
knowledge base is ingested and searched through whatever `build_embeddings`
returns, so a customer's Azure OpenAI account is a change to this file.

Two things travel together here, and that is the point:

  the model      which decides what a vector means; and
  its threshold  the score below which retrieval returns nothing and the
                 assistant abstains.

A threshold is a property of the model, not of the application. Vectors from
two models are not comparable, so their score distributions are not either --
0.42 on one scale can be "almost certainly relevant" and on another "no better
than random". Keeping the two in one table makes it impossible to swap the
model and keep the old number by accident: an uncalibrated model is a refusal,
not a default.

Both numbers below were measured by `make calibrate`, which fits the threshold
on one half of a labelled question set and reports it on the other half. What
it is fitted *against* is a finding of its own -- see the reports under
evals/calibration/.
"""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from patient_ops.config import Settings
from patient_ops.db.models import EMBEDDING_DIM
from patient_ops.errors import ErrorCode, ToolError

FAKE_EMBEDDING_MODEL = "fake-hashing-v1"

# model name -> abstention threshold, measured by `make calibrate`. Each has a
# report under evals/calibration/retrieval-calibration-<model>.md showing the
# score distributions it was fitted on and the holdout numbers it earned.
#
# The two are an order of magnitude apart, which is the whole argument for
# keeping this table: the same 0.345 that is a clean separator for the real
# model would refuse every question the offline stand-in has ever seen.
CALIBRATED_MIN_SCORE: dict[str, float] = {
    # Separable: off-topic questions top out at 0.257, real ones start at 0.444.
    # 100% answer rate, 100% off-topic rejection and hit@4 = 100% on the holdout.
    "text-embedding-3-small": 0.345,
    # A bag of words with no idea what words mean, so its whole score range is
    # compressed and it earns a much lower line. 100% answer rate and 100%
    # off-topic rejection on the holdout, hit@4 75%.
    FAKE_EMBEDDING_MODEL: 0.161,
}


def embedding_model_name(settings: Settings) -> str:
    """The name stored with every chunk, and the key into the threshold table."""
    if settings.llm_provider == "fake":
        return FAKE_EMBEDDING_MODEL
    return settings.embedding_model


def build_embeddings(settings: Settings) -> Embeddings | None:
    """The configured embedder, or None when no provider is configured.

    None rather than an exception, for the same reason as build_chat_model:
    the process must boot without an API key and say so when asked to work.
    """
    if settings.llm_provider == "fake":
        return HashingEmbeddings()
    if not settings.llm_configured:
        return None
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        # Asked for explicitly: kb_chunks.embedding is vector(1536), so a model
        # that would otherwise return a different width is truncated to fit
        # rather than failing one row at a time at INSERT.
        dimensions=EMBEDDING_DIM,
    )


def min_score_for(model: str, override: float | None = None) -> float:
    """The abstention threshold for `model`.

    Raises rather than falling back to a default. A borrowed threshold produces
    confident answers from irrelevant passages, which is the exact failure this
    whole layer exists to prevent.
    """
    if override is not None:
        return override
    try:
        return CALIBRATED_MIN_SCORE[model]
    except KeyError:
        raise ToolError(
            ErrorCode.PERMANENT,
            f"no calibrated abstention threshold for embedding model {model!r}: "
            "run `make calibrate` for it, or set RAG_MIN_SCORE to override",
        ) from None


# ---------------------------------------------------------------------------
# The offline stand-in
# ---------------------------------------------------------------------------
_WORD = re.compile(r"[a-z0-9]+")
# Stripped because they carry no topic: they would otherwise make every pair of
# English sentences look moderately similar.
_STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from had has have how i if in
    is it its may me my no not of on or our should so than that the their them then
    there these they this to up us was we were what when where which who why will
    with would you your""".split()
)
_NGRAM = 4
# Crude, deliberately: enough to put "parking" and "park", or "appointments"
# and "appointment", in the same bucket. A real embedding model handles
# morphology for free; the stand-in has to be told.
_SUFFIXES = ("ements", "ement", "ings", "ing", "ies", "ied", "es", "ed", "ly", "s")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if len(word) - len(suffix) >= 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


class HashingEmbeddings(Embeddings):
    """A deterministic, offline embedder: bag of words, hashed into 1536 slots.

    Not a language model -- it has no idea what words mean. What it does have
    is the one property the test suite needs: texts that share vocabulary come
    out close together, always, on every machine, for free. Retrieval tests can
    therefore assert *which* passage was found, which no real model allows.

    Words are crudely stemmed and their character n-grams hashed alongside, so
    that "where do I park" and "Parking" overlap. Without that the stand-in
    would be far more literal-minded than any real embedding model, and tests
    written against it would teach the wrong lesson about retrieval.

    Hashing is blake2b, never Python's built-in hash(): that one is salted per
    process, so `make ingest` and `make search` would write and read two
    unrelated vector spaces -- a bug whose only symptom is that every score is
    mysteriously low.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> int:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def _features(self, text: str) -> dict[str, float]:
        features: dict[str, float] = {}
        for word in _WORD.findall(text.lower()):
            if word in _STOPWORDS:
                continue
            stem = _stem(word)
            features[stem] = features.get(stem, 0.0) + 1.0
            for i in range(len(word) - _NGRAM + 1):
                gram = f"#{word[i : i + _NGRAM]}"
                features[gram] = features.get(gram, 0.0) + 0.5
        return features

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature, weight in self._features(text).items():
            vector[self._bucket(feature)] += weight
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:  # no content words at all: an all-zero vector scores 0
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]
