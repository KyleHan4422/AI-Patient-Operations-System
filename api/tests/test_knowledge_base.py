"""The corpus itself, checked like code.

Two kinds of assertion live here. The first is hygiene: every file parses,
every category is one the database accepts, no section is empty.

The second is the rule that keeps this system honest:

    One fact, one home.

Exact facts -- which insurance plans are accepted, what a treatment costs,
when a provider works -- live in the tables Phase 1 built and are read by exact
queries. The knowledge base holds prose. A fact written in both places drifts,
and when it drifts nobody can say which copy is wrong. Worse, insurance plan
names are exactly what similarity search blurs together: an embedding model
puts "Delta Dental PPO" and "Cigna DPPO" close to each other, and one of them
is accepted while the other is not.

So the rule is enforced here rather than promised in a style guide.
"""

from __future__ import annotations

import re

import pytest

from patient_ops.config import get_settings
from patient_ops.rag.chunk import MAX_TOKENS, chunk_document, parse_document
from tests.factories import load_script

KB_DIR = get_settings().kb_dir
KB_FILES = sorted(p for p in KB_DIR.glob("*.md") if not p.name.startswith("_"))

seed_script = load_script("seed")
PLAN_NAMES = [name for name, *_ in seed_script.INSURANCE_PLANS]
# The brand is what identifies a plan; the rest of the name ("PPO", "Preferred")
# is shared vocabulary. So the brand alone is enough to flag -- except when the
# brand is also an ordinary English word. "Guardian DentalGuard Preferred" is a
# plan; a "parent or legal guardian" is not, and a check that cannot tell them
# apart is a check somebody deletes. For those, only the full plan name counts.
AMBIGUOUS_BRANDS = {"guardian"}
PLAN_BRANDS = sorted(
    {brand for name in PLAN_NAMES if (brand := name.split()[0]).lower() not in AMBIGUOUS_BRANDS}
)


def _mentions(text: str, phrase: str) -> bool:
    """Case- and spacing-insensitive whole-word search."""
    pattern = r"\b" + r"\s+".join(re.escape(word) for word in phrase.split()) + r"\b"
    return re.search(pattern, text, re.IGNORECASE) is not None


def test_the_corpus_exists():
    assert len(KB_FILES) >= 8, f"expected the clinic's documents in {KB_DIR}"


@pytest.fixture(params=KB_FILES, ids=lambda p: p.name)
def kb_file(request: pytest.FixtureRequest):
    return request.param


def test_frontmatter_is_valid(kb_file):
    parsed = parse_document(kb_file.read_text(encoding="utf-8"))
    assert parsed.frontmatter.title
    assert parsed.body.strip(), "a document with no body has nothing to retrieve"


def test_document_is_chunkable(kb_file):
    chunks = chunk_document(kb_file.read_text(encoding="utf-8"))
    title = parse_document(kb_file.read_text(encoding="utf-8")).frontmatter.title

    assert len(chunks) >= 2, "a document that is one passage is not worth splitting on headings"
    assert all(c.heading_path.startswith(title) for c in chunks)
    assert all(c.content.strip() for c in chunks)
    assert all(c.approx_tokens <= MAX_TOKENS for c in chunks)


def test_headings_are_used(kb_file):
    """Retrieval cites headings, so a document without them cites nothing useful."""
    chunks = chunk_document(kb_file.read_text(encoding="utf-8"))
    assert any(" > " in c.heading_path for c in chunks)


# ---------------------------------------------------------------------------
# One fact, one home
# ---------------------------------------------------------------------------
def test_no_insurance_plan_is_named(kb_file):
    """Which plans are accepted lives in `insurance_plans`, read by an exact query.

    A plan named here would be a second copy of that fact, reachable only by
    similarity search -- the one retrieval method that cannot tell two plan
    names apart.
    """
    text = kb_file.read_text(encoding="utf-8")
    named = [p for p in [*PLAN_BRANDS, *PLAN_NAMES] if _mentions(text, p)]
    assert not named, (
        f"{kb_file.name} names insurance plan(s) {named}. "
        "Plan-specific facts belong in the insurance_plans table; "
        "this page should say to ask instead."
    )


def test_no_prices(kb_file):
    """What treatment costs lives in `procedures`, where a price can be NULL.

    NULL is the answer that matters: WHITENING has no price on file, and the
    pricing tool must say so rather than guess. Prose cannot express that.
    """
    text = kb_file.read_text(encoding="utf-8")
    assert "$" not in text, (
        f"{kb_file.name} quotes a currency amount. Prices live in the procedures table."
    )
