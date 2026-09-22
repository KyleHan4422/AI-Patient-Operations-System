"""The chunker: pure text in, passages out.

Worth testing this hard because every retrieval failure downstream looks the
same from the outside, and half of them start here.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from patient_ops.rag.chunk import (
    _WORD,
    MAX_WORDS,
    OVERLAP_WORDS,
    _windows,
    approx_tokens,
    chunk_document,
    parse_document,
    split_by_headings,
)

FRONTMATTER = """---
title: Aftercare
category: clinical
effective_date: 2026-03-01
authority: official
---
"""


def document(body: str) -> str:
    return FRONTMATTER + body


def words(text: str) -> list[str]:
    return _WORD.findall(text)


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------
def test_frontmatter_is_parsed():
    parsed = parse_document(document("\nBody text.\n"))
    assert parsed.frontmatter.title == "Aftercare"
    assert parsed.frontmatter.category == "clinical"
    assert str(parsed.frontmatter.effective_date) == "2026-03-01"
    assert parsed.body.strip() == "Body text."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("No frontmatter at all\n", "missing YAML frontmatter"),
        ("---\ntitle: [unclosed\n---\nbody\n", "not valid YAML"),
        ("---\njust a string\n---\nbody\n", "must be a mapping"),
        ("---\ncategory: policy\neffective_date: 2026-01-01\n---\nb\n", "title"),
        ("---\ntitle: T\ncategory: nonsense\neffective_date: 2026-01-01\n---\nb\n", "nonsense"),
        ("---\ntitle: T\ncategory: policy\neffective_date: last tuesday\n---\nb\n", "date"),
        ("---\ntitle: '  '\ncategory: policy\neffective_date: 2026-01-01\n---\nb\n", "blank"),
    ],
)
def test_bad_frontmatter_is_rejected_with_a_reason(text: str, expected: str):
    with pytest.raises(ValueError, match=expected):
        parse_document(text)


def test_category_must_be_one_the_database_would_accept():
    """The validator reads the same tuple the CHECK constraint is built from."""
    from patient_ops.db.models import KB_CATEGORIES

    for category in KB_CATEGORIES:
        body = f"---\ntitle: T\ncategory: {category}\neffective_date: 2026-01-01\n---\nb\n"
        assert parse_document(body).frontmatter.category == category


# ---------------------------------------------------------------------------
# Heading structure
# ---------------------------------------------------------------------------
BODY = """
Intro text before any heading.

## After an Extraction

Bite gently on the gauze.

### Dry Socket

A throbbing ache two days later.

## After a Crown

Avoid sticky sweets.
"""


def test_heading_path_carries_the_whole_ancestry():
    sections = split_by_headings(BODY, "Aftercare")
    assert [s.heading_path for s in sections] == [
        "Aftercare",  # the text before the first heading belongs to the document
        "Aftercare > After an Extraction",
        "Aftercare > After an Extraction > Dry Socket",
        "Aftercare > After a Crown",
    ]
    assert sections[2].text == "A throbbing ache two days later."


def test_a_heading_with_no_text_of_its_own_produces_no_section():
    body = "## Container\n\n### Real Section\n\nSome words.\n"
    sections = split_by_headings(body, "Doc")
    assert [s.heading_path for s in sections] == ["Doc > Container > Real Section"]


def test_a_deeper_heading_does_not_pop_its_parent():
    body = "## A\n\na\n\n### B\n\nb\n\n## C\n\nc\n"
    assert [s.heading_path for s in split_by_headings(body, "D")] == [
        "D > A",
        "D > A > B",
        "D > C",
    ]


def test_hashes_inside_a_code_fence_are_not_headings():
    body = "## Real\n\n```\n# not a heading\n```\n\nstill the same section\n"
    sections = split_by_headings(body, "D")
    assert len(sections) == 1
    assert "# not a heading" in sections[0].text


# ---------------------------------------------------------------------------
# Windows: what happens to a section too long for one passage
# ---------------------------------------------------------------------------
def long_text(word_count: int) -> str:
    return " ".join(f"w{i}" for i in range(word_count))


def test_a_short_section_is_one_passage():
    assert _windows("a b c") == ["a b c"]


def test_a_long_section_is_cut_into_overlapping_windows():
    produced = _windows(long_text(MAX_WORDS * 2))
    assert len(produced) > 1
    assert all(len(words(w)) <= MAX_WORDS for w in produced)
    for earlier, later in zip(produced, produced[1:], strict=False):
        assert words(earlier)[-OVERLAP_WORDS:] == words(later)[:OVERLAP_WORDS]


def test_every_window_is_a_verbatim_slice_of_the_original():
    """Passages keep the author's line breaks and list markers."""
    text = "\n".join(f"- item {i} with some words in it" for i in range(200))
    for window in _windows(text):
        assert window in text


@given(
    st.lists(
        st.text(alphabet=st.characters(whitelist_categories=("Ll", "Nd")), min_size=1, max_size=9),
        min_size=1,
        max_size=900,
    ),
    st.lists(st.sampled_from([" ", "\n", "\n\n", "  ", "\t"]), min_size=1, max_size=5),
)
def test_windows_lose_no_words_and_duplicate_none(word_list: list[str], gaps: list[str]):
    """Reassembling the windows, minus each one's overlap, yields the original.

    The property that matters: chunking must not silently drop a sentence at a
    cut, nor repeat it outside the overlap.
    """
    text = "".join(w + gaps[i % len(gaps)] for i, w in enumerate(word_list))
    produced = _windows(text)
    reassembled = words(produced[0])
    for window in produced[1:]:
        reassembled += words(window)[OVERLAP_WORDS:]
    assert reassembled == words(text)


# ---------------------------------------------------------------------------
# Whole documents
# ---------------------------------------------------------------------------
def test_chunks_are_numbered_in_document_order():
    chunks = chunk_document(document(BODY))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.content.strip() for c in chunks)
    assert all(c.approx_tokens > 0 for c in chunks)


def test_embedded_text_carries_the_headings_but_stored_content_does_not():
    chunk = chunk_document(document(BODY))[3]
    assert chunk.content == "Avoid sticky sweets."
    assert chunk.embedding_text.startswith("Aftercare > After a Crown")
    assert chunk.content in chunk.embedding_text


def test_approx_tokens_is_words_times_a_constant():
    assert approx_tokens("") == 0
    assert approx_tokens("one two three") == 4  # ceil(3 * 1.3)
