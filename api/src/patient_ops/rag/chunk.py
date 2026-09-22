"""Markdown in, retrievable passages out. Pure functions, no IO.

Chunking decides more about retrieval quality than the choice of embedding
model does. Two rules:

  split on headings   The author already marked the semantic boundaries. It
                      also yields a `heading_path` -- "Aftercare > After an
                      Extraction > Dry Socket" -- which is the citation a
                      patient can actually check, and which Phase 4 shows
                      alongside the answer.
  overlap long ones   A section too long for one passage is cut into
                      overlapping windows, so an answer that straddles a cut
                      survives in at least one of them.

Sizes are counted in *approximate* tokens: words times a constant. A real
tokenizer (tiktoken) would download a vocabulary file on first use, which is a
hidden network dependency in CI and offline -- and the count is only used to
decide where to cut, never to bill anyone.

CHUNKER_VERSION is part of the key that decides whether a document has to be
re-ingested. Change the rules here and bump it, or `make ingest` will skip
documents whose files did not change while their stored chunks are stale.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date

import yaml
from pydantic import BaseModel, ValidationError, field_validator

from patient_ops.db.models import KB_CATEGORIES

CHUNKER_VERSION = 1

MAX_TOKENS = 400
OVERLAP_TOKENS = 60
# Words, not characters: "how many words is 400 tokens". English prose runs
# around 1.3 tokens per word; the constant only has to be in the right
# neighbourhood.
TOKENS_PER_WORD = 1.3
MAX_WORDS = int(MAX_TOKENS / TOKENS_PER_WORD)
OVERLAP_WORDS = int(OVERLAP_TOKENS / TOKENS_PER_WORD)

_FRONTMATTER = re.compile(r"\A---\r?\n(?P<yaml>.*?)\r?\n---[ \t]*\r?\n?(?P<body>.*)\Z", re.S)
_HEADING = re.compile(r"\A(?P<hashes>#{1,6})\s+(?P<text>.+?)\s*#*\s*\Z")
_WORD = re.compile(r"\S+")


class Frontmatter(BaseModel):
    """The block at the top of every knowledge base file.

    Required, and validated before anything is embedded or written: a typo in
    a category is a mistake to find at `make ingest`, not at retrieval time.
    """

    title: str
    category: str
    # Shown with every citation. A policy answer without a date is a rumour.
    effective_date: date
    authority: str = "official"

    @field_validator("title", "authority")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v.strip()

    @field_validator("category")
    @classmethod
    def _known_category(cls, v: str) -> str:
        # The same tuple the kb_documents CHECK constraint is built from, so a
        # category the database would reject cannot get as far as the database.
        if v not in KB_CATEGORIES:
            raise ValueError(f"unknown category {v!r}; expected one of {', '.join(KB_CATEGORIES)}")
        return v


@dataclass(frozen=True)
class ParsedDocument:
    frontmatter: Frontmatter
    body: str


@dataclass(frozen=True)
class Section:
    """One heading's own text -- not its subsections, which are sections too."""

    heading_path: str
    text: str


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    heading_path: str
    content: str
    approx_tokens: int

    @property
    def embedding_text(self) -> str:
        """What is actually embedded: the passage, under its headings.

        A section titled "Parking" whose text never repeats the word is
        findable only if the heading travels with it. The stored `content`
        stays clean, because that is what a patient is shown.
        """
        return f"{self.heading_path}\n\n{self.content}"


def approx_tokens(text: str) -> int:
    return math.ceil(len(_WORD.findall(text)) * TOKENS_PER_WORD)


def parse_document(text: str) -> ParsedDocument:
    """Split a file into its frontmatter and its body.

    Raises ValueError -- with a reason -- when the frontmatter is missing,
    unparseable or incomplete. The caller adds the file name.
    """
    match = _FRONTMATTER.match(text.lstrip("﻿"))
    if match is None:
        raise ValueError("missing YAML frontmatter: the file must start with a '---' line")
    try:
        loaded = yaml.safe_load(match.group("yaml"))
    except yaml.YAMLError as exc:
        raise ValueError(f"frontmatter is not valid YAML: {exc}") from None
    if not isinstance(loaded, dict):
        raise ValueError("frontmatter must be a mapping of keys to values")
    try:
        frontmatter = Frontmatter.model_validate(loaded)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'frontmatter'}: {e['msg']}"
            for e in exc.errors()
        )
        raise ValueError(f"invalid frontmatter -- {problems}") from None
    return ParsedDocument(frontmatter=frontmatter, body=match.group("body"))


def split_by_headings(body: str, title: str) -> list[Section]:
    """Sections in document order, each labelled with its full heading path.

    A heading with no text of its own (only subheadings under it) produces no
    section; it still appears in the path of its children. Text before the
    first heading belongs to the document itself.
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (heading level, heading text)
    buffer: list[str] = []
    fenced = False

    def flush() -> None:
        text = "\n".join(buffer).strip()
        buffer.clear()
        if text:
            sections.append(Section(" > ".join([title, *(t for _, t in stack)]), text))

    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced  # '#' inside a code fence is not a heading
        heading = None if fenced else _HEADING.match(line.strip())
        if heading is None:
            buffer.append(line)
            continue
        flush()
        level = len(heading.group("hashes"))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading.group("text")))
    flush()
    return sections


def _windows(text: str) -> list[str]:
    """`text` as one passage, or as overlapping windows if it is too long.

    Windows are cut on word boundaries and sliced out of the original string,
    so the passages keep the author's line breaks and list markers -- and so
    that dropping each window's overlap reassembles the text exactly.
    """
    words = list(_WORD.finditer(text))
    if len(words) <= MAX_WORDS:
        return [text.strip()]

    step = MAX_WORDS - OVERLAP_WORDS  # OVERLAP_WORDS < MAX_WORDS, asserted below
    out: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + MAX_WORDS, len(words))
        out.append(text[words[start].start() : words[end - 1].end()])
        if end == len(words):
            break
        start += step
    return out


def chunk_document(text: str, *, title: str | None = None) -> list[Chunk]:
    """A whole file's passages, numbered from zero in document order.

    `title` defaults to the one in the frontmatter; tests pass it to chunk a
    body without a header block.
    """
    parsed = parse_document(text)
    return [
        Chunk(
            ordinal=ordinal,
            heading_path=section.heading_path,
            content=content,
            approx_tokens=approx_tokens(content),
        )
        for ordinal, (section, content) in enumerate(
            (section, window)
            for section in split_by_headings(parsed.body, title or parsed.frontmatter.title)
            for window in _windows(section.text)
        )
    ]


assert 0 < OVERLAP_WORDS < MAX_WORDS, "an overlap at least as wide as a window never advances"
