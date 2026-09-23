"""Deciding whether the agent's answer may be said out loud. No model involved.

Phase 3 measured what retrieval cannot do. Cosine similarity scores *topic*,
not answerhood: "how much is the missed appointment fee" retrieves the section
headed Missed Appointments at one of the highest scores in the whole question
set, and that section states no fee. Seventeen questions in the labelled set
are of that kind, and no threshold separates them -- raising one until it does
only refuses real questions. The calibration report hands that residual here.

So this module is the layer that decides. Four rules, all of them mechanical:

  1. a proposal that says the passages do not answer the question is an
     abstention, and the clinic gets a KB_GAP record naming the document it
     should write;
  2. every citation must be a passage that was actually retrieved this turn --
     a citation the agent invented is not evidence;
  3. an answer with no citation at all is not grounded, however true it sounds;
  4. every figure in the answer -- a price, a fee, a number of days, a
     percentage -- must appear in a passage that was cited. This is the rule
     that makes "never give an approximate figure" a property of the system
     instead of a line in a prompt.

Facts looked up in the clinic's records (insurance, prices, hours) are not
checked here: they are not written by the model at all. They are rendered from
the row by graph/replies.py, and where one of them is *not definite* -- a plan
that is not on file -- the model's prose is dropped even if it passed, because
that is the exact situation in which a model supplies the "no" the records
never contained.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal

from patient_ops.agents.knowledge import FinalAnswer
from patient_ops.graph.replies import ABSTENTION, Citation, facts_block, sources_block
from patient_ops.rag.retrieve import RetrievedChunk
from patient_ops.tools.facts import Fact

# Every value kb_gaps.reason may hold. The CHECK constraint in migration 0004
# is built from this tuple, so a new reason cannot be written without one.
GAP_REASONS = (
    "no_passage_above_threshold",  # nothing was close enough to the question
    "passages_did_not_answer",  # passages came back; they did not answer it
    "answer_without_citation",  # prose with nothing behind it
    "fabricated_citation",  # cited a passage that was never retrieved
    "unsupported_figure",  # a number that appears in no cited passage
    "no_answer",  # the agent never produced a usable verdict
)
GapReason = Literal[
    "no_passage_above_threshold",
    "passages_did_not_answer",
    "answer_without_citation",
    "fabricated_citation",
    "unsupported_figure",
    "no_answer",
]

AnswerKind = Literal["facts", "passages", "facts_and_passages"]

# The three reasons that mean the model wrote something it could not support,
# as opposed to the corpus having nothing to say. On a turn that abstained they
# are the kb_gaps row; on a turn that still had a record to answer from they
# would otherwise vanish, so Grounded carries them out (see `rejected`).
UNSUPPORTED: tuple[str, ...] = (
    "answer_without_citation",
    "fabricated_citation",
    "unsupported_figure",
)

# Not a sentinel the database ever sees: the agent saying "this does not answer
# the question" is refined into one of the two reasons above, depending on
# whether anything was retrieved at all.
_INSUFFICIENT = "insufficient"

# A run of digits, with thousands separators and decimals: "1,400", "24", "50%"
# -> 50, "$180.00" -> 180.00. Figures spelled as words ("twenty-four hours")
# are not checked; prices and fees are written as digits, which is where the
# damage is.
_FIGURE = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class Grounded:
    """An answer that may be said, and what it rests on."""

    text: str
    kind: AnswerKind
    citations: tuple[Citation, ...]
    # Set when prose was refused for being unsupported and the turn was still
    # answerable from the records. The patient is safe either way -- they get
    # the record, not the prose -- but "the model invented a citation" is the
    # signal this module exists to raise, and a turn that happened to also hit
    # an exact table must not swallow it.
    rejected: GapReason | None = None


@dataclass(frozen=True)
class Abstained:
    reason: GapReason
    text: str = ABSTENTION


Decision = Grounded | Abstained


def _figures(text: str) -> set[Decimal]:
    found: set[Decimal] = set()
    for raw in _FIGURE.findall(text):
        try:
            found.add(Decimal(raw.replace(",", "")).normalize())
        except InvalidOperation:  # pragma: no cover -- the pattern cannot produce one
            continue
    return found


def unsupported_figures(answer: str, cited: Sequence[RetrievedChunk]) -> set[Decimal]:
    """Figures in the answer that appear in none of the cited passages.

    The heading path and the effective date count as evidence: both were shown
    to the agent with the passage, so quoting "as of 2026-01-15" is not an
    invention. Everything else has to be in the text the clinic wrote.
    """
    evidence: set[Decimal] = set()
    for chunk in cited:
        evidence |= _figures(f"{chunk.content} {chunk.heading_path} {chunk.effective_date}")
    return _figures(answer) - evidence


def _spoken_facts(facts: Sequence[Fact]) -> list[Fact]:
    """The lookups worth telling the patient about, in the order they were made.

    Identical lookups collapse: an agent that asks the same question twice has
    not found two answers. A *failed* probe followed by a successful lookup of
    something else is still spoken -- "I have no treatment called CROWNS on our
    list" beside a crown's price reads clumsily, but both sentences are true,
    and any rule that dropped one would also drop the second half of "how much
    are a crown and whitening".
    """
    return list(dict.fromkeys(facts))


def _unresolved(facts: Sequence[Fact]) -> bool:
    """True when the last lookup of some kind still could not settle it.

    Per kind, not per fact. An agent that tries `lookup_price("CROWNS")`, gets
    nothing, and then asks for `CROWN` has resolved the price; treating the
    abandoned probe as an unsettled fact would throw away a fully grounded
    answer for no reason.
    """
    last_of_kind = {type(fact): fact for fact in facts}
    return any(not fact.definite for fact in last_of_kind.values())


def _check_prose(
    verdict: FinalAnswer | None, passages: dict[int, RetrievedChunk]
) -> tuple[str, tuple[Citation, ...], str | None]:
    """The four rules, in the order that gives the most useful reason."""
    if verdict is None:
        return "", (), "no_answer"
    if not verdict.sufficient:
        return "", (), _INSUFFICIENT
    answer = verdict.answer.strip()
    if not answer:
        # Expected when the answer came from a records lookup: the agent is
        # told to leave the prose empty there, and facts_block writes it.
        return "", (), "no_answer"
    if not verdict.citations:
        return "", (), "answer_without_citation"
    if any(chunk_id not in passages for chunk_id in verdict.citations):
        return "", (), "fabricated_citation"
    cited = [passages[chunk_id] for chunk_id in dict.fromkeys(verdict.citations)]
    if unsupported_figures(answer, cited):
        return "", (), "unsupported_figure"
    citations = tuple(
        Citation(
            chunk_id=c.chunk_id,
            heading_path=c.heading_path,
            source_path=c.source_path,
            effective_date=c.effective_date,
        )
        for c in cited
    )
    return answer, citations, None


def decide(
    verdict: FinalAnswer | None,
    *,
    facts: Sequence[Fact],
    passages: dict[int, RetrievedChunk],
    searched: bool,
) -> Decision:
    """What the patient is told, and -- when that is nothing -- why."""
    prose, citations, rejected = _check_prose(verdict, passages)
    # Carried onto a Grounded answer, so that a model inventing a citation is
    # still visible on a turn the records could answer anyway.
    unsupported = rejected if rejected in UNSUPPORTED else None

    # A fact the records could not settle is the one case where the model's
    # prose is dropped even though it passed every rule: "we have no plan by
    # that name" and a fluent paragraph about insurance, side by side, is how
    # a patient ends up hearing a no that the clinic never said.
    spoken = _spoken_facts(facts)
    facts_text = facts_block(spoken) if spoken else ""
    if _unresolved(spoken):
        prose, citations = "", ()

    if prose and facts_text:
        text = f"{facts_text}\n\n{prose}\n\n{sources_block(citations)}"
        return Grounded(text=text, kind="facts_and_passages", citations=citations)
    if facts_text:
        return Grounded(text=facts_text, kind="facts", citations=(), rejected=unsupported)
    if prose:
        return Grounded(
            text=f"{prose}\n\n{sources_block(citations)}", kind="passages", citations=citations
        )

    if rejected == _INSUFFICIENT or rejected is None:
        # rejected is None only if prose was dropped by the rule above, and
        # that path has already returned; it is here so this cannot fall
        # through to an assert with nothing to report.
        rejected = (
            "passages_did_not_answer"
            if passages
            else ("no_passage_above_threshold" if searched else "no_answer")
        )
    assert rejected in GAP_REASONS
    return Abstained(reason=rejected)  # type: ignore[arg-type]
