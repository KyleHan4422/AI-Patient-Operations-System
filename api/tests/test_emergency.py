"""G0, the emergency filter, as a pure function -- no database, no model.

Two labelled sets. evals/emergency/cases.yaml is the development set: every
positive must be caught; false positives are printed, not gated, because
over-triage is the failure this filter chooses. evals/emergency/holdout.yaml
was never used to write a rule; its recall is the honest number, guarded
against regression by HOLDOUT_RECALL_FLOOR. The route-level guarantees --
that an emergency is answered with no model, no Redis and no rate-limit budget
-- are in test_emergency_route.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from patient_ops.config import get_settings
from patient_ops.guardrails import emergency
from patient_ops.guardrails.emergency import (
    EmergencyMatch,
    normalize,
    reply,
    screen,
    screen_with_context,
)

EVALS = Path(__file__).resolve().parents[2] / "evals" / "emergency"
CASES = yaml.safe_load((EVALS / "cases.yaml").read_text())
POSITIVES = CASES["positives"]
NEGATIVES = CASES["negatives"]
HOLDOUT = yaml.safe_load((EVALS / "holdout.yaml").read_text())

# A regression guard, not a target: below 1.0 so that an honest new holdout
# case the rules miss is added as a miss, not "fixed" by tuning to it.
HOLDOUT_RECALL_FLOOR = 0.9


def check(case: dict) -> EmergencyMatch | None:
    """A labelled case, with the earlier messages it says came before it."""
    return screen_with_context(case["text"], case.get("context") or [])


@pytest.mark.parametrize("case", POSITIVES, ids=lambda c: c["text"][:40])
def test_every_labelled_emergency_is_caught_as_its_category(case):
    match = check(case)
    assert match is not None, f"missed: {case['text']!r}"
    assert match.category == case["category"], (case["text"], match)


@pytest.mark.parametrize(
    "case",
    [c for c in NEGATIVES if not c.get("known_false_positive")],
    ids=lambda c: c["text"][:40],
)
def test_ordinary_questions_pass_through(case):
    assert check(case) is None, f"new false positive: {case['text']!r}"


def test_known_false_positives_are_still_known():
    """A known false positive that stops matching should lose its label."""
    for case in NEGATIVES:
        if case.get("known_false_positive"):
            assert check(case) is not None, case["text"]


def test_the_set_covers_every_rule():
    """A rule no positive exercises is a rule nobody has checked."""
    exercised = {(m.rule_id, m.lang) for c in POSITIVES if (m := check(c))}
    assert exercised == {(rule.id, rule.lang) for rule in emergency.RULES}


def test_false_positive_rate_is_reported(capsys):
    caught = sum(check(c) is not None for c in NEGATIVES)
    with capsys.disabled():
        print(
            f"\nG0: {len(POSITIVES)}/{len(POSITIVES)} positives caught, "
            f"{caught}/{len(NEGATIVES)} negatives flagged"
        )
    assert len(POSITIVES) >= 40 and len(NEGATIVES) >= 25
    assert len(HOLDOUT["positives"]) >= 30 and len(HOLDOUT["negatives"]) >= 15


def test_holdout_recall_is_reported_and_does_not_regress(capsys):
    """The number to quote. The holdout was never used to write a rule."""
    positives, negatives = HOLDOUT["positives"], HOLDOUT["negatives"]
    caught = [c for c in positives if (m := check(c)) and m.category == c["category"]]
    flagged = [c for c in negatives if check(c) is not None]
    recall = len(caught) / len(positives)
    with capsys.disabled():
        print(
            f"\nG0 holdout: recall {len(caught)}/{len(positives)}, "
            f"{len(flagged)}/{len(negatives)} negatives flagged"
        )
    missed = [c["text"] for c in positives if c not in caught]
    assert recall >= HOLDOUT_RECALL_FLOOR, missed


def test_the_holdout_does_not_overlap_the_development_set():
    """A case in both is a case the rules were written against."""
    dev = {c["text"].lower() for c in POSITIVES + NEGATIVES}
    held = {c["text"].lower() for c in HOLDOUT["positives"] + HOLDOUT["negatives"]}
    assert not dev & held


def test_normalize_folds_what_patients_type():
    assert normalize("I CAN’T  breathe!!") == "i can't breathe"
    assert normalize("ｃａｎｔ") == "cant"


def test_an_er_match_beats_an_urgent_dental_one():
    assert screen("knocked out a tooth and now I can't breathe") == EmergencyMatch(
        "er", "breathing"
    )


# ---------------------------------------------------------------------------
# The replies say what the document says
# ---------------------------------------------------------------------------
def _kb() -> str:
    return (get_settings().kb_dir / emergency.SOURCE_DOCUMENT).read_text()


def test_the_cited_document_still_has_those_headings_and_that_date():
    """An edited document must break the build, not leave the replies stale."""
    text = _kb()
    assert f"effective_date: {emergency.SOURCE_DATE}" in text
    assert f"title: {emergency.SOURCE_TITLE}" in text
    for heading in (emergency.ER_HEADING, emergency.TOOTH_HEADING):
        assert re.search(rf"^## {re.escape(heading)}$", text, re.M), heading


@pytest.mark.parametrize("lang", ["en", "es", "zh"])
@pytest.mark.parametrize("category", ["er", "crisis", "urgent_dental"])
def test_replies_carry_the_phone_number_and_a_source(category, lang):
    text = reply(EmergencyMatch(category, "x", lang), "(212) 555-0199")
    assert text.count("(212) 555-0199") == (1 if lang == "en" else 2), "once in each language"
    source = "Source: 988 " if category == "crisis" else "Source: Dental Emergencies > "
    assert text.splitlines()[-1].startswith(source)


@pytest.mark.parametrize("lang", ["es", "zh"])
def test_a_reply_is_in_the_patients_language_first_then_english(lang):
    text = reply(EmergencyMatch("er", "x", lang), "n/a")
    local, english = text.split("\n\n", 1)
    assert "911" in local and "This could be a medical emergency" not in local
    assert english.startswith("This could be a medical emergency")


def test_the_crisis_reply_says_988_and_911_and_leaves_room_for_pain():
    text = reply(EmergencyMatch("crisis", "x"), "(212) 555-0199")
    first = text.split("\n\n")[0]
    assert "988" in first and "911" in first
    assert "pain is unbearable" in text, "'I want to die' is also said about a toothache"


def test_a_spanish_or_mandarin_match_is_answered_in_it():
    assert screen("No puedo respirar").lang == "es"  # type: ignore[union-attr]
    assert screen("喘不上气").lang == "zh"  # type: ignore[union-attr]
    assert screen("I can't breathe").lang == "en"  # type: ignore[union-attr]


def test_accents_are_folded():
    assert normalize("¿Se cayó el diente?") == "se cayo el diente"


# ---------------------------------------------------------------------------
# An emergency told in pieces
# ---------------------------------------------------------------------------
def test_context_is_only_used_when_the_new_message_adds_something():
    before = ["my cheek is swollen"]
    assert screen_with_context("now it's spreading to my eye", before) == EmergencyMatch(
        "er", "spreading_swelling", "en", from_context=True
    )
    assert screen_with_context("ok thanks", ["I can't breathe"]) is None


def test_a_message_that_is_an_emergency_alone_is_not_marked_as_from_context():
    match = screen_with_context("I can't breathe", ["hello"])
    assert match is not None and not match.from_context


def test_the_er_reply_says_911_first():
    first_line = reply(EmergencyMatch("er", "x"), "n/a").splitlines()[0]
    assert "911" in first_line and "emergency room" in first_line
