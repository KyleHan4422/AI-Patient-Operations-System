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
from patient_ops.guardrails.emergency import EmergencyMatch, normalize, reply, screen

EVALS = Path(__file__).resolve().parents[2] / "evals" / "emergency"
CASES = yaml.safe_load((EVALS / "cases.yaml").read_text())
POSITIVES = CASES["positives"]
NEGATIVES = CASES["negatives"]
HOLDOUT = yaml.safe_load((EVALS / "holdout.yaml").read_text())

# A regression guard, not a target: below 1.0 so that an honest new holdout
# case the rules miss is added as a miss, not "fixed" by tuning to it.
HOLDOUT_RECALL_FLOOR = 0.9


@pytest.mark.parametrize("case", POSITIVES, ids=lambda c: c["text"][:40])
def test_every_labelled_emergency_is_caught_as_its_category(case):
    match = screen(case["text"])
    assert match is not None, f"missed: {case['text']!r}"
    assert match.category == case["category"], (case["text"], match)


@pytest.mark.parametrize(
    "case",
    [c for c in NEGATIVES if not c.get("known_false_positive")],
    ids=lambda c: c["text"][:40],
)
def test_ordinary_questions_pass_through(case):
    assert screen(case["text"]) is None, f"new false positive: {case['text']!r}"


def test_known_false_positives_are_still_known():
    """A known false positive that stops matching should lose its label."""
    for case in NEGATIVES:
        if case.get("known_false_positive"):
            assert screen(case["text"]) is not None, case["text"]


def test_the_set_covers_every_rule():
    """A rule no positive exercises is a rule nobody has checked."""
    exercised = {screen(c["text"]).rule_id for c in POSITIVES}  # type: ignore[union-attr]
    assert exercised == {rule.id for rule in emergency.RULES}


def test_false_positive_rate_is_reported(capsys):
    caught = sum(screen(c["text"]) is not None for c in NEGATIVES)
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
    caught = [c for c in positives if (m := screen(c["text"])) and m.category == c["category"]]
    flagged = [c for c in negatives if screen(c["text"]) is not None]
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


@pytest.mark.parametrize("category", ["er", "urgent_dental"])
def test_replies_carry_the_phone_number_and_a_source(category):
    text = reply(EmergencyMatch(category, "x"), "(212) 555-0199")
    assert "(212) 555-0199" in text
    assert text.splitlines()[-1].startswith("Source: Dental Emergencies > ")


def test_the_er_reply_says_911_first():
    first_line = reply(EmergencyMatch("er", "x"), "n/a").splitlines()[0]
    assert "911" in first_line and "emergency room" in first_line
