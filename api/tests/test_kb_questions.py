"""The labelled question set, checked like code.

A calibration set nobody validates is a calibration set that quietly rots: a
renamed document turns an answerable question into an impossible one, and the
threshold moves to accommodate it.
"""

from __future__ import annotations

import pytest

from patient_ops.config import get_settings
from tests.factories import load_script

calibrate = load_script("calibrate_threshold")
QUESTIONS = calibrate.load_questions()
KB_FILES = {p.name for p in get_settings().kb_dir.glob("*.md")}


def of_kind(*kinds: str):
    return [q for q in QUESTIONS if q.kind in kinds]


def test_the_set_is_big_enough_to_say_anything():
    assert len(QUESTIONS) >= 40
    assert len({q.id for q in QUESTIONS}) == len(QUESTIONS)


def test_every_expected_document_exists():
    """A renamed file must break the question set loudly, not silently."""
    for question in QUESTIONS:
        missing = set(question.expected_sources) - KB_FILES
        assert not missing, f"{question.id} expects {missing}, which is not in the corpus"


def test_both_splits_are_populated_and_mixed():
    for split in ("calibration", "holdout"):
        group = [q for q in QUESTIONS if q.split == split]
        assert len(group) >= 12, f"{split} is too small to measure anything"
        assert any(q.answerable for q in group)
        assert any(not q.answerable for q in group)


def test_the_holdout_is_not_a_token_gesture():
    """It has to be large enough that its numbers mean something."""
    holdout = [q for q in QUESTIONS if q.split == "holdout"]
    assert 0.25 <= len(holdout) / len(QUESTIONS) <= 0.45


def test_every_kind_appears_in_both_splits():
    for kind in (*calibrate.ANSWERABLE_KINDS, *calibrate.UNANSWERABLE_KINDS):
        splits = {q.split for q in QUESTIONS if q.kind == kind}
        assert splits == {"calibration", "holdout"}, f"{kind} is missing from {splits}"


def test_the_hard_negatives_are_the_biggest_negative_class():
    """Near misses are the ones that matter.

    Refusing "who won the World Series" proves nothing. The question set must
    be weighted toward plausible questions the corpus does not answer, or the
    numbers it produces will be flattering and useless.
    """
    near_miss = of_kind("near_miss")
    assert len(near_miss) >= 10
    assert len(near_miss) > len(of_kind("off_topic"))


def test_paraphrases_outnumber_nothing_and_multi_hop_exists():
    assert len(of_kind("paraphrase")) >= 8, "patients do not use the document's wording"
    assert len(of_kind("multi_hop")) >= 3
    assert all(len(q.expected_sources) >= 2 for q in of_kind("multi_hop"))


# ---------------------------------------------------------------------------
# The schema itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ({"kind": "direct", "answerable": True, "expected_sources": []}, "must name the document"),
        (
            {"kind": "near_miss", "answerable": False, "expected_sources": ["aftercare.md"]},
            "cannot have an expected source",
        ),
        ({"kind": "near_miss", "answerable": True}, "not one of"),
        ({"kind": "direct", "answerable": False}, "not one of"),
        ({"kind": "direct", "answerable": True, "split": "training"}, "split"),
    ],
)
def test_inconsistent_labels_are_rejected(item: dict, expected: str):
    base = {
        "id": "x",
        "question": "q?",
        "split": "calibration",
        "expected_sources": ["aftercare.md"],
    }
    with pytest.raises(Exception, match=expected):
        calibrate.Question.model_validate({**base, **item})


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "dupes.yaml"
    path.write_text(
        "- {id: a, question: q, answerable: false, kind: off_topic, split: calibration}\n"
        "- {id: a, question: r, answerable: false, kind: off_topic, split: holdout}\n"
    )
    with pytest.raises(ValueError, match="duplicate question ids"):
        calibrate.load_questions(path)
