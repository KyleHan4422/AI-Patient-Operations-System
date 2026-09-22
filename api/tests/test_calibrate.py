"""Choosing the threshold: the arithmetic, then the whole script end to end.

The arithmetic is pure, so it is tested with invented scores rather than with
the corpus -- including the degenerate case that produced the rule about a
minimum answer rate.
"""

from __future__ import annotations

import pytest

from patient_ops.adapters.llm.embeddings import FAKE_EMBEDDING_MODEL
from tests.factories import load_script

calibrate = load_script("calibrate_threshold")
choose_threshold = calibrate.choose_threshold


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------
def test_separable_scores_put_the_threshold_between_the_clouds():
    choice = choose_threshold([0.6, 0.7, 0.8], [0.1, 0.2, 0.3])
    assert 0.3 < choice.threshold < 0.6
    assert choice.separable
    assert (choice.false_answers, choice.false_abstentions) == (0, 0)


def test_overlapping_scores_are_reported_as_overlapping():
    choice = choose_threshold([0.4, 0.5, 0.6, 0.7], [0.1, 0.45, 0.65])
    assert not choice.separable
    assert choice.false_answers or choice.false_abstentions


@pytest.mark.parametrize("cost", [1.0, 3.0, 10.0, 50.0])
def test_a_higher_price_on_wrong_answers_never_lowers_the_threshold(cost: float):
    """The cost ratio is the knob that encodes 'do not guess', so it must bite."""
    answerable = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    unanswerable = [0.10, 0.20, 0.42, 0.52]
    cheap = choose_threshold(answerable, unanswerable, false_answer_cost=1.0, min_answer_rate=0.0)
    dear = choose_threshold(answerable, unanswerable, false_answer_cost=cost, min_answer_rate=0.0)
    assert dear.threshold >= cheap.threshold


def test_refusing_everything_is_never_the_answer():
    """The degenerate optimum, and the rule that exists because of it.

    With clouds this tangled and a high price on wrong answers, the cheapest
    threshold is one above every score: answer nothing, be wrong never. That is
    not a cautious assistant, it is a broken one, and the cost function has no
    way to know the difference.
    """
    answerable = [0.30, 0.35, 0.40, 0.45]
    unanswerable = [0.34, 0.38, 0.50, 0.60]

    unconstrained = choose_threshold(
        answerable, unanswerable, false_answer_cost=8.0, min_answer_rate=0.0
    )
    constrained = choose_threshold(
        answerable, unanswerable, false_answer_cost=8.0, min_answer_rate=0.6
    )

    assert unconstrained.threshold > max(answerable), "the degenerate optimum, as expected"
    answered = sum(1 for s in answerable if s >= constrained.threshold)
    assert answered / len(answerable) >= 0.6
    assert constrained.floor_binding and not unconstrained.floor_binding


def test_ties_go_to_the_more_cautious_threshold():
    choice = choose_threshold([0.5], [0.1], false_answer_cost=1.0)
    # Every threshold in (0.1, 0.5] costs nothing; the highest one is chosen.
    assert 0.1 < choice.threshold <= 0.5
    assert choice.threshold == pytest.approx(0.3, abs=0.01)


def test_identical_scores_everywhere_do_not_crash():
    choice = choose_threshold([0.5, 0.5], [0.5, 0.5], min_answer_rate=0.0)
    assert choice.threshold > 0
    assert not choice.separable


@pytest.mark.parametrize(("answerable", "unanswerable"), [([], [0.1]), ([0.5], []), ([], [])])
def test_one_sided_input_is_an_error(answerable, unanswerable):
    with pytest.raises(ValueError, match="both answerable and unanswerable"):
        choose_threshold(answerable, unanswerable)


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------
def outcome(kind: str, answerable: bool, score: float, hit: bool | None = None):
    question = calibrate.Question(
        id=f"{kind}-{score}",
        question="q?",
        answerable=answerable,
        kind=kind,
        split="calibration",
        expected_sources=["aftercare.md"] if answerable else [],
    )
    return calibrate.Outcome(question=question, best_score=score, hit=hit, retrieved=[])


def test_rates_separate_what_the_threshold_can_and_cannot_decide():
    outcomes = [
        outcome("direct", True, 0.8, hit=True),
        outcome("paraphrase", True, 0.2, hit=True),  # below the line: refused
        outcome("near_miss", False, 0.7),  # above the line: the residual
        outcome("structured", False, 0.1),
        outcome("off_topic", False, 0.05),
    ]
    measured = calibrate.rates(outcomes, threshold=0.5)

    assert measured["answer_rate"] == 0.5
    assert measured["off_topic_abstain_rate"] == 1.0
    assert measured["undecidable_abstain_rate"] == 0.5  # near_miss and structured
    assert measured["hit_at_k"] == 1.0


# ---------------------------------------------------------------------------
# The whole script
# ---------------------------------------------------------------------------
async def test_the_script_runs_and_writes_its_report(test_settings, tmp_path, monkeypatch):
    """End to end on the offline embedder: ingest, measure, fit, write."""
    monkeypatch.setattr(calibrate, "REPORT_DIR", tmp_path)
    settings = test_settings.model_copy(update={"llm_provider": "fake"})

    exit_code = await calibrate.run(settings, k=4, cost=3.0, min_answer_rate=0.6, plot=False)

    assert exit_code == 0
    report = (tmp_path / f"retrieval-calibration-{FAKE_EMBEDDING_MODEL}.md").read_text()
    assert "Abstention threshold" in report
    assert "holdout" in report
    data = (tmp_path / f"retrieval-calibration-{FAKE_EMBEDDING_MODEL}.json").read_text()
    assert '"threshold"' in data and '"holdout"' in data


def test_the_plot_is_written(tmp_path):
    outcomes = [
        outcome("direct", True, 0.8),
        outcome("near_miss", False, 0.7),
        outcome("off_topic", False, 0.05),
    ]
    choice = choose_threshold([0.8], [0.05])
    path = tmp_path / "plot.png"

    calibrate.plot_distribution(outcomes, choice, "test-model", path)

    assert path.stat().st_size > 5_000  # a real image, not an empty canvas
