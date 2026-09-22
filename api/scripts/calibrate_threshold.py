#!/usr/bin/env python3
"""Measure the abstention threshold instead of guessing it.

Every RAG system needs a number below which it says "I don't have that on
file". Most projects copy one from a blog post. That number is a property of
the embedding model *and* the corpus, so a borrowed one is meaningless: it
either abstains on questions the documents answer, or answers questions they
do not.

This script:

  1. ingests the corpus with the chosen model, so the vectors are current;
  2. searches it with every labelled question in evals/retrieval/;
  3. fits the threshold on the `calibration` split only, using a cost ratio
     that says a wrong answer is worse than a needless abstention;
  4. reports it on the `holdout` split, which had no say in choosing it;
  5. writes a report and a plot of the score distributions.

The holdout numbers are the honest ones, and they are the ones the README
quotes. Fitting and reporting on the same questions measures nothing.

What the threshold is fitted against is itself a finding. The obvious labels
-- "the corpus answers this" against "it does not" -- do not separate, and
measuring showed why: cosine similarity scores *topic*, not answerhood. The
section headed "Missed Appointments" is the closest passage in the corpus to
"how much is the missed appointment fee", and it does not state a fee. No
threshold can tell those apart, because to the geometry they are the same
question.

So the threshold is fitted on what it can decide -- on-topic against
off-topic -- and the questions it cannot decide are reported separately as the
residual, which is work handed to the next layer: Phase 4's agent reads the
passage and sets `sufficient=False`, and the deterministic citation check
turns that into an abstention with a KB_GAP record. Splitting the job this way
is the point; a single number was never going to do both halves.

Usage:  python scripts/calibrate_threshold.py [--provider fake] [--cost 3.0]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError, model_validator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from patient_ops.adapters.llm.embeddings import build_embeddings, embedding_model_name
from patient_ops.config import REPO_ROOT, Settings, get_settings
from patient_ops.db.session import build_engine, build_session_factory
from patient_ops.rag.ingest import ingest
from patient_ops.rag.retrieve import search_knowledge_base

QUESTIONS_PATH = REPO_ROOT / "evals" / "retrieval" / "kb_questions.yaml"
# Committed, unlike evals/reports/, which holds one file per evaluation run and
# is gitignored. These are measurements the code depends on: the thresholds in
# adapters/llm/embeddings.py are only defensible because these reports exist.
REPORT_DIR = REPO_ROOT / "evals" / "calibration"

ANSWERABLE_KINDS = ("direct", "paraphrase", "multi_hop")
UNANSWERABLE_KINDS = ("near_miss", "structured", "off_topic")
# Off topic: the only negatives a similarity score can be expected to reject.
NEGATIVE_KINDS = ("off_topic",)
# On topic, no answer on file. Scores like a hit because the topic *is* in the
# corpus. Measured and reported, never fitted against -- see the module
# docstring.
UNDECIDABLE_KINDS = ("near_miss", "structured")

# How much worse a wrong answer is than a needless abstention. Not 1:1, and
# saying so out loud is the point: a patient who is told "let me have someone
# confirm that" loses a minute, while a patient who is told the wrong thing
# about a medicine or a fee acts on it. Same asymmetry as the emergency
# keyword filter in Phase 7, where over-triage is the correct failure.
FALSE_ANSWER_COST = 3.0

# The floor the cost function does not know about: an assistant that answers
# nothing is not a cautious assistant, it is a broken one. Without this, a
# corpus whose classes overlap badly has a degenerate optimum -- set the
# threshold above every score, refuse everything, pay no penalty for wrong
# answers. So the threshold must let through at least this share of the
# questions the documents demonstrably answer; below that, the problem is the
# corpus or the embedding model, and no threshold can fix it.
MIN_ANSWER_RATE = 0.6


class Question(BaseModel):
    id: str
    question: str
    answerable: bool
    kind: str
    split: Literal["calibration", "holdout"]
    expected_sources: list[str] = []

    @model_validator(mode="after")
    def _consistent(self) -> Question:
        kinds = ANSWERABLE_KINDS if self.answerable else UNANSWERABLE_KINDS
        if self.kind not in kinds:
            raise ValueError(f"kind {self.kind!r} is not one of {kinds} for this question")
        if self.answerable and not self.expected_sources:
            raise ValueError("an answerable question must name the document that answers it")
        if not self.answerable and self.expected_sources:
            raise ValueError("an unanswerable question cannot have an expected source")
        return self


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path.name}: expected a non-empty list of questions")
    try:
        questions = [Question.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise ValueError(f"{path.name}: {exc}") from None
    ids = [q.id for q in questions]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"{path.name}: duplicate question ids {sorted(duplicates)}")
    return questions


# ---------------------------------------------------------------------------
# Choosing the threshold
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    separable: bool  # no unanswerable question scored above an answerable one
    false_answers: int  # unanswerable questions this threshold would answer
    false_abstentions: int  # answerable questions it would refuse
    cost: float
    floor_binding: bool  # the cheapest threshold was rejected for answering too little


def choose_threshold(
    answerable: Sequence[float],
    unanswerable: Sequence[float],
    *,
    false_answer_cost: float = FALSE_ANSWER_COST,
    min_answer_rate: float = MIN_ANSWER_RATE,
) -> ThresholdChoice:
    """The score at which the two clouds are best separated, given the cost ratio.

    A passage qualifies when `score >= threshold`, so the candidates are the
    midpoints between observed scores -- any value between two observations
    behaves identically. Ties go to the higher threshold: when two thresholds
    cost the same, the more cautious one is chosen.

    Candidates that answer less than `min_answer_rate` of the answerable
    questions are excluded first. That constraint is what keeps a heavily
    overlapping corpus from choosing "refuse everything", which the cost
    function alone scores as optimal and which no patient would call correct.
    """
    if not answerable or not unanswerable:
        raise ValueError("need scores from both answerable and unanswerable questions")

    observed = sorted({*answerable, *unanswerable})
    candidates = [
        observed[0] - 0.01,
        *[(a + b) / 2 for a, b in zip(observed, observed[1:], strict=False)],
        observed[-1] + 0.01,
    ]

    def cost(threshold: float) -> float:
        wrong_answers = sum(1 for s in unanswerable if s >= threshold)
        wrong_abstentions = sum(1 for s in answerable if s < threshold)
        return false_answer_cost * wrong_answers + wrong_abstentions

    def answer_rate(threshold: float) -> float:
        return sum(1 for s in answerable if s >= threshold) / len(answerable)

    cheapest = min(candidates, key=lambda t: (cost(t), -t))
    allowed = [t for t in candidates if answer_rate(t) >= min_answer_rate]
    best = min(allowed, key=lambda t: (cost(t), -t)) if allowed else cheapest
    return ThresholdChoice(
        threshold=round(best, 4),
        separable=max(unanswerable) < min(answerable),
        false_answers=sum(1 for s in unanswerable if s >= best),
        false_abstentions=sum(1 for s in answerable if s < best),
        cost=cost(best),
        floor_binding=best != cheapest,
    )


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Outcome:
    question: Question
    best_score: float
    hit: bool | None  # answerable only: every expected document was retrieved
    retrieved: list[str]


async def measure(
    session_factory: async_sessionmaker[AsyncSession],
    embeddings,
    questions: Sequence[Question],
    *,
    model: str,
    k: int,
) -> list[Outcome]:
    outcomes: list[Outcome] = []
    async with session_factory() as session:
        for question in questions:
            # min_score=0 on purpose: this is the measurement that decides what
            # the threshold should be, so it cannot apply one.
            found = await search_knowledge_base(
                session, embeddings, question.question, model=model, k=k, min_score=0.0
            )
            retrieved = [c.source_path for c in found.chunks]
            outcomes.append(
                Outcome(
                    question=question,
                    best_score=found.best_score,
                    hit=set(question.expected_sources) <= set(retrieved)
                    if question.answerable
                    else None,
                    retrieved=retrieved,
                )
            )
    return outcomes


def rates(outcomes: Sequence[Outcome], threshold: float) -> dict[str, float | int]:
    answerable = [o for o in outcomes if o.question.answerable]
    off_topic = [o for o in outcomes if o.question.kind in NEGATIVE_KINDS]
    undecidable = [o for o in outcomes if o.question.kind in UNDECIDABLE_KINDS]

    def share(group: Sequence[Outcome], predicate) -> float:
        return sum(1 for o in group if predicate(o)) / len(group) if group else 0.0

    return {
        "questions": len(outcomes),
        "answerable": len(answerable),
        "off_topic": len(off_topic),
        "undecidable": len(undecidable),
        # Of the questions the corpus answers, how many does the system attempt?
        "answer_rate": share(answerable, lambda o: o.best_score >= threshold),
        # Of the questions that are not about this clinic at all, how many does
        # it correctly refuse? This is the threshold's actual job.
        "off_topic_abstain_rate": share(off_topic, lambda o: o.best_score < threshold),
        # On-topic questions with no answer on file. A low number here is not a
        # broken threshold; it is the share of the work that Phase 4's grounding
        # check has to do, measured rather than assumed.
        "undecidable_abstain_rate": share(undecidable, lambda o: o.best_score < threshold),
        # Retrieval quality, independent of the threshold: was the answering
        # document actually among the passages retrieved?
        "hit_at_k": share(answerable, lambda o: bool(o.hit)),
    }


def by_kind(outcomes: Sequence[Outcome], threshold: float) -> list[dict[str, object]]:
    kinds = [*ANSWERABLE_KINDS, *UNANSWERABLE_KINDS]
    rows: list[dict[str, object]] = []
    for kind in kinds:
        group = [o for o in outcomes if o.question.kind == kind]
        if not group:
            continue
        answerable = group[0].question.answerable
        correct = sum(1 for o in group if (o.best_score >= threshold) == answerable)
        rows.append(
            {
                "kind": kind,
                "answerable": answerable,
                "n": len(group),
                "median_score": round(statistics.median(o.best_score for o in group), 3),
                "max_score": round(max(o.best_score for o in group), 3),
                "correct": correct,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
# Categorical slots 1-3 of the reference palette: the three that validate on
# every pair, in both light and dark, for a scatter form.
ANSWERED_COLOR = "#2a78d6"
UNDECIDABLE_COLOR = "#eb6834"
OFF_TOPIC_COLOR = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"

STRIPS = (
    ("answered by the corpus", ANSWERED_COLOR, lambda q: q.answerable),
    ("on topic, no answer on file", UNDECIDABLE_COLOR, lambda q: q.kind in UNDECIDABLE_KINDS),
    ("off topic", OFF_TOPIC_COLOR, lambda q: q.kind in NEGATIVE_KINDS),
)


def plot_distribution(outcomes: Sequence[Outcome], choice: ThresholdChoice, model: str, path: Path):
    """One strip per class, so every question is visible as its own point.

    Three classes rather than two, because that is the finding: the green
    cloud is what a threshold can reject, and the orange one sits inside the
    blue one no matter where the line goes.
    """
    import matplotlib

    matplotlib.use("Agg")  # no display in CI, and none wanted
    import matplotlib.pyplot as plt

    jitter = random.Random(0)  # fixed, so re-running produces an identical file
    figure, axes = plt.subplots(figsize=(8.4, 3.8), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    rows = list(reversed(range(len(STRIPS))))  # first class on top
    labels = []
    for row, (label, color, belongs) in zip(rows, STRIPS, strict=True):
        scores = [o.best_score for o in outcomes if belongs(o.question)]
        labels.append(f"{label}\n(n={len(scores)})")
        axes.scatter(
            scores,
            [row + jitter.uniform(-0.16, 0.16) for _ in scores],
            s=70,
            color=color,
            edgecolor=SURFACE,  # a surface ring, so overlapping points stay countable
            linewidth=1.2,
            zorder=3,
        )

    top = len(STRIPS) - 1
    axes.axvline(choice.threshold, color=INK, linestyle="--", linewidth=1.2, zorder=2)
    axes.annotate(
        f"threshold {choice.threshold:.3f}",
        xy=(choice.threshold, top + 0.62),
        xytext=(6, 0),
        textcoords="offset points",
        color=INK,
        fontsize=9,
        va="center",
    )
    axes.text(
        choice.threshold - 0.008,
        top + 0.62,
        "abstain",
        color=MUTED,
        fontsize=9,
        ha="right",
        va="center",
    )

    axes.set_yticks(rows)
    axes.set_yticklabels(labels, fontsize=9, color=INK)
    axes.set_ylim(-0.55, top + 0.8)
    axes.set_xlim(0, max(0.05, max(o.best_score for o in outcomes)) * 1.12)
    axes.set_xlabel("best retrieval score (cosine similarity)", fontsize=9, color=MUTED)
    axes.tick_params(axis="x", labelsize=9, colors=MUTED, length=0)
    axes.tick_params(axis="y", length=0)
    axes.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side, spine in axes.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color("#d9d8d3")
    axes.set_title(
        f"What a similarity threshold can and cannot decide · {model}",
        fontsize=11,
        color=INK,
        loc="left",
        pad=12,
    )

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


def render_markdown(
    model: str,
    choice: ThresholdChoice,
    splits: dict[str, dict[str, float | int]],
    kinds: list[dict[str, object]],
    leaks: Sequence[Outcome],
    *,
    k: int,
    cost: float,
    chunks: int,
    plot_name: str,
) -> str:
    def percent(value: float) -> str:
        return f"{value * 100:.0f}%"

    lines = [
        f"# Retrieval calibration -- `{model}`",
        "",
        f"_Generated {datetime.now(UTC):%Y-%m-%d} by `make calibrate`. "
        f"{chunks} chunks, top-k = {k}, a wrong answer counted {cost:g}x a needless "
        "abstention._",
        "",
        f"**Abstention threshold: `{choice.threshold:.3f}`** -- fitted on the calibration",
        "split alone, against off-topic questions. The holdout split had no say in it,",
        "so those are the numbers worth quoting.",
        "",
        f"![score distribution]({plot_name})",
        "",
        "## What the threshold decides",
        "",
        "| split | questions | answers when the corpus answers | rejects off-topic | hit@k |",
        "|---|---|---|---|---|",
    ]
    for name in ("calibration", "holdout"):
        row = splits[name]
        lines.append(
            f"| {name} | {row['questions']} | "
            f"{percent(float(row['answer_rate']))} ({row['answerable']}) | "
            f"{percent(float(row['off_topic_abstain_rate']))} ({row['off_topic']}) | "
            f"{percent(float(row['hit_at_k']))} |"
        )

    lines += [
        "",
        "`hit@k` is retrieval quality and has nothing to do with the threshold: it asks",
        "whether the document that answers the question was among the passages retrieved",
        "at all. A low hit@k is a chunking or corpus problem; a low answer rate with a",
        "high hit@k is a threshold problem.",
        "",
        "## What it cannot decide, and who does",
        "",
        "| split | on-topic questions with no answer on file | stopped by the threshold |",
        "|---|---|---|",
    ]
    for name in ("calibration", "holdout"):
        row = splits[name]
        lines.append(
            f"| {name} | {row['undecidable']} | {percent(float(row['undecidable_abstain_rate']))} |"
        )

    lines += [
        "",
        'Cosine similarity scores *topic*, not answerhood. "How much is the missed',
        'appointment fee" retrieves the section headed "Missed Appointments" -- correctly,',
        "at one of the highest scores in the whole set -- and that section does not state",
        "a fee. To the geometry it is indistinguishable from a question the corpus does",
        "answer, so no threshold separates the two, and raising the threshold until it",
        "does only means refusing real questions.",
        "",
        "That residual is not swept under the rug; it is the measured size of the job",
        "the next layer has to do. Phase 4's agent reads the passage, sets",
        "`sufficient=False`, and a deterministic citation check turns that into an",
        "abstention plus a `KB_GAP` record -- which doubles as the list of documents",
        "this clinic should write. Every question in this class is one of them.",
        "",
        "## By question type (all splits)",
        "",
        "| kind | answered by corpus | n | median score | max score | stopped by threshold |",
        "|---|---|---|---|---|---|",
    ]
    for row in kinds:
        stopped = (
            f"{row['correct']}/{row['n']}"
            if not row["answerable"]
            else f"{int(row['n']) - int(row['correct'])}/{row['n']} refused"
        )
        lines.append(
            f"| {row['kind']} | {'yes' if row['answerable'] else 'no'} | {row['n']} "
            f"| {row['median_score']} | {row['max_score']} | {stopped} |"
        )

    lines += [
        "",
        "## Off-topic questions that got through",
        "",
    ]
    if leaks:
        lines += ["| id | score | what it means |", "|---|---|---|"]
        lines += [
            f"| `{o.question.id}` | {o.best_score:.3f} | "
            "an unrelated question reached the evidence layer |"
            for o in leaks
        ]
    else:
        lines.append(
            "None, in either split. Off-topic questions score far below anything the "
            "corpus covers, which is exactly the separation the threshold is there to "
            "enforce."
        )
    lines.append("")

    if choice.floor_binding:
        lines += [
            f"_The minimum answer rate ({MIN_ANSWER_RATE:.0%}) was binding: the",
            "cost-minimising threshold would have refused more real questions than a",
            "usable assistant can. An assistant that answers nothing is not cautious,",
            "it is broken, and the cost function has no way to know that._",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
async def run(
    settings: Settings, *, k: int, cost: float, min_answer_rate: float, plot: bool
) -> int:
    embeddings = build_embeddings(settings)
    if embeddings is None:
        print("✗ no embedding model configured (set OPENAI_API_KEY or --provider fake)")
        return 1
    model = embedding_model_name(settings)
    questions = load_questions()

    engine = build_engine(settings)
    try:
        session_factory = build_session_factory(engine)
        report = await ingest(session_factory, embeddings, kb_dir=settings.kb_dir, model_name=model)
        print(f"· corpus: {report.summary()}")
        outcomes = await measure(session_factory, embeddings, questions, model=model, k=k)
    finally:
        await engine.dispose()

    fitting = [o for o in outcomes if o.question.split == "calibration"]
    choice = choose_threshold(
        [o.best_score for o in fitting if o.question.answerable],
        # Off-topic only. Fitting against the on-topic-but-unanswered questions
        # would price in an error no threshold can avoid -- see the docstring.
        [o.best_score for o in fitting if o.question.kind in NEGATIVE_KINDS],
        false_answer_cost=cost,
        min_answer_rate=min_answer_rate,
    )
    splits = {
        name: rates([o for o in outcomes if o.question.split == name], choice.threshold)
        for name in ("calibration", "holdout")
    }
    leaks = [
        o
        for o in outcomes
        if o.question.kind in NEGATIVE_KINDS and o.best_score >= choice.threshold
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"retrieval-calibration-{model}"
    plot_name = f"{stem}.png"
    if plot:
        plot_distribution(outcomes, choice, model, REPORT_DIR / plot_name)

    (REPORT_DIR / f"{stem}.json").write_text(
        json.dumps(
            {
                "model": model,
                "generated": datetime.now(UTC).strftime("%Y-%m-%d"),
                "top_k": k,
                "false_answer_cost": cost,
                "min_answer_rate": min_answer_rate,
                "floor_binding": choice.floor_binding,
                "threshold": choice.threshold,
                "separable": choice.separable,
                "splits": splits,
                "by_kind": by_kind(outcomes, choice.threshold),
                "scores": [
                    {
                        "id": o.question.id,
                        "kind": o.question.kind,
                        "split": o.question.split,
                        "answerable": o.question.answerable,
                        "best_score": round(o.best_score, 4),
                        "hit": o.hit,
                        "retrieved": o.retrieved,
                    }
                    for o in outcomes
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / f"{stem}.md").write_text(
        render_markdown(
            model,
            choice,
            splits,
            by_kind(outcomes, choice.threshold),
            leaks,
            k=k,
            cost=cost,
            chunks=report.chunks_total,
            plot_name=plot_name,
        ),
        encoding="utf-8",
    )

    print(
        f"✓ threshold {choice.threshold:.3f} for {model} "
        f"({'separable' if choice.separable else 'overlapping'} against off-topic; "
        f"{choice.false_answers} off-topic would get through, "
        f"{choice.false_abstentions} real questions refused, on the calibration split)"
    )
    for name, row in splits.items():
        print(
            f"  {name:<12} answers {float(row['answer_rate']):.0%} of answerable · "
            f"rejects {float(row['off_topic_abstain_rate']):.0%} of off-topic · "
            f"stops {float(row['undecidable_abstain_rate']):.0%} of on-topic-no-answer · "
            f"hit@{k} {float(row['hit_at_k']):.0%}"
        )
    print(f"  report: evals/calibration/{stem}.md")
    print("  put this number in CALIBRATED_MIN_SCORE in adapters/llm/embeddings.py")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "fake"), default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--cost", type=float, default=FALSE_ANSWER_COST)
    parser.add_argument("--min-answer-rate", type=float, default=MIN_ANSWER_RATE)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    configured = get_settings()
    if args.provider:
        configured = configured.model_copy(update={"llm_provider": args.provider})
    raise SystemExit(
        asyncio.run(
            run(
                configured,
                k=args.k or configured.rag_top_k,
                cost=args.cost,
                min_answer_rate=args.min_answer_rate,
                plot=not args.no_plot,
            )
        )
    )
