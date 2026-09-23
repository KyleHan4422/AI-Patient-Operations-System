#!/usr/bin/env python3
"""Measure what the assistant actually does with a question, end to end.

`make calibrate` measures retrieval: how close the right passage scores. It
stops where cosine similarity stops -- at topic, not at answerhood -- and hands
the rest to this layer. This script measures the rest:

  - does an answerable question get answered, from the document that answers
    it, with a citation;
  - does a question the documents do not answer get refused, and recorded as a
    KB_GAP the clinic can act on;
  - does a question whose answer lives in a table reach the table, and come
    back in the clinic's own words rather than the model's.

Every question comes from evals/retrieval/kb_questions.yaml, so the same set
that calibrated the threshold now grades the layer built on top of it. What a
question kind is *expected* to do lives in evals/knowledge/expectations.yaml,
because that is a Phase 4 judgement and the labels are Phase 3's.

The chat model is part of the measurement, like the embedding model is for
calibration: reports are written per model and the numbers are not comparable
across them. Run with --provider fake and the report says so at the top -- the
offline stand-in answers by keyword and cannot judge whether a passage answers
a question, which is the single hardest thing being graded here.

Usage:  python scripts/eval_knowledge.py [--provider fake] [--limit 5]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from patient_ops.adapters.llm.client import build_chat_model
from patient_ops.adapters.llm.embeddings import (
    build_embeddings,
    embedding_model_name,
    min_score_for,
)
from patient_ops.config import REPO_ROOT, Settings, get_settings
from patient_ops.db.session import build_engine, build_session_factory
from patient_ops.db.transcript import TurnRecord
from patient_ops.graph.build import build_graph
from patient_ops.graph.context import GraphContext
from patient_ops.graph.turn import Final, run_turn, turn_config
from patient_ops.rag.ingest import ingest
from patient_ops.tools.registry import ReadOnlyToolset

QUESTIONS_PATH = REPO_ROOT / "evals" / "retrieval" / "kb_questions.yaml"
EXPECTATIONS_PATH = REPO_ROOT / "evals" / "knowledge" / "expectations.yaml"
REPORT_DIR = REPO_ROOT / "evals" / "knowledge"

# What each kind of question should produce, as state's `answer_kind`.
# Overridden per question by expectations.yaml, which is also where the reason
# for an override is written down.
EXPECTED_BY_KIND: dict[str, str] = {
    "direct": "passages",
    "paraphrase": "passages",
    "multi_hop": "passages",
    "near_miss": "abstained",
    "off_topic": "abstained",
    "structured": "facts",
}


def _load_questions() -> list[Any]:
    """Phase 3's labelled set, loaded by Phase 3's own loader.

    scripts/ is not a package, so it is imported by path -- the same way the
    tests import a script. Copying the Question model instead would let the two
    files disagree about what a label means, which is the one thing that must
    not happen between a calibration report and this one.
    """
    path = Path(__file__).resolve().parent / "calibrate_threshold.py"
    spec = importlib.util.spec_from_file_location("scripts.calibrate_threshold", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: its dataclasses look their own module up while
    # the module body is still executing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_questions(QUESTIONS_PATH)


class Expectation(BaseModel):
    """What one question should produce, when its kind's default is wrong."""

    kind: str
    why: str = ""
    must_say: list[str] = []
    must_not_say: list[str] = []


def load_expectations(path: Path = EXPECTATIONS_PATH) -> dict[str, Expectation]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {key: Expectation.model_validate(value) for key, value in raw.items()}


@dataclass
class Outcome:
    question: Any
    expected: Expectation
    answer_kind: str
    reply: str
    citations: list[int]
    sources: list[str]
    tools: list[str]
    gap: dict[str, Any] | None
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


class NullRecorder:
    """The eval does not write conversations: it grades the answer, not the log."""

    def __init__(self) -> None:
        self.turns: list[TurnRecord] = []

    async def record_turn(self, turn: TurnRecord) -> None:
        self.turns.append(turn)


def grade(outcome: Outcome) -> Outcome:
    expected = outcome.expected
    if outcome.answer_kind != expected.kind:
        # facts_and_passages satisfies an expectation of either half: the
        # record was still read, and the prose still had to pass its check.
        if not (
            outcome.answer_kind == "facts_and_passages" and expected.kind in ("facts", "passages")
        ):
            outcome.failures.append(f"answered as {outcome.answer_kind}, expected {expected.kind}")
    spoken = outcome.reply.lower()
    outcome.failures += [f"never said {s!r}" for s in expected.must_say if s.lower() not in spoken]
    outcome.failures += [f"said {s!r}" for s in expected.must_not_say if s.lower() in spoken]

    wanted = set(getattr(outcome.question, "expected_sources", []) or [])
    if wanted and outcome.answer_kind in ("passages", "facts_and_passages"):
        if not wanted & set(outcome.sources):
            outcome.failures.append(
                f"cited {outcome.sources or '[]'}, expected one of {sorted(wanted)}"
            )
    return outcome


async def ask(
    graph: Any,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    chat_model: Any,
    embeddings: Any,
    question: str,
) -> tuple[str, dict[str, Any], ReadOnlyToolset]:
    model = embedding_model_name(settings)
    toolset = ReadOnlyToolset(
        session_factory=session_factory,
        embeddings=embeddings,
        embedding_model=model,
        min_score=min_score_for(model, settings.rag_min_score),
        top_k=settings.rag_top_k,
        tz=settings.clinic_tz,
    )
    thread_id = uuid.uuid4()
    context = GraphContext(
        thread_id=thread_id,
        request_id="eval",
        chat_model=chat_model,
        recorder=NullRecorder(),
        toolset=toolset,
        max_tool_rounds=settings.agent_max_tool_rounds,
    )
    reply = ""
    async for event in run_turn(graph, text=question, context=context):
        if isinstance(event, Final):
            reply = event.text
    state = (await graph.aget_state(turn_config(thread_id))).values
    return reply, state, toolset


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def by_kind(outcomes: Sequence[Outcome]) -> list[dict[str, Any]]:
    kinds: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        kinds.setdefault(outcome.question.kind, []).append(outcome)
    return [
        {
            "kind": kind,
            "n": len(group),
            "correct": sum(o.ok for o in group),
            "expected": group[0].expected.kind,
        }
        for kind, group in sorted(kinds.items())
    ]


def render_markdown(
    outcomes: Sequence[Outcome], *, chat_model: str, embedding_model: str, chunks: int, fake: bool
) -> str:
    correct = sum(o.ok for o in outcomes)
    gaps = [o for o in outcomes if o.gap]
    reasons = Counter(o.gap["reason"] for o in gaps)

    lines = [
        f"# Knowledge answering -- `{chat_model}`",
        "",
        f"_Generated {datetime.now(UTC):%Y-%m-%d} by `make eval-knowledge`. "
        f"{len(outcomes)} questions, {chunks} chunks, embeddings `{embedding_model}`._",
        "",
        f"**{correct}/{len(outcomes)} questions behaved as expected.**",
        "",
    ]
    if fake:
        lines += [
            "> Run offline, against the stand-in model and the stand-in embedder. Both have",
            "> a ceiling this table runs into, and neither is a defect in the system being",
            "> graded:",
            ">",
            "> - the stand-in model picks a tool by keyword and answers from the best passage",
            ">   it is handed. It cannot judge whether a passage answers the question, which",
            ">   is exactly what `near_miss` measures -- so that row measures the plumbing",
            ">   around the judgement, not the judgement.",
            "> - the stand-in embedder is a bag of words, so it ranks by vocabulary overlap.",
            ">   A `paraphrase` that cites the wrong document usually failed here, not in",
            ">   the grounding check (see the hit@k figures in evals/calibration/).",
            ">",
            "> What this run does establish: which tool each question reaches, that exact",
            "> facts are worded from the record, that every citation was really retrieved,",
            "> that no figure survives that is not in a cited passage, and that an",
            "> abstention leaves a KB_GAP. For answer quality, run against a real model.",
            "",
        ]
    lines += [
        "## By question type",
        "",
        "| kind | expected | n | as expected |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {row['kind']} | {row['expected']} | {row['n']} | {row['correct']}/{row['n']} |"
        for row in by_kind(outcomes)
    ]

    lines += [
        "",
        "`facts` means the reply was written from a row in the clinic's tables by",
        "graph/replies.py; `passages` means the model wrote it and every figure in it was",
        "found in a passage it cited; `abstained` means neither survived, and the clinic",
        "got a KB_GAP instead.",
        "",
        "## What the clinic could not answer",
        "",
    ]
    if gaps:
        lines += [
            "Each row is a document this clinic has not written, or has written in words no",
            "patient uses. That list is the point of abstaining.",
            "",
            "| question | why | closest section | score |",
            "|---|---|---|---|",
        ]
        for outcome in gaps:
            gap = outcome.gap or {}
            score = f"{gap['best_score']:.3f}" if gap.get("best_score") is not None else "--"
            lines.append(
                f"| {outcome.question.question} | `{gap['reason']}` | "
                f"{gap.get('nearest_heading') or '--'} | {score} |"
            )
        lines += ["", "By reason: " + ", ".join(f"`{r}` {n}" for r, n in reasons.most_common()), ""]
    else:
        lines += ["No question was refused in this run.", ""]

    unexpected = [o for o in outcomes if not o.ok]
    lines += ["## Questions that did not behave as expected", ""]
    if unexpected:
        lines += ["| id | kind | what happened |", "|---|---|---|"]
        lines += [
            f"| `{o.question.id}` | {o.question.kind} | {'; '.join(o.failures)} |"
            for o in unexpected
        ]
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
async def run(settings: Settings, *, limit: int | None) -> int:
    chat_model = build_chat_model(settings)
    embeddings = build_embeddings(settings)
    if chat_model is None or embeddings is None:
        print("✗ no model configured (set OPENAI_API_KEY or --provider fake)")
        return 1

    questions = _load_questions()[: limit or None]
    expectations = load_expectations()
    model = embedding_model_name(settings)

    engine = build_engine(settings)
    try:
        session_factory = build_session_factory(engine)
        ingested = await ingest(
            session_factory, embeddings, kb_dir=settings.kb_dir, model_name=model
        )
        print(f"· corpus: {ingested.summary()}")

        graph = build_graph(InMemorySaver())
        outcomes: list[Outcome] = []
        for question in questions:
            reply, state, toolset = await ask(
                graph, session_factory, settings, chat_model, embeddings, question.question
            )
            expected = expectations.get(
                question.id, Expectation(kind=EXPECTED_BY_KIND[question.kind])
            )
            cited = state.get("citations") or []
            outcome = grade(
                Outcome(
                    question=question,
                    expected=expected,
                    answer_kind=state.get("answer_kind") or "none",
                    reply=reply,
                    citations=cited,
                    sources=sorted(
                        {p.source_path for p in toolset.passages.values() if p.chunk_id in cited}
                    ),
                    tools=[call.name for call in toolset.trace],
                    gap=state.get("kb_gap"),
                )
            )
            outcomes.append(outcome)
            print(f"  {'✓' if outcome.ok else '✗'} {question.id:<24} {outcome.answer_kind}")
    finally:
        await engine.dispose()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"knowledge-eval-{settings.llm_model if settings.llm_provider != 'fake' else 'fake'}"
    (REPORT_DIR / f"{stem}.json").write_text(
        json.dumps(
            {
                "generated": datetime.now(UTC).strftime("%Y-%m-%d"),
                "chat_model": settings.llm_model if settings.llm_provider != "fake" else "fake",
                "embedding_model": model,
                "correct": sum(o.ok for o in outcomes),
                "questions": len(outcomes),
                "by_kind": by_kind(outcomes),
                "outcomes": [
                    {
                        "id": o.question.id,
                        "kind": o.question.kind,
                        "expected": o.expected.kind,
                        "answer_kind": o.answer_kind,
                        "tools": o.tools,
                        "citations": o.citations,
                        "gap": o.gap,
                        "failures": o.failures,
                        "reply": o.reply,
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
            outcomes,
            chat_model=settings.llm_model if settings.llm_provider != "fake" else "fake",
            embedding_model=model,
            chunks=ingested.chunks_total,
            fake=settings.llm_provider == "fake",
        ),
        encoding="utf-8",
    )

    correct = sum(o.ok for o in outcomes)
    print(f"\n✓ {correct}/{len(outcomes)} as expected · report: evals/knowledge/{stem}.md")
    return 0 if correct == len(outcomes) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "fake"), default=None)
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    args = parser.parse_args()

    configured = get_settings()
    if args.provider:
        configured = configured.model_copy(update={"llm_provider": args.provider})
    raise SystemExit(asyncio.run(run(configured, limit=args.limit)))
