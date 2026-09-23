"""The read-only toolset: everything an agent is allowed to reach, and nothing else.

One object, built fresh for each turn, holding the four tools the knowledge
agent may call and a record of every call it made. It is handed to the agent
through the graph's runtime context, so `agents/` needs no import of the
database, the retrieval layer or any adapter -- the rule
scripts/check_invariants.py enforces.

Two properties this class is responsible for:

  read-only   Every tool below reads. There is no write path to reach from
              here, so an agent that decides to book an appointment has no way
              to act on it. Autonomy is scoped by making the action space
              small, not by asking the model nicely.
  traceable   Each call appends to `trace`, which the deterministic part of the
              turn uses for three things: to check the answer against the
              evidence, to write the tool_calls rows, and to know which exact
              facts were looked up so the reply can be written from the record
              instead of from the model's prose.

Sessions are opened per call and closed again. A session held open across the
agent's thinking would be a transaction held open across a provider call --
Phase 3's lesson, in a place where the call is a whole reasoning loop.
"""

from __future__ import annotations

import time as clock
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from patient_ops.db import repo
from patient_ops.obs.logging import get_logger
from patient_ops.rag.retrieve import Retrieval, RetrievedChunk, search_knowledge_base
from patient_ops.tools import clinic, knowledge
from patient_ops.tools.facts import Fact, for_model

log = get_logger(__name__)

SEARCH_DOCUMENTS = "search_documents"
LOOKUP_INSURANCE_PLAN = "lookup_insurance_plan"
LOOKUP_PRICE = "lookup_price"
GET_OPENING_HOURS = "get_opening_hours"


# ---------------------------------------------------------------------------
# What the model is asked to fill in
# ---------------------------------------------------------------------------
class SearchArgs(BaseModel):
    query: str = Field(description="The patient's question, in the patient's own words.")


class InsuranceArgs(BaseModel):
    plan_name: str = Field(
        description="The insurance plan exactly as the patient named it, e.g. 'Delta Dental PPO'."
    )


class PriceArgs(BaseModel):
    procedure: str = Field(
        description="A treatment code from the list of treatments on file, e.g. 'CROWN'."
    )


class NoArgs(BaseModel):
    pass


# ---------------------------------------------------------------------------
def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Looked:
    """What a tool body found: a fact, or a set of passages, never both."""

    summary: str
    fact: Fact | None = None
    retrieval: Retrieval | None = None


@dataclass(frozen=True)
class ToolInvocation:
    """One completed call: what was asked, what came back, how long it took.

    Only completed calls are recorded. A tool that fails raises, which ends the
    turn before anything is written -- the failure is in the log, not in
    tool_calls. When Phase 8 makes tool failures survivable, they get their row.
    """

    name: str
    args: dict[str, Any]
    latency_ms: int
    summary: str  # one line, for the log and the tool_calls row
    fact: Fact | None = None
    retrieval: Retrieval | None = None


@dataclass
class ReadOnlyToolset:
    session_factory: async_sessionmaker[AsyncSession]
    embeddings: Embeddings
    embedding_model: str
    min_score: float
    top_k: int = 4
    tz: ZoneInfo = ZoneInfo("UTC")
    now: Callable[[], datetime] = _utc_now
    trace: list[ToolInvocation] = field(default_factory=list)

    # -- what the deterministic half of the turn reads afterwards -------------
    @property
    def facts(self) -> list[Fact]:
        """Exact facts looked up this turn, in the order they were looked up."""
        return [i.fact for i in self.trace if i.fact is not None]

    @property
    def passages(self) -> dict[int, RetrievedChunk]:
        """Every passage the agent was shown, by chunk id -- the evidence a
        citation has to be found in."""
        return {
            chunk.chunk_id: chunk
            for i in self.trace
            if i.retrieval is not None
            for chunk in i.retrieval.chunks
        }

    @property
    def searches(self) -> list[Retrieval]:
        return [i.retrieval for i in self.trace if i.retrieval is not None]

    async def procedure_catalogue(self) -> list[tuple[str, str]]:
        """(code, name) for every treatment on file.

        Goes into the agent's prompt so it passes a code that exists instead of
        inventing one; a lookup tool that has to guess at its own argument is a
        fuzzy match wearing a different hat.
        """
        async with self.session_factory() as session:
            return [(p.code, p.name) for p in await repo.list_procedures(session)]

    # -- the tools themselves -------------------------------------------------
    def tools(self) -> list[BaseTool]:
        return [
            StructuredTool.from_function(
                coroutine=self._search_documents,
                name=SEARCH_DOCUMENTS,
                args_schema=SearchArgs,
                description=(
                    "Search what the clinic has written down: policies, aftercare, what to "
                    "expect, how to prepare, cancellations, privacy. Returns passages to "
                    "quote and cite. Use it for anything that is not an insurance plan, a "
                    "price or opening hours."
                ),
            ),
            StructuredTool.from_function(
                coroutine=self._lookup_insurance_plan,
                name=LOOKUP_INSURANCE_PLAN,
                args_schema=InsuranceArgs,
                description=(
                    "Whether the clinic accepts one named insurance plan. The only way to "
                    "answer that question: the documents do not list plans."
                ),
            ),
            StructuredTool.from_function(
                coroutine=self._lookup_price,
                name=LOOKUP_PRICE,
                args_schema=PriceArgs,
                description=(
                    "The price the clinic has on file for one treatment. The only way to "
                    "answer a price question: the documents do not carry prices."
                ),
            ),
            StructuredTool.from_function(
                coroutine=self._get_opening_hours,
                name=GET_OPENING_HOURS,
                args_schema=NoArgs,
                description=(
                    "The days and times the clinic is open, and the days it is closed. "
                    "This is not appointment availability -- you cannot see free slots."
                ),
            ),
        ]

    async def _search_documents(self, query: str) -> str:
        async def body(session: AsyncSession) -> Looked:
            found = await search_knowledge_base(
                session,
                self.embeddings,
                query,
                model=self.embedding_model,
                k=self.top_k,
                min_score=self.min_score,
            )
            summary = (
                f"{len(found.chunks)} passage(s) at or above {found.threshold:.3f}; "
                f"best {found.best_score:.3f}"
            )
            return Looked(summary=summary, retrieval=found)

        return await self._run(SEARCH_DOCUMENTS, {"query": query}, body)

    async def _lookup_insurance_plan(self, plan_name: str) -> str:
        async def body(session: AsyncSession) -> Looked:
            fact = await clinic.lookup_insurance_plan(session, plan_name)
            return Looked(summary=fact.status, fact=fact)

        return await self._run(LOOKUP_INSURANCE_PLAN, {"plan_name": plan_name}, body)

    async def _lookup_price(self, procedure: str) -> str:
        async def body(session: AsyncSession) -> Looked:
            fact = await clinic.lookup_price(session, procedure)
            return Looked(summary=fact.status, fact=fact)

        return await self._run(LOOKUP_PRICE, {"procedure": procedure}, body)

    async def _get_opening_hours(self) -> str:
        async def body(session: AsyncSession) -> Looked:
            fact = await clinic.opening_hours(session, today=self._today())
            summary = f"{len(fact.windows)} window(s), {len(fact.closures)} closure(s)"
            return Looked(summary=summary, fact=fact)

        return await self._run(GET_OPENING_HOURS, {}, body)

    # -- the one place a tool call is timed, traced, logged and rendered ------
    async def _run(
        self,
        name: str,
        args: dict[str, Any],
        body: Callable[[AsyncSession], Awaitable[Looked]],
    ) -> str:
        started = clock.perf_counter()
        # One short session per call, opened here and closed here: the agent's
        # next thought happens with no transaction open.
        async with self.session_factory() as session:
            looked = await body(session)

        invocation = ToolInvocation(
            name=name,
            args=args,
            latency_ms=round((clock.perf_counter() - started) * 1000),
            summary=looked.summary,
            fact=looked.fact,
            retrieval=looked.retrieval,
        )
        self.trace.append(invocation)
        log.info(
            "tool_call",
            tool=name,
            args=args,
            latency_ms=invocation.latency_ms,
            summary=invocation.summary,
        )
        if looked.retrieval is not None:
            return knowledge.for_model(looked.retrieval)
        assert looked.fact is not None
        return for_model(looked.fact)

    def _today(self) -> date:
        return self.now().astimezone(self.tz).date()
