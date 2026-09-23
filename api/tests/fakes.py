"""Test doubles for the conversation graph and the knowledge base."""

from __future__ import annotations

from typing import Any

import httpx
import openai
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from patient_ops.adapters.llm.embeddings import HashingEmbeddings
from patient_ops.adapters.llm.fake import EchoChatModel
from patient_ops.db.transcript import TurnRecord
from patient_ops.tools import registry


class ListRecorder:
    """In-memory stand-in for db.transcript.SqlTranscript."""

    def __init__(self) -> None:
        self.turns: list[TurnRecord] = []

    async def record_turn(self, turn: TurnRecord) -> None:
        self.turns.append(turn)


INTENT_SCHEMA = "IntentDecision"


def is_intent_call(kwargs: dict[str, Any]) -> bool:
    """True when this call is classify_intent's with_structured_output()."""
    return [t["function"]["name"] for t in kwargs.get("tools") or []] == [INTENT_SCHEMA]


def intent_result(intent: str) -> ChatResult:
    call = {"name": INTENT_SCHEMA, "args": {"intent": intent}, "id": "call_intent"}
    return ChatResult(generations=[ChatGeneration(message=AIMessage("", tool_calls=[call]))])


class PinnedIntent(EchoChatModel):
    """The echo model, routed to one branch whatever the patient wrote.

    The graph's own properties -- memory across turns, how big the window is,
    what gets recorded, what a failed turn leaves behind -- are the same on
    every branch, and small talk is the branch that needs no database and no
    tools. So they are tested there, exactly, and routing gets its own tests
    instead of being smuggled into all of them.
    """

    intent: str = "smalltalk"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        if is_intent_call(kwargs):
            return intent_result(self.intent)
        return super()._generate(messages, stop, run_manager, **kwargs)


class RecordingEcho(PinnedIntent):
    """The echo model, remembering exactly what it was shown on each call.

    Tool-calling rounds are not recorded: `seen` answers "what was the model
    shown before it wrote the reply", and the classifier's own prompt is a
    different question with its own test.
    """

    seen: list[list[BaseMessage]] = Field(default_factory=list)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        if not kwargs.get("tools"):
            self.seen.append(list(messages))
        return await super()._agenerate(messages, stop, run_manager, **kwargs)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs: Any):
        self.seen.append(list(messages))
        async for chunk in super()._astream(messages, stop, run_manager, **kwargs):
            yield chunk


class TimingOutModel(EchoChatModel):
    """A provider that always times out -- after the SDK's own retries.

    Built on the stand-in rather than on BaseChatModel so that it can be bound
    with tools: the first call of every turn is the intent classifier, and a
    double that cannot be bound would fail with NotImplementedError instead of
    the timeout the test is about.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        raise openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com"))


class SilentModel(PinnedIntent):
    """A model whose reply is empty -- one empty chunk, as a real provider sends it."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        if is_intent_call(kwargs):
            return super()._generate(messages, stop, run_manager, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs: Any):
        yield ChatGenerationChunk(message=AIMessageChunk(content=""))


class CountingEmbeddings(HashingEmbeddings):
    """The offline embedder, counting how many vectors it was asked for.

    Ingestion is idempotent only if it stops asking. With a paid provider the
    count is the bill, so the assertion is "zero", not "fast".
    """

    def __init__(self) -> None:
        super().__init__()
        self.embedded: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return super().embed_documents(texts)

    @property
    def calls(self) -> int:
        return len(self.embedded)


class FailingEmbeddings(HashingEmbeddings):
    """Times out once it has embedded `after` texts -- a provider dying mid-run.

    Both entry points fail: ingestion embeds documents, retrieval embeds a
    query, and a double that covers only one of them silently passes the test
    it was written for.
    """

    def __init__(self, after: int = 0) -> None:
        super().__init__()
        self.after = after

    @staticmethod
    def _timeout() -> openai.APITimeoutError:
        return openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(texts) > self.after:
            raise self._timeout()
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        raise self._timeout()


class WrongWidthEmbeddings(HashingEmbeddings):
    """A model whose vectors do not fit the column."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]


class ScriptedModel(EchoChatModel):
    """Returns the replies it was given, in order, whatever it is shown.

    The agent loop is a state machine over what a model happens to return, so
    the only way to test the branches that matter -- a verdict that will not
    parse, a model that answers before it has looked anything up, one that
    never stops looking -- is to say exactly what comes back.
    """

    script: list[AIMessage] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    bound: list[list[str]] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        self.seen.append(list(messages))
        self.bound.append([t["function"]["name"] for t in kwargs.get("tools") or []])
        if len(self.seen) > len(self.script):
            raise AssertionError(
                f"the model was called {len(self.seen)} times, and the script has "
                f"{len(self.script)} replies"
            )
        return ChatResult(generations=[ChatGeneration(message=self.script[len(self.seen) - 1])])


class StubToolset:
    """The read-only toolset, answering from a script.

    Structurally a ReadOnlyToolset as far as the agent is concerned -- it hands
    over tools and remembers what was called -- without a database behind it.
    What each tool *returns* has its own tests; this one is about the loop.
    """

    def __init__(self, **results: str) -> None:
        self.results = results
        self.asked: list[tuple[str, dict[str, Any]]] = []
        self.trace: list[Any] = []

    def _tool(self, name: str, schema: type[BaseModel]) -> StructuredTool:
        async def run(**kwargs: Any) -> str:
            self.asked.append((name, kwargs))
            return self.results.get(name, "")

        return StructuredTool.from_function(
            coroutine=run, name=name, args_schema=schema, description=name
        )

    def tools(self) -> list[StructuredTool]:
        return [
            self._tool(registry.SEARCH_DOCUMENTS, registry.SearchArgs),
            self._tool(registry.LOOKUP_INSURANCE_PLAN, registry.InsuranceArgs),
            self._tool(registry.LOOKUP_PRICE, registry.PriceArgs),
            self._tool(registry.GET_OPENING_HOURS, registry.NoArgs),
        ]

    async def procedure_catalogue(self) -> list[tuple[str, str]]:
        return [("CROWN", "Crown")]
