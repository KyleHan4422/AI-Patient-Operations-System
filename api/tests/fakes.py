"""Test doubles for the conversation graph and the knowledge base."""

from __future__ import annotations

from typing import Any

import httpx
import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from patient_ops.adapters.llm.embeddings import HashingEmbeddings
from patient_ops.adapters.llm.fake import EchoChatModel
from patient_ops.db.transcript import TurnRecord


class ListRecorder:
    """In-memory stand-in for db.transcript.SqlTranscript."""

    def __init__(self) -> None:
        self.turns: list[TurnRecord] = []

    async def record_turn(self, turn: TurnRecord) -> None:
        self.turns.append(turn)


class RecordingEcho(EchoChatModel):
    """The echo model, remembering exactly what it was shown on each call."""

    seen: list[list[BaseMessage]] = Field(default_factory=list)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        self.seen.append(list(messages))
        return await super()._agenerate(messages, stop, run_manager, **kwargs)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs: Any):
        self.seen.append(list(messages))
        async for chunk in super()._astream(messages, stop, run_manager, **kwargs):
            yield chunk


class TimingOutModel(BaseChatModel):
    """A provider that always times out -- after the SDK's own retries."""

    @property
    def _llm_type(self) -> str:
        return "timing-out-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        raise openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com"))


class SilentModel(EchoChatModel):
    """A model whose reply is empty -- one empty chunk, as a real provider sends it."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
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
