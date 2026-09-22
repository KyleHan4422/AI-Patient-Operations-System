"""Test doubles for the conversation graph."""

from __future__ import annotations

from typing import Any

import httpx
import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

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
