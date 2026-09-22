"""A deterministic, offline stand-in for a real chat model.

Its reply is a function of what it was shown -- how many user messages, the
first one and the latest one. That is the whole point: the property Phase 2
has to prove is "the model saw turn one", and a real model's wording cannot be
asserted on, while this reply can.

It streams word by word, because LangGraph's `messages` stream mode can only
forward tokens from a model that implements streaming.

Used by the test suite, by CI (which has no API key) and by anyone running the
demo without one. Settings refuse LLM_PROVIDER=fake when APP_ENV=prod.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


def echo_reply(messages: list[BaseMessage]) -> str:
    said = [m.text for m in messages if isinstance(m, HumanMessage)]
    if not said:
        return "(fake) I haven't seen any message from you yet."
    return (
        f"(fake) I can see {len(said)} message(s) from you. "
        f'The first: "{said[0]}". The latest: "{said[-1]}".'
    )


def _tokens(text: str) -> list[str]:
    # Split on whitespace but keep it, so the chunks join back to the original.
    return [t for t in re.split(r"(\s)", text) if t]


class EchoChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "echo-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(content=echo_reply(messages))
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages)  # no IO, so no executor hop

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for token in _tokens(echo_reply(messages)):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for token in _tokens(echo_reply(messages)):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                await run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk
