"""The one door to a language model.

Nothing else in the codebase names a provider. Graph nodes receive a
`BaseChatModel` (LangChain's provider-neutral interface) by injection and never
import this module -- so moving a deployment onto a customer's Azure OpenAI or
Bedrock account is a change to this file, not a grep across the repository.

Two jobs besides construction:

  retries   Transport-level only: the SDK's own backoff for timeouts, 429s and
            5xx, configured here and invisible to the graph. Retrying because
            the *business* situation changed is a graph edge, never this.
  errors    Every provider failure is mapped onto the shared ErrorCode, so
            callers decide from a code, not from an exception class.
"""

from __future__ import annotations

import openai
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from patient_ops.adapters.llm.fake import EchoChatModel
from patient_ops.config import Settings
from patient_ops.errors import ErrorCode


def build_chat_model(settings: Settings) -> BaseChatModel | None:
    """The configured model, or None when no provider is configured.

    None instead of an exception: boot must succeed without an API key, so
    that the chat endpoint can answer "not configured" rather than the process
    failing to start (Settings already refuses this in prod).
    """
    if settings.llm_provider == "fake":
        return EchoChatModel()
    if not settings.llm_configured:
        return None
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        stream_usage=True,  # token counts arrive on the final streamed chunk
    )


# Ordered: APITimeoutError subclasses APIConnectionError, so it must come first
# in any list that treats them differently. Today they map the same way.
_TRANSIENT = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)
# Retrying these cannot help: a bad key, a missing model, a malformed or
# oversized request.
_PERMANENT = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    openai.BadRequestError,
    openai.UnprocessableEntityError,
)


def classify_llm_error(exc: BaseException) -> ErrorCode | None:
    """ErrorCode for a provider failure; None if `exc` is not a provider error."""
    if isinstance(exc, _TRANSIENT):
        return ErrorCode.TRANSIENT
    if isinstance(exc, _PERMANENT):
        return ErrorCode.PERMANENT
    if isinstance(exc, openai.OpenAIError):
        return ErrorCode.UNKNOWN
    return None
