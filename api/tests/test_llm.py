"""The LLM adapter and its configuration guards. No network, no database."""

from __future__ import annotations

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from patient_ops.adapters.llm.client import build_chat_model, classify_llm_error
from patient_ops.adapters.llm.fake import EchoChatModel, echo_reply
from patient_ops.config import Settings
from patient_ops.errors import ErrorCode

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def status_error(cls: type[openai.APIStatusError], code: int) -> openai.APIStatusError:
    return cls("boom", response=httpx.Response(code, request=REQUEST), body=None)


# ---------------------------------------------------------------------------
# Configuration: dangerous combinations are startup errors
# ---------------------------------------------------------------------------
def test_fake_llm_is_refused_in_prod():
    with pytest.raises(ValidationError, match="LLM_PROVIDER=fake"):
        Settings(app_env="prod", llm_provider="fake")


def test_missing_key_is_refused_in_prod():
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(app_env="prod", llm_provider="openai", openai_api_key="")


def test_prod_with_a_key_starts():
    assert Settings(app_env="prod", llm_provider="openai", openai_api_key="sk-x").llm_configured


def test_missing_key_still_boots_in_dev_but_is_not_configured():
    settings = Settings(app_env="dev", llm_provider="openai", openai_api_key="")
    assert not settings.llm_configured
    assert build_chat_model(settings) is None


def test_api_key_never_appears_in_repr():
    settings = Settings(app_env="dev", openai_api_key="sk-very-secret")
    assert "sk-very-secret" not in repr(settings)
    assert "sk-very-secret" not in str(settings.model_dump())


def test_factory_picks_the_configured_provider():
    assert isinstance(build_chat_model(Settings(llm_provider="fake")), EchoChatModel)
    real = build_chat_model(Settings(llm_provider="openai", openai_api_key="sk-x"))
    assert isinstance(real, ChatOpenAI)  # constructing it makes no network call


# ---------------------------------------------------------------------------
# The fake: its reply is a function of what it saw
# ---------------------------------------------------------------------------
def test_echo_reports_first_and_latest_user_message():
    reply = echo_reply(
        [
            SystemMessage("be nice"),
            HumanMessage("my name is Kyle"),
            AIMessage("hello"),
            HumanMessage("what is my name?"),
        ]
    )
    assert "2 message(s)" in reply
    assert '"my name is Kyle"' in reply
    assert '"what is my name?"' in reply


async def test_echo_streams_chunks_that_join_back_to_the_full_reply():
    model = EchoChatModel()
    messages = [HumanMessage("hello there")]
    chunks = [chunk.text async for chunk in model.astream(messages)]
    assert len(chunks) > 1, "must stream token by token, not in one piece"
    assert "".join(chunks) == (await model.ainvoke(messages)).text


# ---------------------------------------------------------------------------
# Provider errors -> ErrorCode
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (openai.APITimeoutError(request=REQUEST), ErrorCode.TRANSIENT),
        (openai.APIConnectionError(request=REQUEST), ErrorCode.TRANSIENT),
        (status_error(openai.RateLimitError, 429), ErrorCode.TRANSIENT),
        (status_error(openai.InternalServerError, 500), ErrorCode.TRANSIENT),
        (status_error(openai.AuthenticationError, 401), ErrorCode.PERMANENT),
        (status_error(openai.NotFoundError, 404), ErrorCode.PERMANENT),
        (status_error(openai.BadRequestError, 400), ErrorCode.PERMANENT),
        (openai.OpenAIError("unclassified"), ErrorCode.UNKNOWN),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, BaseException) else str(v),
)
def test_provider_errors_map_to_error_codes(exc: BaseException, code: ErrorCode):
    assert classify_llm_error(exc) is code


def test_non_provider_errors_are_not_classified_here():
    assert classify_llm_error(ValueError("not ours")) is None
