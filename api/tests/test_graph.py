"""The conversation graph, end to end, with no database and no network.

An in-memory checkpointer, an in-memory transcript and the echo model: what is
tested here is the graph's own behaviour -- memory across turns, what the
model is shown, what gets saved, and who gets to speak.
"""

from __future__ import annotations

import uuid

import openai
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from patient_ops.adapters.llm.fake import EchoChatModel
from patient_ops.graph.build import build_graph
from patient_ops.graph.context import GraphContext
from patient_ops.graph.nodes.draft_reply import SYSTEM_PROMPT
from patient_ops.graph.nodes.respond import EMPTY_DRAFT_REPLY
from patient_ops.graph.turn import Final, Token, run_turn, turn_config
from tests.fakes import ListRecorder, RecordingEcho, SilentModel, TimingOutModel


class Chat:
    """One conversation thread over one compiled graph."""

    def __init__(self, model: BaseChatModel | None = None, history_limit: int = 20) -> None:
        self.graph = build_graph(InMemorySaver())
        self.thread_id = uuid.uuid4()
        self.model = model or EchoChatModel()
        self.recorder = ListRecorder()
        self.history_limit = history_limit

    def context(self, model: BaseChatModel | None = None) -> GraphContext:
        return GraphContext(
            thread_id=self.thread_id,
            request_id="req-test",
            chat_model=model or self.model,
            recorder=self.recorder,
            history_limit=self.history_limit,
        )

    async def say(self, text: str, model: BaseChatModel | None = None) -> tuple[list[str], str]:
        tokens, final = [], None
        async for event in run_turn(self.graph, text=text, context=self.context(model)):
            if isinstance(event, Token):
                tokens.append(event.text)
            else:
                assert isinstance(event, Final)
                final = event.text
        assert final is not None
        return tokens, final

    async def state(self) -> dict:
        return (await self.graph.aget_state(turn_config(self.thread_id))).values


async def test_later_turns_see_earlier_ones():
    chat = Chat()
    await chat.say("My name is Kyle")
    await chat.say("I have a toothache")
    _, reply = await chat.say("What did I say first?")
    assert "3 message(s)" in reply
    assert '"My name is Kyle"' in reply


async def test_threads_do_not_share_memory():
    chat = Chat()
    await chat.say("My name is Kyle")
    other = Chat()
    other.graph = chat.graph  # same graph and checkpointer, different thread
    _, reply = await other.say("hello")
    assert "Kyle" not in reply


async def test_streamed_tokens_are_exactly_the_final_reply():
    """Only draft_reply's tokens are streamed -- respond's message is not echoed again."""
    tokens, final = await Chat().say("hello there")
    assert len(tokens) > 1
    assert "".join(tokens) == final


async def test_system_prompt_is_shown_but_never_saved():
    model = RecordingEcho()
    chat = Chat(model)
    await chat.say("hi")
    await chat.say("again")

    for shown in model.seen:
        assert isinstance(shown[0], SystemMessage) and shown[0].text == SYSTEM_PROMPT
    saved = (await chat.state())["messages"]
    assert not any(isinstance(m, SystemMessage) for m in saved)
    assert [type(m) for m in saved] == [HumanMessage, AIMessage, HumanMessage, AIMessage]


async def test_model_sees_a_bounded_window_but_everything_is_saved():
    model = RecordingEcho()
    chat = Chat(model, history_limit=4)
    for n in range(1, 5):
        _, reply = await chat.say(f"message {n}")

    shown = model.seen[-1][1:]  # without the system prompt
    assert len(shown) <= 4
    assert isinstance(shown[0], HumanMessage), "the window must start on a user turn"
    assert '"message 3"' in reply, "turn 1 and 2 fell out of the window"
    assert len((await chat.state())["messages"]) == 8, "the checkpoint keeps every message"


async def test_each_turn_is_recorded_once_with_what_was_actually_said():
    chat = Chat()
    _, first = await chat.say("hello")
    _, second = await chat.say("bye")
    assert [(t.user_text, t.assistant_text) for t in chat.recorder.turns] == [
        ("hello", first),
        ("bye", second),
    ]
    assert all(t.thread_id == chat.thread_id for t in chat.recorder.turns)


async def test_a_failed_turn_does_not_inherit_the_previous_reply():
    """Per-turn fields are reset by the input, so a failure leaves them empty.

    Without the reset, the checkpoint would still hold turn 1's reply while
    turn 2 failed -- the same family of bug as confirming a stale slot.
    """
    chat = Chat()
    await chat.say("first")
    with pytest.raises(openai.APITimeoutError):
        await chat.say("second", model=TimingOutModel())

    state = await chat.state()
    assert state["final_response"] is None
    assert state["draft"] is None
    assert len(chat.recorder.turns) == 1, "a failed turn is not written to the transcript"
    _, reply = await chat.say("third")  # the thread carries on
    assert '"third"' in reply


async def test_respond_has_the_last_word_on_an_empty_draft():
    chat = Chat(SilentModel())
    tokens, final = await chat.say("hello")
    assert tokens == []
    assert final == EMPTY_DRAFT_REPLY
    assert chat.recorder.turns[0].assistant_text == EMPTY_DRAFT_REPLY
