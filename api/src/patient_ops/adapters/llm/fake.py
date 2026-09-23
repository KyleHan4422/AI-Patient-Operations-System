"""A deterministic, offline stand-in for a real chat model.

Used by the test suite, by CI (which has no API key) and by anyone running the
demo without one. Settings refuse LLM_PROVIDER=fake when APP_ENV=prod.

It has two modes, because the graph asks two different things of a model.

  no tools bound   It echoes: the reply is a function of what it was shown --
                   how many user messages, the first one and the latest one.
                   That is the point. The property Phase 2 has to prove is
                   "the model saw turn one", and a real model's wording cannot
                   be asserted on, while this reply can. It streams word by
                   word, because LangGraph's `messages` stream mode can only
                   forward tokens from a model that implements streaming.

  tools bound      It plays the front desk, by keyword: it routes an intent,
                   picks a lookup tool, and turns what the tool returned into
                   a FinalAnswer. Enough to run the whole Phase 4 graph
                   offline -- routing, tool choice, the record templates, the
                   grounding check, the KB_GAP -- without a key.

What the stand-in cannot do is the one thing a language model is here for:
judge whether a passage on the right topic actually answers the question. It
quotes the best passage it was given. So an evaluation run against this model
measures the plumbing, never the judgement, and the report says so.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool


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


# ---------------------------------------------------------------------------
# Playing the front desk, by keyword
# ---------------------------------------------------------------------------
# Booking needs a first-person request, not the word "appointment": "how much
# notice do I need to give to cancel an appointment" is a policy question about
# cancellations, and routing it to the booking branch would answer nothing.
_BOOKING = re.compile(
    r"\b(?:book|schedule|rebook|reschedule)\s+(?:me|my|an|a|the)\b"
    r"|\b(?:cancel|move|change)\s+(?:my|the)\s+appointment\b"
    r"|\bwhat times\b|\bany openings\b|\bwhen can i come in\b"
    r"|\b(?:anything|something)\s+free\b|\bfree (?:on|this|next)\b",
    re.I,
)
_SMALLTALK = re.compile(
    r"^\W*(?:hi|hello|hey|good (?:morning|afternoon|evening)|thanks|thank you|"
    r"bye|goodbye|cheers|ok|okay)\b\W*$",
    re.I,
)
# Narrow on purpose. "Can I be added to a waitlist if a slot opens up" and "is
# there an after-hours number" both contain an opening-hours word and neither
# is a question about opening hours; a keyword list wide enough to catch every
# phrasing answers those from the schedule table, confidently and wrongly.
_HOURS = re.compile(
    r"\b(?:are|is) (?:you|the clinic) open\b"
    r"|\bopen (?:on|at|until|till|late|early|today|tomorrow|saturday|sunday|weekends?)\b"
    r"|\b(?:opening|office|clinic|your) hours\b|\bwhat hours\b"
    r"|\bwhat time do you (?:open|close)\b|\bclosed (?:on|for|between)\b",
    re.I,
)
_INSURANCE = re.compile(
    r"\b(?:insurance|insurer|ppo|hmo|dppo|in.network|cover|covers|covered|coverage)\b", re.I
)
_PRICE = re.compile(r"\b(?:price|cost|costs|how much|charge|charges|fee|quote)\b", re.I)
# The plan name is whatever follows "take"/"accept"/"with": up to four words,
# then the trailing filler is trimmed. A patient writes "do you take Aetna
# insurance?" and the tables hold "Aetna".
_PLAN_AFTER = re.compile(
    r"\b(?:take|takes|taking|accept|accepts|cover|covers|with|have|use|using)\s+"
    r"(?P<name>[A-Za-z][\w'&/-]*(?:\s+[A-Za-z][\w'&/-]*){0,3})",
    re.I,
)
_PLAN_FILLER = frozenset(
    "my our your the a an this that insurance insurances plan plans coverage cover "
    "card dental if for and or to at on in as is are do does you we".split()
)
# The agent's prompt ends with "Treatments on file: CROWN (Crown), ...". The
# stand-in reads its own prompt rather than being given a hardcoded list --
# seeded data changes, and a stand-in that lies about the catalogue is worse
# than no stand-in.
_CATALOGUE = re.compile(r"([A-Z][A-Z0-9_]{2,})\s+\(([^)]+)\)")


def _last_human(messages: Sequence[BaseMessage]) -> str:
    return next((m.text for m in reversed(messages) if isinstance(m, HumanMessage)), "")


def classify_intent(question: str) -> str:
    if _BOOKING.search(question):
        return "booking"
    if _SMALLTALK.match(question.strip()):
        return "smalltalk"
    return "knowledge"


def _plan_name(question: str) -> str | None:
    """The plan a patient named, or None if they named no plan.

    "Do you take Aetna?" carries no insurance word at all, so the only signal
    is the proper noun after "take". "Do you take card payments?" has the same
    shape and is not a plan -- the capital letter is what separates them, and
    getting it wrong would answer a payment question with "I have no plan
    called card payments on file".
    """
    match = _PLAN_AFTER.search(question)
    words = match.group("name").split() if match else []
    while words and words[0].lower() in _PLAN_FILLER:
        words.pop(0)
    while words and words[-1].lower() in _PLAN_FILLER:
        words.pop()
    # None, not the words: without a capital letter this is not a plan name,
    # and the caller falls through to the documents. Returning it anyway would
    # answer "does my insurance cover cleanings" with "I have no plan called
    # cleanings on file".
    return " ".join(words) if any(word[0].isupper() for word in words) else None


def _procedure_code(question: str, catalogue: Sequence[tuple[str, str]]) -> str | None:
    asked = question.lower()
    for code, name in catalogue:
        if name.lower() in asked or code.lower() in asked:
            return code
    return None


def _catalogue(messages: Sequence[BaseMessage]) -> list[tuple[str, str]]:
    prompt = next((m.text for m in messages if isinstance(m, SystemMessage)), "")
    listed = prompt.partition("Treatments on file:")[2]
    return _CATALOGUE.findall(listed)


def _pick_lookup(
    question: str, catalogue: Sequence[tuple[str, str]], available: Sequence[str]
) -> tuple[str, dict[str, Any]]:
    """Which tool a front-desk assistant would reach for.

    Order matters: a price question that names no treatment on file ("how much
    is the missed appointment fee") is not a price lookup, it is a question
    about a policy, and the documents are where it belongs.
    """
    if _HOURS.search(question) and "get_opening_hours" in available:
        return "get_opening_hours", {}
    if "lookup_insurance_plan" in available:
        named = _plan_name(question)
        if named and (_INSURANCE.search(question) or named[0].isupper()):
            return "lookup_insurance_plan", {"plan_name": named}
    if _PRICE.search(question) and "lookup_price" in available:
        code = _procedure_code(question, catalogue)
        if code is not None:
            return "lookup_price", {"procedure": code}
    return "search_documents", {"query": question}


def _first_sentences(text: str, count: int = 2) -> str:
    """The opening of a passage, verbatim.

    Verbatim matters: the grounding check requires every figure in an answer to
    appear in a cited passage, and a stand-in that paraphrased would fail its
    own system's rules.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:count]).strip()


def _final_answer(evidence: str) -> dict[str, Any]:
    if "[chunk " in evidence:
        # The first passage only, and its own text only. Answering from one
        # passage while citing another is exactly what the grounding check
        # refuses, and a stand-in that did it would fail its own system.
        label, _source, body = evidence.split("\n\n")[0].split("\n", 2)
        chunk_id = int(re.search(r"\[chunk (\d+)]", label).group(1))
        return {"sufficient": True, "answer": _first_sentences(body), "citations": [chunk_id]}
    if evidence.startswith("No passage"):
        return {"sufficient": False, "answer": "", "citations": []}
    # A record lookup: the reply is written from the row by graph/replies.py,
    # so the answer is deliberately left empty.
    return {"sufficient": True, "answer": "", "citations": []}


def play_front_desk(messages: list[BaseMessage], available: Sequence[str]) -> AIMessage:
    question = _last_human(messages)
    call_id = f"call_{sum(isinstance(m, AIMessage) for m in messages)}"

    if len(available) == 1 and available[0] not in {"FinalAnswer", "search_documents"}:
        # A single bound schema is with_structured_output(), not an agent loop.
        # Today that is only the intent classifier.
        return AIMessage(
            content="",
            tool_calls=[
                {"name": available[0], "args": {"intent": classify_intent(question)}, "id": call_id}
            ],
        )

    last = messages[-1] if messages else None
    if isinstance(last, ToolMessage):
        return AIMessage(
            content="",
            tool_calls=[{"name": "FinalAnswer", "args": _final_answer(last.text), "id": call_id}],
        )

    name, args = _pick_lookup(question, _catalogue(messages), available)
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


# ---------------------------------------------------------------------------
class EchoChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "echo-fake"

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Any:
        # Converted to the OpenAI shape, as a provider would: the stand-in then
        # sees exactly the schema a real model is sent, including the tool
        # descriptions, so a badly described tool fails here too.
        return self.bind(
            tools=[convert_to_openai_tool(tool) for tool in tools],
            tool_choice=tool_choice,
            **kwargs,
        )

    def _should_stream(self, *, async_api: bool, run_manager: Any = None, **kwargs: Any) -> bool:
        # Tool calls are answered in one piece. Streaming them would mean
        # emitting tool_call_chunks, which is provider plumbing this stand-in
        # has no reason to imitate -- and no node streams a tool-calling model.
        if kwargs.get("tools"):
            return False
        return super()._should_stream(async_api=async_api, run_manager=run_manager, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.get("tools")
        if tools:
            names = [tool["function"]["name"] for tool in tools]
            message: BaseMessage = play_front_desk(messages, names)
        else:
            message = AIMessage(content=echo_reply(messages))
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, **kwargs)  # no IO, so no executor hop

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
