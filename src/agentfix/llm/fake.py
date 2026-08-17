"""A scripted stand-in for the model, so the loop can be tested with nothing running.

This is why the test suite is fast, offline and deterministic: instead of mocking the loop's
internals, you hand it a *list of replies* and let the real `run_agent` execute against real
tools in a real temp directory. Nothing is patched — `FakeLLMClient` simply satisfies the
`LLMClient` protocol (see llm/types.py), so the loop cannot tell the difference.

A test reads like a screenplay of a model's turns:

    llm = FakeLLMClient([
        assistant_tool_call("run_tests", {}),
        assistant_tool_call("read_file", {"path": "shopcart/cart.py"}),
        assistant_tool_call("write_file", {"path": "shopcart/cart.py", "content": FIXED}),
        assistant_tool_call("run_tests", {}),
        assistant_text("Fixed the tax rounding."),
    ])

The tests really are red before that write and really are green after it, because the fake
replaces only the model — never the tools, the sandbox, or the loop.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from agentfix.llm.types import LLMReply, ToolCall


def assistant_text(text: str, prompt_tokens: int = 10) -> LLMReply:
    """A prose reply with no tool calls — the model talking rather than acting."""
    return LLMReply(
        message={"role": "assistant", "content": text},
        tool_calls=(),
        prompt_tokens=prompt_tokens,
        # A stand-in for a token count. Nothing asserts on the value; it just has to be
        # plausible so the accounting in run_agent has something to add up.
        completion_tokens=len(text.split()),
    )


def assistant_tool_call(
    name: str, arguments: dict[str, Any], call_id: str = "call_1", prompt_tokens: int = 10
) -> LLMReply:
    """A reply requesting one tool call, in the same two shapes a real reply carries.

    `message` mirrors the wire format exactly — including `arguments` as a JSON *string*,
    which is how the API represents them — while `tool_calls` holds the parsed version. Being
    faithful here is what makes the tests meaningful: the loop appends the raw message to the
    history, so if this shape were wrong the tests would pass against a fiction.
    """
    return assistant_tool_calls(
        [(name, arguments)], call_ids=(call_id,), prompt_tokens=prompt_tokens
    )


def assistant_tool_calls(
    calls: Sequence[tuple[str, dict[str, Any]]],
    call_ids: Sequence[str] | None = None,
    prompt_tokens: int = 10,
) -> LLMReply:
    """A reply requesting SEVERAL tool calls in one turn, which a real model is allowed to do.

    The API permits any number of calls per assistant message, and requires an answer to
    every one of them: skip a `tool_call_id` and the next request is rejected. The loop
    therefore iterates over `reply.tool_calls` — and without this builder there was no way to
    exercise that iteration with more than one element, so a loop answering only the first
    call would have passed every test in the suite.

        assistant_tool_calls([("run_tests", {}), ("list_files", {})], call_ids=("c1", "c2"))

    `call_ids` defaults to call_1..call_N. Ids must be distinct, since the whole point of an
    id is to tell two calls apart.
    """
    if call_ids is None:
        call_ids = [f"call_{index}" for index in range(1, len(calls) + 1)]
    ids = tuple(call_ids)
    assert len(ids) == len(calls), "one call id per call"
    assert len(set(ids)) == len(ids), "call ids must be distinct"

    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            for call_id, (name, arguments) in zip(ids, calls, strict=True)
        ],
    }
    return LLMReply(
        message=message,
        tool_calls=tuple(
            ToolCall(id=call_id, name=name, arguments=arguments)
            for call_id, (name, arguments) in zip(ids, calls, strict=True)
        ),
        prompt_tokens=prompt_tokens,
        completion_tokens=5 * len(calls),
    )


class FakeLLMClient:
    """Scripted client so the agent loop is testable with no model running."""

    def __init__(self, replies: list[LLMReply]) -> None:
        # Copied with `list(...)` so a caller reusing its list cannot mutate this script.
        self._replies = list(replies)
        self._index = 0
        # Every history this client was called with, for tests that assert on what the loop
        # actually sent — that it is append-only, that the tool_call_id came back, and so on.
        self.calls: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,  # noqa: ARG002
    ) -> LLMReply:
        # `tools` is ignored on purpose: the replies are scripted, so nothing here inspects the
        # schemas. The parameter stays because it is part of the LLMClient protocol.
        #
        # An assert, not an IndexError, because the message is the diagnosis: if the loop asks
        # for more turns than the test scripted, the test's model of the loop is wrong. This
        # fires whenever a change makes the loop take an extra step.
        assert self._index < len(self._replies), (
            f"FakeLLMClient script exhausted after {self._index} call(s); "
            "the agent asked for more turns than the test scripted"
        )
        # Snapshot the history at this turn, since the loop keeps appending to the same list.
        self.calls.append(list(messages))
        reply = self._replies[self._index]
        self._index += 1
        return reply
