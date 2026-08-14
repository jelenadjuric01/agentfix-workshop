from __future__ import annotations

import json

from agentfix.llm.types import LLMReply, ToolCall


def assistant_text(text: str, prompt_tokens: int = 10) -> LLMReply:
    return LLMReply(
        message={"role": "assistant", "content": text},
        tool_calls=(),
        prompt_tokens=prompt_tokens,
        completion_tokens=len(text.split()),
    )


def assistant_tool_call(
    name: str, arguments: dict, call_id: str = "call_1", prompt_tokens: int = 10
) -> LLMReply:
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }
    return LLMReply(
        message=message,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
        prompt_tokens=prompt_tokens,
        completion_tokens=5,
    )


class FakeLLMClient:
    """Scripted client so the agent loop is testable with no model running."""

    def __init__(self, replies: list[LLMReply]) -> None:
        self._replies = list(replies)
        self._index = 0
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        assert self._index < len(self._replies), (
            f"FakeLLMClient script exhausted after {self._index} call(s); "
            "the agent asked for more turns than the test scripted"
        )
        self.calls.append(list(messages))
        reply = self._replies[self._index]
        self._index += 1
        return reply
