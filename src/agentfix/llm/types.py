from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

# Key under which an unparseable tool-call argument string is carried through to
# `ToolRegistry.dispatch`, so the model is told its JSON was malformed rather than
# that some argument it thought it sent was missing.
INVALID_ARGUMENTS = "__agentfix_invalid_json__"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMReply:
    message: dict[str, Any]
    tool_calls: tuple[ToolCall, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient(Protocol):
    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LLMReply: ...
