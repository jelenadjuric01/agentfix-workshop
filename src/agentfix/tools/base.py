from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from agentfix.llm.types import INVALID_ARGUMENTS, ToolCall

MAX_TOOL_OUTPUT_CHARS = 2000
MAX_FILE_READ_CHARS = 4000
TRUNCATION_MARKER = "\n[...truncated]"


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + TRUNCATION_MARKER


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str


@dataclass(frozen=True)
class ToolOutcome:
    call_id: str
    name: str
    result: ToolResult

    def as_message(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "name": self.name,
            "content": self.result.content,
        }


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]

    # `*args: Any, **kwargs: Any` is the one signature mypy treats as compatible with ANY
    # concrete signature. Each tool declares its own named parameters (`run(self)`,
    # `run(self, path)`, `run(self, path, content)`) and `dispatch` supplies them by keyword
    # from model-generated JSON, so the Protocol cannot pin them down. A `**kwargs`-only
    # declaration here looks equivalent but is not: it demands an implementation that accepts
    # arbitrary keywords, which none of the tools do.
    def run(self, *args: Any, **kwargs: Any) -> ToolResult: ...


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool]) -> None:
        self._tools: dict[str, Tool] = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def dispatch(self, call: ToolCall) -> ToolOutcome:
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(sorted(self._tools))
            return ToolOutcome(
                call.id,
                call.name,
                ToolResult(False, f"No such tool: {call.name}. Available tools: {available}"),
            )

        if INVALID_ARGUMENTS in call.arguments:
            return ToolOutcome(
                call.id,
                call.name,
                ToolResult(
                    False,
                    f"Your arguments for {call.name} were not valid JSON, so nothing ran. "
                    "Send the call again with a single well-formed JSON object.",
                ),
            )

        missing = [key for key in tool.parameters.get("required", []) if key not in call.arguments]
        if missing:
            return ToolOutcome(
                call.id,
                call.name,
                ToolResult(False, f"Missing required argument(s): {', '.join(missing)}"),
            )

        try:
            result = tool.run(**call.arguments)
        except Exception as error:  # a tool crash must not kill the run
            return ToolOutcome(
                call.id,
                call.name,
                ToolResult(False, f"Tool raised {type(error).__name__}: {error}"),
            )

        return ToolOutcome(call.id, call.name, result)
