from agentfix.llm.types import INVALID_ARGUMENTS, ToolCall
from agentfix.tools.base import ToolRegistry, ToolResult, truncate


class EchoTool:
    name = "echo"
    description = "Echo a message back."
    parameters = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    def run(self, message: str) -> ToolResult:
        return ToolResult(ok=True, content=f"echo: {message}")


def test_truncate_appends_marker_only_when_needed():
    assert truncate("short", 100) == "short"
    long_text = truncate("y" * 500, 100)
    assert long_text.startswith("y" * 100)
    assert "[...truncated]" in long_text


def test_schemas_are_openai_shaped():
    schemas = ToolRegistry([EchoTool()]).schemas()
    assert schemas == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo a message back.",
                "parameters": EchoTool.parameters,
            },
        }
    ]


def test_dispatch_returns_tool_message_with_call_id():
    registry = ToolRegistry([EchoTool()])
    outcome = registry.dispatch(ToolCall(id="call_7", name="echo", arguments={"message": "hi"}))

    assert outcome.result.ok is True
    assert outcome.as_message() == {
        "role": "tool",
        "tool_call_id": "call_7",
        "name": "echo",
        "content": "echo: hi",
    }


def test_unknown_tool_becomes_an_observation_not_an_exception():
    registry = ToolRegistry([EchoTool()])
    outcome = registry.dispatch(ToolCall(id="c1", name="nope", arguments={}))

    assert outcome.result.ok is False
    assert "nope" in outcome.result.content
    assert "echo" in outcome.result.content


def test_missing_required_argument_becomes_an_observation():
    registry = ToolRegistry([EchoTool()])
    outcome = registry.dispatch(ToolCall(id="c1", name="echo", arguments={}))

    assert outcome.result.ok is False
    assert "message" in outcome.result.content


def test_unexpected_argument_becomes_an_observation():
    registry = ToolRegistry([EchoTool()])
    outcome = registry.dispatch(
        ToolCall(id="c1", name="echo", arguments={"message": "hi", "extra": "nope"})
    )

    assert outcome.result.ok is False
    assert "extra" in outcome.result.content


def test_tool_exception_becomes_an_observation():
    class BoomTool:
        name = "boom"
        description = "Always explodes."
        parameters = {"type": "object", "properties": {}}

        def run(self) -> ToolResult:
            raise RuntimeError("kaboom")

    outcome = ToolRegistry([BoomTool()]).dispatch(ToolCall(id="c1", name="boom", arguments={}))
    assert outcome.result.ok is False
    assert "kaboom" in outcome.result.content


def test_malformed_json_arguments_say_so_instead_of_naming_a_missing_argument():
    """ "Missing required argument(s): message" is a misleading thing to self-correct from."""
    registry = ToolRegistry([EchoTool()])
    outcome = registry.dispatch(
        ToolCall(id="c1", name="echo", arguments={INVALID_ARGUMENTS: '{"message": '})
    )

    assert outcome.result.ok is False
    assert "not valid JSON" in outcome.result.content
    assert "Missing required" not in outcome.result.content
