import pytest

from agentfix.config import BASE_MODEL, LLMConfig
from agentfix.llm.client import OllamaClient
from agentfix.llm.fake import (
    FakeLLMClient,
    assistant_text,
    assistant_tool_call,
    assistant_tool_calls,
)
from agentfix.llm.types import INVALID_ARGUMENTS, ToolCall


class _StubFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _StubToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = _StubFunction(name, arguments)


class _StubMessage:
    def __init__(self, content: str | None, tool_calls: list | None, dump: dict) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self._dump = dump

    def model_dump(self, exclude_none: bool = True) -> dict:
        return self._dump


class _StubChoice:
    def __init__(self, message: _StubMessage) -> None:
        self.message = message


class _StubUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _StubResponse:
    def __init__(self, message: _StubMessage, usage: _StubUsage | None) -> None:
        self.choices = [_StubChoice(message)]
        self.usage = usage


def _make_stub_create(response: _StubResponse, captured_kwargs: dict):
    def _create(**kwargs):
        captured_kwargs.update(kwargs)
        return response

    return _create


def test_config_defaults_match_spec():
    config = LLMConfig()
    assert config.base_url == "http://localhost:11434/v1"
    assert config.model == "agentfix-mellum2"
    assert BASE_MODEL == "hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M"
    assert config.temperature == 0.6
    assert config.top_p == 0.95
    assert config.max_tokens == 1024
    assert config.num_ctx == 16384


def test_config_reads_env(monkeypatch):
    monkeypatch.setenv("MELLUM_BASE_URL", "http://gpu-box:8000/v1")
    monkeypatch.setenv("MELLUM_MODEL", "some/other-model")
    config = LLMConfig.from_env()
    assert config.base_url == "http://gpu-box:8000/v1"
    assert config.model == "some/other-model"


def test_assistant_text_builds_reply_with_no_tool_calls():
    reply = assistant_text("all done")
    assert reply.tool_calls == ()
    assert reply.message == {"role": "assistant", "content": "all done"}


def test_assistant_tool_call_builds_openai_shaped_message():
    reply = assistant_tool_call("run_tests", {}, call_id="abc")
    assert reply.tool_calls == (ToolCall(id="abc", name="run_tests", arguments={}),)
    assert reply.message["tool_calls"][0]["function"]["name"] == "run_tests"
    assert reply.message["tool_calls"][0]["id"] == "abc"


def test_assistant_tool_calls_builds_one_message_carrying_several_calls():
    reply = assistant_tool_calls([("run_tests", {}), ("read_file", {"path": "a.py"})])

    assert [call.name for call in reply.tool_calls] == ["run_tests", "read_file"]
    assert [call.id for call in reply.tool_calls] == ["call_1", "call_2"]
    # The wire format holds both calls in a single assistant message, arguments as JSON text.
    wire = reply.message["tool_calls"]
    assert len(wire) == 2
    assert wire[1]["function"]["arguments"] == '{"path": "a.py"}'


def test_assistant_tool_calls_rejects_mismatched_or_duplicate_ids():
    with pytest.raises(AssertionError, match="one call id per call"):
        assistant_tool_calls([("run_tests", {}), ("list_files", {})], call_ids=("only_one",))
    with pytest.raises(AssertionError, match="distinct"):
        assistant_tool_calls([("run_tests", {}), ("list_files", {})], call_ids=("same", "same"))


def test_fake_client_returns_scripted_replies_in_order_and_records_calls():
    client = FakeLLMClient([assistant_tool_call("run_tests", {}), assistant_text("fixed")])

    first = client.chat([{"role": "user", "content": "go"}])
    second = client.chat([{"role": "user", "content": "go"}, {"role": "tool", "content": "fail"}])

    assert first.tool_calls[0].name == "run_tests"
    assert second.message["content"] == "fixed"
    assert len(client.calls) == 2
    assert len(client.calls[1]) == 2


def test_fake_client_raises_when_script_is_exhausted():
    client = FakeLLMClient([assistant_text("only one")])
    client.chat([])
    with pytest.raises(AssertionError, match="exhausted"):
        client.chat([])


def test_ollama_client_extracts_tool_call():
    message = _StubMessage(
        content="",
        tool_calls=[_StubToolCall("call_1", "run_tests", '{"path": "src/"}')],
        dump={"role": "assistant", "content": ""},
    )
    response = _StubResponse(message, usage=_StubUsage(7, 3))
    captured: dict = {}
    client = OllamaClient(LLMConfig())
    client._client.chat.completions.create = _make_stub_create(response, captured)

    reply = client.chat([{"role": "user", "content": "go"}])

    assert reply.tool_calls == (
        ToolCall(id="call_1", name="run_tests", arguments={"path": "src/"}),
    )


def test_ollama_client_plain_text_reply_has_no_tool_calls():
    message = _StubMessage(
        content="all done",
        tool_calls=None,
        dump={"role": "assistant", "content": "all done"},
    )
    response = _StubResponse(message, usage=_StubUsage(4, 2))
    captured: dict = {}
    client = OllamaClient(LLMConfig())
    client._client.chat.completions.create = _make_stub_create(response, captured)

    reply = client.chat([{"role": "user", "content": "go"}])

    assert reply.tool_calls == ()


def test_ollama_client_passes_num_ctx_and_conditionally_includes_tools():
    message = _StubMessage(
        content="ok", tool_calls=None, dump={"role": "assistant", "content": "ok"}
    )
    response = _StubResponse(message, usage=_StubUsage(1, 1))
    captured: dict = {}
    client = OllamaClient(LLMConfig())
    client._client.chat.completions.create = _make_stub_create(response, captured)

    client.chat([{"role": "user", "content": "go"}])
    assert captured["extra_body"] == {"options": {"num_ctx": 16384}}
    assert "tools" not in captured

    captured.clear()
    tools = [{"type": "function", "function": {"name": "run_tests"}}]
    client.chat([{"role": "user", "content": "go"}], tools=tools)
    assert captured["extra_body"] == {"options": {"num_ctx": 16384}}
    assert captured["tools"] == tools


def test_ollama_client_handles_missing_usage():
    message = _StubMessage(
        content="ok", tool_calls=None, dump={"role": "assistant", "content": "ok"}
    )
    response = _StubResponse(message, usage=None)
    captured: dict = {}
    client = OllamaClient(LLMConfig())
    client._client.chat.completions.create = _make_stub_create(response, captured)

    reply = client.chat([{"role": "user", "content": "go"}])

    assert reply.prompt_tokens == 0
    assert reply.completion_tokens == 0


MALFORMED = "{not json"
NOT_AN_OBJECT = "[1, 2, 3]"


def test_ollama_client_marks_malformed_tool_call_arguments():
    message = _StubMessage(
        content="",
        tool_calls=[_StubToolCall("call_1", "run_tests", MALFORMED)],
        dump={"role": "assistant", "content": ""},
    )
    response = _StubResponse(message, usage=_StubUsage(1, 1))
    captured: dict = {}
    client = OllamaClient(LLMConfig())
    client._client.chat.completions.create = _make_stub_create(response, captured)

    reply = client.chat([{"role": "user", "content": "go"}])

    assert reply.tool_calls[0].arguments == {INVALID_ARGUMENTS: MALFORMED}


def test_ollama_client_marks_non_object_tool_call_arguments():
    message = _StubMessage(
        content="",
        tool_calls=[_StubToolCall("call_1", "run_tests", NOT_AN_OBJECT)],
        dump={"role": "assistant", "content": ""},
    )
    response = _StubResponse(message, usage=_StubUsage(1, 1))
    captured: dict = {}
    client = OllamaClient(LLMConfig())
    client._client.chat.completions.create = _make_stub_create(response, captured)

    reply = client.chat([{"role": "user", "content": "go"}])

    assert reply.tool_calls[0].arguments == {INVALID_ARGUMENTS: NOT_AN_OBJECT}


@pytest.mark.llm
def test_live_model_answers_and_reports_usage():
    reply = OllamaClient().chat([{"role": "user", "content": "Reply with exactly: ok"}])
    assert "ok" in (reply.message.get("content") or "").lower()
    assert reply.prompt_tokens > 0
