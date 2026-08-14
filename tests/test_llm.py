import pytest

from agentfix.config import LLMConfig
from agentfix.llm.fake import FakeLLMClient, assistant_text, assistant_tool_call
from agentfix.llm.types import ToolCall


def test_config_defaults_match_spec():
    config = LLMConfig()
    assert config.base_url == "http://localhost:11434/v1"
    assert config.model == "hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M"
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


@pytest.mark.llm
def test_live_model_answers_and_reports_usage():
    from agentfix.llm.client import OllamaClient

    reply = OllamaClient().chat([{"role": "user", "content": "Reply with exactly: ok"}])
    assert "ok" in (reply.message.get("content") or "").lower()
    assert reply.prompt_tokens > 0
