from __future__ import annotations

import json

from openai import OpenAI

from agentfix.config import LLMConfig
from agentfix.llm.types import LLMReply, ToolCall


class OllamaClient:
    """OpenAI-compatible client. Works against Ollama, vLLM, or any /v1 endpoint."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._client = OpenAI(base_url=self.config.base_url, api_key="agentfix")

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        kwargs: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "extra_body": {"options": {"num_ctx": self.config.num_ctx}},
        }
        if tools:
            kwargs["tools"] = tools

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        raw = choice.message

        calls: list[ToolCall] = []
        for call in raw.tool_calls or []:
            calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=_parse_arguments(call.function.arguments),
                )
            )

        return LLMReply(
            message=raw.model_dump(exclude_none=True),
            tool_calls=tuple(calls),
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )


def _parse_arguments(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
