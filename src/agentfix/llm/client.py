"""The real model client: HTTP to Ollama, and normalising what comes back.

The only file in the project that performs network I/O. It exists to satisfy the `LLMClient`
protocol in llm/types.py, and its whole job is to translate between the SDK's objects and
this project's small `LLMReply` — so that nothing downstream deals with HTTP, JSON, or
provider-specific shapes.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from agentfix.config import LLMConfig
from agentfix.llm.types import INVALID_ARGUMENTS, LLMReply, ToolCall


class OllamaClient:
    """OpenAI-compatible client. Works against Ollama, vLLM, or any /v1 endpoint."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()

        # The api_key argument is required: the SDK raises OpenAIError at construction if it
        # is absent and OPENAI_API_KEY is unset. Ollama has no authentication and ignores the
        # value entirely.
        #
        # It is a hardcoded placeholder rather than read from the environment, and that is
        # deliberate in both directions. Reading OPENAI_API_KEY would (a) fail confusingly for
        # the many students who do not have one, in a workshop that never contacts OpenAI, and
        # (b) put a real credential in an Authorization header sent to `base_url` — which is
        # env-configurable, so pointing MELLUM_BASE_URL elsewhere would leak the key there.
        # "agentfix" is not a secret; it is a protocol field the server discards.
        self._client = OpenAI(base_url=self.config.base_url, api_key="agentfix")

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LLMReply:
        """One turn: send the whole history, return a normalised reply."""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            # Ollama's /v1 endpoint DROPS this (measured: `ollama ps` still says 4096).
            # It is kept because vLLM and Ollama's native /api/chat both honour it; on
            # Ollama the context comes from the derived model instead — see Modelfile.
            "extra_body": {"options": {"num_ctx": self.config.num_ctx}},
        }
        # Only send the key when there is something to send: some servers reject
        # `tools: null`. Note this also skips an empty list, not just None.
        if tools:
            kwargs["tools"] = tools

        # `**kwargs` unpacks the dict into keyword arguments. Built as a dict first so the
        # optional `tools` key can be added conditionally above.
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]  # one choice, because n defaults to 1
        raw = choice.message

        # Parse the tool calls once, here, so the loop never touches JSON.
        calls: list[ToolCall] = []
        for call in raw.tool_calls or []:  # `or []` — the field is absent, not empty, if unused
            calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=_parse_arguments(call.function.arguments),
                )
            )

        return LLMReply(
            # `model_dump` converts the SDK's object to a plain dict, and this is passed back
            # to the API verbatim on later turns rather than being rebuilt — see LLMReply in
            # llm/types.py on why byte-stability matters. `exclude_none` drops null fields the
            # server never sent.
            message=raw.model_dump(exclude_none=True),
            tool_calls=tuple(calls),
            # Usage is optional in the protocol, so guard it rather than assuming it is there.
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    """Decode a tool call's JSON arguments, never raising.

    The model sends arguments as a JSON *string*, and a 12B model gets that string wrong often
    enough to matter — truncated, doubled braces, prose mixed in. Raising here would end the
    run over a recoverable mistake, so a failure is recorded under INVALID_ARGUMENTS and
    `ToolRegistry.dispatch` turns it into an observation telling the model to resend.

    The isinstance check catches the second failure mode: valid JSON that is not an object,
    such as a bare list or string, which cannot be unpacked into keyword arguments.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {INVALID_ARGUMENTS: raw}
    return parsed if isinstance(parsed, dict) else {INVALID_ARGUMENTS: raw}
