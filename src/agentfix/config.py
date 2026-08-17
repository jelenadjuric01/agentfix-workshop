"""Settings: where the model is, which model, and how to sample from it."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

# Task fixtures and result files live in the repo, not in whatever directory the student
# happened to be standing in when they ran the CLI.
#
# `__file__` is this file; .parents[2] climbs src/agentfix/ -> src/ -> repo root. It is
# derived rather than hardcoded so the path is right however the CLI was invoked.
REPO_ROOT = Path(__file__).resolve().parents[2]

# The OpenAI-compatible endpoint Ollama serves. Everything talks OpenAI's protocol, which is
# why the same client works against vLLM or any other /v1 server.
DEFAULT_BASE_URL = "http://localhost:11434/v1"

# The model the agent talks to is the one derived by `ollama create -f Modelfile`, not
# the raw GGUF pull: only the derived model carries `num_ctx 16384`, because Ollama's
# /v1 endpoint drops per-request `options`. See Modelfile for the measurement.
BASE_MODEL = "hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M"
DEFAULT_MODEL = "agentfix-mellum2"

# `agentfix doctor` fails if the loaded model reports less than this. A too-small context
# does not error — it silently truncates the middle of the agent's history, which looks like
# a stupid model rather than a misconfigured one. That failure is very hard to diagnose from
# the symptoms, hence a preflight check.
MIN_CONTEXT_LENGTH = 16384


@dataclass(frozen=True)
class LLMConfig:
    """How to reach the model, and how to sample from it. Frozen: read-only after creation."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL

    # Sampling settings. Unlike everything else in this file these are NOT the result of a
    # measurement on this project — they are conventional defaults for this class of model.
    # The predecessor project's 13-config decoding sweep (temperature, top-p, beams,
    # repetition penalty) moved pass@1 by less than its own standard error, while fixing the
    # loop's stop condition moved it 0.50 -> 0.60. So temperature is not where the leverage
    # is here. The trade-off if you do change it: 0.0 makes the published eval numbers
    # reproducible, while a non-zero value gives a stuck model a way out of repeating itself.
    temperature: float = 0.6
    top_p: float = 0.95

    # Cap on ONE reply. Relevant because write_file must emit a complete file: this is the
    # ceiling on how large a file the agent can rewrite in a single turn.
    max_tokens: int = 1024

    # Sent with every request, but Ollama's /v1 endpoint ignores it — see llm/client.py.
    num_ctx: int = 16384

    @property
    def native_api_url(self) -> str:
        """Ollama's own API root. `/v1` cannot report the loaded context length; `/api/ps` can."""
        return self.base_url.rsplit("/v1", 1)[0].rstrip("/")

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Defaults, overridden by MELLUM_BASE_URL and MELLUM_MODEL if they are set.

        `dataclasses.replace(cls(), ...)` builds a *new* instance with those two fields
        changed — the standard way to "modify" a frozen dataclass, since assignment is
        blocked. Only these two fields are env-configurable today; temperature, max_tokens
        and the API key are not, so experimenting with them means editing code.
        """
        return replace(
            cls(),
            base_url=os.environ.get("MELLUM_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("MELLUM_MODEL", DEFAULT_MODEL),
        )
