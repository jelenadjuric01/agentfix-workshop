from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

# Task fixtures and result files live in the repo, not in whatever directory the student
# happened to be standing in when they ran the CLI.
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASE_URL = "http://localhost:11434/v1"

# The model the agent talks to is the one derived by `ollama create -f Modelfile`, not
# the raw GGUF pull: only the derived model carries `num_ctx 16384`, because Ollama's
# /v1 endpoint drops per-request `options`. See Modelfile for the measurement.
BASE_MODEL = "hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M"
DEFAULT_MODEL = "agentfix-mellum2"
MIN_CONTEXT_LENGTH = 16384


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 1024
    num_ctx: int = 16384

    @property
    def native_api_url(self) -> str:
        """Ollama's own API root. `/v1` cannot report the loaded context length; `/api/ps` can."""
        return self.base_url.rsplit("/v1", 1)[0].rstrip("/")

    @classmethod
    def from_env(cls) -> LLMConfig:
        return replace(
            cls(),
            base_url=os.environ.get("MELLUM_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("MELLUM_MODEL", DEFAULT_MODEL),
        )
