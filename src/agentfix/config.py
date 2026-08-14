from __future__ import annotations

import os
from dataclasses import dataclass, replace

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 1024
    num_ctx: int = 16384

    @classmethod
    def from_env(cls) -> LLMConfig:
        return replace(
            cls(),
            base_url=os.environ.get("MELLUM_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("MELLUM_MODEL", DEFAULT_MODEL),
        )
