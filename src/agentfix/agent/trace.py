from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    step: int
    kind: str
    name: str
    detail: str
    prompt_tokens: int
    latency_s: float


class Tracer:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.events: list[TraceEvent] = []

    def record(self, event: TraceEvent) -> None:
        self.events.append(event)
        if self.verbose:
            marker = "→" if event.kind == "llm" else "←"
            detail = event.detail.replace("\n", " ")[:100]
            print(
                f"  step {event.step} {marker} {event.kind}:{event.name}  "
                f"[ctx {event.prompt_tokens} tok, {event.latency_s:.1f}s]  {detail}"
            )

    def as_json(self) -> list[dict[str, Any]]:
        return [asdict(event) for event in self.events]
