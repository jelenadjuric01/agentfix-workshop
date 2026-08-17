"""The entire interface between the agent and a language model.

Three declarations, and nothing here talks to a network. Everything the loop knows about
"a model" is in this file: it hands over a list of messages, it gets an `LLMReply` back.
That is why the same loop can run against a real Ollama server (`llm/client.py`) or a
scripted list of canned replies (`llm/fake.py`) with no changes and no mocking library.

Read this file first — the rest of the codebase speaks the vocabulary defined here.
"""

from __future__ import annotations

# The `from __future__ import annotations` line above makes Python store every type
# annotation as a plain string instead of evaluating it at import time. Two practical
# effects: a class can refer to itself in its own annotations, and annotations cost
# nothing at runtime. It appears at the top of every module in this project.

from dataclasses import dataclass
from typing import Any, Protocol

# A model can emit malformed JSON for its tool arguments — at 12B, fairly often. When it
# does, `client.py` cannot recover the arguments the model meant, so it stores the raw
# string under this key instead. `ToolRegistry.dispatch` looks for exactly this key and
# tells the model its JSON was invalid, which is a fixable mistake, rather than "missing
# required argument: path", which would send it hunting for the wrong problem.
INVALID_ARGUMENTS = "__agentfix_invalid_json__"


@dataclass(frozen=True)
class ToolCall:
    """One request from the model: "run this tool with these arguments"."""

    # `@dataclass` generates __init__, __repr__ and __eq__ from the annotated fields below,
    # replacing about fifteen lines of boilerplate. The generated __eq__ compares field by
    # field, which is what lets tests assert on whole ToolCall values rather than on each
    # attribute separately.
    #
    # `frozen=True` makes instances read-only: assigning to a field after construction
    # raises FrozenInstanceError. Used throughout this project for anything that records
    # something which already happened — a request the model made, a result a tool
    # returned, a run that finished. Rewriting such a record later can only hide a bug.
    # (Frozen is shallow: it stops you replacing `arguments`, not mutating the dict it
    # points at. Use dataclasses.replace() to build a modified copy.)

    id: str  # opaque id from the model; the tool result MUST quote it back — see ToolOutcome
    name: str  # tool name, e.g. "read_file" — may name a tool that does not exist
    arguments: dict[str, Any]  # JSON-decoded, or {INVALID_ARGUMENTS: raw} if that failed


@dataclass(frozen=True)
class LLMReply:
    """One turn of model output, normalised so the loop never touches JSON or HTTP."""

    # This carries the same information in two shapes, on purpose.
    #
    # `message` is the raw assistant message in the exact wire format the API expects to
    # receive back. `run_agent` appends it to the history verbatim and never rebuilds it.
    # That is what keeps the history byte-stable across turns, which keeps the server's KV
    # cache valid for the shared prefix (see ARCHITECTURE.md, "Why history is strictly
    # append-only"). Rebuilding this dict from the parsed fields below would risk small
    # differences — key order, a dropped empty string — that invalidate the cache.
    message: dict[str, Any]

    # `tool_calls` is the same calls parsed for *our* use, so the loop iterates over typed
    # objects with their arguments already decoded. A tuple rather than a list because the
    # field lives on a frozen dataclass and a list would still be mutable from outside.
    # In `tuple[ToolCall, ...]` the `...` is required syntax meaning "any number of
    # elements, all ToolCall"; plain `tuple[ToolCall]` would mean "exactly one".
    tool_calls: tuple[ToolCall, ...] = ()

    # Per-turn usage, defaulting to 0 so a client that cannot report it still satisfies the
    # type. The loop sums these and separately tracks the peak: the total tells you cost,
    # the peak tells you how close a single prompt came to the context limit.
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient(Protocol):
    """Anything with a `chat` method of this shape can drive the agent.

    `Protocol` means *structural* typing: a class satisfies this by having a matching
    method, not by inheriting from it. Notice that neither `OllamaClient` nor
    `FakeLLMClient` imports or subclasses `LLMClient` — they match its shape, so they are
    accepted anywhere an `LLMClient` is required. (Contrast Java or C#, where a type must
    name the interface it implements. Go's interfaces work the way this does.)

    This is the mechanism behind the whole test suite: the tests exercise the real agent
    loop against a scripted fake client, with no model process anywhere. Substituting the
    model needs no inheritance, no registration, and no changes to `agent/loop.py`.
    """

    # The `...` body is not a stub waiting to be filled in — it is the conventional
    # Protocol body meaning "signature only, no implementation". The same three dots mean
    # something completely different inside `tuple[ToolCall, ...]` above.
    #
    # `tools` is optional in two separate ways: `| None` permits the value None, and
    # `= None` makes the argument omittable. Both are needed. The default is None rather
    # than [] because a mutable default is created once and shared by every call.
    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LLMReply: ...
