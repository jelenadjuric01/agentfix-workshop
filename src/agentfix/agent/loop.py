from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentfix.agent.trace import Tracer, TraceEvent
from agentfix.llm.types import LLMClient, LLMReply, ToolCall
from agentfix.tasks.loader import Task
from agentfix.tools.base import ToolRegistry
from agentfix.tools.tests_tool import RunTestsTool

MAX_STEPS = 10
MAX_GUARD_HITS = 3
NUDGE = "The tests have not passed. Read the latest failure and write a fix."


@dataclass(frozen=True)
class AgentResult:
    task_id: str
    solved: bool
    steps_used: int
    prompt_tokens: int
    completion_tokens: int
    duration_s: float
    trace: tuple[TraceEvent, ...]
    peak_prompt_tokens: int = 0


def system_prompt(registry: ToolRegistry) -> str:
    names = ", ".join(schema["function"]["name"] for schema in registry.schemas())
    return (
        "You are a Python bug-fixing agent working in a small project.\n"
        f"You have these tools: {names}.\n"
        "Work in this order: run the tests to see what fails, read the relevant file(s) "
        "before editing, then write the corrected file.\n"
        "Only read files the failure actually implicates — do not read every file "
        "list_files returns.\n"
        "When you call write_file you must supply the COMPLETE file contents, not a diff.\n"
        "Then run the tests again to confirm the fix worked — you are not finished until "
        "they pass. If they still fail, read the new failure and try again.\n"
        "Make the smallest change that fixes the failure. Do not rewrite unrelated code."
    )


def task_prompt(task: Task) -> str:
    return task.prompt


def is_done(run_tests: RunTestsTool) -> bool:
    # TODO(stage-3): the agent is done when the tests actually pass.
    # Not when the model stops calling tools. Not when it says "DONE".
    return False


def _call_signature(call: ToolCall) -> tuple[str, str]:
    return call.name, repr(sorted(call.arguments.items()))


def _guard_observation(name: str, hits: int) -> str:
    if hits == 1:
        return (
            f"You already called {name} with these exact arguments and got the result above. "
            "Try a different tool or different arguments."
        )
    return (
        f"You have now called {name} with identical arguments {hits + 1} times in a row and it "
        "was not executed. Call a different tool or use different arguments — read the file the "
        f"failure names, or call write_file with a fix. After {MAX_GUARD_HITS} repeats this run "
        "is abandoned."
    )


def _guarded(call: ToolCall, step: int, hits: int) -> tuple[dict[str, Any], TraceEvent]:
    """A repeated call gets an observation and a trace line instead of a re-execution."""
    return (
        {
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.name,
            "content": _guard_observation(call.name, hits),
        },
        TraceEvent(
            step=step,
            kind="tool",
            name=call.name,
            detail=f"guarded — identical call #{hits + 1} in a row",
            prompt_tokens=0,
            latency_s=0.0,
        ),
    )


def run_agent(
    task: Task,
    work_dir: Path,
    llm: LLMClient,
    registry: ToolRegistry,
    run_tests: RunTestsTool,
    max_steps: int = MAX_STEPS,
    tracer: Tracer | None = None,
) -> AgentResult:
    tracer = tracer or Tracer()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(registry)},
        {"role": "user", "content": task_prompt(task)},
    ]

    prompt_tokens = 0
    completion_tokens = 0
    peak_prompt_tokens = 0
    previous_signature: tuple[str, str] | None = None
    guard_hits = 0
    steps_used = 0
    started = time.time()

    for step in range(1, max_steps + 1):
        steps_used = step

        call_started = time.time()
        reply = llm.chat(messages, tools=registry.schemas())
        messages.append(reply.message)

        prompt_tokens += reply.prompt_tokens
        completion_tokens += reply.completion_tokens
        peak_prompt_tokens = max(peak_prompt_tokens, reply.prompt_tokens)
        tracer.record(
            TraceEvent(
                step=step,
                kind="llm",
                name="assistant",
                detail=_describe(reply),
                prompt_tokens=reply.prompt_tokens,
                latency_s=round(time.time() - call_started, 2),
            )
        )

        if reply.tool_calls:
            for call in reply.tool_calls:
                signature = _call_signature(call)
                if signature == previous_signature:
                    guard_hits += 1
                    message, event = _guarded(call, step, guard_hits)
                    messages.append(message)
                    tracer.record(event)
                    continue

                guard_hits = 0
                previous_signature = signature
                # TODO(stage-2): dispatch the call through the registry and append the
                # resulting tool message to `messages`. Keep the tool_call_id.
                raise NotImplementedError("stage 2: dispatch the tool call")

            if guard_hits >= MAX_GUARD_HITS:
                break
            continue

        if is_done(run_tests):
            break

        # A text-only reply is NOT a stop condition: the model may have given up, or claimed
        # a fix it never verified. Only passing tests end the run, so nudge and spend a step.
        messages.append({"role": "user", "content": NUDGE})

    return AgentResult(
        task_id=task.task_id,
        solved=is_done(run_tests),
        steps_used=steps_used,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        duration_s=round(time.time() - started, 2),
        trace=tuple(tracer.events),
        peak_prompt_tokens=peak_prompt_tokens,
    )


def _describe(reply: LLMReply) -> str:
    if reply.tool_calls:
        return "calls " + ", ".join(call.name for call in reply.tool_calls)
    return (reply.message.get("content") or "").strip()
