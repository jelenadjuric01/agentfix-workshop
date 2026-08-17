"""The agent itself. If you read one file in this project, read this one.

There is no framework here. An agent is a `for` loop that alternates between asking a model
what to do and doing it, with three things that decide whether it works at all:

1. a bounded number of steps         — an agent with no cap is an unbounded wait and bill
2. a stop condition based on reality — `is_done`, which runs the tests rather than asking
                                       the model whether it is finished
3. a loop guard                      — because a stuck model repeats itself forever

Everything else — the tools, the sandbox, the eval harness — is support for those three.
ARCHITECTURE.md walks through this same function with the design decisions annotated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from agentfix.agent.trace import Tracer, TraceEvent
from agentfix.llm.types import LLMClient, LLMReply, ToolCall
from agentfix.tasks.loader import Task
from agentfix.tools.base import ToolRegistry
from agentfix.tools.tests_tool import RunTestsTool

# Raised to 10 from an original 6 after measurement: this tool granularity needs run_tests +
# list_files + one read_file per implicated file + write_file + a verifying run_tests, which
# is already 8 steps for a three-file read.
MAX_STEPS = 10

# Three identical calls in a row is a stuck model, not slow progress.
MAX_GUARD_HITS = 3

# Sent when the model replies with prose while the tests are still red. See the end of the
# loop for why a text-only reply is not allowed to end the run.
NUDGE = "The tests have not passed. Read the latest failure and write a fix."


@dataclass(frozen=True)
class AgentResult:
    """Everything one run produced: the verdict plus what it cost to reach it.

    Cost is reported, not hidden. `peak_prompt_tokens` is separate from the running total
    because they answer different questions: the total is the bill, the peak is how close a
    single prompt came to the context window.
    """

    task_id: str
    solved: bool
    steps_used: int
    prompt_tokens: int
    completion_tokens: int
    duration_s: float
    trace: tuple[TraceEvent, ...]
    peak_prompt_tokens: int = 0


def system_prompt(registry: ToolRegistry) -> str:
    """The standing instructions, rebuilt from whatever tools are actually registered.

    The tool names are derived from `registry.schemas()` rather than hardcoded, so adding a
    tool updates the prompt and the schema together and they cannot drift apart.

    Most of this text is the result of watching the model fail: it read every file in the
    project, it emitted diffs instead of whole files, it declared victory without re-running
    the tests. Each instruction below is a countermeasure to an observed failure.
    """
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
    """The task's own prompt, verbatim. A seam: the loop never invents task text."""
    return task.prompt


def is_done(run_tests: RunTestsTool) -> bool:  # noqa: ARG001
    """The only thing that can end a run early — so it decides what "solved" means.

    Two failure modes a correct implementation has to rule out: a model that claims a fix it
    never made, and a model that passed the tests once and then wrote a file that broke them
    again. (`RunTestsTool.invalidate` handles the second by clearing the stored result on
    every write, so a stale green result cannot be reused.)

    Until this is implemented it returns False, so the agent runs its full step budget and
    reports NOT SOLVED — the honest behaviour for an agent whose stop condition is missing.
    """
    # The noqa above silences "unused argument" only because this stub ignores `run_tests`.
    # Delete it once your implementation actually reads the parameter.
    # TODO(stage-3): the agent is done when the tests actually pass.
    # Not when the model stops calling tools. Not when it says "DONE".
    return False


def _call_signature(call: ToolCall) -> tuple[str, str]:
    """An identity for "the same call again": tool name plus its arguments.

    `sorted(...)` first so that {"a": 1, "b": 2} and {"b": 2, "a": 1} compare equal — key
    order in the model's JSON is not meaningful. `repr` gives a hashable, comparable string.
    """
    return call.name, repr(sorted(call.arguments.items()))


def _guard_observation(name: str, hits: int) -> str:
    """The text sent back for a repeated call, escalating on the second repeat."""
    if hits == 1:
        return (
            f"You already called {name} with these exact arguments and got the result above. "
            "Try a different tool or different arguments."
        )
    # Second repeat: name the consequence. Being explicit that the run will be abandoned
    # measurably helps a small model break out of the pattern.
    return (
        f"You have now called {name} with identical arguments {hits + 1} times in a row and it "
        "was not executed. Call a different tool or use different arguments — read the file the "
        f"failure names, or call write_file with a fix. After {MAX_GUARD_HITS} repeats this run "
        "is abandoned."
    )


def _guarded(call: ToolCall, step: int, hits: int) -> tuple[dict[str, Any], TraceEvent]:
    """A repeated call gets an observation and a trace line instead of a re-execution.

    Returns the pair the caller needs: the message to append to the history, and the trace
    event to record. Note the message still carries the `tool_call_id` — the API requires an
    answer to every call the model made, even one we refused to run.
    """
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
    llm: LLMClient,
    registry: ToolRegistry,
    run_tests: RunTestsTool,
    max_steps: int = MAX_STEPS,
    tracer: Tracer | None = None,
) -> AgentResult:
    """Run the agent until the tests pass, the step budget runs out, or it gets stuck.

    Note there is no `work_dir` parameter: every tool was constructed with the workspace
    already bound (see runner.py), so the loop itself never touches a path. It does not even
    import pathlib.

    `run_tests` is passed separately as well as being inside `registry` — the loop needs to
    ask it directly, via `is_done`, rather than through the model.
    """
    # `tracer or Tracer()` rather than a default argument of `Tracer()`: a mutable default is
    # evaluated once at function definition and would be shared by every call.
    tracer = tracer or Tracer()

    # The only two messages that exist before the model has done anything.
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

    # Bounded by construction. `is_done` can end this early; nothing else can.
    for step in range(1, max_steps + 1):
        steps_used = step

        call_started = time.time()

        # The whole growing history is re-sent every turn — models are stateless. The tool
        # schemas go with it, because the model has no memory of them either.
        reply = llm.chat(messages, tools=registry.schemas())

        # Append-only. This message is never rewritten, reordered or dropped. That keeps the
        # prefix byte-stable so the server's KV cache stays valid, which is the difference
        # between each turn costing what changed and each turn re-paying for the whole
        # conversation (measured prefill here: ~480 tok/s).
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
            # A turn may contain several calls; each gets its own observation appended.
            for call in reply.tool_calls:
                signature = _call_signature(call)

                # Loop guard. A model that repeats a call verbatim learned nothing from the
                # result, so re-running it would burn a step for the same output. Send an
                # observation instead and let it try something else.
                if signature == previous_signature:
                    guard_hits += 1
                    message, event = _guarded(call, step, guard_hits)
                    messages.append(message)
                    tracer.record(event)
                    continue

                # Progress: reset the counter and remember this call as the new baseline.
                guard_hits = 0
                previous_signature = signature
                # This is the one place where model output becomes a real effect. Two
                # constraints worth knowing before you write it: `registry.dispatch` never
                # raises (see tools/base.py — every failure comes back as an observation), and
                # the API requires every call the model made to be answered by a message
                # carrying its `tool_call_id`.
                # TODO(stage-2): dispatch the call through the registry and append the
                # resulting tool message to `messages`. Keep the tool_call_id.
                raise NotImplementedError("stage 2: dispatch the tool call")

            if guard_hits >= MAX_GUARD_HITS:
                break

            # Tool calls never end the run on their own — always loop back for another turn,
            # so the model can read the results it just received. Note this skips the
            # `is_done` check below on purpose: the check belongs on a turn where the model
            # had nothing more to do.
            continue

        # No tool calls, so the model replied with prose. This is the only place the run can
        # end successfully — and it ends because the tests pass, not because the model
        # stopped calling tools.
        if is_done(run_tests):
            break

        # A text-only reply is NOT a stop condition: the model may have given up, or claimed
        # a fix it never verified. Only passing tests end the run, so nudge and spend a step.
        messages.append({"role": "user", "content": NUDGE})

    return AgentResult(
        task_id=task.task_id,
        # Re-checked here rather than trusting how the loop exited: breaking out on
        # MAX_GUARD_HITS or running out of steps must not be mistaken for success.
        solved=is_done(run_tests),
        steps_used=steps_used,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        duration_s=round(time.time() - started, 2),
        trace=tuple(tracer.events),
        peak_prompt_tokens=peak_prompt_tokens,
    )


def _describe(reply: LLMReply) -> str:
    """One line summarising a model turn, for the trace."""
    if reply.tool_calls:
        return "calls " + ", ".join(call.name for call in reply.tool_calls)
    # `or ""` because a message with no text has content=None, and .strip() would fail on it.
    return (reply.message.get("content") or "").strip()
