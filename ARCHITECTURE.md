# Architecture

No framework. Everything below is real code in this repo, not a simplification of it — the whole
point of the workshop is that a coding agent fits in something you can read end to end.

## The loop, annotated

`src/agentfix/agent/loop.py`, `run_agent`:

```python
def run_agent(task, work_dir, llm, registry, run_tests, max_steps=MAX_STEPS, tracer=None):
    tracer = tracer or Tracer()
    messages = [
        {"role": "system", "content": system_prompt(registry)},
        {"role": "user", "content": task_prompt(task)},
    ]
    # ^ the only two messages that exist before the model has done anything.

    previous_signature = None
    guard_hits = 0
    for step in range(1, max_steps + 1):
        # ^ bounded by construction: an agent with no cap is an unbounded wait and bill.
        # Raised to 10 from an original 6 after measurement — this tool granularity needs
        # run_tests + list_files + one read_file per implicated file + write_file + a
        # verifying run_tests, which is 8 steps for a three-file read.

        reply = llm.chat(messages, tools=registry.schemas())
        messages.append(reply.message)
        # ^ append-only. Never rewritten, reordered, or dropped. See "Why append-only" below.

        if reply.tool_calls:
            for call in reply.tool_calls:
                signature = _call_signature(call)
                if signature == previous_signature:
                    # ^ loop guard: same tool + same args twice in a row gets an observation
                    # instead of a re-execution, with wording that escalates on the second
                    # repeat. See "Loop guard" below.
                    guard_hits += 1
                    message, event = _guarded(call, step, guard_hits)
                    messages.append(message)
                    tracer.record(event)
                    continue
                guard_hits = 0
                previous_signature = signature
                messages.append(registry.dispatch(call).as_message())
                # ^ the only two methods that bridge model output to real effects:
                # `schemas()` above, `dispatch()` here.

            if guard_hits >= MAX_GUARD_HITS:
                break
                # ^ three identical calls in a row is a stuck model, not slow progress.
            continue
            # ^ tool calls never end the run on their own — always loop back for another turn.

        if is_done(run_tests):
            # ^ verification by execution, not by the model's say-so. The highest-value idea
            # in the whole design. `run_tests` is asked, not told: nothing hands the agent the
            # failing test output up front.
            break

        messages.append({"role": "user", "content": NUDGE})
        # ^ the model replied with prose and the tests are still red. That is not a stop
        # condition — it may have given up, or claimed a fix it never verified. So the loop
        # appends one plain nudge ("The tests have not passed. Read the latest failure and
        # write a fix.") and spends another step. Only `is_done` and `max_steps` end a run.

    return AgentResult(...)
```

`is_done` is therefore load-bearing: it is the *only* thing that can end a run early. Before
Stage 3 is implemented it returns `False`, so on a fresh `main` clone the agent runs its full
step budget and reports `NOT SOLVED` — which is the honest behaviour for an agent whose stop
condition has not been written yet.

## Module map

| File | Responsibility |
|---|---|
| `src/agentfix/config.py` | `LLMConfig`, env loading (`MELLUM_BASE_URL`, `MELLUM_MODEL`) |
| `src/agentfix/llm/types.py` | `ToolCall`, `LLMReply`, `LLMClient` Protocol |
| `src/agentfix/llm/client.py` | `OllamaClient` — OpenAI-compatible `chat()` |
| `src/agentfix/llm/fake.py` | `FakeLLMClient` + scripted-reply builders — offline, instant, deterministic |
| `src/agentfix/sandbox/base.py` | `ExecResult`, `ExecutionBackend` Protocol, `get_backend()` |
| `src/agentfix/sandbox/subprocess_backend.py` | hardened subprocess execution (default) |
| `src/agentfix/sandbox/docker_backend.py` | opt-in container execution |
| `src/agentfix/tools/base.py` | `ToolResult`, `ToolOutcome`, `Tool` Protocol, `ToolRegistry`, `truncate` |
| `src/agentfix/tools/fs.py` | `resolve_in_root`, `ListFilesTool`, `ReadFileTool`, `WriteFileTool` |
| `src/agentfix/tools/tests_tool.py` | `RunTestsTool` |
| `src/agentfix/tasks/loader.py` | `Task`, `load_task`, `workspace()` — the copy-to-tempdir context manager |
| `src/agentfix/agent/trace.py` | `TraceEvent`, `Tracer` |
| `src/agentfix/agent/loop.py` | `system_prompt`, `task_prompt`, `is_done`, `run_agent`, `AgentResult`, `MAX_STEPS` |
| `src/agentfix/runner.py` | `solve_task` — wires a `Task` into a fresh workspace, registry, and `run_agent` call |
| `src/agentfix/eval/runner.py` | `EvalReport`, `evaluate`, `run_suite` — pass@1 + steps + tokens + wall time |
| `src/agentfix/eval/humanevalfix.py` | `HumanEvalFixRow`, vendored-subset loader, task-dir generator |
| `src/agentfix/doctor.py` | preflight checks |
| `src/agentfix/cli.py` | argparse entry point: `doctor`, `solve`, `eval` |

Deviation from the original design doc: the spec names `sandbox/subprocess.py` and `tools/tests.py`;
the code uses `subprocess_backend.py` and `tests_tool.py` instead, to avoid shadowing the stdlib
`subprocess` module and colliding with pytest's `test_*` collection convention, respectively.

## Why history is strictly append-only

llama.cpp/Ollama reuse the KV cache for a shared message-history prefix. Measured prefill on this
machine is **~480 tok/s** — a 3,438-token prompt took ~7s before the first output token. Rewriting
or reordering any earlier message invalidates that cached prefix, so every later turn pays full
prefill again instead of only paying for what's new. In a multi-turn agent loop where the same
growing history is re-sent every turn, that difference compounds: append-only keeps each turn's
cost proportional to what changed, not to the whole conversation so far.

## Why `write_file` instead of `apply_diff`

Production agents typically edit via diffs. At 12B, small models reliably emit invalid unified
diffs — drifting line numbers, mismatched context — and burn turns failing to apply a patch rather
than fixing anything. A full-file rewrite of a small file is far more reliable at this model size.
The general lesson, not specific to this repo: tool ergonomics have to match model capability, and
a contract the model can't reliably satisfy is a leading cause of agents that "don't work" even
though the loop and the model are both fine in isolation.

## Why errors are observations, not exceptions

A tool never raises into the loop (`ToolRegistry.dispatch` catches broadly and turns failures into
a `ToolResult(ok=False, content=...)`). A missing file becomes `"No such file: src/typo.py.
Available: ..."`, fed back to the model as a normal tool result — and the model self-corrects on
the next turn. The natural instinct is to wrap the loop in `try/except` and treat a tool failure as
fatal; the fix that actually improves reliability is feeding failures back to the model as
information instead of killing the run over them.

## Two ways an agent passes without fixing anything

`run_tests` is the *only* oracle: `is_done` is `run_tests.last_result.passed` and nothing else. That
is the workshop's central idea, and it also means every way of corrupting that one signal is a way
to fake a fix. Both of these were reproducible in this repo:

1. **Rewriting the specification.** `write_file("tests/test_cart.py", "def test_ok(): assert True")`
   then `run_tests` → green → `solved=True`, with the original bug untouched. `write_file` had no
   path restriction, so deleting the tests was a valid "fix". It now refuses any path under
   `tests/` or named `test_*.py` and returns the observation *"the tests are the specification —
   fix the source instead"*, which is a useful thing for the model to read rather than a silent
   denial.
2. **Trusting a stale pass.** `write_file`(correct) → `run_tests`(green) → `write_file`(breaks it)
   → text reply ⇒ `solved=True` while the tests were red. `last_result` was never invalidated when
   the workspace changed underneath it. A successful `WriteFileTool.run` now clears
   `run_tests.last_result` through an `on_write` callback wired in `runner.py`, so verification is
   evidence about the *current* workspace or it is nothing.

What is still open, and worth saying out loud to students: nothing stops the agent from
special-casing the exact inputs the tests use (`if prices == [10.0, 5.0]: return 16.5`). Every
verification-based agent has this hole, and it is why real systems pair executed tests with held-out
tests and human review rather than treating a green suite as proof. `exercises/stage_3`'s README
poses the first half of this as an extension question.

## Two security boundaries — do not conflate them

1. **Tool-layer path confinement.** Every filesystem tool resolves the requested path against the
   task root and refuses anything that escapes it (`resolve_in_root` in `tools/fs.py`). This guards
   *access* — which files the model-driven tool calls are even allowed to name.
2. **Sandbox execution.** `run_tests` executes model-written code through an `ExecutionBackend`.
   This guards *execution* — what that code can do once it runs: `SubprocessBackend` applies
   `resource.setrlimit` (address space, CPU time, file size, process count), a stripped `env` with
   no inherited secrets, and output truncation at the source; `DockerBackend` adds `--network none`,
   memory/pid/cpu caps, and a non-root user, at the cost of requiring Docker.

These are separate layers on purpose: path confinement stops the model from *naming* a file outside
the project; the sandbox stops code *inside* the project from doing damage once it runs. Neither
substitutes for the other.

### `RLIMIT_AS` is Linux-only

`resource.setrlimit(resource.RLIMIT_AS, ...)` — the address-space cap — aborts CPython startup on
macOS/Apple Silicon; verified empirically on this machine (`_apply_limits` is called via
`preexec_fn`, and the child process fails to start at all when `RLIMIT_AS` is set on this
platform). On macOS, memory is effectively uncapped by `SubprocessBackend`; the CPU-time,
file-size, and process-count limits still apply, plus the wall-clock `timeout_s` on the
`subprocess.run` call itself. Real memory capping on macOS comes only from `DockerBackend`. This
matters less than it sounds: the Kaggle tier (tier 2) runs on Linux, so the memory cap holds
exactly where the weakest laptops in the room are running — the machines with no cap are, by
construction, the ones with 16 GB+ of their own RAM to begin with.

### Docker: the flags are tested, the container is not

`DockerBackend.build_argv` is a pure function, so **every isolation flag is asserted by tests that
need no daemon at all** — `--network none`, `--memory 512m`, `--pids-limit 128`, `--cpus 1`,
`--user runner`, `--cap-drop ALL`, `--security-opt no-new-privileges`, `--read-only`, the read-only
`:ro` mount, `--tmpfs /tmp`, and a unique `--name` per run. That matters because these assertions
used to sit *below* a module-level `pytestmark` skip, so on any machine without Docker (including
this one and CI) nothing checked that `--network none` was present at all. Deleting it would have
been green everywhere it mattered.

The mount is read-only on purpose: the filesystem tools write on the **host**, so the container
needs no write access to the workspace at all. `-p no:cacheprovider` and
`PYTHONDONTWRITEBYTECODE=1` keep pytest from trying, and `--tmpfs /tmp` with `HOME=/tmp` gives it
somewhere scratch to go.

On a `subprocess.TimeoutExpired` only the docker *client* is killed — the container keeps running —
so `run()` names the container and `docker kill`s it by name in the except branch.

**Still unverified: actual container execution.** The five tests that need a live daemon (pass,
fail, no network, unwritable workspace, timeout leaves nothing behind) skip with the reason
`agentfix-sandbox:latest not built (docker build -f Dockerfile.sandbox .)`. On the development
machine the daemon does now run, but its VM cannot reach the Docker registry, so `python:3.12-slim`
could not be pulled and the image could not be built. Build the image and run
`uv run pytest tests/test_docker_backend.py -v` on a machine with registry access before the
sandbox-safety demo depends on the runtime behaviour.

## HumanEvalFix runs as a plain script, not pytest

The workshop tasks (`tasks/workshop/*`) run via `pytest`. HumanEvalFix tasks do not: each vendored
row's `tests` field defines a `check(fn)` function and calls it at module level, so there is no
`test_*` function for pytest to collect — pytest would report "0 collected" and silently pass.
Separately, the dataset's `buggy_solution` is a function *body* only; it is not valid Python until
the matching `declaration` (imports + `def` line) is prepended, which `load_hf_rows` in
`eval/humanevalfix.py` does at vendor time. Both defects independently made pass@1 a structural
0.0 before they were fixed — not a model failure, a harness bug. This is exactly why `Task` carries
its own `test_command`: `tasks/workshop/*/task.json` says `["-m", "pytest", "-q"]`, while
`write_task_dir` generates `["-u", "test_candidate.py"]` for HumanEvalFix rows. Same `Task`
abstraction, same agent, same tools, two different ways of finding out whether the fix worked.

## Ollama's `/v1` endpoint ignores `options`, and that cost us

Plainly: **Ollama's OpenAI-compatible `/v1/chat/completions` silently drops the `options` block.**
`extra_body={"options": {"num_ctx": 16384}}` in `llm/client.py` therefore did nothing on Ollama,
and the same request against the native `/api/chat` sets the context correctly. Measured on ollama
0.32.4 by reading the `CONTEXT` column of `ollama ps` after one request:

| request path | `CONTEXT` |
|---|---|
| `/v1` + `extra_body` options (what this client sends) | 4096 |
| native `/api/chat` + `options.num_ctx` | 16384 |
| `/v1`, with `OLLAMA_CONTEXT_LENGTH` exported in the *client's* shell | 4096 |
| `/v1`, against a model derived with `PARAMETER num_ctx 16384` | **16384** |

What it cost: with `n_ctx=4096` and `max_tokens=1024` reserved for the reply, the usable prompt
window was about **3,072 tokens**, while peak single-call prompts on the longest HumanEvalFix runs
measured **2,875–3,111**. `--context-shift` drops the *earliest* messages first — the system prompt
that says "you are not finished until they pass." So on the longest runs the agent was plausibly
being told less than we thought it was being told, and the "gave up without acting" failure mode
that earlier notes attributed to model capability had a second candidate explanation the whole
time. That is the general lesson worth naming to students: an inference setting you *sent* is not a
setting the server *applied*, and the only way to know is to read it back.

Three consequences in this repo:

1. The shipped model is a **derived** one — `ollama create agentfix-mellum2 -f Modelfile` bakes
   `num_ctx 16384` in. Endpoint-agnostic, one command, identical on every platform.
2. `extra_body` is still sent, because vLLM and `/api/chat` honour it and it is free. It is not
   what makes Ollama work, and the code says so in a comment so nobody re-learns this.
3. `agentfix doctor` reads `/api/ps` and **FAILs** below 16384 with the exact remedy. An invisible
   risk became a preflight signal on the artifact students already run.

`AgentResult.peak_prompt_tokens` records the largest single prompt of a run and `EvalReport` carries
it into the eval JSON and the summary line, so the next person does not have to reconstruct the
number from a trace that `to_json` throws away.

## Loop guard

The guard compares only the *immediately previous* call's signature (tool name + sorted argument
items) and blocks one exact repeat with an observation instead of a re-execution. Originally that
was all it did, and it was not enough: a model calling `run_tests` five times in a row tripped the
guard on each repeat, collected an "already called" observation each time, and burned its whole
budget without trying anything different (measured — see `WORKSHOP.md`'s failure-mode trace 3).

So the guard now counts *consecutive* hits. The first hit gets the plain "you already called this"
observation; later hits get escalated wording naming what to do instead; at `MAX_GUARD_HITS = 3`
the run is abandoned. Guarded calls also record a trace event — without one, `--verbose` showed an
`llm` event calling a tool with no matching `tool` line underneath, which is confusing in exactly
the run a student is trying to read.

What the guard still is *not*: a cycle detector. `run_tests → list_files → run_tests → list_files`
never trips it, because no two consecutive signatures match. It remains a workaround for
small-model capability ("don't waste a turn re-reading the same file") with a bounded escape
hatch, not an architectural solution — say so if a student asks.

## What was deliberately left out

No planning phase, no reflection/self-critique, no parallel tool calls, no multi-agent
orchestration, no retry-at-different-temperature. These are all real techniques in production
agents; they are absent here because the workshop's budget is 90 minutes and three concepts (a
tool, dispatch, a stop condition), and each of these would add a decision point without adding to
that lesson. The natural next step, mentioned but not built, is the Mellum2 **Thinking** variant —
same code, one environment variable, visible `<think>` blocks — which is the smallest possible step
from this repo toward a model that plans before it acts.
