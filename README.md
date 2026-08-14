# agentfix

A teaching repository for a 90-minute workshop that shows developers new to agents how a coding
agent actually works, by having them build one. You write three pieces of a real agent yourself —
a tool and its JSON schema, the loop's tool dispatch, and a verification-based stop condition —
then watch it fix real bugs, locally, for $0, using [JetBrains
Mellum2](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF) served by Ollama. There
is no framework: the loop, the tool registry, and the message history are ~15 lines you can read
end to end. See `ARCHITECTURE.md` for the annotated version.

Every exercise test runs against a scripted fake model, so the workshop does not depend on your
Ollama setup working — real inference is the reward, not a prerequisite.

## Which tier are you?

| Tier | Who | RAM | Endpoint |
|---|---|---|---|
| 1 (default) | 16 GB+ laptop | 16 GB+ | local Ollama, `http://localhost:11434/v1` |
| 2 | weak laptop, can't run an 8 GB model | any | Kaggle notebook — `notebooks/kaggle.ipynb` (**untested — see below**) |
| 3 | last resort / pair up | any | `qwen2.5-coder:1.5b` locally (~1 GB) |

## Tier 1 setup

Do this **before the workshop** — the model pull is 8 GB and venue wifi will not survive 20 people
doing it at once.

```bash
brew install uv ollama          # or your platform's uv/ollama install
ollama pull hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M
uv sync --extra dev
export OLLAMA_CONTEXT_LENGTH=16384   # belt-and-braces — see ARCHITECTURE.md
uv run agentfix doctor
```

`agentfix doctor` checks your Python version, that Ollama is installed and reachable, that the
model is pulled, does one real generation, and runs one sandboxed test execution. It prints
`[PASS]`/`[FAIL]` per check and a final `READY <rate> tok/s` or a remedy for each failure. Send its
output to the instructor days ahead — that turns setup problems into a triage list instead of a
90-minute-clock emergency.

## Tier 2: Kaggle (untested)

If your laptop cannot run an 8 GB model, use `notebooks/kaggle.ipynb`. It runs Ollama, the model,
and this repo entirely *inside* the Kaggle container — there is no tunnel back to your laptop, and
no code changes; `MELLUM_BASE_URL` stays at its default because inside that container,
`localhost:11434` *is* local.

Requirements: a phone-verified Kaggle account (needed to enable internet access in notebook
settings — do this ahead of time) and a GPU accelerator selected for the notebook.

**This path has not been run on Kaggle from this environment — there is no Kaggle access here.**
It is built from the same commands verified for tier 1, but treat it as unverified until someone
runs it end to end. If Ollama does not come up under Kaggle, fall back to tier 3.

## Tier 3: the 1 GB fallback

```bash
ollama pull qwen2.5-coder:1.5b
MELLUM_MODEL=qwen2.5-coder:1.5b uv run agentfix doctor
MELLUM_MODEL=qwen2.5-coder:1.5b uv run agentfix solve tasks/workshop/01-shopcart --verbose
```

Smaller and faster, but noticeably less reliable at multi-step tool use than Mellum2 — expect it
to need more steps, or to fail tasks Mellum2 solves. Good enough to see the loop work; not the
demo model.

## Command reference

```bash
uv run agentfix doctor                                       # preflight check
uv run agentfix solve tasks/workshop/01-shopcart --verbose    # run the agent on one task
uv run agentfix eval --suite workshop --limit 3               # run the agent over a suite
uv run agentfix eval --suite humanevalfix --limit 3
uv run pytest exercises/stage_1                                # or stage_2, stage_3
```

`solve` also accepts `--max-steps N` (default 10). `eval --suite` accepts `workshop` or
`humanevalfix`.

Working through the exercises yourself? Start at `exercises/README.md` — it lays out the three
stages, which file you edit for each, and the `git checkout stage-N-solution` escape hatch if you
fall behind.

## Measured performance

Measured on an Apple M4, 24 GB, against a local Ollama running
`hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M`. Expect roughly 3-4x slower on an older
Intel laptop.

| Metric | Result |
|---|---|
| Generation throughput | 51 tok/s (700 tokens in 13.7s) |
| Prefill throughput | ~480 tok/s (a 3,438-token prompt took ~7s before the first output token) |
| Cold model load | ~3.5s, one-time |
| GGUF size on disk | 8.07 GB |
| Workshop suite (`01`–`03`), pass@1 | 1.00 (3/3), ~39.7s wall clock |
| HumanEvalFix (20 vendored tasks), pass@1 | 0.50 (10/20), median 7 steps, max 10, 167,729 tokens, 8m38s wall clock |

The HumanEvalFix number is exactly why that eval segment is demo-only in the workshop — it does
not fit in a 90-minute session as a live activity. `results/precomputed/` ships both runs so
students can discuss the numbers without waiting for them. `results/legacy/` holds the predecessor
project's Qwen2.5-Coder-1.5B baseline on a GPU-served 164-task HumanEvalFix run (pass@1 ≈ 0.30) —
not an apples-to-apples comparison (different task count, different hardware, different harness),
but a useful contrast: a 12B model with tool access roughly doubles a smaller model's fix rate on
its own predecessor's benchmark.

## Known limitations

- **Docker sandbox is unverified here.** `AGENTFIX_SANDBOX=docker` and `DockerBackend` exist and
  their tests are written, but the Docker daemon is not running on this development machine, so
  `tests/test_docker_backend.py` skips. Exercise it with a real daemon before relying on it for
  the safety demo.
- **`num_ctx` is sent but not confirmed honoured.** `LLMConfig.num_ctx = 16384` is passed to Ollama
  via `extra_body`, but this Ollama instance was observed serving the model with `-c 4096` and
  `--context-shift`. No run so far has needed more than ~9k tokens *across* the whole run, so
  nothing has broken in practice, but if you see a long run behave as though it forgot the early
  tool calls, this is the first thing to check. Setting `OLLAMA_CONTEXT_LENGTH=16384` before
  starting Ollama is the belt-and-braces fix and is included in the tier-1 setup above.
