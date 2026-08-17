# agentfix

A teaching repository for a 90-minute workshop that shows developers new to agents how a coding
agent actually works, by having them build one. You write three pieces of a real agent yourself —
a tool and its JSON schema, the loop's tool dispatch, and a verification-based stop condition —
then watch it fix real bugs, locally, for $0, using [JetBrains
Mellum2](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF) served by Ollama. There
is no framework: the loop itself is about 15 lines and the rest of `run_agent` is tracing and
token accounting. See `ARCHITECTURE.md` for the annotated version.

Every exercise test runs against a scripted fake model, so the workshop does not depend on your
Ollama setup working — real inference is the reward, not a prerequisite.

## Which tier are you?

| Tier | Who | RAM | Endpoint |
|---|---|---|---|
| 1 (default) | 16 GB+ laptop | 16 GB+ | local Ollama, `http://localhost:11434/v1` |
| 2 | weak laptop, can't run an 8 GB model | any | Kaggle notebook — `notebooks/kaggle.ipynb` (**untested — see below**) |
| 3 | last resort / pair up | any | `qwen2.5-coder:1.5b` locally (~1 GB) |

Tiers 1 and 3 run on macOS, Linux, WSL2, and native Windows — the per-OS install commands are in
step 1 below, and the handful of differences that actually matter (sandbox limits, the RAM check) are
in [Platform notes](#platform-notes). Tier 2 needs only a browser, so it is the same on every OS.
Windows users: prefer WSL2.

## Tier 1 setup

Do this **before the workshop** — the model pull is 8 GB and venue wifi will not survive 20 people
doing it at once.

Only step 1 differs by operating system. Steps 2 and 3 are byte-for-byte identical everywhere,
which is the whole reason this repo derives the model with a `Modelfile` instead of setting a server
environment variable (see "The context window" below).

### Step 1 — install `uv` and Ollama

<details open>
<summary><b>macOS</b> (verified — all measurements in this README come from here)</summary>

```bash
brew install uv ollama
ollama serve &                  # or: open -a Ollama   (the app starts the same server)
```

Homebrew's `ollama` and the Ollama.app are the same server on `localhost:11434` — use either, but
not both at once. Without Homebrew: install `uv` with
`curl -LsSf https://astral.sh/uv/install.sh | sh` and Ollama from
[ollama.com/download](https://ollama.com/download).
</details>

<details>
<summary><b>Linux</b></summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://ollama.com/install.sh | sh
```

The install script registers a systemd service, so the server is already listening on
`localhost:11434`; `systemctl status ollama` tells you. If you installed the tarball by hand
instead, run `ollama serve` in its own terminal. A GPU is not required — CPU inference works, it is
just slower than the numbers below.
</details>

<details>
<summary><b>Windows — WSL2 (recommended)</b></summary>

Run the workshop inside WSL2 and treat it as Linux. This is the Windows path to prefer: the
sandbox that executes the agent's test runs is POSIX-shaped (see "Platform notes" below), so WSL2
avoids the one part of this repo that is untested on native Windows.

In PowerShell, once:

```powershell
wsl --install -d Ubuntu
```

Then, inside the Ubuntu shell, follow the **Linux** instructions above and do everything else —
`git clone`, `uv`, `ollama`, the exercises — inside WSL2. Keep the clone on the Linux filesystem
(`~/agentfix-workshop`, not `/mnt/c/...`); pytest across the `/mnt/c` bridge is slow enough to be
annoying.

WSL2 gets a fraction of your total RAM by default (50%, capped at 8 GB on older builds), and that
fraction — not your machine's spec sheet — is what has to hold an 8 GB model. If `free -g` inside
WSL2 shows less than 16 GB total, raise it in `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=16GB
```

then `wsl --shutdown` in PowerShell and reopen the shell.
</details>

<details>
<summary><b>Windows — native PowerShell (works for the exercises; sandbox untested)</b></summary>

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
winget install --id Ollama.Ollama      # or the installer from ollama.com/download
```

The installer runs Ollama as a background app, so the server is already on `localhost:11434` (look
for the tray icon). Then use the same commands as everywhere else — `uv run ...` is identical in
PowerShell, and forward slashes in task paths are fine.

Two caveats, both detailed under "Platform notes": `agentfix doctor` cannot read RAM on Windows and
skips that check rather than failing it, and the subprocess sandbox has not been run on native
Windows. If `uv run agentfix doctor` reports a `sandbox` failure, switch to WSL2 rather than
debugging it during the workshop.
</details>

### Step 2 — pull and derive the model

Identical on macOS, Linux, WSL2, and PowerShell:

```bash
ollama pull hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M
ollama create agentfix-mellum2 -f Modelfile      # required — see "The context window" below
```

### Step 3 — install the project and check it

```bash
uv sync --extra dev
uv run agentfix doctor
```

`agentfix doctor` checks your Python version and free RAM, that Ollama is installed, that its
server answers, that the derived model exists, that the **loaded context window is 16384**, does
one warmed-up timed generation, and runs one sandboxed test execution. It prints `[PASS]`/`[FAIL]`
per check and a final `READY <rate> tok/s`, or a remedy command for each failure. Send its output
to the instructor days ahead — that turns setup problems into a triage list instead of a
90-minute-clock emergency. Expected on a healthy 24 GB M4:

```
[PASS] python: 3.12.9
[PASS] ram: 24.0 GB total, 10.2 GB free
[PASS] ollama installed: /usr/local/bin/ollama
[PASS] ollama server: reachable at http://localhost:11434
[PASS] model present: agentfix-mellum2
[PASS] generation: 51 tok/s (372 tokens in 7.3s)
[PASS] context window: 16384 tokens
[PASS] sandbox: executes tests
```

Two lines read differently off macOS. On Linux the `ollama installed` path is usually
`/usr/local/bin/ollama` or `/usr/bin/ollama`; on native Windows it is the `ollama.exe` under
`%LOCALAPPDATA%\Programs\Ollama`. And the `ram` check only knows how to read `/proc/meminfo`
(Linux) and `sysctl`/`vm_stat` (macOS), so on native Windows it prints
`[PASS] ram: could not read memory on this platform — check by hand` — a deliberate pass, not a
measurement. Check it yourself there: you want 16 GB total and ~9 GB free before loading an 8 GB
model. Inside WSL2 the check works, and reports WSL2's allocation rather than the machine's.

### The context window — do not skip the `ollama create`

The agent re-sends its whole accumulated tool history every turn, and the longest measured
HumanEvalFix prompts reach ~3,100 tokens. Ollama's default is 4,096 with `--context-shift`, which
drops the *earliest* messages first — starting with the system prompt that tells the agent it is
not finished until the tests pass. Measured on ollama 0.32.4, reading the `CONTEXT` column of
`ollama ps`:

| how the context was requested | resulting `CONTEXT` |
|---|---|
| `/v1` + `extra_body={"options": {"num_ctx": 16384}}` | **4096** — silently ignored |
| `export OLLAMA_CONTEXT_LENGTH=16384` in your own shell, then a request | **4096** — wrong process |
| `ollama create agentfix-mellum2 -f Modelfile` (`PARAMETER num_ctx 16384`) | **16384** ✅ |

So the `Modelfile` route is the one this repo uses: one command, the same on every platform, and
it survives whichever endpoint the client talks to. The client still sends `num_ctx` in
`extra_body` because vLLM and Ollama's native `/api/chat` both honour it — but on Ollama's `/v1`
it does nothing, so do not rely on it.

`OLLAMA_CONTEXT_LENGTH` also works, but only if you get it into the **server process's**
environment, which is a different command on every platform — that is exactly why it is the
fallback and not the instruction:

| where the server runs | how to set it |
|---|---|
| any platform, server in your own terminal | `OLLAMA_CONTEXT_LENGTH=16384 ollama serve` |
| macOS, Ollama.app | `launchctl setenv OLLAMA_CONTEXT_LENGTH 16384`, then restart the app |
| Linux, systemd service | `systemctl edit ollama` → `[Service]` / `Environment="OLLAMA_CONTEXT_LENGTH=16384"`, then `systemctl daemon-reload && systemctl restart ollama` |
| Windows, background app | `setx OLLAMA_CONTEXT_LENGTH 16384`, then quit Ollama from the tray and reopen it |

The `ollama create` above makes all four rows unnecessary. Do that instead.

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

macOS, Linux, WSL2:

```bash
ollama pull qwen2.5-coder:1.5b
printf 'FROM qwen2.5-coder:1.5b\nPARAMETER num_ctx 16384\n' > /tmp/Modelfile.agentfix-qwen
ollama create agentfix-qwen -f /tmp/Modelfile.agentfix-qwen
MELLUM_MODEL=agentfix-qwen uv run agentfix doctor
MELLUM_MODEL=agentfix-qwen uv run agentfix solve tasks/workshop/01-shopcart --verbose
```

Native Windows PowerShell — same four steps, but there is no `/tmp` and no `VAR=value command`
prefix, so the Modelfile goes in the repo and the variable is set for the session:

```powershell
ollama pull qwen2.5-coder:1.5b
Set-Content Modelfile.agentfix-qwen @('FROM qwen2.5-coder:1.5b', 'PARAMETER num_ctx 16384')
ollama create agentfix-qwen -f Modelfile.agentfix-qwen
$env:MELLUM_MODEL = 'agentfix-qwen'
uv run agentfix doctor
uv run agentfix solve tasks/workshop/01-shopcart --verbose
```

`$env:MELLUM_MODEL` stays set for the rest of that PowerShell session — use
`Remove-Item Env:\MELLUM_MODEL` to go back to Mellum2. (In `cmd.exe` it is
`set MELLUM_MODEL=agentfix-qwen`.)

The `ollama create` is for the same reason as tier 1: without it the context is 4,096 and long
runs lose their own history. Skipping it makes `doctor`'s `context window` check fail.

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
uv run pytest                                                  # every test that needs no model
uv run pytest --all                                            # add the ones that need Ollama
uv run pytest -m llm                                           # only the ones that need Ollama
```

`solve` also accepts `--max-steps N` (default 10). `eval --suite` accepts `workshop` or
`humanevalfix`.

### Tests and the `llm` marker

One test in this repo talks to a real model; everything else runs offline against a scripted fake
(`src/agentfix/llm/fake.py`). That test carries `@pytest.mark.llm`, and `pyproject.toml` sets
`addopts = "-m 'not llm'"`, so a bare `uv run pytest` is always offline-safe — a broken Ollama
cannot fail your suite. `--all` opts back in, `-m llm` runs that test alone, and combining `--all`
with an explicit `-m` is a usage error rather than a silent override.

Every command above is identical on macOS, Linux, WSL2, and Windows PowerShell — forward slashes in
task paths work in PowerShell too. Only environment variables differ: `MELLUM_MODEL=x uv run ...`
inline on POSIX shells, `$env:MELLUM_MODEL = 'x'` on its own line first in PowerShell. The two
variables the CLI reads are `MELLUM_MODEL` (default `agentfix-mellum2`) and `MELLUM_BASE_URL`
(default `http://localhost:11434/v1`), plus `AGENTFIX_SANDBOX` (`subprocess` or `docker`).

Working through the exercises yourself? Start at `exercises/README.md` — it lays out the three
stages, which file you edit for each, and the `git checkout stage-N-solution` escape hatch if you
fall behind.

## Measured performance

Measured on an Apple M4, 24 GB, against a local Ollama running
`hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M`. Expect roughly 3-4x slower on an older
Intel laptop.

| Metric | Result |
|---|---|
| Generation throughput | 51 tok/s (372 tokens in 7.3s, as `agentfix doctor` reports it) |
| Prefill throughput | ~480 tok/s (a 3,438-token prompt took ~7s before the first output token) |
| Cold model load | ~3.5s, one-time |
| GGUF size on disk | 8.07 GB |
| Loaded context window | 16,384 tokens (`ollama ps`, via the derived model) |
| Workshop suite (`01`–`03`), pass@1 | 1.00 (3/3), 44.5s wall clock, peak prompt 1,456 tok |
| HumanEvalFix (20 vendored tasks), pass@1 | 0.60 (12/20), median 7 steps, max 10, 185,235 tokens, 8m09s wall clock, peak prompt 2,998 tok |

pass@1 on HumanEvalFix was **0.50 before** the loop's stop condition was made real. The old loop
ended a run on any text-only reply, so four failures stopped at 3–5 steps of 10 with the budget
unused; now a text-only reply while the tests are red gets a nudge and another step, and every one
of the 8 remaining failures uses all 10 steps. The context-window fix landed in the same
measurement, so the two effects are not separated — note that the largest single prompt was 2,998
tokens against the old 3,072-token usable window, i.e. the longest runs really were at the edge.
Nothing was tuned to move the number.

The wall-clock figure is exactly why that eval segment is demo-only in the workshop — it does
not fit in a 90-minute session as a live activity. `results/precomputed/` ships both runs so
students can discuss the numbers without waiting for them.

For contrast, the predecessor project's baseline: `Qwen/Qwen2.5-Coder-1.5B-Instruct` on a GPU, over
all 164 HumanEvalFix Python tasks, **pass@1 = 0.305** (50/164) with greedy decoding. That run was
**single-shot** — one patch per task, graded by running the tests, no loop and no tools, which is the
difference that matters here and not the parameter count. It also came with a 13-config decoding
sweep (temperature, top-p, beams, repetition penalty) spanning 0.262–0.317; with one run per config
and n=164, the binomial standard error is ±0.036, so that whole spread is noise. Only 31 of the 164
tasks changed verdict across all 13 configs. Decoding parameters bought almost nothing; adding a
loop and a test-execution oracle roughly doubled the fix rate.

Not an apples-to-apples comparison — different task count, different hardware, different harness —
but the mechanism is the point. The raw reports used to ship in `results/legacy/`; they were removed
from the repo, and `git log --diff-filter=D -- results/legacy` finds the commit that dropped them if
you want them back.

## Platform notes

What actually differs between operating systems, beyond the install commands:

| | macOS | Linux | WSL2 | native Windows |
|---|---|---|---|---|
| verified here | yes (M4, 24 GB) | no | no | no |
| `doctor` RAM check | `sysctl` + `vm_stat` | `/proc/meminfo` | `/proc/meminfo`, reports WSL2's slice | skipped with a PASS — check by hand |
| subprocess sandbox limits | CPU + file size + process count | CPU + file size + process count + 2 GB address space | same as Linux | none applied |
| Docker sandbox | Docker Desktop | Docker Engine | Docker Desktop w/ WSL2 backend, or Engine inside WSL2 | Docker Desktop |

Only the macOS column has been run end to end. Linux, WSL2, and native Windows are documented from
the code and the vendors' own install instructions, not from a run in this environment — if you hit
something the README gets wrong, tell the instructor so this table can be corrected.

The sandbox is the one place where the difference is structural rather than cosmetic.
`src/agentfix/sandbox/subprocess_backend.py` runs each test execution in a child process with a
stripped environment (`PATH=/usr/bin:/bin`) and POSIX `resource` limits applied through
`preexec_fn`. Both of those are POSIX-only: on native Windows the `resource` import fails and
`preexec_fn` is skipped, so the child gets the stripped environment but **no CPU, file-size, or
process caps** — a runaway test cannot be cut off by anything except the 10-second wall-clock
timeout. `RLIMIT_AS` (the 2 GB address-space cap) is deliberately Linux-only as well, because
applying it on Apple Silicon aborts interpreter startup. This is the reason WSL2 is the recommended
Windows path, and it is also why the `sandbox` line in `doctor` is worth reading rather than
skimming: it is the only signal that test execution works at all on your machine.

If you want the tighter isolation on any platform, build the image once and switch backends — the
commands are the same everywhere:

```bash
docker build -t agentfix-sandbox -f Dockerfile.sandbox .
AGENTFIX_SANDBOX=docker uv run agentfix doctor     # PowerShell: $env:AGENTFIX_SANDBOX = 'docker'
```

Verified on macOS: with the image built, `AGENTFIX_SANDBOX=docker uv run agentfix doctor` reports
`[PASS] sandbox: executes tests` and the container test file is 20 passed, no skips.

## Known limitations

- **The Docker sandbox needs one build step, and then it is verified.** Every isolation flag
  (`--network none`, the memory/pid/cpu caps, `--read-only`, `--cap-drop ALL`, …) is asserted by
  tests that run without a daemon. Five more actually start containers and check behaviour; they
  skip until you build the image:
  `docker build -t agentfix-sandbox -f Dockerfile.sandbox .` (note the trailing `.` — it is the
  build context). With the image built, `uv run pytest tests/test_docker_backend.py` is **20
  passed, no skips** — measured, including no-network, unwritable-workspace, and
  timeout-leaves-nothing-behind. The image's pytest is pinned to the version in `uv.lock` so both
  backends verify fixes identically (confirmed in-container: pytest 9.1.1);
  `tests/test_sandbox_image.py` enforces that pin without needing a daemon.
- **The Kaggle notebook has never been run on Kaggle.** `notebooks/kaggle.ipynb` is built from
  commands verified locally, but nobody has executed it end to end in a Kaggle container. Its
  cells now stop with a clear error rather than cloning a placeholder URL, and it clones the
  `solutions` branch so the demo cell works — but treat tier 2 as unverified.
- **Known failure class: the agent's only oracle is `run_tests`.** `write_file` refuses paths under
  `tests/` or named `test_*.py`, and a successful write invalidates the last test result, so
  neither rewriting the specification nor a stale green run can produce a false `SOLVED`. Both were
  reproducible before those guards; see `ARCHITECTURE.md` for what remains open (nothing stops the
  agent from special-casing the exact inputs the tests use).
