# Instructor Runsheet

90 minutes, developers new to agents. Read `README.md`'s tier table and `ARCHITECTURE.md` before
the session; this document is what to run and say, minute by minute.

## Pre-workshop checklist

- [ ] Pre-work email sent (at least a week ahead): pull the 8 GB model, **run
      `ollama create agentfix-mellum2 -f Modelfile`**, run `agentfix doctor`, send the output;
      phone-verify a Kaggle account if planning tier 2; bring a laptop charger.
- [ ] `doctor` outputs collected from attendees and triaged — anyone still failing gets a remedy
      or a tier-3/pair-up plan before the room opens. **`ram` and `context window` are the two
      that decide someone's tier**; a `context window: 4096` line means they skipped the
      `ollama create` and their agent will lose its own system prompt on long runs.
- [ ] USB sticks prepared with the model GGUF, for anyone whose wifi can't do 8 GB on the day.
- [ ] Endpoint decision made per student: tier 1 (local), tier 2 (Kaggle), or tier 3 (1.5B model) —
      `doctor`'s `ram` line now says this for you, so collect it rather than asking people.
- [ ] `git checkout solutions` tested end to end on the instructor machine (see 0:08 below).
- [ ] Docker demo machine has a running daemon **and** a built image, verified same-day:
      `docker info` then `docker build -t agentfix-sandbox -f Dockerfile.sandbox .`. Neither has
      ever succeeded on the machine this repo was written on — the daemon is down there now, and
      before that its VM could not reach the registry — so treat the whole demo as unverified until
      you have run `uv run pytest tests/test_docker_backend.py -v` and seen the five container tests
      pass instead of skip. The isolation-flag assertions and `tests/test_sandbox_image.py` run with
      no daemon at all, so a skip means "unproven at runtime", not "broken".

## Minute-by-minute

| Time | Segment | Command | Say |
|---|---|---|---|
| 0:00–0:08 | Setup triage; what an agent is | — | Loop, tools, verification. An agent is a while-loop around a chat model that can call functions and sees the result. Nothing more magical than that. |
| 0:08–0:16 | Live demo: finished agent fixes task 01 | `git checkout solutions && uv run agentfix solve tasks/workshop/01-shopcart --verbose` | **Run this from `solutions`, not `main`.** `main` is deliberately stubbed for the exercises — `agentfix solve` there prints a legible error and exits 1 by design (verified: `BadRequestError` on the empty tool schema, caught by the CLI, printed to stderr with a pointer to `exercises/README.md`). Running the demo on `main` live would look like a broken demo instead of the intended student starting point. |
| 0:16–0:24 | The tool-calling contract | walk `src/agentfix/tools/base.py` | schema → the model emits a `tool_call` → `ToolRegistry.dispatch` → a `role="tool"` message with the `tool_call_id` → appended to history. Two methods (`schemas`, `dispatch`) are the whole bridge between model output and real effects. |
| 0:24–0:36 | **Stage 1** — `run_tests` + its schema | `git checkout main` (back to the stub); `uv run pytest exercises/stage_1 -v` | Students edit `src/agentfix/tools/tests_tool.py`. Stuck? `git checkout stage-1-solution`. |
| 0:36–0:50 | **Stage 2** — loop dispatch; run on task 01 | `uv run pytest exercises/stage_2 -v` then `uv run agentfix solve tasks/workshop/01-shopcart --verbose` | Students edit the dispatch block in `src/agentfix/agent/loop.py`. The classic bug is forgetting `tool_call_id` — the test names it directly. At `stage-2-solution` the stop condition does not exist yet, so this run will always burn its full 10-step budget and print `NOT SOLVED` — that's expected and the point, not a failure. Checkpoint: `stage-2-solution`. |
| 0:50–1:08 | **Stage 3** — stop condition; run on task 02 | `uv run pytest exercises/stage_3 -v` then `uv run agentfix solve tasks/workshop/02-invoice --verbose` | The naive "model stopped calling tools" or "model said DONE" both fail here — the scripted model in the test claims success while tests still fail. Task 02's bug is not in the file the failing test points at, which is why `list_files`/`read_file` matter. Checkpoint: `stage-3-solution`. **This segment is protected at all costs — cut everything else before this.** |
| 1:08–1:16 | Eval discussion (demo-only) | show `results/precomputed/workshop.json` and `results/precomputed/humanevalfix.json`; do **not** run a live eval | HumanEvalFix (20 tasks) took **8m09s wall clock** and 185,235 tokens on this machine at 51 tok/s — that is exactly why this is discussion, not a live activity. pass@1 is 0.60 (12/20), median 7 steps, max 10 (the cap); every one of the 8 failures spent all 10 steps. The best story here: it was 0.50 until the loop's stop condition stopped being decorative — four failures used to quit at 3–5 steps with budget left. Two lines of loop change moved pass@1 more than any prompt tweak did. The contrast to draw: the predecessor project's 1.5B model scored **0.305 pass@1 (50/164)** on the full HumanEvalFix Python split, GPU-served — but **single-shot**, one patch per task with no loop and no tools. Its 13-config decoding sweep ranged 0.262–0.317, which at n=164 is inside the ±0.036 standard error, and only 31 of 164 tasks ever changed verdict. So: tuning temperature and beams moved nothing; a loop with a test-execution oracle roughly doubled the fix rate. Not apples-to-apples (task count, hardware, harness all differ) — the mechanism is the point. Those raw reports are no longer in the repo; `git log --diff-filter=D -- results/legacy` finds them in history. |
| 1:16–1:30 | Sandbox safety + Docker demo; Thinking variant; next steps | `docker build -t agentfix-sandbox -f Dockerfile.sandbox .` then `AGENTFIX_SANDBOX=docker uv run pytest tests/test_docker_backend.py -v` | Your agent executes model-written code — here is what that means and what production systems do about it. Two boundaries: the tool layer confines *paths* (`resolve_in_root` rejects escapes before a read/write happens); the sandbox confines *execution* (no network, memory/pid/cpu caps, non-root). **Verify the daemon is up and the image is built before this segment** — container execution is unverified in this repo: the image was never built on the authoring machine (registry unreachable from its Docker VM at first, daemon not running there since), so no container has ever actually run these flags, and a live failure here undercuts the safety point. Build the image with the tagged command above, then run `uv run pytest tests/test_docker_backend.py` to confirm it's live before the segment starts. Close with the Mellum2 Thinking variant as the natural next step (same code, one env var, visible `<think>` blocks) and mention planning/reflection/parallel tools as deliberately out of scope (see `ARCHITECTURE.md`). |

## Cut order when the room runs slow

Cut in this order, and only this order:

1. **Eval segment (−8 min).** It is discussion of pre-computed numbers, not hands-on work — the
   easiest thing to compress to "here's the table" or skip with a link to `results/precomputed/`.
2. **Docker demo (−4 min).** Say the two boundaries verbally (tool-layer path confinement vs.
   sandbox execution) and point at `ARCHITECTURE.md` instead of running it live.
3. **Fold the tool-calling contract (0:16–0:24) into the live demo (0:08–0:16).** Narrate the
   schema → tool_call → observation → append cycle while the demo agent is running instead of as a
   separate walkthrough.

**Stage 3 (0:50–1:08) is never cut.** Verification-by-execution is the single highest-value idea in
the workshop.

## Checkpoint-tag rescue

Anyone stuck on a stage checks out the next tag and rejoins the group without falling behind:

```bash
git checkout stage-1-solution   # or stage-2-solution, stage-3-solution
```

This detaches HEAD; tell students that's fine for the workshop and they can
`git checkout main` afterward to get back to a branch.

## Real agent failure modes (instructor material)

Most agent tutorials only ever show a success. These three traces are real, measured runs against
an experimental two-file variant of `03-parser` (bugs split across `tokens.py` and `parser.py`
instead of co-located in one file, to force genuine two-file iteration). The variant solved only
1 of 3 runs — which is exactly why the fixture that ships in this repo keeps both defects in one
file: a student hitting a stuck or give-up run mid-exercise would reasonably conclude they broke
something, not that they're looking at a documented model limitation. Use these traces
*deliberately*, on your terms, rather than leaving them to chance.

### 1. A working iteration (write → fail → write → pass)

```
step 6  → calls write_file   (Wrote 155 characters to config/tokens.py.)
step 7  → calls run_tests    (Tests failed. ..F  — one test still red)
step 8  → calls write_file   (Wrote 361 characters to config/parser.py.)
step 9  → calls run_tests    (All tests passed. ...)
step 10 → text: "The issue was caused by two problems in the parser..."

SOLVED  03-parser  steps=10/10  tokens=14112  17.08s
```

The `..F` at step 7 is the tell: the model read partial progress correctly and made a *second,
different* edit rather than repeating the first one. This is the whole lesson of Stage 3 in
miniature. Note it also spent all 10 of 10 steps — zero margin.

### 2. Giving up with a text-only diagnosis, never editing

```
step 1–5  → run_tests, list_files, read_file, read_file, read_file   (as above)
step 6    → text: "The issue is that the parser does not handle comments starting
                    with `#` and whitespace trimming corr[...]"

NOT SOLVED  03-parser  steps=6/10  tokens=6821  13.19s
```

The model correctly diagnosed both bugs in prose — and then never called `write_file`. The loop
ended because no further tool call came, leaving 4 steps of budget unused and the fixture still
broken. Teaching point: a text-only "I know what's wrong" is not a fix.

**Also a pre-fix recording.** At the time, *any* text-only reply ended the run — `is_done` decided
only the reported `solved` flag, so the loop's stop condition was decorative. It no longer is: a
text-only reply while the tests are red now gets one appended nudge ("The tests have not passed.
Read the latest failure and write a fix.") and the loop spends another step. Use this trace to make
the point concrete — the fix for "the agent gave up with budget to spare" was four lines in the
loop, not a better model.

### 3. Burning the whole budget re-verifying without ever editing again

```
step 1–5  → run_tests, list_files, read_file, read_file, read_file
step 6    → calls run_tests   (Tests failed. .FF)
step 7    → calls run_tests
step 8    → calls run_tests
step 9    → calls run_tests
step 10   → calls run_tests

NOT SOLVED  03-parser  steps=10/10  tokens=17669  34.27s
```

After confirming the failure at step 6, the model called `run_tests` four more times in a row
without ever attempting a second `write_file`. Each call had no arguments, so the guard fired and
injected an "already called" observation each time — and the model kept calling it anyway.

**This is a pre-fix recording, and it is what the loop guard was changed for.** The guard used to
inject the same observation indefinitely, so a stuck model could spend its whole budget collecting
them. It now counts *consecutive* hits, escalates the wording on the second, and abandons the run
at `MAX_GUARD_HITS = 3` — the same behaviour today would end at step 9, and each guarded call now
records a trace event instead of leaving a silent gap under `--verbose`. Show the trace, then say
what changed and why: an agent that cannot make progress should stop, not spend the remaining
budget proving it.

What the guard still does not do is break a longer *cycle*: `run_tests → list_files → run_tests`
never trips it, because no two consecutive signatures match.

Say this plainly to students: agents have budgets, and a stuck model can spend its entire budget
without making progress. That is not a bug in this codebase; it is what a hard step cap is *for* —
it makes the failure visible and bounded instead of an unbounded wait.
