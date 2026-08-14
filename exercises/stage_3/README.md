# Stage 3 — When is it done?

Open `src/agentfix/agent/loop.py`, find `TODO(stage-3)` in `is_done`.

The tempting answers are both wrong:

- "the model stopped calling tools" — it may have given up, or hallucinated success
- "the model said DONE" — models say DONE about code that does not work

The agent is done when **the tests pass**. Verification by execution, not by assertion.
That is the difference between a demo and something you would let near real code.

Until you fix it, `is_done` just returns `False` — so `exercises/stage_3` fails with plain
`AssertionError`s (`solved` stays `False` even once the tests genuinely pass), not a crash.
`agentfix solve` will run to completion and report `NOT SOLVED` rather than raising.

    uv run pytest exercises/stage_3 -v
