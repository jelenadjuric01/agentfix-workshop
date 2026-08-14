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

## Finished early? When does your verification go stale?

`is_done` reads the *last* `run_tests` result. Work out what happens on this sequence:

    write_file(a correct fix) → run_tests(green) → write_file(breaks it again) → "all done"

Nothing re-ran the tests after that second write, so the green result no longer describes the code
on disk — and a naive `is_done` reports SOLVED. This repo closes it by having a successful
`write_file` clear `run_tests.last_result` (see `on_write` in `src/agentfix/runner.py`); find that
wiring and convince yourself it works. Then the harder question, which this repo does **not** solve:
what stops the agent from writing `if prices == [10.0, 5.0]: return 16.5`? See `ARCHITECTURE.md`,
"Two ways an agent passes without fixing anything".
