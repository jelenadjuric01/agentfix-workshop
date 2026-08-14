# Exercises

Three stages. Each one edits real source files in `src/agentfix/` and has tests that
run **without a model** — you can finish every stage offline.

| Stage | You write | File | Test |
|---|---|---|---|
| 1 | the `run_tests` tool + its JSON schema | `src/agentfix/tools/tests_tool.py` | `uv run pytest exercises/stage_1` |
| 2 | the loop's tool dispatch | `src/agentfix/agent/loop.py` | `uv run pytest exercises/stage_2` |
| 3 | the stop condition | `src/agentfix/agent/loop.py` | `uv run pytest exercises/stage_3` |

Stuck? Jump ahead without falling behind the room:

    git checkout stage-1-solution     # or stage-2-solution, stage-3-solution
