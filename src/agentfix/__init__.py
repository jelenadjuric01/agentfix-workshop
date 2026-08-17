"""agentfix — a coding agent small enough to read end to end.

No framework. An agent here is a bounded loop that asks a model what to do, does it, and
verifies the result by running the tests rather than by believing the model.

Suggested reading order:

  1. llm/types.py        the whole interface to a model: 3 declarations, no I/O
  2. tools/base.py       what a tool is; the registry that runs model-requested calls
  3. tasks/loader.py     what a task is; the copy-to-tempdir context manager
  4. tools/fs.py         list_files, read_file, write_file
  5. tools/tests_tool.py run_tests — the agent's only oracle
  6. agent/loop.py       the agent. If you read one file, read this one.
  7. runner.py           how the pieces above are wired together

Then, as needed: llm/client.py and llm/fake.py (the real and scripted models),
sandbox/ (how tests are executed), eval/ (measurement), doctor.py and cli.py (entry points).

ARCHITECTURE.md annotates the same loop with the design decisions and their measurements.
"""

__version__ = "0.1.0"
