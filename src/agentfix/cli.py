"""The command line: `agentfix doctor`, `agentfix solve`, `agentfix eval`.

Registered as a console script in pyproject.toml, which is what makes `uv run agentfix ...`
work. Nothing but argument parsing and printing happens here.

One pattern worth noticing: every subcommand imports its implementation *inside* its own
branch. That keeps `agentfix --version` from importing the OpenAI SDK or the eval harness, and
it means a broken optional dependency cannot stop the other commands from working.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentfix import __version__
from agentfix.agent.loop import MAX_STEPS


def build_parser() -> argparse.ArgumentParser:
    """Declare the CLI. Separate from `main` so tests can inspect it without running anything."""
    parser = argparse.ArgumentParser(prog="agentfix", description="A teaching coding agent")
    parser.add_argument("--version", action="store_true", help="print version and exit")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="check that this machine is ready for the workshop")

    solve = sub.add_parser("solve", help="run the agent on one task")
    # `type=Path` converts the string argparse collects into a Path for us.
    solve.add_argument("task_dir", type=Path)
    solve.add_argument("--verbose", action="store_true", help="print the agent's trace")
    # Defaulting to the loop's own constant, so there is one source of truth for the budget.
    solve.add_argument("--max-steps", type=int, default=MAX_STEPS)

    evaluate = sub.add_parser("eval", help="run the agent over a suite of tasks")
    evaluate.add_argument("--suite", default="workshop", choices=["workshop", "humanevalfix"])
    # Small default because eval against a local model is slow: the full HumanEvalFix subset
    # takes minutes per task. Raise it deliberately, not by accident.
    evaluate.add_argument("--limit", type=int, default=3)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one command and return its exit code. `argv=None` means "read sys.argv"."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse calls sys.exit() on a bad argument or on --help. Catching it and returning
        # the code instead lets tests call main() directly without the process dying.
        return int(exc.code) if exc.code is not None else 2

    if args.version:
        print(f"agentfix {__version__}")
        return 0

    if args.command == "doctor":
        from agentfix.doctor import report, run_checks

        return report(run_checks())

    if args.command == "solve":
        from agentfix.runner import solve_task

        try:
            result = solve_task(args.task_dir, verbose=args.verbose, max_steps=args.max_steps)
        except Exception as error:
            # The most likely exception on the exercise branch is NotImplementedError from an
            # unfinished stage, so the message points there rather than showing a bare
            # traceback. Errors go to stderr so they stay visible when stdout is redirected.
            print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
            print(
                "If you haven't finished the exercises yet, this is expected — see "
                "exercises/README.md.",
                file=sys.stderr,
            )
            return 1
        status = "SOLVED" if result.solved else "NOT SOLVED"
        print(
            f"\n{status}  {result.task_id}  "
            f"steps={result.steps_used}  tokens={result.prompt_tokens + result.completion_tokens}  "
            f"{result.duration_s}s"
        )
        # Exit code follows the verdict, so `agentfix solve ... && echo ok` behaves sensibly
        # and CI can gate on it.
        return 0 if result.solved else 1

    if args.command == "eval":
        from agentfix.eval.runner import run_suite

        return run_suite(args.suite, limit=args.limit)

    # No subcommand given: show help rather than failing silently.
    parser.print_help()
    return 0


def cli_entry() -> None:
    """The console-script entry point. Translates the return value into a process exit code."""
    sys.exit(main())
