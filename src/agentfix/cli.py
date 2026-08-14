from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentfix import __version__
from agentfix.agent.loop import MAX_STEPS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentfix", description="A teaching coding agent")
    parser.add_argument("--version", action="store_true", help="print version and exit")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="check that this machine is ready for the workshop")

    solve = sub.add_parser("solve", help="run the agent on one task")
    solve.add_argument("task_dir", type=Path)
    solve.add_argument("--verbose", action="store_true", help="print the agent's trace")
    solve.add_argument("--max-steps", type=int, default=MAX_STEPS)

    evaluate = sub.add_parser("eval", help="run the agent over a suite of tasks")
    evaluate.add_argument("--suite", default="workshop", choices=["workshop", "humanevalfix"])
    evaluate.add_argument("--limit", type=int, default=3)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 2

    if args.version:
        print(f"agentfix {__version__}")
        return 0

    if args.command == "doctor":
        from agentfix.doctor import report, run_checks

        return report(run_checks())

    if args.command == "solve":
        from agentfix.runner import solve_task

        result = solve_task(args.task_dir, verbose=args.verbose, max_steps=args.max_steps)
        status = "SOLVED" if result.solved else "NOT SOLVED"
        print(
            f"\n{status}  {result.task_id}  "
            f"steps={result.steps_used}  tokens={result.prompt_tokens + result.completion_tokens}  "
            f"{result.duration_s}s"
        )
        return 0 if result.solved else 1

    if args.command == "eval":
        from agentfix.eval.runner import run_suite

        return run_suite(args.suite, limit=args.limit)

    parser.print_help()
    return 0


def cli_entry() -> None:
    sys.exit(main())
