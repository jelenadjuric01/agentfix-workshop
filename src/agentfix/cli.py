from __future__ import annotations

import argparse
import sys

from agentfix import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentfix", description="A teaching coding agent")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    parser.add_argument("command", nargs="?", choices=["doctor", "solve", "eval"])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args, _rest = parser.parse_known_args(argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 2

    if args.version:
        print(f"agentfix {__version__}")
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    print(f"'{args.command}' is not implemented yet")
    return 0


def cli_entry() -> None:
    sys.exit(main())
