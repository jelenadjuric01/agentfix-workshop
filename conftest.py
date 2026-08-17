from __future__ import annotations

import pytest

# `addopts = "-m 'not llm'"` in pyproject.toml means every pytest run excludes the tests
# that need a live Ollama. That default is right -- the whole suite has to pass offline --
# but it leaves no obvious way to opt back in: plain `pytest` cannot, and `-m llm` runs
# ONLY those tests. The incantation for "everything" was `-m "llm or not llm"`, which is
# accurate, unmemorable, and looks like a typo. `--all` is the same thing with a name.

DEFAULT_MARKEXPR = "not llm"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help=(
            "run every test, including the `llm` ones that need a running Ollama "
            "with the model from the README (clears the default -m 'not llm')"
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    if not config.getoption("--all"):
        return

    # An explicit -m and --all contradict each other. Silently discarding the user's
    # filter would run more tests than they asked for, so say so instead.
    markexpr = config.option.markexpr
    if markexpr not in ("", DEFAULT_MARKEXPR):
        raise pytest.UsageError(f"--all cannot be combined with -m {markexpr!r}; pick one")

    config.option.markexpr = ""
