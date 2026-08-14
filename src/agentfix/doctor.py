from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

from agentfix.config import LLMConfig

PULL_HINT = "run: ollama pull hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M"


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _check_python() -> Check:
    version = sys.version_info
    ok = (version.major, version.minor) >= (3, 12)
    return Check("python", ok, f"{sys.version.split()[0]}" + ("" if ok else " — need >= 3.12"))


def _check_ollama_installed() -> Check:
    path = shutil.which("ollama")
    return Check(
        "ollama installed", path is not None, path or "not found — install from ollama.com"
    )


def _check_model_present(config: LLMConfig) -> Check:
    if shutil.which("ollama") is None:
        return Check("model present", False, f"cannot check — {PULL_HINT}")
    listing = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    short = config.model.split("/")[-1].lower()
    ok = short in listing.stdout.lower()
    return Check("model present", ok, config.model if ok else f"missing — {PULL_HINT}")


def _check_generation(config: LLMConfig) -> Check:
    from agentfix.llm.client import OllamaClient

    try:
        started = time.time()
        reply = OllamaClient(config).chat([{"role": "user", "content": "Reply with: ok"}])
        elapsed = max(time.time() - started, 1e-6)
        rate = reply.completion_tokens / elapsed
        return Check("generation", True, f"{rate:.0f} tok/s")
    except Exception as error:
        return Check("generation", False, f"{type(error).__name__}: {error}")


def _check_sandbox() -> Check:
    import tempfile
    from pathlib import Path

    from agentfix.sandbox.base import get_backend

    with tempfile.TemporaryDirectory() as temp:
        Path(temp, "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        result = get_backend().run(Path(temp), (sys.executable, "-m", "pytest", "-q"))
    return Check("sandbox", result.passed, "executes tests" if result.passed else result.output)


def run_checks() -> list[Check]:
    config = LLMConfig.from_env()
    checks = [_check_python(), _check_ollama_installed(), _check_model_present(config)]
    if checks[-1].ok:
        checks.append(_check_generation(config))
    checks.append(_check_sandbox())
    return checks


def report(checks: list[Check]) -> int:
    for check in checks:
        mark = "PASS" if check.ok else "FAIL"
        print(f"[{mark}] {check.name}: {check.detail}")

    failed = [check for check in checks if not check.ok]
    if failed:
        print(f"\nNOT READY — {len(failed)} check(s) failed. Fix the FAIL lines above.")
        return 1

    rate = next((c.detail for c in checks if c.name == "generation"), "")
    print(f"\nREADY {rate}".rstrip())
    return 0
