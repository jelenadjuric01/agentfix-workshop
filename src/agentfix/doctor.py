from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentfix.config import BASE_MODEL, MIN_CONTEXT_LENGTH, LLMConfig

PULL_HINT = f"run: ollama pull {BASE_MODEL}"
CREATE_HINT = "run: ollama create agentfix-mellum2 -f Modelfile"
SERVE_HINT = (
    "the Ollama server is not answering — start it "
    "(macOS: open the Ollama app; otherwise: ollama serve)"
)

MIN_TOTAL_RAM_BYTES = 16 * 1024**3
COMFORTABLE_FREE_RAM_BYTES = 9 * 1024**3

# One short call measures cold model load and prefill, not generation. So warm up first,
# throw that call away, then time a prompt long enough that per-call overhead stops
# dominating the arithmetic.
WARMUP_PROMPT = "Reply with: ok"
THROUGHPUT_PROMPT = "Count from 1 to 120, one number per line, digits only."


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _get_json(url: str, timeout_s: float = 5.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    # json.loads returns Any; every caller indexes this like an object, so a list or a
    # bare string from an unexpected endpoint should read as "no data", not crash later.
    return payload if isinstance(payload, dict) else None


def _check_python() -> Check:
    version = sys.version_info
    ok = (version.major, version.minor) >= (3, 12)
    return Check("python", ok, f"{sys.version.split()[0]}" + ("" if ok else " — need >= 3.12"))


def _capture(command: list[str]) -> str:
    """stdout of a successful command. Raises CalledProcessError/OSError otherwise."""
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def _memory_bytes() -> tuple[int | None, int | None]:
    """(total, free). Either may be None on a platform we cannot read."""
    if sys.platform.startswith("linux"):
        try:
            fields = dict(
                (line.split(":", 1)[0], int(line.split()[1]) * 1024)
                for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
                if ":" in line and line.split()[1].isdigit()
            )
        except OSError:
            return None, None
        return fields.get("MemTotal"), fields.get("MemAvailable")

    if sys.platform == "darwin":
        try:
            total = int(_capture(["sysctl", "-n", "hw.memsize"]))
            page = int(_capture(["sysctl", "-n", "hw.pagesize"]))
            counts = {
                line.split(":")[0].strip(): int(line.split(":")[1].strip().rstrip("."))
                for line in _capture(["vm_stat"]).splitlines()
                if ":" in line and line.split(":")[1].strip().rstrip(".").isdigit()
            }
        except (OSError, ValueError, subprocess.CalledProcessError):
            return None, None
        reclaimable = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
        return total, sum(counts.get(key, 0) for key in reclaimable) * page

    return None, None


def _check_ram() -> Check:
    total, free = _memory_bytes()
    if total is None:
        return Check("ram", True, "could not read memory on this platform — check by hand")

    detail = f"{total / 1024**3:.1f} GB total"
    if free is not None:
        detail += f", {free / 1024**3:.1f} GB free"

    if total < MIN_TOTAL_RAM_BYTES:
        return Check(
            "ram",
            False,
            f"{detail} — under 16 GB, use tier 2 (Kaggle) or tier 3 (qwen2.5-coder:1.5b); "
            "see README.md",
        )
    if free is not None and free < COMFORTABLE_FREE_RAM_BYTES:
        detail += " — tight for an 8 GB model unless it is already loaded; close other apps"
    return Check("ram", True, detail)


def _check_ollama_installed() -> Check:
    path = shutil.which("ollama")
    return Check(
        "ollama installed", path is not None, path or "not found — install from ollama.com"
    )


def _check_server(config: LLMConfig) -> Check:
    payload = _get_json(f"{config.native_api_url}/api/tags")
    if payload is None:
        return Check("ollama server", False, SERVE_HINT)
    return Check("ollama server", True, f"reachable at {config.native_api_url}")


def _check_model_present(config: LLMConfig) -> Check:
    payload = _get_json(f"{config.native_api_url}/api/tags") or {}
    names = [model.get("name", "") for model in payload.get("models", [])]
    if any(name.split(":")[0] == config.model for name in names):
        return Check("model present", True, config.model)

    if any(name.split(":")[0] == BASE_MODEL for name in names):
        return Check(
            "model present", False, f"{BASE_MODEL} is pulled but not derived — {CREATE_HINT}"
        )
    return Check(
        "model present", False, f"{config.model} missing — {PULL_HINT}, then {CREATE_HINT}"
    )


def _check_generation(config: LLMConfig) -> Check:
    from agentfix.llm.client import OllamaClient

    client = OllamaClient(config)
    try:
        client.chat([{"role": "user", "content": WARMUP_PROMPT}])
    except Exception as error:
        return Check("generation", False, f"{type(error).__name__}: {error}")

    try:
        started = time.time()
        reply = client.chat([{"role": "user", "content": THROUGHPUT_PROMPT}])
        elapsed = max(time.time() - started, 1e-6)
    except Exception as error:
        return Check("generation", False, f"{type(error).__name__}: {error}")

    rate = reply.completion_tokens / elapsed
    return Check(
        "generation",
        True,
        f"{rate:.0f} tok/s ({reply.completion_tokens} tokens in {elapsed:.1f}s)",
    )


def _check_context(config: LLMConfig) -> Check:
    """The single most consequential setting, and the one nothing else will tell you about."""
    payload = _get_json(f"{config.native_api_url}/api/ps")
    if payload is None:
        return Check("context window", False, f"cannot read /api/ps — {SERVE_HINT}")

    loaded = [
        model
        for model in payload.get("models", [])
        if model.get("name", "").split(":")[0] == config.model
    ]
    if not loaded:
        return Check(
            "context window",
            False,
            f"{config.model} is not loaded, so its context cannot be read — "
            "run this check again after a generation succeeds",
        )

    length = loaded[0].get("context_length") or 0
    if length < MIN_CONTEXT_LENGTH:
        return Check(
            "context window",
            False,
            f"{length} tokens — too small; the agent's history will be silently truncated "
            f"mid-run. {CREATE_HINT}",
        )
    return Check("context window", True, f"{length} tokens")


def _check_sandbox() -> Check:
    import tempfile

    from agentfix.sandbox.base import get_backend

    with tempfile.TemporaryDirectory() as temp:
        Path(temp, "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        result = get_backend().run(Path(temp), (sys.executable, "-m", "pytest", "-q"))
    return Check("sandbox", result.passed, "executes tests" if result.passed else result.output)


def run_checks() -> list[Check]:
    config = LLMConfig.from_env()
    checks = [_check_python(), _check_ram(), _check_ollama_installed(), _check_server(config)]

    if checks[-1].ok:
        checks.append(_check_model_present(config))
        if checks[-1].ok:
            checks.append(_check_generation(config))
            checks.append(_check_context(config))

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
