"""Regenerate tasks/humanevalfix/subset.json. Needs: uv sync --extra eval"""

from pathlib import Path

from agentfix.eval.humanevalfix import VENDORED_SUBSET, dump_rows, load_hf_rows

if __name__ == "__main__":
    dump_rows(load_hf_rows(sample=20, seed=42), Path(VENDORED_SUBSET))
    print(f"wrote {VENDORED_SUBSET}")
