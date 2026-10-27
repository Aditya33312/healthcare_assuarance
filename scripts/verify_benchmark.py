"""Verify that the checked clinical benchmark artifact matches the current engine."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backend.app.benchmark.harness import run_benchmark
from scripts.generate_critical_diagnostics import (
    OUTPUT as CHECKED_DIAGNOSTICS,
)
from scripts.generate_critical_diagnostics import (
    build_artifact as build_critical_diagnostics,
)

CORPUS = Path("benchmark/clinical_faithfulness/v0/cases.jsonl")
CHECKED_RESULT = Path("benchmark/clinical_faithfulness/v0/benchmark-result.json")


def main() -> int:
    try:
        checked = CHECKED_RESULT.read_bytes()
        checked_payload = json.loads(checked)
    except FileNotFoundError:
        print(f"Missing checked benchmark artifact: {CHECKED_RESULT}")
        return 1
    report = run_benchmark(
        CORPUS,
        source_revision=str(checked_payload["source_revision"]),
        repository_state=str(checked_payload["repository_state"]),
        python_version=str(checked_payload["python_version"]),
    )
    generated = (json.dumps(report.as_dict(), indent=2) + "\n").encode()
    if generated != checked:
        print(
            "Checked benchmark artifact is stale: "
            f"expected sha256:{hashlib.sha256(generated).hexdigest()}, "
            f"found sha256:{hashlib.sha256(checked).hexdigest()}"
        )
        return 1
    generated_diagnostics = (json.dumps(build_critical_diagnostics(), indent=2) + "\n").encode()
    try:
        checked_diagnostics = CHECKED_DIAGNOSTICS.read_bytes()
    except FileNotFoundError:
        print(f"Missing checked CRITICAL diagnostics: {CHECKED_DIAGNOSTICS}")
        return 1
    if generated_diagnostics != checked_diagnostics:
        print(
            "Checked CRITICAL diagnostics are stale: "
            f"expected sha256:{hashlib.sha256(generated_diagnostics).hexdigest()}, "
            f"found sha256:{hashlib.sha256(checked_diagnostics).hexdigest()}"
        )
        return 1
    print(
        f"Benchmark artifact verified ({report.cases_total} cases, "
        f"sha256:{hashlib.sha256(checked).hexdigest()})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
