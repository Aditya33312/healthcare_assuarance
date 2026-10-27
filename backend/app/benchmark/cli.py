"""Command-line benchmark report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from backend.app.benchmark.harness import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the Clinical Compare Engine")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("benchmark/clinical_faithfulness/v0/cases.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-f1", type=float, default=0.0)
    parser.add_argument(
        "--min-critical-recall",
        type=float,
        default=0.01,
        help="Provisional development gate; defaults above zero and remains configurable",
    )
    parser.add_argument(
        "--source-revision",
        default=os.environ.get("GATEKEEPER_SOURCE_REVISION", "UNRECORDED"),
        help="Source revision recorded in the generated artifact",
    )
    parser.add_argument(
        "--repository-state",
        default=os.environ.get("GATEKEEPER_REPOSITORY_STATE", "UNRECORDED"),
        help="Repository state recorded in the generated artifact",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.min_f1 <= 1.0:
        raise SystemExit("--min-f1 must be between 0 and 1")
    if not 0.0 < args.min_critical_recall <= 1.0:
        raise SystemExit("--min-critical-recall must be greater than 0 and at most 1")
    report = run_benchmark(
        args.path,
        source_revision=args.source_revision,
        repository_state=args.repository_state,
    )
    serialized = json.dumps(report.as_dict(), indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")

    critical_recall = report.normalized_critical_error_recall or 0.0
    if report.cases_unassessed:
        return 2
    if report.normalized_overall.f1 < args.min_f1:
        return 1
    if critical_recall < args.min_critical_recall:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
