#!/usr/bin/env python3
"""Invoke Gatekeeper from GitHub Actions or another CI runner."""

from __future__ import annotations

import argparse
import os

from backend.app.ci import GateConfig, run_release_gate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--system-version", required=True)
    parser.add_argument("--environment", default="staging")
    parser.add_argument("--test-suite-id", required=True)
    parser.add_argument("--test-suite-version", required=True)
    parser.add_argument("--test-plan-id", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--github-run-id", required=True)
    parser.add_argument("--github-run-attempt", required=True, type=int)
    parser.add_argument("--pull-request", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=1_800)
    parser.add_argument("--fail-on-warn", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    api_key = os.environ.get("GATEKEEPER_API_KEY")
    if not api_key:
        print("Assurance could not complete: GATEKEEPER_API_KEY is not configured.")
        return 1
    outcome = run_release_gate(
        GateConfig(
            api_url=args.api_url,
            api_key=api_key,
            system_id=args.system_id,
            system_version=args.system_version,
            environment=args.environment,
            test_suite_id=args.test_suite_id,
            test_suite_version=args.test_suite_version,
            test_plan_id=args.test_plan_id,
            commit_sha=args.commit_sha,
            branch=args.branch,
            github_run_id=args.github_run_id,
            github_run_attempt=args.github_run_attempt,
            pull_request=args.pull_request,
            timeout_seconds=args.timeout_seconds,
            fail_on_warn=args.fail_on_warn,
        )
    )
    print(outcome.message)
    if outcome.run_id:
        print(f"Run: {outcome.run_id}")
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
