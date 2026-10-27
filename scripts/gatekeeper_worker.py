#!/usr/bin/env python3
"""Run the durable Gatekeeper test-run worker.

Run this process beside the API in staging/production.  It is intentionally a
separate process so API requests stay responsive while application endpoints
are being exercised.
"""

from __future__ import annotations

import argparse
import asyncio

from backend.app.core.config import get_settings
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.orchestration.worker import TestRunWorker
from backend.app.security import EncryptedJsonStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute queued Gatekeeper test runs")
    parser.add_argument("--once", action="store_true", help="process one queued run and exit")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=5)
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = create_database_engine(settings.database_url)
    sessions = create_session_factory(engine)
    worker = TestRunWorker(
        sessions,
        EncryptedJsonStore(settings.data_encryption_keys),
        timeout_seconds=args.timeout_seconds or settings.customer_ai_timeout_seconds,
        max_attempts=args.max_attempts,
        max_concurrency=args.max_concurrency,
    )
    try:
        if args.once:
            await worker.execute_next()
            return 0
        if args.poll_seconds <= 0:
            raise ValueError("--poll-seconds must be greater than zero")
        while True:
            run_id = await worker.execute_next()
            if run_id is None:
                await asyncio.sleep(args.poll_seconds)
    finally:
        engine.dispose()


def main() -> int:
    return asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
