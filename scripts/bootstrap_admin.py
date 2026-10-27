#!/usr/bin/env python3
"""Create the first Gatekeeper organization and all-scopes API key."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from sqlalchemy.exc import SQLAlchemyError

from backend.app.auth.bootstrap import (
    BootstrapAlreadyInitializedError,
    bootstrap_first_admin,
)
from backend.app.core.config import get_settings
from backend.app.database.session import create_database_engine, create_session_factory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap an empty Gatekeeper database with its first administrator key."
    )
    parser.add_argument("--organization-name", required=True)
    parser.add_argument("--key-environment", required=True, choices=("test", "live"))
    parser.add_argument(
        "--confirm-empty-database",
        action="store_true",
        help="Confirm that this command is intended for a freshly migrated, empty database.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.confirm_empty_database:
        print(
            "Bootstrap refused: pass --confirm-empty-database after verifying the target database.",
            file=sys.stderr,
        )
        return 2

    engine = None
    try:
        settings = get_settings()
        engine = create_database_engine(settings.database_url)
        session_factory = create_session_factory(engine)
        with session_factory() as session:
            result = bootstrap_first_admin(
                session,
                organization_name=args.organization_name,
                environment=args.key_environment,
                pepper=settings.api_key_pepper,
            )
    except BootstrapAlreadyInitializedError:
        print(
            "Bootstrap refused: organizations and API keys must both be empty.",
            file=sys.stderr,
        )
        return 2
    except (SQLAlchemyError, ValueError):
        print(
            "Bootstrap failed: verify the configured settings, migrations, and database access.",
            file=sys.stderr,
        )
        return 1
    finally:
        if engine is not None:
            engine.dispose()

    print("Bootstrap complete. Store this API key now; it cannot be recovered.")
    print(result.api_key.secret)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
