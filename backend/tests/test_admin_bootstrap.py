from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.auth import ApiKeyAuthenticator, ApiScope
from backend.app.auth.bootstrap import (
    BootstrapAlreadyInitializedError,
    bootstrap_first_admin,
)
from backend.app.core.config import get_settings
from backend.app.database.base import Base
from backend.app.database.models import ApiKey, Organization
from backend.app.database.session import create_database_engine
from scripts.bootstrap_admin import main

PEPPER = "unit-test-bootstrap-pepper"


@pytest.fixture
def empty_session() -> Generator[Session, None, None]:
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def _count(session: Session, model: type[Organization] | type[ApiKey]) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


def test_bootstrap_creates_active_organization_and_all_scopes_key(
    empty_session: Session,
) -> None:
    result = bootstrap_first_admin(
        empty_session,
        organization_name="  Clinical Assurance  ",
        environment="live",
        pepper=PEPPER,
    )

    assert result.organization.name == "Clinical Assurance"
    assert result.organization.status == "ACTIVE"
    assert result.api_key.record.organization_id == result.organization.id
    assert result.api_key.record.status == "ACTIVE"
    assert result.api_key.record.environment == "live"
    assert set(result.api_key.record.scopes) == {scope.value for scope in ApiScope}
    assert ApiScope.API_KEYS_MANAGE.value in result.api_key.record.scopes
    assert result.api_key.secret not in repr(result)
    principal = ApiKeyAuthenticator(empty_session, PEPPER).authenticate(
        result.api_key.secret,
        required_scopes={ApiScope.API_KEYS_MANAGE},
    )
    assert principal.organization_id == result.organization.id


def test_bootstrap_refuses_an_initialized_database(empty_session: Session) -> None:
    bootstrap_first_admin(
        empty_session,
        organization_name="First",
        environment="test",
        pepper=PEPPER,
    )

    with pytest.raises(BootstrapAlreadyInitializedError):
        bootstrap_first_admin(
            empty_session,
            organization_name="Second",
            environment="test",
            pepper=PEPPER,
        )

    assert _count(empty_session, Organization) == 1
    assert _count(empty_session, ApiKey) == 1


@pytest.mark.parametrize("name", ["", "   ", "line\nbreak", "x" * 201])
def test_bootstrap_rejects_invalid_organization_names(
    empty_session: Session,
    name: str,
) -> None:
    with pytest.raises(ValueError):
        bootstrap_first_admin(
            empty_session,
            organization_name=name,
            environment="test",
            pepper=PEPPER,
        )

    assert _count(empty_session, Organization) == 0
    assert _count(empty_session, ApiKey) == 0


def test_bootstrap_rolls_back_the_organization_when_key_creation_fails(
    empty_session: Session,
) -> None:
    with pytest.raises(ValueError, match="pepper"):
        bootstrap_first_admin(
            empty_session,
            organization_name="Clinical Assurance",
            environment="test",
            pepper="too-short",
        )

    assert _count(empty_session, Organization) == 0
    assert _count(empty_session, ApiKey) == 0


def test_cli_uses_configured_database_and_prints_secret_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = tmp_path / "bootstrap.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path}"
    engine = create_database_engine(database_url)
    Base.metadata.create_all(engine)
    engine.dispose()
    monkeypatch.setenv("GATEKEEPER_ENVIRONMENT", "test")
    monkeypatch.setenv("GATEKEEPER_DATABASE_URL", database_url)
    monkeypatch.setenv("GATEKEEPER_API_KEY_PEPPER", PEPPER)
    get_settings.cache_clear()

    try:
        exit_code = main(
            [
                "--organization-name",
                "CLI Organization",
                "--key-environment",
                "test",
                "--confirm-empty-database",
            ]
        )
    finally:
        get_settings.cache_clear()

    captured = capsys.readouterr()
    output_lines = captured.out.strip().splitlines()
    assert exit_code == 0
    assert captured.err == ""
    assert len(output_lines) == 2
    assert output_lines[1].startswith("gk_test_")
    assert captured.out.count(output_lines[1]) == 1
    assert database_url not in captured.out
    assert PEPPER not in captured.out

    verification_engine = create_database_engine(database_url)
    with Session(verification_engine) as session:
        organization = session.scalar(select(Organization))
        api_key = session.scalar(select(ApiKey))
        assert organization is not None
        assert organization.name == "CLI Organization"
        assert api_key is not None
        assert api_key.organization_id == organization.id
        assert set(api_key.scopes) == {scope.value for scope in ApiScope}
    verification_engine.dispose()

    get_settings.cache_clear()
    try:
        repeat_exit_code = main(
            [
                "--organization-name",
                "Second Organization",
                "--key-environment",
                "test",
                "--confirm-empty-database",
            ]
        )
    finally:
        get_settings.cache_clear()
    repeat_capture = capsys.readouterr()
    assert repeat_exit_code == 2
    assert repeat_capture.out == ""
    assert output_lines[1] not in repeat_capture.err
    assert "must both be empty" in repeat_capture.err


def test_cli_requires_explicit_empty_database_confirmation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--organization-name", "Clinical Assurance", "--key-environment", "test"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "--confirm-empty-database" in captured.err
