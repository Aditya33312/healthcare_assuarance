"""Database engine/session construction without import-time connections."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.core.config import get_settings


def create_database_engine(url: str | None = None) -> Engine:
    database_url = url or get_settings().database_url
    kwargs: dict[str, object] = {"pool_pre_ping": True}
    if database_url in {"sqlite://", "sqlite+pysqlite:///:memory:"}:
        kwargs.update(
            {
                "connect_args": {"check_same_thread": False},
                "poolclass": StaticPool,
            }
        )
    return create_engine(database_url, **kwargs)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
