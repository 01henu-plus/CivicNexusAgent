from __future__ import annotations
from collections.abc import Iterator
from contextlib import contextmanager
from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool
from civicnexus.core.config import get_settings


class Base(DeclarativeBase):
    pass


def create_db_engine(database_url: str, *, echo: bool = False) -> Engine:
    """Create a PostgreSQL production engine or a SQLite test engine."""
    options: dict[str, object] = {"echo": echo}
    if database_url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
        if database_url in {"sqlite://", "sqlite:///:memory:"}:
            options["poolclass"] = StaticPool
    else:
        options["pool_pre_ping"] = True
    db_engine = create_engine(database_url, **options)
    if database_url.startswith("sqlite"):

        @event.listens_for(db_engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _connection_record):  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return db_engine


def create_session_factory(
    database_url: str | Engine, *, echo: bool = False
) -> sessionmaker[Session]:
    bound_engine = (
        database_url
        if isinstance(database_url, Engine)
        else create_db_engine(database_url, echo=echo)
    )
    return sessionmaker(
        bind=bound_engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )


settings = get_settings()
engine = create_db_engine(settings.database_url)
SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    expire_on_commit=False,
)


def init_db(db_engine: Engine | None = None) -> None:
    """Create all tables. Production schema migration can later replace this."""
    from civicnexus.persistence import models  # noqa: F401

    target = db_engine or engine
    Base.metadata.create_all(target)
    _additive_schema_upgrade(target)


def _additive_schema_upgrade(target: Engine) -> None:
    """Add fields introduced by the harness without destructive migrations.
    The demo ships with a small SQLite database that may have been created by
    an earlier skeleton.  ``create_all`` intentionally does not alter existing
    tables, so these nullable additions keep that database compatible while
    remaining safe for PostgreSQL deployments.
    """
    additions: dict[str, dict[str, str]] = {
        "run_events": {"state_before": "VARCHAR(64)", "state_after": "VARCHAR(64)"},
        "facts": {"metadata_json": "JSON"},
        "context_snapshots": {
            "status": "VARCHAR(64)",
            "summary": "TEXT",
            "facts": "JSON",
            "evidence": "JSON",
            "recent_messages": "JSON",
            "memory_refs": "JSON",
            "state": "JSON",
            "stats": "JSON",
        },
    }
    with target.begin() as connection:
        inspector = inspect(connection)
        for table, columns in additions.items():
            if table not in inspector.get_table_names():
                continue
            existing = {column["name"] for column in inspector.get_columns(table)}
            for column, column_type in columns.items():
                if column not in existing:
                    connection.execute(
                        text(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {column_type}')
                    )


@contextmanager
def session_scope(factory: sessionmaker[Session] = SessionLocal) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a transaction-scoped SQLAlchemy session."""
    with session_scope() as session:
        yield session


__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "create_db_engine",
    "create_session_factory",
    "init_db",
    "session_scope",
    "get_db",
]
