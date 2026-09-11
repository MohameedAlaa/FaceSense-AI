"""
Database engine, session management, and FastAPI dependency injection.
Supports PostgreSQL (production) and SQLite (testing/local).
"""

import logging
from typing import Generator, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker] = None


def get_engine() -> Optional[Engine]:
    """
    Get or lazily initialize the SQLAlchemy engine.
    Returns None if settings.DATABASE_URL is not configured.
    """
    global _engine
    if _engine is not None:
        return _engine

    db_url = settings.DATABASE_URL
    if not db_url or not db_url.strip():
        logger.info("DATABASE_URL is not set. Database integration is in fallback mode.")
        return None

    try:
        engine_kwargs = {
            "echo": settings.DATABASE_ECHO,
        }

        if "sqlite" in db_url.lower():
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        else:
            engine_kwargs["pool_size"] = settings.DATABASE_POOL_SIZE
            engine_kwargs["max_overflow"] = settings.DATABASE_MAX_OVERFLOW
            engine_kwargs["pool_timeout"] = settings.DATABASE_POOL_TIMEOUT
            engine_kwargs["pool_pre_ping"] = True

        _engine = create_engine(db_url, **engine_kwargs)
        logger.info("Database engine initialized successfully.")
        return _engine
    except Exception as e:
        logger.error(f"Failed to initialize database engine: {e}")
        return None


def get_session_factory() -> Optional[sessionmaker]:
    """
    Get or lazily initialize the sessionmaker.
    Returns None if engine is not available.
    """
    global _SessionFactory
    if _SessionFactory is not None:
        return _SessionFactory

    engine = get_engine()
    if engine is None:
        return None

    _SessionFactory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False,
    )
    return _SessionFactory


def reset_engine() -> None:
    """Reset the global engine and session factory (useful in test teardown)."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None


def get_db() -> Generator[Optional[Session], None, None]:
    """
    FastAPI dependency that yields an active database Session if the database is configured.
    Yields None if the database is not configured.
    Guarantees session cleanup (rollback on unhandled error, close on exit).
    """
    factory = get_session_factory()
    if factory is None:
        yield None
        return

    session: Session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# Alias for endpoints that gracefully handle missing database connection
get_db_optional = get_db


def check_db_connection(session: Optional[Session] = None) -> bool:
    """
    Check if the database connection is alive by executing `SELECT 1`.
    Can accept an existing session or open a brief check session.
    """
    if session is not None:
        try:
            session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.warning(f"Database connection check failed on session: {e}")
            return False

    engine = get_engine()
    if engine is None:
        return False

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.warning(f"Database connection check failed: {e}")
        return False
