"""
Database package for FaceSense AI.
Provides declarative Base, engine initialization, session factory, and dependencies.
"""

from backend.app.db.base import Base
from backend.app.db.session import (
    get_engine,
    get_session_factory,
    get_db,
    check_db_connection,
    reset_engine,
)

__all__ = [
    "Base",
    "get_engine",
    "get_session_factory",
    "get_db",
    "check_db_connection",
    "reset_engine",
]
