"""
database.py – Engine & session factory for PostgreSQL.

Reads DATABASE_URL from environment with a sensible local default
matching the docker-compose.yml configuration.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://admin:admin123@localhost:5432/invoices",
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,       # reconnect on stale connections
    pool_size=5,               # production-sensible defaults
    max_overflow=10,
    echo=False,                # flip to True for SQL debugging
)

SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


def get_db() -> Session:
    """Dependency-injection helper (e.g. for FastAPI).

    Yields a session and guarantees it is closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
