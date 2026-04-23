from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.shared.config import get_settings


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # Verify connections before use
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """Dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
