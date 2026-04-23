from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.shared.config import get_settings
from src.shared.database import Base, get_db  # noqa: F401 - exported for test overrides


@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine.

    Uses the same database as development for now.
    In production, this would use a separate test database.
    """
    settings = get_settings()
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
    )
    return engine


@pytest.fixture(scope="session")
def test_session_factory(test_engine):
    """Create session factory for tests."""
    return sessionmaker(
        bind=test_engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


@pytest.fixture(scope="function")
def db_session(test_engine, test_session_factory) -> Generator[Session, None, None]:
    """Provide a transactional database session for tests.

    Each test runs in a transaction that is rolled back after the test.
    """
    # Create all tables
    Base.metadata.create_all(bind=test_engine)

    # Clean all tables before test
    session = test_session_factory()
    for table in reversed(Base.metadata.sorted_tables):
        session.execute(table.delete())
    session.commit()
    session.close()

    # Create a new session for the test
    session = test_session_factory()

    try:
        yield session
    finally:
        # Rollback any uncommitted changes
        session.rollback()
        session.close()


@pytest.fixture(scope="function")
def override_get_db(db_session: Session):
    """Fixture for overriding get_db dependency in FastAPI tests.

    Usage:
        def test_something(client, override_get_db):
            app.dependency_overrides[get_db] = override_get_db
            # ... test code ...
    """

    def _override_get_db() -> Generator[Session, None, None]:
        yield db_session

    return _override_get_db
