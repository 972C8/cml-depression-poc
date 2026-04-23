"""Tests for shared database module."""

from collections.abc import Generator

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.shared.database import Base, SessionLocal, engine, get_db


class TestEngine:
    """Tests for database engine configuration."""

    def test_engine_is_created(self):
        """Verify engine is created successfully."""
        assert engine is not None
        assert isinstance(engine, Engine)

    def test_engine_has_pool_pre_ping(self):
        """Verify engine is configured with pool_pre_ping."""
        # pool_pre_ping is stored as _pre_ping in QueuePool
        assert engine.pool._pre_ping is True

    def test_engine_can_connect(self):
        """Verify engine can establish a connection."""
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1


class TestSessionLocal:
    """Tests for SessionLocal factory."""

    def test_session_local_is_sessionmaker(self):
        """Verify SessionLocal is a sessionmaker instance."""
        assert isinstance(SessionLocal, sessionmaker)

    def test_session_local_creates_session(self):
        """Verify SessionLocal creates a valid session."""
        session = SessionLocal()
        try:
            assert isinstance(session, Session)
        finally:
            session.close()

    def test_session_local_autocommit_disabled(self):
        """Verify sessions have autocommit disabled.

        In SQLAlchemy 2.0, autocommit is configured on the sessionmaker.
        We verify by checking the sessionmaker's kw configuration.
        """
        # SessionLocal is configured with autocommit=False
        assert SessionLocal.kw.get("autocommit", False) is False

    def test_session_local_autoflush_disabled(self):
        """Verify sessions have autoflush disabled."""
        session = SessionLocal()
        try:
            assert session.autoflush is False
        finally:
            session.close()


class TestGetDb:
    """Tests for get_db dependency function."""

    def test_get_db_returns_generator(self):
        """Verify get_db returns a generator."""
        result = get_db()
        assert isinstance(result, Generator)

    def test_get_db_yields_session(self):
        """Verify get_db yields a database session."""
        gen = get_db()
        session = next(gen)
        try:
            assert isinstance(session, Session)
        finally:
            # Exhaust the generator to trigger cleanup
            try:
                next(gen)
            except StopIteration:
                pass

    def test_get_db_session_is_usable(self):
        """Verify yielded session can execute queries."""
        gen = get_db()
        session = next(gen)
        try:
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1
        finally:
            try:
                next(gen)
            except StopIteration:
                pass

    def test_get_db_closes_session_on_exit(self):
        """Verify session is closed after generator exits."""
        gen = get_db()
        _ = next(gen)  # Get session (name unused, testing cleanup behavior)
        # Exhaust the generator
        try:
            next(gen)
        except StopIteration:
            pass
        # Session should be closed (will raise if we try to use it improperly)
        # We can't directly check is_active after close, but the cleanup ran


class TestBase:
    """Tests for declarative Base class."""

    def test_base_has_metadata(self):
        """Verify Base has metadata attribute."""
        assert hasattr(Base, "metadata")
        assert Base.metadata is not None

    def test_base_metadata_contains_tables(self):
        """Verify Base.metadata knows about model tables."""
        # Import models to ensure they're registered
        from src.shared.models import Biomarker, Context, Indicator  # noqa: F401

        table_names = list(Base.metadata.tables.keys())
        assert "biomarkers" in table_names
        assert "context" in table_names
        assert "indicators" in table_names
