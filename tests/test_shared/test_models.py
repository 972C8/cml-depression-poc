"""Tests for SQLAlchemy ORM models."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import inspect

from src.shared.database import Base, engine
from src.shared.models import Biomarker, BiomarkerType, Context, Indicator, init_db


class TestInitDb:
    """Tests for init_db function and table creation."""

    def test_init_db_creates_tables(self):
        """Verify init_db creates all expected tables."""
        # Create tables
        init_db()

        # Check tables exist
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        assert "biomarkers" in table_names
        assert "context" in table_names
        assert "indicators" in table_names

    def test_base_metadata_has_tables(self):
        """Verify Base.metadata contains all model tables."""
        table_names = list(Base.metadata.tables.keys())

        assert "biomarkers" in table_names
        assert "context" in table_names
        assert "indicators" in table_names


class TestBiomarkerModel:
    """Tests for Biomarker model."""

    def test_biomarker_has_required_columns(self):
        """Verify Biomarker model has all required columns."""
        columns = {c.name for c in Biomarker.__table__.columns}

        assert "id" in columns
        assert "user_id" in columns
        assert "timestamp" in columns
        assert "biomarker_type" in columns
        assert "value" in columns
        assert "metadata" in columns  # Column name, not Python attr
        assert "created_at" in columns

    def test_biomarker_type_enum(self):
        """Verify BiomarkerType enum values."""
        assert BiomarkerType.SPEECH.value == "speech"
        assert BiomarkerType.NETWORK.value == "network"

    def test_biomarker_user_timestamp_index(self):
        """Verify composite index on user_id and timestamp."""
        indexes = {idx.name for idx in Biomarker.__table__.indexes}
        assert "ix_biomarkers_user_timestamp" in indexes

    def test_biomarker_create(self, db_session):
        """Test creating a Biomarker instance."""
        biomarker = Biomarker(
            user_id="test-user-1",
            timestamp=datetime.now(UTC),
            biomarker_type=BiomarkerType.SPEECH.value,
            value={"feature1": 0.5, "feature2": 0.8},
            metadata_={"source": "test"},
        )
        db_session.add(biomarker)
        db_session.commit()

        assert biomarker.id is not None
        assert isinstance(biomarker.id, uuid.UUID)
        assert biomarker.created_at is not None


class TestContextModel:
    """Tests for Context model."""

    def test_context_has_required_columns(self):
        """Verify Context model has all required columns."""
        columns = {c.name for c in Context.__table__.columns}

        assert "id" in columns
        assert "user_id" in columns
        assert "timestamp" in columns
        assert "context_type" in columns
        assert "value" in columns
        assert "metadata" in columns
        assert "created_at" in columns

    def test_context_user_timestamp_index(self):
        """Verify composite index on user_id and timestamp."""
        indexes = {idx.name for idx in Context.__table__.indexes}
        assert "ix_context_user_timestamp" in indexes

    def test_context_create(self, db_session):
        """Test creating a Context instance."""
        context = Context(
            user_id="test-user-1",
            timestamp=datetime.now(UTC),
            context_type="location",
            value={"lat": 40.7128, "lng": -74.0060},
        )
        db_session.add(context)
        db_session.commit()

        assert context.id is not None
        assert isinstance(context.id, uuid.UUID)


class TestIndicatorModel:
    """Tests for Indicator model."""

    def test_indicator_has_required_columns(self):
        """Verify Indicator model has all required columns."""
        columns = {c.name for c in Indicator.__table__.columns}

        assert "id" in columns
        assert "user_id" in columns
        assert "timestamp" in columns
        assert "indicator_type" in columns
        assert "value" in columns
        assert "data_reliability_score" in columns
        assert "analysis_run_id" in columns
        assert "config_snapshot" in columns
        assert "created_at" in columns

    def test_indicator_user_timestamp_index(self):
        """Verify composite index on user_id and timestamp."""
        indexes = {idx.name for idx in Indicator.__table__.indexes}
        assert "ix_indicators_user_timestamp" in indexes

    def test_indicator_analysis_run_id_index(self):
        """Verify index on analysis_run_id."""
        # Check if analysis_run_id column has index
        for idx in Indicator.__table__.indexes:
            cols = [c.name for c in idx.columns]
            if "analysis_run_id" in cols and len(cols) == 1:
                return  # Found single-column index
        # Also check column-level index
        col = Indicator.__table__.c.analysis_run_id
        assert col.index is True

    def test_indicator_create(self, db_session):
        """Test creating an Indicator instance."""
        run_id = uuid.uuid4()
        indicator = Indicator(
            user_id="test-user-1",
            timestamp=datetime.now(UTC),
            indicator_type="stress_level",
            value=0.75,
            data_reliability_score=0.9,
            analysis_run_id=run_id,
            config_snapshot={"algorithm": "v1.0"},
        )
        db_session.add(indicator)
        db_session.commit()

        assert indicator.id is not None
        assert indicator.analysis_run_id == run_id
