"""Tests for UserBaseline ORM model."""

import uuid
from datetime import datetime, timezone

from src.shared.models import UserBaseline


class TestUserBaselineModel:
    """Test UserBaseline model creation and fields."""

    def test_user_baseline_creation(self):
        """Test creating a UserBaseline instance."""
        now = datetime.now(timezone.utc)
        baseline = UserBaseline(
            id=uuid.uuid4(),
            user_id="test_user_123",
            biomarker_name="speech_activity",
            mean=0.75,
            std=0.15,
            percentile_25=0.65,
            percentile_75=0.85,
            data_points=30,
            window_start=now,
            window_end=now,
            updated_at=now,
        )

        assert baseline.user_id == "test_user_123"
        assert baseline.biomarker_name == "speech_activity"
        assert baseline.mean == 0.75
        assert baseline.std == 0.15
        assert baseline.percentile_25 == 0.65
        assert baseline.percentile_75 == 0.85
        assert baseline.data_points == 30

    def test_user_baseline_optional_percentiles(self):
        """Test UserBaseline with None percentiles."""
        now = datetime.now(timezone.utc)
        baseline = UserBaseline(
            id=uuid.uuid4(),
            user_id="test_user_456",
            biomarker_name="voice_energy",
            mean=0.5,
            std=0.2,
            percentile_25=None,
            percentile_75=None,
            data_points=5,
            window_start=now,
            window_end=now,
            updated_at=now,
        )

        assert baseline.percentile_25 is None
        assert baseline.percentile_75 is None
