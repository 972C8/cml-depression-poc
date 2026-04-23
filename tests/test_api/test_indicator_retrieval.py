"""Tests for indicator retrieval endpoint."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.shared.database import get_db
from src.shared.models import Indicator

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_indicators(db_session):
    """Clean indicators table before each test."""
    db_session.query(Indicator).delete()
    db_session.commit()


class TestGetIndicators:
    """Tests for GET /indicators endpoint."""

    @patch("src.api.dependencies.get_settings")
    def test_get_indicators_success(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test successful retrieval with all filters."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed test data
        now = datetime.now(UTC)
        analysis_run_id = uuid4()

        indicator1 = Indicator(
            user_id="user-123",
            timestamp=now - timedelta(hours=2),
            indicator_type="stress",
            value=0.75,
            data_reliability_score=0.85,
            analysis_run_id=analysis_run_id,
            config_snapshot={"weights": {"speech": 0.6, "network": 0.4}},
        )
        indicator2 = Indicator(
            user_id="user-123",
            timestamp=now - timedelta(hours=1),
            indicator_type="stress",
            value=0.65,
            data_reliability_score=0.90,
            analysis_run_id=analysis_run_id,
        )
        db_session.add_all([indicator1, indicator2])
        db_session.commit()

        response = client.get(
            "/indicators",
            params={
                "user_id": "user-123",
                "start_time": (now - timedelta(hours=3)).isoformat(),
                "end_time": now.isoformat(),
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "meta" in data
        assert isinstance(data["data"], list)
        assert data["meta"]["count"] == 2

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_get_indicators_user_id_filter(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test filtering by user_id (required parameter)."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed data for multiple users
        now = datetime.now(UTC)
        analysis_run_id = uuid4()

        indicator1 = Indicator(
            user_id="user-123",
            timestamp=now,
            indicator_type="stress",
            value=0.75,
            analysis_run_id=analysis_run_id,
        )
        indicator2 = Indicator(
            user_id="user-456",
            timestamp=now,
            indicator_type="stress",
            value=0.65,
            analysis_run_id=analysis_run_id,
        )
        db_session.add_all([indicator1, indicator2])
        db_session.commit()

        response = client.get(
            "/indicators",
            params={"user_id": "user-123"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["meta"]["count"] == 1
        assert data["data"][0]["user_id"] == "user-123"

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_get_indicators_time_range_filters(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test start_time and end_time filters."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed data across different times
        now = datetime.now(UTC)
        analysis_run_id = uuid4()

        ind_old = Indicator(
            user_id="user-123",
            timestamp=now - timedelta(hours=10),
            indicator_type="stress",
            value=0.75,
            analysis_run_id=analysis_run_id,
        )
        ind_recent = Indicator(
            user_id="user-123",
            timestamp=now - timedelta(hours=2),
            indicator_type="stress",
            value=0.65,
            analysis_run_id=analysis_run_id,
        )
        db_session.add_all([ind_old, ind_recent])
        db_session.commit()

        # Request with specific time range (last 3 hours)
        response = client.get(
            "/indicators",
            params={
                "user_id": "user-123",
                "start_time": (now - timedelta(hours=3)).isoformat(),
                "end_time": now.isoformat(),
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        # Should only get the recent one
        assert data["meta"]["count"] == 1

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_get_indicators_filtered_by_type(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test filtering by indicator_type."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed multiple types
        now = datetime.now(UTC)
        analysis_run_id = uuid4()

        stress_ind = Indicator(
            user_id="user-123",
            timestamp=now,
            indicator_type="stress",
            value=0.75,
            analysis_run_id=analysis_run_id,
        )
        mood_ind = Indicator(
            user_id="user-123",
            timestamp=now,
            indicator_type="mood",
            value=0.60,
            analysis_run_id=analysis_run_id,
        )
        db_session.add_all([stress_ind, mood_ind])
        db_session.commit()

        response = client.get(
            "/indicators",
            params={"user_id": "user-123", "indicator_type": "stress"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["meta"]["count"] == 1
        assert data["data"][0]["indicator_type"] == "stress"

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_get_indicators_default_time_range(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test default time range is last 24 hours."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed data: one within 24h, one older
        now = datetime.now(UTC)
        analysis_run_id = uuid4()

        recent_ind = Indicator(
            user_id="user-123",
            timestamp=now - timedelta(hours=12),
            indicator_type="stress",
            value=0.75,
            analysis_run_id=analysis_run_id,
        )
        old_ind = Indicator(
            user_id="user-123",
            timestamp=now - timedelta(days=2),
            indicator_type="stress",
            value=0.65,
            analysis_run_id=analysis_run_id,
        )
        db_session.add_all([recent_ind, old_ind])
        db_session.commit()

        # Request without time range
        response = client.get(
            "/indicators",
            params={"user_id": "user-123"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        # Should only get the recent one
        assert data["meta"]["count"] == 1

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_get_indicators_ordered_by_timestamp(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test results are ordered by timestamp ascending (NFR3)."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed in non-chronological order
        now = datetime.now(UTC)
        analysis_run_id = uuid4()

        ind3 = Indicator(
            user_id="user-123",
            timestamp=now - timedelta(hours=1),
            indicator_type="stress",
            value=0.75,
            analysis_run_id=analysis_run_id,
        )
        ind1 = Indicator(
            user_id="user-123",
            timestamp=now - timedelta(hours=3),
            indicator_type="stress",
            value=0.65,
            analysis_run_id=analysis_run_id,
        )
        ind2 = Indicator(
            user_id="user-123",
            timestamp=now - timedelta(hours=2),
            indicator_type="stress",
            value=0.70,
            analysis_run_id=analysis_run_id,
        )
        db_session.add_all([ind3, ind1, ind2])
        db_session.commit()

        response = client.get(
            "/indicators",
            params={"user_id": "user-123"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        timestamps = [item["timestamp"] for item in data["data"]]
        # Verify ascending order
        assert timestamps == sorted(timestamps)

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_get_indicators_requires_api_key(self, mock_get_settings: MagicMock):
        """Test API key authentication is required."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        # Request without Authorization header
        response = client.get(
            "/indicators",
            params={"user_id": "user-123"},
        )

        assert response.status_code == 422  # Missing header caught by validation

    @patch("src.api.dependencies.get_settings")
    def test_get_indicators_missing_user_id_returns_422(
        self, mock_get_settings: MagicMock
    ):
        """Test missing required user_id returns 422 validation error."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        response = client.get(
            "/indicators",
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "VALIDATION_ERROR"
        errors = data["error"]["details"]["errors"]
        assert any("user_id" in err["field"] for err in errors)

    @patch("src.api.dependencies.get_settings")
    def test_get_indicators_invalid_timestamp_format_returns_422(
        self, mock_get_settings: MagicMock
    ):
        """Test invalid timestamp format returns 422 with clear error message."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        response = client.get(
            "/indicators",
            params={
                "user_id": "user-123",
                "start_time": "invalid-timestamp",
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "VALIDATION_ERROR"

    @patch("src.api.dependencies.get_settings")
    def test_get_indicators_empty_results(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test empty results when no data matches."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        response = client.get(
            "/indicators",
            params={"user_id": "nonexistent-user"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []
        assert data["meta"]["count"] == 0

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_get_indicators_includes_analysis_run_id(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test that analysis_run_id is included for traceability (AC10)."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed indicator
        now = datetime.now(UTC)
        analysis_run_id = uuid4()

        indicator = Indicator(
            user_id="user-123",
            timestamp=now,
            indicator_type="stress",
            value=0.75,
            data_reliability_score=0.85,
            analysis_run_id=analysis_run_id,
            config_snapshot={"weights": {"speech": 0.6, "network": 0.4}},
        )
        db_session.add(indicator)
        db_session.commit()

        response = client.get(
            "/indicators",
            params={"user_id": "user-123"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1

        indicator_data = data["data"][0]
        assert "analysis_run_id" in indicator_data
        assert indicator_data["analysis_run_id"] == str(analysis_run_id)
        assert "value" in indicator_data
        assert indicator_data["value"] == 0.75
        assert "data_reliability_score" in indicator_data
        assert indicator_data["data_reliability_score"] == 0.85
        assert "config_snapshot" in indicator_data

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_get_indicators_data_reliability_and_config_snapshot(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test that data_reliability_score and config_snapshot fields are included."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed indicator with all optional fields
        now = datetime.now(UTC)
        analysis_run_id = uuid4()

        indicator = Indicator(
            user_id="user-123",
            timestamp=now,
            indicator_type="stress",
            value=0.75,
            data_reliability_score=0.92,
            analysis_run_id=analysis_run_id,
            config_snapshot={
                "weights": {"speech": 0.6, "network": 0.4},
                "thresholds": {"low": 0.3, "high": 0.7},
            },
        )
        db_session.add(indicator)
        db_session.commit()

        response = client.get(
            "/indicators",
            params={"user_id": "user-123"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        indicator_data = data["data"][0]

        # Verify data_reliability_score field
        assert "data_reliability_score" in indicator_data
        assert indicator_data["data_reliability_score"] == 0.92

        # Verify config_snapshot field
        assert "config_snapshot" in indicator_data
        assert indicator_data["config_snapshot"]["weights"]["speech"] == 0.6
        assert indicator_data["config_snapshot"]["thresholds"]["low"] == 0.3

        app.dependency_overrides.clear()
