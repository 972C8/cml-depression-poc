"""Tests for biomarker retrieval endpoint."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.shared.database import get_db
from src.shared.models import Biomarker

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_biomarkers(db_session):
    """Clean biomarkers table before each test."""
    db_session.query(Biomarker).delete()
    db_session.commit()


class TestGetBiomarkers:
    """Tests for GET /biomarkers endpoint."""

    @patch("src.api.dependencies.get_settings")
    def test_get_biomarkers_success_with_all_filters(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test successful retrieval with all filters (AC1, AC2, AC3, AC6)."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed test data
        now = datetime.now(UTC)
        biomarker1 = Biomarker(
            user_id="user-123",
            timestamp=now - timedelta(hours=2),
            biomarker_type="speech",
            value={"pitch": 120.5, "volume": 85},
        )
        biomarker2 = Biomarker(
            user_id="user-123",
            timestamp=now - timedelta(hours=1),
            biomarker_type="speech",
            value={"pitch": 125.0, "volume": 90},
        )
        db_session.add_all([biomarker1, biomarker2])
        db_session.commit()

        response = client.get(
            "/biomarkers",
            params={
                "user_id": "user-123",
                "start_time": (now - timedelta(hours=3)).isoformat(),
                "end_time": now.isoformat(),
                "biomarker_type": "speech",
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
    def test_get_biomarkers_user_id_filter(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test filtering by user_id (AC2)."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed data for different users
        now = datetime.now(UTC)
        biomarker_user1 = Biomarker(
            user_id="user-123",
            timestamp=now,
            biomarker_type="speech",
            value={"pitch": 120.5},
        )
        biomarker_user2 = Biomarker(
            user_id="user-456",
            timestamp=now,
            biomarker_type="speech",
            value={"pitch": 115.0},
        )
        db_session.add_all([biomarker_user1, biomarker_user2])
        db_session.commit()

        response = client.get(
            "/biomarkers",
            params={"user_id": "user-123"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["meta"]["count"] == 1
        assert data["data"][0]["user_id"] == "user-123"

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_get_biomarkers_time_range_filters(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test filtering by start_time and end_time (AC2)."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed data with different timestamps
        now = datetime.now(UTC)
        biomarker_old = Biomarker(
            user_id="user-123",
            timestamp=now - timedelta(hours=5),
            biomarker_type="speech",
            value={"pitch": 110.0},
        )
        biomarker_middle = Biomarker(
            user_id="user-123",
            timestamp=now - timedelta(hours=2),
            biomarker_type="speech",
            value={"pitch": 120.5},
        )
        biomarker_recent = Biomarker(
            user_id="user-123",
            timestamp=now - timedelta(minutes=30),
            biomarker_type="speech",
            value={"pitch": 125.0},
        )
        db_session.add_all([biomarker_old, biomarker_middle, biomarker_recent])
        db_session.commit()

        # Query with time range that excludes the old one
        response = client.get(
            "/biomarkers",
            params={
                "user_id": "user-123",
                "start_time": (now - timedelta(hours=3)).isoformat(),
                "end_time": now.isoformat(),
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["meta"]["count"] == 2  # Should not include biomarker_old

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_get_biomarkers_type_filter(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test filtering by biomarker_type (AC2)."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed data with different types
        now = datetime.now(UTC)
        speech_biomarker = Biomarker(
            user_id="user-123",
            timestamp=now,
            biomarker_type="speech",
            value={"pitch": 120.5},
        )
        network_biomarker = Biomarker(
            user_id="user-123",
            timestamp=now,
            biomarker_type="network",
            value={"contacts": 5},
        )
        db_session.add_all([speech_biomarker, network_biomarker])
        db_session.commit()

        response = client.get(
            "/biomarkers",
            params={"user_id": "user-123", "biomarker_type": "speech"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["meta"]["count"] == 1
        assert data["data"][0]["biomarker_type"] == "speech"

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_get_biomarkers_default_time_range(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test default time range is last 24 hours (AC9)."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed data: one within 24h, one older
        now = datetime.now(UTC)
        recent_biomarker = Biomarker(
            user_id="user-123",
            timestamp=now - timedelta(hours=12),
            biomarker_type="speech",
            value={"pitch": 120.5},
        )
        old_biomarker = Biomarker(
            user_id="user-123",
            timestamp=now - timedelta(days=2),
            biomarker_type="speech",
            value={"pitch": 110.0},
        )
        db_session.add_all([recent_biomarker, old_biomarker])
        db_session.commit()

        # Request without time range - should default to last 24h
        response = client.get(
            "/biomarkers",
            params={"user_id": "user-123"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        # Should only get the recent one
        assert data["meta"]["count"] == 1

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_get_biomarkers_ordered_by_timestamp(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test results are ordered by timestamp ascending (AC4, NFR3)."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed data in non-chronological order
        now = datetime.now(UTC)
        biomarker3 = Biomarker(
            user_id="user-123",
            timestamp=now - timedelta(hours=1),
            biomarker_type="speech",
            value={"pitch": 125.0},
        )
        biomarker1 = Biomarker(
            user_id="user-123",
            timestamp=now - timedelta(hours=3),
            biomarker_type="speech",
            value={"pitch": 115.0},
        )
        biomarker2 = Biomarker(
            user_id="user-123",
            timestamp=now - timedelta(hours=2),
            biomarker_type="speech",
            value={"pitch": 120.0},
        )
        db_session.add_all([biomarker3, biomarker1, biomarker2])
        db_session.commit()

        response = client.get(
            "/biomarkers",
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
    def test_get_biomarkers_requires_api_key(self, mock_get_settings: MagicMock):
        """Test API key authentication is required (AC5)."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        # Request without Authorization header
        response = client.get(
            "/biomarkers",
            params={"user_id": "user-123"},
        )

        # FastAPI catches missing header as validation error
        assert response.status_code == 422

    @patch("src.api.dependencies.get_settings")
    def test_get_biomarkers_missing_user_id_returns_422(
        self, mock_get_settings: MagicMock
    ):
        """Test missing required user_id returns 422 validation error (AC7)."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        response = client.get(
            "/biomarkers",
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "VALIDATION_ERROR"
        # Check field name in error details
        errors = data["error"]["details"]["errors"]
        assert any("user_id" in err["field"] for err in errors)

    @patch("src.api.dependencies.get_settings")
    def test_get_biomarkers_invalid_timestamp_format_returns_422(
        self, mock_get_settings: MagicMock
    ):
        """Test invalid timestamp format returns 422 with clear error (AC8)."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        response = client.get(
            "/biomarkers",
            params={
                "user_id": "user-123",
                "start_time": "not-a-valid-timestamp",
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "VALIDATION_ERROR"
        errors = data["error"]["details"]["errors"]
        assert any("start_time" in err["field"] for err in errors)

    @patch("src.api.dependencies.get_settings")
    def test_get_biomarkers_empty_results(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test empty results when no data matches filters."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        response = client.get(
            "/biomarkers",
            params={"user_id": "nonexistent-user"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []
        assert data["meta"]["count"] == 0

        app.dependency_overrides.clear()
