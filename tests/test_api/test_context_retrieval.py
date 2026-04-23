"""Tests for context retrieval endpoint."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.shared.database import get_db
from src.shared.models import Context

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_contexts(db_session):
    """Clean contexts table before each test."""
    db_session.query(Context).delete()
    db_session.commit()


class TestGetContext:
    """Tests for GET /context endpoint."""

    @patch("src.api.dependencies.get_settings")
    def test_get_context_success(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test successful retrieval with all filters."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed test data
        now = datetime.now(UTC)
        context1 = Context(
            user_id="user-123",
            timestamp=now - timedelta(hours=2),
            context_type="location",
            value={"lat": 37.7749, "lon": -122.4194},
        )
        context2 = Context(
            user_id="user-123",
            timestamp=now - timedelta(hours=1),
            context_type="activity",
            value={"activity": "walking"},
        )
        db_session.add_all([context1, context2])
        db_session.commit()

        response = client.get(
            "/context",
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
    def test_get_context_user_id_filter(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test filtering by user_id (required parameter)."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed data for multiple users
        now = datetime.now(UTC)
        context1 = Context(
            user_id="user-123",
            timestamp=now,
            context_type="location",
            value={},
        )
        context2 = Context(
            user_id="user-456",
            timestamp=now,
            context_type="location",
            value={},
        )
        db_session.add_all([context1, context2])
        db_session.commit()

        response = client.get(
            "/context",
            params={"user_id": "user-123"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["meta"]["count"] == 1
        assert data["data"][0]["user_id"] == "user-123"

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_get_context_time_range_filters(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test start_time and end_time filters."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed data across different times
        now = datetime.now(UTC)
        ctx_old = Context(
            user_id="user-123",
            timestamp=now - timedelta(hours=10),
            context_type="location",
            value={},
        )
        ctx_recent = Context(
            user_id="user-123",
            timestamp=now - timedelta(hours=2),
            context_type="location",
            value={},
        )
        db_session.add_all([ctx_old, ctx_recent])
        db_session.commit()

        # Request with specific time range (last 3 hours)
        response = client.get(
            "/context",
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
    def test_get_context_filtered_by_type(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test filtering by context_type."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed test data with multiple types
        now = datetime.now(UTC)
        location_ctx = Context(
            user_id="user-123",
            timestamp=now,
            context_type="location",
            value={"lat": 37.7749, "lon": -122.4194},
        )
        activity_ctx = Context(
            user_id="user-123",
            timestamp=now,
            context_type="activity",
            value={"activity": "walking"},
        )
        db_session.add_all([location_ctx, activity_ctx])
        db_session.commit()

        response = client.get(
            "/context",
            params={"user_id": "user-123", "context_type": "location"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["meta"]["count"] == 1
        assert data["data"][0]["context_type"] == "location"

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_get_context_default_time_range(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test default time range is last 24 hours."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed data: one within 24h, one older
        now = datetime.now(UTC)
        recent_ctx = Context(
            user_id="user-123",
            timestamp=now - timedelta(hours=12),
            context_type="location",
            value={},
        )
        old_ctx = Context(
            user_id="user-123",
            timestamp=now - timedelta(days=2),
            context_type="location",
            value={},
        )
        db_session.add_all([recent_ctx, old_ctx])
        db_session.commit()

        # Request without time range - should default to last 24h
        response = client.get(
            "/context",
            params={"user_id": "user-123"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        # Should only get the recent one
        assert data["meta"]["count"] == 1

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_get_context_ordered_by_timestamp(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test results are ordered by timestamp ascending (NFR3)."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Seed data in non-chronological order
        now = datetime.now(UTC)
        ctx3 = Context(
            user_id="user-123",
            timestamp=now - timedelta(hours=1),
            context_type="location",
            value={},
        )
        ctx1 = Context(
            user_id="user-123",
            timestamp=now - timedelta(hours=3),
            context_type="location",
            value={},
        )
        ctx2 = Context(
            user_id="user-123",
            timestamp=now - timedelta(hours=2),
            context_type="location",
            value={},
        )
        db_session.add_all([ctx3, ctx1, ctx2])
        db_session.commit()

        response = client.get(
            "/context",
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
    def test_get_context_requires_api_key(self, mock_get_settings: MagicMock):
        """Test API key authentication is required."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        # Request without Authorization header
        response = client.get(
            "/context",
            params={"user_id": "user-123"},
        )

        assert response.status_code == 422  # Missing header caught by validation

    @patch("src.api.dependencies.get_settings")
    def test_get_context_missing_user_id_returns_422(
        self, mock_get_settings: MagicMock
    ):
        """Test missing required user_id returns 422 validation error."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        response = client.get(
            "/context",
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
    def test_get_context_invalid_timestamp_format_returns_422(
        self, mock_get_settings: MagicMock
    ):
        """Test invalid timestamp format returns 422 with clear error message."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        response = client.get(
            "/context",
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
    def test_get_context_empty_results(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test empty results when no data matches filters."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        response = client.get(
            "/context",
            params={"user_id": "nonexistent-user"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []
        assert data["meta"]["count"] == 0

        app.dependency_overrides.clear()
