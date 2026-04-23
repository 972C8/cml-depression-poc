"""Tests for context API endpoints."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.main import app
from src.shared.database import get_db

client = TestClient(app)


class TestCreateContext:
    """Tests for POST /context endpoint."""

    @patch("src.api.dependencies.get_settings")
    def test_create_location_context_returns_201(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test successful creation of location context returns 201."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        response = client.post(
            "/context",
            json={
                "user_id": "user-123",
                "timestamp": "2025-12-16T10:30:00Z",
                "context_type": "location",
                "value": {"lat": 40.7128, "lng": -74.0060, "accuracy": 10},
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 201
        data = response.json()
        assert "data" in data
        assert data["data"]["context_type"] == "location"
        assert data["data"]["user_id"] == "user-123"
        assert data["data"]["value"] == {"lat": 40.7128, "lng": -74.0060, "accuracy": 10}
        assert "id" in data["data"]
        assert "created_at" in data["data"]

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_create_activity_context_returns_201(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test successful creation of activity context returns 201."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        response = client.post(
            "/context",
            json={
                "user_id": "user-456",
                "timestamp": "2025-12-16T11:00:00Z",
                "context_type": "activity",
                "value": {"type": "walking", "steps": 150, "duration_seconds": 120},
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["data"]["context_type"] == "activity"

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_create_environment_context_returns_201(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test successful creation of environment context returns 201."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        response = client.post(
            "/context",
            json={
                "user_id": "user-789",
                "timestamp": "2025-12-16T12:00:00Z",
                "context_type": "environment",
                "value": {"temperature": 22.5, "humidity": 45, "noise_level": "low"},
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["data"]["context_type"] == "environment"

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_create_context_with_metadata(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test context creation with optional metadata."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        response = client.post(
            "/context",
            json={
                "user_id": "user-123",
                "timestamp": "2025-12-16T12:00:00Z",
                "context_type": "location",
                "value": {"lat": 40.7128, "lng": -74.0060},
                "metadata": {"source": "gps", "accuracy_meters": 5},
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["data"]["metadata"] == {"source": "gps", "accuracy_meters": 5}

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_custom_context_type_accepted(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test that custom context_type values are accepted (flexible string)."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        response = client.post(
            "/context",
            json={
                "user_id": "user-123",
                "timestamp": "2025-12-16T10:30:00Z",
                "context_type": "custom_sensor_data",
                "value": {"sensor_reading": 42},
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["data"]["context_type"] == "custom_sensor_data"

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_missing_required_fields_returns_422(self, mock_get_settings: MagicMock):
        """Test missing required fields returns 422 validation error."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        # Missing user_id
        response = client.post(
            "/context",
            json={
                "timestamp": "2025-12-16T10:30:00Z",
                "context_type": "location",
                "value": {"data": 1},
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422

    @patch("src.api.dependencies.get_settings")
    def test_empty_user_id_returns_422(self, mock_get_settings: MagicMock):
        """Test empty user_id returns 422 validation error."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        response = client.post(
            "/context",
            json={
                "user_id": "",
                "timestamp": "2025-12-16T10:30:00Z",
                "context_type": "location",
                "value": {"data": 1},
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422

    @patch("src.api.dependencies.get_settings")
    def test_empty_context_type_returns_422(self, mock_get_settings: MagicMock):
        """Test empty context_type returns 422 validation error."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        response = client.post(
            "/context",
            json={
                "user_id": "user-123",
                "timestamp": "2025-12-16T10:30:00Z",
                "context_type": "",
                "value": {"data": 1},
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422

    def test_missing_api_key_returns_422(self):
        """Test missing API key returns 422 (FastAPI required header)."""
        response = client.post(
            "/context",
            json={
                "user_id": "user-123",
                "timestamp": "2025-12-16T10:30:00Z",
                "context_type": "location",
                "value": {"data": 1},
            },
        )

        assert response.status_code == 422

    @patch("src.api.dependencies.get_settings")
    def test_invalid_api_key_returns_401(self, mock_get_settings: MagicMock):
        """Test invalid API key returns 401 Unauthorized."""
        mock_get_settings.return_value.api_keys = ["valid-key"]

        response = client.post(
            "/context",
            json={
                "user_id": "user-123",
                "timestamp": "2025-12-16T10:30:00Z",
                "context_type": "location",
                "value": {"data": 1},
            },
            headers={"Authorization": "Bearer wrong-key"},
        )

        assert response.status_code == 401

    @patch("src.api.dependencies.get_settings")
    def test_data_persisted_to_database(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test context data is actually persisted to database."""
        from uuid import UUID

        from src.shared.models import Context

        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Count before
        count_before = db_session.query(Context).count()

        response = client.post(
            "/context",
            json={
                "user_id": "persist-test-user",
                "timestamp": "2025-12-16T10:30:00Z",
                "context_type": "location",
                "value": {"test": "persistence"},
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 201

        # Count after
        count_after = db_session.query(Context).count()
        assert count_after == count_before + 1

        # Verify the data
        created_id = response.json()["data"]["id"]
        context = db_session.query(Context).filter(
            Context.id == UUID(created_id)
        ).first()
        assert context is not None
        assert context.user_id == "persist-test-user"
        assert context.value == {"test": "persistence"}

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_response_wrapped_in_data_structure(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test response follows ApiResponse structure with data wrapper."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        response = client.post(
            "/context",
            json={
                "user_id": "user-123",
                "timestamp": "2025-12-16T10:30:00Z",
                "context_type": "location",
                "value": {"data": 1},
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 201
        data = response.json()
        assert "data" in data
        assert "meta" in data  # Can be null but key should exist

        app.dependency_overrides.clear()


class TestContextBatch:
    """Tests for POST /context/batch endpoint."""

    @patch("src.api.dependencies.get_settings")
    def test_batch_create_returns_201(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test successful batch creation returns 201 with count."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        response = client.post(
            "/context/batch",
            json={
                "items": [
                    {
                        "user_id": "user-123",
                        "timestamp": "2025-12-16T10:30:00Z",
                        "context_type": "location",
                        "value": {"lat": 40.7, "lng": -74.0},
                    },
                    {
                        "user_id": "user-123",
                        "timestamp": "2025-12-16T10:31:00Z",
                        "context_type": "activity",
                        "value": {"type": "walking"},
                    },
                ]
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["data"]["created_count"] == 2

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_batch_all_items_persisted(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test all items in batch are persisted to database."""
        from src.shared.models import Context

        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        count_before = db_session.query(Context).count()

        response = client.post(
            "/context/batch",
            json={
                "items": [
                    {
                        "user_id": "batch-user",
                        "timestamp": "2025-12-16T10:30:00Z",
                        "context_type": "location",
                        "value": {"item": 1},
                    },
                    {
                        "user_id": "batch-user",
                        "timestamp": "2025-12-16T10:31:00Z",
                        "context_type": "activity",
                        "value": {"item": 2},
                    },
                    {
                        "user_id": "batch-user",
                        "timestamp": "2025-12-16T10:32:00Z",
                        "context_type": "environment",
                        "value": {"item": 3},
                    },
                ]
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 201
        count_after = db_session.query(Context).count()
        assert count_after == count_before + 3

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_batch_invalid_item_fails_entire_batch(self, mock_get_settings: MagicMock):
        """Test single invalid item fails entire batch."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        response = client.post(
            "/context/batch",
            json={
                "items": [
                    {
                        "user_id": "user-123",
                        "timestamp": "2025-12-16T10:30:00Z",
                        "context_type": "location",
                        "value": {"valid": True},
                    },
                    {
                        "user_id": "",  # Invalid - empty user_id
                        "timestamp": "2025-12-16T10:31:00Z",
                        "context_type": "activity",
                        "value": {"invalid": True},
                    },
                ]
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422

    @patch("src.api.dependencies.get_settings")
    def test_batch_size_limit_exceeded_returns_422(self, mock_get_settings: MagicMock):
        """Test batch exceeding 1000 items is rejected."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        items = [
            {
                "user_id": f"user-{i}",
                "timestamp": "2025-12-16T10:30:00Z",
                "context_type": "location",
                "value": {"index": i},
            }
            for i in range(1001)
        ]

        response = client.post(
            "/context/batch",
            json={"items": items},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422

    @patch("src.api.dependencies.get_settings")
    def test_empty_batch_returns_422(self, mock_get_settings: MagicMock):
        """Test empty batch is rejected."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        response = client.post(
            "/context/batch",
            json={"items": []},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422

    @patch("src.api.dependencies.get_settings")
    def test_batch_requires_api_key(self, mock_get_settings: MagicMock):
        """Test batch endpoint requires API key authentication."""
        mock_get_settings.return_value.api_keys = ["valid-key"]

        response = client.post(
            "/context/batch",
            json={
                "items": [
                    {
                        "user_id": "user-123",
                        "timestamp": "2025-12-16T10:30:00Z",
                        "context_type": "location",
                        "value": {"data": 1},
                    }
                ]
            },
            headers={"Authorization": "Bearer wrong-key"},
        )

        assert response.status_code == 401
