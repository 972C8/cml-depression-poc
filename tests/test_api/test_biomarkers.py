"""Tests for biomarkers API endpoints."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.main import app
from src.shared.database import get_db

client = TestClient(app)


class TestCreateBiomarker:
    """Tests for POST /biomarkers endpoint."""

    @patch("src.api.dependencies.get_settings")
    def test_create_speech_biomarker_returns_201(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test successful creation of speech biomarker returns 201."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        response = client.post(
            "/biomarkers",
            json={
                "user_id": "user-123",
                "timestamp": "2025-12-16T10:30:00Z",
                "biomarker_type": "speech",
                "value": {"pitch": 120.5, "tempo": 85.2},
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 201
        data = response.json()
        assert "data" in data
        assert data["data"]["biomarker_type"] == "speech"
        assert data["data"]["user_id"] == "user-123"
        assert data["data"]["value"] == {"pitch": 120.5, "tempo": 85.2}
        assert "id" in data["data"]
        assert "created_at" in data["data"]

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_create_network_biomarker_returns_201(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test successful creation of network biomarker returns 201."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        response = client.post(
            "/biomarkers",
            json={
                "user_id": "user-456",
                "timestamp": "2025-12-16T11:00:00Z",
                "biomarker_type": "network",
                "value": {"connections": 15, "activity_score": 0.75},
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["data"]["biomarker_type"] == "network"

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_create_biomarker_with_metadata(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test biomarker creation with optional metadata."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        response = client.post(
            "/biomarkers",
            json={
                "user_id": "user-789",
                "timestamp": "2025-12-16T12:00:00Z",
                "biomarker_type": "speech",
                "value": {"data": 1},
                "metadata": {"source": "mobile-app", "version": "1.0"},
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["data"]["metadata"] == {"source": "mobile-app", "version": "1.0"}

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_invalid_biomarker_type_returns_422(self, mock_get_settings: MagicMock):
        """Test invalid biomarker type returns 422 validation error."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        response = client.post(
            "/biomarkers",
            json={
                "user_id": "user-123",
                "timestamp": "2025-12-16T10:30:00Z",
                "biomarker_type": "invalid_type",
                "value": {"data": 1},
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422

    @patch("src.api.dependencies.get_settings")
    def test_missing_required_fields_returns_422(self, mock_get_settings: MagicMock):
        """Test missing required fields returns 422 validation error."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        # Missing user_id
        response = client.post(
            "/biomarkers",
            json={
                "timestamp": "2025-12-16T10:30:00Z",
                "biomarker_type": "speech",
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
            "/biomarkers",
            json={
                "user_id": "",
                "timestamp": "2025-12-16T10:30:00Z",
                "biomarker_type": "speech",
                "value": {"data": 1},
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422

    def test_missing_api_key_returns_422(self):
        """Test missing API key returns 422 (FastAPI required header)."""
        response = client.post(
            "/biomarkers",
            json={
                "user_id": "user-123",
                "timestamp": "2025-12-16T10:30:00Z",
                "biomarker_type": "speech",
                "value": {"data": 1},
            },
        )

        assert response.status_code == 422

    @patch("src.api.dependencies.get_settings")
    def test_invalid_api_key_returns_401(self, mock_get_settings: MagicMock):
        """Test invalid API key returns 401 Unauthorized."""
        mock_get_settings.return_value.api_keys = ["valid-key"]

        response = client.post(
            "/biomarkers",
            json={
                "user_id": "user-123",
                "timestamp": "2025-12-16T10:30:00Z",
                "biomarker_type": "speech",
                "value": {"data": 1},
            },
            headers={"Authorization": "Bearer wrong-key"},
        )

        assert response.status_code == 401

    @patch("src.api.dependencies.get_settings")
    def test_data_persisted_to_database(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test biomarker data is actually persisted to database."""
        from src.shared.models import Biomarker

        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        # Count before
        count_before = db_session.query(Biomarker).count()

        response = client.post(
            "/biomarkers",
            json={
                "user_id": "persist-test-user",
                "timestamp": "2025-12-16T10:30:00Z",
                "biomarker_type": "speech",
                "value": {"test": "persistence"},
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 201

        # Count after
        count_after = db_session.query(Biomarker).count()
        assert count_after == count_before + 1

        # Verify the data
        created_id = response.json()["data"]["id"]
        from uuid import UUID

        biomarker = db_session.query(Biomarker).filter(
            Biomarker.id == UUID(created_id)
        ).first()
        assert biomarker is not None
        assert biomarker.user_id == "persist-test-user"
        assert biomarker.value == {"test": "persistence"}

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_response_wrapped_in_data_structure(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test response follows ApiResponse structure with data wrapper."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        response = client.post(
            "/biomarkers",
            json={
                "user_id": "user-123",
                "timestamp": "2025-12-16T10:30:00Z",
                "biomarker_type": "speech",
                "value": {"data": 1},
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 201
        data = response.json()
        assert "data" in data
        assert "meta" in data  # Can be null but key should exist

        app.dependency_overrides.clear()


class TestBiomarkerBatch:
    """Tests for POST /biomarkers/batch endpoint."""

    @patch("src.api.dependencies.get_settings")
    def test_batch_create_returns_201(
        self, mock_get_settings: MagicMock, db_session, override_get_db
    ):
        """Test successful batch creation returns 201 with count."""
        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        response = client.post(
            "/biomarkers/batch",
            json={
                "items": [
                    {
                        "user_id": "user-123",
                        "timestamp": "2025-12-16T10:30:00Z",
                        "biomarker_type": "speech",
                        "value": {"pitch": 120.5},
                    },
                    {
                        "user_id": "user-123",
                        "timestamp": "2025-12-16T10:31:00Z",
                        "biomarker_type": "network",
                        "value": {"latency": 50},
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
        from src.shared.models import Biomarker

        mock_get_settings.return_value.api_keys = ["test-key"]
        app.dependency_overrides[get_db] = override_get_db

        count_before = db_session.query(Biomarker).count()

        response = client.post(
            "/biomarkers/batch",
            json={
                "items": [
                    {
                        "user_id": "batch-user",
                        "timestamp": "2025-12-16T10:30:00Z",
                        "biomarker_type": "speech",
                        "value": {"item": 1},
                    },
                    {
                        "user_id": "batch-user",
                        "timestamp": "2025-12-16T10:31:00Z",
                        "biomarker_type": "speech",
                        "value": {"item": 2},
                    },
                    {
                        "user_id": "batch-user",
                        "timestamp": "2025-12-16T10:32:00Z",
                        "biomarker_type": "network",
                        "value": {"item": 3},
                    },
                ]
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 201
        count_after = db_session.query(Biomarker).count()
        assert count_after == count_before + 3

        app.dependency_overrides.clear()

    @patch("src.api.dependencies.get_settings")
    def test_batch_invalid_item_fails_entire_batch(self, mock_get_settings: MagicMock):
        """Test single invalid item fails entire batch."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        response = client.post(
            "/biomarkers/batch",
            json={
                "items": [
                    {
                        "user_id": "user-123",
                        "timestamp": "2025-12-16T10:30:00Z",
                        "biomarker_type": "speech",
                        "value": {"valid": True},
                    },
                    {
                        "user_id": "user-123",
                        "timestamp": "2025-12-16T10:31:00Z",
                        "biomarker_type": "invalid_type",
                        "value": {"invalid": True},
                    },
                ]
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422

    @patch("src.api.dependencies.get_settings")
    def test_batch_error_includes_item_index(self, mock_get_settings: MagicMock):
        """Test validation error includes item index in location."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        response = client.post(
            "/biomarkers/batch",
            json={
                "items": [
                    {
                        "user_id": "user-123",
                        "timestamp": "2025-12-16T10:30:00Z",
                        "biomarker_type": "speech",
                        "value": {"valid": True},
                    },
                    {
                        "user_id": "user-123",
                        "timestamp": "2025-12-16T10:31:00Z",
                        "biomarker_type": "invalid_type",
                        "value": {"invalid": True},
                    },
                ]
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422
        data = response.json()
        errors = data["error"]["details"]["errors"]
        # Verify error includes item index information
        assert any("items" in err["field"] for err in errors)
        assert any("1" in err["field"] for err in errors)

    @patch("src.api.dependencies.get_settings")
    def test_batch_size_limit_exceeded_returns_422(self, mock_get_settings: MagicMock):
        """Test batch exceeding 1000 items is rejected."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        items = [
            {
                "user_id": f"user-{i}",
                "timestamp": "2025-12-16T10:30:00Z",
                "biomarker_type": "speech",
                "value": {"index": i},
            }
            for i in range(1001)
        ]

        response = client.post(
            "/biomarkers/batch",
            json={"items": items},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422

    @patch("src.api.dependencies.get_settings")
    def test_empty_batch_returns_422(self, mock_get_settings: MagicMock):
        """Test empty batch is rejected."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        response = client.post(
            "/biomarkers/batch",
            json={"items": []},
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422

    @patch("src.api.dependencies.get_settings")
    def test_batch_requires_api_key(self, mock_get_settings: MagicMock):
        """Test batch endpoint requires API key authentication."""
        mock_get_settings.return_value.api_keys = ["valid-key"]

        response = client.post(
            "/biomarkers/batch",
            json={
                "items": [
                    {
                        "user_id": "user-123",
                        "timestamp": "2025-12-16T10:30:00Z",
                        "biomarker_type": "speech",
                        "value": {"data": 1},
                    }
                ]
            },
            headers={"Authorization": "Bearer wrong-key"},
        )

        assert response.status_code == 401
