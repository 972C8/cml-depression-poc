"""Tests for API exception handlers."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestJSONDecodeError:
    """Tests for invalid JSON error handling."""

    @patch("src.api.dependencies.get_settings")
    def test_invalid_json_returns_400_with_error_structure(self, mock_get_settings):
        """Test that invalid JSON returns 400 with structured error (AC1, AC5)."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        # Send invalid JSON (malformed)
        response = client.post(
            "/biomarkers",
            content='{"user_id": "user-123", "invalid": }',  # Invalid JSON
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "INVALID_JSON"
        assert data["error"]["message"] == "Request body contains invalid JSON"
        assert "details" in data["error"]
        assert "request_id" in data["error"]["details"]


class TestValidationErrors:
    """Tests for Pydantic validation error handling."""

    @patch("src.api.dependencies.get_settings")
    def test_missing_field_returns_422_with_field_name(self, mock_get_settings):
        """Test that missing required field returns 422 with field name (AC2, AC5)."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        # Missing required field: timestamp
        response = client.post(
            "/biomarkers",
            json={
                "user_id": "user-123",
                "biomarker_type": "speech",
                "value": {"pitch": 120.5},
                # Missing timestamp
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert data["error"]["message"] == "Request validation failed"
        assert "details" in data["error"]
        assert "errors" in data["error"]["details"]
        assert any("timestamp" in err["field"] for err in data["error"]["details"]["errors"])
        assert "request_id" in data["error"]["details"]

    @patch("src.api.dependencies.get_settings")
    def test_invalid_type_returns_422_with_expected_type(self, mock_get_settings):
        """Test that invalid data type returns 422 with expected type (AC3, AC5)."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        # Invalid type: value should be dict
        response = client.post(
            "/biomarkers",
            json={
                "user_id": "user-123",
                "timestamp": "2025-12-16T10:30:00Z",
                "biomarker_type": "speech",
                "value": "invalid-not-a-dict",  # Should be dict
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert "details" in data["error"]
        assert "errors" in data["error"]["details"]
        errors = data["error"]["details"]["errors"]
        assert any("value" in err["field"] for err in errors)

    @patch("src.api.dependencies.get_settings")
    def test_invalid_timestamp_format_returns_422(self, mock_get_settings):
        """Test that invalid timestamp format returns 422 with error message (AC4, AC5)."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        # Invalid timestamp format
        response = client.post(
            "/biomarkers",
            json={
                "user_id": "user-123",
                "timestamp": "not-a-valid-timestamp",
                "biomarker_type": "speech",
                "value": {"pitch": 120.5},
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert "details" in data["error"]
        assert "errors" in data["error"]["details"]
        errors = data["error"]["details"]["errors"]
        assert any("timestamp" in err["field"] for err in errors)


class TestRequestID:
    """Tests for request_id in error responses and headers."""

    @patch("src.api.dependencies.get_settings")
    def test_request_id_present_in_error_responses(self, mock_get_settings):
        """Test that request_id is included in error responses (AC6)."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        response = client.post(
            "/biomarkers",
            content='{"invalid": }',  # Invalid JSON
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
        )

        data = response.json()
        assert "error" in data
        assert "details" in data["error"]
        assert "request_id" in data["error"]["details"]
        # Should be a UUID format
        request_id = data["error"]["details"]["request_id"]
        assert len(request_id) == 36  # UUID format length

    @patch("src.api.dependencies.get_settings")
    def test_x_request_id_header_present(self, mock_get_settings):
        """Test that X-Request-ID header is present in responses (AC6)."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        response = client.post(
            "/biomarkers",
            content='{"invalid": }',  # Invalid JSON
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
        )

        assert "x-request-id" in response.headers
        request_id = response.headers["x-request-id"]
        assert len(request_id) == 36  # UUID format length


class TestConsistentErrorFormat:
    """Tests for consistent error format across all endpoints."""

    @patch("src.api.dependencies.get_settings")
    def test_missing_auth_header_returns_validation_error(self, mock_get_settings):
        """Test that missing auth header returns validation error (AC7)."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        # Missing Authorization header - FastAPI catches this as validation error
        response = client.post(
            "/biomarkers",
            json={
                "user_id": "user-123",
                "timestamp": "2025-12-16T10:30:00Z",
                "biomarker_type": "speech",
                "value": {"pitch": 120.5},
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert any(
            "Authorization" in err["field"]
            for err in data["error"]["details"]["errors"]
        )

    @patch("src.api.dependencies.get_settings")
    def test_invalid_api_key_uses_standard_format(self, mock_get_settings):
        """Test that invalid API key error uses standard format (AC7)."""
        mock_get_settings.return_value.api_keys = ["test-key"]

        response = client.post(
            "/biomarkers",
            json={
                "user_id": "user-123",
                "timestamp": "2025-12-16T10:30:00Z",
                "biomarker_type": "speech",
                "value": {"pitch": 120.5},
            },
            headers={"Authorization": "Bearer wrong-key"},
        )

        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "INVALID_API_KEY"
        assert data["error"]["message"] == "Invalid API key"
        assert "details" in data["error"]
        assert "request_id" in data["error"]["details"]
