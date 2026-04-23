"""Tests for API dependencies module."""

from unittest.mock import MagicMock, patch

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import verify_api_key

# Create test app with protected endpoint
app = FastAPI()


@app.get("/protected", dependencies=[Depends(verify_api_key)])
def protected_endpoint():
    return {"message": "success"}


client = TestClient(app)


class TestVerifyApiKey:
    """Tests for verify_api_key dependency."""

    @patch("src.api.dependencies.get_settings")
    def test_valid_api_key_succeeds(self, mock_get_settings: MagicMock):
        """Test that valid API key allows request to proceed."""
        mock_settings = MagicMock()
        mock_settings.api_keys = ["valid-key-123"]
        mock_get_settings.return_value = mock_settings

        response = client.get(
            "/protected", headers={"Authorization": "Bearer valid-key-123"}
        )
        assert response.status_code == 200
        assert response.json() == {"message": "success"}

    @patch("src.api.dependencies.get_settings")
    def test_invalid_api_key_returns_401(self, mock_get_settings: MagicMock):
        """Test that invalid API key returns 401 Unauthorized."""
        mock_settings = MagicMock()
        mock_settings.api_keys = ["valid-key-123"]
        mock_get_settings.return_value = mock_settings

        response = client.get(
            "/protected", headers={"Authorization": "Bearer wrong-key"}
        )
        assert response.status_code == 401
        assert "detail" in response.json()
        assert response.json()["detail"]["error"]["code"] == "INVALID_API_KEY"

    def test_missing_authorization_header_returns_422(self):
        """Test that missing Authorization header returns 422 (FastAPI required header)."""
        response = client.get("/protected")
        assert response.status_code == 422  # FastAPI returns 422 for missing required header

    @patch("src.api.dependencies.get_settings")
    def test_malformed_bearer_token_returns_401(self, mock_get_settings: MagicMock):
        """Test that malformed Bearer token returns 401."""
        mock_settings = MagicMock()
        mock_settings.api_keys = ["valid-key-123"]
        mock_get_settings.return_value = mock_settings

        response = client.get("/protected", headers={"Authorization": "Basic invalid"})
        assert response.status_code == 401
        assert "detail" in response.json()
        assert response.json()["detail"]["error"]["code"] == "INVALID_AUTH_FORMAT"

    @patch("src.api.dependencies.get_settings")
    def test_multiple_valid_keys_all_work(self, mock_get_settings: MagicMock):
        """Test that multiple configured API keys all work."""
        mock_settings = MagicMock()
        mock_settings.api_keys = ["key-one", "key-two", "key-three"]
        mock_get_settings.return_value = mock_settings

        # Test first key
        response = client.get(
            "/protected", headers={"Authorization": "Bearer key-one"}
        )
        assert response.status_code == 200

        # Test second key
        response = client.get(
            "/protected", headers={"Authorization": "Bearer key-two"}
        )
        assert response.status_code == 200

        # Test third key
        response = client.get(
            "/protected", headers={"Authorization": "Bearer key-three"}
        )
        assert response.status_code == 200

    @patch("src.api.dependencies.get_settings")
    def test_case_sensitive_key_comparison(self, mock_get_settings: MagicMock):
        """Test that API key comparison is case-sensitive."""
        mock_settings = MagicMock()
        mock_settings.api_keys = ["MySecretKey"]
        mock_get_settings.return_value = mock_settings

        # Exact case should work
        response = client.get(
            "/protected", headers={"Authorization": "Bearer MySecretKey"}
        )
        assert response.status_code == 200

        # Wrong case should fail
        response = client.get(
            "/protected", headers={"Authorization": "Bearer mysecretkey"}
        )
        assert response.status_code == 401

        response = client.get(
            "/protected", headers={"Authorization": "Bearer MYSECRETKEY"}
        )
        assert response.status_code == 401

    @patch("src.api.dependencies.get_settings")
    def test_empty_token_returns_401(self, mock_get_settings: MagicMock):
        """Test that empty token after Bearer returns 401."""
        mock_settings = MagicMock()
        mock_settings.api_keys = ["valid-key"]
        mock_get_settings.return_value = mock_settings

        response = client.get("/protected", headers={"Authorization": "Bearer "})
        assert response.status_code == 401
