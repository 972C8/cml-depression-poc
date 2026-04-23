"""Tests for shared configuration module."""

import os
from unittest.mock import patch

from src.shared.config import Settings, get_settings


class TestSettings:
    """Tests for Settings class."""

    def test_settings_has_database_fields(self):
        """Verify Settings has all database configuration fields."""
        settings = Settings()
        assert hasattr(settings, "postgres_host")
        assert hasattr(settings, "postgres_port")
        assert hasattr(settings, "postgres_user")
        assert hasattr(settings, "postgres_password")
        assert hasattr(settings, "postgres_db")

    def test_settings_has_api_fields(self):
        """Verify Settings has all API configuration fields."""
        settings = Settings()
        assert hasattr(settings, "api_host")
        assert hasattr(settings, "api_port")
        assert hasattr(settings, "api_keys")

    def test_settings_has_dashboard_fields(self):
        """Verify Settings has dashboard configuration fields."""
        settings = Settings()
        assert hasattr(settings, "streamlit_port")

    def test_settings_default_values(self):
        """Verify Settings has sensible defaults."""
        settings = Settings()
        assert settings.postgres_host == "localhost"
        assert settings.postgres_port == 5432
        assert settings.postgres_user == "mt_poc"
        assert settings.postgres_db == "mt_poc"
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.streamlit_port == 8501

    def test_database_url_property(self):
        """Verify database_url property builds correct connection string."""
        settings = Settings()
        url = settings.database_url
        assert url.startswith("postgresql://")
        assert settings.postgres_user in url
        assert settings.postgres_host in url
        assert str(settings.postgres_port) in url
        assert settings.postgres_db in url

    def test_password_is_secret_str(self):
        """Verify postgres_password uses SecretStr for security."""
        settings = Settings()
        # SecretStr should not reveal value in string representation
        password_repr = repr(settings.postgres_password)
        assert "changeme" not in password_repr
        assert "**" in password_repr

    def test_repr_hides_sensitive_data(self):
        """Verify __repr__ does not expose sensitive information."""
        settings = Settings()
        repr_str = repr(settings)
        # Should not contain password
        assert "changeme" not in repr_str
        assert "password" not in repr_str.lower()
        # Should contain non-sensitive info
        assert "postgres_host" in repr_str
        assert "postgres_db" in repr_str

    @patch.dict(os.environ, {"POSTGRES_HOST": "custom-host", "POSTGRES_PORT": "5433"})
    def test_settings_loads_from_environment(self):
        """Verify Settings loads values from environment variables."""
        # Clear cache to pick up new env vars
        get_settings.cache_clear()
        settings = Settings()
        assert settings.postgres_host == "custom-host"
        assert settings.postgres_port == 5433
        # Reset cache
        get_settings.cache_clear()


class TestApiKeysParsing:
    """Tests for api_keys field parsing."""

    @patch.dict(os.environ, {"API_KEYS": ""}, clear=False)
    def test_api_keys_default_is_empty_list(self):
        """Verify api_keys defaults to empty list when env var is empty."""
        get_settings.cache_clear()
        settings = Settings(_env_file=None)  # Skip .env file
        assert settings.api_keys == []
        get_settings.cache_clear()

    @patch.dict(os.environ, {"API_KEYS": "key1,key2,key3"})
    def test_api_keys_parses_comma_separated_string(self):
        """Verify comma-separated API keys are parsed into list."""
        get_settings.cache_clear()
        settings = Settings()
        assert settings.api_keys == ["key1", "key2", "key3"]
        get_settings.cache_clear()

    @patch.dict(os.environ, {"API_KEYS": " key1 , key2 , key3 "})
    def test_api_keys_trims_whitespace(self):
        """Verify whitespace is trimmed from API keys."""
        get_settings.cache_clear()
        settings = Settings()
        assert settings.api_keys == ["key1", "key2", "key3"]
        get_settings.cache_clear()

    @patch.dict(os.environ, {"API_KEYS": ""})
    def test_api_keys_empty_string_returns_empty_list(self):
        """Verify empty string results in empty list."""
        get_settings.cache_clear()
        settings = Settings()
        assert settings.api_keys == []
        get_settings.cache_clear()

    @patch.dict(os.environ, {"API_KEYS": "single-key"})
    def test_api_keys_single_key_works(self):
        """Verify single API key without comma works."""
        get_settings.cache_clear()
        settings = Settings()
        assert settings.api_keys == ["single-key"]
        get_settings.cache_clear()

    @patch.dict(os.environ, {"API_KEYS": "key1,,key2"})
    def test_api_keys_ignores_empty_entries(self):
        """Verify empty entries from double commas are ignored."""
        get_settings.cache_clear()
        settings = Settings()
        assert settings.api_keys == ["key1", "key2"]
        get_settings.cache_clear()


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_returns_settings_instance(self):
        """Verify get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_is_cached(self):
        """Verify get_settings returns the same cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_get_settings_cache_can_be_cleared(self):
        """Verify cache can be cleared for testing."""
        settings1 = get_settings()
        get_settings.cache_clear()
        settings2 = get_settings()
        # After cache clear, should be different instances
        # (though with same values)
        assert settings1 is not settings2
