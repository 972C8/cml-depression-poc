from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "mt_poc"
    postgres_password: SecretStr = SecretStr("changeme")
    postgres_db: str = "mt_poc"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_keys_raw: str = Field(default="", alias="API_KEYS")

    @property
    def api_keys(self) -> list[str]:
        """Parse comma-separated API keys string into list."""
        if not self.api_keys_raw:
            return []
        return [k.strip() for k in self.api_keys_raw.split(",") if k.strip()]

    # Dashboard
    streamlit_port: int = 8501

    # CORS
    cors_allowed_origins_raw: str = Field(
        default="http://localhost:8501", alias="CORS_ALLOWED_ORIGINS"
    )

    @property
    def cors_allowed_origins(self) -> list[str]:
        """Parse comma-separated CORS origins string into list.

        Returns:
            List of allowed origins. Returns ["*"] if "*" is specified.
        """
        if not self.cors_allowed_origins_raw:
            return ["http://localhost:8501"]
        origins = [o.strip() for o in self.cors_allowed_origins_raw.split(",") if o.strip()]
        return origins if origins else ["http://localhost:8501"]

    @property
    def database_url(self) -> str:
        """Build PostgreSQL connection URL."""
        password = self.postgres_password.get_secret_value()
        return (
            f"postgresql://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def __repr__(self) -> str:
        """Hide sensitive values in repr."""
        return f"Settings(postgres_host={self.postgres_host}, postgres_db={self.postgres_db})"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
