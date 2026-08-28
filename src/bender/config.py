"""Configuration module — loads and validates environment variables."""

import logging
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Required: Slack tokens
    slack_bot_token: str
    slack_app_token: str

    # Required: Claude Code authentication (at least one)
    anthropic_api_key: str | None = None
    claude_code_oauth_token: str | None = None

    # Optional
    bender_workspace: Path = Path.cwd()
    bender_api_port: int = 8080
    bender_db_path: Path = Path.cwd() / "bender_sessions.sqlite3"
    log_level: str = "info"

    # Optional: API key for authenticating external HTTP requests
    bender_api_key: str | None = None

    model_config = {"case_sensitive": False}

    def validate_auth(self) -> None:
        """Ensure at least one Claude Code authentication method is configured."""
        if not self.anthropic_api_key and not self.claude_code_oauth_token:
            raise ValueError(
                "At least one authentication method is required: "
                "ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN"
            )


def configure_logging(level: str) -> None:
    """Configure application-wide logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_settings() -> Settings:
    """Load settings from environment, validate, and configure logging."""
    settings = Settings()
    settings.validate_auth()
    configure_logging(settings.log_level)
    return settings
