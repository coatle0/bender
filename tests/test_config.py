"""Tests for the configuration module."""

import logging
from pathlib import Path

import pytest

from bender.config import Settings, configure_logging, load_settings


class TestSettings:
    """Tests for the Settings class."""

    def test_settings_with_all_values(self, tmp_path: Path) -> None:
        """Settings loads all fields correctly."""
        s = Settings(
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test",
            anthropic_api_key="sk-ant-test",
            bender_workspace=tmp_path,
            bender_api_port=9090,
            log_level="debug",
        )
        assert s.slack_bot_token == "xoxb-test"
        assert s.slack_app_token == "xapp-test"
        assert s.anthropic_api_key == "sk-ant-test"
        assert s.bender_workspace == tmp_path
        assert s.bender_api_port == 9090
        assert s.log_level == "debug"

    def test_settings_defaults(self) -> None:
        """Settings uses correct defaults for optional fields."""
        s = Settings(
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test",
            anthropic_api_key="sk-ant-test",
        )
        assert s.bender_workspace == Path.cwd()
        assert s.bender_api_port == 8080
        assert s.bender_db_path == Path.cwd() / "bender_sessions.sqlite3"
        assert s.bender_backend == "claude"
        assert s.log_level == "info"

    def test_settings_auth_with_api_key(self) -> None:
        """Settings accepts ANTHROPIC_API_KEY as auth method."""
        s = Settings(
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test",
            anthropic_api_key="sk-ant-test",
        )
        s.validate_auth()  # Should not raise

    def test_settings_auth_with_oauth_token(self) -> None:
        """Settings accepts CLAUDE_CODE_OAUTH_TOKEN as auth method."""
        s = Settings(
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test",
            claude_code_oauth_token="oauth-test-token",
        )
        s.validate_auth()  # Should not raise

    def test_settings_auth_with_both(self) -> None:
        """Settings accepts both auth methods simultaneously."""
        s = Settings(
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test",
            anthropic_api_key="sk-ant-test",
            claude_code_oauth_token="oauth-test-token",
        )
        s.validate_auth()  # Should not raise

    def test_settings_auth_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings raises ValueError when no auth method is configured."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        s = Settings(
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test",
        )
        with pytest.raises(ValueError, match="At least one authentication method"):
            s.validate_auth()

    def test_settings_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings loads from environment variables."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-from-env")
        monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-from-env")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
        monkeypatch.setenv("BENDER_API_PORT", "3000")
        monkeypatch.setenv("LOG_LEVEL", "warning")

        s = Settings()
        assert s.slack_bot_token == "xoxb-from-env"
        assert s.slack_app_token == "xapp-from-env"
        assert s.anthropic_api_key == "sk-ant-from-env"
        assert s.bender_api_port == 3000
        assert s.log_level == "warning"

    def test_settings_db_path_from_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Settings loads BENDER_DB_PATH from the environment."""
        custom_path = tmp_path / "custom_sessions.sqlite3"
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("BENDER_DB_PATH", str(custom_path))

        s = Settings()
        assert s.bender_db_path == custom_path

    def test_settings_backend_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings loads BENDER_BACKEND from the environment."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("BENDER_BACKEND", "codex")

        s = Settings()
        assert s.bender_backend == "codex"


class TestConfigureLogging:
    """Tests for the configure_logging function."""

    def test_configure_logging_valid_level(self) -> None:
        """configure_logging sets the correct log level."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        configure_logging("debug")
        assert root_logger.level == logging.DEBUG

    def test_configure_logging_case_insensitive(self) -> None:
        """configure_logging handles mixed-case level strings."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        configure_logging("WARNING")
        assert root_logger.level == logging.WARNING

    def test_configure_logging_invalid_falls_back_to_info(self) -> None:
        """configure_logging falls back to INFO for invalid level strings."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        configure_logging("nonexistent_level")
        assert root_logger.level == logging.INFO


class TestLoadSettings:
    """Tests for the load_settings function."""

    def test_load_settings_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """load_settings returns configured Settings on valid env."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        s = load_settings()
        assert s.slack_bot_token == "xoxb-test"

    def test_load_settings_missing_auth_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """load_settings raises when no auth is provided."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
        # Clear any pre-existing auth env vars
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

        with pytest.raises(ValueError, match="At least one authentication method"):
            load_settings()
