"""Shared fixtures for Bender test suite."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from bender.config import Settings
from bender.session_manager import SessionManager


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Create a Settings instance with test values."""
    return Settings(
        slack_bot_token="xoxb-test-token",
        slack_app_token="xapp-test-token",
        anthropic_api_key="sk-ant-test-key",
        bender_workspace=tmp_path,
        bender_api_port=9999,
        bender_db_path=tmp_path / "sessions.sqlite3",
        log_level="debug",
    )


@pytest.fixture
def session_manager() -> SessionManager:
    """Create a fresh SessionManager instance."""
    return SessionManager()


@pytest.fixture
def mock_say() -> AsyncMock:
    """Create a mock Slack say function."""
    return AsyncMock()


@pytest.fixture
def mock_slack_client() -> AsyncMock:
    """Create a mock Slack WebClient."""
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ts": "1234567890.123456"})
    return client
