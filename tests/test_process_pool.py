"""Tests for the ProcessPool — per-thread long-lived Claude process registry."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from bender.claude_process import ClaudeProcessError
from bender.process_pool import ProcessPool
from bender.session_manager import SessionManager


def _mock_codex_process(session_id: str = "new-codex-id") -> AsyncMock:
    """A fake CodexProcess instance with canned start/send/close."""
    instance = AsyncMock()
    instance.is_alive = True
    instance.session_id = session_id
    instance.start = AsyncMock()
    instance.send = AsyncMock(return_value="codex reply")
    instance.close = AsyncMock()
    return instance


@pytest.fixture
def sessions(tmp_path: Path) -> SessionManager:
    return SessionManager(db_path=tmp_path / "sessions.sqlite3")


@pytest.fixture
def pool(tmp_path: Path, sessions: SessionManager) -> ProcessPool:
    return ProcessPool(workspace=tmp_path, sessions=sessions, idle_timeout=999, reap_interval=999)


def _mock_claude_process(session_id: str = "new-id") -> AsyncMock:
    """A fake ClaudeProcess instance with canned start/send/close."""
    instance = AsyncMock()
    instance.is_alive = True
    instance.session_id = session_id
    instance.start = AsyncMock()
    instance.send = AsyncMock(return_value="reply")
    instance.close = AsyncMock()
    return instance


class TestSend:
    async def test_first_message_starts_a_fresh_process(
        self, pool: ProcessPool, sessions: SessionManager
    ) -> None:
        """No prior session -> starts a new (non-resumed) process."""
        instance = _mock_claude_process(session_id="fresh-id")
        with patch("bender.process_pool.ClaudeProcess", return_value=instance) as ctor:
            result = await pool.send("thread-1", "hello")

        assert result == "reply"
        ctor.assert_called_once_with(workspace=pool._workspace, session_id=None)
        instance.start.assert_called_once_with(resume=False)
        assert await sessions.get_session("thread-1") == "fresh-id"

    async def test_second_message_same_thread_reuses_live_process(
        self, pool: ProcessPool
    ) -> None:
        """A second send() for the same thread does not spawn a second process."""
        instance = _mock_claude_process()
        with patch("bender.process_pool.ClaudeProcess", return_value=instance) as ctor:
            await pool.send("thread-1", "first")
            await pool.send("thread-1", "second")

        ctor.assert_called_once()
        assert instance.send.await_count == 2

    async def test_persisted_session_resumed_after_pool_restart(
        self, pool: ProcessPool, sessions: SessionManager
    ) -> None:
        """If a session was persisted but no live process exists, resume it."""
        await sessions.set_session("thread-2", "old-session-id")
        instance = _mock_claude_process(session_id="old-session-id")

        with patch("bender.process_pool.ClaudeProcess", return_value=instance) as ctor:
            await pool.send("thread-2", "continue please")

        ctor.assert_called_once_with(workspace=pool._workspace, session_id="old-session-id")
        instance.start.assert_called_once_with(resume=True)

    async def test_dead_process_error_drops_it_from_pool(self, pool: ProcessPool) -> None:
        """A ClaudeProcessError during send() removes the process from the pool
        so the next call starts a fresh one instead of reusing a dead handle."""
        instance = _mock_claude_process()
        instance.send.side_effect = ClaudeProcessError("crashed")

        with patch("bender.process_pool.ClaudeProcess", return_value=instance) as ctor:
            with pytest.raises(ClaudeProcessError):
                await pool.send("thread-3", "hello")
            assert "thread-3" not in pool._processes

            # next call spawns a new process rather than reusing the dead one
            instance2 = _mock_claude_process()
            ctor.return_value = instance2
            await pool.send("thread-3", "retry")

        assert ctor.call_count == 2

    async def test_independent_threads_get_independent_processes(
        self, pool: ProcessPool
    ) -> None:
        instance_a = _mock_claude_process(session_id="a")
        instance_b = _mock_claude_process(session_id="b")

        with patch(
            "bender.process_pool.ClaudeProcess", side_effect=[instance_a, instance_b]
        ):
            await pool.send("thread-a", "hi")
            await pool.send("thread-b", "hi")

        assert pool._processes["thread-a"] is instance_a
        assert pool._processes["thread-b"] is instance_b


class TestBackendSelection:
    def test_rejects_unknown_backend(self, tmp_path: Path, sessions: SessionManager) -> None:
        with pytest.raises(ValueError, match="backend"):
            ProcessPool(workspace=tmp_path, sessions=sessions, backend="gpt5")

    def test_defaults_to_claude_backend(self, tmp_path: Path, sessions: SessionManager) -> None:
        pool = ProcessPool(workspace=tmp_path, sessions=sessions)
        assert pool._backend == "claude"

    async def test_claude_backend_constructs_claude_process(
        self, tmp_path: Path, sessions: SessionManager
    ) -> None:
        pool = ProcessPool(workspace=tmp_path, sessions=sessions, backend="claude")
        instance = _mock_claude_process()
        with patch("bender.process_pool.ClaudeProcess", return_value=instance) as ctor, patch(
            "bender.process_pool.CodexProcess"
        ) as codex_ctor:
            await pool.send("thread-1", "hi")

        ctor.assert_called_once()
        codex_ctor.assert_not_called()

    async def test_codex_backend_constructs_codex_process(
        self, tmp_path: Path, sessions: SessionManager
    ) -> None:
        pool = ProcessPool(workspace=tmp_path, sessions=sessions, backend="codex")
        instance = _mock_codex_process()
        with patch("bender.process_pool.CodexProcess", return_value=instance) as ctor, patch(
            "bender.process_pool.ClaudeProcess"
        ) as claude_ctor:
            result = await pool.send("thread-1", "hi")

        assert result == "codex reply"
        ctor.assert_called_once_with(workspace=pool._workspace, session_id=None)
        claude_ctor.assert_not_called()
        assert await sessions.get_session("thread-1") == "new-codex-id"


class TestReaper:
    async def test_reap_idle_closes_and_removes_stale_processes(
        self, tmp_path: Path, sessions: SessionManager
    ) -> None:
        """Processes idle past idle_timeout get closed on the next reap pass."""
        pool = ProcessPool(workspace=tmp_path, sessions=sessions, idle_timeout=0, reap_interval=999)
        instance = _mock_claude_process()

        with patch("bender.process_pool.ClaudeProcess", return_value=instance):
            await pool.send("thread-1", "hi")

        await pool._reap_idle()

        instance.close.assert_called_once()
        assert "thread-1" not in pool._processes

    async def test_reap_idle_leaves_recently_used_processes(self, pool: ProcessPool) -> None:
        """A pool with a long idle_timeout does not reap a just-used process."""
        instance = _mock_claude_process()
        with patch("bender.process_pool.ClaudeProcess", return_value=instance):
            await pool.send("thread-1", "hi")

        await pool._reap_idle()

        instance.close.assert_not_called()
        assert "thread-1" in pool._processes


class TestStop:
    async def test_stop_closes_all_live_processes(self, pool: ProcessPool) -> None:
        instance_a = _mock_claude_process()
        instance_b = _mock_claude_process()
        with patch(
            "bender.process_pool.ClaudeProcess", side_effect=[instance_a, instance_b]
        ):
            await pool.send("thread-a", "hi")
            await pool.send("thread-b", "hi")

        await pool.stop()

        instance_a.close.assert_called_once()
        instance_b.close.assert_called_once()
        assert pool._processes == {}

    async def test_stop_cancels_reap_task(self, pool: ProcessPool) -> None:
        pool.start_reaper()
        assert pool._reap_task is not None
        await pool.stop()
        assert pool._reap_task is None
