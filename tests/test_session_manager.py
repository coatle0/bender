"""Tests for the session manager module."""

from pathlib import Path

from bender.session_manager import SessionManager


class TestSessionManager:
    """Tests for the SessionManager class."""

    async def test_create_session_returns_uuid(self, session_manager: SessionManager) -> None:
        """create_session returns a valid UUID string."""
        session_id = await session_manager.create_session("1234567890.000001")
        assert isinstance(session_id, str)
        assert len(session_id) == 36  # UUID format: 8-4-4-4-12

    async def test_create_session_stores_mapping(self, session_manager: SessionManager) -> None:
        """create_session stores the thread_ts -> session_id mapping."""
        thread_ts = "1234567890.000001"
        session_id = await session_manager.create_session(thread_ts)
        retrieved = await session_manager.get_session(thread_ts)
        assert retrieved == session_id

    async def test_create_session_unique_ids(self, session_manager: SessionManager) -> None:
        """Each call to create_session generates a unique session ID."""
        id1 = await session_manager.create_session("1234567890.000001")
        id2 = await session_manager.create_session("1234567890.000002")
        assert id1 != id2

    async def test_get_session_nonexistent_returns_none(
        self, session_manager: SessionManager
    ) -> None:
        """get_session returns None for unknown thread_ts."""
        result = await session_manager.get_session("nonexistent")
        assert result is None

    async def test_has_session_true(self, session_manager: SessionManager) -> None:
        """has_session returns True for existing sessions."""
        thread_ts = "1234567890.000001"
        await session_manager.create_session(thread_ts)
        assert await session_manager.has_session(thread_ts) is True

    async def test_has_session_false(self, session_manager: SessionManager) -> None:
        """has_session returns False for non-existing sessions."""
        assert await session_manager.has_session("nonexistent") is False

    async def test_set_session_explicit(self, session_manager: SessionManager) -> None:
        """set_session allows explicit mapping of thread_ts to session_id."""
        thread_ts = "1234567890.000001"
        session_id = "explicit-session-id"
        await session_manager.set_session(thread_ts, session_id)

        result = await session_manager.get_session(thread_ts)
        assert result == session_id

    async def test_set_session_overwrites_existing(
        self, session_manager: SessionManager
    ) -> None:
        """set_session overwrites a previously created session."""
        thread_ts = "1234567890.000001"
        await session_manager.create_session(thread_ts)
        new_id = "overwritten-session-id"
        await session_manager.set_session(thread_ts, new_id)

        result = await session_manager.get_session(thread_ts)
        assert result == new_id

    async def test_clear_session_removes_mapping(
        self, session_manager: SessionManager
    ) -> None:
        """clear_session drops a thread's mapping so it reads back as
        unset -- for a persisted session_id that's become permanently
        unresumable, this is what lets the thread's next message start
        fresh instead of retrying the same broken resume forever."""
        thread_ts = "1234567890.000001"
        await session_manager.set_session(thread_ts, "now-broken-session-id")

        await session_manager.clear_session(thread_ts)

        assert await session_manager.get_session(thread_ts) is None
        assert await session_manager.has_session(thread_ts) is False

    async def test_clear_session_nonexistent_thread_is_noop(
        self, session_manager: SessionManager
    ) -> None:
        """Clearing a thread with no mapping must not raise."""
        await session_manager.clear_session("9999999999.999999")  # must not raise

    async def test_multiple_threads_independent(
        self, session_manager: SessionManager
    ) -> None:
        """Different threads maintain independent sessions."""
        ts1 = "1234567890.000001"
        ts2 = "1234567890.000002"
        ts3 = "1234567890.000003"

        id1 = await session_manager.create_session(ts1)
        id2 = await session_manager.create_session(ts2)
        id3 = await session_manager.create_session(ts3)

        assert await session_manager.get_session(ts1) == id1
        assert await session_manager.get_session(ts2) == id2
        assert await session_manager.get_session(ts3) == id3


class TestSessionManagerPersistence:
    """Tests for SQLite-backed persistence across restarts."""

    async def test_mapping_survives_new_instance_same_db_file(self, tmp_path: Path) -> None:
        """A fresh SessionManager pointed at the same db file sees prior mappings."""
        db_path = tmp_path / "sessions.sqlite3"
        thread_ts = "1234567890.000001"

        first = SessionManager(db_path=db_path)
        session_id = await first.create_session(thread_ts)
        first.close()

        second = SessionManager(db_path=db_path)
        try:
            assert await second.get_session(thread_ts) == session_id
            assert await second.has_session(thread_ts) is True
        finally:
            second.close()

    async def test_set_session_survives_restart(self, tmp_path: Path) -> None:
        """Explicit set_session mappings also survive a restart."""
        db_path = tmp_path / "sessions.sqlite3"
        thread_ts = "1234567890.000002"

        first = SessionManager(db_path=db_path)
        await first.set_session(thread_ts, "explicit-session-id")
        first.close()

        second = SessionManager(db_path=db_path)
        try:
            assert await second.get_session(thread_ts) == "explicit-session-id"
        finally:
            second.close()

    async def test_db_file_created_on_disk(self, tmp_path: Path) -> None:
        """A real db_path creates a file on disk, not just an in-memory table."""
        db_path = tmp_path / "sessions.sqlite3"
        manager = SessionManager(db_path=db_path)
        try:
            await manager.create_session("1234567890.000003")
        finally:
            manager.close()
        assert db_path.exists()

    async def test_default_memory_db_does_not_touch_disk(self, tmp_path: Path) -> None:
        """The default (no db_path) constructor stays in-memory."""
        manager = SessionManager()
        try:
            await manager.create_session("1234567890.000004")
        finally:
            manager.close()
        # No file should appear in an otherwise-empty temp dir.
        assert list(tmp_path.iterdir()) == []

    async def test_restart_recovery_matches_bender_reliability_goal(
        self, tmp_path: Path
    ) -> None:
        """End-to-end: create several threads, 'restart', verify all survive."""
        db_path = tmp_path / "sessions.sqlite3"
        threads = {f"restart-thread-{i}": None for i in range(5)}

        first = SessionManager(db_path=db_path)
        for thread_ts in threads:
            threads[thread_ts] = await first.create_session(thread_ts)
        first.close()

        second = SessionManager(db_path=db_path)
        try:
            for thread_ts, expected_id in threads.items():
                assert await second.get_session(thread_ts) == expected_id
        finally:
            second.close()
