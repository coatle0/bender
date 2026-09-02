"""Session manager — maps Slack threads to Claude Code sessions.

Backed by SQLite so the thread -> session mapping survives process restarts
(the underlying Claude Code sessions already persist to ~/.claude/projects/;
this store just remembers which Slack thread goes with which session_id).
"""

import logging
import sqlite3
import uuid
from asyncio import Lock
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

MEMORY_DB_PATH = ":memory:"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SessionManager:
    """Thread-safe, SQLite-backed mapping between Slack thread timestamps
    and Claude Code session IDs.

    Each Slack thread maps to exactly one Claude Code session, enabling
    multi-turn conversations with context preserved. The mapping is
    persisted to disk by default, so a Bender restart does not orphan
    in-flight threads.
    """

    def __init__(self, db_path: Path | str = MEMORY_DB_PATH) -> None:
        self._db_path = str(db_path)
        self._lock = Lock()
        # A single long-lived connection is safe here: asyncio runs
        # coroutines on one OS thread, and self._lock already serializes
        # every access to it.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        if self._db_path != MEMORY_DB_PATH:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS thread_sessions (
                thread_ts TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        self._conn.commit()
        logger.info("SessionManager storing thread mappings at %s", self._db_path)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    async def create_session(self, thread_ts: str) -> str:
        """Create a new session for a Slack thread.

        Args:
            thread_ts: The Slack thread timestamp identifier.

        Returns:
            The newly generated session ID.
        """
        session_id = str(uuid.uuid4())
        await self._upsert(thread_ts, session_id)
        logger.info("Created session %s for thread %s", session_id, thread_ts)
        return session_id

    async def get_session(self, thread_ts: str) -> str | None:
        """Get the session ID for a Slack thread, if one exists.

        Args:
            thread_ts: The Slack thread timestamp identifier.

        Returns:
            The session ID, or None if no session exists for this thread.
        """
        async with self._lock:
            row = self._conn.execute(
                "SELECT session_id FROM thread_sessions WHERE thread_ts = ?",
                (thread_ts,),
            ).fetchone()
        return row[0] if row else None

    async def has_session(self, thread_ts: str) -> bool:
        """Check whether a Slack thread has an existing session.

        Args:
            thread_ts: The Slack thread timestamp identifier.

        Returns:
            True if the thread has an associated session.
        """
        return await self.get_session(thread_ts) is not None

    async def set_session(self, thread_ts: str, session_id: str) -> None:
        """Explicitly set the session ID for a thread (e.g., from API-created sessions).

        Args:
            thread_ts: The Slack thread timestamp identifier.
            session_id: The Claude Code session ID to associate.
        """
        await self._upsert(thread_ts, session_id)
        logger.info("Set session %s for thread %s", session_id, thread_ts)

    async def clear_session(self, thread_ts: str) -> None:
        """Drop a thread's persisted session mapping.

        For when the persisted session_id itself is the problem: a backend
        can fail to resume a session (transport error, corrupted resume
        state, etc.) in a way that will keep failing identically forever,
        since ProcessPool evicting the in-memory process still leaves this
        same broken session_id on disk for the *next* message to resume
        into. Observed live: a thread's session_id became permanently
        unresumable after its app-server connection dropped mid-turn, and
        every retry failed in under a second with the same transport
        error. Clearing this mapping is what lets the next message start
        a fresh session instead of repeating that failure indefinitely.

        Args:
            thread_ts: The Slack thread timestamp identifier.
        """
        async with self._lock:
            self._conn.execute(
                "DELETE FROM thread_sessions WHERE thread_ts = ?", (thread_ts,)
            )
            self._conn.commit()
        logger.info("Cleared session mapping for thread %s", thread_ts)

    async def _upsert(self, thread_ts: str, session_id: str) -> None:
        async with self._lock:
            self._conn.execute(
                """INSERT INTO thread_sessions (thread_ts, session_id, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(thread_ts) DO UPDATE SET
                     session_id = excluded.session_id,
                     updated_at = excluded.updated_at""",
                (thread_ts, session_id, _utc_now_iso()),
            )
            self._conn.commit()
