"""Pool of long-lived conversation-thread backends, one per active Slack
thread.

Idle processes are reaped after `idle_timeout` so resources aren't held
forever. The underlying session survives on disk regardless
(SessionManager persists thread_ts -> session_id), so a reaped thread
simply pays the startup cost again on its next message, exactly like a
Bender restart would.

`backend` selects which CLI a given ProcessPool (and therefore a given
Bender instance) drives -- "claude" (ClaudeProcess, long-lived
stream-json subprocess) or "codex" (CodexProcess, long-lived `codex
app-server` connection). One Bender process only ever runs one
backend; running both means running two separate Bender instances
(two Slack apps, two .env files, two ports).
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Protocol

from bender.claude_process import ClaudeProcess
from bender.codex_process import CodexProcess
from bender.errors import ProcessError
from bender.session_manager import SessionManager

logger = logging.getLogger(__name__)

DEFAULT_IDLE_TIMEOUT_SECONDS = 15 * 60
DEFAULT_REAP_INTERVAL_SECONDS = 60
VALID_BACKENDS = ("claude", "codex")


class ThreadBackend(Protocol):
    """Structural interface both ClaudeProcess and CodexProcess satisfy."""

    session_id: str | None

    async def start(self, resume: bool = False) -> None: ...

    @property
    def is_alive(self) -> bool: ...

    async def send(self, prompt: str) -> str: ...

    async def close(self) -> None: ...


class ProcessPool:
    """Thread-safe registry of live ThreadBackend instances keyed by
    Slack thread_ts, with idle reaping and crash recovery."""

    def __init__(
        self,
        workspace: Path,
        sessions: SessionManager,
        backend: str = "claude",
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
        reap_interval: float = DEFAULT_REAP_INTERVAL_SECONDS,
    ) -> None:
        if backend not in VALID_BACKENDS:
            raise ValueError(f"backend must be one of {VALID_BACKENDS}, got {backend!r}")
        self._workspace = workspace
        self._sessions = sessions
        self._backend = backend
        self._idle_timeout = idle_timeout
        self._reap_interval = reap_interval
        self._processes: dict[str, ThreadBackend] = {}
        self._last_used: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._thread_locks: dict[str, asyncio.Lock] = {}
        self._reap_task: asyncio.Task | None = None

    def start_reaper(self) -> None:
        """Start the background idle-reaper loop. No-op if already running."""
        if self._reap_task is None:
            self._reap_task = asyncio.create_task(self._reap_loop())

    async def stop(self) -> None:
        """Cancel the reaper and close every live process."""
        if self._reap_task is not None:
            self._reap_task.cancel()
            self._reap_task = None
        async with self._lock:
            processes = list(self._processes.values())
            self._processes.clear()
            self._last_used.clear()
        for proc in processes:
            await proc.close()

    async def send(self, thread_ts: str, prompt: str) -> str:
        """Route one Slack message to the thread's live process, starting
        one (fresh or resumed) if none is currently running."""
        proc = await self._get_or_start(thread_ts)
        # Known before send() only for a resumed process -- a fresh one's
        # session_id is None until a successful turn assigns it. Needed
        # below to tell "this resume attempt is what just failed" apart
        # from "this thread never had a persisted session to begin with".
        was_resumed = proc.session_id is not None
        try:
            result = await proc.send(prompt)
        except ProcessError as exc:
            # The live process died mid-turn. Drop it so the next message
            # starts a fresh process instead of reusing the dead handle.
            logger.warning(
                "Evicting process for thread %s after error: %s", thread_ts, exc
            )
            async with self._lock:
                self._processes.pop(thread_ts, None)
                self._last_used.pop(thread_ts, None)
            if was_resumed:
                # Dropping the in-memory process alone isn't enough here:
                # the persisted session_id it was resuming is still on
                # disk, so the *next* message would try to resume that
                # exact same session and hit the exact same failure --
                # observed live as a session whose app-server connection
                # dropped mid-turn becoming permanently unresumable,
                # failing every retry in under a second. Clearing the
                # mapping lets the next message start fresh instead.
                await self._sessions.clear_session(thread_ts)
            raise
        if proc.session_id:
            await self._sessions.set_session(thread_ts, proc.session_id)
        async with self._lock:
            self._last_used[thread_ts] = time.monotonic()
        return result

    async def _get_or_start(self, thread_ts: str) -> ThreadBackend:
        # Slack delivers both an `app_mention` and a `message` event for
        # one physical message that's a threaded reply containing a
        # mention, and slack_handler.py dispatches both to pool.send().
        # Without per-thread serialization here, both concurrent calls
        # would see "no live process yet" under the lock below, release
        # it, and each build + start their own backend connection
        # resuming the same persisted session_id -- the codex app-server
        # then rejects the second concurrent resume with "thread not
        # found", and the loser of the self._processes[thread_ts] write
        # race becomes an orphaned, untracked process that keeps running
        # its own send() until it independently hits the turn timeout
        # (observed live: a "thread not found" failure immediately
        # followed ten minutes later by an unrelated-looking timeout for
        # the same thread). Holding one lock per thread_ts across the
        # entire check-create-register sequence makes the second caller
        # simply wait for the first's result instead of racing it.
        async with self._lock:
            thread_lock = self._thread_locks.setdefault(thread_ts, asyncio.Lock())

        async with thread_lock:
            existing = self._processes.get(thread_ts)
            if existing is not None and existing.is_alive:
                async with self._lock:
                    self._last_used[thread_ts] = time.monotonic()
                return existing

            existing_session_id = await self._sessions.get_session(thread_ts)
            if self._backend == "codex":
                proc: ThreadBackend = CodexProcess(
                    workspace=self._workspace, session_id=existing_session_id, thread_ts=thread_ts
                )
            else:
                proc = ClaudeProcess(
                    workspace=self._workspace, session_id=existing_session_id, thread_ts=thread_ts
                )
            await proc.start(resume=existing_session_id is not None)

            async with self._lock:
                self._processes[thread_ts] = proc
                self._last_used[thread_ts] = time.monotonic()
            return proc

    async def _reap_loop(self) -> None:
        while True:
            await asyncio.sleep(self._reap_interval)
            await self._reap_idle()

    async def _reap_idle(self) -> None:
        now = time.monotonic()
        async with self._lock:
            stale = [
                thread_ts
                for thread_ts, last in self._last_used.items()
                if now - last > self._idle_timeout
            ]
            to_close = [(ts, self._processes.pop(ts)) for ts in stale if ts in self._processes]
            for thread_ts, _ in to_close:
                self._last_used.pop(thread_ts, None)
                # Safe to drop even if a coroutine is mid-wait on this
                # Lock object: dropping it from the dict only affects
                # future _get_or_start calls, which will create a fresh
                # Lock via setdefault; anyone already holding a reference
                # to this one is unaffected.
                self._thread_locks.pop(thread_ts, None)
        for thread_ts, proc in to_close:
            logger.info("Reaping idle Claude process for thread %s", thread_ts)
            await proc.close()
