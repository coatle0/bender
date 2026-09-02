"""Long-lived Codex CLI subprocess.

Wraps `codex app-server`, a JSON-RPC daemon reached via the
`codex-app-server-sdk` package, instead of spawning a fresh `codex exec`
/ `codex exec resume` process per turn. The app-server subprocess is
spawned once in start() and its MCP servers stay warm for every
subsequent send() in the thread -- verified: first turn ~17.5s
(includes MCP startup), second turn on the same connection ~6.9s.

`codex-app-server-sdk` is AGPL-3.0 licensed (third-party, PyPI). This
was flagged and accepted before adoption.

On Windows, `codex` resolves to an npm `.cmd` shim rather than a native
binary, and the SDK's stdio transport does not resolve PATHEXT the way
a shell does, so the executable name below is `codex.cmd` explicitly,
verified working.
"""

import asyncio
import logging
import time
from pathlib import Path

from codex_app_server_sdk import CodexClient, ThreadConfig
from codex_app_server_sdk.errors import CodexError, CodexTimeoutError

from bender.claude_process import _subprocess_env
from bender.errors import ProcessError

logger = logging.getLogger(__name__)

DEFAULT_TURN_TIMEOUT_SECONDS = 300
CODEX_EXECUTABLE = "codex.cmd"

# Matches the bypassPermissions choice made for the Claude backend: MCP
# tool calls (e.g. the `slack` server) fail closed under any approval
# policy other than "never" -- verified against the real `slack` MCP
# server, both in exec mode (--dangerously-bypass-approvals-and-sandbox)
# and here.
_THREAD_CONFIG = ThreadConfig(approval_policy="never", sandbox="danger-full-access")


class CodexProcessError(ProcessError):
    """Raised when the long-lived Codex app-server connection fails or
    returns no reply."""


class CodexProcess:
    """One long-lived `codex app-server` connection bound to a single
    Codex thread. Call `start()` once, then `send()` for each user turn."""

    def __init__(
        self, workspace: Path, session_id: str | None = None, thread_ts: str | None = None
    ) -> None:
        self.workspace = workspace
        self.session_id = session_id
        # Slack thread timestamp -- included in every log line below so a
        # specific thread's whole lifecycle can be grepped out of the log
        # by one stable id. self.session_id (the Codex thread_id) doesn't
        # work for this: it's None until the first turn's result assigns
        # one, so a turn that never completes has nothing to grep by
        # unless thread_ts is threaded through separately.
        self.thread_ts = thread_ts
        self._client: CodexClient | None = None
        self._lock = asyncio.Lock()

    async def start(self, resume: bool = False) -> None:
        """Spawn the app-server subprocess and complete the JSON-RPC
        handshake. `resume` has no separate code path here -- an
        existing session_id is simply passed as thread_id on the first
        send() (see send()), same as `codex exec resume` did."""
        logger.info(
            "Starting long-lived Codex app-server (thread=%s, session=%s, resume=%s, workspace=%s)",
            self.thread_ts,
            self.session_id,
            resume,
            self.workspace,
        )
        self._client = CodexClient.connect_stdio(
            command=[CODEX_EXECUTABLE, "app-server"],
            cwd=str(self.workspace),
            env=_subprocess_env(),
        )
        await self._client.start()
        await self._client.initialize()

    @property
    def is_alive(self) -> bool:
        return self._client is not None

    async def send(self, prompt: str, timeout: int = DEFAULT_TURN_TIMEOUT_SECONDS) -> str:
        """Send one user turn and wait for the matching reply. Turns are
        serialized: concurrent callers queue behind the lock so two
        messages in the same thread never interleave on the connection."""
        if self._client is None:
            raise CodexProcessError("process not started")

        async with self._lock:
            logger.info(
                "Sending codex app-server turn (thread=%s, session=%s, workspace=%s)",
                self.thread_ts,
                self.session_id,
                self.workspace,
            )
            start = time.monotonic()
            try:
                # The SDK's own inactivity_timeout is a sliding window: it
                # resets on every turn event, not just the final one. A
                # turn stuck in a retry/progress loop keeps producing
                # events and can run for hours without ever tripping it --
                # observed live as a thread wedged for ~44h. wait_for adds
                # a hard cap on top, bounded by wall-clock time from the
                # start of the call regardless of intermediate activity.
                result = await asyncio.wait_for(
                    self._client.chat_once(
                        prompt,
                        thread_id=self.session_id,
                        thread_config=_THREAD_CONFIG,
                        inactivity_timeout=timeout,
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError as exc:
                await self._force_close_after_timeout()
                logger.warning(
                    "Codex turn timed out after %.1fs hard cap (thread=%s, session=%s)",
                    time.monotonic() - start,
                    self.thread_ts,
                    self.session_id,
                )
                raise CodexProcessError(
                    f"Codex turn timed out after {timeout}s (hard cap)"
                ) from exc
            except CodexTimeoutError as exc:
                await self._force_close_after_timeout()
                logger.warning(
                    "Codex turn timed out after %.1fs (thread=%s, session=%s)",
                    time.monotonic() - start,
                    self.thread_ts,
                    self.session_id,
                )
                raise CodexProcessError(f"Codex turn timed out after {timeout}s") from exc
            except CodexError as exc:
                raise CodexProcessError(f"codex app-server turn failed: {exc}") from exc

            self.session_id = result.thread_id
            if not result.final_text:
                raise CodexProcessError("Codex produced no agent_message")
            logger.info(
                "Codex turn completed in %.1fs (thread=%s, session=%s)",
                time.monotonic() - start,
                self.thread_ts,
                self.session_id,
            )
            return result.final_text

    async def _force_close_after_timeout(self) -> None:
        """Tear down a wedged connection after a hard timeout.

        CodexProcessError is a ProcessError, which ProcessPool.send()
        catches by evicting this instance so the thread's next message
        starts a fresh one. Without this, the old app-server subprocess
        (and whatever turn state the SDK still holds for it) would be
        orphaned instead of terminated -- clearing _client also makes
        is_alive report False immediately, rather than leaving the pool
        able to reuse a connection this call just gave up on.
        """
        client, self._client = self._client, None
        if client is None:
            return
        try:
            await client.close()
        except Exception:
            logger.warning("Failed to close wedged Codex client after timeout", exc_info=True)

    async def close(self) -> None:
        """Close the app-server connection, terminating its subprocess."""
        if self._client is None:
            return
        client, self._client = self._client, None
        await client.close()
