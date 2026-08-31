"""Long-lived Claude Code subprocess.

Wraps `claude --print --input-format stream-json --output-format
stream-json`, which stays running and accepts one JSON line per user turn
on stdin instead of exiting after a single reply. Keeping the process
alive across a Slack thread's turns means MCP servers only start once per
thread instead of once per message.
"""

import asyncio
import json
import logging
import os
from collections import deque
from pathlib import Path

from bender.errors import ProcessError

logger = logging.getLogger(__name__)

DEFAULT_TURN_TIMEOUT_SECONDS = 300

# How many trailing stderr lines to keep for error reporting. Unbounded
# retention isn't needed -- this only ever surfaces in the "process exited
# unexpectedly" error message.
_STDERR_TAIL_LINES = 50

# asyncio.StreamReader.readline() defaults to a 64KiB limit and raises
# ValueError if a single line exceeds it before hitting the separator.
# stream-json emits one JSON object per line, and a single line embeds the
# full content of any tool_use/tool_result in that turn -- an MCP call
# that returns a large payload (seen in practice: an unfiltered financial
# data dump north of 1MB) produces a stdout line well past the default,
# which crashed turn parsing outright rather than merely being slow.
_STDOUT_LINE_LIMIT = 16 * 1024 * 1024

# Bender's own Slack app credentials (its Bolt/Socket Mode connection) must
# never leak into the Claude Code subprocess's environment: other MCP
# servers (e.g. the project's `slack` tool) resolve SLACK_BOT_TOKEN too, and
# would otherwise silently pick up Bender's narrow-scoped bot instead of the
# intended one.
_ENV_VARS_TO_STRIP = ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN")


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in _ENV_VARS_TO_STRIP:
        env.pop(name, None)
    return env


class ClaudeProcessError(ProcessError):
    """Raised when the long-lived Claude Code process fails or errors out."""


class ClaudeProcess:
    """One long-lived `claude` subprocess bound to a single Claude Code
    session. Call `start()` once, then `send()` for each user turn."""

    def __init__(self, workspace: Path, session_id: str | None = None) -> None:
        self.workspace = workspace
        self.session_id = session_id
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._stderr_tail: deque[bytes] = deque(maxlen=_STDERR_TAIL_LINES)
        self._stderr_task: asyncio.Task[None] | None = None

    async def start(self, resume: bool = False) -> None:
        """Spawn the subprocess. If resume=True and session_id is set,
        resumes that session; if session_id is set without resume, seeds
        a new session with that id; otherwise starts a fresh session."""
        cmd = [
            "claude",
            "--print",
            "--output-format",
            "stream-json",
            "--input-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            "bypassPermissions",
        ]
        if resume and self.session_id:
            cmd.extend(["--resume", self.session_id])
        elif self.session_id:
            cmd.extend(["--session-id", self.session_id])

        logger.info(
            "Starting long-lived Claude process (session=%s, resume=%s, workspace=%s)",
            self.session_id,
            resume,
            self.workspace,
        )
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.workspace,
            env=_subprocess_env(),
            limit=_STDOUT_LINE_LIMIT,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def send(self, prompt: str, timeout: int = DEFAULT_TURN_TIMEOUT_SECONDS) -> str:
        """Send one user turn and wait for the matching result message.
        Turns are serialized: concurrent callers queue behind the lock so
        two messages in the same thread never interleave on stdin."""
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise ClaudeProcessError("process not started")

        async with self._lock:
            payload = json.dumps({"type": "user", "message": {"role": "user", "content": prompt}})
            self._process.stdin.write((payload + "\n").encode())
            await self._process.stdin.drain()

            try:
                return await asyncio.wait_for(self._read_until_result(), timeout=timeout)
            except asyncio.TimeoutError as exc:
                raise ClaudeProcessError(f"Claude Code turn timed out after {timeout}s") from exc

    async def _drain_stderr(self) -> None:
        """Continuously read stderr into a bounded tail buffer.

        `--verbose` mode can write enough diagnostic output mid-turn to
        fill the OS pipe buffer; if nothing drains stderr, the `claude`
        subprocess blocks on that write and never reaches the point of
        emitting its stdout `result` line -- the turn then hangs until
        the caller's timeout fires. Reading stderr lazily (only at
        stdout EOF, the previous approach) doesn't prevent this because
        the deadlock happens *before* EOF. Draining continuously in the
        background avoids it; only the tail is kept, for error messages.
        """
        assert self._process is not None and self._process.stderr is not None
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    return
                self._stderr_tail.append(line)
        except (asyncio.CancelledError, ValueError):
            pass

    async def _read_until_result(self) -> str:
        assert self._process is not None and self._process.stdout is not None
        while True:
            line = await self._process.stdout.readline()
            if not line:
                # The process exited (or closed stdout). Give the stderr
                # drain task a brief moment to flush its final lines --
                # stdout EOF and the last stderr writes can arrive in
                # either order, and an empty error message here would
                # hide the actual crash reason.
                if self._stderr_task is not None:
                    try:
                        await asyncio.wait_for(asyncio.shield(self._stderr_task), timeout=1)
                    except (TimeoutError, asyncio.TimeoutError):
                        pass
                stderr = b"".join(self._stderr_tail)
                raise ClaudeProcessError(
                    "Claude Code process ended unexpectedly: "
                    f"{stderr.decode(errors='replace')[-500:]}"
                )
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") != "result":
                continue
            if data.get("session_id"):
                self.session_id = data["session_id"]
            if data.get("is_error"):
                raise ClaudeProcessError(str(data.get("result", "unknown error")))
            return str(data.get("result", ""))

    async def close(self) -> None:
        """Terminate the subprocess, giving it a moment to exit cleanly."""
        if self._process is None:
            return
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5)
        except (asyncio.TimeoutError, ProcessLookupError):
            self._process.kill()
            await self._process.wait()
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            self._stderr_task = None
        self._process = None
