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
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TURN_TIMEOUT_SECONDS = 300


class ClaudeProcessError(Exception):
    """Raised when the long-lived Claude Code process fails or errors out."""


class ClaudeProcess:
    """One long-lived `claude` subprocess bound to a single Claude Code
    session. Call `start()` once, then `send()` for each user turn."""

    def __init__(self, workspace: Path, session_id: str | None = None) -> None:
        self.workspace = workspace
        self.session_id = session_id
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

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
        )

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

    async def _read_until_result(self) -> str:
        assert self._process is not None and self._process.stdout is not None
        while True:
            line = await self._process.stdout.readline()
            if not line:
                stderr = b""
                if self._process.stderr is not None:
                    stderr = await self._process.stderr.read()
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
        self._process = None
