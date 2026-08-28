"""Codex CLI subprocess wrapper.

Codex has no stdin-streaming multi-turn primitive reachable from a plain
subprocess the way Claude Code's `--input-format stream-json` does --
that would require Codex's `app-server` daemon plus its `queue`/websocket
protocol, which is not wired here. Each turn is instead a fresh
`codex exec` (first turn) or `codex exec resume --json <thread_id>`
(later turns) invocation, matching Claude Code's original one-shot
design (bender's ClaudeProcess only became a long-lived process because
Claude Code specifically supports it).

On Windows, `codex` resolves to an npm `.cmd` shim rather than a native
binary, and `asyncio.create_subprocess_exec(["codex", ...])` fails with
FileNotFoundError because CreateProcess does not resolve PATHEXT the way
a shell does. The executable name below is `codex.cmd` explicitly to
avoid that.
"""

import asyncio
import json
import logging
from pathlib import Path

from bender.claude_process import _subprocess_env
from bender.errors import ProcessError

logger = logging.getLogger(__name__)

DEFAULT_TURN_TIMEOUT_SECONDS = 300
CODEX_EXECUTABLE = "codex.cmd"


class CodexProcessError(ProcessError):
    """Raised when a Codex CLI invocation fails or returns no reply."""


class CodexProcess:
    """One Codex CLI conversation thread.

    Despite the name, this does not hold a long-lived OS process handle
    (see module docstring) -- it holds the Codex `thread_id` and spawns
    one `codex exec` invocation per `send()` call.
    """

    def __init__(self, workspace: Path, session_id: str | None = None) -> None:
        self.workspace = workspace
        self.session_id = session_id
        self._started = False

    async def start(self, resume: bool = False) -> None:
        """No process to spawn yet; send() does that per turn. Kept for
        interface parity with ClaudeProcess / ProcessPool."""
        self._started = True

    @property
    def is_alive(self) -> bool:
        """Always true once started -- there is no OS process to go
        stale, so ProcessPool keeps reusing this object (and its
        session_id) for every message in the thread."""
        return self._started

    async def send(self, prompt: str, timeout: int = DEFAULT_TURN_TIMEOUT_SECONDS) -> str:
        if not self._started:
            raise CodexProcessError("process not started")

        if self.session_id:
            cmd = [
                CODEX_EXECUTABLE, "exec", "resume", "--json", self.session_id, prompt,
            ]
        else:
            cmd = [CODEX_EXECUTABLE, "exec", prompt, "--json"]
        cmd += ["--sandbox", "workspace-write", "--skip-git-repo-check"]

        logger.info(
            "Running codex exec (session=%s, resume=%s, workspace=%s)",
            self.session_id,
            bool(self.session_id),
            self.workspace,
        )

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.workspace,
            env=_subprocess_env(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise CodexProcessError(f"Codex turn timed out after {timeout}s") from exc

        if process.returncode != 0:
            raise CodexProcessError(
                f"codex exec exited {process.returncode}: "
                f"{stderr.decode(errors='replace')[-500:]}"
            )

        return self._parse_jsonl(stdout.decode())

    def _parse_jsonl(self, raw: str) -> str:
        """Codex --json prints one JSON object per line: thread.started,
        turn.started, item.completed (agent_message carries the reply
        text), turn.completed. Grabs the thread_id and the last
        agent_message text."""
        text = ""
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") == "thread.started" and data.get("thread_id"):
                self.session_id = data["thread_id"]
            elif data.get("type") == "item.completed":
                item = data.get("item") or {}
                if item.get("type") == "agent_message" and item.get("text"):
                    text = item["text"]
        if not text:
            raise CodexProcessError("Codex produced no agent_message")
        return text

    async def close(self) -> None:
        """No OS process is held between turns, so there is nothing to
        terminate -- just mark the thread as no longer usable."""
        self._started = False
