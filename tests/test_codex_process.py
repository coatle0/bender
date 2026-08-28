"""Tests for the Codex CLI subprocess wrapper."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bender.codex_process import CODEX_EXECUTABLE, CodexProcess, CodexProcessError


def _fake_process(stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=returncode)
    return proc


def _thread_started(thread_id: str) -> bytes:
    return (json.dumps({"type": "thread.started", "thread_id": thread_id}) + "\n").encode()


def _agent_message(text: str) -> bytes:
    item = {"id": "item_0", "type": "agent_message", "text": text}
    payload = {"type": "item.completed", "item": item}
    return (json.dumps(payload) + "\n").encode()


def _turn_completed() -> bytes:
    return (json.dumps({"type": "turn.completed", "usage": {}}) + "\n").encode()


class TestCodexProcessStart:
    async def test_is_alive_after_start(self, tmp_path: Path) -> None:
        proc = CodexProcess(workspace=tmp_path)
        assert proc.is_alive is False
        await proc.start()
        assert proc.is_alive is True

    async def test_close_marks_not_alive(self, tmp_path: Path) -> None:
        proc = CodexProcess(workspace=tmp_path)
        await proc.start()
        await proc.close()
        assert proc.is_alive is False


class TestCodexProcessSend:
    async def test_send_before_start_raises(self, tmp_path: Path) -> None:
        proc = CodexProcess(workspace=tmp_path)
        with pytest.raises(CodexProcessError, match="not started"):
            await proc.send("hi")

    async def test_first_send_uses_exec_without_resume(self, tmp_path: Path) -> None:
        """A fresh thread (no session_id yet) runs `codex exec <prompt> --json`,
        not `codex exec resume`."""
        proc = CodexProcess(workspace=tmp_path)
        await proc.start()
        stdout = _thread_started("new-thread-id") + _agent_message("hello back") + _turn_completed()
        fake = _fake_process(stdout)

        with patch(
            "bender.codex_process.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake,
        ) as mock_exec:
            result = await proc.send("hi")

        assert result == "hello back"
        assert proc.session_id == "new-thread-id"
        args = mock_exec.call_args[0]
        assert args[0] == CODEX_EXECUTABLE
        assert "exec" in args
        assert "resume" not in args
        assert "hi" in args

    async def test_send_bypasses_approvals_for_mcp_tool_calls(self, tmp_path: Path) -> None:
        """MCP tool calls (e.g. the `slack` server) fail closed under plain
        --sandbox workspace-write ("approval policy is never"); only
        --dangerously-bypass-approvals-and-sandbox lets them run, verified
        against the real slack MCP server."""
        proc = CodexProcess(workspace=tmp_path)
        await proc.start()
        stdout = _thread_started("t1") + _agent_message("ok") + _turn_completed()
        fake = _fake_process(stdout)

        with patch(
            "bender.codex_process.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake,
        ) as mock_exec:
            await proc.send("hi")

        args = mock_exec.call_args[0]
        assert "--dangerously-bypass-approvals-and-sandbox" in args
        assert "--sandbox" not in args

    async def test_second_send_resumes_thread(self, tmp_path: Path) -> None:
        """Once a thread_id is known, later turns use `codex exec resume --json <id>`."""
        proc = CodexProcess(workspace=tmp_path, session_id="existing-thread")
        await proc.start()
        stdout = (
            _thread_started("existing-thread") + _agent_message("remembered") + _turn_completed()
        )
        fake = _fake_process(stdout)

        with patch(
            "bender.codex_process.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake,
        ) as mock_exec:
            result = await proc.send("what did I say?")

        assert result == "remembered"
        args = mock_exec.call_args[0]
        assert "resume" in args
        assert "--json" in args
        assert "existing-thread" in args
        assert "what did I say?" in args

    async def test_skips_non_agent_message_items(self, tmp_path: Path) -> None:
        """Only the agent_message item's text is returned; other event
        types are ignored."""
        proc = CodexProcess(workspace=tmp_path)
        await proc.start()
        stdout = (
            _thread_started("t1")
            + (json.dumps({"type": "turn.started"}) + "\n").encode()
            + (
                json.dumps({"type": "item.completed", "item": {"type": "command", "text": "ls"}})
                + "\n"
            ).encode()
            + _agent_message("final answer")
            + _turn_completed()
        )
        fake = _fake_process(stdout)

        with patch(
            "bender.codex_process.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake,
        ):
            assert await proc.send("hi") == "final answer"

    async def test_raises_on_nonzero_exit(self, tmp_path: Path) -> None:
        proc = CodexProcess(workspace=tmp_path)
        await proc.start()
        fake = _fake_process(b"", stderr=b"auth error", returncode=1)

        with patch(
            "bender.codex_process.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake,
        ):
            with pytest.raises(CodexProcessError, match="auth error"):
                await proc.send("hi")

    async def test_raises_when_no_agent_message_present(self, tmp_path: Path) -> None:
        proc = CodexProcess(workspace=tmp_path)
        await proc.start()
        stdout = _thread_started("t1") + _turn_completed()
        fake = _fake_process(stdout)

        with patch(
            "bender.codex_process.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake,
        ):
            with pytest.raises(CodexProcessError, match="no agent_message"):
                await proc.send("hi")

    async def test_timeout_kills_process(self, tmp_path: Path) -> None:
        proc = CodexProcess(workspace=tmp_path)
        await proc.start()
        fake = MagicMock()
        fake.communicate = AsyncMock(return_value=(b"", b""))
        fake.kill = MagicMock()
        fake.wait = AsyncMock(return_value=None)

        with patch(
            "bender.codex_process.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake,
        ), patch(
            "bender.codex_process.asyncio.wait_for",
            new_callable=AsyncMock,
            side_effect=TimeoutError(),
        ):
            with pytest.raises(CodexProcessError, match="timed out"):
                await proc.send("hi", timeout=1)

        fake.kill.assert_called_once()

    async def test_uses_stripped_subprocess_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bender's own Slack tokens must not leak into the codex subprocess either."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-benders-own-token")
        proc = CodexProcess(workspace=tmp_path)
        await proc.start()
        stdout = _thread_started("t1") + _agent_message("ok") + _turn_completed()
        fake = _fake_process(stdout)

        with patch(
            "bender.codex_process.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake,
        ) as mock_exec:
            await proc.send("hi")

        assert "SLACK_BOT_TOKEN" not in mock_exec.call_args.kwargs["env"]
