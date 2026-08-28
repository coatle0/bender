"""Tests for the long-lived Claude Code subprocess wrapper."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bender.claude_process import ClaudeProcess, ClaudeProcessError


def _fake_process(stdout_lines: list[bytes], stderr: bytes = b"") -> MagicMock:
    """Build a MagicMock standing in for asyncio.subprocess.Process."""
    proc = MagicMock()
    proc.returncode = None
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdin.close = MagicMock()

    remaining = list(stdout_lines)

    async def readline():
        if remaining:
            return remaining.pop(0)
        return b""

    proc.stdout = MagicMock()
    proc.stdout.readline = AsyncMock(side_effect=readline)

    proc.stderr = MagicMock()
    proc.stderr.read = AsyncMock(return_value=stderr)

    proc.wait = AsyncMock(return_value=0)
    proc.kill = MagicMock()
    return proc


def _result_line(text: str, session_id: str = "s1", is_error: bool = False) -> bytes:
    payload = {"type": "result", "result": text, "session_id": session_id, "is_error": is_error}
    return (json.dumps(payload) + "\n").encode()


class TestClaudeProcessStart:
    async def test_start_builds_bypass_permissions_command(self, tmp_path: Path) -> None:
        """The spawned command always includes bypassPermissions and stream-json I/O."""
        proc = ClaudeProcess(workspace=tmp_path)
        fake = _fake_process([])
        with patch(
            "bender.claude_process.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake,
        ) as mock_exec:
            await proc.start()

        args = mock_exec.call_args[0]
        assert "claude" in args
        assert "--permission-mode" in args
        assert "bypassPermissions" in args
        assert "stream-json" in args

    async def test_start_with_resume_passes_resume_flag(self, tmp_path: Path) -> None:
        """resume=True with an existing session_id passes --resume <id>."""
        proc = ClaudeProcess(workspace=tmp_path, session_id="existing-session")
        fake = _fake_process([])
        with patch(
            "bender.claude_process.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake,
        ) as mock_exec:
            await proc.start(resume=True)

        args = mock_exec.call_args[0]
        assert "--resume" in args
        assert "existing-session" in args

    async def test_is_alive_reflects_process_state(self, tmp_path: Path) -> None:
        proc = ClaudeProcess(workspace=tmp_path)
        assert proc.is_alive is False

        fake = _fake_process([])
        with patch(
            "bender.claude_process.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake,
        ):
            await proc.start()
        assert proc.is_alive is True

        fake.returncode = 0
        assert proc.is_alive is False


class TestClaudeProcessSend:
    async def test_send_writes_stdin_and_returns_result_text(self, tmp_path: Path) -> None:
        """send() writes a user-turn JSON line and returns the result text."""
        proc = ClaudeProcess(workspace=tmp_path)
        fake = _fake_process([_result_line("TURN1_OK")])
        with patch(
            "bender.claude_process.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake,
        ):
            await proc.start()

        result = await proc.send("hello")

        assert result == "TURN1_OK"
        written = fake.stdin.write.call_args[0][0]
        payload = json.loads(written.decode().strip())
        assert payload == {"type": "user", "message": {"role": "user", "content": "hello"}}

    async def test_send_captures_session_id_from_result(self, tmp_path: Path) -> None:
        """A fresh process picks up the session_id Claude Code assigns."""
        proc = ClaudeProcess(workspace=tmp_path)
        fake = _fake_process([_result_line("ok", session_id="assigned-id")])
        with patch(
            "bender.claude_process.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake,
        ):
            await proc.start()
        await proc.send("hi")

        assert proc.session_id == "assigned-id"

    async def test_send_skips_non_result_lines(self, tmp_path: Path) -> None:
        """Intermediate system/assistant stream-json lines are skipped."""
        proc = ClaudeProcess(workspace=tmp_path)
        lines = [
            (json.dumps({"type": "system", "subtype": "init"}) + "\n").encode(),
            (json.dumps({"type": "assistant", "message": {"content": []}}) + "\n").encode(),
            _result_line("final answer"),
        ]
        fake = _fake_process(lines)
        with patch(
            "bender.claude_process.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake,
        ):
            await proc.start()

        assert await proc.send("hi") == "final answer"

    async def test_send_raises_on_is_error_result(self, tmp_path: Path) -> None:
        """An is_error result raises ClaudeProcessError."""
        proc = ClaudeProcess(workspace=tmp_path)
        fake = _fake_process([_result_line("boom", is_error=True)])
        with patch(
            "bender.claude_process.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake,
        ):
            await proc.start()

        with pytest.raises(ClaudeProcessError, match="boom"):
            await proc.send("hi")

    async def test_send_raises_when_process_exits_without_result(self, tmp_path: Path) -> None:
        """EOF before a result message raises ClaudeProcessError with stderr."""
        proc = ClaudeProcess(workspace=tmp_path)
        fake = _fake_process([], stderr=b"segfault")
        with patch(
            "bender.claude_process.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake,
        ):
            await proc.start()

        with pytest.raises(ClaudeProcessError, match="segfault"):
            await proc.send("hi")

    async def test_send_before_start_raises(self, tmp_path: Path) -> None:
        """send() on a never-started process raises ClaudeProcessError."""
        proc = ClaudeProcess(workspace=tmp_path)
        with pytest.raises(ClaudeProcessError, match="not started"):
            await proc.send("hi")

    async def test_two_turns_reuse_same_process(self, tmp_path: Path) -> None:
        """Two send() calls only spawn one subprocess (the whole point)."""
        proc = ClaudeProcess(workspace=tmp_path)
        fake = _fake_process([_result_line("first"), _result_line("second")])
        with patch(
            "bender.claude_process.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake,
        ) as mock_exec:
            await proc.start()
            r1 = await proc.send("turn 1")
            r2 = await proc.send("turn 2")

        assert (r1, r2) == ("first", "second")
        mock_exec.assert_called_once()


class TestClaudeProcessClose:
    async def test_close_closes_stdin_and_waits(self, tmp_path: Path) -> None:
        proc = ClaudeProcess(workspace=tmp_path)
        fake = _fake_process([])
        with patch(
            "bender.claude_process.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake,
        ):
            await proc.start()

        await proc.close()

        fake.stdin.close.assert_called_once()
        fake.wait.assert_awaited()
        assert proc.is_alive is False

    async def test_close_before_start_is_noop(self, tmp_path: Path) -> None:
        proc = ClaudeProcess(workspace=tmp_path)
        await proc.close()  # must not raise
