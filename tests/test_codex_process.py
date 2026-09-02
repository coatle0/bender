"""Tests for the long-lived Codex app-server subprocess wrapper."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from codex_app_server_sdk.errors import (
    CodexProtocolError,
    CodexTimeoutError,
    CodexTransportError,
)

from bender.codex_process import CODEX_EXECUTABLE, CodexProcess, CodexProcessError


def _fake_client(**chat_once_results: object) -> MagicMock:
    """A fake CodexClient. `chat_once_results` is unused directly; tests
    configure `client.chat_once.side_effect` / `.return_value` themselves."""
    client = MagicMock()
    client.start = AsyncMock()
    client.initialize = AsyncMock()
    client.chat_once = AsyncMock()
    client.close = AsyncMock()
    return client


def _chat_result(thread_id: str, final_text: str) -> MagicMock:
    result = MagicMock()
    result.thread_id = thread_id
    result.final_text = final_text
    return result


class TestCodexProcessStart:
    async def test_is_alive_after_start(self, tmp_path: Path) -> None:
        proc = CodexProcess(workspace=tmp_path)
        assert proc.is_alive is False
        with patch(
            "bender.codex_process.CodexClient.connect_stdio",
            return_value=_fake_client(),
        ):
            await proc.start()
        assert proc.is_alive is True

    async def test_close_marks_not_alive(self, tmp_path: Path) -> None:
        proc = CodexProcess(workspace=tmp_path)
        fake = _fake_client()
        with patch("bender.codex_process.CodexClient.connect_stdio", return_value=fake):
            await proc.start()
        await proc.close()
        assert proc.is_alive is False
        fake.close.assert_awaited_once()

    async def test_start_spawns_app_server_via_cmd_shim(self, tmp_path: Path) -> None:
        """Windows resolves `codex` to an npm .cmd shim; the SDK's
        stdio transport needs the explicit .cmd name."""
        proc = CodexProcess(workspace=tmp_path)
        with patch(
            "bender.codex_process.CodexClient.connect_stdio",
            return_value=_fake_client(),
        ) as mock_connect:
            await proc.start()

        assert mock_connect.call_args.kwargs["command"] == [CODEX_EXECUTABLE, "app-server"]


class TestCodexProcessSend:
    async def test_send_before_start_raises(self, tmp_path: Path) -> None:
        proc = CodexProcess(workspace=tmp_path)
        with pytest.raises(CodexProcessError, match="not started"):
            await proc.send("hi")

    async def test_first_send_passes_no_thread_id(self, tmp_path: Path) -> None:
        """A fresh thread (no session_id yet) omits thread_id so the
        app-server starts a new one."""
        proc = CodexProcess(workspace=tmp_path)
        fake = _fake_client()
        fake.chat_once.return_value = _chat_result("new-thread-id", "hello back")
        with patch("bender.codex_process.CodexClient.connect_stdio", return_value=fake):
            await proc.start()

        result = await proc.send("hi")

        assert result == "hello back"
        assert proc.session_id == "new-thread-id"
        assert fake.chat_once.call_args.kwargs["thread_id"] is None
        assert fake.chat_once.call_args.args[0] == "hi"

    async def test_second_send_reuses_client_and_resumes_thread(self, tmp_path: Path) -> None:
        """Once a thread_id is known, the SAME app-server connection
        (no re-spawn) is reused, passing the known thread_id."""
        proc = CodexProcess(workspace=tmp_path, session_id="existing-thread")
        fake = _fake_client()
        fake.chat_once.return_value = _chat_result("existing-thread", "remembered")
        with patch(
            "bender.codex_process.CodexClient.connect_stdio", return_value=fake
        ) as mock_connect:
            await proc.start(resume=True)
            await proc.send("what did I say?")
            await proc.send("and again?")

        assert mock_connect.call_count == 1
        assert fake.chat_once.call_count == 2
        for call in fake.chat_once.call_args_list:
            assert call.kwargs["thread_id"] == "existing-thread"

    async def test_send_bypasses_approvals_for_mcp_tool_calls(self, tmp_path: Path) -> None:
        """MCP tool calls (e.g. the `slack` server) fail closed under
        any approval policy other than 'never'; only
        approval_policy='never' + sandbox='danger-full-access' lets
        them run, verified against the real slack MCP server."""
        proc = CodexProcess(workspace=tmp_path)
        fake = _fake_client()
        fake.chat_once.return_value = _chat_result("t1", "ok")
        with patch("bender.codex_process.CodexClient.connect_stdio", return_value=fake):
            await proc.start()
            await proc.send("hi")

        cfg = fake.chat_once.call_args.kwargs["thread_config"]
        assert cfg.approval_policy == "never"
        assert cfg.sandbox == "danger-full-access"

    async def test_raises_when_no_final_text(self, tmp_path: Path) -> None:
        proc = CodexProcess(workspace=tmp_path)
        fake = _fake_client()
        fake.chat_once.return_value = _chat_result("t1", "")
        with patch("bender.codex_process.CodexClient.connect_stdio", return_value=fake):
            await proc.start()
            with pytest.raises(CodexProcessError, match="no agent_message"):
                await proc.send("hi")

    async def test_transport_error_raises_codex_process_error(self, tmp_path: Path) -> None:
        proc = CodexProcess(workspace=tmp_path)
        fake = _fake_client()
        fake.chat_once.side_effect = CodexTransportError("connection dropped")
        with patch("bender.codex_process.CodexClient.connect_stdio", return_value=fake):
            await proc.start()
            with pytest.raises(CodexProcessError, match="connection dropped"):
                await proc.send("hi")

    async def test_protocol_error_raises_codex_process_error(self, tmp_path: Path) -> None:
        proc = CodexProcess(workspace=tmp_path)
        fake = _fake_client()
        fake.chat_once.side_effect = CodexProtocolError("bad params")
        with patch("bender.codex_process.CodexClient.connect_stdio", return_value=fake):
            await proc.start()
            with pytest.raises(CodexProcessError, match="bad params"):
                await proc.send("hi")

    async def test_timeout_raises_codex_process_error(self, tmp_path: Path) -> None:
        proc = CodexProcess(workspace=tmp_path)
        fake = _fake_client()
        fake.chat_once.side_effect = CodexTimeoutError("inactivity timeout")
        with patch("bender.codex_process.CodexClient.connect_stdio", return_value=fake):
            await proc.start()
            with pytest.raises(CodexProcessError, match="timed out"):
                await proc.send("hi", timeout=1)
        fake.close.assert_awaited_once()
        assert proc.is_alive is False

    async def test_hard_timeout_fires_when_sdk_inactivity_timer_never_does(
        self, tmp_path: Path
    ) -> None:
        """A turn stuck in a retry/progress loop keeps the SDK's sliding
        inactivity window alive indefinitely (observed live: a Slack
        thread wedged ~44h). The outer wait_for must still cut it off,
        close the connection, and mark the process dead so the pool
        starts a fresh one on the thread's next message."""
        import asyncio

        proc = CodexProcess(workspace=tmp_path)
        fake = _fake_client()

        async def never_finishes(*args: object, **kwargs: object) -> MagicMock:
            await asyncio.sleep(3600)
            return _chat_result("t1", "should never get here")

        fake.chat_once.side_effect = never_finishes
        with patch("bender.codex_process.CodexClient.connect_stdio", return_value=fake):
            await proc.start()
            with pytest.raises(CodexProcessError, match="hard cap"):
                await proc.send("hi", timeout=0.05)

        fake.close.assert_awaited_once()
        assert proc.is_alive is False

    async def test_uses_stripped_subprocess_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bender's own Slack tokens must not leak into the codex
        app-server subprocess either."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-benders-own-token")
        proc = CodexProcess(workspace=tmp_path)
        with patch(
            "bender.codex_process.CodexClient.connect_stdio",
            return_value=_fake_client(),
        ) as mock_connect:
            await proc.start()

        assert "SLACK_BOT_TOKEN" not in mock_connect.call_args.kwargs["env"]

    async def test_send_is_serialized_under_lock(self, tmp_path: Path) -> None:
        """Two concurrent send() calls on the same thread must not
        interleave chat_once calls on the shared connection."""
        import asyncio

        proc = CodexProcess(workspace=tmp_path)
        fake = _fake_client()
        call_order: list[str] = []

        async def slow_chat_once(*args: object, **kwargs: object) -> MagicMock:
            call_order.append("start")
            await asyncio.sleep(0.01)
            call_order.append("end")
            return _chat_result("t1", "ok")

        fake.chat_once.side_effect = slow_chat_once
        with patch("bender.codex_process.CodexClient.connect_stdio", return_value=fake):
            await proc.start()
            await asyncio.gather(proc.send("a"), proc.send("b"))

        assert call_order == ["start", "end", "start", "end"]
