"""Tests for the Slack event handlers module."""

from unittest.mock import AsyncMock

import pytest

from bender.claude_process import ClaudeProcessError
from bender.session_manager import SessionManager
from bender.slack_handler import _strip_mention, register_handlers


class TestStripMention:
    """Tests for the _strip_mention helper."""

    def test_removes_single_mention(self) -> None:
        """Removes a single <@UXXXX> mention."""
        assert _strip_mention("<@U12345ABC> hello world") == "hello world"

    def test_removes_multiple_mentions(self) -> None:
        """Removes multiple mentions."""
        assert _strip_mention("<@U111> <@U222> do something") == "do something"

    def test_no_mention_returns_text(self) -> None:
        """Returns text unchanged when no mention is present."""
        assert _strip_mention("no mention here") == "no mention here"

    def test_empty_string(self) -> None:
        """Handles empty string gracefully."""
        assert _strip_mention("") == ""

    def test_only_mention_returns_empty(self) -> None:
        """Returns empty string when text is just a mention."""
        assert _strip_mention("<@U12345ABC>") == ""

    def test_removes_bot_mention(self) -> None:
        """Removes <@BXXXX> bot mentions."""
        assert _strip_mention("<@B12345ABC> hello") == "hello"

    def test_removes_workspace_mention(self) -> None:
        """Removes <@WXXXX> workspace mentions."""
        assert _strip_mention("<@W12345ABC> hello") == "hello"


@pytest.fixture
def pool() -> AsyncMock:
    """A stand-in ProcessPool whose send() is mocked per-test."""
    return AsyncMock()


class TestHandleMention:
    """Tests for the app_mention event handler."""

    @pytest.fixture
    def setup_handler(self, session_manager: SessionManager, pool: AsyncMock):
        """Set up a mock bolt app and register handlers."""
        mock_app = AsyncMock()
        handlers = {}

        def capture_event(event_type):
            def decorator(func):
                handlers[event_type] = func
                return func
            return decorator

        mock_app.event = capture_event
        register_handlers(mock_app, session_manager, pool)
        return handlers

    async def test_mention_routes_through_pool(
        self, setup_handler, pool: AsyncMock, mock_say: AsyncMock
    ) -> None:
        """New mention is sent to the thread's process pool entry."""
        handler = setup_handler["app_mention"]
        event = {
            "text": "<@U12345> check the logs",
            "ts": "1234567890.000001",
            "channel": "C123",
        }
        pool.send.return_value = "Logs look fine"

        await handler(event=event, say=mock_say)

        pool.send.assert_called_once_with("1234567890.000001", "check the logs")
        mock_say.assert_called_once_with(text="Logs look fine", thread_ts="1234567890.000001")

    async def test_mention_reply_in_tracked_thread_uses_parent_thread_ts(
        self,
        setup_handler,
        session_manager: SessionManager,
        pool: AsyncMock,
        mock_say: AsyncMock,
    ) -> None:
        """A mention posted as a reply inside an already-tracked thread
        must route to the thread's parent ts, not its own -- using its
        own ts here would silently fork off a disconnected new session.
        Observed live: a threaded "ACK 필요" reminder became its own
        isolated thread and Codex never saw the request it referred to."""
        parent_ts = "1000000000.000001"
        reply_ts = "1000000000.500002"
        await session_manager.create_session(parent_ts)  # thread already tracked

        handler = setup_handler["app_mention"]
        event = {
            "text": "<@U12345> 위 요청 확인 바람. ACK 필요.",
            "ts": reply_ts,
            "thread_ts": parent_ts,
            "channel": "C123",
        }
        pool.send.return_value = "ACK."

        await handler(event=event, say=mock_say)

        pool.send.assert_called_once_with(parent_ts, "위 요청 확인 바람. ACK 필요.")
        say_kwargs = mock_say.call_args.kwargs
        assert say_kwargs["thread_ts"] == parent_ts
        assert say_kwargs["text"].startswith("ACK.")

    async def test_mention_reply_in_untracked_thread_backfills_context(
        self,
        setup_handler,
        session_manager: SessionManager,
        pool: AsyncMock,
        mock_say: AsyncMock,
    ) -> None:
        """A mention that's the *first* bot involvement in a thread, but
        arrives as a reply partway through pre-existing (bot-free) thread
        history, must pull that history in -- otherwise the prompt is
        just the reply's own text, with no idea what it's about. This is
        exactly what happened live: a request was posted, then a plain
        reminder reply first mentioned the bot, and Codex only ever saw
        the reminder."""
        parent_ts = "2000000000.000001"
        reply_ts = "2000000000.500002"
        mock_client = AsyncMock()
        mock_client.conversations_replies.return_value = {
            "messages": [
                {"ts": parent_ts, "text": "[요청] 1. 파일 확인 2. 귀속 확인"},
                {"ts": reply_ts, "text": "<@U12345> 위 요청 확인 바람. ACK 필요."},
            ]
        }

        handler = setup_handler["app_mention"]
        event = {
            "text": "<@U12345> 위 요청 확인 바람. ACK 필요.",
            "ts": reply_ts,
            "thread_ts": parent_ts,
            "channel": "C123",
        }
        pool.send.return_value = "ACK, will do."

        await handler(event=event, say=mock_say, client=mock_client)

        mock_client.conversations_replies.assert_called_once_with(
            channel="C123", ts=parent_ts, limit=50
        )
        sent_prompt = pool.send.call_args[0][1]
        assert "[요청] 1. 파일 확인 2. 귀속 확인" in sent_prompt
        assert "위 요청 확인 바람. ACK 필요." in sent_prompt
        # The reply's own text (fetched back from the API) isn't
        # double-counted as "prior" context -- only messages strictly
        # before it are.
        assert sent_prompt.count("위 요청 확인 바람") == 1

    async def test_mention_empty_text_responds_help(
        self, setup_handler, pool: AsyncMock, mock_say: AsyncMock
    ) -> None:
        """Empty mention text gets a help response without touching the pool."""
        handler = setup_handler["app_mention"]
        event = {"text": "<@U12345>", "ts": "1234567890.000001", "channel": "C123"}

        await handler(event=event, say=mock_say)

        pool.send.assert_not_called()
        mock_say.assert_called_once_with(text="How can I help?", thread_ts="1234567890.000001")

    async def test_mention_appends_session_id_footer(
        self,
        setup_handler,
        session_manager: SessionManager,
        pool: AsyncMock,
        mock_say: AsyncMock,
    ) -> None:
        """The first-turn reply is tagged with the session id ProcessPool
        just persisted, so a stuck thread can be grepped out of the logs
        by that id instead of reconstructed from Slack timestamps."""
        handler = setup_handler["app_mention"]
        event = {
            "text": "<@U12345> check the logs",
            "ts": "1234567890.000001",
            "channel": "C123",
        }
        pool.send.return_value = "Logs look fine"
        # Stand in for what the real ProcessPool.send() does internally
        # (persist proc.session_id via SessionManager) before the handler
        # reads it back.
        await session_manager.set_session("1234567890.000001", "sess-abc-123")

        await handler(event=event, say=mock_say)

        mock_say.assert_called_once_with(
            text="Logs look fine\n\n_(session: `sess-abc-123`)_",
            thread_ts="1234567890.000001",
        )

    async def test_mention_claude_error_posts_error(
        self, setup_handler, pool: AsyncMock, mock_say: AsyncMock
    ) -> None:
        """Posts error message when the pool raises ClaudeProcessError."""
        handler = setup_handler["app_mention"]
        event = {"text": "<@U12345> do something", "ts": "1234567890.000001", "channel": "C123"}
        pool.send.side_effect = ClaudeProcessError("CLI crashed")

        await handler(event=event, say=mock_say)

        mock_say.assert_called_once()
        call_kwargs = mock_say.call_args[1]
        assert "Sorry, something went wrong" in call_kwargs["text"]
        assert "CLI crashed" in call_kwargs["text"]


class TestHandleMessage:
    """Tests for the message event handler (thread replies)."""

    @pytest.fixture
    def setup_handler(self, session_manager: SessionManager, pool: AsyncMock):
        """Set up a mock bolt app and register handlers."""
        mock_app = AsyncMock()
        handlers = {}

        def capture_event(event_type):
            def decorator(func):
                handlers[event_type] = func
                return func
            return decorator

        mock_app.event = capture_event
        register_handlers(mock_app, session_manager, pool)
        return handlers

    async def test_thread_reply_routes_through_pool(
        self, setup_handler, session_manager: SessionManager, pool: AsyncMock, mock_say: AsyncMock
    ) -> None:
        """Thread reply on a tracked thread is sent to the process pool."""
        handler = setup_handler["message"]
        thread_ts = "1234567890.000001"
        await session_manager.create_session(thread_ts)

        event = {
            "text": "yes, go ahead",
            "thread_ts": thread_ts,
            "channel": "C123",
        }
        pool.send.return_value = "Done!"

        await handler(event=event, say=mock_say)

        pool.send.assert_called_once_with(thread_ts, "yes, go ahead")
        mock_say.assert_called_once_with(text="Done!", thread_ts=thread_ts)

    async def test_thread_reply_ignores_bot_messages(
        self, setup_handler, session_manager: SessionManager, pool: AsyncMock, mock_say: AsyncMock
    ) -> None:
        """Bot messages are ignored to prevent loops."""
        handler = setup_handler["message"]
        thread_ts = "1234567890.000001"
        await session_manager.create_session(thread_ts)

        event = {
            "text": "bot response",
            "thread_ts": thread_ts,
            "bot_id": "B12345",
            "channel": "C123",
        }

        await handler(event=event, say=mock_say)
        pool.send.assert_not_called()
        mock_say.assert_not_called()

    async def test_thread_reply_ignores_subtype_messages(
        self, setup_handler, session_manager: SessionManager, pool: AsyncMock, mock_say: AsyncMock
    ) -> None:
        """Messages with subtype (e.g., channel_join) are ignored."""
        handler = setup_handler["message"]
        thread_ts = "1234567890.000001"
        await session_manager.create_session(thread_ts)

        event = {
            "text": "joined the channel",
            "thread_ts": thread_ts,
            "subtype": "channel_join",
            "channel": "C123",
        }

        await handler(event=event, say=mock_say)
        mock_say.assert_not_called()

    async def test_non_thread_message_ignored(
        self, setup_handler, pool: AsyncMock, mock_say: AsyncMock
    ) -> None:
        """Non-thread messages are ignored."""
        handler = setup_handler["message"]
        event = {"text": "hello", "channel": "C123"}

        await handler(event=event, say=mock_say)
        pool.send.assert_not_called()
        mock_say.assert_not_called()

    async def test_untracked_thread_ignored(
        self, setup_handler, pool: AsyncMock, mock_say: AsyncMock
    ) -> None:
        """Thread replies in untracked threads are ignored."""
        handler = setup_handler["message"]
        event = {
            "text": "hello",
            "thread_ts": "9999999999.999999",
            "channel": "C123",
        }

        await handler(event=event, say=mock_say)
        pool.send.assert_not_called()
        mock_say.assert_not_called()

    async def test_thread_reply_empty_text_ignored(
        self, setup_handler, session_manager: SessionManager, pool: AsyncMock, mock_say: AsyncMock
    ) -> None:
        """Empty thread replies are ignored."""
        handler = setup_handler["message"]
        thread_ts = "1234567890.000001"
        await session_manager.create_session(thread_ts)

        event = {"text": "<@U12345>", "thread_ts": thread_ts, "channel": "C123"}

        await handler(event=event, say=mock_say)
        pool.send.assert_not_called()
        mock_say.assert_not_called()

    async def test_thread_reply_claude_error(
        self, setup_handler, session_manager: SessionManager, pool: AsyncMock, mock_say: AsyncMock
    ) -> None:
        """Posts error message when the pool raises ClaudeProcessError."""
        handler = setup_handler["message"]
        thread_ts = "1234567890.000001"
        await session_manager.create_session(thread_ts)

        event = {
            "text": "do something",
            "thread_ts": thread_ts,
            "channel": "C123",
        }
        pool.send.side_effect = ClaudeProcessError("timeout")

        await handler(event=event, say=mock_say)

        mock_say.assert_called_once()
        assert "Sorry, something went wrong" in mock_say.call_args[1]["text"]
