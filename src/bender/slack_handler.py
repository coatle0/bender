"""Slack event handlers — @mentions and thread replies via slack-bolt."""

import logging
import re

from slack_bolt.async_app import AsyncApp

from bender.errors import ProcessError
from bender.process_pool import ProcessPool
from bender.session_manager import SessionManager
from bender.slack_utils import SLACK_MSG_LIMIT, md_to_mrkdwn, split_text

logger = logging.getLogger(__name__)


def register_handlers(
    app: AsyncApp, sessions: SessionManager, pool: ProcessPool
) -> None:
    """Register Slack event handlers on the bolt app."""

    @app.event("reaction_added")
    async def handle_reaction_added(event: dict) -> None:
        """Ignore reaction_added events — Bender doesn't need to track reactions."""
        pass

    @app.event("reaction_removed")
    async def handle_reaction_removed(event: dict) -> None:
        """Ignore reaction_removed events — Bender doesn't need to track reactions."""
        pass

    @app.event("app_mention")
    async def handle_mention(event: dict, say) -> None:
        """Handle new @Bender mentions — starts (or reuses) the thread's
        long-lived Claude Code process."""
        text = _strip_mention(event.get("text", ""))
        thread_ts = event.get("ts", "")
        channel = event.get("channel", "")

        if not text.strip():
            await say(text="How can I help?", thread_ts=thread_ts)
            return

        logger.info("New mention in channel=%s thread=%s", channel, thread_ts)

        try:
            result = await pool.send(thread_ts, text)
            session_id = await sessions.get_session(thread_ts)
            if session_id:
                # Surfaced only on the thread's first reply, once per thread
                # rather than on every turn -- lets a stuck/misbehaving
                # thread be grepped straight out of the logs by session_id
                # instead of reconstructed from Slack timestamps after the
                # fact (the exact gap hit debugging a hung Codexy turn).
                result = f"{result}\n\n_(session: `{session_id}`)_"
            await _post_response(say, result, thread_ts)
        except ProcessError as exc:
            logger.error("Backend invocation failed: %s", exc)
            await say(text=f"Sorry, something went wrong: {exc}", thread_ts=thread_ts)

    @app.event("message")
    async def handle_message(event: dict, say) -> None:
        """Handle thread replies — resume the thread's live process (or
        start one, resuming its persisted session, if none is running)."""
        # Ignore bot messages to avoid loops
        if event.get("bot_id") or event.get("subtype"):
            return

        thread_ts = event.get("thread_ts")
        if not thread_ts:
            # Not a thread reply, ignore
            return

        if not await sessions.has_session(thread_ts):
            # Thread not tracked by Bender, ignore
            return

        text = _strip_mention(event.get("text", ""))
        if not text.strip():
            return

        channel = event.get("channel", "")
        logger.info("Thread reply in channel=%s thread=%s", channel, thread_ts)

        try:
            result = await pool.send(thread_ts, text)
            await _post_response(say, result, thread_ts)
        except ProcessError as exc:
            logger.error("Backend invocation failed: %s", exc)
            await say(text=f"Sorry, something went wrong: {exc}", thread_ts=thread_ts)


def _strip_mention(text: str) -> str:
    """Remove Slack mention tags (<@U...>, <@B...>, <@W...>) from the message text."""
    return re.sub(r"<@[UBW][A-Z0-9]+>", "", text).strip()


async def _post_response(say, text: str, thread_ts: str) -> None:
    """Post a response in the thread, splitting if it exceeds Slack's limit."""
    text = md_to_mrkdwn(text)

    if len(text) <= SLACK_MSG_LIMIT:
        await say(text=text, thread_ts=thread_ts)
        return

    chunks = split_text(text, SLACK_MSG_LIMIT)
    for chunk in chunks:
        await say(text=chunk, thread_ts=thread_ts)
