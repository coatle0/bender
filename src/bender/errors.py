"""Shared error types for the process backends."""


class ProcessError(Exception):
    """Raised when a conversation-thread backend (Claude or Codex) fails.
    ProcessPool and slack_handler catch this base class so they don't need
    to know which concrete backend a given Bender instance is running."""
