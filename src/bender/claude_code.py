"""Claude Code CLI invocation — subprocess wrapper for headless mode."""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from bender.claude_process import _subprocess_env

logger = logging.getLogger(__name__)

# Default timeout for Claude Code invocations (5 minutes)
DEFAULT_TIMEOUT_SECONDS = 300


@dataclass
class ClaudeResponse:
    """Parsed response from Claude Code CLI."""

    result: str
    session_id: str
    is_error: bool = False


class ClaudeCodeError(Exception):
    """Raised when Claude Code CLI invocation fails."""


async def invoke_claude(
    prompt: str,
    workspace: Path,
    session_id: str | None = None,
    resume: bool = False,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> ClaudeResponse:
    """Invoke Claude Code CLI in headless mode via subprocess.

    Args:
        prompt: The message/prompt to send to Claude Code.
        workspace: Working directory where Claude Code runs.
        session_id: Session ID for new or resumed sessions.
        resume: Whether to resume an existing session.
        timeout: Maximum execution time in seconds.

    Returns:
        ClaudeResponse with the parsed result.

    Raises:
        ClaudeCodeError: If the CLI invocation fails.
    """
    cmd = ["claude", "--print", "--output-format", "json"]

    if resume and session_id:
        cmd.extend(["--resume", session_id])
    elif session_id:
        cmd.extend(["--session-id", session_id])

    cmd.extend(["--", prompt])

    logger.info(
        "Invoking Claude Code (session=%s, resume=%s, workspace=%s)",
        session_id,
        resume,
        workspace,
    )

    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace,
            env=_subprocess_env(),
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )
    except FileNotFoundError:
        raise ClaudeCodeError(
            "Claude Code CLI not found. Ensure 'claude' is installed and in PATH."
        )
    except asyncio.TimeoutError:
        if process is not None:
            process.kill()
            await process.wait()
        raise ClaudeCodeError(f"Claude Code timed out after {timeout}s")

    if process.returncode != 0:
        error_msg = stderr.decode().strip() if stderr else "Unknown error"
        logger.error("Claude Code failed (exit=%d): %s", process.returncode, error_msg)
        raise ClaudeCodeError(f"Claude Code exited with code {process.returncode}: {error_msg}")

    return _parse_response(stdout.decode(), session_id or "")


def _parse_response(raw_output: str, session_id: str) -> ClaudeResponse:
    """Parse JSON output from Claude Code CLI."""
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        # If output is not valid JSON, treat the raw text as the result
        logger.warning("Claude Code output is not valid JSON, using raw text")
        return ClaudeResponse(result=raw_output.strip(), session_id=session_id)

    # Claude Code --print --output-format json returns a structured response
    result = data.get("result", raw_output.strip())
    returned_session_id = data.get("session_id", session_id)
    is_error = data.get("is_error", False)

    return ClaudeResponse(
        result=result,
        session_id=returned_session_id,
        is_error=is_error,
    )
