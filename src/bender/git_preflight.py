"""Reusable git preflight/postflight checks for repo-changing work.

Extracted from `order_dispatcher.py`'s `ensure_clean_and_current()` (the
retired Order protocol's common\\scripts\\order_dispatcher.py) with every
Order-number/Order-MD/OrderLock dependency removed. Takes a plain repo path
and nothing else, so both the Bender/Codexy Slack-channel execution path and
a direct CLI session can call the exact same check before touching any repo.

Invocation (installed editable, so reachable as `python -m bender.git_preflight`
from any cwd):

    python -m bender.git_preflight preflight <repo_path> [--no-pull]
    python -m bender.git_preflight postflight <repo_path> [--before <sha>]

Never runs `git reset --hard`, `git clean`, `git push --force`, or any repo
create/delete/remote-change command. Never auto-cleans on failure -- a
BLOCKED/NOT_VERIFIED verdict just reports state and leaves it untouched.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


class GitCheckError(RuntimeError):
    """Raised when a git command needed to produce a verdict itself fails."""


def _run(args: list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _git(cwd: Path, *args: str, timeout: int = 120) -> str:
    result = _run(["git", *args], cwd, timeout=timeout)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise GitCheckError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _porcelain_lines(cwd: Path) -> list[str]:
    out = _git(cwd, "status", "--porcelain=v1")
    return out.splitlines() if out else []


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def _upstream_ref(cwd: Path) -> str | None:
    """The branch's actual configured upstream (e.g. `origin/main`), not an
    assumed `origin/<branch>` -- those differ whenever the local branch name
    doesn't match its remote-tracking branch."""
    try:
        return _git(cwd, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    except GitCheckError:
        return None


def _ahead_behind(cwd: Path, upstream: str | None) -> tuple[int | None, int | None]:
    if not upstream:
        return None, None
    try:
        counts = _git(cwd, "rev-list", "--left-right", "--count", f"HEAD...{upstream}")
        ahead_s, behind_s = counts.split()
        return int(ahead_s), int(behind_s)
    except GitCheckError:
        return None, None


def preflight(repo_path: Path, pull: bool = True) -> dict[str, Any]:
    """Verify a repo is clean before any new change work starts.

    Mirrors ensure_clean_and_current(): refuse if the tree is dirty, then
    (optionally) fast-forward from origin, then refuse again if that pull
    somehow left the tree dirty. `git pull --ff-only` is atomic -- it either
    fast-forwards cleanly or fails without touching the working tree, so a
    failed pull is reported but does not by itself BLOCK the preflight;
    remote sync is advisory (the branch may have no remote, be offline, or
    be legitimately ahead-only).
    """
    repo_path = Path(repo_path).resolve()
    report: dict[str, Any] = {"path": str(repo_path), "checked_at": _now()}

    if not repo_path.is_dir():
        return {**report, "verdict": "BLOCKED", "reason": f"path does not exist: {repo_path}"}

    try:
        report["repo_root"] = _git(repo_path, "rev-parse", "--show-toplevel")
    except GitCheckError as exc:
        return {**report, "verdict": "BLOCKED", "reason": f"not a git repository: {exc}"}

    dirty = _porcelain_lines(repo_path)
    if dirty:
        return {
            **report,
            "verdict": "BLOCKED",
            "reason": "working tree is not clean; refusing new changes",
            "dirty": True,
            "changes": dirty,
        }

    head_before = _git(repo_path, "rev-parse", "HEAD")
    branch = _git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")

    pulled = False
    if pull:
        try:
            _git(repo_path, "fetch", "origin", timeout=180)
            _git(repo_path, "pull", "--ff-only", timeout=180)
            pulled = True
        except GitCheckError as exc:
            report["pull_error"] = str(exc)

    dirty_after_pull = _porcelain_lines(repo_path)
    if dirty_after_pull:
        return {
            **report,
            "verdict": "BLOCKED",
            "reason": "tree became dirty after pull; refusing new changes",
            "dirty": True,
            "changes": dirty_after_pull,
        }

    head = _git(repo_path, "rev-parse", "HEAD")
    try:
        remote = _git(repo_path, "remote", "get-url", "origin")
    except GitCheckError:
        remote = None
    upstream = _upstream_ref(repo_path)
    ahead, behind = _ahead_behind(repo_path, upstream)

    return {
        **report,
        "verdict": "PASS",
        "dirty": False,
        "branch": branch,
        "upstream": upstream,
        "head_before": head_before,
        "head": head,
        "pulled": pulled,
        "remote": remote,
        "ahead": ahead,
        "behind": behind,
    }


def postflight(repo_path: Path, before_head: str | None = None) -> dict[str, Any]:
    """Verify the result of change work already done by the caller.

    Read-only: reports diff/commit/push/final-status, but performs no git
    write of any kind (no add/commit/push here -- that stays the agent's own
    job under the new policy). Never resets/cleans/stashes on a bad verdict.
    """
    repo_path = Path(repo_path).resolve()
    report: dict[str, Any] = {"path": str(repo_path), "checked_at": _now()}

    try:
        _git(repo_path, "rev-parse", "--show-toplevel")
    except GitCheckError as exc:
        return {**report, "verdict": "NOT_VERIFIED", "reason": f"not a git repository: {exc}"}

    dirty = _porcelain_lines(repo_path)
    head = _git(repo_path, "rev-parse", "HEAD")
    branch = _git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")

    changed_since_before = None
    if before_head:
        try:
            diff_out = _git(repo_path, "diff", "--name-status", before_head, head)
            changed_since_before = diff_out.splitlines() if diff_out else []
        except GitCheckError as exc:
            report["diff_error"] = str(exc)

    upstream = _upstream_ref(repo_path)
    ahead, behind = _ahead_behind(repo_path, upstream)

    if dirty:
        verdict, reason = "NOT_VERIFIED", "working tree not clean after task"
    elif upstream is None:
        verdict, reason = "NOT_VERIFIED", "no upstream configured; push status unknown"
    elif ahead and ahead > 0:
        verdict, reason = "UNPUSHED", f"{ahead} commit(s) ahead of {upstream}; not pushed"
    else:
        verdict, reason = "PASS", "clean and pushed to upstream"

    return {
        **report,
        "verdict": verdict,
        "reason": reason,
        "dirty": bool(dirty),
        "changes": dirty,
        "branch": branch,
        "upstream": upstream,
        "head_before": before_head,
        "head": head,
        "changed_since_before": changed_since_before,
        "ahead": ahead,
        "behind": behind,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preflight")
    p_pre.add_argument("repo_path")
    p_pre.add_argument("--no-pull", action="store_true")

    p_post = sub.add_parser("postflight")
    p_post.add_argument("repo_path")
    p_post.add_argument("--before")

    args = parser.parse_args(argv)

    if args.command == "preflight":
        result = preflight(Path(args.repo_path), pull=not args.no_pull)
    else:
        result = postflight(Path(args.repo_path), before_head=args.before)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
