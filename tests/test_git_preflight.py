"""Tests for the reusable git preflight/postflight checks."""

import subprocess
from pathlib import Path

import pytest

from bender.git_preflight import postflight, preflight


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


@pytest.fixture
def clean_repo(tmp_path: Path) -> Path:
    """A repo with a real (local, bare) remote and an upstream-tracked
    branch, so postflight's ahead/behind-against-upstream logic has
    something real to check -- not just a repo with no remote at all."""
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "remote", "add", "origin", str(remote))
    (repo / "a.txt").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial")
    _git(repo, "push", "-q", "-u", "origin", "main")
    return repo


class TestPreflight:
    def test_clean_repo_passes(self, clean_repo: Path) -> None:
        result = preflight(clean_repo, pull=False)
        assert result["verdict"] == "PASS"
        assert result["dirty"] is False
        assert result["head"] == result["head_before"]
        assert result["upstream"] == "origin/main"

    def test_dirty_repo_is_blocked(self, clean_repo: Path) -> None:
        (clean_repo / "a.txt").write_text("changed\n", encoding="utf-8")
        result = preflight(clean_repo, pull=False)
        assert result["verdict"] == "BLOCKED"
        assert result["dirty"] is True
        assert any("a.txt" in line for line in result["changes"])

    def test_untracked_file_is_blocked(self, clean_repo: Path) -> None:
        (clean_repo / "new.txt").write_text("new\n", encoding="utf-8")
        result = preflight(clean_repo, pull=False)
        assert result["verdict"] == "BLOCKED"
        assert result["dirty"] is True

    def test_nonexistent_path_is_blocked(self, tmp_path: Path) -> None:
        result = preflight(tmp_path / "does-not-exist", pull=False)
        assert result["verdict"] == "BLOCKED"
        assert "does not exist" in result["reason"]

    def test_non_git_directory_is_blocked(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        result = preflight(plain, pull=False)
        assert result["verdict"] == "BLOCKED"
        assert "not a git repository" in result["reason"]

    def test_repeated_calls_are_deterministic(self, clean_repo: Path) -> None:
        first = preflight(clean_repo, pull=False)
        second = preflight(clean_repo, pull=False)
        assert first["verdict"] == second["verdict"] == "PASS"
        assert first["head"] == second["head"]

    def test_never_mutates_a_dirty_tree(self, clean_repo: Path) -> None:
        (clean_repo / "a.txt").write_text("changed\n", encoding="utf-8")
        before_status = subprocess.run(
            ["git", "status", "--porcelain=v1"], cwd=str(clean_repo), capture_output=True, text=True
        ).stdout
        preflight(clean_repo, pull=False)
        after_status = subprocess.run(
            ["git", "status", "--porcelain=v1"], cwd=str(clean_repo), capture_output=True, text=True
        ).stdout
        assert before_status == after_status


class TestPostflight:
    def test_clean_and_pushed_passes(self, clean_repo: Path) -> None:
        before = preflight(clean_repo, pull=False)["head"]
        (clean_repo / "a.txt").write_text("changed\n", encoding="utf-8")
        _git(clean_repo, "add", "-A")
        _git(clean_repo, "commit", "-q", "-m", "change")
        _git(clean_repo, "push", "-q")
        result = postflight(clean_repo, before_head=before)
        assert result["verdict"] == "PASS"
        assert result["dirty"] is False
        assert result["ahead"] == 0
        assert result["changed_since_before"] == ["M\ta.txt"]

    def test_committed_but_unpushed_is_unpushed(self, clean_repo: Path) -> None:
        before = preflight(clean_repo, pull=False)["head"]
        (clean_repo / "a.txt").write_text("changed\n", encoding="utf-8")
        _git(clean_repo, "add", "-A")
        _git(clean_repo, "commit", "-q", "-m", "change")
        result = postflight(clean_repo, before_head=before)
        assert result["verdict"] == "UNPUSHED"
        assert result["dirty"] is False
        assert result["ahead"] == 1

    def test_uncommitted_leftovers_are_not_verified(self, clean_repo: Path) -> None:
        before = preflight(clean_repo, pull=False)["head"]
        (clean_repo / "a.txt").write_text("changed\n", encoding="utf-8")
        result = postflight(clean_repo, before_head=before)
        assert result["verdict"] == "NOT_VERIFIED"
        assert result["dirty"] is True

    def test_no_upstream_is_not_verified(self, tmp_path: Path) -> None:
        repo = tmp_path / "solo"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "Test")
        (repo / "a.txt").write_text("hello\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "initial")
        result = postflight(repo)
        assert result["verdict"] == "NOT_VERIFIED"
        assert result["upstream"] is None

    def test_never_resets_or_cleans_on_bad_verdict(self, clean_repo: Path) -> None:
        before = preflight(clean_repo, pull=False)["head"]
        (clean_repo / "a.txt").write_text("changed\n", encoding="utf-8")
        (clean_repo / "untracked.txt").write_text("stray\n", encoding="utf-8")
        postflight(clean_repo, before_head=before)
        # The leftover changes must still be there -- postflight only reports.
        assert (clean_repo / "untracked.txt").exists()
        assert (clean_repo / "a.txt").read_text(encoding="utf-8") == "changed\n"

    def test_non_git_directory_is_not_verified(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        result = postflight(plain)
        assert result["verdict"] == "NOT_VERIFIED"
