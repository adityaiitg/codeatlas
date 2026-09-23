"""Unit tests for GitTracker."""

from pathlib import Path

from codeatlas.git.tracker import GitTracker


def test_git_tracker_non_git(tmp_path: Path):
    tracker = GitTracker(tmp_path)
    assert not tracker.is_git_repo
    assert tracker.get_head_commit() is None
    assert tracker.get_file_churn() == {}
    assert tracker.get_changed_files() == []


def test_git_tracker_valid_repo():
    # Use current git repository path
    current_repo = Path(__file__).parent.parent.parent
    tracker = GitTracker(current_repo)
    assert tracker.is_git_repo
    head = tracker.get_head_commit()
    assert head is not None
    assert len(head) == 40
    churn = tracker.get_file_churn(max_commits=5)
    assert isinstance(churn, dict)


def test_git_tracker_hooks(tmp_path: Path):
    import git

    repo = git.Repo.init(tmp_path)
    tracker = GitTracker(tmp_path)
    assert tracker.is_git_repo

    installed = tracker.install_hooks()
    assert "post-commit" in installed
    assert (tmp_path / ".git" / "hooks" / "post-commit").exists()

    # Re-install should be idempotent
    reinstalled = tracker.install_hooks()
    assert any("already installed" in h for h in reinstalled)

    uninstalled = tracker.uninstall_hooks()
    assert "post-commit" in uninstalled
    assert not (tmp_path / ".git" / "hooks" / "post-commit").exists()
    repo.close()
