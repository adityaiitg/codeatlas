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
