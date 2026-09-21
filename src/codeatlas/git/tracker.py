"""Git repository integration: commit history, churn metrics, and diff tracking."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GitTracker:
    """Extracts commit history, file churn, and diffs from a git repository."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self._repo = None
        self._is_git = False

        try:
            import git

            self._repo = git.Repo(repo_path, search_parent_directories=True)
            self._is_git = True
        except ImportError:
            logger.debug("GitPython not installed, git features disabled")
            self._is_git = False
        except Exception as exc:
            logger.debug("Not a git repository at %s: %s", repo_path, exc)
            self._is_git = False

    @property
    def is_git_repo(self) -> bool:
        """Return True if path is within a valid git repository."""
        return self._is_git

    def get_head_commit(self) -> str | None:
        """Return the current HEAD commit hash."""
        if not self._is_git or self._repo is None:
            return None
        try:
            return self._repo.head.commit.hexsha
        except Exception:
            return None

    def get_file_churn(self, max_commits: int = 200) -> dict[str, int]:
        """Compute commit counts per file over recent commits."""
        if not self._is_git or self._repo is None:
            return {}

        churn: dict[str, int] = {}
        try:
            for commit in self._repo.iter_commits(max_count=max_commits):
                for file_path in commit.stats.files:
                    full_path = str(self.repo_path / file_path)
                    churn[full_path] = churn.get(full_path, 0) + 1
        except Exception as e:
            logger.debug(f"Could not compute file churn: {e}")

        return churn

    def get_changed_files(self, since_commit: str | None = None) -> list[str]:
        """Get list of files modified, added, or deleted since a specific commit."""
        if not self._is_git or self._repo is None:
            return []

        changed: set[str] = set()
        try:
            if since_commit:
                diff = self._repo.commit(since_commit).diff(None)
                for item in diff:
                    if item.a_path:
                        changed.add(str(self.repo_path / item.a_path))
                    if item.b_path:
                        changed.add(str(self.repo_path / item.b_path))
            else:
                # Uncommitted working tree changes
                for item in self._repo.index.diff(None):
                    if item.a_path:
                        changed.add(str(self.repo_path / item.a_path))
                # Untracked files
                for untracked in self._repo.untracked_files:
                    changed.add(str(self.repo_path / untracked))
        except Exception as e:
            logger.debug(f"Could not get diff: {e}")

        return sorted(changed)
