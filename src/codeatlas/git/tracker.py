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

    def install_hooks(self) -> list[str]:
        """Install post-commit and post-merge git hooks for automatic incremental re-indexing."""
        import stat

        if not self._is_git or self._repo is None:
            raise ValueError(f"Not a git repository at {self.repo_path}")

        hooks_dir = Path(self._repo.git_dir) / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)

        hook_code = (
            "\n# >>> CodeAtlas Hook >>>\n"
            "if command -v codeatlas >/dev/null 2>&1; then\n"
            "    (codeatlas index --fast >/dev/null 2>&1 &)\n"
            "fi\n"
            "# <<< CodeAtlas Hook <<<\n"
        )

        installed_hooks = []
        for hook_name in ("post-commit", "post-merge", "post-checkout"):
            hook_file = hooks_dir / hook_name
            if hook_file.exists():
                existing = hook_file.read_text(encoding="utf-8")
                if "# >>> CodeAtlas Hook >>>" in existing:
                    installed_hooks.append(f"{hook_name} (already installed)")
                    continue
                new_content = existing.rstrip() + "\n" + hook_code
            else:
                new_content = "#!/bin/sh\n" + hook_code

            hook_file.write_text(new_content, encoding="utf-8")
            # Make executable
            current_mode = hook_file.stat().st_mode
            hook_file.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            installed_hooks.append(hook_name)

        return installed_hooks

    def uninstall_hooks(self) -> list[str]:
        """Remove CodeAtlas git hooks from the repository."""
        if not self._is_git or self._repo is None:
            raise ValueError(f"Not a git repository at {self.repo_path}")

        hooks_dir = Path(self._repo.git_dir) / "hooks"
        if not hooks_dir.exists():
            return []

        uninstalled_hooks = []
        for hook_name in ("post-commit", "post-merge", "post-checkout"):
            hook_file = hooks_dir / hook_name
            if not hook_file.exists():
                continue

            content = hook_file.read_text(encoding="utf-8")
            if "# >>> CodeAtlas Hook >>>" not in content:
                continue

            # Strip the codeatlas block
            start_marker = "# >>> CodeAtlas Hook >>>"
            end_marker = "# <<< CodeAtlas Hook <<<"
            lines = content.splitlines(keepends=True)
            filtered = []
            skipping = False
            for line in lines:
                if start_marker in line:
                    skipping = True
                    continue
                if end_marker in line:
                    skipping = False
                    continue
                if not skipping:
                    filtered.append(line)

            remaining = "".join(filtered).strip()
            # If nothing left or only #!/bin/sh, remove file
            if not remaining or remaining == "#!/bin/sh":
                hook_file.unlink()
            else:
                hook_file.write_text(remaining + "\n", encoding="utf-8")

            uninstalled_hooks.append(hook_name)

        return uninstalled_hooks

