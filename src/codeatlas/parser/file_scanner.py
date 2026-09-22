"""File discovery and language detection."""

from __future__ import annotations

import os
from pathlib import Path

from codeatlas.config import Settings

LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
}


class FileScanner:
    """Walks a repository directory tree and discovers source files."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.repo_resolved = self.settings.repo_path.resolve()

    def scan(self) -> list[tuple[Path, str]]:
        """Return a list of (file_path, language) tuples for all source files."""
        results: list[tuple[Path, str]] = []
        repo_path = self.settings.repo_path
        repo_str = str(repo_path)
        repo_resolved = self.repo_resolved

        for root, dirs, files in os.walk(repo_str, followlinks=False):
            # In-place directory pruning: do not traverse ignored or hidden directories
            dirs[:] = [d for d in dirs if not self._is_dir_ignored(d)]

            root_path = Path(root)
            for f in sorted(files):
                if f.startswith("."):
                    continue
                ext = os.path.splitext(f)[1]
                lang = LANGUAGE_MAP.get(ext)
                if not lang:
                    continue

                full_path = root_path / f
                if full_path.is_symlink():
                    try:
                        resolved = full_path.resolve()
                        if not resolved.is_relative_to(repo_resolved):
                            continue
                    except (OSError, ValueError):
                        continue

                results.append((full_path, lang))

        return sorted(results, key=lambda x: str(x[0]))

    def _is_dir_ignored(self, dir_name: str) -> bool:
        """Check if a directory name should be ignored."""
        if dir_name.startswith("."):
            return True
        for pattern in self.settings.ignore_patterns:
            if pattern.startswith("*"):
                if dir_name.endswith(pattern[1:]):
                    return True
            elif dir_name == pattern:
                return True
        return False

    def _is_ignored(self, path: Path) -> bool:
        """Check if a path matches any ignore pattern or is a hidden/AppleDouble file."""
        if path.name.startswith("."):
            return True
        try:
            parts = path.relative_to(self.settings.repo_path).parts
        except ValueError:
            parts = path.parts
        for part in parts:
            if part.startswith(".") and part != ".":
                return True
            for pattern in self.settings.ignore_patterns:
                if pattern.startswith("*"):
                    if part.endswith(pattern[1:]):
                        return True
                elif part == pattern:
                    return True
        return False

