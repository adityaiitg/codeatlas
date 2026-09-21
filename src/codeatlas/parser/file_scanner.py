"""File discovery and language detection."""

from __future__ import annotations

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

    def scan(self) -> list[tuple[Path, str]]:
        """Return a list of (file_path, language) tuples for all source files."""
        results: list[tuple[Path, str]] = []
        for path in sorted(self.settings.repo_path.rglob("*")):
            if path.is_file() and not self._is_ignored(path):
                lang = LANGUAGE_MAP.get(path.suffix)
                if lang:
                    results.append((path, lang))
        return results

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
