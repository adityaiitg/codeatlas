"""CodeAtlas parser package."""

from codeatlas.parser.base import LanguageParser
from codeatlas.parser.file_scanner import FileScanner
from codeatlas.parser.python_parser import PythonParser, split_identifier

__all__ = [
    "LanguageParser",
    "FileScanner",
    "PythonParser",
    "split_identifier",
]
