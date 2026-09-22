"""Unit tests for the Python AST parser and identifier tokenization."""

from pathlib import Path

from codeatlas.models.relationships import EdgeType
from codeatlas.models.symbols import SymbolKind
from codeatlas.parser.python_parser import PythonParser, split_identifier


def test_split_identifier():
    assert split_identifier("camelCase") == ["camel", "case"]
    assert split_identifier("PascalCase") == ["pascal", "case"]
    assert split_identifier("snake_case_variable") == ["snake", "case", "variable"]
    assert split_identifier("AuthMiddleware") == ["auth", "middleware"]
    assert split_identifier("get_recommendations") == ["get", "recommendations"]


def test_parse_sample_file(tmp_path: Path):
    sample_code = '''"""Test module docstring."""

from auth.token import parse_jwt


class SampleService:
    """A sample service for testing."""

    def __init__(self, key: str):
        self.key = key

    def process(self, token: str) -> dict:
        """Process an input token."""
        return parse_jwt(token)
'''
    test_file = tmp_path / "sample.py"
    test_file.write_text(sample_code, encoding="utf-8")

    parser = PythonParser()
    symbols, chunks, edges = parser.parse_file(test_file)

    # Validate symbols
    symbol_names = [s.name for s in symbols]
    assert "sample" in symbol_names  # module
    assert "SampleService" in symbol_names  # class
    assert "process" in symbol_names  # method

    # Validate class symbol
    class_sym = next(s for s in symbols if s.name == "SampleService")
    assert class_sym.kind == SymbolKind.CLASS
    assert "sample service" in (class_sym.docstring or "").lower()

    # Validate method symbol
    method_sym = next(s for s in symbols if s.name == "process")
    assert method_sym.kind == SymbolKind.METHOD
    assert method_sym.parent_symbol == class_sym.node_id
    assert "parse_jwt" in method_sym.calls

    # Validate edges
    edge_types = [e.edge_type for e in edges]
    assert EdgeType.DEFINES in edge_types
    assert EdgeType.IMPORTS in edge_types
    assert EdgeType.CALLS in edge_types

    # Validate chunks
    assert len(chunks) >= 2
    assert any(c.is_definition for c in chunks)
    assert any("process" in c.identifiers for c in chunks)


def test_parse_decorated_functions_and_classes(tmp_path: Path):
    code = """from functools import lru_cache

@decorator_a
@decorator_b(param=1)
class DecoratedClass:
    @property
    @lru_cache(maxsize=128)
    def compute(self):
        return 42
"""
    test_file = tmp_path / "decorated.py"
    test_file.write_text(code, encoding="utf-8")

    parser = PythonParser()
    symbols, chunks, _ = parser.parse_file(test_file)

    class_sym = next(s for s in symbols if s.name == "DecoratedClass")
    assert "@decorator_a" in class_sym.source_code
    assert "@decorator_b" in class_sym.source_code
    assert class_sym.start_line == 3

    func_sym = next(s for s in symbols if s.name == "compute")
    assert "@property" in func_sym.source_code
    assert "@lru_cache" in func_sym.source_code
    assert func_sym.start_line == 6


def test_file_scanner_symlink_and_pruning(tmp_path: Path):
    from codeatlas.config import Settings
    from codeatlas.parser.file_scanner import FileScanner

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "main.py").write_text("print('hello')", encoding="utf-8")

    # Ignored directory
    venv_dir = repo_dir / ".venv"
    venv_dir.mkdir()
    (venv_dir / "ignored.py").write_text("print('ignored')", encoding="utf-8")

    # Outside directory with sensitive file
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    secret_file = outside_dir / "secret.py"
    secret_file.write_text("password = 'secret'", encoding="utf-8")

    # Symlink pointing outside repo
    symlink_file = repo_dir / "leak.py"
    try:
        symlink_file.symlink_to(secret_file)
    except OSError:
        pass  # In case OS permissions disallow symlinks

    # Symlink pointing inside repo
    internal_symlink = repo_dir / "main_link.py"
    try:
        internal_symlink.symlink_to(repo_dir / "main.py")
    except OSError:
        pass

    scanner = FileScanner(Settings(repo_path=repo_dir))
    scanned = scanner.scan()
    scanned_paths = [p.name for p, _ in scanned]

    assert "main.py" in scanned_paths
    assert "ignored.py" not in scanned_paths
    assert "leak.py" not in scanned_paths
    if internal_symlink.exists():
        assert "main_link.py" in scanned_paths


