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
