"""Python AST parser using standard ast module (Tree-sitter loaded for future multi-language support)."""

from __future__ import annotations

import ast
import hashlib
import logging
import re
from pathlib import Path

from codeatlas.models.chunks import ChunkType, CodeChunk
from codeatlas.models.relationships import Edge, EdgeType
from codeatlas.models.symbols import Symbol, SymbolKind
from codeatlas.parser.base import LanguageParser

logger = logging.getLogger(__name__)


def split_identifier(identifier: str) -> list[str]:
    """Split camelCase and snake_case identifiers into individual words."""
    # Split by underscore or transitions from lower to upper case
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", identifier)
    words = [w.lower() for w in re.split(r"[_\W]+", s1) if w]
    return words


class PythonParser(LanguageParser):
    """Parses Python source code into canonical symbols, chunks, and edges."""

    def __init__(self) -> None:
        pass

    def parse_file(self, file_path: Path) -> tuple[list[Symbol], list[CodeChunk], list[Edge]]:
        """Parse a Python source file and extract symbols, chunks, and edges."""
        try:
            code = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Cannot read %s: %s", file_path, exc)
            return [], [], []

        file_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        symbols: list[Symbol] = []
        chunks: list[CodeChunk] = []
        edges: list[Edge] = []

        # We use Python AST for complete semantic fidelity (docstrings, type hints, calls)
        # and Tree-sitter for structural validation if available
        try:
            tree = ast.parse(code, filename=str(file_path))
        except SyntaxError:
            # If standard AST fails on invalid syntax (e.g. active editing), extract symbols via regex fallback
            lines = code.splitlines()
            fallback_symbols: list[Symbol] = []
            found_identifiers: list[str] = []

            for lineno, line in enumerate(lines, start=1):
                match = re.match(r"^\s*(def|class)\s+([a-zA-Z_0-9]+)", line)
                if match:
                    kind = SymbolKind.CLASS if match.group(1) == "class" else SymbolKind.FUNCTION
                    name = match.group(2)
                    nid = f"{file_path}:{name}"
                    found_identifiers.append(name)
                    fallback_symbols.append(
                        Symbol(
                            node_id=nid,
                            file_path=str(file_path),
                            kind=kind,
                            name=name,
                            start_line=lineno,
                            end_line=lineno,
                            source_code=line,
                            content_hash=hashlib.sha256(line.encode("utf-8")).hexdigest(),
                            language="python",
                        )
                    )

            id_tokens = [t for s in found_identifiers for t in split_identifier(s)]
            chunk = CodeChunk(
                chunk_id=f"{file_path}:1-{len(lines)}",
                symbol_id=None,
                file_path=str(file_path),
                chunk_type=ChunkType.CODE,
                content=code,
                start_line=1,
                end_line=len(lines) or 1,
                language="python",
                content_hash=file_hash,
                is_definition=False,
                identifiers=found_identifiers,
                identifier_tokens=list(set(id_tokens)),
            )
            return fallback_symbols, [chunk], []

        lines = code.splitlines(keepends=True)
        module_doc = ast.get_docstring(tree)

        # 1. Module-level Symbol
        module_id = f"file:{file_path}"
        symbols.append(
            Symbol(
                node_id=module_id,
                file_path=str(file_path),
                kind=SymbolKind.MODULE,
                name=file_path.stem,
                docstring=module_doc,
                start_line=1,
                end_line=len(lines) or 1,
                source_code=code,
                content_hash=file_hash,
                language="python",
            )
        )

        # File chunk
        file_identifiers = [file_path.stem]
        file_tokens = split_identifier(file_path.stem)
        chunks.append(
            CodeChunk(
                chunk_id=f"chunk:{module_id}",
                symbol_id=module_id,
                file_path=str(file_path),
                chunk_type=ChunkType.DOCS if "tests" not in str(file_path) else ChunkType.TEST,
                content=f"# {file_path.name}\n{module_doc or ''}",
                start_line=1,
                end_line=min(20, len(lines) or 1),
                language="python",
                content_hash=hashlib.sha256((module_doc or "").encode()).hexdigest(),
                is_definition=False,
                identifiers=file_identifiers,
                identifier_tokens=file_tokens,
            )
        )

        # 2. Extract Imports
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    edges.append(
                        Edge(
                            source_id=module_id,
                            target_id=f"module:{alias.name}",
                            edge_type=EdgeType.IMPORTS,
                            metadata={"alias": alias.asname, "line": node.lineno},
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                mod_name = node.module or ""
                for alias in node.names:
                    target = f"{mod_name}.{alias.name}" if mod_name else alias.name
                    edges.append(
                        Edge(
                            source_id=module_id,
                            target_id=f"symbol:{target}",
                            edge_type=EdgeType.IMPORTS,
                            metadata={
                                "module": mod_name,
                                "name": alias.name,
                                "alias": alias.asname,
                                "line": node.lineno,
                            },
                        )
                    )

        # 3. Walk classes and functions
        class Visitor(ast.NodeVisitor):
            def __init__(self, parser_inst: PythonParser):
                self.parser = parser_inst
                self.current_class: str | None = None
                self.class_node_id: str | None = None

            def visit_ClassDef(self, node: ast.ClassDef):
                start = node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno
                end = getattr(node, "end_lineno", start)
                class_src = "".join(lines[start - 1 : end])
                class_hash = hashlib.sha256(class_src.encode()).hexdigest()
                doc = ast.get_docstring(node)
                class_node_id = f"{file_path}:{node.name}"

                bases = [self.parser._resolve_callee_expr(b) or ast.unparse(b) for b in node.bases]
                # Extract calls only from class-level statements, avoiding walking methods twice
                class_calls: list[str] = []
                for stmt in node.body:
                    if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        class_calls.extend(self.parser._extract_calls(stmt))
                calls = list(dict.fromkeys(class_calls))

                sym = Symbol(
                    node_id=class_node_id,
                    file_path=str(file_path),
                    kind=SymbolKind.CLASS,
                    name=node.name,
                    parent_symbol=module_id,
                    docstring=doc,
                    start_line=start,
                    end_line=end,
                    source_code=class_src,
                    content_hash=class_hash,
                    language="python",
                    calls=calls,
                    referenced_types=bases,
                )
                symbols.append(sym)

                # Edge: module defines class
                edges.append(
                    Edge(
                        source_id=module_id,
                        target_id=class_node_id,
                        edge_type=EdgeType.DEFINES,
                    )
                )

                # Edge: class inheritance
                for base in bases:
                    edges.append(
                        Edge(
                            source_id=class_node_id,
                            target_id=f"symbol:{base}",
                            edge_type=EdgeType.INHERITS,
                        )
                    )

                # Class chunk (for search: if class is large, index class header + docstring + overview up to 40 lines;
                # methods have their own individual chunks)
                if end - start > 40:
                    class_chunk_src = "".join(lines[start - 1 : min(start + 40, end)])
                    class_chunk_end = min(start + 40, end)
                else:
                    class_chunk_src = class_src
                    class_chunk_end = end

                id_tokens = split_identifier(node.name)
                chunks.append(
                    CodeChunk(
                        chunk_id=f"chunk:{class_node_id}",
                        symbol_id=class_node_id,
                        file_path=str(file_path),
                        chunk_type=ChunkType.TEST
                        if "test" in str(file_path).lower()
                        else ChunkType.CODE,
                        content=class_chunk_src,
                        start_line=start,
                        end_line=class_chunk_end,
                        language="python",
                        content_hash=hashlib.sha256(class_chunk_src.encode()).hexdigest(),
                        is_definition=True,
                        identifiers=[node.name],
                        identifier_tokens=id_tokens,
                    )
                )

                prev_class = self.current_class
                prev_class_id = self.class_node_id
                self.current_class = node.name
                self.class_node_id = class_node_id

                self.generic_visit(node)

                self.current_class = prev_class
                self.class_node_id = prev_class_id

            def visit_FunctionDef(self, node: ast.FunctionDef):
                self._handle_function(node, is_async=False)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                self._handle_function(node, is_async=True)

            def _handle_function(
                self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool
            ):
                start = node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno
                end = getattr(node, "end_lineno", start)
                fn_src = "".join(lines[start - 1 : end])
                fn_hash = hashlib.sha256(fn_src.encode()).hexdigest()
                doc = ast.get_docstring(node)

                prefix = f"{self.current_class}." if self.current_class else ""
                full_name = f"{prefix}{node.name}"
                fn_node_id = f"{file_path}:{full_name}"
                kind = SymbolKind.METHOD if self.current_class else SymbolKind.FUNCTION

                # Build signature
                args_str = ast.unparse(node.args)
                returns_str = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                async_prefix = "async " if is_async else ""
                sig = f"{async_prefix}def {node.name}({args_str}){returns_str}:"

                calls = self.parser._extract_calls(node)

                parent = self.class_node_id if self.class_node_id else module_id
                sym = Symbol(
                    node_id=fn_node_id,
                    file_path=str(file_path),
                    kind=kind,
                    name=node.name,
                    parent_symbol=parent,
                    signature=sig,
                    docstring=doc,
                    start_line=start,
                    end_line=end,
                    source_code=fn_src,
                    content_hash=fn_hash,
                    language="python",
                    calls=calls,
                )
                symbols.append(sym)

                # Edge: parent defines function/method
                edges.append(
                    Edge(
                        source_id=parent,
                        target_id=fn_node_id,
                        edge_type=EdgeType.DEFINES,
                    )
                )

                # Edges: function calls
                for callee in calls:
                    edges.append(
                        Edge(
                            source_id=fn_node_id,
                            target_id=f"call:{callee}",
                            edge_type=EdgeType.CALLS,
                        )
                    )

                # Test relationship
                if "test" in str(file_path).lower() or node.name.startswith("test_"):
                    # Extract what it might be testing
                    target_name = node.name.removeprefix("test_")
                    edges.append(
                        Edge(
                            source_id=fn_node_id,
                            target_id=f"symbol:{target_name}",
                            edge_type=EdgeType.TESTED_BY,
                        )
                    )

                # Chunk
                id_tokens = split_identifier(node.name)
                if self.current_class:
                    id_tokens.extend(split_identifier(self.current_class))
                chunk_type = (
                    ChunkType.TEST
                    if "test" in str(file_path).lower() or node.name.startswith("test_")
                    else ChunkType.CODE
                )

                chunks.append(
                    CodeChunk(
                        chunk_id=f"chunk:{fn_node_id}",
                        symbol_id=fn_node_id,
                        file_path=str(file_path),
                        chunk_type=chunk_type,
                        content=fn_src,
                        start_line=start,
                        end_line=end,
                        language="python",
                        content_hash=fn_hash,
                        is_definition=True,
                        identifiers=[node.name]
                        + ([self.current_class] if self.current_class else []),
                        identifier_tokens=list(set(id_tokens)),
                    )
                )

                # Visit nested statements without re-processing this function
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        self.visit(child)

        visitor = Visitor(self)
        for stmt in tree.body:
            if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                visitor.visit(stmt)

        return symbols, chunks, edges

    @staticmethod
    def _resolve_callee_expr(node: ast.AST) -> str | None:
        """Fast iterative attribute chain resolver avoiding ast.unparse overhead."""
        parts: list[str] = []
        curr: ast.AST = node
        while isinstance(curr, ast.Attribute):
            parts.append(curr.attr)
            curr = curr.value
        if isinstance(curr, ast.Name):
            parts.append(curr.id)
            return ".".join(reversed(parts))
        try:
            return ast.unparse(node)
        except Exception:
            return None

    def _extract_calls(self, node: ast.AST) -> list[str]:
        """Extract all function/method call names inside an AST subtree."""
        calls: list[str] = []
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                callee = self._resolve_callee_expr(n.func)
                if callee:
                    calls.append(callee)
        return list(dict.fromkeys(calls))  # preserve order, unique
