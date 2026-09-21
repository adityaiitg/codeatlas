"""Generic multi-language parser for TypeScript, JavaScript, Go, Rust, and other languages."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from codeatlas.models.chunks import ChunkType, CodeChunk
from codeatlas.models.relationships import Edge, EdgeType
from codeatlas.models.symbols import Symbol, SymbolKind
from codeatlas.parser.base import LanguageParser
from codeatlas.parser.python_parser import split_identifier

logger = logging.getLogger(__name__)

# Patterns for extracting definitions across modern languages
CLASS_PATTERN = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?(?:class|interface|struct|type)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)

FUNCTION_PATTERN = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:pub\s+)?(?:fn|func|function)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)

CONST_FUNC_PATTERN = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_][A-Za-z0-9_]*)\s*=>",
    re.MULTILINE,
)

IMPORT_PATTERN = re.compile(
    r"""(?:import\s+(?:\{[^}]*\}|\*\s+as\s+[A-Za-z0-9_]+|[A-Za-z0-9_]+)\s+from\s+['"]([^'"]+)['"]|import\s+['"]([^'"]+)['"]|use\s+([A-Za-z0-9_:]+);)""",
    re.MULTILINE,
)


class GenericCodeParser(LanguageParser):
    """Extracts symbols, chunks, and import relationships from non-Python codebases."""

    def __init__(self, language: str):
        self.language = language

    def parse_file(self, file_path: Path) -> tuple[list[Symbol], list[CodeChunk], list[Edge]]:
        """Parse source file extracting classes, functions, and import edges."""
        try:
            code = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Cannot read file %s: %s", file_path, exc)
            return [], [], []

        file_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        symbols: list[Symbol] = []
        chunks: list[CodeChunk] = []
        edges: list[Edge] = []
        lines = code.splitlines(keepends=True)
        total_lines = len(lines) or 1
        module_id = f"file:{file_path}"

        # 1. Module level symbol
        symbols.append(
            Symbol(
                node_id=module_id,
                file_path=str(file_path),
                kind=SymbolKind.MODULE,
                name=file_path.stem,
                start_line=1,
                end_line=total_lines,
                source_code=code,
                content_hash=file_hash,
                language=self.language,
            )
        )

        # File chunk
        chunks.append(
            CodeChunk(
                chunk_id=f"chunk:{module_id}",
                symbol_id=module_id,
                file_path=str(file_path),
                chunk_type=ChunkType.DOCS
                if "test" not in str(file_path).lower()
                else ChunkType.TEST,
                content=f"# {file_path.name}\n" + "".join(lines[:30]),
                start_line=1,
                end_line=min(30, total_lines),
                language=self.language,
                content_hash=hashlib.sha256("".join(lines[:30]).encode()).hexdigest(),
                is_definition=False,
                identifiers=[file_path.stem],
                identifier_tokens=split_identifier(file_path.stem),
            )
        )

        # 2. Extract Imports
        for match in IMPORT_PATTERN.finditer(code):
            target = next(g for g in match.groups() if g is not None)
            edges.append(
                Edge(
                    source_id=module_id,
                    target_id=f"module:{target}",
                    edge_type=EdgeType.IMPORTS,
                    metadata={"target": target},
                )
            )

        # Helper to find line numbers
        def get_line_num(char_idx: int) -> int:
            return code[:char_idx].count("\n") + 1

        # 3. Extract Classes / Interfaces / Structs
        for match in CLASS_PATTERN.finditer(code):
            name = match.group(1)
            start_l = get_line_num(match.start())
            # Approximate end line (up to next empty line or next class/func)
            end_l = min(start_l + 40, total_lines)
            class_src = "".join(lines[start_l - 1 : end_l])
            node_id = f"{file_path}:{name}"

            symbols.append(
                Symbol(
                    node_id=node_id,
                    file_path=str(file_path),
                    kind=SymbolKind.CLASS,
                    name=name,
                    parent_symbol=module_id,
                    start_line=start_l,
                    end_line=end_l,
                    source_code=class_src,
                    content_hash=hashlib.sha256(class_src.encode()).hexdigest(),
                    language=self.language,
                )
            )

            edges.append(
                Edge(
                    source_id=module_id,
                    target_id=node_id,
                    edge_type=EdgeType.DEFINES,
                )
            )

            chunks.append(
                CodeChunk(
                    chunk_id=f"chunk:{node_id}",
                    symbol_id=node_id,
                    file_path=str(file_path),
                    chunk_type=ChunkType.CODE,
                    content=class_src,
                    start_line=start_l,
                    end_line=end_l,
                    language=self.language,
                    content_hash=hashlib.sha256(class_src.encode()).hexdigest(),
                    is_definition=True,
                    identifiers=[name],
                    identifier_tokens=split_identifier(name),
                )
            )

        # 4. Extract Functions
        seen_funcs = set()
        for pattern in (FUNCTION_PATTERN, CONST_FUNC_PATTERN):
            for match in pattern.finditer(code):
                name = match.group(1)
                if name in seen_funcs:
                    continue
                seen_funcs.add(name)

                start_l = get_line_num(match.start())
                end_l = min(start_l + 30, total_lines)
                fn_src = "".join(lines[start_l - 1 : end_l])
                node_id = f"{file_path}:{name}"

                symbols.append(
                    Symbol(
                        node_id=node_id,
                        file_path=str(file_path),
                        kind=SymbolKind.FUNCTION,
                        name=name,
                        parent_symbol=module_id,
                        start_line=start_l,
                        end_line=end_l,
                        source_code=fn_src,
                        content_hash=hashlib.sha256(fn_src.encode()).hexdigest(),
                        language=self.language,
                    )
                )

                edges.append(
                    Edge(
                        source_id=module_id,
                        target_id=node_id,
                        edge_type=EdgeType.DEFINES,
                    )
                )

                chunks.append(
                    CodeChunk(
                        chunk_id=f"chunk:{node_id}",
                        symbol_id=node_id,
                        file_path=str(file_path),
                        chunk_type=ChunkType.CODE,
                        content=fn_src,
                        start_line=start_l,
                        end_line=end_l,
                        language=self.language,
                        content_hash=hashlib.sha256(fn_src.encode()).hexdigest(),
                        is_definition=True,
                        identifiers=[name],
                        identifier_tokens=split_identifier(name),
                    )
                )

        return symbols, chunks, edges
