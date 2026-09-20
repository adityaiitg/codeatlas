"""Living wiki generator synthesizing architecture, modules, and workflows."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

from codeatlas.config import Settings
from codeatlas.graph.builder import GraphBuilder
from codeatlas.graph.schema import connect_db
from codeatlas.llm.client import LLMClient
from codeatlas.models.chunks import ChunkType, CodeChunk


class WikiGenerator:
    """Generates a structured, living repository wiki with Mermaid diagrams."""

    def __init__(self, settings: Settings, use_llm: bool = False):
        self.settings = settings
        self.wiki_dir = settings.wiki_dir
        self.db_path = settings.db_path
        self.llm = LLMClient(settings) if use_llm else None

    def generate(self) -> list[Path]:
        """Generate full living wiki documentation tree from SQLite graph."""
        if not self.db_path.exists():
            return []

        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        modules_dir = self.wiki_dir / "modules"
        modules_dir.mkdir(parents=True, exist_ok=True)

        conn = connect_db(self.db_path)
        conn.row_factory = sqlite3.Row

        # Load symbols
        symbols_rows = conn.execute(
            """SELECT node_id, file_path, kind, name, parent_id, signature, docstring,
                      start_line, end_line
               FROM symbols ORDER BY file_path, start_line"""
        ).fetchall()

        # Load edges
        edges_rows = conn.execute(
            """SELECT source_id, target_id, edge_type, metadata FROM edges"""
        ).fetchall()
        conn.close()

        # Organize by file / module
        files_map = defaultdict(list)
        for row in symbols_rows:
            files_map[row["file_path"]].append(dict(row))

        generated_files: list[Path] = []

        # 1. Generate Architecture Overview
        arch_path = self._generate_architecture(files_map, edges_rows)
        generated_files.append(arch_path)

        # 2. Generate Module Pages
        for file_path, syms in files_map.items():
            mod_path = self._generate_module_page(file_path, syms, edges_rows, modules_dir)
            if mod_path:
                generated_files.append(mod_path)

        # 3. Generate Workflows Trace
        wf_path = self._generate_workflows(files_map, edges_rows)
        generated_files.append(wf_path)

        # 4. Generate Main Index
        index_path = self._generate_index(files_map, generated_files)
        generated_files.insert(0, index_path)

        # 5. Index generated wiki into CodeAtlas search index
        self._index_wiki_chunks(generated_files)

        return generated_files

    def _generate_architecture(self, files_map: dict, edges_rows: list) -> Path:
        """Generate architecture.md with Mermaid diagrams."""
        arch_file = self.wiki_dir / "architecture.md"

        # Build module dependency links for Mermaid
        mermaid_lines = ["```mermaid", "flowchart TD"]
        module_nodes = set()
        dep_edges = set()

        for edge in edges_rows:
            if edge["edge_type"] in ("IMPORTS", "CALLS"):
                src = edge["source_id"].split(":")[0].replace("/", "_").replace(".", "_")
                tgt = edge["target_id"].split(":")[0].replace("/", "_").replace(".", "_")
                if src and tgt and src != tgt and len(src) < 40 and len(tgt) < 40:
                    module_nodes.add(src)
                    module_nodes.add(tgt)
                    dep_edges.add((src, tgt))

        for node in sorted(module_nodes)[:20]:
            clean_name = node.split("_")[-1]
            mermaid_lines.append(f'    {node}["{clean_name}"]')

        for src, tgt in sorted(dep_edges)[:30]:
            mermaid_lines.append(f"    {src} --> {tgt}")

        mermaid_lines.append("```")
        diagram_block = "\n".join(mermaid_lines) if dep_edges else "_No cross-file relations detected._"

        content = f"""# System Architecture

## Overview
This living architectural specification is automatically generated from the AST and Knowledge Graph of the codebase.

## Subsystem Dependency Graph
{diagram_block}

## Discovered Modules & Components
| File Path | Defined Symbols | Summary |
| :--- | :--- | :--- |
"""
        for fpath, syms in sorted(files_map.items()):
            names = ", ".join(f"`{s['name']}`" for s in syms if s["kind"] in ("class", "function"))
            summary = next((s["docstring"] for s in syms if s["docstring"]), "Source module")
            first_line = summary.split("\n")[0] if summary else ""
            rel_path = Path(fpath).name
            content += f"| [`{rel_path}`](./modules/{Path(fpath).stem}.md) | {names or 'Module'} | {first_line} |\n"

        arch_file.write_text(content, encoding="utf-8")
        return arch_file

    def _generate_module_page(
        self, file_path: str, syms: list[dict], edges_rows: list, modules_dir: Path
    ) -> Path:
        """Generate individual module documentation file."""
        stem = Path(file_path).stem
        mod_file = modules_dir / f"{stem}.md"

        file_doc = next((s["docstring"] for s in syms if s["kind"] == "module" and s["docstring"]), "")

        content = f"""# Module: `{Path(file_path).name}`

**Source Location:** `{file_path}`

## Description
{file_doc or 'Source implementation module.'}

## Defined Classes & Functions
"""
        for s in syms:
            if s["kind"] in ("class", "function", "method"):
                sig = f"\n```python\n{s['signature']}\n```\n" if s["signature"] else ""
                doc = f"> {s['docstring']}\n" if s["docstring"] else ""
                content += f"### `{s['name']}` ({s['kind']})\n"
                content += f"- **Lines:** {s['start_line']}–{s['end_line']}\n"
                if sig:
                    content += sig
                if doc:
                    content += doc
                content += "\n"

        # Outgoing calls from this file
        file_calls = [
            e["target_id"].removeprefix("call:")
            for e in edges_rows
            if file_path in e["source_id"] and e["edge_type"] == "CALLS"
        ]
        if file_calls:
            content += "## Invoked External Symbols\n"
            for call in sorted(set(file_calls)):
                content += f"- `{call}`\n"

        mod_file.write_text(content, encoding="utf-8")
        return mod_file

    def _generate_workflows(self, files_map: dict, edges_rows: list) -> Path:
        """Generate workflows.md tracing call sequences."""
        wf_file = self.wiki_dir / "workflows.md"

        calls_map = defaultdict(list)
        for e in edges_rows:
            if e["edge_type"] == "CALLS":
                calls_map[e["source_id"]].append(e["target_id"])

        content = """# End-to-End Workflows & Execution Paths

This document maps execution paths through the call graph.

## Detected Call Sequences
"""
        for src, tgts in list(calls_map.items())[:15]:
            src_name = src.split(":")[-1]
            content += f"### Flow from `{src_name}`\n"
            content += "```mermaid\nsequenceDiagram\n"
            for tgt in tgts:
                tgt_name = tgt.removeprefix("call:").replace(".", "_")
                clean_src = src_name.replace(".", "_")
                content += f"    {clean_src}->>{tgt_name}: invoke\n"
            content += "```\n\n"

        if not calls_map:
            content += "\n_No multi-step call sequences detected in current graph._\n"

        wf_file.write_text(content, encoding="utf-8")
        return wf_file

    def _generate_index(self, files_map: dict, generated_files: list[Path]) -> Path:
        """Generate index.md linking to all chapters."""
        idx_file = self.wiki_dir / "index.md"
        content = """# CodeAtlas Living Wiki

Welcome to the automated living documentation for this repository.

## Table of Contents
- [System Architecture](./architecture.md)
- [Execution Workflows](./workflows.md)
- [Module Directory](./modules/)

## Module Index
"""
        for fpath in sorted(files_map.keys()):
            stem = Path(fpath).stem
            content += f"- [{Path(fpath).name}](./modules/{stem}.md)\n"

        idx_file.write_text(content, encoding="utf-8")
        return idx_file

    def _index_wiki_chunks(self, wiki_files: list[Path]):
        """Index generated wiki pages into CodeAtlas search index."""
        builder = GraphBuilder(self.db_path)
        wiki_chunks: list[CodeChunk] = []

        for p in wiki_files:
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue

            lines = text.splitlines()
            chunk_id = f"wiki:{p.stem}"
            wiki_chunks.append(
                CodeChunk(
                    chunk_id=chunk_id,
                    symbol_id=None,
                    file_path=str(p),
                    chunk_type=ChunkType.WIKI,
                    content=text,
                    start_line=1,
                    end_line=len(lines) or 1,
                    language="markdown",
                    content_hash=str(hash(text)),
                    is_definition=False,
                    identifiers=[p.stem, "wiki", "documentation"],
                    identifier_tokens=[p.stem, "wiki", "docs"],
                )
            )

        builder.add_chunks(wiki_chunks)
        builder.close()
