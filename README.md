# CodeAtlas

Local codebase intelligence engine: AST-driven code IR, deterministic knowledge graph, living repo wiki, and hybrid search CLI.

## Features

- **Canonical Code IR**: Extracts semantic code symbols (classes, functions, methods, docstrings, calls, types) using Tree-sitter.
- **Code Knowledge Graph**: Tracks caller/callee relationships, imports, type hierarchies, and references in SQLite and NetworkX.
- **Hybrid Search Engine**: Reciprocal Rank Fusion (RRF) combining SQLite FTS5 (BM25 lexical search) and dense vector embeddings (`fastembed`).
- **Graph Neighborhood Expansion**: Expands search retrieval with 1–2 hop dependency context.
- **Living Repo Wiki**: Automatically generates and maintains architecture and module documentation with Mermaid diagrams.
- **Interactive CLI**: Fast terminal commands for indexing, querying, graph visualization, impact analysis, and wiki browsing.

## Installation

```bash
pip install codeatlas-cli
```
Or for local development:
```bash
uv pip install -e ".[dev]"
```

## Quickstart

```bash
# Index current repository
codeatlas index .

# Search code and architecture
codeatlas search "authentication middleware"

# Inspect symbol call graph
codeatlas graph "AuthMiddleware.authenticate"

# Analyze change blast radius
codeatlas impact "TokenService"

# Start MCP server
codeatlas mcp
```
