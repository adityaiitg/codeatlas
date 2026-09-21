# CodeAtlas

Local codebase intelligence engine: multi-language code IR, deterministic knowledge graph, living repo wiki, native vector search, Model Context Protocol (MCP) server, and hybrid search CLI.

## Features

- **Multi-Language Code IR**: Extracts semantic code symbols (classes, functions, methods, docstrings, calls, imports) across Python, TypeScript, JavaScript, Go, Rust, Java, and C/C++.
- **Code Knowledge Graph**: Tracks caller/callee relationships, imports, type hierarchies, and references in SQLite and NetworkX.
- **Native Hybrid Search Engine**: Reciprocal Rank Fusion (RRF) combining SQLite FTS5 (BM25 lexical search) and hardware-accelerated dense vector embeddings with `sqlite-vec`.
- **Graph Neighborhood Expansion**: Expands search retrieval with 1–2 hop dependency context in a single batched query.
- **Living Repo Wiki**: Automatically generates and maintains architecture and module documentation with Mermaid sequence/flow diagrams, with optional AI architectural synthesis.
- **Model Context Protocol (MCP) Server**: Integrates directly with AI coding assistants (Claude Desktop, Cursor, Zed, Antigravity) via stdio JSON-RPC.
- **Graph Export**: Dumps knowledge graphs to JSON, GraphML (for Gephi / Cytoscape), and DOT formats.
- **Interactive CLI**: Fast terminal commands for indexing, querying, graph visualization, impact analysis, configuration, and wiki browsing.

## Installation

### Via Homebrew (macOS / Linux)

```bash
brew install adityaiitg/tap/codeatlas
```

Or tap first:
```bash
brew tap adityaiitg/tap
brew install codeatlas
```

### Via PyPI

```bash
pip install codeatlas-cli
```

### For Local Development

```bash
uv pip install -e ".[dev]"
```

## Quickstart

```bash
# Index repository (extracts symbols, chunks, edges, and dense vectors)
codeatlas index .

# Hybrid search across code and architecture
codeatlas search "authentication middleware"

# Inspect symbol call graph
codeatlas graph "AuthMiddleware.authenticate"

# Analyze change blast radius
codeatlas impact "TokenService"

# Generate living wiki documentation (optionally with --llm for AI synthesis)
codeatlas wiki generate
codeatlas wiki view --topic architecture

# Export knowledge graph for Gephi / Cytoscape visualization
codeatlas export --format json -o graph.json
codeatlas export --format graphml -o graph.graphml

# Inspect active configuration and paths
codeatlas config

# Start MCP server for AI coding agents
codeatlas mcp
```


