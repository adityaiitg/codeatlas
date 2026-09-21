# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-21

### Added
- **Multi-Language AST Extraction**: Support for parsing TypeScript, JavaScript, Go, Rust, Java, C, and C++ into canonical IR symbols and chunks.
- **Model Context Protocol (MCP) Server**: Built-in stdio JSON-RPC MCP server (`codeatlas mcp`) providing code search, symbol graph inspection, and impact analysis to AI coding tools.
- **Knowledge Graph Export**: `codeatlas export` command exporting graph structures to JSON, GraphML (Gephi / Cytoscape), and DOT formats.
- **CLI Configuration Inspector**: `codeatlas config` command to view active settings and environment variables.
- **Hardware-Accelerated Vector ANN**: Integrated `sqlite-vec` `vec0` virtual table for native database-level semantic search with fallback.
- **AI-Powered Living Wiki**: Optional `--llm` flag in `codeatlas wiki generate` synthesizing high-level architecture overviews and module summaries.
- **Version Flag**: `--version` / `-V` CLI flag with graceful package metadata resolution.
- **Verbose & Quiet Logging**: RichHandler logging with `--verbose` and `--quiet` flags.

### Fixed
- **Impact Analysis Algorithm**: Fixed inverted dependency traversal bug; blast radius analysis now accurately traces callers and defined children with BFS.
- **Wiki View Security**: Fixed path traversal vulnerability in `wiki view` command with path resolution checks.
- **Broad Exception Swallowing**: Replaced bare and broad `except Exception` blocks with specific exception catches and structured logging.
- **N+1 Neighborhood Resolution**: Consolidated search neighborhood lookups into a single batched query.
- **Database Leaks**: Wrapped database access in connection context managers across GraphBuilder, Retriever, and Indexer.
