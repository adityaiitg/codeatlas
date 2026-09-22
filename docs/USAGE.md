# CodeAtlas (Python) — Complete Usage Guide 🗺️

CodeAtlas is a local codebase intelligence engine providing multi-language code information retrieval (IR), a deterministic knowledge graph, a living repository documentation wiki, multi-signal hybrid search, and a Model Context Protocol (MCP) server for AI coding assistants.

---

## Table of Contents

1. [Installation & Setup](#1-installation--setup)
   - [PyPI (Standard & Fast CPU Embedding Modes)](#pypi-standard--fast-cpu-embedding-modes)
   - [Homebrew (macOS / Linux)](#homebrew-macos--linux)
   - [Development Setup with `uv`](#development-setup-with-uv)
   - [Verifying Installation](#verifying-installation)
2. [Core Architecture & Retrieval Innovation](#2-core-architecture--retrieval-innovation)
   - [Fast Model2Vec vs Dense Embeddings](#fast-model2vec-vs-dense-embeddings)
   - [40-Line Fine-Grained Class Chunking](#40-line-fine-grained-class-chunking)
   - [Multi-Signal Ranking & Noise Penalties](#multi-signal-ranking--noise-penalties)
3. [CLI Command Reference](#3-cli-command-reference)
   - [`codeatlas index`](#codeatlas-index)
   - [`codeatlas search`](#codeatlas-search)
   - [`codeatlas impact`](#codeatlas-impact)
   - [`codeatlas wiki`](#codeatlas-wiki)
   - [`codeatlas graph`](#codeatlas-graph)
   - [`codeatlas install`](#codeatlas-install)
   - [`codeatlas mcp`](#codeatlas-mcp)
4. [AI Agent & Editor Skill Integration](#4-ai-agent--editor-skill-integration)
   - [Why Use CLI Skills for AI Coding Agents?](#why-use-cli-skills-for-ai-coding-agents)
   - [One-Command Skill Installation](#one-command-skill-installation)
   - [Antigravity (AGY)](#antigravity-agy)
   - [Cursor](#cursor)
   - [GitHub Copilot](#github-copilot)
   - [Universal Open Agent Standard](#universal-open-agent-standard)
5. [Model Context Protocol (MCP) Server](#5-model-context-protocol-mcp-server)
   - [Starting the MCP Server](#starting-the-mcp-server)
   - [Claude Desktop Configuration](#claude-desktop-configuration)
   - [Cursor MCP Configuration](#cursor-mcp-configuration)
   - [Available MCP Tools](#available-mcp-tools)
6. [Python Library / SDK API](#6-python-library--sdk-api)
   - [Programmatic Indexing](#programmatic-indexing)
   - [Hybrid Retrieval](#hybrid-retrieval)
   - [Knowledge Graph Queries](#knowledge-graph-queries)
   - [Wiki Generation](#wiki-generation)
7. [Running Benchmarks](#7-running-benchmarks)
   - [Reproducing the Semble Benchmark](#reproducing-the-semble-benchmark)
8. [Troubleshooting & FAQ](#8-troubleshooting--faq)

---

## 1. Installation & Setup

CodeAtlas requires **Python 3.10+**.

### PyPI (Standard & Fast CPU Embedding Modes)

```bash
# Recommended: Ultra-fast CPU embeddings via Model2Vec (517ms index, no PyTorch overhead)
pip install "codeatlas-cli[fast]"

# Standard installation (includes BAAI/bge-small-en-v1.5 sentence-transformers)
pip install codeatlas-cli
```

### Homebrew (macOS / Linux)

```bash
brew tap adityaiitg/tap
brew install codeatlas
```

### Development Setup with `uv`

```bash
git clone https://github.com/adityaiitg/codeatlas.git
cd codeatlas

# Install development dependencies with uv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,fast]"
```

### Verifying Installation

```bash
codeatlas --version
codeatlas --help
```

---

## 2. Core Architecture & Retrieval Innovation

### Fast Model2Vec vs Dense Embeddings

| Feature | Standard Dense (`bge-small-en-v1.5`) | Fast Mode (`--fast` Model2Vec) |
| :--- | :--- | :--- |
| **Model** | BAAI/bge-small-en-v1.5 (384-dim) | potion-code-16M-v2 (256-dim) |
| **Cold Indexing** | ~38,000 ms (Flask benchmark) | **517 ms** (7.5x faster) |
| **Memory / Deps** | Requires PyTorch (~2.5 GB) | Zero PyTorch, lightweight NumPy |
| **NDCG@10 Score** | 0.7657 | **0.8978** (Beats Semble) |
| **Activation** | `codeatlas index .` | `codeatlas index --fast .` |

### 40-Line Fine-Grained Class Chunking

Standard code search tools dump entire multi-hundred-line classes into LLM prompts, leading to:
- Frequent context window truncation (>512 tokens).
- Exhausting the context window budget.
- Lower retrieval accuracy because individual methods are buried in boilerplate.

CodeAtlas caps class overview chunks at **40 lines** and indexes methods as individual granular chunks with parent-child linkage.
- **70% LLM token savings** (2,778 tokens vs 9,196 tokens on Top-10 queries).
- **0% context truncation**.

### Multi-Signal Ranking & Noise Penalties

Search scores are composed via Reciprocal Rank Fusion (RRF) and dynamic signal weighting:
- **Lexical Score**: SQLite FTS5 BM25 with Porter stemming and CamelCase identifier splitting.
- **Dense Vector Similarity**: Cosine similarity against code embedding chunks.
- **Exact Symbol Boost (2.0x)**: Exact matching identifier definitions are prioritized above references.
- **File Stem Boost (1.4x)**: Files whose name matches the query receive a relevance boost.
- **Noise Penalties**: Test fixtures and compatibility shims receive damping penalties (`0.35x` for tests, `0.5x` for `compat.py`).
- **File Coherence Boost (0.2x)**: Files containing multiple relevant chunks receive a contextual clustering bonus.

---

## 3. CLI Command Reference

### `codeatlas index`

Parses source code into AST chunks, builds graph relationships, generates embeddings, and populates SQLite (`.codeatlas/index.db`).

```bash
# Ultra-fast CPU indexing with Model2Vec (<1 second)
codeatlas index --fast

# Standard indexing with BAAI embeddings
codeatlas index .

# Force a full re-index (ignores incremental hash manifest)
codeatlas index --full

# Index an external codebase
codeatlas index /path/to/project
```

**Options:**
- `[PATH]`: Path to index (default: `.`)
- `--fast`: Use lightweight Model2Vec CPU embeddings
- `--full`: Force complete re-indexing of all files
- `--db-path`: Custom SQLite database path

---

### `codeatlas search`

Executes multi-signal hybrid retrieval across the indexed codebase.

```bash
# Search for symbols, functions, or concepts
codeatlas search "render_template"

# Attach 1-hop graph neighborhood (callers, callees, classes, and imports)
codeatlas search "Retriever" --expand-graph

# Return fewer results
codeatlas search "Database" --limit 5

# Format results as machine-readable JSON
codeatlas search "authenticate" --json
```

**Options:**
- `<QUERY>`: Search string or symbol name
- `--limit`: Number of results (default: `10`)
- `--expand-graph`: Attach 1-hop knowledge graph neighborhood
- `--json`: Format output as JSON

**Example Output:**
```text
CodeAtlas Search for "Retriever" (10 results):

1. src/codeatlas/retrieval/retriever.py 44:84 (score: 0.0600) [definition]
   Symbol: src/codeatlas/retrieval/retriever.py:Retriever
   │ class Retriever:
   │     """Performs hybrid lexical-semantic search with graph neighborhood expansion."""
   │ 
   │     def __init__(self, settings: Settings):
   └ 1-hop: _attach_neighborhoods (method), __enter__ (method), _search_lexical (method)
```

---

### `codeatlas impact`

Performs reverse BFS traversal over the code knowledge graph to calculate the change blast radius before refactoring.

```bash
# Find all files and callers dependent on a class or function
codeatlas impact "Retriever"
```

**Example Output:**
```text
Reverse Impact Analysis for "Retriever" (6 callers/dependents):

  ← file:.../benchmarks/compare_semble.py
  ← file:.../benchmarks/benchmark_suite.py
  ← file:.../src/codeatlas/cli.py
  ← file:.../src/codeatlas/mcp/server.py
  ← file:.../tests/integration/test_end_to_end.py
  ← file:.../tests/unit/test_retriever.py
```

---

### `codeatlas wiki`

Generates living architecture documentation with Mermaid flowcharts and sequence diagrams.

```bash
# Generate architecture wiki in default directory (.codeatlas/wiki/)
codeatlas wiki .codeatlas/wiki

# Generate in a custom docs directory
codeatlas wiki docs/architecture
```

**Generated Documentation:**
- `index.md`: Complete codebase overview, symbol tallies, and chapter navigation.
- `architecture.md`: Visual Mermaid flowchart diagram (`graph TD`) of module dependencies.
- `workflows.md`: Mermaid sequence diagrams (`sequenceDiagram`) tracing execution flow.
- `modules/*.md`: Exhaustive module documentation with exported symbols, callers, and chunks.

---

### `codeatlas graph`

Inspects and exports the knowledge graph:

```bash
# Print knowledge graph statistics (files, symbols, edges)
codeatlas graph stats

# Export graph as JSON
codeatlas graph export --format json -o graph.json

# Export graph as Graphviz DOT
codeatlas graph export --format dot -o graph.dot
```

---

### `codeatlas install`

Configures the CodeAtlas MCP server across AI coding assistants with one command:

```bash
# Automatically configure all detected assistants
codeatlas install all

# Configure specific assistant
codeatlas install claude    # Updates ~/.claude.json
codeatlas install cursor    # Updates .cursor/mcp.json
codeatlas install opencode  # Updates ~/.opencode/config.json
codeatlas install codex     # Updates ~/.codex/mcp.json
```

---

### `codeatlas mcp`

Runs the Model Context Protocol stdio server:

```bash
codeatlas mcp
```

---

### `codeatlas watch`

Continuously polls the codebase for file changes and triggers ultra-fast incremental re-indexing in the background:

```bash
# Watch current repository with default 3-second polling interval
codeatlas watch .

# Custom polling interval
codeatlas watch . --interval 5
```

---

### `codeatlas hook`

Installs or removes Git hooks (`post-commit`, `post-checkout`, `post-merge`) so that every commit or branch switch automatically updates the CodeAtlas index in the background:

```bash
# Install git hooks
codeatlas hook install

# Remove git hooks
codeatlas hook uninstall
```

---

### `codeatlas serve`

Spins up a local web server with an interactive, force-directed Knowledge Graph visualization (D3.js) and real-time symbol search in your browser:

```bash
# Serve knowledge graph on default port 8765 and open in browser
codeatlas serve

# Custom port without opening browser automatically
codeatlas serve --port 9000 --no-open
```

---

## 4. AI Agent & Editor Skill Integration

### Why Use CLI Skills for AI Coding Agents?

AI agents (Cursor, Copilot, Antigravity) frequently guess callers or dump whole files when planning refactors. The CodeAtlas CLI Skill equips them to:
1. **Find exact definitions instantly** without scrolling through hundreds of lines.
2. **Evaluate change blast radius** with `codeatlas impact` before touching code.
3. **Consume 70% fewer prompt tokens** using 40-line bounded chunks.

### One-Command Skill Installation

```bash
# Run in the repository root:
./scripts/install-skill.sh

# Or target another repository:
./scripts/install-skill.sh /path/to/my-project
```

### Antigravity (AGY)

CodeAtlas installs globally into `~/.gemini/config/skills/codeatlas/SKILL.md`. AGY triggers `codeatlas` whenever code search, symbol lookup, or blast-radius analysis is requested.

### Cursor

The installer creates `.cursor/rules/codeatlas.mdc` in the project root. Cursor Composer and Chat consult CodeAtlas before modifying functions or analyzing project structure.

### GitHub Copilot

The installer writes `.github/copilot-instructions.md`. GitHub Copilot Workspace and CLI utilize CodeAtlas commands for symbol navigation and dependency tracing.

### Universal Open Agent Standard

The skill is standardized at `.agents/skills/codeatlas/SKILL.md` for portable integration across any AI agent environment.

---

## 5. Model Context Protocol (MCP) Server

### Starting the MCP Server

```bash
codeatlas-mcp
# or
codeatlas mcp
```

### Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "codeatlas": {
      "command": "codeatlas",
      "args": ["mcp"]
    }
  }
}
```

### Cursor MCP Configuration

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "codeatlas": {
      "command": "codeatlas",
      "args": ["mcp"]
    }
  }
}
```

### Available MCP Tools

- `codeatlas_search`: Hybrid lexical-semantic search with 1-hop graph neighborhood.
- `codeatlas_symbol_graph`: Inspect incoming callers and outgoing dependencies.
- `codeatlas_impact_analysis`: Reverse BFS change blast radius.
- `codeatlas_status`: Report index size, symbols, and edge counts.

---

## 6. Python Library / SDK API

You can use CodeAtlas programmatically in your own Python pipelines and applications:

### Programmatic Indexing

```python
from pathlib import Path
from codeatlas.config import Settings
from codeatlas.indexer.indexer import CodebaseIndexer

settings = Settings(
    db_path=Path(".codeatlas/index.db"),
    use_fast_embeddings=True,  # Enables Model2Vec
)

indexer = CodebaseIndexer(settings)
stats = indexer.index(root_path=Path("."), full_reindex=False)
print(f"Indexed {stats['files_indexed']} files, {stats['symbols']} symbols in {stats['duration_ms']}ms")
```

### Hybrid Retrieval

```python
from pathlib import Path
from codeatlas.config import Settings
from codeatlas.retrieval.retriever import Retriever

settings = Settings(db_path=Path(".codeatlas/index.db"))

with Retriever(settings) as retriever:
    results = retriever.search(
        query="render_template",
        limit=5,
        expand_graph=True,
    )
    for res in results:
        print(f"File: {res.file_path}:{res.start_line} (score: {res.score:.4f})")
        print(f"Neighbors: {res.graph_neighborhood}")
```

### Knowledge Graph Queries

```python
from pathlib import Path
from codeatlas.graph.queries import GraphQueries

queries = GraphQueries(Path(".codeatlas/index.db"))

# Find all callers of a function (reverse impact analysis)
callers = queries.get_callers_transitive("render_template", max_depth=3)
print(f"Affected callers: {callers}")

# Get outgoing dependencies
callees = queries.get_callees("render_template")
print(f"Calls: {callees}")
```

### Wiki Generation

```python
from pathlib import Path
from codeatlas.wiki.generator import WikiGenerator

wiki_gen = WikiGenerator(
    db_path=Path(".codeatlas/index.db"),
    output_dir=Path(".codeatlas/wiki"),
)
chapters = wiki_gen.generate()
print(f"Generated {len(chapters)} documentation chapters")
```

---

## 7. Running Benchmarks

### Reproducing the Semble Benchmark

To evaluate CodeAtlas against Semble on the reference Flask codebase (evaluating NDCG@10, top-1 accuracy, indexing latency, and token consumption):

```bash
# Run comparison benchmark
python benchmarks/compare_semble.py

# Or via Makefile:
make bench-compare
```

**Flask Benchmark Summary:**
- **NDCG@10**: `0.8978` (CodeAtlas `--fast`) vs `0.8872` (Semble).
- **Cold Indexing**: `517ms` (CodeAtlas `--fast`) vs `3,920ms` (Semble) — **7.5x faster**.
- **Context Tokens**: `2,778 tokens` (CodeAtlas) vs `9,196 tokens` (Semble) — **70% token savings**.
- **Context Truncation**: `0%` (CodeAtlas) vs frequent truncation in other tools.

---

## 8. Troubleshooting & FAQ

### Q: Why do I see a warning about sqlite-vec?
CodeAtlas natively supports the `sqlite-vec` extension for vector ANN search. If `sqlite-vec` is not compiled on your system, CodeAtlas automatically falls back to exact cosine similarity over NumPy embeddings with no loss of retrieval accuracy.

### Q: How do I switch between Model2Vec and SentenceTransformers?
Use `--fast` for Model2Vec (`potion-code-16M-v2`), which requires no PyTorch and indexes in under a second on CPU. Omit `--fast` to use `BAAI/bge-small-en-v1.5` via `sentence-transformers`.

### Q: How do I clear the local index?
```bash
rm -rf .codeatlas/
codeatlas index --fast .
```

### Q: Looking for even higher performance?
Check out **[CodeAtlas-rs](https://github.com/adityaiitg/codeatlas-rs)** — our native Rust rewrite with **107ms cold indexing**, **6ms incremental updates**, and a single 6MB standalone binary with zero dependencies.
