# CodeAtlas 🗺️

**Local codebase intelligence engine: multi-language code IR, deterministic knowledge graph, living repo wiki, multi-signal hybrid search, and Model Context Protocol (MCP) server for AI coding agents.**

[![PyPI version](https://img.shields.io/pypi/v/codeatlas-cli.svg)](https://pypi.org/project/codeatlas-cli/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Benchmark NDCG@10](https://img.shields.io/badge/NDCG%4010-0.8978%20(beats%20Semble)-brightgreen.svg)]()
[![Rust Port](https://img.shields.io/badge/Rust%20Port-codeatlas--rs-orange.svg)](https://github.com/adityaiitg/codeatlas-rs)

> ⚡ **Looking for maximum performance? Check out [CodeAtlas-rs (Rust)](https://github.com/adityaiitg/codeatlas-rs)** — our native Rust rewrite featuring **107ms cold indexing**, **6ms incremental updates**, **<1ms search latency**, and a single 6MB standalone binary.

---

## 🏆 Benchmark: CodeAtlas vs Semble

We replicated the standard coding agent retrieval benchmark on the **Flask** reference repository (24 files, 331 symbols, 21 retrieval tasks) evaluating NDCG@10, Top-1 accuracy, index latency, and context token consumption:

| Metric | Semble (Hybrid BM25 + dense) | CodeAtlas (Standard BAAI) | **CodeAtlas (`--fast` Model2Vec)** | **CodeAtlas-rs (Rust)** |
| :--- | :--- | :--- | :--- | :--- |
| **NDCG@10** | 0.8872 | 0.7657 | **0.8978** (+1.2% vs Semble) | **0.8978+** |
| **Top-1 Accuracy** | 90.5% | 76.2% | **90.5%** (tied best) | **90.5%** |
| **Cold Indexing Time** | 3,920 ms | 38,400 ms | **517 ms** (7.5x faster) | **107 ms** (36x faster) |
| **Incremental Re-index**| 450 ms | 1,200 ms | **82 ms** (5.5x faster) | **6 ms** (75x faster) |
| **Query Latency** | 38 ms | 45 ms | **12 ms** (3x faster) | **< 1 ms** (40x faster) |
| **Context Tokens / Top-10**| 9,196 tokens | 9,196 tokens | **2,778 tokens** (70% savings) | **2,778 tokens** |
| **Context Truncation** | Frequent (>512 tokens) | Frequent (>512 tokens) | **0% truncation** (capped 40 lines) | **0% truncation** |

*Run the benchmark yourself: `python benchmarks/compare_semble.py` or `make bench-compare`.*

---

## ✨ Features

- **Multi-Signal Hybrid Retrieval**:
  - SQLite FTS5 BM25 lexical tokenization with Porter stemming.
  - Reciprocal Rank Fusion (RRF) combining lexical score and dense embeddings.
  - Exact symbol matching boost (2.0x).
  - File-stem semantic boost (1.4x).
  - Noise penalties for test and compatibility paths (0.35x / 0.5x).
  - Multi-chunk file coherence boosting (0.2x).
- **Ultra-Fast Embedding Mode (`--fast`)**:
  - Model2Vec (`potion-code-16M-v2`) drops cold indexing time from 38s to **517ms** on CPU without PyTorch overhead.
- **Fine-Grained Class Chunking**:
  - Class overview chunks capped at 40 lines prevent 512-token truncation and save 70% of LLM prompt token budget.
- **Deterministic Knowledge Graph**:
  - SQLite + NetworkX graph tracking callers, callees, inheritance, and imports.
  - BFS 1-hop expansion and reverse BFS change blast radius analysis.
- **Living Repo Wiki**:
  - Automatically synthesizes architectural specifications with Mermaid flowchart and sequence diagrams.
- **Model Context Protocol (MCP) Server**:
  - Stdio JSON-RPC 2.0 server compatible with Claude Code, Cursor, OpenCode, Codex, and Windsurf.
- **One-Click Agent Auto-Installer**:
  - `codeatlas install all` automatically configures MCP across all detected coding agents.

---

## 📦 Installation

### Via Homebrew (macOS / Linux)

```bash
brew tap adityaiitg/tap
brew install codeatlas
```

### Via PyPI

```bash
pip install codeatlas-cli
```

### For Ultra-Fast CPU Embeddings (Recommended)

```bash
pip install "codeatlas-cli[fast]"  # installs model2vec
```

### From Source

```bash
git clone https://github.com/adityaiitg/codeatlas.git
cd codeatlas
uv pip install -e ".[dev]"
```

---

## 🚀 Quickstart

### 1. Indexing
```bash
# Ultra-fast CPU indexing with Model2Vec (<1 second)
codeatlas index --fast

# Standard dense embedding indexing (BAAI/bge-small-en-v1.5)
codeatlas index .

# Force full re-index
codeatlas index --force
```

### 2. Hybrid Code Search
```bash
# Multi-signal hybrid retrieval
codeatlas search "render_template"

# Adjust limit or mode
codeatlas search "authentication middleware" --limit 5 --mode hybrid
```

### 3. Change Blast Radius & Knowledge Graph
```bash
# Discover what breaks if a symbol is modified (reverse BFS)
codeatlas impact "Flask.render_template"

# Inspect callers and callees
codeatlas graph "AuthMiddleware.authenticate"

# Export graph for Gephi / Cytoscape
codeatlas export --format json -o graph.json
codeatlas export --format dot -o graph.dot
```

### 4. Living Architecture Wiki
```bash
# Generate architecture documentation and Mermaid diagrams in .codeatlas/wiki/
codeatlas wiki generate

# View wiki in browser or terminal
codeatlas wiki view --topic architecture
```

### 5. Auto-Configure AI Coding Agents (MCP)
```bash
# Auto-configure Claude Code, Cursor, OpenCode, and Codex
codeatlas install all

# Install for a specific agent
codeatlas install claude
codeatlas install cursor
codeatlas install opencode
codeatlas install codex

# Start MCP stdio server manually
codeatlas mcp
```

---

## ⚡ Rust Version (`codeatlas-rs`)

For production environments, CI pipelines, or developers needing instant indexing on large codebases, see the native Rust port:
👉 **[codeatlas-rs](https://github.com/adityaiitg/codeatlas-rs)**

- **107ms** cold index / **6ms** incremental re-index
- **<1ms** search latency
- Single **6.1 MB** static binary with zero external dependencies
- Prebuilt binaries available for macOS (Apple Silicon & Intel) and Linux x86_64

---

## 🧪 Testing & Benchmarks

```bash
# Run unit & integration tests
pytest -v

# Run head-to-head benchmark comparison against Semble
make bench-compare
# or
python benchmarks/compare_semble.py
```

---

## 📄 License

MIT License © 2026 Aditya Pratap Singh
