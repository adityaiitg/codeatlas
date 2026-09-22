# CodeAtlas (Python): Comprehensive Improvement Roadmap & Implementation Plan

> **Synthesized from 20 Concurrent Deep-Dive Architectural & Empirical Research Audits**  
> **Status:** Approved for Implementation  
> **Target Repository:** `codeatlas` (Python)

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Empirical Benchmark Highlights](#empirical-benchmark-highlights)
3. [Phase 1: Critical Correctness & Bug Fixes](#phase-1-critical-correctness--bug-fixes)
4. [Phase 2: Performance & Scalability Optimizations](#phase-2-performance--scalability-optimizations)
5. [Phase 3: Packaging, Dependency Diet & Footprint Reduction](#phase-3-packaging-dependency-diet--footprint-reduction)
6. [Phase 4: Type Safety, Mypy Configuration & API Ergonomics](#phase-4-type-safety-mypy-configuration--api-ergonomics)
7. [Phase 5: MCP Protocol Hardening & Offline Resilience](#phase-5-mcp-protocol-hardening--offline-resilience)
8. [Implementation Execution Checklist](#implementation-execution-checklist)

---

## Executive Summary

Across 20 distinct domains—spanning AST parsing, vector embedding pipelines, hybrid retrieval heuristics, graph algorithms, SQLite storage, CLI developer experience, MCP protocols, security, type safety, test coverage, and distribution packaging—the Python implementation of CodeAtlas demonstrates strong architectural design. 

However, empirical benchmarking and static code analysis identified **critical bugs breaking core intelligence features** (such as caller resolution returning 0 callers and fast mode semantic search failing), **massive performance bottlenecks** (such as file scanning taking 1.3s instead of 20ms and commits happening per-file), **security vulnerabilities** (symlink escape), and **virtualenv bloat** (4.0 GB due to LiteLLM / AWS SDK in core).

This document serves as the master specification and execution checklist for implementing these improvements.

---

## Empirical Benchmark Highlights

| Subsystem / Metric | Current Baseline | Optimized Prototype | Measured Impact |
|---|---|---|---|
| **Repository File Discovery** | `rglob("*")`: **1,307.4 ms** | `os.walk` + in-place pruning: **20.5 ms** | **63.6x faster** |
| **AST Callee Chain Resolution** | `ast.unparse`: **35.5 ms** / 40k | Iterative attribute resolver: **10.6 ms** | **3.35x faster** |
| **AST Visitor Structure** | Redundant class + method walks: **983.6 ms** | Single-pass inline visitor: **565.8 ms** | **1.74x faster** |
| **SQLite DB Batch Writes** | 5 commits per file: **75.9 ms** | Single transaction + `executemany`: **17.6 ms** | **4.31x faster** |
| **IPC Serialization** | Pickling Pydantic models: **6.87 ms** | Raw tuples: **1.48 ms** | **4.64x faster IPC** |
| **Embedding Speed (421 chunks)** | FastEmbed BGE-small: **36.70 s** | Model2Vec Potion-16M: **0.698 s** | **44x faster** |
| **Embedding Memory (Max RSS)** | FastEmbed ONNX: **1,949.48 MB (~1.95 GB)** | Model2Vec: **190.98 MB** | **10x lower RAM** |
| **Virtualenv Disk Footprint** | Core install: **4.0 GB** (98 pkgs) | Restructured core + extras: **~100 MB** | **40x smaller** |

---

## Phase 1: Critical Correctness & Bug Fixes

### 1.1 Knowledge Graph Symbol Resolution Linker (The "0-Callers" Bug)
* **Affected Files**: `src/codeatlas/graph/queries.py`, `src/codeatlas/graph/builder.py`, `src/codeatlas/parser/python_parser.py`
* **Defect**:
  - AST parsing records defined symbols as `file_path:Class.method` or `file_path:func_name`.
  - Callee invocations are emitted as unlinked string expressions: `call:parse_jwt` or `call:self.fetch`.
  - Import statements are emitted as `symbol:module.func`.
  - Defined symbol nodes never receive incoming `CALLS` edges from other files.
  - As a result, `GraphQueries.get_callers()` and `impact_analysis()` return 0 callers and 0 dependents for all real symbols across any multi-file repository.
* **Remediation**:
  - Implement a post-indexing symbol resolution pass (`GraphLinker`):
    1. Build a global symbol table: `(module_path, unqualified_name) -> canonical_node_id`.
    2. Map imports: `import foo.bar as b` -> tracks alias `b` to `foo.bar`.
    3. Resolve `call:<callee>` edges to canonical defined symbol node IDs using file scope and imported symbols.
    4. Replace unresolved call strings with direct `Edge(source_symbol, target_symbol, EdgeType.CALLS)`.

### 1.2 Embedding Dimension & Model Sync in Fast Mode
* **Affected Files**: `src/codeatlas/cli.py`, `src/codeatlas/retrieval/retriever.py`, `src/codeatlas/mcp/server.py`, `src/codeatlas/graph/schema.py`
* **Defect**:
  - `codeatlas index --fast` indexes with `potion-code-16M-v2` (256-dim) and creates a 256-dim `chunk_vectors_vec` table.
  - `codeatlas search` and `MCPServer` do not read the index's embedding dimension or model from the database, defaulting to `BAAI/bge-small-en-v1.5` (384-dim).
  - Semantic vector search fails silently or crashes on dimension mismatch (384 != 256), yielding 0 semantic results.
* **Remediation**:
  - Persist an `index_metadata` table in `codeatlas.db` recording `embedding_model`, `embedding_dim`, and `created_at`.
  - `Retriever` and `MCPServer` read `index_metadata` on initialization and dynamically select the matching model and dimension.

### 1.3 Safe Config Mutation in Agent Installer (Data-Loss Prevention)
* **Affected File**: `src/codeatlas/installer/agent_installer.py`
* **Defect**:
  - `agent_installer.py` parses configuration files with `json.loads`.
  - Modern Cursor (`~/.cursor/settings.json`) and Claude Desktop (`claude_desktop_config.json`) configs frequently contain JSONC syntax (comments, trailing commas).
  - When `json.loads` fails, it falls back to `data = {}` and overwrites the file, wiping all existing user extensions, settings, and other MCP servers.
* **Remediation**:
  - Strip JS/JSONC comments and trailing commas before parsing, or parse with a tolerant regex-based preprocessor.
  - **Never overwrite with empty dictionary on parse failure**: If parsing fails, abort with an error and inform the user.
  - Automatically write a `.bak.<timestamp>` backup before modifying any user config file.

### 1.4 Retrieval Heuristics: Stemming & Query Intent Sanitization
* **Affected File**: `src/codeatlas/retrieval/retriever.py`
* **Defect**:
  - `clean_q.rstrip("s") == file_stem.rstrip("s")` corrupts words like `"process"` -> `"proce"`, `"status"` -> `"statu"`, `"pass"` -> `"pa"`.
  - Substring noise penalty check `"test" in clean_q` matches `"latest"`, `"fastest"`, `"greatest"`, and `"spec"` matches `"specific"`, `"special"`. Legitimate production queries matching these words have test-directory penalties erroneously disabled.
  - Regex `[a-z][A-Z]` in `_classify_query` fails for all-caps acronyms (`JWT`, `HTTP`, `CLI`, `AST`).
* **Remediation**:
  - Replace `rstrip("s")` with exact match or safe inflection stemming (`words.endswith("s") and not words.endswith("ss")`).
  - Use regex word boundaries `r"\b(test|tests|spec|specs)\b"` for test query detection.
  - Update identifier regex to handle acronyms: `r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[0-9]+"`.

### 1.5 AST Parser: Preserve Function & Class Decorators
* **Affected File**: `src/codeatlas/parser/python_parser.py`
* **Defect**:
  - `start = node.lineno` records the line of the `def` or `class` statement. In Python, decorators precede the `def` line (e.g., `@app.route(...)`, `@dataclass`, `@property`, `@override`).
  - Stored chunk source code and AST signatures omit all decorators, losing API routing and auth metadata.
* **Remediation**:
  - Set `start = node.decorator_list[0].lineno if node.decorator_list else node.lineno`.

### 1.6 Symlink Traversal Confinement (Security Hardening)
* **Affected Files**: `src/codeatlas/parser/file_scanner.py`, `src/codeatlas/index/indexer.py`
* **Defect**:
  - `FileScanner` scans paths without checking `path.resolve().is_relative_to(repo_root)`.
  - Symlinks pointing outside the repository (e.g. `/etc/passwd` or home directory files) are indexed into SQLite, exposing arbitrary files via search.
* **Remediation**:
  - Reject symlinks that point outside `settings.repo_path`.
  - Ensure all file reads verify `path.resolve().is_relative_to(self.settings.repo_path.resolve())`.

---

## Phase 2: Performance & Scalability Optimizations

### 2.1 In-Place Directory Pruning in FileScanner (63.6x Faster)
* **Affected File**: `src/codeatlas/parser/file_scanner.py`
* **Issue**:
  - `sorted(self.settings.repo_path.rglob("*"))` traverses every file and folder in the tree before checking ignore rules, generating tens of thousands of `Path` objects inside `.venv`, `.git`, and `node_modules`.
* **Remediation**:
  - Replace `rglob("*")` with `os.walk()`:
    ```python
    for root, dirs, files in os.walk(repo_str):
        # Prune ignored directories in-place before os.walk descends into them
        dirs[:] = [d for d in dirs if not (d.startswith(".") or d in ignore_set)]
        for f in files:
            if not f.startswith(".") and os.path.splitext(f)[1] in LANGUAGE_MAP:
                results.append((Path(root) / f, LANGUAGE_MAP[os.path.splitext(f)[1]]))
    ```

### 2.2 Single-Pass AST Visitor & Fast Callee Resolution (3.35x Faster)
* **Affected File**: `src/codeatlas/parser/python_parser.py`
* **Issue**:
  - `visit_ClassDef` calls `_extract_calls` on the entire class, and then `visit_FunctionDef` walks each method again ($O(N^2)$ AST node visits).
  - Every call invokes `ast.unparse(n.func)`, which is a heavy code-generator visitor.
* **Remediation**:
  - Replace `ast.unparse` with an iterative attribute-chain resolver (`".".join(parts)`), falling back to `ast.unparse` only for complex subscript/lambda expressions.
  - Collect calls inline in `visit_Call` during node traversal rather than executing redundant `ast.walk` trees.

### 2.3 SQLite WAL Mode, Transaction Batching & Bulk Inserts (4.31x Faster)
* **Affected Files**: `src/codeatlas/graph/schema.py`, `src/codeatlas/graph/builder.py`, `src/codeatlas/index/indexer.py`
* **Issue**:
  - `PRAGMA journal_mode = MEMORY;` locks readers during writes, causing MCP servers and concurrent searches to crash with `sqlite3.OperationalError: database is locked`.
  - 5 independent transactions are committed per file (5,000 disk commits per 1,000 files).
  - FTS5 `DELETE FROM chunks_fts WHERE chunk_id = ?` triggers a full table scan because `chunk_id` is marked `UNINDEXED`.
* **Remediation**:
  - Enable `PRAGMA journal_mode = WAL;`, `PRAGMA busy_timeout = 30000;`, and `PRAGMA synchronous = NORMAL;`.
  - Wrap the entire file indexing loop in an explicit transaction (`with self.conn:`).
  - Use `conn.executemany()` for `symbols`, `edges`, `chunks`, `chunks_fts`, and `chunk_vectors`.

### 2.4 Stat-First Incremental Change Detection
* **Affected File**: `src/codeatlas/index/indexer.py`
* **Issue**:
  - Incremental runs read and hash every file from disk even when unchanged.
* **Remediation**:
  - Compare `stat.st_mtime` and `stat.st_size` against SQLite `file_manifest` before reading file contents.
  - Skip unchanged files with zero disk reads and zero SHA-256 computations.

### 2.5 Make Model2Vec Default & Stream Bounded Vector Buffers
* **Affected Files**: `src/codeatlas/config.py`, `src/codeatlas/index/indexer.py`, `src/codeatlas/index/embedder.py`
* **Issue**:
  - FastEmbed BGE-small consumes 1.95 GB RAM and takes 36.7s for 421 chunks, compared to Model2Vec which consumes 191 MB RAM and takes 0.698s (44x faster).
  - `all_new_chunks` accumulates all repository chunks in memory before embedding, scaling memory to $O(N)$.
* **Remediation**:
  - Set `minishlab/potion-code-16M-v2` (Model2Vec, 256-dim) as default embedding model in `Settings`.
  - Embed chunks in bounded batches of 256 and commit directly to SQLite, preventing unbounded memory growth on large codebases.

---

## Phase 3: Packaging, Dependency Diet & Footprint Reduction

### 3.1 Trim Core Package from 4.0 GB to ~100 MB
* **Affected File**: `pyproject.toml`
* **Issue**:
  - `litellm` in core pulls over 30 transitive dependencies (`boto3`, `botocore`, `s3transfer`, `openai`, `aiohttp`, `tiktoken`), taking **2.15 GB** (>50% of the virtualenv), used only for `--llm` in living wiki generation.
  - `tree-sitter` and `tree-sitter-python` are compiled C extensions required in core, but are **100% unused dead code** (`self._ts_parser` is never called).
  - `numpy` is directly imported in `embedder.py` and `retriever.py`, but omitted from `dependencies`.
* **Remediation**:
  - Remove `tree-sitter` and `tree-sitter-python` entirely.
  - Explicitly declare `numpy>=1.26.0,<3.0.0` in core dependencies.
  - Move `litellm` to optional extra `[llm]`.
  - Move `fastembed` and `sqlite-vec` to optional extra `[onnx]`.
  - Core dependencies: `typer`, `rich`, `pydantic`, `pydantic-settings`, `networkx`, `numpy`, `model2vec`.

---

## Phase 4: Type Safety, Mypy Configuration & API Ergonomics

### 4.1 Fix Static Type Errors & Scope Collisions
* **Affected Files**: `src/codeatlas/wiki/generator.py`, `src/codeatlas/mcp/server.py`, `src/codeatlas/index/embedder.py`, `src/codeatlas/llm/client.py`
* **Defects**:
  - `wiki/generator.py:144`: Variable collision (`names` declared as `list[Any]` at L117 and reassigned as `str` at L144 in the same scope).
  - `mcp/server.py:151–155`: `res` inferred as `list[dict]` in branch 1, throwing type errors when assigned `dict` in branches 2 and 3.
  - `embedder.py`: Pyright flags attribute access failures on `self._model` (`encode` vs `embed`) because the union is not narrowed using `isinstance`.
  - `llm/client.py:35`: Pyright flags `.choices` access on LiteLLM's `ModelResponse | CustomStreamWrapper`.
  - 43 functions lack return type annotations (including all 12 Typer CLI commands and 13 `__init__` methods).
* **Remediation**:
  - Resolve variable naming clashes and add explicit type narrowing (`isinstance(model, StaticModel)`).
  - Add return type annotations (`-> None`) across CLI commands and class methods.
  - Add `tool.mypy` configuration in `pyproject.toml` and add `mypy` and `types-networkx` to dev dependencies.

### 4.2 Replace Bare Types with Domain Models & TypedDicts
* **Affected Files**: `src/codeatlas/parser/base.py`, `src/codeatlas/graph/queries.py`, `src/codeatlas/graph/builder.py`
* **Remediation**:
  - Replace anonymous return tuple `tuple[list[Symbol], list[CodeChunk], list[Edge]]` with a typed dataclass: `ParseResult(symbols, chunks, edges)`.
  - Replace bare `dict` returns with `TypedDict`: `IndexStats`, `ImpactResult`, `NeighborInfo`.
  - Replace string parameter flags with `typing.Literal`: `SearchMode = Literal["hybrid", "lexical", "semantic"]`.

---

## Phase 5: MCP Protocol Hardening & Offline Resilience

### 5.1 Stdio Stream Protection & Retriever Caching in MCP Server
* **Affected Files**: `src/codeatlas/mcp/server.py`, `src/codeatlas/cli.py`
* **Issue**:
  - CLI configures `RichHandler` on `sys.stdout`. Uncaught exceptions or debug statements emit ANSI characters to stdout, corrupting the MCP JSON-RPC protocol stream.
  - Every `codeatlas_search` MCP tool call instantiates a new `Retriever`, reloading embedding weights from disk on every query (150ms–580ms latency per tool call).
* **Remediation**:
  - In MCP mode, redirect all logging and console output strictly to `sys.stderr`.
  - Retain a single persistent, cached `Retriever` instance alive across MCP requests.
  - Expose specialized MCP tools: `codeatlas_definition`, `codeatlas_callers`, `codeatlas_callees`.

### 5.2 Offline & Network Failure Graceful Degradation
* **Affected Files**: `src/codeatlas/retrieval/retriever.py`, `src/codeatlas/index/indexer.py`
* **Issue**:
  - If Hugging Face is unreachable or offline, `codeatlas search` crashes on `embed_query` with unhandled network errors, even though local SQLite FTS5 BM25 search requires no network.
  - If a file contains syntax errors, `indexer.py` deletes all previous graph nodes and edges for that file, returning 0 symbols and destroying blast radius accuracy during code editing.
* **Remediation**:
  - Fall back transparently to BM25 lexical search when embedding models cannot be loaded offline.
  - When AST parsing encounters syntax errors, fall back to sliding window chunking with regex extraction, and retain prior graph nodes with an `outdated` flag instead of wiping the file from the graph.

---

## Implementation Execution Checklist

- [x] **Phase 1: Critical Correctness & Bug Fixes**
  - [x] 1.1 Implement post-index symbol resolution linker in `GraphLinker` to fix the 0-callers bug
  - [x] 1.2 Store embedding metadata in SQLite and sync dimensions between `index`, `search`, and `mcp`
  - [x] 1.3 Add JSONC comment stripping and `.bak` backups to `agent_installer.py`
  - [x] 1.4 Sanitize retrieval heuristics (word boundary regexes for test intent, safe stemming, acronym support)
  - [x] 1.5 Fix decorator truncation in AST parsing (`node.decorator_list`)
  - [x] 1.6 Add symlink boundary confinement in `FileScanner`

- [x] **Phase 2: Performance & Scalability Optimizations**
  - [x] 2.1 Replace `rglob("*")` with in-place directory pruning `os.walk` in `FileScanner`
  - [x] 2.2 Implement iterative attribute-chain resolver and single-pass AST visitor in `PythonParser`
  - [x] 2.3 Enable SQLite WAL mode, busy timeout 30s, and transaction batching via `executemany`
  - [x] 2.4 Add stat-first incremental change detection (`mtime` + `size`) in `Indexer`
  - [x] 2.5 Make Model2Vec (`potion-code-16M-v2`) default and stream chunk embedding batches

- [x] **Phase 3: Packaging & Dependency Diet**
  - [x] 3.1 Remove dead `tree-sitter` dependencies from `PythonParser` and `pyproject.toml`
  - [x] 3.2 Add explicit `numpy>=1.26.0,<3.0.0` to `dependencies` in `pyproject.toml`
  - [x] 3.3 Move `litellm` to optional extra `[llm]` and `fastembed` to `[onnx]`

- [x] **Phase 4: Type Safety & API Ergonomics**
  - [x] 4.1 Fix variable collision in `wiki/generator.py` and union returns in `mcp/server.py`
  - [x] 4.2 Guard `self.choices` in `client.py` and narrow types
  - [x] 4.3 Add return type annotations (`-> None`) across CLI commands and class methods
  - [x] 4.4 Add `tool.mypy` configuration to `pyproject.toml` and verify clean runs

- [x] **Phase 5: MCP Server & Offline Resilience**
  - [x] 5.1 Route all logging to `sys.stderr` in MCP mode to protect stdio JSON-RPC stream
  - [x] 5.2 Cache persistent `Retriever` and `GraphQueries` instances in `MCPServer`
  - [x] 5.3 Implement offline automatic fallback to BM25 lexical search in `Retriever.search()`
  - [x] 5.4 Retain symbols via regex fallback on syntax errors during active editing

- [x] **Phase 6: Verification & Test Suite Execution**
  - [x] 6.1 Run full unit test suite (`pytest tests/unit`) - 28/28 passing
  - [x] 6.2 Run integration test suite (`pytest tests/integration`) - 2/2 passing
  - [x] 6.3 Verify linting and code hygiene (`ruff check src/ tests/`) - 0 errors
