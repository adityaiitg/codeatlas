"""Benchmark comparison replicating MinishLab/semble evaluation methodology for CodeAtlas."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from codeatlas.config import Settings
from codeatlas.index.indexer import Indexer
from codeatlas.retrieval.retriever import Retriever

try:
    from semble import SembleIndex

    SEMBLE_AVAILABLE = True
except ImportError:
    SEMBLE_AVAILABLE = False

console = Console()


def dcg(relevances: list[int]) -> float:
    """Compute Discounted Cumulative Gain for a ranked relevance list."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(relevant_ranks: list[int], n_relevant: int, k: int) -> float:
    """Compute NDCG@k given 1-based ranks of relevant results and total relevant count."""
    if n_relevant == 0:
        return 0.0
    relevances = [0] * k
    for rank in relevant_ranks:
        if 1 <= rank <= k:
            relevances[rank - 1] = 1
    ideal = dcg([1] * min(k, n_relevant))
    return dcg(relevances) / ideal if ideal > 0 else 0.0


def path_matches(file_path: str, target_path: str) -> bool:
    """Return True if either path is a suffix of the other."""
    norm_file = file_path.replace("\\", "/")
    norm_target = target_path.replace("\\", "/")
    return (
        norm_file == norm_target
        or norm_file.endswith(f"/{norm_target}")
        or norm_target.endswith(f"/{norm_file}")
    )


@dataclass(frozen=True)
class Target:
    """Ground truth target location."""

    path: str
    start_line: int | None = None
    end_line: int | None = None

    @property
    def has_span(self) -> bool:
        return self.start_line is not None and self.end_line is not None


def target_matches(file_path: str, start_line: int, end_line: int, target: Target) -> bool:
    """Check whether a chunk covers the target."""
    if not path_matches(file_path, target.path):
        return False
    if not target.has_span:
        return True
    return not (end_line < target.start_line or start_line > target.end_line)  # type: ignore[operator]


@dataclass
class BenchmarkTask:
    """A benchmark query with ground truth targets and query category."""

    query: str
    category: str
    relevant: list[Target]
    secondary: list[Target]

    @property
    def all_targets(self) -> list[Target]:
        return self.relevant + self.secondary


def load_tasks(annotation_file: Path) -> list[BenchmarkTask]:
    """Load benchmark tasks from Semble-formatted annotation JSON."""
    raw = json.loads(annotation_file.read_text(encoding="utf-8"))
    tasks: list[BenchmarkTask] = []
    for item in raw:
        relevant: list[Target] = []
        for t in item.get("relevant", []):
            if isinstance(t, str):
                relevant.append(Target(path=t))
            elif isinstance(t, dict):
                relevant.append(
                    Target(
                        path=str(t["path"]),
                        start_line=t.get("start_line"),
                        end_line=t.get("end_line"),
                    )
                )

        secondary: list[Target] = []
        for t in item.get("secondary", []):
            if isinstance(t, str):
                secondary.append(Target(path=t))
            elif isinstance(t, dict):
                secondary.append(
                    Target(
                        path=str(t["path"]),
                        start_line=t.get("start_line"),
                        end_line=t.get("end_line"),
                    )
                )

        cat = item.get("category")
        if not cat:
            q = item["query"].lower()
            if " " not in q:
                cat = "symbol"
            elif q.startswith(("how ", "how does", "how are")):
                cat = "architecture"
            else:
                cat = "semantic"

        tasks.append(
            BenchmarkTask(
                query=item["query"],
                category=cat,
                relevant=relevant,
                secondary=secondary,
            )
        )
    return tasks


@dataclass
class EngineBenchmarkResult:
    """Benchmark evaluation summary for a specific search engine or mode."""

    engine_name: str
    index_time_ms: float
    chunks_or_symbols: int
    ndcg5: float
    ndcg10: float
    ndcg10_by_category: dict[str, float] = field(default_factory=dict)
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    mean_ms: float = 0.0
    avg_tokens_retrieved: float = 0.0
    top1_accuracy: float = 0.0


def run_engine_evaluation(
    engine_name: str,
    tasks: list[BenchmarkTask],
    search_fn: Any,
    index_time_ms: float,
    chunk_count: int,
    runs: int = 5,
) -> EngineBenchmarkResult:
    """Run latency and NDCG retrieval evaluation over all benchmark tasks."""
    ndcg5_scores: list[float] = []
    ndcg10_scores: list[float] = []
    category_ndcg10: dict[str, list[float]] = {}
    latencies: list[float] = []
    token_counts: list[int] = []
    top1_hits = 0

    for task in tasks:
        task_latencies = []
        results = []
        # Latency runs
        for _ in range(runs):
            t0 = time.perf_counter()
            results = search_fn(task.query, 10)
            task_latencies.append((time.perf_counter() - t0) * 1000)

        latencies.append(float(np.median(task_latencies)))

        # Token calculation (approx 4 characters per token)
        total_chars = sum(len(getattr(r, "content", "")) for r in results)
        token_counts.append(total_chars // 4)

        # NDCG calculation
        targets = task.all_targets
        matched_ranks = []
        for target in targets:
            for rank, r in enumerate(results, 1):
                fpath = getattr(r, "file_path", "")
                sline = getattr(r, "start_line", 0)
                eline = getattr(r, "end_line", 0)
                if target_matches(fpath, sline, eline, target):
                    matched_ranks.append(rank)
                    break

        n_rel = len(targets)
        q_ndcg5 = ndcg_at_k(matched_ranks, n_rel, 5)
        q_ndcg10 = ndcg_at_k(matched_ranks, n_rel, 10)

        ndcg5_scores.append(q_ndcg5)
        ndcg10_scores.append(q_ndcg10)
        category_ndcg10.setdefault(task.category, []).append(q_ndcg10)

        # Top-1 accuracy
        if matched_ranks and min(matched_ranks) == 1:
            top1_hits += 1

    latencies.sort()
    p50 = float(np.percentile(latencies, 50))
    p90 = float(np.percentile(latencies, 90))
    p95 = float(np.percentile(latencies, 95))

    cat_summary = {cat: float(np.mean(scores)) for cat, scores in category_ndcg10.items()}

    return EngineBenchmarkResult(
        engine_name=engine_name,
        index_time_ms=index_time_ms,
        chunks_or_symbols=chunk_count,
        ndcg5=float(np.mean(ndcg5_scores)),
        ndcg10=float(np.mean(ndcg10_scores)),
        ndcg10_by_category=cat_summary,
        p50_ms=p50,
        p90_ms=p90,
        p95_ms=p95,
        mean_ms=float(statistics.mean(latencies)),
        avg_tokens_retrieved=float(np.mean(token_counts)),
        top1_accuracy=top1_hits / len(tasks) if tasks else 0.0,
    )


def compare_codeatlas_and_semble(
    repo_root: Path,
    annotation_file: Path,
    runs: int = 5,
    export_json: Path | None = None,
) -> None:
    """Execute head-to-head benchmark comparison between CodeAtlas and Semble."""
    tasks = load_tasks(annotation_file)
    console.print(
        Panel.fit(
            f"[bold cyan]Head-to-Head Benchmark: CodeAtlas vs. MinishLab/semble[/bold cyan]\n"
            f"Repository: [underline]{repo_root}[/underline]\n"
            f"Tasks: [bold]{len(tasks)}[/bold] queries loaded from [italic]{annotation_file.name}[/italic]",
            title="Benchmark Setup",
            border_style="cyan",
        )
    )

    results: list[EngineBenchmarkResult] = []

    # -------------------------------------------------------------
    # 1. Semble Benchmark
    # -------------------------------------------------------------
    if SEMBLE_AVAILABLE:
        console.print(
            "\n[bold green]▶ Running Semble (potion-code-16M-v2 + BM25 + Reranker)...[/bold green]"
        )
        t0 = time.perf_counter()
        s_index = SembleIndex.from_path(str(repo_root))
        semble_idx_ms = (time.perf_counter() - t0) * 1000

        def semble_search(q: str, limit: int = 10):
            res = s_index.search(q, top_k=limit)

            # Adapt chunk format to unified interface
            class Item:
                def __init__(self, c):
                    self.file_path = c.file_path
                    self.start_line = c.start_line
                    self.end_line = c.end_line
                    self.content = c.content

            return [Item(r.chunk) for r in res]

        s_bench = run_engine_evaluation(
            engine_name="Semble (Hybrid + Reranker)",
            tasks=tasks,
            search_fn=semble_search,
            index_time_ms=semble_idx_ms,
            chunk_count=len(s_index.chunks),
            runs=runs,
        )
        results.append(s_bench)
    else:
        console.print("[yellow]Semble not installed; skipping Semble runs.[/yellow]")

    # -------------------------------------------------------------
    # 2. CodeAtlas Full Hybrid Benchmark
    # -------------------------------------------------------------
    console.print(
        "\n[bold magenta]▶ Running CodeAtlas Hybrid (BGE Transformer + sqlite-vec + FTS5 + 1-Hop Graph)...[/bold magenta]"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        settings = Settings(repo_path=repo_root, data_dir=Path(tmp_dir) / ".codeatlas")
        t0 = time.perf_counter()
        with Indexer(settings, embed_vectors=True) as indexer:
            stats = indexer.index_repository(force=True)
        ca_idx_ms = (time.perf_counter() - t0) * 1000

        with Retriever(settings) as retriever:
            ca_bench = run_engine_evaluation(
                engine_name="CodeAtlas (Hybrid + Graph)",
                tasks=tasks,
                search_fn=lambda q, limit: retriever.search(
                    q, limit=limit, mode="hybrid", expand_graph=True
                ),
                index_time_ms=ca_idx_ms,
                chunk_count=stats.get("chunks", 0),
                runs=runs,
            )
            results.append(ca_bench)

            # -------------------------------------------------------------
            # 3. CodeAtlas Lexical-Only (FTS5 BM25)
            # -------------------------------------------------------------
            console.print(
                "[bold blue]▶ Running CodeAtlas Lexical (FTS5 BM25 + Code Boosts)...[/bold blue]"
            )
            ca_lex_bench = run_engine_evaluation(
                engine_name="CodeAtlas (BM25 Lexical)",
                tasks=tasks,
                search_fn=lambda q, limit: retriever.search(
                    q, limit=limit, mode="lexical", expand_graph=False
                ),
                index_time_ms=ca_idx_ms,
                chunk_count=stats.get("chunks", 0),
                runs=runs,
            )
            results.append(ca_lex_bench)

            # -------------------------------------------------------------
            # 4. CodeAtlas Semantic-Only (sqlite-vec ANN)
            # -------------------------------------------------------------
            console.print(
                "[bold yellow]▶ Running CodeAtlas Semantic (fastembed BGE + sqlite-vec)...[/bold yellow]"
            )
            ca_sem_bench = run_engine_evaluation(
                engine_name="CodeAtlas (Dense Vector ANN)",
                tasks=tasks,
                search_fn=lambda q, limit: retriever.search(
                    q, limit=limit, mode="semantic", expand_graph=False
                ),
                index_time_ms=ca_idx_ms,
                chunk_count=stats.get("chunks", 0),
                runs=runs,
            )
            results.append(ca_sem_bench)

    # -------------------------------------------------------------
    # Render Tables
    # -------------------------------------------------------------
    # Table 1: Retrieval Quality
    tbl_quality = Table(title="1. Retrieval Quality (NDCG & Accuracy)", header_style="bold magenta")
    tbl_quality.add_column("Engine / Mode", style="cyan")
    tbl_quality.add_column("NDCG@5", justify="right")
    tbl_quality.add_column("NDCG@10", justify="right", style="bold green")
    tbl_quality.add_column("Top-1 Accuracy", justify="right")
    tbl_quality.add_column("Avg Tokens / Query", justify="right")

    for r in results:
        tbl_quality.add_row(
            r.engine_name,
            f"{r.ndcg5:.4f}",
            f"{r.ndcg10:.4f}",
            f"{r.top1_accuracy * 100:.1f}%",
            f"{int(r.avg_tokens_retrieved)}",
        )
    console.print("\n", tbl_quality)

    # Table 2: By Category Breakdown
    categories = sorted({c for r in results for c in r.ndcg10_by_category})
    tbl_cats = Table(
        title="2. Retrieval Quality by Query Category (NDCG@10)", header_style="bold green"
    )
    tbl_cats.add_column("Engine / Mode", style="cyan")
    for cat in categories:
        tbl_cats.add_column(cat.capitalize(), justify="right")

    for r in results:
        row = [r.engine_name]
        for cat in categories:
            val = r.ndcg10_by_category.get(cat, 0.0)
            row.append(f"{val:.4f}")
        tbl_cats.add_row(*row)
    console.print("\n", tbl_cats)

    # Table 3: Latency & Speed
    tbl_speed = Table(title="3. Latency & Throughput (ms)", header_style="bold blue")
    tbl_speed.add_column("Engine / Mode", style="cyan")
    tbl_speed.add_column("Cold Index", justify="right")
    tbl_speed.add_column("Query p50", justify="right", style="bold green")
    tbl_speed.add_column("Query p90", justify="right")
    tbl_speed.add_column("Query p95", justify="right", style="yellow")
    tbl_speed.add_column("Indexed Items", justify="right")

    for r in results:
        tbl_speed.add_row(
            r.engine_name,
            f"{r.index_time_ms:.1f} ms",
            f"{r.p50_ms:.2f} ms",
            f"{r.p90_ms:.2f} ms",
            f"{r.p95_ms:.2f} ms",
            f"{r.chunks_or_symbols}",
        )
    console.print("\n", tbl_speed)

    # Table 4: Architectural Matrix
    tbl_arch = Table(title="4. Architectural & Capability Comparison", header_style="bold yellow")
    tbl_arch.add_column("Dimension / Capability", style="cyan")
    tbl_arch.add_column("MinishLab/semble", style="green")
    tbl_arch.add_column("CodeAtlas", style="magenta")

    tbl_arch.add_row(
        "Primary Architecture",
        "Snippet/Chunk Search Engine",
        "Knowledge Graph & Code Intelligence Engine",
    )
    tbl_arch.add_row(
        "Embedding Model",
        "potion-code-16M-v2 (Model2Vec)",
        "BAAI/bge-small-en-v1.5 (33M fastembed)",
    )
    tbl_arch.add_row(
        "Vector Search Engine",
        "Vicinity / NumPy dot product",
        "Native sqlite-vec ANN (vec0 virtual table)",
    )
    tbl_arch.add_row(
        "Lexical Engine", "In-memory custom BM25", "SQLite FTS5 (Porter Stemming, full-text)"
    )
    tbl_arch.add_row(
        "AST Parser", "tree-sitter (py-tree-sitter)", "Python AST + Generic Multi-Lang Parser"
    )
    tbl_arch.add_row(
        "Graph Capabilities",
        "None (isolated chunks)",
        "Full Call/Import/Inheritance Knowledge Graph",
    )
    tbl_arch.add_row(
        "Context Expansion", "Heuristic reranker", "1-Hop Graph Neighborhood SQL Expansion"
    )
    tbl_arch.add_row(
        "Blast Radius Analysis", "Not supported", "Multi-hop BFS impact analysis for PRs"
    )
    tbl_arch.add_row(
        "Living Wiki Generator", "Not supported", "Topological living architecture documentation"
    )
    tbl_arch.add_row(
        "Agent MCP Integration",
        "Supported (search, find_related)",
        "Supported (search, graph, impact, wiki)",
    )
    tbl_arch.add_row(
        "Storage Model",
        "OS cache directory files",
        "Zero-dependency single SQLite file (.codeatlas/codeatlas.db)",
    )

    console.print("\n", tbl_arch)

    # Export JSON if requested
    if export_json:
        export_json.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "target_repo": str(repo_root),
            "total_tasks": len(tasks),
            "results": [
                {
                    "engine": r.engine_name,
                    "index_time_ms": r.index_time_ms,
                    "ndcg5": r.ndcg5,
                    "ndcg10": r.ndcg10,
                    "category_ndcg10": r.ndcg10_by_category,
                    "p50_ms": r.p50_ms,
                    "p90_ms": r.p90_ms,
                    "p95_ms": r.p95_ms,
                    "top1_accuracy": r.top1_accuracy,
                    "avg_tokens": r.avg_tokens_retrieved,
                }
                for r in results
            ],
        }
        export_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
        console.print(
            f"\n[bold green]✓ Benchmark results exported to: [underline]{export_json}[/underline][/bold green]"
        )


def main() -> None:
    """CLI entry point for benchmark comparison."""
    parser = argparse.ArgumentParser(description="Replicate Semble benchmarks for CodeAtlas.")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.home() / ".cache" / "semble-bench" / "flask" / "src" / "flask",
        help="Path to repository source directory to index and evaluate.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("benchmarks/data/flask.json"),
        help="Path to annotation JSON file with benchmark tasks.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of latency runs per query.",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        default=Path("benchmarks/results/codeatlas_vs_semble.json"),
        help="Path to export JSON benchmark summary.",
    )
    args = parser.parse_args()

    if not args.repo.exists():
        console.print(
            f"[bold red]Error: Target repository does not exist at {args.repo}[/bold red]"
        )
        raise SystemExit(1)

    if not args.annotations.exists():
        console.print(
            f"[bold red]Error: Annotations file does not exist at {args.annotations}[/bold red]"
        )
        raise SystemExit(1)

    compare_codeatlas_and_semble(
        repo_root=args.repo.resolve(),
        annotation_file=args.annotations.resolve(),
        runs=args.runs,
        export_json=args.export_json,
    )
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
