"""Comprehensive performance benchmark suite for CodeAtlas."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from codeatlas.config import Settings
from codeatlas.graph.queries import GraphQueries
from codeatlas.index.indexer import Indexer
from codeatlas.retrieval.retriever import Retriever
from codeatlas.wiki.generator import WikiGenerator

console = Console()


def format_ms(val_s: float) -> str:
    """Format seconds into milliseconds."""
    return f"{val_s * 1000:.2f} ms"


def run_latency_benchmark(func, iterations: int = 50) -> dict[str, float]:
    """Run a function multiple times and return latency statistics in seconds."""
    times: list[float] = []
    # Warmup
    func()

    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times.sort()
    p50_idx = int(iterations * 0.50)
    p95_idx = int(iterations * 0.95)

    return {
        "min": min(times),
        "mean": statistics.mean(times),
        "p50": times[p50_idx],
        "p95": times[p95_idx],
        "max": max(times),
    }


def benchmark_indexing(repo_path: Path, tmp_dir: Path) -> dict[str, float]:
    """Benchmark full repository indexing and incremental re-indexing."""
    settings = Settings(repo_path=repo_path, data_dir=tmp_dir / ".codeatlas_bench")

    # 1. Cold full index (AST + Graph + FTS5, no vector download in test)
    t0 = time.perf_counter()
    with Indexer(settings, embed_vectors=False) as indexer:
        stats = indexer.index_repository(force=True)
    cold_time = time.perf_counter() - t0

    # 2. Incremental re-index (all files cached)
    t0 = time.perf_counter()
    with Indexer(settings, embed_vectors=False) as indexer:
        inc_stats = indexer.index_repository(force=False)
    inc_time = time.perf_counter() - t0

    return {
        "cold_index_time_s": cold_time,
        "incremental_index_time_s": inc_time,
        "files_indexed": stats.get("files", 0),
        "symbols_indexed": stats.get("symbols", 0),
        "edges_indexed": stats.get("edges", 0),
        "chunks_indexed": stats.get("chunks", 0),
        "incremental_files_processed": inc_stats.get("indexed_files_this_run", 0),
    }


def run_full_benchmark(repo_path: Path | None = None) -> None:
    """Execute complete CodeAtlas performance benchmarking."""
    target_repo = (repo_path or Path(".")).resolve()
    settings = Settings(repo_path=target_repo, data_dir=target_repo / ".codeatlas")

    console.print(
        f"\n[bold cyan]═══ CodeAtlas Performance Benchmark Suite ═══[/bold cyan]\nTarget Repository: [underline]{target_repo}[/underline]\n"
    )

    # -------------------------------------------------------------
    # 1. Indexing Throughput
    # -------------------------------------------------------------
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_bench_dir:
        bench_tmp = Path(tmp_bench_dir)
        idx_res = benchmark_indexing(target_repo, bench_tmp)

    table_idx = Table(
        title="1. Indexing & Parsing Throughput",
        show_header=True,
        header_style="bold magenta",
    )
    table_idx.add_column("Pipeline Stage", style="cyan")
    table_idx.add_column("Result / Speed", style="green", justify="right")

    files = idx_res["files_indexed"]
    cold_s = idx_res["cold_index_time_s"]
    throughput = files / cold_s if cold_s > 0 else 0

    table_idx.add_row("Total Files Scanned & Indexed", f"{files} files")
    table_idx.add_row("Symbols Discovered (Classes/Funcs)", f"{idx_res['symbols_indexed']}")
    table_idx.add_row("Graph Relationships (Edges)", f"{idx_res['edges_indexed']}")
    table_idx.add_row("Search Chunks Generated", f"{idx_res['chunks_indexed']}")
    table_idx.add_row("Cold Index Duration (AST + Graph + FTS)", f"{cold_s:.3f} s")
    table_idx.add_row("Parsing Throughput", f"{throughput:.1f} files/sec")
    table_idx.add_row(
        "Incremental Re-Index (Manifest Cache Hit)",
        f"{idx_res['incremental_index_time_s'] * 1000:.2f} ms",
    )

    console.print(table_idx)

    # -------------------------------------------------------------
    # 2. Search Latencies
    # -------------------------------------------------------------
    if not settings.db_path.exists():
        console.print("[yellow]Database not found, indexing repo first...[/yellow]")
        with Indexer(settings, embed_vectors=False) as idx:
            idx.index_repository()

    retriever = Retriever(settings)

    # Benchmark Lexical
    lex_stats = run_latency_benchmark(
        lambda: retriever.search("Indexer", limit=10, mode="lexical", expand_graph=False),
        iterations=50,
    )

    # Benchmark Semantic (Vector ANN via sqlite-vec or in-memory)
    sem_stats = run_latency_benchmark(
        lambda: retriever.search(
            "AST parser and graph constructor", limit=10, mode="semantic", expand_graph=False
        ),
        iterations=20,
    )

    # Benchmark Hybrid + Graph Expansion (Full pipeline)
    hybrid_stats = run_latency_benchmark(
        lambda: retriever.search(
            "Indexer.index_repository", limit=10, mode="hybrid", expand_graph=True
        ),
        iterations=20,
    )

    table_search = Table(
        title="2. Search Query Latency (p50 / p95 / mean)",
        show_header=True,
        header_style="bold green",
    )
    table_search.add_column("Search Mode", style="cyan")
    table_search.add_column("Min", justify="right")
    table_search.add_column("Mean", justify="right")
    table_search.add_column("Median (p50)", justify="right", style="bold green")
    table_search.add_column("p95", justify="right", style="yellow")
    table_search.add_column("Max", justify="right")

    for name, s in [
        ("FTS5 BM25 Lexical", lex_stats),
        ("Dense Vector ANN (fastembed + sqlite-vec)", sem_stats),
        ("Hybrid (RRF + 1-Hop Graph Context)", hybrid_stats),
    ]:
        table_search.add_row(
            name,
            format_ms(s["min"]),
            format_ms(s["mean"]),
            format_ms(s["p50"]),
            format_ms(s["p95"]),
            format_ms(s["max"]),
        )

    console.print(table_search)

    # -------------------------------------------------------------
    # 3. Graph Operations Latency
    # -------------------------------------------------------------
    gq = GraphQueries(settings.db_path)
    test_node = next(iter(gq.G.nodes)) if gq.G.nodes else "Indexer"

    # 1-hop expansion
    exp_stats = run_latency_benchmark(
        lambda: gq.expand_neighborhood([test_node], depth=1),
        iterations=50,
    )

    # Impact analysis BFS
    impact_stats = run_latency_benchmark(
        lambda: gq.impact_analysis(test_node),
        iterations=50,
    )

    # Export JSON
    export_stats = run_latency_benchmark(
        lambda: gq.export_graph(settings.data_dir / "bench_export.json", format="json"),
        iterations=20,
    )

    table_graph = Table(
        title="3. Knowledge Graph & Analysis Latency",
        show_header=True,
        header_style="bold blue",
    )
    table_graph.add_column("Operation", style="cyan")
    table_graph.add_column("p50", justify="right", style="bold green")
    table_graph.add_column("p95", justify="right", style="yellow")
    table_graph.add_column("Mean", justify="right")

    table_graph.add_row(
        "1-Hop Neighborhood Expansion (SQL)",
        format_ms(exp_stats["p50"]),
        format_ms(exp_stats["p95"]),
        format_ms(exp_stats["mean"]),
    )
    table_graph.add_row(
        "Change Blast Radius BFS Impact Analysis",
        format_ms(impact_stats["p50"]),
        format_ms(impact_stats["p95"]),
        format_ms(impact_stats["mean"]),
    )
    table_graph.add_row(
        f"Graph Export to JSON ({len(gq.G.nodes)} nodes, {len(gq.G.edges)} edges)",
        format_ms(export_stats["p50"]),
        format_ms(export_stats["p95"]),
        format_ms(export_stats["mean"]),
    )

    console.print(table_graph)

    # Clean up bench export file
    bench_file = settings.data_dir / "bench_export.json"
    if bench_file.exists():
        bench_file.unlink()

    # -------------------------------------------------------------
    # 4. Living Wiki Generation
    # -------------------------------------------------------------
    wiki_gen = WikiGenerator(settings, use_llm=False)
    t0 = time.perf_counter()
    wiki_files = wiki_gen.generate()
    wiki_time = time.perf_counter() - t0

    table_wiki = Table(
        title="4. Documentation Generation Throughput",
        show_header=True,
        header_style="bold magenta",
    )
    table_wiki.add_column("Metric", style="cyan")
    table_wiki.add_column("Value", style="green", justify="right")
    table_wiki.add_row("Generated Wiki Chapters", f"{len(wiki_files)} chapters")
    table_wiki.add_row("Mermaid Architecture & Workflows", "Included")
    table_wiki.add_row("Total Generation Time", f"{wiki_time:.3f} s")
    table_wiki.add_row(
        "Generation Speed",
        f"{len(wiki_files) / wiki_time:.1f} chapters/sec" if wiki_time > 0 else "N/A",
    )

    console.print(table_wiki)

    # -------------------------------------------------------------
    # 5. Database Footprint
    # -------------------------------------------------------------
    db_size_kb = settings.db_path.stat().st_size / 1024 if settings.db_path.exists() else 0
    table_db = Table(
        title="5. Storage Footprint",
        show_header=True,
        header_style="bold yellow",
    )
    table_db.add_column("Asset", style="cyan")
    table_db.add_column("Size", style="green", justify="right")
    table_db.add_row(
        "SQLite DB (Symbols + Edges + Chunks + Vectors + FTS5)", f"{db_size_kb:.1f} KB"
    )

    console.print(table_db)
    console.print("[bold green]✓ Benchmark Suite Completed Successfully![/bold green]\n")


if __name__ == "__main__":
    run_full_benchmark()
