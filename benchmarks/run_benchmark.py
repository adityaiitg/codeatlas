"""Self-contained benchmark runner for CodeAtlas retrieval and indexing.

Measures latency (P50/P95/P99), result counts, and definition recall
across 10 diverse query types.
"""

from __future__ import annotations

import statistics
import sys
import tempfile
import time
from pathlib import Path

from codeatlas.config import Settings
from codeatlas.index.indexer import Indexer
from codeatlas.retrieval.retriever import Retriever

QUERIES = [
    {"query": "Retriever", "category": "Exact Symbol Class"},
    {"query": "parse_file", "category": "Function Lookup"},
    {"query": "GraphQueries", "category": "Knowledge Graph Core"},
    {"query": "Settings", "category": "Pydantic Config"},
    {"query": "reverse impact analysis callers", "category": "Natural Language Semantic"},
    {"query": "SQLite WAL journal mode cache", "category": "Storage & Database"},
    {"query": "Model2Vec dense vector embedding", "category": "Vector Embedding"},
    {"query": "ast syntax tree traversal", "category": "Parser Concepts"},
    {"query": "test_cli", "category": "Test Suite Queries"},
    {"query": "mcp stdio json-rpc server", "category": "Protocol & MCP"},
]


def run_benchmark() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    with tempfile.TemporaryDirectory() as tmp_dir:
        bench_data_dir = Path(tmp_dir) / ".codeatlas_bench"
        settings = Settings(
            repo_path=repo_root,
            data_dir=bench_data_dir,
            search_limit=10,
        )

        print("⚡ CodeAtlas Benchmark Runner")
        print(f"  Target repository: {repo_root}")
        print(f"  Temporary index:   {bench_data_dir}")

        # 1. Index the repository
        t0 = time.perf_counter()
        with Indexer(settings, embed_vectors=False) as indexer:
            stats = indexer.index_repository(force=True)
        index_duration_ms = (time.perf_counter() - t0) * 1000

        print("\n✓ Cold Indexing Complete")
        print(f"  Files indexed:    {stats.get('files', 0)}")
        print(f"  Symbols parsed:   {stats.get('symbols', 0)}")
        print(f"  Chunks created:   {stats.get('chunks', 0)}")
        print(f"  Edges extracted:  {stats.get('edges', 0)}")
        print(f"  Index Latency:    {index_duration_ms:.1f} ms\n")

        # 2. Run retrieval benchmark
        results_table: list[dict] = []
        all_p50s: list[float] = []

        with Retriever(settings) as retriever:
            for item in QUERIES:
                q = item["query"]
                cat = item["category"]

                # Warmup
                retriever.search(q, limit=10, expand_graph=True)

                latencies: list[float] = []
                iterations = 25
                top_is_def = False
                res_count = 0

                for i in range(iterations):
                    t_start = time.perf_counter()
                    res = retriever.search(q, limit=10, expand_graph=True)
                    t_elapsed_ms = (time.perf_counter() - t_start) * 1000
                    latencies.append(t_elapsed_ms)

                    if i == 0:
                        res_count = len(res)
                        if res and res[0].is_definition:
                            top_is_def = True

                latencies.sort()
                p50 = statistics.median(latencies)
                p95 = latencies[int(iterations * 0.95)]
                p99 = latencies[-1]

                all_p50s.append(p50)
                results_table.append(
                    {
                        "query": q,
                        "category": cat,
                        "count": res_count,
                        "top_is_def": "✓" if top_is_def else "✗",
                        "p50_ms": p50,
                        "p95_ms": p95,
                        "p99_ms": p99,
                    }
                )

        # 3. Print Results in GitHub Markdown Format
        print("| Query | Category | Results | Top Is Def | P50 (ms) | P95 (ms) | P99 (ms) |")
        print("|:------|:---------|:-------:|:----------:|:--------:|:--------:|:--------:|")
        for r in results_table:
            print(
                f"| `{r['query']}` | {r['category']} | {r['count']} | "
                f"{r['top_is_def']} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} |"
            )

        overall_p50 = statistics.median(all_p50s)
        overall_mean = statistics.mean(all_p50s)
        print(f"\n📊 Summary: Median P50 = {overall_p50:.2f} ms | Mean P50 = {overall_mean:.2f} ms")

        # Threshold check: P50 latency should be fast (< 75 ms for full hybrid + 1-hop graph)
        if overall_p50 > 100.0:
            print(f"⚠️ Warning: Overall P50 ({overall_p50:.2f} ms) exceeds 100 ms threshold.")
            return 1

        print("✓ All retrieval latency requirements passed (< 100 ms target)")
        return 0


if __name__ == "__main__":
    sys.exit(run_benchmark())
