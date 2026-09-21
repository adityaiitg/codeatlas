"""Graph query algorithms for neighborhood expansion and impact analysis."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from codeatlas.graph.schema import connect_db
from codeatlas.models.relationships import EdgeType


class GraphQueries:
    """In-memory graph backed by NetworkX, loaded from SQLite."""

    def __init__(self, db_path: Path | str):
        self.G = nx.DiGraph()
        self._load_from_sqlite(Path(db_path))

    def _load_from_sqlite(self, db_path: Path):
        """Load the graph from the SQLite edges table."""
        if not db_path.exists():
            return
        conn = connect_db(db_path)
        rows = conn.execute("SELECT source_id, target_id, edge_type FROM edges").fetchall()
        for source, target, etype in rows:
            self.G.add_edge(source, target, type=etype)
        conn.close()

    def expand_neighborhood(
        self,
        node_ids: list[str],
        depth: int = 1,
        edge_types: list[EdgeType] | None = None,
    ) -> list[str]:
        """1-hop or 2-hop graph expansion from initial retrieval results."""
        type_set = {e.value for e in edge_types} if edge_types else None
        expanded = set(node_ids)
        frontier = set(node_ids)

        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid not in self.G:
                    continue
                # Outgoing edges
                for _, neighbor, data in self.G.edges(nid, data=True):
                    if type_set is None or data.get("type") in type_set:
                        next_frontier.add(neighbor)
                # Incoming edges
                for predecessor, _, data in self.G.in_edges(nid, data=True):
                    if type_set is None or data.get("type") in type_set:
                        next_frontier.add(predecessor)
            frontier = next_frontier - expanded
            expanded |= frontier

        return list(expanded)

    def impact_analysis(self, node_id: str) -> dict:
        """What would break if this symbol changes?

        Walks the dependency graph to find all affected nodes:
        - Callers (predecessors via CALLS edges) — code that invokes this symbol
        - Defined children (successors via DEFINES edges) — methods/functions owned by this symbol
        """
        if node_id not in self.G:
            return {"direct_dependents": [], "all_affected": [], "affected_count": 0}

        direct: list[str] = []
        # Callers: anything that calls this symbol would be affected
        for src, _, d in self.G.in_edges(node_id, data=True):
            if d.get("type") == EdgeType.CALLS.value:
                direct.append(src)
        # Defined children: methods/classes defined by this symbol would be affected
        for _, tgt, d in self.G.edges(node_id, data=True):
            if d.get("type") == EdgeType.DEFINES.value:
                direct.append(tgt)

        # BFS to find all transitively affected nodes
        all_affected: set[str] = set()
        frontier = set(direct)
        while frontier:
            all_affected |= frontier
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid not in self.G:
                    continue
                for src, _, d in self.G.in_edges(nid, data=True):
                    if d.get("type") == EdgeType.CALLS.value and src not in all_affected:
                        next_frontier.add(src)
                for _, tgt, d in self.G.edges(nid, data=True):
                    if d.get("type") == EdgeType.DEFINES.value and tgt not in all_affected:
                        next_frontier.add(tgt)
            frontier = next_frontier

        return {
            "direct_dependents": direct,
            "all_affected": list(all_affected),
            "affected_count": len(all_affected),
        }

    def get_callers(self, node_id: str) -> list[str]:
        """Find all symbols that call this one."""
        return [
            src
            for src, _, d in self.G.in_edges(node_id, data=True)
            if d.get("type") == EdgeType.CALLS.value
        ]

    def get_callees(self, node_id: str) -> list[str]:
        """Find all symbols called by this one."""
        return [
            tgt
            for _, tgt, d in self.G.edges(node_id, data=True)
            if d.get("type") == EdgeType.CALLS.value
        ]

    def trace_flow(self, entrypoint: str, max_depth: int = 10) -> list[str]:
        """Trace execution from entrypoint through call graph (DFS)."""
        path: list[str] = []
        visited: set[str] = set()
        self._dfs_calls(entrypoint, path, visited, max_depth, 0)
        return path

    def _dfs_calls(
        self,
        node_id: str,
        path: list[str],
        visited: set[str],
        max_depth: int,
        depth: int,
    ):
        if depth > max_depth or node_id in visited or node_id not in self.G:
            return
        visited.add(node_id)
        path.append(node_id)
        for callee in self.get_callees(node_id):
            self._dfs_calls(callee, path, visited, max_depth, depth + 1)
