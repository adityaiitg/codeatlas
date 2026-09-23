"""Symbol resolution and call-graph linking pass."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from codeatlas.graph.schema import get_db_connection
from codeatlas.models.relationships import EdgeType

logger = logging.getLogger(__name__)


class GraphLinker:
    """Resolves symbolic call and import edges to canonical defined symbol node IDs."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def link(self) -> int:
        """Run post-indexing symbol resolution pass over all edges.

        Returns the number of call/inheritance edges successfully resolved.
        """
        if not self.db_path.exists():
            return 0

        with get_db_connection(self.db_path) as conn:
            # 1. Load all defined symbols
            cursor = conn.execute("SELECT node_id, file_path, name, kind, parent_id FROM symbols")
            symbols = cursor.fetchall()
            if not symbols:
                return 0

            # Symbol lookup structures
            by_name: dict[str, list[dict]] = {}
            by_file_and_name: dict[tuple[str, str], str] = {}
            sym_by_id: dict[str, dict] = {}

            for nid, fpath, name, kind, parent_id in symbols:
                sym_info = {
                    "node_id": nid,
                    "file_path": fpath,
                    "name": name,
                    "kind": kind,
                    "parent_id": parent_id,
                }
                by_name.setdefault(name, []).append(sym_info)
                by_file_and_name[(fpath, name)] = nid
                sym_by_id[nid] = sym_info

            # 2. Load all import edges to track file-level imports
            file_imports: dict[str, list[dict]] = {}
            imp_cursor = conn.execute(
                "SELECT source_id, target_id, metadata FROM edges WHERE edge_type = ?",
                (EdgeType.IMPORTS.value,),
            )
            for src, tgt, meta_str in imp_cursor.fetchall():
                meta = json.loads(meta_str) if meta_str else {}
                file_imports.setdefault(src, []).append(
                    {
                        "target_id": tgt,
                        "module": meta.get("module", ""),
                        "alias": meta.get("alias", ""),
                        "name": meta.get("name", ""),
                    }
                )

            # 3. Load all CALLS and INHERITS edges that need resolution
            edge_cursor = conn.execute(
                """SELECT source_id, target_id, edge_type, metadata
                   FROM edges
                   WHERE edge_type IN (?, ?) AND (target_id LIKE 'call:%' OR target_id LIKE 'symbol:%')""",
                (EdgeType.CALLS.value, EdgeType.INHERITS.value),
            )
            edges_to_resolve = edge_cursor.fetchall()
            if not edges_to_resolve:
                return 0

            resolved_edges: list[tuple[str, str, str, str]] = []
            edges_to_delete: list[tuple[str, str, str]] = []

            for source_id, raw_target, edge_type, meta_str in edges_to_resolve:
                caller_sym = sym_by_id.get(source_id)
                caller_file = caller_sym["file_path"] if caller_sym else None
                resolved_target: str | None = None

                if edge_type == EdgeType.CALLS.value and raw_target.startswith("call:"):
                    callee = raw_target[5:]
                    resolved_target = self._resolve_callee(
                        callee=callee,
                        caller_sym=caller_sym,
                        caller_file=caller_file,
                        by_name=by_name,
                        by_file_and_name=by_file_and_name,
                        file_imports=file_imports,
                    )
                elif edge_type == EdgeType.INHERITS.value and raw_target.startswith("symbol:"):
                    base_name = raw_target[7:]
                    resolved_target = self._resolve_symbol_name(
                        name=base_name,
                        caller_file=caller_file,
                        by_name=by_name,
                        by_file_and_name=by_file_and_name,
                    )

                if resolved_target and resolved_target != source_id:
                    meta = json.loads(meta_str) if meta_str else {}
                    meta["original_target"] = raw_target
                    meta["resolved"] = True
                    resolved_edges.append(
                        (
                            source_id,
                            resolved_target,
                            edge_type,
                            json.dumps(meta),
                        )
                    )
                    edges_to_delete.append((source_id, raw_target, edge_type))

            # 4. Write resolved edges to SQLite
            if resolved_edges:
                conn.executemany(
                    """INSERT OR REPLACE INTO edges (source_id, target_id, edge_type, metadata)
                       VALUES (?, ?, ?, ?)""",
                    resolved_edges,
                )
                conn.executemany(
                    "DELETE FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
                    edges_to_delete,
                )
                conn.commit()

            return len(resolved_edges)

    def _resolve_callee(
        self,
        callee: str,
        caller_sym: dict | None,
        caller_file: str | None,
        by_name: dict[str, list[dict]],
        by_file_and_name: dict[tuple[str, str], str],
        file_imports: dict[str, list[dict]],
    ) -> str | None:
        if not callee:
            return None

        # 1. Method call on self/cls: e.g. "self.helper" or "cls.helper"
        if caller_sym and (callee.startswith("self.") or callee.startswith("cls.")):
            method_name = callee.split(".", 1)[1]
            if caller_file:
                node_parts = caller_sym["node_id"].split(":")[-1].split(".")
                if len(node_parts) >= 2:
                    class_name = node_parts[0]
                    target_nid = f"{caller_file}:{class_name}.{method_name}"
                    if (caller_file, f"{class_name}.{method_name}") in by_file_and_name:
                        return target_nid

        # 2. Local function in the same file e.g. "helper()"
        if caller_file:
            if (caller_file, callee) in by_file_and_name:
                return by_file_and_name[(caller_file, callee)]

        # 3. Check imported symbols
        if caller_file:
            mod_key = f"file:{caller_file}"
            imports = file_imports.get(mod_key, [])
            for imp in imports:
                imp_name = imp.get("name")
                imp_alias = imp.get("alias")
                imp_mod = imp.get("module", "")

                # Direct import: from module import func
                if imp_name == callee or imp_alias == callee:
                    candidates = by_name.get(imp_name, [])
                    for cand in candidates:
                        cand_fpath = cand["file_path"].replace("/", ".").replace("\\", ".")
                        if imp_mod and imp_mod in cand_fpath:
                            return cand["node_id"]
                    if candidates:
                        return candidates[0]["node_id"]

                # Module call: import module; module.func()
                if "." in callee:
                    mod_part, fn_part = callee.rsplit(".", 1)
                    if imp_alias == mod_part or imp_name == mod_part or imp_mod.endswith(mod_part):
                        candidates = by_name.get(fn_part, [])
                        for cand in candidates:
                            cand_fpath = cand["file_path"].replace("/", ".").replace("\\", ".")
                            if mod_part in cand_fpath:
                                return cand["node_id"]

        # 4. If callee has dot qualification e.g. "Class.method"
        if "." in callee:
            parts = callee.split(".")
            short_name = parts[-1]
            candidates = by_name.get(short_name, [])
            if len(candidates) == 1:
                return candidates[0]["node_id"]

        # 5. Global unique symbol match
        candidates = by_name.get(callee, [])
        if len(candidates) == 1:
            return candidates[0]["node_id"]

        return None

    def _resolve_symbol_name(
        self,
        name: str,
        caller_file: str | None,
        by_name: dict[str, list[dict]],
        by_file_and_name: dict[tuple[str, str], str],
    ) -> str | None:
        if not name:
            return None
        if caller_file and (caller_file, name) in by_file_and_name:
            return by_file_and_name[(caller_file, name)]

        candidates = by_name.get(name, [])
        if len(candidates) == 1:
            return candidates[0]["node_id"]
        return None
