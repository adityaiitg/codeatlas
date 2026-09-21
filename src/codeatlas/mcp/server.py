"""Model Context Protocol (MCP) stdio server exposing CodeAtlas code intelligence tools."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from codeatlas.config import Settings
from codeatlas.graph.builder import GraphBuilder
from codeatlas.graph.queries import GraphQueries
from codeatlas.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

SERVER_NAME = "codeatlas"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "codeatlas_search",
        "description": "Perform hybrid code search using BM25, dense embeddings, and graph context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language or symbol query to search in code",
                },
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "lexical", "semantic"],
                    "default": "hybrid",
                    "description": "Retrieval mode",
                },
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of chunks to return",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "codeatlas_symbol_graph",
        "description": "Inspect callers, callees, and connected context for a specific code symbol.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Symbol name or node identifier to inspect",
                },
                "depth": {
                    "type": "integer",
                    "default": 1,
                    "description": "Neighborhood expansion depth",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "codeatlas_impact_analysis",
        "description": "Analyze change blast radius to discover all symbols that would break if a given symbol is modified.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Symbol name to analyze for change blast radius",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "codeatlas_status",
        "description": "Get CodeAtlas index stats (file count, symbol count, edges, chunks, vectors).",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


class MCPServer:
    """Stdio-based JSON-RPC 2.0 Model Context Protocol server."""

    def __init__(self, repo_path: Path | None = None):
        target_path = (repo_path or Path(".")).resolve()
        data_dir = target_path / ".codeatlas"
        self.settings = Settings(repo_path=target_path, data_dir=data_dir)

    def run(self) -> None:
        """Run the main MCP request-response loop reading from standard input."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                self._send_error(None, -32700, f"Parse error: {exc}")
                continue

            self._handle_request(request)

    def _handle_request(self, request: dict[str, Any]) -> None:
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if method == "initialize":
            self._send_response(
                req_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION,
                    },
                },
            )
        elif method == "notifications/initialized":
            pass  # Client notification, no response required
        elif method == "ping":
            self._send_response(req_id, {})
        elif method == "tools/list":
            self._send_response(req_id, {"tools": TOOLS})
        elif method == "tools/call":
            self._handle_tool_call(req_id, params)
        else:
            self._send_error(req_id, -32601, f"Method not found: {method}")

    def _handle_tool_call(self, req_id: Any, params: dict[str, Any]) -> None:
        tool_name = params.get("name")
        args = params.get("arguments", {})

        try:
            if tool_name == "codeatlas_search":
                res = self._tool_search(args)
            elif tool_name == "codeatlas_symbol_graph":
                res = self._tool_graph(args)
            elif tool_name == "codeatlas_impact_analysis":
                res = self._tool_impact(args)
            elif tool_name == "codeatlas_status":
                res = self._tool_status()
            else:
                self._send_error(req_id, -32602, f"Unknown tool: {tool_name}")
                return

            self._send_response(
                req_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(res, indent=2),
                        }
                    ]
                },
            )
        except Exception as exc:
            logger.exception("Tool %s execution failed", tool_name)
            self._send_response(
                req_id,
                {
                    "isError": True,
                    "content": [
                        {
                            "type": "text",
                            "text": f"Tool execution failed: {exc}",
                        }
                    ],
                },
            )

    def _tool_search(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        query = args.get("query", "")
        mode = args.get("mode", "hybrid")
        limit = int(args.get("limit", 5))

        with Retriever(self.settings) as retriever:
            results = retriever.search(query=query, limit=limit, mode=mode, expand_graph=True)
            return [
                {
                    "file_path": r.file_path,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "score": round(r.score, 4),
                    "is_definition": r.is_definition,
                    "content": r.content,
                    "neighbors": [
                        {"name": n["name"], "kind": n["kind"], "file": n["file_path"]}
                        for n in r.neighbors[:3]
                    ],
                }
                for r in results
            ]

    def _tool_graph(self, args: dict[str, Any]) -> dict[str, Any]:
        symbol = args.get("symbol", "")
        depth = int(args.get("depth", 1))

        gq = GraphQueries(self.settings.db_path)
        matches = [n for n in gq.G.nodes if symbol in n]
        if not matches:
            return {"error": f"Symbol '{symbol}' not found in knowledge graph"}

        target_node = matches[0]
        callers = gq.get_callers(target_node)
        callees = gq.get_callees(target_node)
        neighbors = gq.expand_neighborhood([target_node], depth=depth)

        return {
            "symbol": target_node,
            "incoming_callers": callers,
            "outgoing_callees": callees,
            "connected_context": [n for n in neighbors if n != target_node],
        }

    def _tool_impact(self, args: dict[str, Any]) -> dict[str, Any]:
        symbol = args.get("symbol", "")
        gq = GraphQueries(self.settings.db_path)
        matches = [n for n in gq.G.nodes if symbol in n]
        if not matches:
            return {"error": f"Symbol '{symbol}' not found in knowledge graph"}

        target_node = matches[0]
        analysis = gq.impact_analysis(target_node)
        return {
            "symbol": target_node,
            "direct_dependents": analysis["direct_dependents"],
            "all_affected": analysis["all_affected"],
            "affected_count": analysis["affected_count"],
        }

    def _tool_status(self) -> dict[str, Any]:
        if not self.settings.db_path.exists():
            return {"status": "unindexed", "message": "Repository not indexed yet."}

        with GraphBuilder(self.settings.db_path) as gb:
            stats = gb.get_stats()

        return {
            "status": "ready",
            "repo_path": str(self.settings.repo_path),
            "database_path": str(self.settings.db_path),
            "stats": stats,
        }

    def _send_response(self, req_id: Any, result: dict[str, Any]) -> None:
        payload = {"jsonrpc": "2.0", "id": req_id, "result": result}
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()

    def _send_error(self, req_id: Any, code: int, message: str) -> None:
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
