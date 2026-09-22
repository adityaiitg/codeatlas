"""Interactive Web Viewer for CodeAtlas Knowledge Graph and Search."""

from __future__ import annotations

import json
import logging
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from codeatlas.config import Settings
from codeatlas.graph.queries import GraphQueries
from codeatlas.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CodeAtlas 🗺️ Interactive Knowledge Graph</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --text-dim: #8b949e;
      --cyan: #58a6ff;
      --green: #3fb950;
      --yellow: #d29922;
      --purple: #bc8cff;
      --red: #f85149;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      overflow: hidden;
      display: flex;
      height: 100vh;
    }
    #sidebar {
      width: 360px;
      background: var(--card-bg);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      z-index: 10;
    }
    .header {
      padding: 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .header h1 { font-size: 18px; font-weight: 600; color: #fff; }
    .search-box {
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
    }
    .search-input {
      width: 100%;
      padding: 8px 12px;
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      color: #fff;
      font-size: 14px;
      outline: none;
    }
    .search-input:focus { border-color: var(--cyan); }
    .stats-bar {
      padding: 8px 16px;
      background: #11161d;
      border-bottom: 1px solid var(--border);
      display: flex;
      gap: 12px;
      font-size: 12px;
      color: var(--text-dim);
    }
    .badge {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 500;
    }
    .badge-class { background: #1f3b5c; color: var(--cyan); }
    .badge-func { background: #1c3d27; color: var(--green); }
    .badge-mod { background: #3d3118; color: var(--yellow); }
    #details {
      flex: 1;
      padding: 16px;
      overflow-y: auto;
      font-size: 13px;
      line-height: 1.5;
    }
    #graph-container {
      flex: 1;
      position: relative;
    }
    svg { width: 100%; height: 100%; }
    .node text {
      font-size: 11px;
      fill: var(--text);
      pointer-events: none;
      user-select: none;
    }
    .link { stroke: #30363d; stroke-opacity: 0.6; stroke-width: 1.5px; }
    .tooltip {
      position: absolute;
      background: #21262d;
      padding: 8px 12px;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 12px;
      pointer-events: none;
      display: none;
      z-index: 20;
    }
    .code-snippet {
      background: #0d1117;
      border: 1px solid var(--border);
      padding: 8px;
      border-radius: 4px;
      font-family: monospace;
      font-size: 11px;
      max-height: 180px;
      overflow-y: auto;
      white-space: pre-wrap;
      margin-top: 8px;
    }
    .caller-list { margin-top: 8px; }
    .caller-item {
      padding: 4px 6px;
      background: #0d1117;
      border-radius: 4px;
      margin-bottom: 4px;
      font-family: monospace;
      font-size: 11px;
      cursor: pointer;
    }
    .caller-item:hover { background: #1f3b5c; }
  </style>
</head>
<body>
  <div id="sidebar">
    <div class="header">
      <h1>CodeAtlas 🗺️</h1>
      <span style="font-size: 12px; color: var(--green);">● Live</span>
    </div>
    <div class="search-box">
      <input type="text" id="search" class="search-input" placeholder="Search symbols, functions, files...">
    </div>
    <div class="stats-bar" id="stats">
      <span>Loading graph data...</span>
    </div>
    <div id="details">
      <p style="color: var(--text-dim);">Click any node in the knowledge graph to view its definitions, callers, callees, and change blast radius.</p>
    </div>
  </div>
  <div id="graph-container">
    <div id="tooltip" class="tooltip"></div>
    <svg id="canvas"></svg>
  </div>

  <script>
    let graphData = { nodes: [], links: [] };
    let simulation, link, node;

    fetch('/api/graph')
      .then(res => res.json())
      .then(data => {
        graphData = data;
        document.getElementById('stats').innerHTML =
          `<span>Nodes: <b>${data.nodes.length}</b></span><span>Edges: <b>${data.links.length}</b></span>`;
        initGraph(data);
      })
      .catch(err => {
        document.getElementById('stats').innerHTML = '<span style="color:red">Failed to load graph</span>';
      });

    function initGraph(data) {
      const svg = d3.select("#canvas");
      const width = document.getElementById("graph-container").clientWidth;
      const height = document.getElementById("graph-container").clientHeight;

      const g = svg.append("g");

      svg.call(d3.zoom().scaleExtent([0.1, 4]).on("zoom", (e) => {
        g.attr("transform", e.transform);
      }));

      simulation = d3.forceSimulation(data.nodes)
        .force("link", d3.forceLink(data.links).id(d => d.id).distance(70))
        .force("charge", d3.forceManyBody().strength(-120))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collide", d3.forceCollide().radius(22));

      link = g.append("g")
        .selectAll("line")
        .data(data.links)
        .join("line")
        .attr("class", "link");

      node = g.append("g")
        .selectAll("g")
        .data(data.nodes)
        .join("g")
        .attr("class", "node")
        .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended))
        .on("click", (e, d) => inspectNode(d));

      node.append("circle")
        .attr("r", d => d.kind === 'module' ? 10 : (d.kind === 'class' ? 8 : 6))
        .attr("fill", d => {
          if (d.kind === 'class') return 'var(--cyan)';
          if (d.kind === 'function' || d.kind === 'method') return 'var(--green)';
          if (d.kind === 'module') return 'var(--yellow)';
          return 'var(--purple)';
        });

      node.append("text")
        .attr("x", 10)
        .attr("y", 3)
        .text(d => d.name || d.id.split(':').pop());

      simulation.on("tick", () => {
        link
          .attr("x1", d => d.source.x)
          .attr("y1", d => d.source.y)
          .attr("x2", d => d.target.x)
          .attr("y2", d => d.target.y);

        node.attr("transform", d => `translate(${d.x},${d.y})`);
      });
    }

    function inspectNode(d) {
      fetch(`/api/impact?target=${encodeURIComponent(d.id)}`)
        .then(res => res.json())
        .then(impact => {
          renderDetails(d, impact);
        })
        .catch(() => renderDetails(d, null));
    }

    function renderDetails(d, impact) {
      let callersHtml = '';
      if (impact && impact.direct_dependents && impact.direct_dependents.length > 0) {
        callersHtml = `<div style="margin-top:12px;"><b>Incoming Callers (${impact.direct_dependents.length}):</b><div class="caller-list">` +
          impact.direct_dependents.map(c => `<div class="caller-item" onclick="focusNode('${c}')">← ${c}</div>`).join('') +
          `</div></div>`;
      }

      const blastRadius = impact ? impact.affected_count : 0;
      const blastColor = blastRadius > 3 ? 'var(--red)' : 'var(--green)';

      document.getElementById('details').innerHTML = `
        <div style="margin-bottom: 12px;">
          <span class="badge badge-${d.kind}">${d.kind.toUpperCase()}</span>
          <h2 style="font-size: 16px; margin: 6px 0; color: #fff;">${d.name}</h2>
          <div style="color: var(--text-dim); font-size: 11px;">${d.file_path || d.id}</div>
        </div>
        <div style="background: #11161d; padding: 8px; border-radius: 4px; margin-bottom: 12px;">
          <div>Change Blast Radius: <b style="color: ${blastColor}">${blastRadius} affected node(s)</b></div>
        </div>
        ${callersHtml}
      `;
    }

    function focusNode(id) {
      const target = graphData.nodes.find(n => n.id === id || n.id.endsWith(':' + id));
      if (target) inspectNode(target);
    }

    function dragstarted(event) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }
    function dragged(event) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }
    function dragended(event) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }

    document.getElementById('search').addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase().trim();
      if (!q) {
        node.style("opacity", 1);
        link.style("opacity", 0.6);
        return;
      }
      node.style("opacity", d => {
        const match = (d.name && d.name.toLowerCase().includes(q)) || d.id.toLowerCase().includes(q);
        return match ? 1 : 0.15;
      });
      link.style("opacity", 0.05);
    });
  </script>
</body>
</html>
"""


class ViewerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for CodeAtlas web graph visualizer."""

    settings: Settings
    gq: GraphQueries

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default stdout logging to keep terminal clean."""
        logger.debug(format, *args)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query_params = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        if path == "/api/graph":
            self._handle_api_graph()
            return

        if path == "/api/impact":
            target = query_params.get("target", [""])[0]
            self._handle_api_impact(target)
            return

        if path == "/api/search":
            q = query_params.get("q", [""])[0]
            self._handle_api_search(q)
            return

        self.send_error(404, "Not Found")

    def _handle_api_graph(self) -> None:
        nodes = []
        for n in self.gq.G.nodes:
            kind = "module" if n.startswith("file:") else ("class" if ":" in n and "." not in n.split(":")[-1] else "function")
            name = n.split(":")[-1]
            nodes.append({"id": n, "name": name, "kind": kind, "file_path": n.split(":")[0]})

        links = []
        for u, v, data in self.gq.G.edges(data=True):
            links.append({"source": u, "target": v, "type": data.get("type", "calls")})

        payload = {"nodes": nodes, "links": links}
        self._send_json(payload)

    def _handle_api_impact(self, target: str) -> None:
        if not target or target not in self.gq.G:
            # Try fuzzy match
            matches = [n for n in self.gq.G.nodes if target in n]
            if matches:
                target = matches[0]
            else:
                self._send_json({"target": target, "direct_dependents": [], "affected_count": 0})
                return

        analysis = self.gq.impact_analysis(target)
        self._send_json(analysis)

    def _handle_api_search(self, query: str) -> None:
        with Retriever(self.settings) as retriever:
            results = retriever.search(query=query, limit=10)
        out = [
            {
                "chunk_id": r.chunk_id,
                "symbol_id": r.symbol_id,
                "file_path": r.file_path,
                "start_line": r.start_line,
                "end_line": r.end_line,
                "score": r.score,
                "content": r.content,
            }
            for r in results
        ]
        self._send_json(out)

    def _send_json(self, data: Any) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_viewer(
    repo_path: Path, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True
) -> None:
    """Start local web viewer server."""
    settings = Settings(repo_path=repo_path, data_dir=repo_path / ".codeatlas")
    if not settings.db_path.exists():
        raise FileNotFoundError(f"Database not found at {settings.db_path}. Please run 'codeatlas index' first.")

    gq = GraphQueries(settings.db_path)

    # Inject dependencies onto handler class
    ViewerHandler.settings = settings
    ViewerHandler.gq = gq

    server = ThreadingHTTPServer((host, port), ViewerHandler)
    url = f"http://{host}:{port}"
    logger.info("Serving CodeAtlas web viewer at %s", url)

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
