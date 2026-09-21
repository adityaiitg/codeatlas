"""Unit tests for Model Context Protocol (MCP) server."""

from pathlib import Path

from codeatlas.mcp.server import TOOLS, MCPServer


def test_mcp_server_initialization_and_tools(tmp_path: Path):
    server = MCPServer(repo_path=tmp_path)

    # 1. Tools list verification
    assert len(TOOLS) >= 4
    tool_names = [t["name"] for t in TOOLS]
    assert "codeatlas_search" in tool_names
    assert "codeatlas_symbol_graph" in tool_names
    assert "codeatlas_impact_analysis" in tool_names
    assert "codeatlas_status" in tool_names

    # 2. Test status tool handler
    status_res = server._tool_status()
    assert "status" in status_res

    # 3. Test empty graph tool calls return appropriate error messages
    graph_res = server._tool_graph({"symbol": "non_existent"})
    assert "error" in graph_res

    impact_res = server._tool_impact({"symbol": "non_existent"})
    assert "error" in impact_res


def test_mcp_request_dispatch(tmp_path: Path):
    server = MCPServer(repo_path=tmp_path)
    responses = []

    # Mock send_response and send_error
    server._send_response = lambda req_id, result: responses.append((req_id, result))
    server._send_error = lambda req_id, code, msg: responses.append((req_id, code, msg))

    # Test initialize
    server._handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert len(responses) == 1
    assert responses[0][0] == 1
    assert "serverInfo" in responses[0][1]

    # Test tools/list
    server._handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert len(responses) == 2
    assert "tools" in responses[1][1]

    # Test ping
    server._handle_request({"jsonrpc": "2.0", "id": 3, "method": "ping"})
    assert len(responses) == 3
    assert responses[2][1] == {}
