"""Unit tests for agent MCP auto-installer."""

from __future__ import annotations

import json
from pathlib import Path

from codeatlas.installer.agent_installer import configure_agent


def test_configure_agent_claude(tmp_path: Path, monkeypatch):
    test_claude_cfg = tmp_path / ".claude.json"
    monkeypatch.setattr(
        "codeatlas.installer.agent_installer.AGENT_CONFIG_PATHS",
        {"claude": [test_claude_cfg]},
    )

    # 1. Install
    results = configure_agent("claude", remove=False)
    assert len(results) == 1
    assert results[0].action == "configured"
    assert test_claude_cfg.exists()

    data = json.loads(test_claude_cfg.read_text(encoding="utf-8"))
    assert "codeatlas" in data["mcpServers"]
    assert data["mcpServers"]["codeatlas"]["command"] == "codeatlas"

    # 2. Re-install (already present)
    results2 = configure_agent("claude", remove=False)
    assert results2[0].action == "already_present"

    # 3. Uninstall
    results3 = configure_agent("claude", remove=True)
    assert results3[0].action == "removed"
    data3 = json.loads(test_claude_cfg.read_text(encoding="utf-8"))
    assert "codeatlas" not in data3.get("mcpServers", {})


def test_configure_agent_jsonc_and_backup(tmp_path: Path, monkeypatch):
    test_cfg = tmp_path / "mcp.json"
    jsonc_content = """{
        // Existing user server
        "mcpServers": {
            "my-custom-server": {
                "command": "custom",
                "args": ["serve"], /* inline comment */
            },
        },
    }"""
    test_cfg.write_text(jsonc_content, encoding="utf-8")
    monkeypatch.setattr(
        "codeatlas.installer.agent_installer.AGENT_CONFIG_PATHS",
        {"cursor": [test_cfg]},
    )

    results = configure_agent("cursor", remove=False)
    assert len(results) == 1
    assert results[0].action == "configured"

    # Verify backup was created
    backup_file = tmp_path / "mcp.json.bak"
    assert backup_file.exists()
    assert backup_file.read_text(encoding="utf-8") == jsonc_content

    # Verify existing server was preserved alongside codeatlas
    data = json.loads(test_cfg.read_text(encoding="utf-8"))
    assert "my-custom-server" in data["mcpServers"]
    assert "codeatlas" in data["mcpServers"]


def test_configure_agent_corrupted_safeguard(tmp_path: Path, monkeypatch):
    test_cfg = tmp_path / "broken.json"
    broken_content = "INVALID {{{ JSON ::::"
    test_cfg.write_text(broken_content, encoding="utf-8")
    monkeypatch.setattr(
        "codeatlas.installer.agent_installer.AGENT_CONFIG_PATHS",
        {"claude": [test_cfg]},
    )

    results = configure_agent("claude", remove=False)
    assert len(results) == 1
    assert results[0].action == "error_parse_failed"

    # Crucial: content was NOT overwritten or wiped!
    assert test_cfg.read_text(encoding="utf-8") == broken_content
