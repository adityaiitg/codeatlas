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
