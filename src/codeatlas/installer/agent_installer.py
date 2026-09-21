"""Auto-configure CodeAtlas MCP server in installed coding agents (Claude Code, Cursor, OpenCode, Codex)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HOME = Path.home()

AGENT_CONFIG_PATHS: dict[str, list[Path]] = {
    "claude": [HOME / ".claude.json"],
    "cursor": [HOME / ".cursor" / "mcp.json", Path(".cursor") / "mcp.json"],
    "opencode": [HOME / ".config" / "opencode" / "config.json"],
    "codex": [HOME / ".codex" / "config.json"],
}


@dataclass
class InstallResult:
    agent: str
    config_path: Path
    action: str  # "configured", "already_present", "not_found", "removed"


def get_mcp_config_entry(agent: str) -> dict[str, Any]:
    """Return agent-specific MCP server configuration dict."""
    if agent == "opencode":
        return {
            "type": "local",
            "command": ["codeatlas", "mcp"],
            "enabled": True,
        }
    return {
        "command": "codeatlas",
        "args": ["mcp"],
        "type": "stdio",
    }


def configure_agent(agent: str, remove: bool = False) -> list[InstallResult]:
    """Configure or remove CodeAtlas MCP in target agent's config file."""
    results: list[InstallResult] = []
    paths = AGENT_CONFIG_PATHS.get(agent, [])

    for config_file in paths:
        if not config_file.parent.exists():
            if remove:
                continue
            config_file.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {}
        if config_file.exists():
            try:
                data = json.loads(config_file.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Could not parse %s: %s", config_file, exc)
                data = {}

        if remove:
            # Remove configuration
            modified = False
            if agent == "opencode":
                mcp_dict = data.get("mcp", {})
                if "codeatlas" in mcp_dict:
                    del mcp_dict["codeatlas"]
                    modified = True
            else:
                mcp_dict = data.get("mcpServers", {})
                if "codeatlas" in mcp_dict:
                    del mcp_dict["codeatlas"]
                    modified = True

            if modified:
                config_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
                results.append(InstallResult(agent=agent, config_path=config_file, action="removed"))
        else:
            # Add configuration
            if agent == "opencode":
                mcp_dict = data.setdefault("mcp", {})
                if "codeatlas" in mcp_dict:
                    results.append(InstallResult(agent=agent, config_path=config_file, action="already_present"))
                    continue
                mcp_dict["codeatlas"] = get_mcp_config_entry(agent)
            else:
                mcp_dict = data.setdefault("mcpServers", {})
                if "codeatlas" in mcp_dict:
                    results.append(InstallResult(agent=agent, config_path=config_file, action="already_present"))
                    continue
                mcp_dict["codeatlas"] = get_mcp_config_entry(agent)

            config_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            results.append(InstallResult(agent=agent, config_path=config_file, action="configured"))

    return results


def install_all(agents: list[str] | None = None) -> list[InstallResult]:
    """Configure CodeAtlas MCP server across all or specified agents."""
    targets = agents or list(AGENT_CONFIG_PATHS.keys())
    all_results = []
    for agent in targets:
        all_results.extend(configure_agent(agent, remove=False))
    return all_results


def uninstall_all(agents: list[str] | None = None) -> list[InstallResult]:
    """Remove CodeAtlas MCP server across all or specified agents."""
    targets = agents or list(AGENT_CONFIG_PATHS.keys())
    all_results = []
    for agent in targets:
        all_results.extend(configure_agent(agent, remove=True))
    return all_results
