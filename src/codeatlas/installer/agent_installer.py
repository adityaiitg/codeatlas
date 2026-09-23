"""Auto-configure CodeAtlas MCP server in installed coding agents (Claude Code, Cursor, OpenCode, Codex)."""

from __future__ import annotations

import json
import logging
import re
import shutil
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
    action: str  # "configured", "already_present", "not_found", "removed", "error_parse_failed"


def strip_jsonc_comments(content: str) -> str:
    """Strip JavaScript single-line, multi-line comments and trailing commas from JSONC."""
    # Remove multi-line comments /* ... */
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    # Remove single-line comments // ... without matching inside quotes
    pattern = re.compile(r'("(?:\\.|[^"\\])*")|//.*')

    def replace(match: re.Match) -> str:
        if match.group(1):
            return match.group(1)
        return ""

    content = pattern.sub(replace, content)
    # Remove trailing commas before } or ]
    content = re.sub(r",\s*([\]}])", r"\1", content)
    return content


def safe_load_config(config_file: Path) -> dict[str, Any] | None:
    """Safely load JSON/JSONC config. Returns None if file exists but is invalid."""
    if not config_file.exists():
        return {}
    raw = config_file.read_text(encoding="utf-8")
    if not raw.strip():
        return {}
    # Try standard json first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try stripping JSONC comments and trailing commas
    try:
        cleaned = strip_jsonc_comments(raw)
        return json.loads(cleaned)
    except Exception as exc:
        logger.error("Failed to parse config %s: %s", config_file, exc)
        return None


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
            parsed_data = safe_load_config(config_file)
            if parsed_data is None:
                # Do NOT overwrite existing user configuration if it fails to parse!
                logger.error("Skipping %s due to parse failure to prevent data loss.", config_file)
                results.append(
                    InstallResult(
                        agent=agent,
                        config_path=config_file,
                        action="error_parse_failed",
                    )
                )
                continue
            data = parsed_data

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
                backup_path = config_file.with_name(f"{config_file.name}.bak")
                try:
                    shutil.copy2(config_file, backup_path)
                except OSError as exc:
                    logger.warning("Could not create backup for %s: %s", config_file, exc)
                config_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
                results.append(
                    InstallResult(agent=agent, config_path=config_file, action="removed")
                )
        else:
            # Add configuration
            if agent == "opencode":
                mcp_dict = data.setdefault("mcp", {})
                if "codeatlas" in mcp_dict:
                    results.append(
                        InstallResult(
                            agent=agent, config_path=config_file, action="already_present"
                        )
                    )
                    continue
                mcp_dict["codeatlas"] = get_mcp_config_entry(agent)
            else:
                mcp_dict = data.setdefault("mcpServers", {})
                if "codeatlas" in mcp_dict:
                    results.append(
                        InstallResult(
                            agent=agent, config_path=config_file, action="already_present"
                        )
                    )
                    continue
                mcp_dict["codeatlas"] = get_mcp_config_entry(agent)

            if config_file.exists():
                backup_path = config_file.with_name(f"{config_file.name}.bak")
                try:
                    shutil.copy2(config_file, backup_path)
                except OSError as exc:
                    logger.warning("Could not create backup for %s: %s", config_file, exc)
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
