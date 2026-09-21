"""Unit tests for CodeAtlas CLI commands using Typer's CliRunner."""

from pathlib import Path

from typer.testing import CliRunner

from codeatlas.cli import app

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "codeatlas" in result.stdout


def test_cli_config(tmp_path: Path):
    result = runner.invoke(app, ["config", "--path", str(tmp_path)])
    assert result.exit_code == 0
    assert "CodeAtlas Active Configuration" in result.stdout
    assert "Embedding Model" in result.stdout


def test_cli_status_unindexed(tmp_path: Path):
    result = runner.invoke(app, ["status", "--path", str(tmp_path)])
    assert result.exit_code == 0
    assert "CodeAtlas Repository Status" in result.stdout
    assert "Not created" in result.stdout


def test_cli_diff(tmp_path: Path):
    result = runner.invoke(app, ["diff", "--path", str(tmp_path)])
    assert result.exit_code == 0


def test_cli_index_search_graph_impact_wiki_export(tmp_path: Path):
    # 1. Create a sample Python repo
    repo = tmp_path / "repo"
    repo.mkdir()
    py_file = repo / "calculator.py"
    py_file.write_text(
        """
class Calculator:
    def add(self, a, b):
        return a + b

    def compute(self, x, y):
        return self.add(x, y)
""",
        encoding="utf-8",
    )

    # 2. Test codeatlas index
    res_index = runner.invoke(app, ["index", str(repo), "--no-vectors"])
    assert res_index.exit_code == 0
    assert "Indexing complete" in res_index.stdout

    # 3. Test codeatlas status after indexing
    res_status = runner.invoke(app, ["status", "--path", str(repo)])
    assert res_status.exit_code == 0
    assert "Indexed Files" in res_status.stdout

    # 4. Test codeatlas search
    res_search = runner.invoke(
        app, ["search", "Calculator", "--path", str(repo), "--mode", "lexical"]
    )
    assert res_search.exit_code == 0
    assert "Calculator" in res_search.stdout

    # 5. Test codeatlas graph
    res_graph = runner.invoke(app, ["graph", "Calculator", "--path", str(repo)])
    assert res_graph.exit_code == 0
    assert "Calculator" in res_graph.stdout

    # 6. Test codeatlas impact
    res_impact = runner.invoke(app, ["impact", "Calculator", "--path", str(repo)])
    assert res_impact.exit_code == 0
    assert "Blast Radius Assessment" in res_impact.stdout

    # 7. Test codeatlas export
    export_json = repo / "graph_out.json"
    res_export = runner.invoke(
        app,
        ["export", "--output", str(export_json), "--format", "json", "--path", str(repo)],
    )
    assert res_export.exit_code == 0
    assert export_json.exists()

    # 8. Test codeatlas wiki generate and view
    res_wiki_gen = runner.invoke(app, ["wiki", "generate", "--path", str(repo)])
    assert res_wiki_gen.exit_code == 0
    assert "Living Wiki generated" in res_wiki_gen.stdout

    res_wiki_view = runner.invoke(
        app, ["wiki", "view", "--topic", "architecture", "--path", str(repo)]
    )
    assert res_wiki_view.exit_code == 0
    assert "System Architecture" in res_wiki_view.stdout
