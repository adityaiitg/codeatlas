"""CodeAtlas Command Line Interface."""

from __future__ import annotations

import logging
from importlib.metadata import version as pkg_version
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from codeatlas.config import Settings
from codeatlas.git.tracker import GitTracker
from codeatlas.graph.queries import GraphQueries
from codeatlas.index.indexer import Indexer
from codeatlas.retrieval.retriever import Retriever
from codeatlas.wiki.generator import WikiGenerator

try:
    __version__ = pkg_version("codeatlas-cli")
except Exception:
    try:
        __version__ = pkg_version("codeatlas")
    except Exception:
        __version__ = "0.1.0"

console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"codeatlas {__version__}")
        raise typer.Exit()


def _configure_logging(verbose: bool, quiet: bool) -> None:
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
        force=True,
    )


app = typer.Typer(
    name="codeatlas",
    help="CodeAtlas: Local codebase intelligence, knowledge graph, living wiki & hybrid search.",
    add_completion=False,
    callback=lambda version: None,  # placeholder, real callback below
)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable detailed debug logging.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress all logs except warnings and errors.",
    ),
) -> None:
    """CodeAtlas: Local codebase intelligence, knowledge graph, living wiki & hybrid search."""
    _configure_logging(verbose=verbose, quiet=quiet)


def get_settings(repo_path: Path | None = None) -> Settings:
    """Instantiate settings for the given repository path."""
    target_path = (repo_path or Path(".")).resolve()
    data_dir = target_path / ".codeatlas"
    return Settings(repo_path=target_path, data_dir=data_dir)


@app.command()
def index(
    path: Path = typer.Argument(Path("."), help="Path to repository to index"),
    force: bool = typer.Option(False, "--force", "-f", help="Force full re-indexing of all files"),
    no_vectors: bool = typer.Option(False, "--no-vectors", help="Skip dense vector embeddings"),
):
    """Scan, parse AST, construct code graph, and index repository."""
    settings = get_settings(path)
    console.print(f"[bold blue]Indexing repository:[/bold blue] {settings.repo_path}")

    with console.status("[cyan]Scanning, parsing AST, and building graph...[/cyan]"):
        with Indexer(settings, embed_vectors=not no_vectors) as indexer:
            stats = indexer.index_repository(force=force)

    table = Table(title="CodeAtlas Indexing Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")

    table.add_row("Total Files Indexed", str(stats.get("files", 0)))
    table.add_row("Files Processed (This Run)", str(stats.get("indexed_files_this_run", 0)))
    table.add_row("Symbols (Classes/Functions)", str(stats.get("symbols", 0)))
    table.add_row("Graph Edges (Calls/Imports)", str(stats.get("edges", 0)))
    table.add_row("Search Chunks", str(stats.get("chunks", 0)))
    table.add_row("Vector Embeddings", str(stats.get("vectors", 0)))

    console.print(table)
    console.print(
        f"[bold green]✓ Indexing complete![/bold green] Database saved to [italic]{settings.db_path}[/italic]\n"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query (natural language or symbol name)"),
    path: Path = typer.Option(Path("."), "--path", "-p", help="Repository path"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of results to return"),
    mode: str = typer.Option(
        "hybrid", "--mode", "-m", help="Search mode: hybrid, lexical, semantic"
    ),
    expand: bool = typer.Option(
        True, "--expand/--no-expand", help="Expand graph neighborhood context"
    ),
):
    """Hybrid code search with BM25, dense vectors, and graph neighborhood expansion."""
    settings = get_settings(path)
    if not settings.db_path.exists():
        console.print("[bold red]Error:[/bold red] Repository has not been indexed yet.")
        console.print("Run [bold cyan]codeatlas index[/bold cyan] first.")
        raise typer.Exit(code=1)

    with Retriever(settings) as retriever:
        results = retriever.search(query=query, limit=limit, mode=mode, expand_graph=expand)

    if not results:
        console.print(f"[yellow]No results found for query:[/yellow] '{query}'")
        return

    console.print(
        f"\n[bold green]Found {len(results)} matches for:[/bold green] '{query}' (Mode: {mode})\n"
    )

    for idx, r in enumerate(results, start=1):
        rel_path = Path(r.file_path).name
        title = f"#{idx} | {rel_path}:{r.start_line}-{r.end_line} | Score: {r.score:.4f}"
        if r.is_definition:
            title += " [bold green][DEF][/bold green]"

        syntax = Syntax(
            r.content,
            lexer="python" if r.file_path.endswith(".py") else "text",
            line_numbers=True,
            start_line=r.start_line,
            theme="monokai",
        )

        console.print(Panel(syntax, title=title, expand=False, border_style="blue"))

        # Display expanded graph neighbors if present
        if r.neighbors:
            neighbor_strs = [
                f"[bold cyan]{n['kind']}[/bold cyan] [underline]{n['name']}[/underline] ({Path(n['file_path']).name}:{n['start_line']})"
                for n in r.neighbors[:4]
            ]
            console.print(
                "  [italic dim]↳ Graph Neighbors:[/italic dim] " + " | ".join(neighbor_strs)
            )
            console.print()


@app.command()
def graph(
    symbol: str = typer.Argument(..., help="Symbol name or node ID to inspect"),
    path: Path = typer.Option(Path("."), "--path", "-p", help="Repository path"),
    depth: int = typer.Option(1, "--depth", "-d", help="Neighborhood expansion depth"),
):
    """Explore callers, callees, and dependencies for a symbol."""
    settings = get_settings(path)
    if not settings.db_path.exists():
        console.print(
            "[bold red]Error:[/bold red] Repository not indexed. Run 'codeatlas index' first."
        )
        raise typer.Exit(code=1)

    gq = GraphQueries(settings.db_path)

    # Find matching nodes in graph, prioritizing defined symbols over call targets
    matches = [n for n in gq.G.nodes if symbol in n]
    if not matches:
        console.print(f"[yellow]Symbol '{symbol}' not found in knowledge graph.[/yellow]")
        return

    matches.sort(
        key=lambda x: (
            x.startswith("call:"),
            x.startswith("module:"),
            x.startswith("symbol:"),
            len(x),
        )
    )
    target_node = matches[0]
    tree = Tree(f"[bold cyan]Symbol:[/bold cyan] {target_node}")

    # Callers (who calls this)
    callers = gq.get_callers(target_node)
    callers_branch = tree.add(f"[bold yellow]Incoming Callers ({len(callers)}):[/bold yellow]")
    for c in callers:
        callers_branch.add(f"[green]←[/green] {c}")

    # Callees (who this calls)
    callees = gq.get_callees(target_node)
    callees_branch = tree.add(f"[bold yellow]Outgoing Calls ({len(callees)}):[/bold yellow]")
    for c in callees:
        callees_branch.add(f"[blue]→[/blue] {c}")

    # Neighborhood expansion
    expanded = gq.expand_neighborhood([target_node], depth=depth)
    other_neighbors = [
        n for n in expanded if n != target_node and n not in callers and n not in callees
    ]
    if other_neighbors:
        other_branch = tree.add(
            f"[bold magenta]Connected Context ({len(other_neighbors)}):[/bold magenta]"
        )
        for o in other_neighbors[:10]:
            other_branch.add(f"[dim]{o}[/dim]")

    console.print(tree)


@app.command()
def impact(
    symbol: str = typer.Argument(..., help="Symbol name to analyze for change blast radius"),
    path: Path = typer.Option(Path("."), "--path", "-p", help="Repository path"),
):
    """Analyze change impact and blast radius if a symbol is modified."""
    settings = get_settings(path)
    if not settings.db_path.exists():
        console.print(
            "[bold red]Error:[/bold red] Repository not indexed. Run 'codeatlas index' first."
        )
        raise typer.Exit(code=1)

    gq = GraphQueries(settings.db_path)
    matches = [n for n in gq.G.nodes if symbol in n]
    if not matches:
        console.print(f"[yellow]Symbol '{symbol}' not found in knowledge graph.[/yellow]")
        return

    matches.sort(
        key=lambda x: (
            x.startswith("call:"),
            x.startswith("module:"),
            x.startswith("symbol:"),
            len(x),
        )
    )
    target_node = matches[0]
    analysis = gq.impact_analysis(target_node)

    console.print(
        Panel(
            f"[bold red]Impact Analysis for:[/bold red] [underline]{target_node}[/underline]\n\n"
            f"• [bold]Direct Dependents:[/bold] {len(analysis['direct_dependents'])}\n"
            f"• [bold]Total Downstream Affected Nodes:[/bold] {analysis['affected_count']}",
            title="Blast Radius Assessment",
            border_style="red" if analysis["affected_count"] > 5 else "yellow",
        )
    )

    if analysis["direct_dependents"]:
        table = Table(title="Directly Dependent Symbols", show_header=True)
        table.add_column("Dependent Symbol", style="cyan")
        for dep in analysis["direct_dependents"]:
            table.add_row(dep)
        console.print(table)


@app.command()
def wiki(
    action: str = typer.Argument("generate", help="Action: 'generate' or 'view'"),
    topic: str = typer.Option(
        "index", "--topic", "-t", help="Topic to view (e.g. index, architecture, workflows)"
    ),
    llm: bool = typer.Option(
        False, "--llm", help="Use configured LLM for AI-powered chapter summarization"
    ),
    path: Path = typer.Option(Path("."), "--path", "-p", help="Repository path"),
):
    """Generate or view the living repository wiki."""
    settings = get_settings(path)
    if not settings.db_path.exists():
        console.print(
            "[bold red]Error:[/bold red] Repository not indexed. Run 'codeatlas index' first."
        )
        raise typer.Exit(code=1)

    if action == "generate":
        with console.status("[cyan]Generating living wiki specifications...[/cyan]"):
            generator = WikiGenerator(settings, use_llm=llm)
            files = generator.generate()

        console.print(f"[bold green]✓ Living Wiki generated![/bold green] ({len(files)} chapters)")
        console.print(f"Output directory: [italic]{settings.wiki_dir}[/italic]")
        for f in files:
            console.print(f"  • {f.relative_to(settings.repo_path)}")

    elif action == "view":
        # Sanitize topic to prevent path traversal
        safe_topic = Path(topic).name
        target_file = settings.wiki_dir / f"{safe_topic}.md"
        if not target_file.exists():
            # Try in modules
            target_file = settings.wiki_dir / "modules" / f"{safe_topic}.md"

        if not target_file.resolve().is_relative_to(settings.wiki_dir.resolve()):
            console.print("[bold red]Error:[/bold red] Invalid topic name.")
            raise typer.Exit(code=1)

        if not target_file.exists():
            console.print(
                f"[yellow]Wiki chapter '{safe_topic}' not found in {settings.wiki_dir}[/yellow]"
            )
            return

        content = target_file.read_text(encoding="utf-8")
        console.print(Markdown(content))


@app.command()
def status(
    path: Path = typer.Option(Path("."), "--path", "-p", help="Repository path"),
):
    """Display current index status, database size, and git state."""
    settings = get_settings(path)
    git_tracker = GitTracker(settings.repo_path)

    head_commit = (
        git_tracker.get_head_commit() if git_tracker.is_git_repo else "Not a git repository"
    )
    changed = git_tracker.get_changed_files() if git_tracker.is_git_repo else []

    db_exists = settings.db_path.exists()
    db_size = f"{settings.db_path.stat().st_size / 1024:.1f} KB" if db_exists else "Not created"

    table = Table(title="CodeAtlas Repository Status", show_header=True, header_style="bold blue")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Repository Path", str(settings.repo_path))
    table.add_row("Database Path", str(settings.db_path))
    table.add_row("Database Size", db_size)
    table.add_row("Git HEAD Commit", str(head_commit)[:12])
    table.add_row("Unindexed/Modified Files", str(len(changed)))

    if db_exists:
        from codeatlas.graph.builder import GraphBuilder

        with GraphBuilder(settings.db_path) as gb:
            stats = gb.get_stats()
        table.add_row("Indexed Files", str(stats.get("files", 0)))
        table.add_row("Indexed Symbols", str(stats.get("symbols", 0)))
        table.add_row("Graph Edges", str(stats.get("edges", 0)))

    console.print(table)


@app.command()
def diff(
    path: Path = typer.Option(Path("."), "--path", "-p", help="Repository path"),
):
    """Show files changed since last commit that may need re-indexing."""
    settings = get_settings(path)
    git_tracker = GitTracker(settings.repo_path)
    if not git_tracker.is_git_repo:
        console.print("[yellow]Path is not a git repository.[/yellow]")
        return

    changed = git_tracker.get_changed_files()
    if not changed:
        console.print("[bold green]✓ Working tree is clean. Index is up to date.[/bold green]")
        return

    console.print(f"[bold yellow]Found {len(changed)} changed files:[/bold yellow]")
    for f in changed:
        console.print(f"  • {f}")


@app.command()
def export(
    output: Path = typer.Option(
        Path("codeatlas_graph.json"), "--output", "-o", help="Output file path"
    ),
    format: str = typer.Option("json", "--format", "-f", help="Export format: json, graphml, dot"),
    path: Path = typer.Option(Path("."), "--path", "-p", help="Repository path"),
):
    """Export the codebase knowledge graph to JSON, GraphML, or DOT."""
    settings = get_settings(path)
    if not settings.db_path.exists():
        console.print(
            "[bold red]Error:[/bold red] Repository not indexed. Run 'codeatlas index' first."
        )
        raise typer.Exit(code=1)

    gq = GraphQueries(settings.db_path)
    try:
        out = gq.export_graph(output_path=output, format=format)
        console.print(
            f"[bold green]✓ Exported knowledge graph to:[/bold green] {out} ({len(gq.G.nodes)} nodes, {len(gq.G.edges)} edges)"
        )
    except Exception as exc:
        console.print(f"[bold red]Export failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None


@app.command()
def config(
    path: Path = typer.Option(Path("."), "--path", "-p", help="Repository path"),
):
    """View active CodeAtlas settings, paths, and model configurations."""
    settings = get_settings(path)
    table = Table(
        title="CodeAtlas Active Configuration", show_header=True, header_style="bold cyan"
    )
    table.add_column("Setting", style="yellow")
    table.add_column("Value", style="green")
    table.add_column("Env Variable", style="dim")

    configs = [
        ("Repository Path", str(settings.repo_path), "CODEATLAS_REPO_PATH"),
        ("Data Directory", str(settings.data_dir), "CODEATLAS_DATA_DIR"),
        ("Database Path", str(settings.db_path), "-"),
        ("Wiki Directory", str(settings.wiki_dir), "-"),
        ("Embedding Model", settings.embedding_model, "CODEATLAS_EMBEDDING_MODEL"),
        ("Embedding Dim", str(settings.embedding_dim), "CODEATLAS_EMBEDDING_DIM"),
        ("LLM Provider", settings.llm_provider, "CODEATLAS_LLM_PROVIDER"),
        ("LLM Model", settings.llm_model, "CODEATLAS_LLM_MODEL"),
        ("BM25 Weight", str(settings.bm25_weight), "CODEATLAS_BM25_WEIGHT"),
        ("Semantic Weight", str(settings.semantic_weight), "CODEATLAS_SEMANTIC_WEIGHT"),
        ("RRF k", str(settings.rrf_k), "CODEATLAS_RRF_K"),
        ("Search Limit", str(settings.search_limit), "CODEATLAS_SEARCH_LIMIT"),
        (
            "Expansion Depth",
            str(settings.graph_expansion_depth),
            "CODEATLAS_GRAPH_EXPANSION_DEPTH",
        ),
    ]

    for name, val, env in configs:
        table.add_row(name, val, env)

    console.print(table)


@app.command()
def mcp(
    path: Path = typer.Option(Path("."), "--path", "-p", help="Repository path"),
):
    """Start the Model Context Protocol (MCP) server over standard I/O."""
    from codeatlas.mcp.server import MCPServer

    server = MCPServer(repo_path=path)
    server.run()


if __name__ == "__main__":
    app()
