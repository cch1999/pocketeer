"""Command-line interface for pocketeer."""

import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .api import find_pockets
from .core import Pocket
from .utils import (
    SUPPORTED_FORMATS,
    load_structure,
    write_individual_pocket_jsons,
    write_pockets_as_pdb,
    write_pockets_json,
    write_summary,
)

app = typer.Typer(
    name="pocketeer",
    help="A minimal, fpocket-style pocket finder in pure Python/Numpy.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def _validate_input_file(structure_file: str) -> None:
    """Validate input structure file."""
    structure_path = Path(structure_file)
    if not structure_path.exists():
        console.print(
            f"[bold red]Error: Structure file '{structure_file}' does not exist[/bold red]"
        )
        raise typer.Exit(1)

    # Check if it's a supported format
    if structure_path.suffix.lower() not in SUPPORTED_FORMATS:
        console.print(
            f"[bold red]Error: Unsupported file format '{structure_path.suffix}'. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}[/bold red]"
        )
        raise typer.Exit(1)


def _make_table(pockets: list[Pocket]) -> Table:
    """Make a table of the detected pockets."""
    table = Table(
        title="Detected Pockets",
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
    )
    table.add_column("ID", style="cyan", justify="center")
    table.add_column("Score", style="magenta", justify="center")
    table.add_column("Volume (A³)", style="green", justify="center")
    table.add_column("Spheres", style="yellow", justify="center")
    table.add_column("Residues", style="blue", justify="center")
    for pocket in pockets:
        table.add_row(
            str(pocket.pocket_id),
            f"{pocket.score:.2f}",
            f"{pocket.volume:.1f}",
            str(pocket.n_spheres),
            str(len(pocket.residues)),
        )
    return table


@app.command()
def detect(
    structure_file: str = typer.Argument(..., help="Input structure file (PDB, mmCIF, etc.)"),
    out_dir: str = typer.Option("pockets", "--out-dir", "-o", help="Output directory"),
    *,
    r_min: float = typer.Option(3.0, "--r-min", help="Minimum sphere radius (Å)"),
    r_max: float = typer.Option(6.0, "--r-max", help="Maximum sphere radius (Å)"),
    polar_probe_radius: float = typer.Option(
        1.4, "--polar-probe", help="Probe radius for polarity (Å)"
    ),
    sasa_threshold: float = typer.Option(
        20.0, "--sasa-threshold", help="SASA threshold for buried spheres (Å²)"
    ),
    merge_distance: float = typer.Option(
        1.75, "--merge-distance", help="Distance threshold for merging (Å)"
    ),
    min_spheres: int = typer.Option(35, "--min-spheres", help="Minimum spheres per pocket cluster"),
    ignore_hydrogens: bool = typer.Option(True, "--ignore-hydrogens", help="Ignore hydrogen atoms"),
    ignore_water: bool = typer.Option(True, "--ignore-water", help="Ignore water molecules"),
    ignore_hetero: bool = typer.Option(True, "--ignore-hetero", help="Ignore hetero atoms"),
    no_summary: bool = typer.Option(False, "--no-summary", help="Do not write summary file"),
) -> None:
    """Detect binding pockets in a protein structure."""
    # Validate input file
    _validate_input_file(structure_file)

    # Read structure using Biotite
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Reading structure...", total=None)
        atoms = load_structure(structure_file)
        progress.update(task, description=f"Loaded {len(atoms)} atoms")

    start_time = time.time()

    # Detect pockets
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Detecting pockets...", total=None)
        pockets = find_pockets(
            atoms,
            r_min=r_min,
            r_max=r_max,
            polar_probe_radius=polar_probe_radius,
            sasa_threshold=sasa_threshold,
            merge_distance=merge_distance,
            min_spheres=min_spheres,
            ignore_hydrogens=ignore_hydrogens,
            ignore_water=ignore_water,
            ignore_hetero=ignore_hetero,
        )
        progress.update(task, description="Pocket detection complete")

    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Write outputs
    console.print("Writing output files...")
    write_pockets_json(str(out_path / "pockets.json"), pockets)
    write_individual_pocket_jsons(str(out_path), pockets)
    write_pockets_as_pdb(str(out_path / "alphaspheres.pdb"), pockets)
    if not no_summary:
        write_summary(str(out_path / "summary.txt"), pockets, structure_file)

    # Display results
    console.print(
        f"\n[bold green]Found {len(pockets)} pocket(s) in "
        f"{time.time() - start_time:.2f} seconds[/bold green]\n"
    )

    if pockets:
        table = _make_table(pockets)
        console.print(table)
    else:
        console.print("[yellow]No pockets detected with current parameters[/yellow]")

    console.print(f"\n[cyan]Results written to {out_dir}/[/cyan]")


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
