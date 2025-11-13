# CLI Guide

## Basic Usage

Detect pockets in a protein structure:

```bash
pocketeer protein.pdb  # also works with .cif files
```

## Common Options

- `-o DIR` or `--out-dir DIR` : Output directory (default: `pockets/`)
- `--r-min X` : Minimum sphere radius (Å, default: 3.0)
- `--r-max X` : Maximum sphere radius (Å, default: 6.0)
- `--min-spheres N` : Minimum spheres per pocket (default: 35)
- `--merge-distance X` : Distance threshold for clustering (Å, default: 2.0)
- `--ignore-hetero False` : Keep ligands/ions (default: True)
- `--no-summary` : Skip summary file generation

## Examples

### Custom Output Directory

```bash
pocketeer protein.pdb -o results/
```

### Custom Parameters

```bash
pocketeer protein.pdb --r-min 2.5 --r-max 7.0 --min-spheres 25
```

### Batch Processing

```bash
for pdb in *.pdb; do
    pocketeer "$pdb" -o "results/${pdb%.pdb}"
done
```

## Output Files

The CLI generates these output files in the specified output directory:

- `pockets.json` - All pocket descriptors in JSON format
- `json/` - Subdirectory with individual JSON files for each pocket (`pocket_0.json`, `pocket_1.json`, etc.)
- `alphaspheres.pdb` - Alpha-spheres as PDB file for visualization in PyMOL or ChimeraX
- `summary.txt` - Human-readable summary (unless `--no-summary` is used)

## Getting Help

For help and all available options:

```bash
pocketeer --help
```

Open `alphaspheres.pdb` in PyMOL or ChimeraX to visualize detected pockets.
