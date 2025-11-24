[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![Tests](https://github.com/cch1999/pocketeer/actions/workflows/run_tests.yml/badge.svg)](https://github.com/cch1999/pocketeer/actions/workflows/run_tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
[![PyPI version](https://img.shields.io/pypi/v/pocketeer)](https://pypi.org/project/pocketeer/)
[![Docs](https://img.shields.io/badge/docs-latest-blueviolet.svg)](https://pocketeer.readthedocs.io/en/latest/)

# Pocketeer

<div align="center">
  <img src="docs/assets/logo.png" alt="Pocketeer Logo" width="250"/>
</div>

A lightweight, fast pocket finder in Python.

Pocketeer detects binding pockets and cavities in protein structures using the alpha-sphere method based on Delaunay tessellation—similar to the popular fpocket, but with a modern, Pythonic interface. It's lightweight, fast, and easy to use as both a Python library and a command-line tool. Pocketeer natively supports `atomarrays` from [`biotite`](https://www.biotite-python.org/latest/), making it fully compatible with [`atomworks`](https://rosettacommons.github.io/atomworks/latest/).

**Read the [full documentation here](https://pocketeer.readthedocs.io/en/latest/).**

> 🚧 **Warning:** This software is in alpha. Some features (e.g., similarity calculation) are not yet implemented and there may be bugs. If you encounter any issues, please open an issue on the repository. 🚧


## Installation

```bash
pip install pocketeer
```


<details>
<summary><strong>Other options</strong></summary>

Install with notebook viewer:

```bash
pip install pocketeer[vis]
```

Install using [uv](https://github.com/astral-sh/uv):

```bash
uv add pocketeer
```

Install with extra dependencies, e.g. for notebook/visualization support:

```bash
uv add 'pocketeer[vis]'
uv add 'pocketeer[dev, vis]'
```


</details>




## Quick Start

<div align="center">
  <img src="docs/assets/pocketeer_example.png" alt="Example pocketeer output" width="650"/>
  <p><em>Example output image from Pocketeer</em></p>
</div>



A couple quick tips before you start:
- Pocketeer automatically removes hydrogens from your structure, and is designed to work with them absent, so you don't need to worry about them.
- By default, Pocketeer removes all waters and hetero atoms from your input structure. Set `ignore_hetero` to `False` if you don't want this behavior (e.g. to take into account a metal ion in the pocket).

### Python API

```python
import pocketeer as pt

# Load structure from PDB file
atomarray = pt.load_structure("protein.pdb")

# Detect pockets with default parameters
pockets = pt.find_pockets(atomarray)

# Print results
for pocket in pockets:
    print(f"Pocket {pocket.pocket_id}:")
    print(f"  Score: {pocket.score:.2f}")
    print(f"  Volume: {pocket.volume:.1f} Å³")
    print(f"  Spheres: {pocket.n_spheres}")
```

### Custom Parameters

```python
import pocketeer as pt

# Load structure
atomarray = pt.load_structure("protein.pdb")

# Find pockets with custom parameters
pockets = pt.find_pockets(
    atomarray,
    r_min=3.0,           # Minimum sphere radius (Å)
    r_max=6.0,           # Maximum sphere radius (Å)
    min_spheres=30,      # Minimum spheres per pocket cluster
    merge_distance=2.5,  # Distance threshold for clustering
    sasa_threshold=25.0,  # SASA threshold for buried spheres (Å²)
)
```

### Command-Line Interface

```bash
# Basic usage
pocketeer protein.pdb # also works with .cif files 

# Custom output directory
pocketeer protein.pdb --o results/

# Adjust parameters
pocketeer protein.pdb --r-min 2.5 --r-max 7.0 --min-spheres 25

# Fine-tune buried sphere detection
pocketeer protein.pdb --sasa-threshold 25.0 --polar-probe 1.6
```

### Output Files

The CLI generates these output files:

- `pockets.json` - All pocket descriptors in JSON format
- `json/` - Subdirectory with individual JSON files for each pocket (`pocket_0.json`, `pocket_1.json`, etc.)
- `summary.txt` - Human-readable summary (unless `--no-summary` is used)

## Algorithm

Pocketeer implements a simplified version of the fpocket algorithm:

1. **Delaunay Tessellation** - Compute Delaunay triangulation of protein atoms
2. **Alpha-Sphere Detection** - Extract circumspheres of tetrahedra within radius bounds
3. **Polarity Labeling** - Calculate SASA (solvent accessible surface area) for atoms defining each sphere
4. **Surface Filtering** - Filter to buried spheres using SASA threshold (spheres with mean SASA < threshold)
5. **Clustering** - Group buried spheres into pockets using graph connectivity
6. **Scoring** - Rank pockets by volume and geometric features

## Limitations

- Simplified scoring compared to fpocket (no hydrophobicity, flexibility, etc.)
- Volume estimation is approximate (voxel-based)

## Visualization in Notebooks

Pocketeer integrates smoothly with Jupyter and scientific Python notebooks. You can directly visualize detected pockets for rapid exploration:

```python
import pocketeer as pt

# Load structure
atomarray = pt.load_structure("protein.pdb")
pockets = pt.find_pockets(atomarray)

# Visualize the pockets in your notebook
pt.view_pockets(atomarray, pockets)
```

<div align="center">
  <img src="docs/assets/vis_example_2.png" alt="Pocketeer notebook visualization example" width="600"/>
  <p><em>Example notebook visualization of predicted binding pockets colored by cluster.</em></p>
</div>

## Citation

If you use Pocketeer in your research, please cite the original fpocket paper:

> Le Guilloux, V., Schmidtke, P., & Tuffery, P. (2009). 
> Fpocket: An open source platform for ligand pocket detection. 
> BMC Bioinformatics, 10(1), 168.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run `ruff check --fix && ruff format` before committing (or run `just all`)
5. Submit a pull request

### Development installation

```bash
git clone https://github.com/cch1999/pocketeer.git
cd pocketeer
pip install -e ".[dev]" # or uv sync --dev
```

## Support

For bugs and feature requests, please open an issue on GitHub.

