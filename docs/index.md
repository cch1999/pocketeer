# Pocketeer

<div align="center">
  <img src="assets/logo.png" alt="Pocketeer Logo" width="200"/>
</div>

**A lightweight, fast pocket finder in Python**

Pocketeer detects binding pockets and cavities in protein structures using the alpha-sphere method based on Delaunay tessellation, similar to the popular fpocket, but more modern. It's lightweight, fast, and designed to be easy to use both as a Python library and command-line tool.

> 🚧 **Warning:** This software is in alpha. Some features (e.g., similarity calculation) are not yet implemented and there may be bugs. If you encounter any issues, please open an issue on the repository. 🚧

## Key Features

- **Modern Python implementation** - installable with `pip` or `uv`, works on Apple Silicon
- **Flexible Python API** - built on `biotite` atom arrays
- **Command-line interface** - simple CLI for batch processing
- **JIT-compiled performance** - `numba` acceleration for volume calculations

## Quick Start

### Installation

```bash
pip install pocketeer
```

or 

```bash
uv add pocketeer
```

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

### Command-Line Interface

```bash
# Basic usage
pocketeer protein.pdb # also works with .cif files 

# Custom output directory
pocketeer protein.pdb --o results/

# Adjust parameters
pocketeer protein.pdb --r-min 2.5 --r-max 7.0 --min-spheres 25
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

## What's Next?

- **[Getting Started](getting-started.md)** - Complete tutorial and workflow
- **[API Reference](api-reference.md)** - Full API documentation
- **[CLI Guide](cli-guide.md)** - Command-line interface usage
- **[Algorithm](algorithm.md)** - Technical details about the alpha-sphere method
- **[Visualization Guide](visualization-guide.md)** - Notebook visualization examples

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
4. Run `ruff check --fix && ruff format` before committing
5. Submit a pull request

### Development installation

```bash
git clone https://github.com/cch1999/pocketeer.git
cd pocketeer
pip install -e ".[dev]"
```

## Support

For bugs and feature requests, please open an issue on GitHub.


