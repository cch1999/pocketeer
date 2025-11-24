# Getting Started

This guide will walk you through installing Pocketeer and running your first pocket detection analysis.

## Installation

```bash
pip install pocketeer
```

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

## Quick Tips

Before you start:
- Pocketeer automatically removes hydrogens from your structure, and is designed to work with them absent, so you don't need to worry about them.
- By default, Pocketeer removes all waters and hetero atoms from your input structure. Set `ignore_hetero` to `False` if you don't want this behavior (e.g. to take into account a metal ion in the pocket).

## Basic Workflow

The typical Pocketeer workflow consists of four main steps:

1. **Load structure** - Read a PDB file
2. **Configure parameters** - Set detection parameters (optional)
3. **Detect pockets** - Run the analysis
4. **Interpret results** - Analyze and save results

### Step 1: Load a Structure

```python
import pocketeer as pt

# Load a protein structure from PDB file
atomarray = pt.load_structure("protein.pdb")
print(f"Loaded {len(atomarray)} atoms")
```

!!! tip "Supported Formats"
    Pocketeer uses Biotite for structure loading, which supports PDB, mmCIF, and other common formats.

### Step 2: Detect Pockets

```python
# Run pocket detection with default parameters
pockets = pt.find_pockets(atomarray)

# Or customize parameters
pockets = pt.find_pockets(
    atomarray,
    r_min=3.0,           # Minimum sphere radius (Å)
    r_max=6.0,           # Maximum sphere radius (Å)
    min_spheres=30,      # Minimum spheres per pocket cluster
    merge_distance=2.5,  # Distance threshold for clustering
    sasa_threshold=25.0,  # SASA threshold for buried spheres (Å²)
)

print(f"Found {len(pockets)} pockets")
```

### Step 3: Interpret Results

```python
# Print pocket information
for pocket in pockets:
    print(f"Pocket {pocket.pocket_id}:")
    print(f"  Score: {pocket.score:.2f}")
    print(f"  Volume: {pocket.volume:.1f} Å³")
    print(f"  Spheres: {pocket.n_spheres}")
```

## Understanding Parameters

### Radius Parameters

- **`r_min`** (default: 3.0 Å): Minimum alpha-sphere radius
  - Smaller values detect smaller cavities
  - Too small may detect noise
  
- **`r_max`** (default: 6.0 Å): Maximum alpha-sphere radius
  - Larger values detect larger cavities
  - Too large may miss specific binding sites

### Clustering Parameters

- **`min_spheres`** (default: 35): Minimum spheres per pocket cluster
  - Higher values require more evidence for a pocket
  - Lower values may detect smaller or less well-defined pockets

- **`merge_distance`** (default: 1.75 Å): Distance threshold for merging nearby spheres
  - Smaller values create more fragmented pockets
  - Larger values merge nearby pockets together

### Polarity Parameters

- **`polar_probe_radius`** (default: 1.4 Å): Probe radius for SASA calculation
  - Used to compute solvent accessible surface area (SASA) of atoms
  - Should be similar to water molecule radius (~1.4 Å)
  - Lower values give more detailed surface calculations

- **`sasa_threshold`** (default: 20.0 Å²): Threshold for mean SASA to determine if a sphere is buried
  - Spheres with mean SASA below this threshold are considered buried (interior) and kept
  - Higher values include more surface-exposed spheres in pocket detection
  - Lower values filter more aggressively, keeping only deeply buried spheres
  - Typical range: 15-30 Å²

## Saving Results

### Save to Files

```python
# Save all pockets to JSON
pt.write_pockets_json("pockets.json", pockets)

# Save individual pocket files
pt.write_individual_pocket_jsons("output/", pockets)

# Save alpha-spheres as PDB for visualization
pt.write_pockets_as_pdb("alphaspheres.pdb", pockets)

# Save human-readable summary
pt.write_summary("summary.txt", pockets, "protein.pdb")
```

### Visualize Results

```python
# If atomworks is installed
try:
    pt.view_pockets(atomarray, pockets)
except ImportError:
    print("Install atomworks for visualization: pip install pocketeer[vis]")
```

## Common Issues

### No Pockets Detected

If no pockets are found, try:

1. **Lower `min_spheres`** - Require fewer spheres per pocket
2. **Adjust radius range** - Try different `r_min`/`r_max` values
3. **Increase `merge_distance`** - Merge nearby spheres more aggressively
4. **Check input structure** - Ensure it's a complete protein structure

### Too Many Small Pockets

If you get many small, irrelevant pockets:

1. **Increase `min_spheres`** - Require more evidence per pocket
2. **Increase `r_min`** - Filter out very small cavities
3. **Decrease `merge_distance`** - Keep pockets more separate

### Performance Issues

For large structures (>5000 atoms):

1. **Use default parameters** - They're optimized for typical proteins
2. **Consider filtering atoms** - Remove waters, ions if not needed
3. **Run on multiple cores** - Pocketeer uses NumPy/SciPy which can utilize multiple cores

## Next Steps

- **[API Reference](api-reference.md)** - Complete function documentation
- **[CLI Guide](cli-guide.md)** - Command-line usage
- **[Visualization Guide](visualization-guide.md)** - Notebook visualization examples
- **[Algorithm](algorithm.md)** - Technical details

