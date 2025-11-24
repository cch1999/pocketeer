# API Reference

This page documents all public functions and classes in Pocketeer. There are really only 


| Function                                     | Description                                                          |
|-----------------------------------------------|----------------------------------------------------------------------|
| `load_structure()`                    | Loads a structure from a PDB file as a Biotite AtomArray.            |
| `find_pockets()`   | Detects binding pockets in protein structures.                       |
| `view_pockets()`                        | Visualizes pockets and protein structures in a 3D viewer (e.g., notebook).                |



## Core Functions

### `find_pockets()`

Main function for detecting binding pockets in protein structures.

```python
pocketeer.find_pockets(
    atomarray: biotite.structure.AtomArray,
    *,
    r_min: float = 3.0,
    r_max: float = 6.0,
    polar_probe_radius: float = 1.4,
    sasa_threshold: float = 20.0,
    merge_distance: float = 1.75,
    min_spheres: int = 35,
    ignore_hydrogens: bool = True,
    ignore_water: bool = True,
    ignore_hetero: bool = True
) -> list[Pocket]
```

**Parameters:**

- **`atomarray`**: Biotite AtomArray with protein structure data
- **`r_min`**: Minimum alpha-sphere radius (Å). Default: 3.0
- **`r_max`**: Maximum alpha-sphere radius (Å). Default: 6.0
- **`polar_probe_radius`**: Probe radius for SASA calculation (Å). Default: 1.4
- **`sasa_threshold`**: Threshold for SASA value to determine if a sphere is buried (Å²). Spheres with mean SASA below this threshold are considered buried and kept for pocket detection. Default: 20.0
- **`merge_distance`**: Distance threshold for merging nearby sphere clusters (Å). Default: 1.75
- **`min_spheres`**: Minimum number of spheres per pocket cluster. Default: 35
- **`ignore_hydrogens`**: Remove hydrogen atoms (recommended: True)
- **`ignore_water`**: Remove water molecules (recommended: True)  
- **`ignore_hetero`**: Remove hetero atoms/ligands (recommended: True)

**Returns:**

List of `Pocket` objects, sorted by score (highest first).

**Example:**

```python
import pocketeer as pt

# Load structure
atomarray = pt.load_structure("protein.pdb")

# Detect pockets with default parameters
pockets = pt.find_pockets(atomarray)

# With custom parameters
pockets = pt.find_pockets(atomarray, r_min=2.5, r_max=7.0, min_spheres=25)
```

### `load_structure()`

Load a protein structure from various file formats.

```python
pocketeer.load_structure(structure_path: str, model: int = 1) -> biotite.structure.AtomArray
```

**Parameters:**

- **`structure_path`**: Path to structure file (PDB, mmCIF/CIF, BinaryCIF, MOL, SDF)
- **`model`**: Model number to load (default: 1)

**Returns:**

Biotite AtomArray with structure data.

**Example:**

```python
atomarray = pocketeer.load_structure("protein.pdb")
print(f"Loaded {len(atomarray)} atoms")
```

## Parameter Reference

All parameters for `find_pockets()` can be passed directly as keyword arguments:

- **`r_min`** (default: 3.0 Å): Minimum alpha-sphere radius
- **`r_max`** (default: 6.0 Å): Maximum alpha-sphere radius  
- **`polar_probe_radius`** (default: 1.4 Å): Probe radius for SASA calculation. Used to compute solvent accessible surface area of atoms defining each sphere
- **`sasa_threshold`** (default: 20.0 Å²): Threshold for mean SASA value to determine if a sphere is buried. Spheres with mean SASA below this threshold are considered buried (interior) and kept for pocket detection. Higher values include more surface-exposed spheres
- **`merge_distance`** (default: 1.75 Å): Distance threshold for merging nearby sphere clusters
- **`min_spheres`** (default: 35): Minimum number of spheres per pocket cluster

## Visualization (Optional)

### `view_pockets()`

Visualize protein structure with detected pockets in a Jupyter notebook or other supported viewer.
This function is a wrapper around [atomworks.io.utils.visualize.view](https://baker-laboratory.github.io/atomworks-dev/latest/io/utils/visualize.html) and shows pockets as colored spheres.

```python
pocketeer.view_pockets(
    atomarray,                     # biotite.structure.AtomArray with protein structure data
    pockets,                       # list of Pocket objects from find_pockets()
    color_scheme="rainbow",        # 'rainbow', 'grayscale', or 'red_blue'
    sphere_opacity=0.7,            # Opacity of pocket spheres (float, 0.0–1.0)
    sphere_scale=1.0,              # Size scaling of spheres
    receptor_cartoon=True,         # Show main structure as cartoon
    receptor_surface=False,        # Show surface of protein
    **kwargs                       # Extra arguments passed to atomworks "view"
)
# Returns: atomworks viewer instance (e.g. py3Dmol)
```

**Arguments:**

| Name                | Type                             | Default         | Description                                                             |
|---------------------|----------------------------------|-----------------|-------------------------------------------------------------------------|
| `atomarray`         | `biotite.structure.AtomArray`    | *required*      | Protein structure to display                                            |
| `pockets`           | `list[Pocket]`                   | *required*      | List of Pocket objects (from `find_pockets`)                            |
| `color_scheme`      | `str`                            | `"rainbow"`     | How to color pockets: `"rainbow"`, `"grayscale"`, `"red_blue"`          |
| `sphere_opacity`    | `float`                          | `0.7`           | Opacity of pocket spheres (0=transparent, 1=opaque)                      |
| `sphere_scale`      | `float`                          | `1.0`           | Scale factor for alpha-sphere display                                   |
| `receptor_cartoon`  | `bool`                           | `True`          | Show main structure as cartoon                                          |
| `receptor_surface`  | `bool`                           | `False`         | Show protein surface                                                    |
| `**kwargs`          | any                              | N/A             | Additional arguments passed to atomworks `view()`                       |

**Returns:**

- **atomworks viewer instance** (e.g. based on py3Dmol) for interactive use

**Raises:**

- `ImportError` if atomworks is not installed
- `TypeError` if atomarray is not a Biotite AtomArray
- `ValueError` if no pockets given or unknown color scheme

**Example:**

```python
import pocketeer as pt

atomarray = pt.load_structure("protein.pdb")
pockets = pt.find_pockets(atomarray)

# Standard visualization
viewer = pt.view_pockets(atomarray, pockets)
viewer.show()
```

**Note:** Requires the `atomworks` package. Install with `pip install pocketeer[vis]`.

