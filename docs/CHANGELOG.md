# Changelog

All notable changes to pocketeer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-XX

### Added
- Residue information in `Pocket` dataclass (`residues` field with chain_id, res_id, res_name)
- Boolean mask (`mask` field) for selecting atoms from original AtomArray
- Residue count displayed in CLI output

### Changed
- `create_pocket()` now requires `atomarray` parameter (no longer optional)

## [0.1.0] - 2025-11-24

### Added
- `sasa_threshold` parameter to `find_pockets()` for filtering buried vs surface-exposed spheres
- SASA (Solvent Accessible Surface Area) calculation using `polar_probe_radius` to distinguish interior pockets from surface features
- Enhanced documentation for SASA threshold parameter across all documentation files

### Changed
- Default `polar_probe_radius` corrected to 1.4 Å (was incorrectly documented as 1.8 Å)
- Default `merge_distance` corrected to 1.75 Å (was incorrectly documented as 1.2 Å)

## [0.0.0] - 2025-11-13

### Added
- Initial release of pocketeer
- Core pocket detection using alpha-sphere method
- Delaunay tessellation-based geometry
- Graph-based clustering of alpha-spheres
- Volume estimation using voxel grid method with Numba JIT compilation
- Simple druggability scoring
- PDB I/O with simple parser and optional Biotite support
- CLI with Typer (built-in) and optional Rich for pretty output
- Comprehensive test suite
- Full API documentation in README

### Features
- `find_pockets()` - Main API function
- `AlphaSphere` and `Pocket` data types
- Multiple output formats (PDB, JSON, TXT)
- Clean, minimal dependencies (numpy, scipy, numba, typer)

### Performance
- Handles typical proteins (1000-5000 atoms) in 3-5 seconds
- Numba JIT compilation for volume calculation (fast and memory-efficient)
- Vectorized NumPy operations throughout
- First run slightly slower due to JIT compilation, subsequent runs are fast

[0.1.0]: https://github.com/cch1999/pocketeer/releases/tag/v0.1.0


## 0.0.3

### Fixes
- Fixed `view_pockets` import error