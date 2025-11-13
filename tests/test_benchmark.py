"""Benchmark tests for pocketeer performance."""

import subprocess
import tempfile
import time
from pathlib import Path

import pytest

# PDB files are in the project root
DATA_ROOT = Path(__file__).parent / "data"
PROJECT_ROOT = Path(__file__).parent.parent
TEST_FILES = [
    "1wcc.pdb",
    "2xjx.pdb",
    "4hhb.pdb",
    "4jnc.pdb",
    "4tos_receptor.pdb",
    "4tos.cif",  # mmCIF file
    "5o3l.pdb",
    "6qrd.pdb",
    "7gaw.pdb",
    "7m3z.pdb",
    "8azr.pdb",
]


@pytest.mark.parametrize("pdb_file", TEST_FILES)
def test_main_benchmark(pdb_file):
    """Benchmark the main program on test PDB files."""
    pdb_path = DATA_ROOT / pdb_file

    if not pdb_path.exists():
        pytest.skip(f"PDB file not found: {pdb_path}")

    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Run the main program as a subprocess and time it
        start = time.perf_counter()
        result = subprocess.run(
            ["uv", "run", "pocketeer", str(pdb_path), "--out-dir", str(tmp_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        elapsed = time.perf_counter() - start

        print(f"\nBenchmark for {pdb_file}: {elapsed:.3f} seconds")
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

        # Check that the program ran successfully
        assert result.returncode == 0, f"Program failed for {pdb_file}"

        # Verify output files were created
        assert (tmp_path / "pockets.json").exists()
        assert (tmp_path / "summary.txt").exists()
