"""Download P2Rank benchmark datasets (chen11 train, coach420 evaluation).

Structures come from the public P2Rank dataset repository:
    https://github.com/rdk/p2rank-datasets

Files are cached under ``benchmarks/ml/.cache/<dataset>/`` (gitignored).

Usage:
    python benchmarks/ml/fetch_datasets.py            # both datasets
    python benchmarks/ml/fetch_datasets.py chen11     # one dataset
"""

from __future__ import annotations

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CACHE = Path(__file__).parent / ".cache"
API = "https://api.github.com/repos/rdk/p2rank-datasets/contents/{}?per_page=1000"
DATASETS = ("chen11", "coach420")


def list_pdbs(dataset: str) -> list[tuple[str, str]]:
    """Return (filename, download_url) for every .pdb in a dataset directory."""
    with urllib.request.urlopen(API.format(dataset)) as response:
        items = json.load(response)
    return [(item["name"], item["download_url"]) for item in items if item["name"].endswith(".pdb")]


def fetch_dataset(dataset: str) -> None:
    out = CACHE / dataset
    out.mkdir(parents=True, exist_ok=True)
    pdbs = list_pdbs(dataset)
    print(f"{dataset}: {len(pdbs)} structures")

    def grab(item: tuple[str, str]) -> None:
        name, url = item
        dest = out / name
        if dest.exists() and dest.stat().st_size > 0:
            return
        try:
            urllib.request.urlretrieve(url, dest)  # trusted GitHub host
        except Exception as error:  # noqa: BLE001 - report individual download failures
            print(f"  failed {name}: {error}")

    with ThreadPoolExecutor(max_workers=12) as executor:
        list(executor.map(grab, pdbs))
    have = len(list(out.glob("*.pdb")))
    print(f"{dataset}: {have}/{len(pdbs)} downloaded -> {out}")


def main() -> None:
    targets = sys.argv[1:] or list(DATASETS)
    for dataset in targets:
        fetch_dataset(dataset)


if __name__ == "__main__":
    main()
