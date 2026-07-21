# Pocket-ranking model

This directory trains and evaluates the logistic-regression ligandability model
used by `pocketeer.core.scoring.score_pocket`. Its coefficients are baked into
`pocketeer.utils.constants`; no ML dependency or model file is needed at runtime.

## Data

Structures come from [rdk/p2rank-datasets](https://github.com/rdk/p2rank-datasets):

- **chen11** (252 proteins): training set
- **coach420** (420 proteins): evaluation set, disjoint from chen11

A pocket is positive if it lies within 4 A (DCA) of any relevant co-crystallised
ligand. Cofactors are kept; water, ions, buffers, sugars, and detergents are dropped.

## Reproduce

```bash
uv run python benchmarks/ml/fetch_datasets.py
uv run python benchmarks/ml/featurize.py chen11
uv run python benchmarks/ml/featurize.py coach420
uv run python benchmarks/ml/train_ranker.py
```

The last command prints a `SCORE_LR_*` block ready to paste into
`src/pocketeer/utils/constants.py`. It uses only NumPy. If scikit-learn happens
to be installed, it also reports optional RandomForest and GradientBoosting
comparisons; scikit-learn is not a project dependency.

To explore detector clustering parameters and retrain the ranker for every
configuration:

```bash
uv run python benchmarks/ml/grid_search_detector.py
```

The grid search reports detection recall, decoys per structure, ROC-AUC, and
conditional/end-to-end Top-1 and Top-3. Structures rejected by the current SASA
atom annotation are counted and excluded rather than silently treated as
detection misses.

## Model evaluation

The model is fitted only on chen11. Detector clustering defaults were selected
using coach420, however, so these figures are engineering evaluation results and
must not be presented as performance on an untouched held-out set.

The six features are lining-residue count, alpha-sphere count, volume per sphere,
mean sphere radius, mean sphere SASA, and aromatic-residue fraction.

| Ranker | ROC-AUC | Top-1 | Top-3 |
| --- | --- | --- | --- |
| **Logistic regression (shipped)** | **0.85** | **87%** | **97%** |
| original size-based heuristic | 0.78 | 72% | 91% |
| pure volume | 0.76 | 66% | 88% |

With the default detector (`merge_distance=2.75`, `min_spheres=40`), 388 of
405 eligible coach420 structures have a detected site. Counting detection misses
as failures, the shipped model reaches 83% Top-1 and 93% Top-3 end to end.

The logistic regression is shipped because Top-1 ranking is the primary goal,
and because the compact model is interpretable and dependency-free.
