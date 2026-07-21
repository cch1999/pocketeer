"""Train and evaluate the dependency-free pocket-ranking model."""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

np.seterr(all="ignore")  # spurious BLAS matmul warnings on some platforms

CACHE = Path(__file__).parent / ".cache"
META = {"struct", "det_rank", "dca", "label", "heuristic_score", "n_ligands"}
COMPACT6 = [
    "n_residues",
    "n_spheres",
    "vol_per_sphere",
    "radius_mean",
    "sasa_mean",
    "frac_aromatic",
]
_MINR, _MAXR = 3.5, 5.0


def load(path, keep=None):
    """Load a feature cache into rows, a matrix, labels, and feature names."""
    with path.open("rb") as file:
        rows = pickle.load(file)
    keys = list(keep) if keep else sorted(key for key in rows[0] if key not in META)
    features = np.nan_to_num(np.array([[row[key] for key in keys] for row in rows], float))
    labels = np.array([row["label"] for row in rows], float)
    return rows, features, labels, keys


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def train_lr(features, labels, l2=1.0, iterations=4000, learning_rate=0.5):
    """Fit class-balanced L2 logistic regression via gradient descent."""
    mean, std = features.mean(0), features.std(0) + 1e-9
    design = np.hstack([(features - mean) / std, np.ones((len(features), 1))])
    weights = np.zeros(design.shape[1])
    n = len(features)
    sample_weights = np.where(labels == 1, n / (2 * labels.sum()), n / (2 * (n - labels.sum())))
    for _ in range(iterations):
        gradient = design.T @ (sample_weights * (_sigmoid(design @ weights) - labels)) / n
        gradient[:-1] += l2 * weights[:-1] / n
        weights -= learning_rate * gradient
    return weights, mean, std


def predict(weights, mean, std, features):
    design = np.hstack([(features - mean) / std, np.ones((len(features), 1))])
    return _sigmoid(design @ weights)


def auc(labels, scores):
    """Compute ROC-AUC from ranks."""
    scores = np.asarray(scores, float)
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_positive = labels.sum()
    n_negative = len(labels) - n_positive
    return (ranks[labels == 1].sum() - n_positive * (n_positive + 1) / 2) / (
        n_positive * n_negative
    )


def rank_metrics(rows, scores):
    """Compute per-structure Top-1, Top-3, and Top-(n+2) success."""
    grouped = defaultdict(list)
    for row, score in zip(rows, scores, strict=False):
        grouped[row["struct"]].append((score, row["label"], row["n_ligands"]))
    top1 = top3 = topn = n = 0
    for items in grouped.values():
        if not any(label for _, label, _ in items):
            continue
        n += 1
        labels = [label for _, label, _ in sorted(items, key=lambda item: -item[0])]
        top1 += labels[0] == 1
        top3 += 1 in labels[:3]
        topn += 1 in labels[: items[0][2] + 2]
    return top1, top3, topn, n


def baseline_orig(row):
    band = 2 if _MINR <= row["radius_mean"] <= _MAXR else 0
    return row["volume"] / 500 + row["n_spheres"] / 50 + band


def main() -> None:
    train_path = CACHE / "chen11.features.pkl"
    test_path = CACHE / "coach420.features.pkl"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit("Missing feature caches. Run fetch_datasets.py then featurize.py.")

    train_rows, train_x, train_y, _ = load(train_path, keep=COMPACT6)
    test_rows, test_x, test_y, _ = load(test_path, keep=COMPACT6)
    print(
        f"train {len(train_rows)} pockets ({int(train_y.sum())} pos) | "
        f"test {len(test_rows)} pockets ({int(test_y.sum())} pos)\n"
    )
    weights, mean, std = train_lr(train_x, train_y)
    print(f"{'ranker':<22}{'AUC':>7}{'Top-1':>13}{'Top-3':>13}{'Top-(n+2)':>13}")

    def report(name, rows, labels, scores):
        top1, top3, topn, n = rank_metrics(rows, np.asarray(scores, float))
        print(
            f"{name:<22}{auc(labels, scores):>7.3f}{top1:>5}/{n}({100 * top1 / n:>3.0f}%)"
            f"{top3:>5}/{n}({100 * top3 / n:>3.0f}%)"
            f"{topn:>5}/{n}({100 * topn / n:>3.0f}%)"
        )

    report("LogReg (re-trained)", test_rows, test_y, predict(weights, mean, std, test_x))
    report("package score", test_rows, test_y, [row["heuristic_score"] for row in test_rows])
    report("baseline-orig", test_rows, test_y, [baseline_orig(row) for row in test_rows])
    report("pure-volume", test_rows, test_y, [row["volume"] for row in test_rows])

    try:
        from sklearn.ensemble import (  # noqa: PLC0415 - optional comparison dependency
            GradientBoostingClassifier,
            RandomForestClassifier,
        )

        forest = RandomForestClassifier(
            n_estimators=400,
            max_depth=7,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=0,
            n_jobs=-1,
        ).fit(train_x, train_y)
        report("RandomForest", test_rows, test_y, forest.predict_proba(test_x)[:, 1])
        boosting = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            random_state=0,
        ).fit(train_x, train_y)
        report("GradientBoosting", test_rows, test_y, boosting.predict_proba(test_x)[:, 1])
    except ImportError:
        print("(install scikit-learn for the RandomForest/GradientBoosting comparison)")

    print("\n# ---- paste into pocketeer/utils/constants.py ----")
    print(f"# Feature order: {tuple(COMPACT6)}")
    print(f"SCORE_LR_MEAN = {tuple(round(float(value), 5) for value in mean)}")
    print(f"SCORE_LR_STD = {tuple(round(float(value), 5) for value in std)}")
    print(f"SCORE_LR_COEF = {tuple(round(float(value), 5) for value in weights[:-1])}")
    print(f"SCORE_LR_BIAS = {round(float(weights[-1]), 5)}")


if __name__ == "__main__":
    main()
