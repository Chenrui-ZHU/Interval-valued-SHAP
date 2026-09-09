from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from interval_attribution import (
    _prepare_rf_instance_payloads,
    _rf_global_greedy_signed_endpoint_bounds_for_instance,
    _stable_order_from_scores,
    abs_bounds_from_signed_bounds,
    extract_sklearn_tree,
    features_used_in_tree,
    leaf_class_counts,
)


N_SAMPLES = 120
N_REPEATS = 5
N_TREES = 40
EVAL_SIZE = 20
S = 2
POWER_R_VALUES = (0.00, 0.10, 0.20, 0.30)

EXP3_N_SAMPLES = 2000
EXP3_N_FEATURES = 50
EXP3_RELEVANT_FEATURES = 5
EXP3_RELEVANT_POOL = 10
EXP3_N_REPEATS = 5
EXP3_N_TREES = 60
EXP3_EVAL_SIZE = 80
EXP3_MAX_LEAF_NODES = 8

FEATURE_NAMES = ("X1_cont", "X2_binary_signal", "X3_cat4", "X4_cat10", "X5_cat20")
CARDINALITIES = ("continuous", "2", "4", "10", "20")
PLOT_FEATURE_LABELS = ("X1\ncont", "X2\n2", "X3\n4", "X4\n10", "X5\n20")


@dataclass(frozen=True)
class ImportanceScores:
    nominal: np.ndarray
    pessimistic: np.ndarray
    oob: np.ndarray


@dataclass(frozen=True)
class LocalImportanceScores:
    nominal: np.ndarray
    pessimistic: np.ndarray
    oob: np.ndarray


@dataclass(frozen=True)
class IntervalImportanceScores:
    nominal_absolute: np.ndarray
    absolute_lower: np.ndarray
    absolute_upper: np.ndarray


@dataclass(frozen=True)
class DebiasConfig:
    random_state: int = 20260526
    n_samples: int = N_SAMPLES
    n_repeats: int = N_REPEATS
    n_trees: int = N_TREES
    eval_size: int = EVAL_SIZE
    s: int = S
    power_r_values: tuple[float, ...] = POWER_R_VALUES
    auc_n_samples: int = EXP3_N_SAMPLES
    auc_n_features: int = EXP3_N_FEATURES
    auc_n_relevant: int = EXP3_RELEVANT_FEATURES
    auc_relevant_pool: int = EXP3_RELEVANT_POOL
    auc_n_repeats: int = EXP3_N_REPEATS
    auc_n_trees: int = EXP3_N_TREES
    auc_eval_size: int = EXP3_EVAL_SIZE
    auc_max_leaf_nodes: int = EXP3_MAX_LEAF_NODES
    results_dir: str = "results_submission/debias"
    figures_dir: str = "figures/debias"


# ----------------------------
# Synthetic data
# ----------------------------
def generate_loecher_features(
    *,
    rng: np.random.Generator,
    n_samples: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "X1_cont": rng.normal(size=n_samples),
            "X2_binary_signal": rng.integers(1, 3, size=n_samples),
            "X3_cat4": rng.integers(1, 5, size=n_samples),
            "X4_cat10": rng.integers(1, 11, size=n_samples),
            "X5_cat20": rng.integers(1, 21, size=n_samples),
        }
    )


def loecher_probabilities(X: pd.DataFrame, r: float | None) -> np.ndarray:
    if r is None:
        return np.full(len(X), 0.5, dtype=float)
    x2 = X["X2_binary_signal"].to_numpy(dtype=int)
    return np.where(x2 == 1, 0.5 - float(r), 0.5 + float(r))


def generate_loecher_data(
    *,
    rng: np.random.Generator,
    n_samples: int,
    r: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    X = generate_loecher_features(rng=rng, n_samples=n_samples)
    y = rng.binomial(1, loecher_probabilities(X, r)).astype(int)
    return X, y


def generate_loecher_many_feature_data(
    *,
    rng: np.random.Generator,
    n_samples: int,
    n_features: int,
    n_relevant: int,
    relevant_pool: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    columns: dict[str, np.ndarray] = {}
    for feature_id in range(int(n_features)):
        columns[f"X{feature_id + 1:02d}_k{feature_id + 2}"] = rng.integers(
            0,
            feature_id + 2,
            size=int(n_samples),
        )
    X = pd.DataFrame(columns)

    relevant = np.sort(rng.choice(int(relevant_pool), size=int(n_relevant), replace=False))
    labels = np.zeros(int(n_features), dtype=int)
    labels[relevant] = 1

    denominators = relevant.astype(float) + 1.0
    signal_terms = X.iloc[:, relevant].to_numpy(dtype=float) / denominators[None, :]
    logits = (2.0 / 5.0) * signal_terms.sum(axis=1) - 1.0
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probabilities).astype(int)
    return X, y, labels


def fit_forest(
    X: pd.DataFrame,
    y: np.ndarray,
    random_state: int,
    *,
    n_estimators: int,
    max_features: int | str | None,
    max_leaf_nodes: int | None = None,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1,
    ).fit(X, y)


# ----------------------------
# Forest attribution scores
# ----------------------------
def prepare_indexed_forest_tree_payloads(forest: RandomForestClassifier) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for tree_index, dt in enumerate(forest.estimators_):
        tree_struct = extract_sklearn_tree(dt)
        feature_ids = features_used_in_tree(tree_struct)
        if not feature_ids:
            continue
        payloads.append(
            {
                "tree_index": tree_index,
                "tree": tree_struct,
                "counts": leaf_class_counts(dt, tree_struct),
                "U": list(feature_ids),
            }
        )
    return payloads


def forest_inbag_masks(forest: RandomForestClassifier, n_samples: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for samples in forest.estimators_samples_:
        mask = np.zeros(int(n_samples), dtype=bool)
        mask[np.asarray(samples, dtype=int)] = True
        masks.append(mask)
    return masks


def no_intercept_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    numerator = np.nansum(x * y, axis=0)
    denominator = np.nansum(x * x, axis=0)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 1e-12,
    )


def rf_signed_shap_matrix(
    forest: RandomForestClassifier,
    X_bg: pd.DataFrame,
    X_eval: pd.DataFrame,
) -> np.ndarray:
    n_features = X_bg.shape[1]
    tree_payloads = prepare_indexed_forest_tree_payloads(forest)
    rows: list[np.ndarray] = []
    for _, x_star in X_eval.iterrows():
        instance_payloads = _prepare_rf_instance_payloads(
            tree_payloads,
            x_star=x_star,
            X_bg=X_bg,
            kind="shapley",
            semantics="path_dependent",
        )
        nominal = np.zeros(n_features, dtype=float)
        for payload in instance_payloads:
            feature_ids = np.asarray(payload["U"], dtype=int)
            nominal[feature_ids] += np.asarray(payload["nominal"], dtype=float)
        if instance_payloads:
            nominal /= len(instance_payloads)
        rows.append(nominal)
    return np.vstack(rows)


def rf_train_test_shrunk_scores(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    random_state: int,
    n_trees: int,
    max_features: int | str | None,
    max_leaf_nodes: int | None,
    eval_size: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(random_state))
    indices = rng.permutation(len(X))
    split_at = max(1, min(len(X) - 1, len(X) // 2))
    left_idx = indices[:split_at]
    right_idx = indices[split_at:]

    X_left = X.iloc[left_idx].reset_index(drop=True)
    y_left = np.asarray(y, dtype=int)[left_idx]
    X_right = X.iloc[right_idx].reset_index(drop=True)
    y_right = np.asarray(y, dtype=int)[right_idx]

    forest_left = fit_forest(
        X_left,
        y_left,
        random_state=int(random_state) + 1,
        n_estimators=n_trees,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
    )
    forest_right = fit_forest(
        X_right,
        y_right,
        random_state=int(random_state) + 2,
        n_estimators=n_trees,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
    )

    n_left_eval = min(len(X_left), max(1, int(eval_size) // 2))
    n_right_eval = min(len(X_right), max(1, int(eval_size) - n_left_eval))
    X_left_eval = X_left.iloc[:n_left_eval]
    X_right_eval = X_right.iloc[:n_right_eval]

    shap_in = np.vstack(
        [
            rf_signed_shap_matrix(forest_left, X_left, X_left_eval),
            rf_signed_shap_matrix(forest_right, X_right, X_right_eval),
        ]
    )
    shap_out = np.vstack(
        [
            rf_signed_shap_matrix(forest_right, X_right, X_left_eval),
            rf_signed_shap_matrix(forest_left, X_left, X_right_eval),
        ]
    )
    beta = no_intercept_slopes(shap_in, shap_out)
    return np.nanmean(np.abs(shap_in * beta[None, :]), axis=0)


def rf_local_importance_scores(
    forest: RandomForestClassifier,
    X: pd.DataFrame,
    *,
    s: int,
    eval_size: int,
) -> LocalImportanceScores:
    n_features = X.shape[1]
    tree_payloads = prepare_indexed_forest_tree_payloads(forest)
    inbag_masks = forest_inbag_masks(forest, n_samples=len(X))
    X_eval = X.iloc[: min(int(eval_size), len(X))]

    nominal_rows: list[np.ndarray] = []
    pessimistic_rows: list[np.ndarray] = []
    shap_inbag_rows: list[np.ndarray] = []
    shap_oob_rows: list[np.ndarray] = []

    for row_position, (_, x_star) in enumerate(X_eval.iterrows()):
        instance_payloads = _prepare_rf_instance_payloads(
            tree_payloads,
            x_star=x_star,
            X_bg=X,
            kind="shapley",
            semantics="path_dependent",
        )

        nominal = np.zeros(n_features, dtype=float)
        shap_inbag = np.zeros(n_features, dtype=float)
        shap_oob = np.zeros(n_features, dtype=float)
        n_inbag = 0
        n_oob = 0

        for tree_payload, payload in zip(tree_payloads, instance_payloads):
            feature_ids = np.asarray(payload["U"], dtype=int)
            tree_values = np.zeros(n_features, dtype=float)
            tree_values[feature_ids] = np.asarray(payload["nominal"], dtype=float)
            nominal += tree_values

            tree_index = int(tree_payload["tree_index"])
            if inbag_masks[tree_index][row_position]:
                shap_inbag += tree_values
                n_inbag += 1
            else:
                shap_oob += tree_values
                n_oob += 1

        if instance_payloads:
            nominal /= len(instance_payloads)
        shap_inbag = shap_inbag / n_inbag if n_inbag else np.full(n_features, np.nan)
        shap_oob = shap_oob / n_oob if n_oob else np.full(n_features, np.nan)

        signed_lower, signed_upper = _rf_global_greedy_signed_endpoint_bounds_for_instance(
            instance_payloads,
            n_features=n_features,
            s_values=[int(s)],
        )[int(s)]
        absolute_lower, _ = abs_bounds_from_signed_bounds(signed_lower, signed_upper)

        nominal_rows.append(np.abs(nominal))
        pessimistic_rows.append(absolute_lower)
        shap_inbag_rows.append(shap_inbag)
        shap_oob_rows.append(shap_oob)

    shap_inbag_matrix = np.vstack(shap_inbag_rows)
    shap_oob_matrix = np.vstack(shap_oob_rows)
    beta = no_intercept_slopes(shap_inbag_matrix, shap_oob_matrix)
    oob_rows = np.abs(shap_inbag_matrix * beta[None, :])
    return LocalImportanceScores(
        nominal=np.vstack(nominal_rows),
        pessimistic=np.vstack(pessimistic_rows),
        oob=oob_rows,
    )


def rf_nominal_and_pessimistic_scores(
    forest: RandomForestClassifier,
    X: pd.DataFrame,
    *,
    s: int,
    eval_size: int,
) -> ImportanceScores:
    local = rf_local_importance_scores(forest, X, s=s, eval_size=eval_size)
    return ImportanceScores(
        nominal=np.mean(local.nominal, axis=0),
        pessimistic=np.mean(local.pessimistic, axis=0),
        oob=np.nanmean(local.oob, axis=0),
    )


def rf_nominal_and_interval_scores(
    forest: RandomForestClassifier,
    X: pd.DataFrame,
    *,
    s: int,
    eval_size: int,
) -> IntervalImportanceScores:
    n_features = X.shape[1]
    tree_payloads = prepare_indexed_forest_tree_payloads(forest)
    X_eval = X.iloc[: min(int(eval_size), len(X))]
    nominal_rows: list[np.ndarray] = []
    lower_rows: list[np.ndarray] = []
    upper_rows: list[np.ndarray] = []

    for _, x_star in X_eval.iterrows():
        instance_payloads = _prepare_rf_instance_payloads(
            tree_payloads,
            x_star=x_star,
            X_bg=X,
            kind="shapley",
            semantics="path_dependent",
        )
        nominal_signed = np.zeros(n_features, dtype=float)
        for payload in instance_payloads:
            feature_ids = np.asarray(payload["U"], dtype=int)
            nominal_signed[feature_ids] += np.asarray(payload["nominal"], dtype=float)
        if instance_payloads:
            nominal_signed /= len(instance_payloads)

        signed_lower, signed_upper = _rf_global_greedy_signed_endpoint_bounds_for_instance(
            instance_payloads,
            n_features=n_features,
            s_values=[int(s)],
        )[int(s)]
        absolute_lower, absolute_upper = abs_bounds_from_signed_bounds(signed_lower, signed_upper)

        nominal_rows.append(np.abs(nominal_signed))
        lower_rows.append(absolute_lower)
        upper_rows.append(absolute_upper)

    return IntervalImportanceScores(
        nominal_absolute=np.mean(np.vstack(nominal_rows), axis=0),
        absolute_lower=np.mean(np.vstack(lower_rows), axis=0),
        absolute_upper=np.mean(np.vstack(upper_rows), axis=0),
    )


# ----------------------------
# Experiments
# ----------------------------
def rank_of_x2(scores: np.ndarray) -> int:
    order = _stable_order_from_scores(np.asarray(scores, dtype=float), descending=True)
    return int(order.index(1) + 1)


def gap_for_x2(scores: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    noisy = np.delete(scores, 1)
    return float(scores[1] - noisy.max())


def binary_feature_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return np.nan
    greater = positive[:, None] > negative[None, :]
    ties = positive[:, None] == negative[None, :]
    return float(np.mean(greater + 0.5 * ties))


def summarize_power(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return (
        df.groupby(["r", "method"], as_index=False)
        .agg(
            mean_rank_x2=("rank_x2", "mean"),
            rank1_rate=("rank_x2", lambda values: float(np.mean(np.asarray(values) == 1))),
            mean_gap_x2=("gap_x2", "mean"),
            sd_gap_x2=("gap_x2", "std"),
        )
        .sort_values(["r", "method"])
    )


def summarize_auc(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return (
        df.groupby("method", as_index=False)
        .agg(mean_auc=("auc", "mean"), sd_auc=("auc", "std"))
        .sort_values("mean_auc", ascending=False)
    )


def run_power_boxplot_experiment(config: DebiasConfig, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    for r in config.power_r_values:
        for repeat in range(config.n_repeats):
            X, y = generate_loecher_data(rng=rng, n_samples=config.n_samples, r=float(r))
            forest = fit_forest(
                X,
                y,
                random_state=20_000 + int(100 * r) + repeat,
                n_estimators=config.n_trees,
                max_features=2,
            )
            scores = rf_nominal_and_pessimistic_scores(
                forest,
                X,
                s=config.s,
                eval_size=config.eval_size,
            )
            shrunk = rf_train_test_shrunk_scores(
                X,
                y,
                random_state=40_000 + int(100 * r) + repeat,
                n_trees=config.n_trees,
                max_features=2,
                max_leaf_nodes=None,
                eval_size=config.eval_size,
            )

            for method, values in (
                ("nominal", scores.nominal),
                ("pessimistic", scores.pessimistic),
                ("shrunk", shrunk),
                ("oob", scores.oob),
            ):
                metric_rows.append(
                    {
                        "r": float(r),
                        "repeat": repeat,
                        "method": method,
                        "rank_x2": rank_of_x2(values),
                        "gap_x2": gap_for_x2(values),
                    }
                )
                for feature_id, importance in enumerate(values):
                    importance_rows.append(
                        {
                            "r": float(r),
                            "repeat": repeat,
                            "method": method,
                            "feature": FEATURE_NAMES[feature_id],
                            "cardinality": CARDINALITIES[feature_id],
                            "importance": float(importance),
                            "scenario": f"r={float(r):.2f}",
                        }
                    )
    return summarize_power(metric_rows), pd.DataFrame(importance_rows)


def run_50_feature_auc_experiment(config: DebiasConfig, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for repeat in range(config.auc_n_repeats):
        X, y, labels = generate_loecher_many_feature_data(
            rng=rng,
            n_samples=config.auc_n_samples,
            n_features=config.auc_n_features,
            n_relevant=config.auc_n_relevant,
            relevant_pool=config.auc_relevant_pool,
        )
        forest = fit_forest(
            X,
            y,
            random_state=30_000 + repeat,
            n_estimators=config.auc_n_trees,
            max_features=3,
            max_leaf_nodes=config.auc_max_leaf_nodes,
        )
        scores = rf_nominal_and_pessimistic_scores(
            forest,
            X,
            s=config.s,
            eval_size=config.auc_eval_size,
        )
        interval_scores = rf_nominal_and_interval_scores(
            forest,
            X,
            s=config.s,
            eval_size=config.auc_eval_size,
        )
        shrunk = rf_train_test_shrunk_scores(
            X,
            y,
            random_state=50_000 + repeat,
            n_trees=config.auc_n_trees,
            max_features=3,
            max_leaf_nodes=config.auc_max_leaf_nodes,
            eval_size=config.auc_eval_size,
        )

        for method, values in (
            ("nominal", scores.nominal),
            ("pessimistic", scores.pessimistic),
            ("shrunk", shrunk),
            ("oob", scores.oob),
            ("safe", interval_scores.absolute_upper),
        ):
            rows.append(
                {
                    "repeat": repeat,
                    "method": method,
                    "auc": binary_feature_auc(values, labels),
                    "relevant_features": ",".join(f"X{idx + 1}" for idx in np.where(labels == 1)[0]),
                }
            )

    auc = pd.DataFrame(rows)
    return summarize_auc(rows), auc


# ----------------------------
# Plots and output
# ----------------------------
def plot_5_feature_boxplots(
    importance: pd.DataFrame,
    *,
    r_values: tuple[float, ...],
    output_path: str,
) -> Path:
    plot_r_values = [float(r) for r in r_values if not np.isclose(float(r), 0.10)]
    scenarios = [f"r={r:.2f}" for r in plot_r_values]
    methods = [
        ("nominal", r"$\mathbf{SHAP}$"),
        ("pessimistic", r"$\mathbf{SHAP}^{\mathbf{Pess}}$"),
        ("shrunk", r"$\mathbf{SHAP}^{\mathbf{Shrunk}}$"),
        ("oob", r"$\mathbf{SHAP}^{\mathbf{oob}}$"),
    ]

    fig, axes = plt.subplots(
        nrows=len(methods),
        ncols=len(scenarios),
        figsize=(10.5, 2.15 * len(methods)),
        sharey=True,
        squeeze=False,
    )
    y_max = float(importance["importance"].max())

    for row_idx, (method, method_label) in enumerate(methods):
        for col_idx, scenario in enumerate(scenarios):
            ax = axes[row_idx, col_idx]
            frame = importance[
                (importance["scenario"] == scenario)
                & (importance["method"] == method)
            ]
            data_values = [
                frame.loc[frame["feature"] == feature_name, "importance"].to_numpy()
                for feature_name in FEATURE_NAMES
            ]
            box = ax.boxplot(
                data_values,
                tick_labels=PLOT_FEATURE_LABELS,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.2},
                whiskerprops={"color": "#555555"},
                capprops={"color": "#555555"},
            )
            for feature_id, patch in enumerate(box["boxes"]):
                patch.set_facecolor("#93c47d" if feature_id == 1 else "#8fb3d9")
                patch.set_edgecolor("#4f4f4f")

            if row_idx == 0:
                rho = float(scenario.split("=")[1])
                ax.set_title(rf"$\rho = {rho:.1f}$")
            if col_idx == 0:
                ax.set_ylabel(
                    method_label,
                    rotation=90,
                    labelpad=12,
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )
            ax.set_ylim(0.0, y_max * 1.12 if y_max > 0 else 1.0)
            ax.grid(axis="y", color="#dddddd", linewidth=0.7)
            ax.set_axisbelow(True)

    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_50_feature_auc_boxplot(
    auc: pd.DataFrame,
    *,
    output_path: str,
) -> Path:
    methods = [
        ("nominal", "SHAP"),
        ("pessimistic", r"SHAP$^{Pess}$"),
        ("shrunk", r"SHAP$^{Shrunk}$"),
        ("oob", r"SHAP$^{oob}$"),
        ("safe", r"SHAP$^{safe}$"),
    ]
    data_values = [
        auc.loc[auc["method"] == method, "auc"].to_numpy(dtype=float)
        for method, _ in methods
    ]

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    box = ax.boxplot(
        data_values,
        tick_labels=[label for _, label in methods],
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    colors = ("#8fb3d9", "#93c47d", "#d9a066", "#9b8fd9", "#d98f8f")
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#4f4f4f")

    ax.set_title("Feature-level AUC on Loecher-style 50-feature simulation")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def print_table(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    print(frame.to_string(index=False, float_format=lambda value: f"{value: .4f}"))


# ----------------------------
# CLI
# ----------------------------
def _parse_r_values(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("At least one r value is required.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean synthetic debias experiments: 5-feature boxplots and 50-feature AUC."
    )
    parser.add_argument("--task", choices=["five_feature_boxplot", "fifty_feature_auc", "all"], default="all")
    parser.add_argument("--random-state", type=int, default=20260526)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n-trees", type=int, default=N_TREES)
    parser.add_argument("--eval-size", type=int, default=EVAL_SIZE)
    parser.add_argument("--s", type=int, default=S)
    parser.add_argument("--r-values", default=",".join(str(v) for v in POWER_R_VALUES))
    parser.add_argument("--auc-n-repeats", type=int, default=EXP3_N_REPEATS)
    parser.add_argument("--auc-n-samples", type=int, default=EXP3_N_SAMPLES)
    parser.add_argument("--auc-n-features", type=int, default=EXP3_N_FEATURES)
    parser.add_argument("--auc-n-relevant", type=int, default=EXP3_RELEVANT_FEATURES)
    parser.add_argument("--auc-relevant-pool", type=int, default=EXP3_RELEVANT_POOL)
    parser.add_argument("--auc-n-trees", type=int, default=EXP3_N_TREES)
    parser.add_argument("--auc-eval-size", type=int, default=EXP3_EVAL_SIZE)
    parser.add_argument("--auc-max-leaf-nodes", type=int, default=EXP3_MAX_LEAF_NODES)
    parser.add_argument("--results-dir", default="results_submission/debias")
    parser.add_argument("--figures-dir", default="figures/debias")
    return parser


def config_from_args(args: argparse.Namespace) -> DebiasConfig:
    return DebiasConfig(
        random_state=args.random_state,
        n_samples=args.n_samples,
        n_repeats=args.n_repeats,
        n_trees=args.n_trees,
        eval_size=args.eval_size,
        s=args.s,
        power_r_values=_parse_r_values(args.r_values),
        auc_n_samples=args.auc_n_samples,
        auc_n_features=args.auc_n_features,
        auc_n_relevant=args.auc_n_relevant,
        auc_relevant_pool=args.auc_relevant_pool,
        auc_n_repeats=args.auc_n_repeats,
        auc_n_trees=args.auc_n_trees,
        auc_eval_size=args.auc_eval_size,
        auc_max_leaf_nodes=args.auc_max_leaf_nodes,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    rng = np.random.default_rng(config.random_state)

    power_summary = pd.DataFrame()
    importance = pd.DataFrame()
    auc_summary = pd.DataFrame()
    auc = pd.DataFrame()

    if args.task in {"five_feature_boxplot", "all"}:
        print("Running 5-feature synthetic boxplot experiment...")
        power_summary, importance = run_power_boxplot_experiment(config, rng)
        print_table("5-feature rank summary", power_summary)

    if args.task in {"fifty_feature_auc", "all"}:
        print("Running 50-feature synthetic AUC experiment...")
        auc_summary, auc = run_50_feature_auc_experiment(config, rng)
        print_table("50-feature AUC summary", auc_summary)

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []

    if not power_summary.empty:
        power_summary.to_csv(results_dir / "synthetic_5_feature_boxplot_summary.csv", index=False)
        importance.to_csv(results_dir / "synthetic_5_feature_boxplot_importance.csv", index=False)
        figure_paths.append(
            plot_5_feature_boxplots(
                importance,
                r_values=config.power_r_values,
                output_path=str(Path(config.figures_dir) / "synthetic_5_feature_boxplots.png"),
            )
        )

    if not auc_summary.empty:
        auc_summary.to_csv(results_dir / "synthetic_50_feature_auc_summary.csv", index=False)
        auc.to_csv(results_dir / "synthetic_50_feature_auc.csv", index=False)
        figure_paths.append(
            plot_50_feature_auc_boxplot(
                auc,
                output_path=str(Path(config.figures_dir) / "synthetic_50_feature_auc_boxplot.png"),
            )
        )

    for path in figure_paths:
        print(f"Saved figure: {path}")
    print(f"Saved CSVs under: {results_dir}")


if __name__ == "__main__":
    main()
