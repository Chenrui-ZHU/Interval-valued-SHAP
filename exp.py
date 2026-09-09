from __future__ import annotations

import argparse
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import math
import os
import time
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

import data
import models
from interval_attribution import (
    _prepare_forest_tree_payloads,
    _rf_global_allocation_signed_endpoint_bounds_for_instance,
    _stable_order_from_scores,
    compute_interval_attribution,
    compute_nominal_attribution,
    extract_sklearn_tree,
    features_used_in_tree,
    leaf_class_counts,
    point_leaf_probabilities,
    topk_set_endpoint_score,
    tree_attribution_coefficient_matrix,
)


DEFAULT_DATASETS = ("breast_cancer", "diabetes", "ionosphere", "nhanes")

_RETRAINED_BOOTSTRAP_VALUES_STATE: dict[str, Any] | None = None
_METHOD_INTERVAL_STATE: dict[str, Any] | None = None
_TASK6_INSTANCE_STATE: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    datasets: tuple[str, ...]
    kind: str = "shapley"
    s_fixed: int = 1
    k_top: int = 3
    cv: int = 5
    n_bootstrap: int = 100
    semantics: str = "interventional"
    allocation_method: str = "greedy"
    topk_basis: str = "absolute"
    bg_size: int = 100
    target_class: int = 1
    tol: float = 1e-6
    random_state: int = 0
    n_eval: int | None = None
    inner_cv: int = 3
    scoring: str = "balanced_accuracy"
    max_tree_leaf_nodes: int = 8
    forest_n_estimators: int = 100
    n_jobs: int = 1
    save_dir: str = "results_submission"
    make_plots: bool = False


# ----------------------------
# Statistics
# ----------------------------
def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x = x[mask] - x[mask].mean()
    y = y[mask] - y[mask].mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 0.0:
        return np.nan
    return float(np.sum(x * y) / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    def rankdata(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
                j += 1
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return ranks

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    return pearson_corr(rankdata(x[mask]), rankdata(y[mask]))


def finite_pair_count(x: np.ndarray, y: np.ndarray) -> int:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return int((np.isfinite(x) & np.isfinite(y)).sum())


def width_rank_positions(widths: np.ndarray) -> np.ndarray:
    order = _stable_order_from_scores(np.asarray(widths, dtype=float), descending=True)
    ranks = np.empty(len(order), dtype=int)
    for position, feature_id in enumerate(order, start=1):
        ranks[int(feature_id)] = int(position)
    return ranks


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not np.isfinite(x) or not np.isfinite(a) or not np.isfinite(b) or a <= 0.0 or b <= 0.0:
        return np.nan
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    def beta_continued_fraction(aa: float, bb: float, xx: float) -> float:
        max_iter = 200
        eps = 3.0e-14
        fpmin = 1.0e-300
        qab = aa + bb
        qap = aa + 1.0
        qam = aa - 1.0
        c = 1.0
        d = 1.0 - qab * xx / qap
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0 / d
        h = d
        for m in range(1, max_iter + 1):
            m2 = 2 * m
            term = m * (bb - m) * xx / ((qam + m2) * (aa + m2))
            d = 1.0 + term * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + term / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            h *= d * c

            term = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2))
            d = 1.0 + term * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + term / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps:
                break
        return h

    log_beta_term = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    beta_term = math.exp(log_beta_term)
    if x < (a + 1.0) / (a + b + 2.0):
        value = beta_term * beta_continued_fraction(a, b, x) / a
    else:
        value = 1.0 - beta_term * beta_continued_fraction(b, a, 1.0 - x) / b
    return float(min(1.0, max(0.0, value)))


def _student_t_two_sided_p_value(t_statistic: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0 or not np.isfinite(t_statistic):
        if np.isposinf(abs(t_statistic)) and degrees_of_freedom > 0:
            return 0.0
        return np.nan
    t_abs = abs(float(t_statistic))
    x = degrees_of_freedom / (degrees_of_freedom + t_abs * t_abs)
    return _regularized_incomplete_beta(0.5 * degrees_of_freedom, 0.5, x)


def correlation_p_value(correlation: float, n_observations: int) -> float:
    if n_observations < 3 or not np.isfinite(correlation):
        return np.nan
    r = float(np.clip(correlation, -1.0, 1.0))
    if abs(r) >= 1.0:
        return 0.0
    denominator = max(1.0 - r * r, 0.0)
    if denominator <= 0.0:
        return 0.0
    t_statistic = r * math.sqrt((n_observations - 2) / denominator)
    return _student_t_two_sided_p_value(t_statistic, n_observations - 2)


def fold_level_correlation_p_value(fold_correlations: Sequence[float]) -> float:
    values = np.asarray(fold_correlations, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    std = float(values.std(ddof=1))
    mean = float(values.mean())
    if std <= 0.0:
        return 1.0 if mean == 0.0 else 0.0
    t_statistic = mean / (std / math.sqrt(len(values)))
    return _student_t_two_sided_p_value(t_statistic, len(values) - 1)


def _finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    return float(values[mask].mean()) if mask.any() else np.nan


def _finite_max(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    return float(values[mask].max()) if mask.any() else np.nan


def _resolve_n_jobs(n_jobs: int | None, n_tasks: int) -> int:
    if n_jobs is None:
        n_jobs = 1
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    return max(1, min(int(n_jobs), max(1, int(n_tasks))))


# ----------------------------
# Data/model helpers
# ----------------------------
def _task4_is_categorical_column(series: pd.Series) -> bool:
    dtype = series.dtype
    return bool(
        pd.api.types.is_object_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
    )


def _deduplicate_names(names: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique_names: list[str] = []
    for raw_name in names:
        name = str(raw_name)
        count = seen.get(name, 0)
        unique_names.append(name if count == 0 else f"{name}__{count}")
        seen[name] = count + 1
    return unique_names


def _encode_fold_features(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    encoded_names: list[str] = []
    group_rows: list[dict[str, Any]] = []

    for original_id, column in enumerate(X_train.columns):
        original_name = str(column)
        is_categorical = _task4_is_categorical_column(X_train[column])

        if is_categorical:
            train_series = (
                X_train[column]
                .astype("object")
                .where(pd.notna(X_train[column]), "__missing__")
                .astype(str)
            )
            valid_series = (
                X_valid[column]
                .astype("object")
                .where(pd.notna(X_valid[column]), "__missing__")
                .astype(str)
            )
            train_block = pd.get_dummies(train_series, prefix=original_name, prefix_sep="=", dtype=float)
            valid_block = pd.get_dummies(valid_series, prefix=original_name, prefix_sep="=", dtype=float)
            valid_block = valid_block.reindex(columns=train_block.columns, fill_value=0.0)
        else:
            train_block = pd.DataFrame(
                {original_name: pd.to_numeric(X_train[column], errors="coerce").astype(float)},
                index=X_train.index,
            )
            valid_block = pd.DataFrame(
                {original_name: pd.to_numeric(X_valid[column], errors="coerce").astype(float)},
                index=X_valid.index,
            )

        for encoded_name in train_block.columns:
            encoded_names.append(str(encoded_name))
            group_rows.append(
                {
                    "encoded_feature_id": len(group_rows),
                    "encoded_feature_name": str(encoded_name),
                    "original_feature_id": int(original_id),
                    "original_feature_name": original_name,
                    "is_categorical": bool(is_categorical),
                }
            )
        train_parts.append(train_block.reset_index(drop=True))
        valid_parts.append(valid_block.reset_index(drop=True))

    unique_names = _deduplicate_names(encoded_names)
    X_train_encoded = pd.concat(train_parts, axis=1)
    X_valid_encoded = pd.concat(valid_parts, axis=1)
    X_train_encoded.columns = unique_names
    X_valid_encoded.columns = unique_names

    group_frame = pd.DataFrame(group_rows)
    group_frame["encoded_feature_name"] = unique_names
    return X_train_encoded, X_valid_encoded, group_frame


def _select_model_by_type(
    model_type: str,
    X_train,
    y_train,
    *,
    inner_cv: int,
    scoring: str,
    random_state: int,
    max_tree_leaf_nodes: int | None = None,
    forest_n_estimators: int = 100,
):
    if model_type == "tree":
        return models.select_decision_tree_cv(
            X_train,
            y_train,
            cv=inner_cv,
            scoring=scoring,
            random_state=random_state,
            tree_specs=(
                models.certifiable_tree_specs(
                    random_state=random_state,
                    max_leaf_nodes=max_tree_leaf_nodes,
                )
                if max_tree_leaf_nodes is not None
                else None
            ),
        )
    if model_type == "forest":
        return models.select_random_forest_cv(
            X_train,
            y_train,
            cv=inner_cv,
            scoring=scoring,
            random_state=random_state,
            forest_specs=(
                models.certifiable_random_forest_specs(
                    random_state=random_state,
                    max_leaf_nodes=max_tree_leaf_nodes,
                    n_estimators=forest_n_estimators,
                )
                if max_tree_leaf_nodes is not None
                else None
            ),
        )
    raise ValueError("model_type must be 'tree' or 'forest'.")


def _fit_retrained_bootstrap_estimators(
    reference_estimator,
    X_train,
    y_train,
    *,
    n_bootstrap: int,
    random_state: int,
) -> list[Any]:
    rng = np.random.default_rng(random_state)
    estimators: list[Any] = []
    for boot_id in range(n_bootstrap):
        sample_idx = rng.integers(0, len(X_train), size=len(X_train))
        estimator = clone(reference_estimator)
        params = estimator.get_params(deep=False)
        if "random_state" in params:
            estimator.set_params(random_state=random_state + boot_id + 1)
        estimator.fit(
            X_train.iloc[sample_idx].reset_index(drop=True),
            y_train.iloc[sample_idx].reset_index(drop=True),
        )
        estimators.append(estimator)
    return estimators


# ----------------------------
# Task 1: interval-rank
# ----------------------------
def _init_retrained_bootstrap_values_worker(state: dict[str, Any]) -> None:
    global _RETRAINED_BOOTSTRAP_VALUES_STATE
    _RETRAINED_BOOTSTRAP_VALUES_STATE = state


def _retrained_bootstrap_nominal_values(
    estimator,
    state: dict[str, Any],
    *,
    boot_id: int,
) -> list[tuple[int, np.ndarray]]:
    random_state = int(state["random_state"]) + int(boot_id) + 1
    X_reference = state["X_reference"]
    X_eval = state["X_eval"]
    X_bg = (
        X_reference
        if len(X_reference) <= int(state["bg_size"])
        else X_reference.sample(n=int(state["bg_size"]), random_state=random_state)
    )
    n_features = X_reference.shape[1]
    rows: list[tuple[int, np.ndarray]] = []

    if isinstance(estimator, DecisionTreeClassifier):
        tree_struct = extract_sklearn_tree(estimator)
        feature_ids = features_used_in_tree(tree_struct)
        if not feature_ids:
            values = np.zeros(n_features, dtype=float)
            return [(int(idx), values.copy()) for idx in state["eval_idx"]]

        counts = leaf_class_counts(estimator, tree_struct)
        w_hat = point_leaf_probabilities(counts, target_class=state["target_class"])
        feature_idx = np.asarray(feature_ids, dtype=int)
        for idx in state["eval_idx"]:
            C_local = tree_attribution_coefficient_matrix(
                tree_struct,
                X_eval.iloc[int(idx)],
                feature_ids,
                kind=state["kind"],
                semantics=state["semantics"],
                X_bg=X_bg,
            )
            values = np.zeros(n_features, dtype=float)
            values[feature_idx] = C_local @ w_hat
            rows.append((int(idx), values))
        return rows

    if isinstance(estimator, RandomForestClassifier):
        tree_payloads = _prepare_forest_tree_payloads(estimator)
        for idx in state["eval_idx"]:
            x_star = X_eval.iloc[int(idx)]
            values = np.zeros(n_features, dtype=float)
            for payload in tree_payloads:
                C_local = tree_attribution_coefficient_matrix(
                    payload["tree"],
                    x_star,
                    payload["U"],
                    kind=state["kind"],
                    semantics=state["semantics"],
                    X_bg=X_bg,
                )
                w_hat = point_leaf_probabilities(payload["counts"], target_class=state["target_class"])
                values[np.asarray(payload["U"], dtype=int)] += C_local @ w_hat
            if tree_payloads:
                values /= len(tree_payloads)
            rows.append((int(idx), values))
        return rows

    for idx in state["eval_idx"]:
        nominal = compute_nominal_attribution(
            estimator,
            X_reference,
            X_eval.iloc[int(idx)],
            kind=state["kind"],
            semantics=state["semantics"],
            target_class=state["target_class"],
            bg_size=state["bg_size"],
            random_state=random_state,
        )
        rows.append((int(idx), np.asarray(nominal.values, dtype=float)))
    return rows


def _retrained_bootstrap_values_worker(task: tuple[int, Any]) -> tuple[int, list[tuple[int, np.ndarray]]]:
    state = _RETRAINED_BOOTSTRAP_VALUES_STATE
    if state is None:
        raise RuntimeError("Retrained bootstrap worker state is not initialized.")
    boot_id, estimator = task
    return int(boot_id), _retrained_bootstrap_nominal_values(estimator, state, boot_id=int(boot_id))


def _init_method_interval_worker(state: dict[str, Any]) -> None:
    global _METHOD_INTERVAL_STATE
    _METHOD_INTERVAL_STATE = state


def _method_interval_worker(idx: int) -> tuple[int, np.ndarray, np.ndarray]:
    state = _METHOD_INTERVAL_STATE
    if state is None:
        raise RuntimeError("Method interval worker state is not initialized.")

    idx = int(idx)
    x_star = state["X_eval"].iloc[idx]
    random_state = int(state["random_state"]) + idx

    if state["model_type"] == "forest":
        X_reference = state["X_reference"]
        bg_size = int(state["bg_size"])
        X_bg = (
            X_reference
            if len(X_reference) <= bg_size
            else X_reference.sample(n=bg_size, random_state=random_state)
        )
        signed_lower, signed_upper = _rf_global_allocation_signed_endpoint_bounds_for_instance(
            state["tree_payloads"],
            n_features=int(state["n_features"]),
            x_star=x_star,
            X_bg=X_bg,
            s_values=[int(state["s_fixed"])],
            allocation_method=state["allocation_method"],
            kind=state["kind"],
            semantics=state["semantics"],
            target_class=int(state["target_class"]),
        )[int(state["s_fixed"])]
    else:
        interval = compute_interval_attribution(
            state["estimator"],
            state["X_reference"],
            x_star,
            s=int(state["s_fixed"]),
            kind=state["kind"],
            semantics=state["semantics"],
            allocation=state["allocation_method"],
            target_class=int(state["target_class"]),
            bg_size=int(state["bg_size"]),
            random_state=random_state,
        )
        signed_lower = np.asarray(interval.signed_lower, dtype=float)
        signed_upper = np.asarray(interval.signed_upper, dtype=float)

    return idx, np.asarray(signed_lower, dtype=float), np.asarray(signed_upper, dtype=float)


def task1_bootstrap_interval_rank_correlation_cv(
    dataset_name: str,
    model_type: str = "forest",
    kind: str = "shapley",
    s_fixed: int = 1,
    cv: int = 5,
    n_bootstrap: int = 100,
    semantics: str = "interventional",
    bg_size: int = 100,
    target_class: int = 1,
    allocation_method: str = "greedy",
    random_state: int = 0,
    n_eval: Optional[int] = None,
    inner_cv: int = 3,
    scoring: str = "balanced_accuracy",
    max_tree_leaf_nodes: Optional[int] = 8,
    forest_n_estimators: int = 100,
    n_jobs: int = 1,
    save_dir: Optional[str] = "results_submission",
) -> dict[str, pd.DataFrame]:
    if model_type not in {"tree", "forest"}:
        raise ValueError("model_type must be 'tree' or 'forest'.")
    if model_type == "tree" and allocation_method not in {"greedy", "multinomial", "box"}:
        raise ValueError("Tree allocation_method must be 'greedy', 'multinomial', or 'box'.")
    if model_type == "forest" and allocation_method not in {"greedy", "multinomial"}:
        raise ValueError("Forest allocation_method must be 'greedy' or 'multinomial'.")
    if model_type == "forest" and semantics != "interventional":
        raise ValueError("Random forest interval attribution supports interventional semantics here.")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2.")

    X, y, info = data.load_dataset(dataset_name)
    print(info)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    instance_rows: list[dict[str, Any]] = []

    for fold_id, (tr_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
        start_time = time.time()
        Xtr = X.iloc[tr_idx].reset_index(drop=True)
        ytr = y.iloc[tr_idx].reset_index(drop=True)
        Xval = X.iloc[val_idx].reset_index(drop=True)
        yval = y.iloc[val_idx].reset_index(drop=True)
        Xtr_model, Xval_model, _feature_group_frame = _encode_fold_features(Xtr, Xval)

        if n_eval is None or n_eval >= len(Xval_model):
            eval_idx = np.arange(len(Xval_model), dtype=int)
        else:
            rng_eval = np.random.default_rng(random_state + 10_000 + fold_id)
            eval_idx = np.sort(rng_eval.choice(len(Xval_model), size=n_eval, replace=False))

        selected_model = _select_model_by_type(
            model_type,
            Xtr_model,
            ytr,
            inner_cv=inner_cv,
            scoring=scoring,
            random_state=random_state + fold_id,
            max_tree_leaf_nodes=max_tree_leaf_nodes,
            forest_n_estimators=forest_n_estimators,
        )
        bootstrap_estimators = _fit_retrained_bootstrap_estimators(
            selected_model.estimator,
            Xtr_model,
            ytr,
            n_bootstrap=n_bootstrap,
            random_state=random_state + 20_000 * fold_id,
        )
        print(
            f"[fold {fold_id}] task 1 interval-rank setup: {time.time() - start_time:.1f}s "
            f"(eval_instances={len(eval_idx)}, n_bootstrap={n_bootstrap})"
        )

        worker_state = {
            "X_reference": Xtr_model,
            "X_eval": Xval_model,
            "eval_idx": np.asarray(eval_idx, dtype=int),
            "kind": kind,
            "semantics": semantics,
            "target_class": target_class,
            "bg_size": bg_size,
            "random_state": random_state + 30_000 * fold_id,
        }
        worker_jobs = _resolve_n_jobs(n_jobs, len(bootstrap_estimators))
        worker_tasks = list(enumerate(bootstrap_estimators))
        if worker_jobs > 1:
            with ProcessPoolExecutor(
                max_workers=worker_jobs,
                initializer=_init_retrained_bootstrap_values_worker,
                initargs=(worker_state,),
            ) as executor:
                worker_results = list(executor.map(_retrained_bootstrap_values_worker, worker_tasks))
        else:
            _init_retrained_bootstrap_values_worker(worker_state)
            worker_results = [_retrained_bootstrap_values_worker(task) for task in worker_tasks]

        bootstrap_matrices: list[np.ndarray] = []
        for _, rows in sorted(worker_results, key=lambda item: item[0]):
            values_by_idx = {int(idx): np.asarray(values, dtype=float) for idx, values in rows}
            bootstrap_matrices.append(np.vstack([values_by_idx[int(idx)] for idx in eval_idx]))
        bootstrap_cube = np.stack(bootstrap_matrices, axis=0)
        bootstrap_lower_matrix = np.min(bootstrap_cube, axis=0)
        bootstrap_upper_matrix = np.max(bootstrap_cube, axis=0)
        bootstrap_width_matrix = bootstrap_upper_matrix - bootstrap_lower_matrix

        interval_state = {
            "model_type": model_type,
            "estimator": selected_model.estimator,
            "tree_payloads": (
                _prepare_forest_tree_payloads(selected_model.estimator)
                if model_type == "forest"
                else None
            ),
            "X_reference": Xtr_model,
            "X_eval": Xval_model,
            "n_features": int(Xtr_model.shape[1]),
            "s_fixed": int(s_fixed),
            "kind": kind,
            "semantics": semantics,
            "allocation_method": allocation_method,
            "target_class": int(target_class),
            "bg_size": int(bg_size),
            "random_state": random_state + 40_000 * fold_id,
        }
        interval_jobs = _resolve_n_jobs(n_jobs, len(eval_idx))
        if interval_jobs > 1:
            with ProcessPoolExecutor(
                max_workers=interval_jobs,
                initializer=_init_method_interval_worker,
                initargs=(interval_state,),
            ) as executor:
                interval_results = list(executor.map(_method_interval_worker, [int(idx) for idx in eval_idx]))
        else:
            _init_method_interval_worker(interval_state)
            interval_results = [_method_interval_worker(int(idx)) for idx in eval_idx]

        method_lower_by_idx: dict[int, np.ndarray] = {}
        method_upper_by_idx: dict[int, np.ndarray] = {}
        method_widths_by_idx: dict[int, np.ndarray] = {}
        for idx, method_lower, method_upper in interval_results:
            method_lower_by_idx[int(idx)] = method_lower
            method_upper_by_idx[int(idx)] = method_upper
            method_widths_by_idx[int(idx)] = method_upper - method_lower

        n_features = Xtr_model.shape[1]
        for row_position, idx in enumerate(eval_idx):
            method_width = method_widths_by_idx[int(idx)]
            bootstrap_width = bootstrap_width_matrix[row_position]
            method_rank = width_rank_positions(method_width)
            bootstrap_rank = width_rank_positions(bootstrap_width)
            n_pairs = finite_pair_count(method_width, bootstrap_width)
            pearson_r = pearson_corr(method_width, bootstrap_width)
            spearman_r = spearman_corr(method_width, bootstrap_width)
            rank_pearson_r = pearson_corr(method_rank, bootstrap_rank)
            instance_rows.append(
                {
                    "dataset": dataset_name,
                    "model_type": model_type,
                    "fold_id": int(fold_id),
                    "instance_id": int(idx),
                    "y_true": int(yval.iloc[int(idx)]),
                    "kind": kind,
                    "s_fixed": int(s_fixed),
                    "allocation_method": allocation_method,
                    "semantics": semantics,
                    "n_bootstrap": int(n_bootstrap),
                    "n_features": int(n_features),
                    "n_pairs": int(n_pairs),
                    "pearson_width_correlation": pearson_r,
                    "pearson_width_p_value": correlation_p_value(pearson_r, n_pairs),
                    "spearman_width_correlation": spearman_r,
                    "spearman_width_p_value": correlation_p_value(spearman_r, n_pairs),
                    "pearson_rank_correlation": rank_pearson_r,
                    "pearson_rank_p_value": correlation_p_value(rank_pearson_r, n_pairs),
                    "method_signed_interval_lower": tuple(float(v) for v in method_lower_by_idx[int(idx)]),
                    "method_signed_interval_upper": tuple(float(v) for v in method_upper_by_idx[int(idx)]),
                    "method_signed_width": tuple(float(v) for v in method_width),
                    "bootstrap_signed_interval_lower": tuple(float(v) for v in bootstrap_lower_matrix[row_position]),
                    "bootstrap_signed_interval_upper": tuple(float(v) for v in bootstrap_upper_matrix[row_position]),
                    "bootstrap_signed_width": tuple(float(v) for v in bootstrap_width),
                    "method_width_rank": tuple(int(v) for v in method_rank),
                    "bootstrap_width_rank": tuple(int(v) for v in bootstrap_rank),
                    "base_inner_cv_score": (
                        float(selected_model.cv_score) if selected_model.cv_score is not None else np.nan
                    ),
                }
            )
        print(
            f"[fold {fold_id}] task 1 interval-rank completed in {time.time() - start_time:.1f}s "
            f"(bootstrap_n_jobs={worker_jobs}, interval_n_jobs={interval_jobs})"
        )

    if not instance_rows:
        raise RuntimeError("No Task 1 interval-rank rows were produced.")

    df_instances = pd.DataFrame(instance_rows)
    df_fold_summary = (
        df_instances
        .groupby("fold_id", as_index=False)
        .agg(
            n_instances=("instance_id", "count"),
            mean_pearson_width_correlation=("pearson_width_correlation", "mean"),
            mean_spearman_width_correlation=("spearman_width_correlation", "mean"),
            mean_pearson_rank_correlation=("pearson_rank_correlation", "mean"),
            pearson_width_mean_p_value=(
                "pearson_width_correlation",
                lambda values: fold_level_correlation_p_value(values.to_numpy(dtype=float)),
            ),
            spearman_width_mean_p_value=(
                "spearman_width_correlation",
                lambda values: fold_level_correlation_p_value(values.to_numpy(dtype=float)),
            ),
            pearson_rank_mean_p_value=(
                "pearson_rank_correlation",
                lambda values: fold_level_correlation_p_value(values.to_numpy(dtype=float)),
            ),
        )
    )
    df_summary = pd.DataFrame(
        [
            {"metric": "mean_pearson_width_correlation", "value": float(df_instances["pearson_width_correlation"].mean())},
            {"metric": "mean_pearson_width_p_value", "value": fold_level_correlation_p_value(df_instances["pearson_width_correlation"])},
            {"metric": "median_pearson_width_correlation", "value": float(df_instances["pearson_width_correlation"].median())},
            {"metric": "mean_spearman_width_correlation", "value": float(df_instances["spearman_width_correlation"].mean())},
            {"metric": "mean_spearman_width_p_value", "value": fold_level_correlation_p_value(df_instances["spearman_width_correlation"])},
            {"metric": "median_spearman_width_correlation", "value": float(df_instances["spearman_width_correlation"].median())},
            {"metric": "mean_pearson_rank_correlation", "value": float(df_instances["pearson_rank_correlation"].mean())},
            {"metric": "mean_pearson_rank_p_value", "value": fold_level_correlation_p_value(df_instances["pearson_rank_correlation"])},
            {"metric": "n_instances", "value": float(len(df_instances))},
        ]
    )

    if save_dir is not None:
        task_save_dir = f"{save_dir}/task1/{dataset_name}"
        os.makedirs(task_save_dir, exist_ok=True)
        stem = (
            f"{dataset_name}_{model_type}_{kind}_s={s_fixed}_b={n_bootstrap}_cv={cv}_"
            f"{semantics}_{allocation_method}_bootstrap_interval_rank"
        )
        df_instances.to_csv(f"{task_save_dir}/bootstrap_interval_rank_instances_{stem}.csv", index=False)
        df_fold_summary.to_csv(f"{task_save_dir}/bootstrap_interval_rank_fold_summary_{stem}.csv", index=False)
        df_summary.to_csv(f"{task_save_dir}/bootstrap_interval_rank_summary_{stem}.csv", index=False)

    return {"instances": df_instances, "fold_summary": df_fold_summary, "summary": df_summary}


# ----------------------------
# Task 6: forest boundary distance
# ----------------------------
def _init_task6_instance_worker(state: dict[str, Any]) -> None:
    global _TASK6_INSTANCE_STATE
    _TASK6_INSTANCE_STATE = state


def _probability_margin(estimator, x_star) -> tuple[float, Any, float, float]:
    x_frame = x_star.to_frame().T if hasattr(x_star, "to_frame") else np.asarray(x_star).reshape(1, -1)
    proba = np.asarray(estimator.predict_proba(x_frame)[0], dtype=float)
    classes = np.asarray(estimator.classes_)
    order = np.argsort(proba)[::-1]
    top1 = int(order[0])
    top2_prob = float(proba[int(order[1])]) if len(order) > 1 else 0.0
    return float(proba[top1] - top2_prob), classes[top1], float(proba[top1]), top2_prob


def _forest_boundary_signal(estimator, x_star) -> tuple[dict[str, Any], Any, float, float, float]:
    margin, predicted_label, top1_prob, top2_prob = _probability_margin(estimator, x_star)
    boundary = {
        "boundary_distance": float(margin),
        "boundary_distance_kind": "forest_probability_margin",
        "signed_boundary_distance": np.nan,
        "dk_probability_target_class": np.nan,
        "dk_probability_margin": np.nan,
    }
    return boundary, predicted_label, float(margin), float(top1_prob), float(top2_prob)


def _signed_interval_sensitivity_scores(
    nominal_values: np.ndarray,
    signed_lower: np.ndarray,
    signed_upper: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nominal_values = np.asarray(nominal_values, dtype=float)
    signed_lower = np.asarray(signed_lower, dtype=float)
    signed_upper = np.asarray(signed_upper, dtype=float)
    a = np.minimum(signed_lower, signed_upper)
    b = np.maximum(signed_lower, signed_upper)
    width = b - a

    below = nominal_values <= a
    above = nominal_values >= b
    inside = ~(below | above)

    sensitivity_integral = np.empty_like(nominal_values, dtype=float)
    sensitivity_integral[below] = 0.5 * (
        (b[below] - nominal_values[below]) ** 2 - (a[below] - nominal_values[below]) ** 2
    )
    sensitivity_integral[above] = 0.5 * (
        (nominal_values[above] - a[above]) ** 2 - (nominal_values[above] - b[above]) ** 2
    )
    sensitivity_integral[inside] = 0.5 * (
        (nominal_values[inside] - a[inside]) ** 2 + (b[inside] - nominal_values[inside]) ** 2
    )

    degenerate_sensitivity = np.abs(a - nominal_values)
    mean_sensitivity = np.divide(
        sensitivity_integral,
        width,
        out=degenerate_sensitivity.copy(),
        where=width > epsilon,
    )
    relative_sensitivity = mean_sensitivity / (np.abs(nominal_values) + epsilon)
    sensitivity_aware_score = nominal_values / (1.0 + relative_sensitivity)
    return mean_sensitivity, relative_sensitivity, sensitivity_aware_score


def _task6_instance_worker(task: tuple[int, Any]) -> dict[str, Any]:
    state = _TASK6_INSTANCE_STATE
    if state is None:
        raise RuntimeError("Task 6 worker state is not initialized.")

    idx, x_star = task
    nominal = compute_nominal_attribution(
        state["estimator"],
        state["Xtr"],
        x_star,
        kind=state["kind"],
        semantics=state["semantics"],
        target_class=state["target_class"],
        bg_size=state["bg_size"],
        random_state=state["random_state"],
    )
    interval = compute_interval_attribution(
        state["estimator"],
        state["Xtr"],
        x_star,
        s=state["s_fixed"],
        kind=state["kind"],
        semantics=state["semantics"],
        allocation=state["allocation_method"],
        target_class=state["target_class"],
        bg_size=state["bg_size"],
        random_state=state["random_state"],
    )

    nominal_values = np.asarray(nominal.values, dtype=float)
    nominal_abs = np.abs(nominal_values)
    signed_lower = np.asarray(interval.signed_lower, dtype=float)
    signed_upper = np.asarray(interval.signed_upper, dtype=float)
    absolute_lower = np.asarray(interval.absolute_lower, dtype=float)
    absolute_upper = np.asarray(interval.absolute_upper, dtype=float)
    signed_width = signed_upper - signed_lower
    absolute_width = absolute_upper - absolute_lower
    _mean_sens, relative_sensitivity, sensitivity_aware_score = _signed_interval_sensitivity_scores(
        nominal_values,
        signed_lower,
        signed_upper,
        epsilon=state["epsilon"],
    )

    k_top = min(int(state["k_top"]), int(state["n_features"]))
    if state["topk_basis"] == "signed":
        topk_order = _stable_order_from_scores(nominal_values, descending=True)
        topk_set_current = set(topk_order[:k_top])
        certification_score = topk_set_endpoint_score(signed_lower, signed_upper, topk_set_current)
    else:
        topk_order = _stable_order_from_scores(nominal_abs, descending=True)
        topk_set_current = set(topk_order[:k_top])
        certification_score = topk_set_endpoint_score(absolute_lower, absolute_upper, topk_set_current)

    certified = bool(
        certification_score > state["tol"]
        if state["strict"]
        else certification_score >= -state["tol"]
    )
    topk_idx = np.asarray(topk_order[:k_top], dtype=int)
    boundary_info, predicted_label, margin, top1_prob, top2_prob = _forest_boundary_signal(
        state["estimator"],
        x_star,
    )
    y_true = state["y_true_by_idx"][int(idx)]

    row = {
        "fold_id": int(state["fold_id"]),
        "instance_id": int(idx),
        "dataset": state["dataset_name"],
        "model_type": "forest",
        "y_true": y_true,
        "y_pred": predicted_label,
        "is_correct": bool(predicted_label == y_true),
        "probability_margin": float(margin),
        "top1_probability": float(top1_prob),
        "top2_probability": float(top2_prob),
        "kind": state["kind"],
        "k_top": int(k_top),
        "s_fixed": int(state["s_fixed"]),
        "allocation_method": state["allocation_method"],
        "semantics": state["semantics"],
        "topk_basis": state["topk_basis"],
        "nominal_topk_set": tuple(sorted(int(v) for v in topk_set_current)),
        "certified_topk_set": bool(certified),
        "certification_score": float(certification_score),
        "nominal_values": tuple(float(v) for v in nominal_values),
        "nominal_absolute_values": tuple(float(v) for v in nominal_abs),
        "signed_widths": tuple(float(v) for v in signed_width),
        "absolute_widths": tuple(float(v) for v in absolute_width),
        "signed_relative_sensitivity_values": tuple(float(v) for v in relative_sensitivity),
        "signed_sensitivity_aware_scores": tuple(float(v) for v in sensitivity_aware_score),
        "mean_signed_width": _finite_mean(signed_width),
        "mean_absolute_width": _finite_mean(absolute_width),
        "topk_mean_signed_width": _finite_mean(signed_width[topk_idx]) if len(topk_idx) else np.nan,
        "topk_mean_absolute_width": _finite_mean(absolute_width[topk_idx]) if len(topk_idx) else np.nan,
        "topk_max_signed_width": _finite_max(signed_width[topk_idx]) if len(topk_idx) else np.nan,
        "topk_max_absolute_width": _finite_max(absolute_width[topk_idx]) if len(topk_idx) else np.nan,
        "max_signed_width": _finite_max(signed_width),
        "max_absolute_width": _finite_max(absolute_width),
        "mean_relative_sensitivity": _finite_mean(relative_sensitivity),
        "mean_sensitivity_aware_score": _finite_mean(sensitivity_aware_score),
        "inner_cv_score": float(state["inner_cv_score"]),
    }
    row.update(boundary_info)
    return row


def _task6_boundary_group_labels(distances: pd.Series, n_groups: int = 3) -> pd.Series:
    values = distances.astype(float)
    finite = values[np.isfinite(values)]
    labels = pd.Series("unknown", index=distances.index, dtype=object)
    if finite.empty:
        return labels
    ranks = finite.rank(method="average")
    q = min(int(n_groups), int(ranks.nunique()), len(ranks))
    if q <= 1:
        labels.loc[finite.index] = "all"
        return labels
    codes = pd.qcut(ranks, q=q, labels=False, duplicates="drop")
    n_bins = int(np.nanmax(codes)) + 1
    group_names = ["near", "far"] if n_bins == 2 else ["near", "middle", "far"][:n_bins]
    labels.loc[finite.index] = [group_names[int(code)] for code in codes]
    return labels


def _task6_boundary_distance_summary(df_instances: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[Any, pd.DataFrame]] = [("all", df_instances)]
    groups.extend((int(fold_id), df_fold) for fold_id, df_fold in df_instances.groupby("fold_id", sort=True))

    for fold_id, df_group in groups:
        distances = df_group["boundary_distance"].astype(float)
        finite = distances[np.isfinite(distances)]
        counts = finite.value_counts(dropna=True)
        n_observations = int(len(finite))
        if counts.empty:
            rows.append(
                {
                    "dataset": df_instances["dataset"].iloc[0] if len(df_instances) else "",
                    "model_type": "forest",
                    "fold_id": fold_id,
                    "n_observations": n_observations,
                    "n_unique_boundary_distances": 0,
                    "largest_tie_count": 0,
                    "largest_tie_fraction": np.nan,
                    "tied_observation_fraction": np.nan,
                    "min_boundary_distance": np.nan,
                    "min_distance_count": 0,
                    "min_distance_fraction": np.nan,
                    "zero_distance_count": 0,
                    "zero_distance_fraction": np.nan,
                }
            )
            continue

        largest_tie_count = int(counts.max())
        tied_observations = int(counts[counts > 1].sum())
        min_distance = float(finite.min())
        min_distance_count = int(np.isclose(finite, min_distance).sum())
        zero_distance_count = int(np.isclose(finite, 0.0).sum())
        rows.append(
            {
                "dataset": df_instances["dataset"].iloc[0],
                "model_type": "forest",
                "fold_id": fold_id,
                "n_observations": n_observations,
                "n_unique_boundary_distances": int(len(counts)),
                "largest_tie_count": largest_tie_count,
                "largest_tie_fraction": float(largest_tie_count / n_observations),
                "tied_observation_fraction": float(tied_observations / n_observations),
                "min_boundary_distance": min_distance,
                "min_distance_count": min_distance_count,
                "min_distance_fraction": float(min_distance_count / n_observations),
                "zero_distance_count": zero_distance_count,
                "zero_distance_fraction": float(zero_distance_count / n_observations),
            }
        )
    return pd.DataFrame(rows)


def _task6_feature_width_tables(
    df_instances: pd.DataFrame,
    feature_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_features = len(feature_names)
    rows: list[dict[str, Any]] = []
    for _, instance_row in df_instances.iterrows():
        signed_widths = np.asarray(instance_row["signed_widths"], dtype=float)
        absolute_widths = np.asarray(instance_row["absolute_widths"], dtype=float)
        nominal_values = np.asarray(instance_row["nominal_values"], dtype=float)
        nominal_abs = np.asarray(instance_row["nominal_absolute_values"], dtype=float)
        relative_sensitivity = np.asarray(instance_row["signed_relative_sensitivity_values"], dtype=float)
        sensitivity_scores = np.asarray(instance_row["signed_sensitivity_aware_scores"], dtype=float)
        if signed_widths.shape[0] != n_features:
            continue
        nominal_topk = set(int(v) for v in instance_row.get("nominal_topk_set", ()))
        for feature_id, feature_name in enumerate(feature_names):
            rows.append(
                {
                    "dataset": instance_row["dataset"],
                    "model_type": "forest",
                    "fold_id": int(instance_row["fold_id"]),
                    "instance_id": int(instance_row["instance_id"]),
                    "boundary_group": instance_row.get("boundary_group"),
                    "boundary_distance": float(instance_row["boundary_distance"]),
                    "probability_margin": float(instance_row["probability_margin"]),
                    "is_correct": bool(instance_row["is_correct"]),
                    "certified_topk_set": bool(instance_row["certified_topk_set"]),
                    "feature_id": int(feature_id),
                    "feature_name": str(feature_name),
                    "is_nominal_topk": bool(feature_id in nominal_topk),
                    "nominal_value": float(nominal_values[feature_id]),
                    "nominal_absolute_value": float(nominal_abs[feature_id]),
                    "signed_width": float(signed_widths[feature_id]),
                    "absolute_width": float(absolute_widths[feature_id]),
                    "signed_relative_sensitivity": float(relative_sensitivity[feature_id]),
                    "signed_sensitivity_aware_score": float(sensitivity_scores[feature_id]),
                }
            )

    df_feature_widths = pd.DataFrame(rows)
    if df_feature_widths.empty:
        return df_feature_widths, pd.DataFrame(), pd.DataFrame()

    df_feature_summary = (
        df_feature_widths
        .groupby(["dataset", "model_type", "boundary_group", "feature_id", "feature_name"], as_index=False)
        .agg(
            n_instances=("instance_id", "count"),
            mean_boundary_distance=("boundary_distance", "mean"),
            mean_probability_margin=("probability_margin", "mean"),
            topk_instance_rate=("is_nominal_topk", "mean"),
            certified_topk_rate=("certified_topk_set", "mean"),
            mean_nominal_absolute_value=("nominal_absolute_value", "mean"),
            mean_signed_width=("signed_width", "mean"),
            mean_absolute_width=("absolute_width", "mean"),
            max_signed_width=("signed_width", "max"),
            max_absolute_width=("absolute_width", "max"),
            mean_signed_relative_sensitivity=("signed_relative_sensitivity", "mean"),
            mean_signed_sensitivity_aware_score=("signed_sensitivity_aware_score", "mean"),
        )
    )

    correlation_rows: list[dict[str, Any]] = []
    for feature_id, df_feature in df_feature_widths.groupby("feature_id", sort=True):
        for metric in ["signed_width", "absolute_width"]:
            boundary_values = df_feature["boundary_distance"].astype(float).to_numpy()
            metric_values = df_feature[metric].astype(float).to_numpy()
            n_observations = finite_pair_count(boundary_values, metric_values)
            pearson_r = pearson_corr(boundary_values, metric_values)
            spearman_r = spearman_corr(boundary_values, metric_values)
            correlation_rows.append(
                {
                    "dataset": df_feature["dataset"].iloc[0],
                    "model_type": "forest",
                    "feature_id": int(feature_id),
                    "feature_name": str(df_feature["feature_name"].iloc[0]),
                    "metric": metric,
                    "n_observations": n_observations,
                    "pearson_margin_vs_metric": pearson_r,
                    "pearson_p_value": correlation_p_value(pearson_r, n_observations),
                    "spearman_margin_vs_metric": spearman_r,
                    "spearman_p_value": correlation_p_value(spearman_r, n_observations),
                    "topk_instance_rate": float(df_feature["is_nominal_topk"].mean()),
                    "mean_metric": float(df_feature[metric].mean()),
                }
            )
    return df_feature_widths, df_feature_summary, pd.DataFrame(correlation_rows)


def task6_boundary_distance_interval_width_forest_cv(
    dataset_name: str,
    kind: str = "shapley",
    k_top: int = 3,
    s_fixed: int = 1,
    cv: int = 5,
    semantics: str = "interventional",
    bg_size: int = 100,
    target_class: int = 1,
    allocation_method: str = "greedy",
    topk_basis: str = "absolute",
    tol: float = 1e-6,
    strict: bool = True,
    random_state: int = 0,
    n_eval: Optional[int] = None,
    inner_cv: int = 3,
    scoring: str = "balanced_accuracy",
    max_tree_leaf_nodes: Optional[int] = 8,
    epsilon: float = 1e-12,
    n_jobs: int = 1,
    save_dir: Optional[str] = "results_submission",
    make_plots: bool = False,
) -> dict[str, Any]:
    del make_plots
    if semantics != "interventional":
        raise ValueError("Task 6 forest supports interventional semantics here.")
    if allocation_method not in {"greedy", "multinomial"}:
        raise ValueError("Forest allocation_method must be 'greedy' or 'multinomial'.")
    if topk_basis not in {"absolute", "signed"}:
        raise ValueError("topk_basis must be 'absolute' or 'signed'.")

    X, y, info = data.load_dataset(dataset_name)
    print(info)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows: list[dict[str, Any]] = []

    for fold_id, (tr_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
        start_time = time.time()
        Xtr = X.iloc[tr_idx].reset_index(drop=True)
        ytr = y.iloc[tr_idx].reset_index(drop=True)
        Xval = X.iloc[val_idx].reset_index(drop=True)
        yval = y.iloc[val_idx].reset_index(drop=True)
        Xtr_model, Xval_model, _feature_group_frame = _encode_fold_features(Xtr, Xval)

        if n_eval is None or n_eval >= len(Xval_model):
            eval_idx = np.arange(len(Xval_model), dtype=int)
        else:
            rng_eval = np.random.default_rng(random_state + 20_000 + fold_id)
            eval_idx = np.sort(rng_eval.choice(len(Xval_model), size=n_eval, replace=False))

        selected_model = _select_model_by_type(
            "forest",
            Xtr_model,
            ytr,
            inner_cv=inner_cv,
            scoring=scoring,
            random_state=random_state + fold_id,
            max_tree_leaf_nodes=max_tree_leaf_nodes,
        )
        worker_state = {
            "estimator": selected_model.estimator,
            "Xtr": Xtr_model,
            "n_features": int(Xtr_model.shape[1]),
            "dataset_name": dataset_name,
            "fold_id": int(fold_id),
            "y_true_by_idx": yval.to_numpy(),
            "kind": kind,
            "k_top": int(k_top),
            "s_fixed": int(s_fixed),
            "allocation_method": allocation_method,
            "semantics": semantics,
            "target_class": target_class,
            "bg_size": bg_size,
            "topk_basis": topk_basis,
            "tol": float(tol),
            "strict": bool(strict),
            "random_state": random_state + fold_id,
            "epsilon": float(epsilon),
            "inner_cv_score": float(selected_model.cv_score) if selected_model.cv_score is not None else np.nan,
        }
        worker_jobs = _resolve_n_jobs(n_jobs, len(eval_idx))
        worker_tasks = [(int(idx), Xval_model.iloc[int(idx)]) for idx in eval_idx]
        if worker_jobs > 1:
            with ProcessPoolExecutor(
                max_workers=worker_jobs,
                initializer=_init_task6_instance_worker,
                initargs=(worker_state,),
            ) as executor:
                fold_rows = list(executor.map(_task6_instance_worker, worker_tasks))
        else:
            _init_task6_instance_worker(worker_state)
            fold_rows = [_task6_instance_worker(task) for task in worker_tasks]

        rows.extend(fold_rows)
        print(
            f"[fold {fold_id}] task 6 forest: {time.time() - start_time:.1f}s "
            f"(eval_instances={len(eval_idx)}, n_jobs={worker_jobs})"
        )

    if not rows:
        raise RuntimeError("No Task 6 rows were produced.")

    df_instances = pd.DataFrame(rows)
    metric_columns = [
        "mean_signed_width",
        "mean_absolute_width",
        "topk_mean_signed_width",
        "topk_mean_absolute_width",
        "topk_max_signed_width",
        "topk_max_absolute_width",
        "max_signed_width",
        "max_absolute_width",
        "mean_relative_sensitivity",
        "mean_sensitivity_aware_score",
        "certified_topk_set",
    ]
    correlation_rows: list[dict[str, Any]] = []
    fold_correlation_rows: list[dict[str, Any]] = []
    for metric in metric_columns:
        metric_values = df_instances[metric].astype(float).to_numpy()
        boundary_values = df_instances["boundary_distance"].astype(float).to_numpy()
        n_observations = finite_pair_count(boundary_values, metric_values)
        pearson_r = pearson_corr(boundary_values, metric_values)
        spearman_r = spearman_corr(boundary_values, metric_values)
        correlation_rows.append(
            {
                "dataset": dataset_name,
                "model_type": "forest",
                "kind": kind,
                "k_top": int(k_top),
                "s_fixed": int(s_fixed),
                "allocation_method": allocation_method,
                "semantics": semantics,
                "topk_basis": topk_basis,
                "metric": metric,
                "n_observations": n_observations,
                "pearson_margin_vs_metric": pearson_r,
                "pearson_p_value": correlation_p_value(pearson_r, n_observations),
                "spearman_margin_vs_metric": spearman_r,
                "spearman_p_value": correlation_p_value(spearman_r, n_observations),
            }
        )
        for fold_id, df_fold in df_instances.groupby("fold_id", sort=True):
            fold_metric_values = df_fold[metric].astype(float).to_numpy()
            fold_boundary_values = df_fold["boundary_distance"].astype(float).to_numpy()
            fold_n = finite_pair_count(fold_boundary_values, fold_metric_values)
            fold_pearson_r = pearson_corr(fold_boundary_values, fold_metric_values)
            fold_spearman_r = spearman_corr(fold_boundary_values, fold_metric_values)
            fold_correlation_rows.append(
                {
                    "dataset": dataset_name,
                    "model_type": "forest",
                    "kind": kind,
                    "k_top": int(k_top),
                    "s_fixed": int(s_fixed),
                    "allocation_method": allocation_method,
                    "semantics": semantics,
                    "topk_basis": topk_basis,
                    "fold_id": int(fold_id),
                    "metric": metric,
                    "n_observations": fold_n,
                    "pearson_margin_vs_metric": fold_pearson_r,
                    "pearson_p_value": correlation_p_value(fold_pearson_r, fold_n),
                    "spearman_margin_vs_metric": fold_spearman_r,
                    "spearman_p_value": correlation_p_value(fold_spearman_r, fold_n),
                }
            )
    df_correlations = pd.DataFrame(correlation_rows)
    df_fold_correlations = pd.DataFrame(fold_correlation_rows)
    df_fold_correlation_summary = (
        df_fold_correlations
        .groupby("metric", as_index=False)
        .agg(
            n_folds=("fold_id", "nunique"),
            pearson_fold_mean=("pearson_margin_vs_metric", "mean"),
            spearman_fold_mean=("spearman_margin_vs_metric", "mean"),
        )
    )

    df_boundary_distance_summary = _task6_boundary_distance_summary(df_instances)
    df_instances["boundary_group"] = _task6_boundary_group_labels(df_instances["boundary_distance"])
    df_feature_widths, df_feature_summary, df_feature_correlations = _task6_feature_width_tables(
        df_instances,
        [str(column) for column in Xtr_model.columns],
    )
    df_group_summary = (
        df_instances
        .groupby(["dataset", "model_type", "boundary_group"], as_index=False)
        .agg(
            n_instances=("instance_id", "count"),
            mean_boundary_distance=("boundary_distance", "mean"),
            mean_probability_margin=("probability_margin", "mean"),
            mean_signed_width=("mean_signed_width", "mean"),
            mean_absolute_width=("mean_absolute_width", "mean"),
            topk_mean_signed_width=("topk_mean_signed_width", "mean"),
            topk_mean_absolute_width=("topk_mean_absolute_width", "mean"),
            mean_relative_sensitivity=("mean_relative_sensitivity", "mean"),
            mean_sensitivity_aware_score=("mean_sensitivity_aware_score", "mean"),
            certified_topk_rate=("certified_topk_set", "mean"),
        )
    )
    df_overall_summary = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "model_type": "forest",
                "kind": kind,
                "k_top": int(k_top),
                "s_fixed": int(s_fixed),
                "allocation_method": allocation_method,
                "semantics": semantics,
                "topk_basis": topk_basis,
                "n_instances": int(len(df_instances)),
                "mean_boundary_distance": float(df_instances["boundary_distance"].mean()),
                "mean_probability_margin": float(df_instances["probability_margin"].mean()),
                "mean_signed_width": float(df_instances["mean_signed_width"].mean()),
                "mean_absolute_width": float(df_instances["mean_absolute_width"].mean()),
                "topk_mean_signed_width": float(df_instances["topk_mean_signed_width"].mean()),
                "topk_mean_absolute_width": float(df_instances["topk_mean_absolute_width"].mean()),
                "mean_relative_sensitivity": float(df_instances["mean_relative_sensitivity"].mean()),
                "mean_sensitivity_aware_score": float(df_instances["mean_sensitivity_aware_score"].mean()),
                "certified_topk_rate": float(df_instances["certified_topk_set"].mean()),
            }
        ]
    )

    if save_dir is not None:
        task_save_dir = f"{save_dir}/task6/{dataset_name}"
        os.makedirs(task_save_dir, exist_ok=True)
        stem = (
            f"{dataset_name}_forest_{kind}_k={k_top}_s={s_fixed}_cv={cv}_"
            f"{semantics}_{allocation_method}_topk={topk_basis}_task6"
        )
        df_instances.to_csv(f"{task_save_dir}/task6_boundary_instances_{stem}.csv", index=False)
        df_correlations.to_csv(f"{task_save_dir}/task6_boundary_correlations_{stem}.csv", index=False)
        df_fold_correlations.to_csv(f"{task_save_dir}/task6_boundary_fold_correlations_{stem}.csv", index=False)
        df_fold_correlation_summary.to_csv(f"{task_save_dir}/task6_boundary_fold_correlation_summary_{stem}.csv", index=False)
        df_boundary_distance_summary.to_csv(f"{task_save_dir}/task6_boundary_distance_summary_{stem}.csv", index=False)
        df_feature_widths.to_csv(f"{task_save_dir}/task6_boundary_feature_widths_{stem}.csv", index=False)
        df_feature_summary.to_csv(f"{task_save_dir}/task6_boundary_feature_summary_{stem}.csv", index=False)
        df_feature_correlations.to_csv(f"{task_save_dir}/task6_boundary_feature_correlations_{stem}.csv", index=False)
        df_group_summary.to_csv(f"{task_save_dir}/task6_boundary_groups_{stem}.csv", index=False)
        df_overall_summary.to_csv(f"{task_save_dir}/task6_boundary_summary_{stem}.csv", index=False)

    return {
        "instances": df_instances,
        "correlations": df_correlations,
        "fold_correlations": df_fold_correlations,
        "fold_correlation_summary": df_fold_correlation_summary,
        "distance_summary": df_boundary_distance_summary,
        "feature_widths": df_feature_widths,
        "feature_summary": df_feature_summary,
        "feature_correlations": df_feature_correlations,
        "group_summary": df_group_summary,
        "summary": df_overall_summary,
    }


# ----------------------------
# CLI
# ----------------------------
def _parse_datasets(raw: str) -> tuple[str, ...]:
    datasets = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not datasets:
        raise ValueError("At least one dataset name is required.")
    return datasets


def _print_frame(title: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        print(f"\n{title}: empty")
        return
    print(f"\n{title}")
    print(frame.to_string(index=False))


def run_task1_interval_rank(config: ExperimentConfig, *, model_type: str = "forest") -> dict[str, dict[str, pd.DataFrame]]:
    outputs: dict[str, dict[str, pd.DataFrame]] = {}
    for dataset_name in config.datasets:
        print(f"\n=== Task 1 interval-rank | {dataset_name} | {model_type} ===")
        result = task1_bootstrap_interval_rank_correlation_cv(
            dataset_name=dataset_name,
            model_type=model_type,
            kind=config.kind,
            s_fixed=config.s_fixed,
            cv=config.cv,
            n_bootstrap=config.n_bootstrap,
            semantics=config.semantics,
            bg_size=config.bg_size,
            target_class=config.target_class,
            allocation_method=config.allocation_method,
            random_state=config.random_state,
            n_eval=config.n_eval,
            inner_cv=config.inner_cv,
            scoring=config.scoring,
            max_tree_leaf_nodes=config.max_tree_leaf_nodes,
            forest_n_estimators=config.forest_n_estimators,
            n_jobs=config.n_jobs,
            save_dir=config.save_dir,
        )
        outputs[dataset_name] = result
        _print_frame("Task 1 summary", result["summary"])
    return outputs


def run_task6_forest(config: ExperimentConfig) -> dict[str, dict[str, object]]:
    outputs: dict[str, dict[str, object]] = {}
    for dataset_name in config.datasets:
        print(f"\n=== Task 6 boundary distance | {dataset_name} | forest ===")
        result = task6_boundary_distance_interval_width_forest_cv(
            dataset_name=dataset_name,
            kind=config.kind,
            k_top=config.k_top,
            s_fixed=config.s_fixed,
            cv=config.cv,
            semantics=config.semantics,
            bg_size=config.bg_size,
            target_class=config.target_class,
            allocation_method=config.allocation_method,
            topk_basis=config.topk_basis,
            tol=config.tol,
            random_state=config.random_state,
            n_eval=config.n_eval,
            inner_cv=config.inner_cv,
            scoring=config.scoring,
            max_tree_leaf_nodes=config.max_tree_leaf_nodes,
            n_jobs=config.n_jobs,
            save_dir=config.save_dir,
            make_plots=config.make_plots,
        )
        outputs[dataset_name] = result
        _print_frame("Task 6 summary", result["summary"])
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submission experiments: Task 1 interval-rank and forest-only Task 6."
    )
    parser.add_argument("--task", choices=["task1_interval_rank", "task6_forest", "all"], default="all")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--task1-model-type", choices=["tree", "forest"], default="forest")
    parser.add_argument("--kind", choices=["shapley", "banzhaf"], default="shapley")
    parser.add_argument("--s-fixed", type=int, default=1)
    parser.add_argument("--k-top", type=int, default=3)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--semantics", choices=["interventional"], default="interventional")
    parser.add_argument("--allocation-method", choices=["greedy", "multinomial"], default="greedy")
    parser.add_argument("--topk-basis", choices=["absolute", "signed"], default="absolute")
    parser.add_argument("--bg-size", type=int, default=100)
    parser.add_argument("--target-class", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument("--inner-cv", type=int, default=3)
    parser.add_argument("--scoring", default="balanced_accuracy")
    parser.add_argument("--max-tree-leaf-nodes", type=int, default=8)
    parser.add_argument("--forest-n-estimators", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--save-dir", default="results_submission")
    parser.add_argument("--make-plots", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        datasets=_parse_datasets(args.datasets),
        kind=args.kind,
        s_fixed=args.s_fixed,
        k_top=args.k_top,
        cv=args.cv,
        n_bootstrap=args.n_bootstrap,
        semantics=args.semantics,
        allocation_method=args.allocation_method,
        topk_basis=args.topk_basis,
        bg_size=args.bg_size,
        target_class=args.target_class,
        tol=args.tol,
        random_state=args.random_state,
        n_eval=args.n_eval,
        inner_cv=args.inner_cv,
        scoring=args.scoring,
        max_tree_leaf_nodes=args.max_tree_leaf_nodes,
        forest_n_estimators=args.forest_n_estimators,
        n_jobs=args.n_jobs,
        save_dir=args.save_dir,
        make_plots=args.make_plots,
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    start = time.time()

    if args.task in {"task1_interval_rank", "all"}:
        run_task1_interval_rank(config, model_type=args.task1_model_type)
    if args.task in {"task6_forest", "all"}:
        run_task6_forest(config)

    print(f"\nDone in {time.time() - start:.1f}s. Results saved under {config.save_dir}/")


if __name__ == "__main__":
    main()
