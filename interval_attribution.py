from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


AttributionKind = Literal["shapley", "banzhaf"]


@dataclass(frozen=True)
class TreeStruct:
    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    cover: np.ndarray
    is_leaf: np.ndarray
    leaf_id_of_node: np.ndarray
    n_leaves: int


@dataclass
class AttributionResult:
    values: np.ndarray
    order: list[int]
    feature_ids: list[int]
    metadata: dict[str, Any]


@dataclass
class AttributionIntervalResult:
    signed_lower: np.ndarray
    signed_upper: np.ndarray
    absolute_lower: np.ndarray
    absolute_upper: np.ndarray
    metadata: dict[str, Any]


# ----------------------------
# Tree Extraction
# ----------------------------
def extract_sklearn_tree(dt) -> TreeStruct:
    tree = dt.tree_
    children_left = tree.children_left.astype(int, copy=False)
    children_right = tree.children_right.astype(int, copy=False)
    feature = tree.feature.astype(int, copy=False)
    threshold = tree.threshold.astype(float, copy=False)
    cover = getattr(tree, "weighted_n_node_samples", None)
    if cover is None:
        cover = tree.n_node_samples
    cover = np.asarray(cover, dtype=float)

    is_leaf = (children_left == -1) & (children_right == -1)
    leaf_id_of_node = np.full_like(feature, fill_value=-1, dtype=int)
    for leaf_id, node in enumerate(np.where(is_leaf)[0]):
        leaf_id_of_node[node] = leaf_id

    return TreeStruct(
        children_left=children_left,
        children_right=children_right,
        feature=feature,
        threshold=threshold,
        cover=cover,
        is_leaf=is_leaf,
        leaf_id_of_node=leaf_id_of_node,
        n_leaves=int(is_leaf.sum()),
    )


def leaf_class_counts(dt, tree: TreeStruct) -> np.ndarray:
    value = np.asarray(dt.tree_.value, dtype=float)
    leaf_totals = getattr(dt.tree_, "weighted_n_node_samples", None)
    if leaf_totals is None:
        leaf_totals = dt.tree_.n_node_samples
    leaf_totals = np.asarray(leaf_totals, dtype=float)

    n_classes = value.shape[2]
    counts = np.zeros((tree.n_leaves, n_classes), dtype=float)
    for node in np.where(tree.is_leaf)[0]:
        row = value[node, 0, :].astype(float, copy=False)
        row_sum = float(row.sum())
        total = float(leaf_totals[node])
        if row_sum > 0.0 and not np.isclose(row_sum, total):
            row = row * (total / row_sum)
        counts[tree.leaf_id_of_node[node], :] = row
    return counts


def features_used_in_tree(tree: TreeStruct) -> list[int]:
    return sorted(set(int(feature_id) for feature_id in tree.feature[~tree.is_leaf] if feature_id >= 0))


# ----------------------------
# Coalition Routing
# ----------------------------
def _safe_branch_weights(tree: TreeStruct, node: int) -> tuple[float, float]:
    left = int(tree.children_left[node])
    right = int(tree.children_right[node])
    left_cover = float(tree.cover[left])
    right_cover = float(tree.cover[right])
    total = left_cover + right_cover
    if total <= 0.0:
        return 0.5, 0.5
    return left_cover / total, right_cover / total


def q_vector_path_dependent(
    tree: TreeStruct,
    x_star,
    known_features: set[int],
    node: int = 0,
) -> np.ndarray:
    if tree.is_leaf[node]:
        q = np.zeros(tree.n_leaves, dtype=float)
        q[tree.leaf_id_of_node[node]] = 1.0
        return q

    feature_id = int(tree.feature[node])
    left = int(tree.children_left[node])
    right = int(tree.children_right[node])

    if feature_id in known_features:
        x_value = x_star.iloc[feature_id] if hasattr(x_star, "iloc") else x_star[feature_id]
        next_node = left if x_value <= tree.threshold[node] else right
        return q_vector_path_dependent(tree, x_star, known_features, node=next_node)

    p_left, p_right = _safe_branch_weights(tree, node)
    return (
        p_left * q_vector_path_dependent(tree, x_star, known_features, node=left)
        + p_right * q_vector_path_dependent(tree, x_star, known_features, node=right)
    )


def leaf_id_for_x(tree: TreeStruct, x, node: int = 0) -> int:
    x_np = x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)
    while not tree.is_leaf[node]:
        feature_id = int(tree.feature[node])
        node = (
            int(tree.children_left[node])
            if x_np[feature_id] <= tree.threshold[node]
            else int(tree.children_right[node])
        )
    return int(tree.leaf_id_of_node[node])


def q_vector_interventional_empirical(
    tree: TreeStruct,
    x_star,
    known_features: set[int],
    X_bg,
    bg_weights: np.ndarray | None = None,
) -> np.ndarray:
    x_star_np = x_star.to_numpy() if hasattr(x_star, "to_numpy") else np.asarray(x_star)
    X_bg_np = X_bg.to_numpy() if hasattr(X_bg, "to_numpy") else np.asarray(X_bg)
    if bg_weights is None:
        bg_weights = np.ones(X_bg_np.shape[0], dtype=float)

    counts = np.zeros(tree.n_leaves, dtype=float)
    known_idx = np.array(sorted(known_features), dtype=int) if known_features else np.array([], dtype=int)
    for row_id in range(X_bg_np.shape[0]):
        z = X_bg_np[row_id].copy()
        if known_idx.size:
            z[known_idx] = x_star_np[known_idx]
        counts[leaf_id_for_x(tree, z)] += bg_weights[row_id]

    total = counts.sum()
    return counts / total if total > 0.0 else counts


def precompute_q_by_mask(
    tree: TreeStruct,
    x_star,
    feature_ids: Sequence[int],
    *,
    semantics: str,
    X_bg=None,
) -> np.ndarray:
    m = len(feature_ids)
    n_masks = 1 << m
    n_cells = n_masks * int(tree.n_leaves)
    if n_cells > 50_000_000:
        raise ValueError(
            "Attribution coalition cache is too large "
            f"({n_masks} masks x {tree.n_leaves} leaves). "
            "Use a smaller tree, e.g. lower max_tree_leaf_nodes."
        )

    Q = np.zeros((n_masks, tree.n_leaves), dtype=float)
    feature_ids = list(feature_ids)
    for mask in range(n_masks):
        known = {feature_ids[i] for i in range(m) if (mask >> i) & 1}
        if semantics == "interventional":
            Q[mask, :] = q_vector_interventional_empirical(tree, x_star, known, X_bg=X_bg)
        elif semantics == "path_dependent":
            Q[mask, :] = q_vector_path_dependent(tree, x_star, known)
        else:
            raise ValueError(f"Invalid semantics={semantics!r}.")

    denom = Q.sum(axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return Q / denom


# ----------------------------
# Nominal Attribution
# ----------------------------
def _shapley_weight(m: int, k: int) -> float:
    return math.factorial(k) * math.factorial(m - k - 1) / math.factorial(m)


def coefficient_matrix_C(Q: np.ndarray, kind: AttributionKind = "shapley") -> np.ndarray:
    n_masks, n_leaves = Q.shape
    m = int(round(math.log2(n_masks)))
    if (1 << m) != n_masks:
        raise ValueError("Q must have 2^m rows.")

    C = np.zeros((m, n_leaves), dtype=float)
    if kind == "banzhaf":
        coef = 1.0 / (2 ** (m - 1))
        for i in range(m):
            bit = 1 << i
            for mask in range(n_masks):
                if not (mask & bit):
                    C[i, :] += coef * (Q[mask | bit, :] - Q[mask, :])
        return C

    if kind == "shapley":
        popcount = np.array([int(mask).bit_count() for mask in range(n_masks)], dtype=int)
        weights = np.array([_shapley_weight(m, k) for k in range(m)], dtype=float)
        for i in range(m):
            bit = 1 << i
            for mask in range(n_masks):
                if not (mask & bit):
                    C[i, :] += weights[popcount[mask]] * (Q[mask | bit, :] - Q[mask, :])
        return C

    raise ValueError(f"Invalid kind={kind!r}.")


def tree_shap_coefficient_matrix(
    tree: TreeStruct,
    x_star,
    feature_ids: Sequence[int],
    *,
    semantics: str,
    X_bg=None,
) -> np.ndarray:
    """
    Compute the leafwise Shapley coefficient matrix without coalition masks.

    A vector-valued copy of the tree is built whose output at leaf ``ell`` is
    the one-hot vector ``e_ell``. The coalition value for output ``ell`` is
    therefore the leaf reach probability ``q_ell(S)``, so TreeSHAP output
    ``(i, ell)`` is exactly the coefficient ``a_{i, ell}`` in

        phi_i(w) = sum_ell a_{i, ell} w_ell.

    This is an exact TreeSHAP reduction, not an approximation. In particular,
    it avoids constructing the exponential ``2**m x n_leaves`` coalition
    cache used by :func:`precompute_q_by_mask`.
    """
    if semantics not in {"interventional", "path_dependent"}:
        raise ValueError(f"Invalid semantics={semantics!r}.")

    feature_ids = [int(feature_id) for feature_id in feature_ids]
    if not feature_ids:
        return np.zeros((0, tree.n_leaves), dtype=float)

    try:
        import shap
    except ImportError as exc:  # pragma: no cover - depends on environment setup
        raise ImportError(
            "TreeSHAP coefficient computation requires the 'shap' package. "
            "Install the dependencies from requirements.txt."
        ) from exc

    n_nodes = len(tree.feature)
    leaf_values = np.zeros((n_nodes, tree.n_leaves), dtype=float)
    for node in np.flatnonzero(tree.is_leaf):
        leaf_values[int(node), int(tree.leaf_id_of_node[int(node)])] = 1.0

    # SHAP's documented custom-tree representation. Copies are important:
    # TreeExplainer may recompute node weights from an interventional
    # background, and that must not mutate the TreeStruct used elsewhere.
    shap_tree = {
        "children_left": np.asarray(tree.children_left, dtype=np.int32).copy(),
        "children_right": np.asarray(tree.children_right, dtype=np.int32).copy(),
        # The current routing code sends NaNs to the right because ``NaN <=
        # threshold`` is false. Encoded experiment inputs should not contain
        # NaNs, but matching the default branch keeps both representations
        # consistent if one is encountered.
        "children_default": np.asarray(tree.children_right, dtype=np.int32).copy(),
        "features": np.asarray(tree.feature, dtype=np.int32).copy(),
        "thresholds": np.asarray(tree.threshold, dtype=float).copy(),
        "values": leaf_values,
        "node_sample_weight": np.asarray(tree.cover, dtype=float).copy(),
    }
    shap_model = {
        "trees": [shap_tree],
        "internal_dtype": np.float64,
        "input_dtype": np.float64,
        "tree_output": "raw_value",
        "base_offset": np.zeros(tree.n_leaves, dtype=float),
    }

    x_np = x_star.to_numpy() if hasattr(x_star, "to_numpy") else np.asarray(x_star)
    x_np = np.asarray(x_np, dtype=float).reshape(1, -1)

    if semantics == "interventional":
        if X_bg is None:
            raise ValueError("X_bg is required for interventional TreeSHAP coefficients.")
        X_bg_np = X_bg.to_numpy() if hasattr(X_bg, "to_numpy") else np.asarray(X_bg)
        X_bg_np = np.asarray(X_bg_np, dtype=float)
        if X_bg_np.ndim != 2 or X_bg_np.shape[0] == 0:
            raise ValueError("X_bg must be a non-empty two-dimensional background array.")
        # Passing a raw array lets SHAP's default Independent masker silently
        # summarize backgrounds larger than 100 rows. An explicit masker
        # preserves the exact empirical distribution used by the old routine.
        masker = shap.maskers.Independent(X_bg_np, max_samples=X_bg_np.shape[0])
        explainer = shap.TreeExplainer(
            shap_model,
            data=masker,
            feature_perturbation="interventional",
            model_output="raw",
        )
    else:
        # Do not pass background data here: path-dependent routing must use the
        # training covers stored in ``node_sample_weight``.
        explainer = shap.TreeExplainer(
            shap_model,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )

    explanation = explainer(x_np, check_additivity=False)
    values = np.asarray(explanation.values, dtype=float)
    expected_shape = (1, x_np.shape[1], tree.n_leaves)
    if values.shape != expected_shape:
        raise RuntimeError(
            "Unexpected multi-output TreeSHAP result shape: "
            f"expected {expected_shape}, got {values.shape}."
        )

    coefficients = values[0, np.asarray(feature_ids, dtype=int), :]
    if not np.isfinite(coefficients).all():
        raise RuntimeError("TreeSHAP returned non-finite leafwise coefficients.")
    return coefficients


def tree_attribution_coefficient_matrix(
    tree: TreeStruct,
    x_star,
    feature_ids: Sequence[int],
    *,
    kind: AttributionKind = "shapley",
    semantics: str,
    X_bg=None,
) -> np.ndarray:
    """
    Dispatch exact tree coefficient construction by attribution kind.

    Shapley coefficients use the polynomial TreeSHAP reduction. Banzhaf
    coefficients retain exact coalition enumeration until a Banzhaf-specific
    tree dynamic program is implemented.
    """
    if kind == "shapley":
        return tree_shap_coefficient_matrix(
            tree,
            x_star,
            feature_ids,
            semantics=semantics,
            X_bg=X_bg,
        )
    if kind == "banzhaf":
        Q = precompute_q_by_mask(
            tree,
            x_star,
            feature_ids,
            semantics=semantics,
            X_bg=X_bg,
        )
        return coefficient_matrix_C(Q, kind=kind)
    raise ValueError(f"Invalid kind={kind!r}.")


def point_leaf_probabilities(leaf_counts: np.ndarray, target_class: int = 1) -> np.ndarray:
    counts = np.asarray(leaf_counts, dtype=float)
    numerator = counts[:, target_class]
    denominator = counts.sum(axis=1)
    return np.where(denominator > 0.0, numerator / denominator, 0.5)


def ranking_from_point_attributions(C: np.ndarray, w_hat: np.ndarray, descending: bool = True) -> list[int]:
    scores = C @ w_hat
    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    return order.tolist()


def _stable_order_from_scores(scores: np.ndarray, descending: bool = True) -> list[int]:
    scores = np.asarray(scores, dtype=float)
    feature_ids = np.arange(scores.shape[0], dtype=int)
    if descending:
        order = np.lexsort((feature_ids, -scores))
    else:
        order = np.lexsort((feature_ids, scores))
    return order.tolist()


# ----------------------------
# Leaf uncertainty and intervals
# ----------------------------
def idm_leaf_bounds_binary(
    leaf_counts: np.ndarray,
    *,
    target_class: int = 1,
    s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(leaf_counts, dtype=float)
    numerator = counts[:, target_class]
    denominator = counts.sum(axis=1) + s
    w_low = np.where(denominator > 0.0, numerator / denominator, 0.0)
    w_high = np.where(denominator > 0.0, (numerator + s) / denominator, 1.0)
    return np.clip(w_low, 0.0, 1.0), np.clip(w_high, 0.0, 1.0)


def leaf_prob(
    leaf_counts: np.ndarray,
    mode: Literal["uniform", "mass"] = "mass",
) -> np.ndarray:
    n_leaves = leaf_counts.shape[0]
    if mode == "uniform":
        return np.full(n_leaves, 1.0 / n_leaves, dtype=float)
    if mode == "mass":
        mass = leaf_counts.sum(axis=1).astype(float)
        total = mass.sum()
        return mass / total if total > 0.0 else np.full(n_leaves, 1.0 / n_leaves, dtype=float)
    raise ValueError(f"Invalid mode={mode!r}.")


def _binom_pmf_matrix(s: int, p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    r = np.arange(s + 1, dtype=float)[None, :]
    log_comb = (
        math.lgamma(s + 1.0)
        - np.vectorize(math.lgamma)(r + 1.0)
        - np.vectorize(math.lgamma)((s - r) + 1.0)
    )
    p = p[:, None]
    logp = np.where(p > 0.0, np.log(p), -np.inf)
    log1mp = np.where(p < 1.0, np.log1p(-p), -np.inf)
    pmf = np.exp(log_comb + r * logp + (s - r) * log1mp)
    flat = p[:, 0]
    pmf[flat == 0.0, :] = 0.0
    pmf[flat == 0.0, 0] = 1.0
    pmf[flat == 1.0, :] = 0.0
    pmf[flat == 1.0, s] = 1.0
    return pmf


def _expected_fraction_from_binomial(
    totals: np.ndarray,
    *,
    s_total: int,
    leaf_probs: np.ndarray,
) -> np.ndarray:
    totals = np.asarray(totals, dtype=float)
    r = np.arange(int(s_total) + 1, dtype=float)[None, :]
    denom = totals[:, None] + r
    frac = np.divide(
        r,
        denom,
        out=np.zeros((totals.shape[0], int(s_total) + 1), dtype=float),
        where=denom > 0.0,
    )
    pmf = _binom_pmf_matrix(int(s_total), np.asarray(leaf_probs, dtype=float))
    return (frac * pmf).sum(axis=1)


def attribution_shrinkage_components(
    C: np.ndarray,
    leaf_counts: np.ndarray,
    *,
    target_class: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    C = np.asarray(C, dtype=float)
    counts = np.asarray(leaf_counts, dtype=float)
    w_hat = point_leaf_probabilities(counts, target_class=target_class)
    totals = counts.sum(axis=1)
    c_pos = np.maximum(C, 0.0)
    c_neg = np.maximum(-C, 0.0)
    nominal = C @ w_hat
    alpha_minus = c_pos * w_hat[None, :] + c_neg * (1.0 - w_hat[None, :])
    alpha_plus = c_pos * (1.0 - w_hat[None, :]) + c_neg * w_hat[None, :]
    return nominal, alpha_minus, alpha_plus, totals


def attribution_endpoints_from_shrinkage(
    nominal: np.ndarray,
    alpha_minus: np.ndarray,
    alpha_plus: np.ndarray,
    fractions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fractions = np.asarray(fractions, dtype=float)
    return nominal - alpha_minus @ fractions, nominal + alpha_plus @ fractions


def expected_attribution_endpoints_random_placement(
    C: np.ndarray,
    leaf_counts: np.ndarray,
    *,
    s_total: int,
    pi: Optional[np.ndarray] = None,
    pi_mode: Literal["uniform", "mass"] = "mass",
    target_class: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if pi is None:
        pi = leaf_prob(leaf_counts, mode=pi_mode)
    nominal, alpha_minus, alpha_plus, totals = attribution_shrinkage_components(
        C,
        leaf_counts,
        target_class=target_class,
    )
    expected_fraction = _expected_fraction_from_binomial(
        totals,
        s_total=s_total,
        leaf_probs=np.asarray(pi, dtype=float),
    )
    return attribution_endpoints_from_shrinkage(nominal, alpha_minus, alpha_plus, expected_fraction)


def box_min_dot(u: np.ndarray, w_low: np.ndarray, w_high: np.ndarray) -> float:
    return float(np.sum(np.where(u >= 0.0, u * w_low, u * w_high)))


def attribution_bounds(C: np.ndarray, w_low: np.ndarray, w_high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([box_min_dot(C[i], w_low, w_high) for i in range(C.shape[0])], dtype=float)
    upper = np.array(
        [float(np.sum(np.where(C[i] >= 0.0, C[i] * w_high, C[i] * w_low))) for i in range(C.shape[0])],
        dtype=float,
    )
    return lower, upper


def abs_bounds_from_signed_bounds(
    signed_lower: np.ndarray,
    signed_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    signed_lower = np.asarray(signed_lower, dtype=float)
    signed_upper = np.asarray(signed_upper, dtype=float)
    crosses_zero = (signed_lower <= 0.0) & (signed_upper >= 0.0)
    absolute_lower = np.where(crosses_zero, 0.0, np.minimum(np.abs(signed_lower), np.abs(signed_upper)))
    absolute_upper = np.maximum(np.abs(signed_lower), np.abs(signed_upper))
    return absolute_lower, absolute_upper


def _fraction_from_additions(totals: np.ndarray, additions: np.ndarray) -> np.ndarray:
    totals = np.asarray(totals, dtype=float)
    additions = np.asarray(additions, dtype=float)
    denom = totals + additions
    return np.divide(additions, denom, out=np.zeros_like(additions, dtype=float), where=denom > 0.0)


def _single_tree_greedy_signed_bounds(
    C: np.ndarray,
    leaf_counts: np.ndarray,
    *,
    s_values: Sequence[int],
    target_class: int = 1,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    s_values = sorted(set(int(s) for s in s_values))
    if not s_values:
        return {}

    nominal, alpha_minus, alpha_plus, totals = attribution_shrinkage_components(
        C,
        leaf_counts,
        target_class=target_class,
    )

    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if 0 in s_values:
        results[0] = (nominal.copy(), nominal.copy())

    n_features, n_leaves = alpha_minus.shape
    row_idx = np.arange(n_features, dtype=int)
    totals_row = totals[None, :]
    lower_additions = np.zeros((n_features, n_leaves), dtype=int)
    upper_additions = np.zeros((n_features, n_leaves), dtype=int)
    lower_correction = np.zeros(n_features, dtype=float)
    upper_correction = np.zeros(n_features, dtype=float)

    for step in range(1, s_values[-1] + 1):
        lower_r = lower_additions + 1.0
        upper_r = upper_additions + 1.0
        lower_gain = alpha_minus * totals_row / ((totals_row + lower_r - 1.0) * (totals_row + lower_r))
        upper_gain = alpha_plus * totals_row / ((totals_row + upper_r - 1.0) * (totals_row + upper_r))

        best_lower_leaf = np.argmax(lower_gain, axis=1)
        best_upper_leaf = np.argmax(upper_gain, axis=1)
        lower_correction += lower_gain[row_idx, best_lower_leaf]
        upper_correction += upper_gain[row_idx, best_upper_leaf]
        lower_additions[row_idx, best_lower_leaf] += 1
        upper_additions[row_idx, best_upper_leaf] += 1

        if step in s_values:
            results[step] = (nominal - lower_correction, nominal + upper_correction)
    return results


def topk_set_endpoint_score(
    interval_lower: np.ndarray,
    interval_upper: np.ndarray,
    topk_set: Sequence[int],
) -> float:
    n_features = interval_lower.shape[0]
    topk = set(int(i) for i in topk_set)
    if not topk:
        return float("inf")
    margin = float("inf")
    for i in topk:
        for j in range(n_features):
            if j in topk:
                continue
            margin = min(margin, float(interval_lower[i] - interval_upper[j]))
            if margin < 0.0:
                return margin
    return margin


# ----------------------------
# Payload Preparation
# ----------------------------
def _background_sample(X_reference, bg_size: int, random_state: int):
    if len(X_reference) <= bg_size:
        return X_reference
    return X_reference.sample(n=bg_size, random_state=random_state)


def _full_length_matrix(C_local: np.ndarray, feature_ids: Sequence[int], n_features: int) -> np.ndarray:
    C_full = np.zeros((n_features, C_local.shape[1]), dtype=float)
    C_full[np.asarray(feature_ids, dtype=int), :] = C_local
    return C_full


def _prepare_single_tree_instance_payload(
    estimator,
    X_reference,
    x_star,
    kind: str = "shapley",
    semantics: str = "interventional",
    target_class: int = 1,
    bg_size: int = 200,
    random_state: int = 0,
) -> Optional[dict[str, Any]]:
    tree_struct = extract_sklearn_tree(estimator)
    feature_ids = features_used_in_tree(tree_struct)
    if not feature_ids:
        return None

    X_bg = _background_sample(X_reference, bg_size=bg_size, random_state=random_state)
    C_local = tree_attribution_coefficient_matrix(
        tree_struct,
        x_star,
        feature_ids,
        kind=kind,
        semantics=semantics,
        X_bg=X_bg,
    )
    counts = leaf_class_counts(estimator, tree_struct)
    w_hat = point_leaf_probabilities(counts, target_class=target_class)
    psi_local = C_local @ w_hat
    psi_full = np.zeros(X_reference.shape[1], dtype=float)
    psi_full[np.asarray(feature_ids, dtype=int)] = psi_local

    return {
        "tree": tree_struct,
        "counts": counts,
        "U": list(feature_ids),
        "C_local": C_local,
        "w_hat": w_hat,
        "nominal_local": psi_local,
        "psi_local": psi_local,
        "psi_full": psi_full,
        "order_local": ranking_from_point_attributions(C_local, w_hat, descending=True),
        "order_full": _stable_order_from_scores(psi_full, descending=True),
    }


def _prepare_forest_tree_payloads(estimator) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for dt in estimator.estimators_:
        tree_struct = extract_sklearn_tree(dt)
        feature_ids = features_used_in_tree(tree_struct)
        if not feature_ids:
            continue
        payloads.append(
            {
                "tree": tree_struct,
                "counts": leaf_class_counts(dt, tree_struct),
                "U": list(feature_ids),
            }
        )
    return payloads


def _prepare_rf_instance_payloads(
    tree_payloads: list[dict[str, Any]],
    *,
    x_star,
    X_bg,
    kind: str,
    semantics: str = "interventional",
) -> list[dict[str, Any]]:
    instance_payloads: list[dict[str, Any]] = []
    for payload in tree_payloads:
        C = tree_attribution_coefficient_matrix(
            payload["tree"],
            x_star,
            payload["U"],
            kind=kind,
            semantics=semantics,
            X_bg=X_bg,
        )
        nominal, alpha_minus, alpha_plus, leaf_totals = attribution_shrinkage_components(
            C,
            payload["counts"],
        )
        instance_payloads.append(
            {
                "C": C,
                "counts": payload["counts"],
                "U": payload["U"],
                "nominal": nominal,
                "alpha_minus": alpha_minus,
                "alpha_plus": alpha_plus,
                "leaf_totals": leaf_totals,
            }
        )
    return instance_payloads


# ----------------------------
# Random-forest Allocation
# ----------------------------
def _rf_global_multinomial_signed_endpoint_bounds_for_instance(
    instance_payloads: list[dict[str, Any]],
    *,
    n_features: int,
    s_values: Sequence[int],
    target_class: int = 1,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    s_values = sorted(set(int(s) for s in s_values))
    if not instance_payloads:
        zeros = np.zeros(n_features, dtype=float)
        return {int(s): (zeros.copy(), zeros.copy()) for s in s_values}

    n_trees = len(instance_payloads)
    signed_totals = {
        s: (np.zeros(n_features, dtype=float), np.zeros(n_features, dtype=float))
        for s in s_values
    }
    for payload in instance_payloads:
        feature_ids = np.asarray(payload["U"], dtype=int)
        leaf_probs = leaf_prob(payload["counts"], mode="mass")
        for s in s_values:
            expected_fraction = _expected_fraction_from_binomial(
                payload["leaf_totals"],
                s_total=int(s),
                leaf_probs=leaf_probs,
            )
            local_lower, local_upper = attribution_endpoints_from_shrinkage(
                payload["nominal"],
                payload["alpha_minus"],
                payload["alpha_plus"],
                expected_fraction,
            )
            signed_totals[s][0][feature_ids] += local_lower
            signed_totals[s][1][feature_ids] += local_upper

    return {s: (signed_totals[s][0] / n_trees, signed_totals[s][1] / n_trees) for s in s_values}


def _rf_global_greedy_signed_endpoint_bounds_for_instance(
    instance_payloads: list[dict[str, Any]],
    *,
    n_features: int,
    s_values: Sequence[int],
    target_class: int = 1,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    s_values = sorted(set(int(s) for s in s_values))
    if not instance_payloads:
        zeros = np.zeros(n_features, dtype=float)
        return {int(s): (zeros.copy(), zeros.copy()) for s in s_values}

    n_trees = len(instance_payloads)
    signed_totals = {
        s: (np.zeros(n_features, dtype=float), np.zeros(n_features, dtype=float))
        for s in s_values
    }
    for payload in instance_payloads:
        feature_ids = np.asarray(payload["U"], dtype=int)
        tree_bounds = _single_tree_greedy_signed_bounds(
            payload["C"],
            payload["counts"],
            s_values=s_values,
            target_class=target_class,
        )
        for s in s_values:
            local_lower, local_upper = tree_bounds[s]
            signed_totals[s][0][feature_ids] += local_lower
            signed_totals[s][1][feature_ids] += local_upper

    return {s: (signed_totals[s][0] / n_trees, signed_totals[s][1] / n_trees) for s in s_values}


def _rf_global_allocation_signed_endpoint_bounds_for_instance(
    tree_payloads: list[dict[str, Any]],
    *,
    n_features: int,
    x_star,
    X_bg,
    s_values: Sequence[int],
    allocation_method: Literal["multinomial", "greedy"],
    kind: str,
    semantics: str = "interventional",
    target_class: int = 1,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    instance_payloads = _prepare_rf_instance_payloads(
        tree_payloads,
        x_star=x_star,
        X_bg=X_bg,
        kind=kind,
        semantics=semantics,
    )
    if allocation_method == "multinomial":
        return _rf_global_multinomial_signed_endpoint_bounds_for_instance(
            instance_payloads,
            n_features=n_features,
            s_values=s_values,
            target_class=target_class,
        )
    if allocation_method == "greedy":
        return _rf_global_greedy_signed_endpoint_bounds_for_instance(
            instance_payloads,
            n_features=n_features,
            s_values=s_values,
            target_class=target_class,
        )
    raise ValueError(f"Invalid allocation_method={allocation_method!r}.")


# ----------------------------
# Public Attribution API
# ----------------------------
def compute_nominal_attribution(
    estimator,
    X_reference,
    x_star,
    *,
    kind: str = "shapley",
    semantics: str = "interventional",
    target_class: int = 1,
    bg_size: int = 200,
    random_state: int = 0,
) -> AttributionResult:
    if isinstance(estimator, DecisionTreeClassifier):
        payload = _prepare_single_tree_instance_payload(
            estimator,
            X_reference=X_reference,
            x_star=x_star,
            kind=kind,
            semantics=semantics,
            bg_size=bg_size,
            random_state=random_state,
            target_class=target_class,
        )
        if payload is None:
            values = np.zeros(X_reference.shape[1], dtype=float)
            return AttributionResult(values=values, order=list(range(len(values))), feature_ids=[], metadata={})
        return AttributionResult(
            values=payload["psi_full"],
            order=payload["order_full"],
            feature_ids=list(payload["U"]),
            metadata=payload,
        )

    if isinstance(estimator, RandomForestClassifier):
        tree_payloads = _prepare_forest_tree_payloads(estimator)
        X_bg = _background_sample(X_reference, bg_size=bg_size, random_state=random_state)
        n_features = X_reference.shape[1]
        values = np.zeros(n_features, dtype=float)

        for payload in tree_payloads:
            C_local = tree_attribution_coefficient_matrix(
                payload["tree"],
                x_star,
                payload["U"],
                kind=kind,
                semantics=semantics,
                X_bg=X_bg,
            )
            w_hat = point_leaf_probabilities(payload["counts"], target_class=target_class)
            values[np.asarray(payload["U"], dtype=int)] += C_local @ w_hat

        if tree_payloads:
            values /= len(tree_payloads)

        return AttributionResult(
            values=values,
            order=_stable_order_from_scores(values, descending=True),
            feature_ids=list(range(n_features)),
            metadata={"n_trees_used": len(tree_payloads)},
        )

    raise TypeError(f"Unsupported estimator type: {type(estimator)!r}")


def compute_interval_attribution(
    estimator,
    X_reference,
    x_star,
    *,
    s: int,
    kind: str = "shapley",
    semantics: str = "interventional",
    allocation: str = "greedy",
    target_class: int = 1,
    bg_size: int = 200,
    random_state: int = 0,
) -> AttributionIntervalResult:
    if isinstance(estimator, DecisionTreeClassifier):
        payload = _prepare_single_tree_instance_payload(
            estimator,
            X_reference=X_reference,
            x_star=x_star,
            kind=kind,
            semantics=semantics,
            bg_size=bg_size,
            random_state=random_state,
            target_class=target_class,
        )
        if payload is None:
            zeros = np.zeros(X_reference.shape[1], dtype=float)
            return AttributionIntervalResult(zeros, zeros, zeros, zeros, {})

        C_full = _full_length_matrix(payload["C_local"], payload["U"], X_reference.shape[1])
        counts = payload["counts"]
        if allocation == "greedy":
            signed_lower, signed_upper = _single_tree_greedy_signed_bounds(
                C_full,
                counts,
                s_values=[s],
                target_class=target_class,
            )[s]
        elif allocation == "multinomial":
            signed_lower, signed_upper = expected_attribution_endpoints_random_placement(
                C_full,
                counts,
                s_total=s,
                pi=leaf_prob(counts, mode="mass"),
                target_class=target_class,
            )
        elif allocation == "box":
            w_low, w_high = idm_leaf_bounds_binary(counts, target_class=target_class, s=s)
            signed_lower, signed_upper = attribution_bounds(C_full, w_low, w_high)
        else:
            raise ValueError(f"Invalid allocation={allocation!r}.")

        absolute_lower, absolute_upper = abs_bounds_from_signed_bounds(signed_lower, signed_upper)
        return AttributionIntervalResult(
            signed_lower=signed_lower,
            signed_upper=signed_upper,
            absolute_lower=absolute_lower,
            absolute_upper=absolute_upper,
            metadata=payload,
        )

    if isinstance(estimator, RandomForestClassifier):
        tree_payloads = _prepare_forest_tree_payloads(estimator)
        X_bg = _background_sample(X_reference, bg_size=bg_size, random_state=random_state)
        signed_lower, signed_upper = _rf_global_allocation_signed_endpoint_bounds_for_instance(
            tree_payloads,
            n_features=X_reference.shape[1],
            x_star=x_star,
            X_bg=X_bg,
            s_values=[s],
            allocation_method=allocation,
            kind=kind,
            semantics=semantics,
            target_class=target_class,
        )[s]
        absolute_lower, absolute_upper = abs_bounds_from_signed_bounds(signed_lower, signed_upper)
        return AttributionIntervalResult(
            signed_lower=signed_lower,
            signed_upper=signed_upper,
            absolute_lower=absolute_lower,
            absolute_upper=absolute_upper,
            metadata={"n_trees_used": len(tree_payloads), "semantics": semantics},
        )

    raise TypeError(f"Unsupported estimator type: {type(estimator)!r}")
