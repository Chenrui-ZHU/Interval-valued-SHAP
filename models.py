from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

import data


# ----------------------------
# Model Specs
# ----------------------------
@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any


@dataclass
class TrainedModel:
    name: str
    estimator: Any
    cv_score: Optional[float]
    params: Dict[str, Any]


# ----------------------------
# Training
# ----------------------------
def train_decision_tree(
    X_train,
    y_train,
    *,
    random_state: int = 0,
    **kwargs: Any,
) -> DecisionTreeClassifier:
    """
    Fit one decision tree classifier.
    """
    model = DecisionTreeClassifier(random_state=random_state, **kwargs)
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train,
    y_train,
    *,
    random_state: int = 0,
    **kwargs: Any,
) -> RandomForestClassifier:
    """
    Fit one random forest classifier.
    """
    model = RandomForestClassifier(random_state=random_state, **kwargs)
    model.fit(X_train, y_train)
    return model


def default_tree_specs(random_state: int = 0) -> list[ModelSpec]:
    """
    Small default inner-CV search space for decision trees.
    """
    specs: list[ModelSpec] = []
    for max_depth in [5, 7, 10, None]:
        for min_samples_leaf in [5, 10, 20]:
            specs.append(
                ModelSpec(
                    name=(
                        "decision_tree"
                        f"_depth={max_depth if max_depth is not None else 'none'}"
                        f"_leaf={min_samples_leaf}"
                    ),
                    estimator=DecisionTreeClassifier(
                        random_state=random_state,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                    ),
                )
            )
    return specs


def certifiable_tree_specs(
    random_state: int = 0,
    *,
    max_leaf_nodes: int = 16,
) -> list[ModelSpec]:
    """
    Decision-tree search space for exact interval attribution.

    Attribution enumerates coalitions over the distinct split features used by
    a tree. Limiting leaf nodes bounds the number of split nodes, and therefore
    the number of split features, by max_leaf_nodes - 1.
    """
    if max_leaf_nodes < 2:
        raise ValueError("max_leaf_nodes must be at least 2.")

    specs: list[ModelSpec] = []
    for min_samples_leaf in [5, 10, 20]:
        specs.append(
            ModelSpec(
                name=f"decision_tree_maxleaf={max_leaf_nodes}_leaf={min_samples_leaf}",
                estimator=DecisionTreeClassifier(
                    random_state=random_state,
                    max_leaf_nodes=max_leaf_nodes,
                    min_samples_leaf=min_samples_leaf,
                ),
            )
        )
    return specs


def default_model_specs(random_state: int = 0) -> list[ModelSpec]:
    """
    Small default model pool for binary tabular classification.
    """
    return [
        ModelSpec(
            name="decision_tree",
            estimator=DecisionTreeClassifier(
                random_state=random_state,
                min_samples_leaf=5,
            ),
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                random_state=random_state,
                n_estimators=200,
                min_samples_leaf=5,
                n_jobs=-1,
            ),
        ),
    ]


def default_random_forest_specs(
    random_state: int = 0,
    *,
    max_leaf_nodes: int | None = None,
) -> list[ModelSpec]:
    """
    Small default inner-CV search space for random forests.
    """
    specs: list[ModelSpec] = []
    for n_estimators in [100]:
        for max_depth in [5, 7, 10, None]:
            for min_samples_leaf in [5, 10, 20]:
                specs.append(
                    ModelSpec(
                        name=(
                            "random_forest"
                            f"_n={n_estimators}"
                            f"_depth={max_depth if max_depth is not None else 'none'}"
                            f"_leaf={min_samples_leaf}"
                        ),
                        estimator=RandomForestClassifier(
                            random_state=random_state,
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            max_leaf_nodes=max_leaf_nodes,
                            min_samples_leaf=min_samples_leaf,
                            n_jobs=-1,
                        ),
                    )
                )
    return specs


def certifiable_random_forest_specs(
    random_state: int = 0,
    *,
    max_leaf_nodes: int = 8,
    n_estimators: int = 100,
) -> list[ModelSpec]:
    """
    Random-forest search space for exact interval certification.

    The attribution code enumerates coalitions over the distinct split features
    used by each tree. Limiting leaf nodes bounds that count by
    max_leaf_nodes - 1 and prevents exponential coalition caches from exploding
    on wider datasets such as NHANES.
    """
    if max_leaf_nodes < 2:
        raise ValueError("max_leaf_nodes must be at least 2.")
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1.")

    specs: list[ModelSpec] = []
    for min_samples_leaf in [5, 10, 20]:
        specs.append(
            ModelSpec(
                name=(
                    "random_forest"
                    f"_n={n_estimators}"
                    f"_maxleaf={max_leaf_nodes}"
                    f"_leaf={min_samples_leaf}"
                ),
                estimator=RandomForestClassifier(
                    random_state=random_state,
                    n_estimators=n_estimators,
                    max_leaf_nodes=max_leaf_nodes,
                    min_samples_leaf=min_samples_leaf,
                    n_jobs=-1,
                ),
            )
        )
    return specs


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_classifier(estimator, X_test, y_test) -> Dict[str, float]:
    """
    Evaluate a fitted classifier on one holdout split.
    """
    y_pred = estimator.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
    }

    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X_test)[:, 1]
        metrics["auprc"] = float(average_precision_score(y_test, y_score))

    return metrics


# ----------------------------
# Model Selection
# ----------------------------
def select_preferred_model(
    X_train,
    y_train,
    *,
    cv: int = 3,
    scoring: str = "balanced_accuracy",
    random_state: int = 0,
    model_specs: Optional[Iterable[ModelSpec]] = None,
) -> TrainedModel:
    """
    Select the best model by mean CV score, then refit it on the full training set.
    """
    specs = list(model_specs or default_model_specs(random_state=random_state))
    if not specs:
        raise ValueError("model_specs must contain at least one candidate.")

    splitter = data.make_stratified_cv(cv=cv, random_state=random_state)

    best_name = ""
    best_estimator = None
    best_score = -np.inf
    best_params: Dict[str, Any] = {}

    for spec in specs:
        estimator = clone(spec.estimator)
        scores = cross_val_score(
            estimator,
            X_train,
            y_train,
            cv=splitter,
            scoring=scoring,
            n_jobs=None,
        )
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_name = spec.name
            best_estimator = estimator
            best_score = mean_score
            best_params = estimator.get_params(deep=False)

    assert best_estimator is not None
    best_estimator.fit(X_train, y_train)
    return TrainedModel(
        name=best_name,
        estimator=best_estimator,
        cv_score=best_score,
        params=best_params,
    )


def select_decision_tree_cv(
    X_train,
    y_train,
    *,
    cv: int = 3,
    scoring: str = "balanced_accuracy",
    random_state: int = 0,
    tree_specs: Optional[Iterable[ModelSpec]] = None,
) -> TrainedModel:
    """
    Select a decision tree by inner CV and refit it on the full training fold.
    """
    specs = list(tree_specs or default_tree_specs(random_state=random_state))
    specs = [spec for spec in specs if isinstance(spec.estimator, DecisionTreeClassifier)]
    if not specs:
        raise ValueError("tree_specs must contain at least one DecisionTreeClassifier candidate.")

    return select_preferred_model(
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        model_specs=specs,
    )


def select_random_forest_cv(
    X_train,
    y_train,
    *,
    cv: int = 3,
    scoring: str = "balanced_accuracy",
    random_state: int = 0,
    forest_specs: Optional[Iterable[ModelSpec]] = None,
) -> TrainedModel:
    """
    Select a random forest by inner CV and refit it on the full training fold.
    """
    specs = list(forest_specs or default_random_forest_specs(random_state=random_state))
    specs = [spec for spec in specs if isinstance(spec.estimator, RandomForestClassifier)]
    if not specs:
        raise ValueError("forest_specs must contain at least one RandomForestClassifier candidate.")

    return select_preferred_model(
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        model_specs=specs,
    )
