from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from sklearn.impute import SimpleImputer
import re

import numpy as np
import pandas as pd

from sklearn.datasets import (
    load_breast_cancer,
    load_wine,
    load_digits,
    fetch_openml,
    make_classification,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

import loadnhanes


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    task: str
    n_samples: int
    n_features: int
    classes: Tuple[str, str] | Tuple[int, int]
    class_mapping: Dict[object, int]  # original label -> {0,1}


def _to_frame(X, feature_names=None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=list(feature_names))


def _to_series(y, name: str = "y") -> pd.Series:
    if isinstance(y, pd.Series):
        if y.name is None:
            y = y.rename(name)
        return y
    return pd.Series(y, name=name)


def _binarize_y(
    y: pd.Series,
    *,
    positive_label: Optional[object] = None,
) -> Tuple[pd.Series, Dict[object, int]]:
    """
    Map a 2-class label vector to {0,1}.
    If positive_label is provided, map that label -> 1 and the other -> 0.
    Otherwise, map sorted unique labels to 0/1 and record the mapping.
    """
    y = y.copy()

    # Drop missing target values
    y = y.dropna()
    uniq = list(pd.unique(y))

    if len(uniq) != 2:
        raise ValueError(
            f"Expected binary target with 2 classes, got {len(uniq)} classes: {uniq}. "
            "Either choose a binary dataset or specify how to binarize (one-vs-rest)."
        )

    if positive_label is not None:
        if positive_label not in uniq:
            raise ValueError(f"positive_label={positive_label!r} not in observed labels: {uniq}")
        neg_label = uniq[0] if uniq[1] == positive_label else uniq[1]
        mapping = {neg_label: 0, positive_label: 1}
    else:
        # Stable default: lexicographic sort for strings, numeric sort for numbers
        try:
            labels_sorted = sorted(uniq)
        except TypeError:
            labels_sorted = uniq  # fallback: keep observed order
        mapping = {labels_sorted[0]: 0, labels_sorted[1]: 1}

    y_bin = y.map(mapping).astype(int)
    return y_bin, mapping


def _drop_missing(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Drop rows with any missing values in X or y.
    (DecisionTreeClassifier does not accept NaNs.)
    """
    df = X.copy()
    df["_y_tmp_"] = y
    df = df.dropna(axis=0)
    y2 = df.pop("_y_tmp_").astype(int)
    return df, y2


def load_dataset(
    dataset_name: str,
    *,
    dropna: bool = True,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.Series, DatasetInfo]:
    """
    Load a dataset by name and return (X: DataFrame, y: Series in {0,1}, info).

    Supported dataset_name values:
        - “diabetes"                     (8 numeric features, binary, OpenML id=37)
        - "wine_0_vs_rest"               (13 numeric features; derived binary)
        - "breast_cancer"                (30 numeric features, binary)
        - "ionosphere"                   (34 numeric features, binary, OpenML id=59)    
        - "spambase"                     (57 numeric features, binary, OpenML id=44)
        - "digits_0_vs_rest"             (64 numeric features; derived binary)
        - "adult"                        (14 mixed features; OpenML id=1590)
        - "bank_marketing"               (16 mixed features; OpenML id=1461)
        - "credit_g"                     (20 mixed features; OpenML id=31)
        - "nhanes"                       (79 mixed features)
        - "synthetic_<p>" e.g. "synthetic_50"  (make_classification with p features)
        - "synthetic_task6" or "synthetic_task6_<n>_<p>" (numeric Task 6 stress test)

    Notes:
        - OpenML fetch requires internet access on your machine.
        - For mixed-type datasets, you'll one-hot encode later in your pipeline.
    """
    name = dataset_name.strip().lower()

    # --- sklearn built-ins ---
    if name == "breast_cancer":
        data = load_breast_cancer(as_frame=True)
        X = data.data
        y_raw = _to_series(data.target, name="target")
        # Already 0/1
        y = y_raw.astype(int)
        mapping = {0: 0, 1: 1}

    elif name == "digits_0_vs_rest":
        data = load_digits(as_frame=True)
        X = data.data
        y_raw = _to_series(data.target, name="target")
        # binary: 1 iff digit==0
        y = (y_raw == 0).astype(int)
        mapping = {0: 1, "not_0": 0}  # informational only

    elif name == "wine_0_vs_rest":
        data = load_wine(as_frame=True)
        X = data.data
        y_raw = _to_series(data.target, name="target")
        y = (y_raw == 0).astype(int)
        mapping = {0: 1, "not_0": 0}

    elif name == "nhanes":
        X, y = loadnhanes._load()
        for c in X.columns:
            if c.endswith("_isBlank"):
                del X[c]   
        X["bmi"] = 10000 * X["weight"].values.copy() / (X["height"].values.copy() * X["height"].values.copy())
        del X["weight"]
        del X["height"]
        del X["urine_hematest_isTrace"] # would have no variance in the strain set
        del X["SGOT_isBlankbutapplicable"] # would have no variance in the strain set
        del X["calcium_isBlankbutapplicable"] # would have no variance in the strain set
        del X["uric_acid_isBlankbutapplicable"] # would only have one true value in the train set
        del X["urine_hematest_isVerylarge"] # would only have one true value in the train set
        del X["total_bilirubin_isBlankbutapplicable"] # would only have one true value in the train set
        del X["alkaline_phosphatase_isBlankbutapplicable"] # would only have one true value in the train set
        del X["hemoglobin_isUnacceptable"] # redundant with hematocrit_isUnacceptable
        rows = np.where(np.invert(np.isnan(X["systolic_blood_pressure"]) | np.isnan(X["bmi"])))[0]

        X = X.iloc[rows,:]
        # drop very sparse columns if desired
        missing_frac = X.isna().mean()
        X = X.loc[:, missing_frac < 0.8]
        # impute
        imp = SimpleImputer(strategy="median")
        X = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)

        y_raw = y[rows]
        # if y < 0 then the patient is alive at follow-up, if y > 0 then the patient is deceased at follow-up
        y = (y_raw > 0).astype(int)
        mapping = {0: 0, 1: 1}

    # --- OpenML datasets (binary classification) ---
    elif name in {"spambase", "ionosphere", "adult", "bank_marketing", "credit_g", "diabetes"}:
        openml_ids = {
            "spambase": 44,
            "ionosphere": 59,
            "adult": 1590,
            "bank_marketing": 1461,
            "credit_g": 31,
            "diabetes": 37,
        }
        data_id = openml_ids[name]

        # as_frame=True so you keep mixed types as pandas columns
        X_raw, y_raw = fetch_openml(data_id=data_id, as_frame=True, return_X_y=True)
        X = _to_frame(X_raw)
        y_raw = _to_series(y_raw, name="target")

        # Choose positive class for nicer interpretation (optional but helpful)
        positive_label = None
        if name == "adult":
            # typical labels: "<=50K" and ">50K"
            positive_label = ">50K"
        elif name == "bank_marketing":
            # typical labels: "no" and "yes"
            positive_label = "yes"
        elif name == "credit_g":
            # typical labels: "good" and "bad" (we usually treat "bad" as positive)
            positive_label = "bad"

        y, mapping = _binarize_y(y_raw, positive_label=positive_label)

        # Align X with non-missing y index
        X = X.loc[y.index]

    # --- synthetic ---
    elif name.startswith("synthetic_task6"):
        m = re.match(r"synthetic_task6(?:[_-](\d+)[_-](\d+))?$", name)
        if not m:
            raise ValueError('Use "synthetic_task6" or "synthetic_task6_<n>_<p>", e.g. "synthetic_task6_3000_20".')
        n_samples = int(m.group(1)) if m.group(1) is not None else 3000
        p = int(m.group(2)) if m.group(2) is not None else 20
        if n_samples < 50:
            raise ValueError("synthetic_task6 requires at least 50 samples.")
        if p < 4:
            raise ValueError("synthetic_task6 requires at least 4 features.")
        X_np, y_np = make_classification(
            n_samples=n_samples,
            n_features=p,
            n_informative=max(3, min(p - 1, p // 3)),
            n_redundant=max(1, min(p // 4, p - max(3, min(p - 1, p // 3)))),
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=2,
            weights=[0.5, 0.5],
            flip_y=0.03,
            class_sep=0.75,
            random_state=random_state,
        )
        X = _to_frame(X_np, [f"x{i}" for i in range(p)])
        y = _to_series(y_np, name="target").astype(int)
        mapping = {0: 0, 1: 1}

    elif name.startswith("synthetic"):
        m = re.match(r"synthetic[_-](\d+)$", name)
        if not m:
            raise ValueError('Use "synthetic_<p>", e.g. "synthetic_50".')
        p = int(m.group(1))
        X_np, y_np = make_classification(
            n_samples=5000,
            n_features=p,
            n_informative=max(2, p // 5),
            n_redundant=max(0, p // 5),
            n_repeated=0,
            n_classes=2,
            weights=None,
            flip_y=0.01,
            class_sep=1.0,
            random_state=random_state,
        )
        X = _to_frame(X_np, [f"x{i}" for i in range(p)])
        y = _to_series(y_np, name="target").astype(int)
        mapping = {0: 0, 1: 1}

    else:
        raise ValueError(
            f"Unknown dataset_name={dataset_name!r}. "
            "Try one of: breast_cancer, ionosphere, spambase, adult, bank_marketing, credit_g, "
            "digits_0_vs_rest, wine_0_vs_rest, synthetic_<p>, synthetic_task6, "
            "synthetic_task6_<n>_<p>."
        )

    # Optional: drop rows with missing values (recommended for DecisionTreeClassifier)
    if dropna:
        X, y = _drop_missing(X, y)

    # Build info
    classes = tuple(sorted(mapping.keys(), key=lambda z: str(z)))  # just for display
    info = DatasetInfo(
        name=dataset_name,
        task="binary_classification",
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        classes=classes if len(classes) == 2 else (0, 1),
        class_mapping=mapping,
    )
    return X, y, info


def stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Return a reproducible stratified train/test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def make_stratified_cv(
    cv: int = 5,
    *,
    random_state: int = 0,
) -> StratifiedKFold:
    """
    Build a shuffled stratified cross-validation splitter.
    """
    return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
