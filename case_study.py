"""Run the two-patient Diabetes case study from the paper."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from interval_attribution import compute_interval_attribution, compute_nominal_attribution


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "datasets" / "diabetes" / "diabetes.csv"
OUTPUT_PATH = ROOT / "results" / "diabetes" / "case_study.csv"

PATIENTS = {"patient_1": 122, "patient_2": 14}
FEATURE_NAMES = ["preg", "plas", "pres", "skin", "insu", "mass", "pred", "age"]

# Produces the depth-6, 12-leaf tree used for the case study.
CCP_ALPHA = 0.0044697622217002055


def run_case_study() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    y = data.pop("Outcome").astype(int)
    X = data.set_axis(FEATURE_NAMES, axis="columns").astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    model = DecisionTreeClassifier(random_state=0, ccp_alpha=CCP_ALPHA)
    model.fit(X_train, y_train)

    tables = []
    predictions = []
    options = {
        "kind": "shapley",
        "semantics": "interventional",
        "target_class": 1,
        "bg_size": 200,
        "random_state": 0,
    }

    for patient, row_id in PATIENTS.items():
        x = X_test.loc[row_id]
        nominal = compute_nominal_attribution(model, X_train, x, **options)
        pessimistic = compute_interval_attribution(
            model, X_train, x, s=1, allocation="greedy", **options
        )
        averaging = compute_interval_attribution(
            model, X_train, x, s=1, allocation="multinomial", **options
        )

        tables.append(
            pd.DataFrame(
                {
                    "patient": patient,
                    "row_id": row_id,
                    "feature": X.columns,
                    "feature_value": x.to_numpy(),
                    "nominal": nominal.values,
                    "pessimistic_lower": pessimistic.signed_lower,
                    "pessimistic_upper": pessimistic.signed_upper,
                    "averaging_lower": averaging.signed_lower,
                    "averaging_upper": averaging.signed_upper,
                }
            )
        )
        predictions.append(
            {
                "patient": patient,
                "row_id": row_id,
                "true_label": int(y_test.loc[row_id]),
                "predicted_label": int(model.predict(x.to_frame().T)[0]),
            }
        )

    result = pd.concat(tables, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(
        f"Model: depth={model.get_depth()}, leaves={model.get_n_leaves()}, "
        f"test accuracy={model.score(X_test, y_test):.3f}"
    )
    print(pd.DataFrame(predictions).to_string(index=False))
    print()
    print(result.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved {OUTPUT_PATH.relative_to(ROOT)}")
    return result


if __name__ == "__main__":
    run_case_study()
