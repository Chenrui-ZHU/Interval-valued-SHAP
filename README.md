# Interval-valued SHAP

Code for the experiments in *Interval-valued SHAP in Tree-Based Models*. The repository studies Shapley and Banzhaf feature attributions for binary tabular classifiers when tree-leaf probabilities are uncertain.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Diabetes case study

The smallest complete example is the two-patient Diabetes case study from Section IV-B of the paper

```bash
python case_study.py
```

The script uses the local dataset, the fixed train/test split, and the two representative rows from the paper. It reports nominal signed SHAP values and intervals under both allocation principles at uncertainty level `s = 1`:

- `pessimistic`: worst-case allocation, computed by the greedy method;
- `averaging`: expected allocation under the empirical leaf-mass distribution.

The full table is printed and saved to `results/diabetes/case_study.csv`. The `results/` directory is generated automatically and is not tracked by Git.

## Main experiments

Run a short smoke test:

```bash
python main.py --datasets breast_cancer --cv 2 --inner-cv 2 --n-bootstrap 5 --n-eval 10
```

Run all default experiments:

```bash
python main.py
```

Run one experiment only:

```bash
python main.py --task task1_interval_rank --datasets breast_cancer
python main.py --task task6_forest --datasets breast_cancer
```

Run the synthetic debiasing experiment:

```bash
python debias.py
```

The Diabetes and NHANES data used by the documented examples are included under `datasets/`. Other datasets are downloaded from OpenML on first use and therefore require internet access.

## Repository layout

- `case_study.py` — compact Diabetes case study.
- `main.py` — command-line entry point for the main experiments.
- `exp.py` — experiment definitions.
- `interval_attribution.py` — attribution and interval computations.
- `data.py` and `loadnhanes.py` — dataset loaders.
- `models.py` — model training and selection.
- `debias.py` — synthetic debiasing experiment.
