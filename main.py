from __future__ import annotations

import argparse
import importlib
import time
from collections.abc import Iterable


DEFAULT_DATASETS = ("breast_cancer", "diabetes", "ionosphere", "nhanes")


def _parse_datasets(raw: str) -> tuple[str, ...]:
    datasets = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not datasets:
        raise ValueError("At least one dataset name is required.")
    return datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean entry point for the focused new_exp experiments."
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


def _load_new_exp():
    try:
        return importlib.import_module("exp")
    except ModuleNotFoundError as exc:
        if exc.name == "exp":
            raise SystemExit(
                "Could not import exp.py. Save or restore exp.py in this directory, "
                "then rerun new_main.py."
            ) from exc
        raise


def _config_from_args(exp, args: argparse.Namespace):
    return exp.ExperimentConfig(
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
    new_exp = _load_new_exp()
    config = _config_from_args(new_exp, args)
    start_time = time.time()

    if args.task in {"task1_interval_rank", "all"}:
        new_exp.run_task1_interval_rank(config, model_type=args.task1_model_type)

    if args.task in {"task6_forest", "all"}:
        new_exp.run_task6_forest(config)

    print(f"\n[new_main] total runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
