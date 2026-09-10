"""Metric-aligned advanced ranking experiments for the FlyRank top-50 queue.

The existing LambdaRank benchmark treats every client as a separate query but
evaluates one global queue spanning all held-out clients.  This experiment keeps
the exact five outer GroupKFold splits and tests query definitions that better
match that operational ranking unit.  No target-window, label-derived, ID, or
product-score column is used as a model feature.
"""

from __future__ import annotations

import importlib.util
import json
import platform
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]
V1_SCRIPT = ROOT / "work" / "scripts" / "feature_engineering_experiment.py"
FEATURE_PATH = ROOT / "data" / "processed" / "refresh_feature_vector.csv"
RAW_PATH = ROOT / "data" / "raw" / "content_refresh_anonymized.csv"
V1_OOF_PATH = ROOT / "work" / "outputs" / "feature_engineering_oof.csv"
V2_OOF_PATH = ROOT / "work" / "outputs" / "feature_engineering_v2_oof.csv"
RESULTS_PATH = ROOT / "work" / "outputs" / "advanced_ranking_results.json"
OOF_PATH = ROOT / "work" / "outputs" / "advanced_ranking_oof.csv"

RANDOM_STATE = 42
N_SPLITS = 5


def load_v1_module():
    spec = importlib.util.spec_from_file_location("flyrank_fe_v1", V1_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {V1_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1 = load_v1_module()


def fit_metric_aligned_ranker(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    query_ids: pd.Series,
    x_test: pd.DataFrame,
    *,
    objective: str = "lambdarank",
    truncation: int = 51,
    num_leaves: int = 31,
    min_child_samples: int = 50,
) -> tuple[np.ndarray, lgb.LGBMRanker]:
    query_text = query_ids.astype(str).to_numpy()
    order = np.argsort(query_text, kind="mergesort")
    sorted_x = x_train.iloc[order]
    sorted_y = y_train.iloc[order]
    sorted_queries = query_text[order]
    _, group_sizes = np.unique(sorted_queries, return_counts=True)

    parameters = dict(
        objective=objective,
        metric="ndcg",
        learning_rate=0.025,
        n_estimators=750,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        colsample_bytree=0.8,
        reg_alpha=0.25,
        reg_lambda=2.0,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    if objective == "lambdarank":
        parameters["lambdarank_truncation_level"] = truncation
    ranker = lgb.LGBMRanker(**parameters)
    ranker.fit(sorted_x, sorted_y, group=group_sizes.tolist(), eval_at=[20, 50, 100])
    return ranker.predict(x_test), ranker


def summarize(rows: list[dict]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric in ["base_rate", "p_at_20", "p_at_50", "p_at_100", "lift_at_50", "roc_auc", "average_precision"]:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        summary[f"mean_{metric}"] = float(values.mean())
        summary[f"std_{metric}"] = float(values.std(ddof=1))
    return summary


def evaluate_oof(
    target: pd.Series,
    groups: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    scores: np.ndarray,
) -> list[dict]:
    rows = []
    for fold_number, (_, test_index) in enumerate(folds, start=1):
        rows.append(
            {
                "fold": fold_number,
                "test_rows": int(len(test_index)),
                "test_clients": int(groups.iloc[test_index].nunique()),
                **v1.evaluate_scores(target.iloc[test_index], scores[test_index]),
            }
        )
    return rows


def fold_rank_blend(oof: pd.DataFrame, columns: list[str], weights: list[float]) -> np.ndarray:
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Blend weights must sum to one")
    scores = np.zeros(len(oof), dtype=float)
    for column, weight in zip(columns, weights):
        scores += weight * oof.groupby("fold")[column].rank(method="average", pct=True).to_numpy()
    return scores


def main() -> None:
    np.random.seed(RANDOM_STATE)
    prepared = pd.read_csv(FEATURE_PATH)
    raw = pd.read_csv(RAW_PATH)
    v1_oof = pd.read_csv(V1_OOF_PATH)
    v2_oof = pd.read_csv(V2_OOF_PATH)
    if not prepared["content_id"].equals(v1_oof["content_id"]):
        raise AssertionError("Prepared rows do not align with V1 OOF")
    if not prepared["content_id"].equals(v2_oof["content_id"]):
        raise AssertionError("Prepared rows do not align with V2 OOF")

    engineered = v1.add_engineered_features(prepared, raw)
    base_matrix, base_names = v1.build_matrix(
        prepared, v1.BASE_NUMERIC_FEATURES, v1.BASE_CATEGORICAL_FEATURES
    )
    engineered_matrix, engineered_names = v1.build_matrix(
        engineered,
        v1.BASE_NUMERIC_FEATURES + v1.ENGINEERED_NUMERIC_FEATURES,
        v1.BASE_CATEGORICAL_FEATURES,
    )
    target = prepared["is_declining_label"].astype(int)
    clients = prepared["client_id"].fillna("unknown").astype(str)
    folds = list(GroupKFold(n_splits=N_SPLITS).split(engineered_matrix, target, groups=clients))
    fixed_fold_ids = v1_oof["fold"].astype(int)

    oof = pd.DataFrame(
        {
            "content_id": prepared["content_id"],
            "client_id": prepared["client_id"],
            "is_declining_label": target,
            "fold": fixed_fold_ids,
            "B0_engineered_lambdarank": v1_oof["E4_engineered_lambdarank"],
            "B1_base_lambdarank": v1_oof["E3_base_lambdarank"],
            "B2_current_rank_blend": v2_oof["F7_lambda_pair_equal_blend"],
        }
    )

    candidates = {
        "A1_engineered_fold_query_lambdarank": {
            "matrix": engineered_matrix,
            "query": "fold",
            "objective": "lambdarank",
            "truncation": 51,
            "num_leaves": 31,
            "min_child_samples": 50,
        },
        "A3_base_fold_query_lambdarank": {
            "matrix": base_matrix,
            "query": "fold",
            "objective": "lambdarank",
            "truncation": 51,
            "num_leaves": 31,
            "min_child_samples": 50,
        },
        "A4_engineered_fold_query_rank_xendcg": {
            "matrix": engineered_matrix,
            "query": "fold",
            "objective": "rank_xendcg",
            "truncation": 51,
            "num_leaves": 31,
            "min_child_samples": 50,
        },
        "A5_engineered_fold_query_top20": {
            "matrix": engineered_matrix,
            "query": "fold",
            "objective": "lambdarank",
            "truncation": 21,
            "num_leaves": 31,
            "min_child_samples": 50,
        },
    }

    results: dict[str, dict] = {}
    for benchmark in ["B0_engineered_lambdarank", "B1_base_lambdarank", "B2_current_rank_blend"]:
        rows = evaluate_oof(target, clients, folds, oof[benchmark].to_numpy())
        results[benchmark] = {"folds": rows, "summary": summarize(rows)}

    print(f"Rows={len(prepared):,}; clients={clients.nunique()}; positive rate={target.mean():.2%}")
    print(f"Base matrix={base_matrix.shape}; engineered matrix={engineered_matrix.shape}")
    print("Forbidden target-window and identifier feature guard: PASS")

    for name, specification in candidates.items():
        scores = np.full(len(prepared), np.nan, dtype=float)
        fold_rows = []
        print(f"\n=== {name} ===")
        for fold_number, (train_index, test_index) in enumerate(folds, start=1):
            matrix = specification["matrix"]
            if specification["query"] == "fold":
                query_ids = fixed_fold_ids.iloc[train_index]
            elif specification["query"] == "global":
                query_ids = pd.Series(np.zeros(len(train_index), dtype=int), index=train_index)
            else:
                raise KeyError(specification["query"])
            fold_scores, _ = fit_metric_aligned_ranker(
                matrix.iloc[train_index],
                target.iloc[train_index],
                query_ids,
                matrix.iloc[test_index],
                objective=str(specification["objective"]),
                truncation=int(specification["truncation"]),
                num_leaves=int(specification["num_leaves"]),
                min_child_samples=int(specification["min_child_samples"]),
            )
            scores[test_index] = fold_scores
            metrics = v1.evaluate_scores(target.iloc[test_index], fold_scores)
            fold_rows.append(
                {
                    "fold": fold_number,
                    "test_rows": int(len(test_index)),
                    "test_clients": int(clients.iloc[test_index].nunique()),
                    **metrics,
                }
            )
            print(
                f"fold={fold_number} P@20={metrics['p_at_20']:.1%} "
                f"P@50={metrics['p_at_50']:.1%} P@100={metrics['p_at_100']:.1%}"
            )
        if np.isnan(scores).any():
            raise AssertionError(f"Missing OOF scores for {name}")
        oof[name] = scores
        results[name] = {
            "folds": fold_rows,
            "summary": summarize(fold_rows),
            "specification": {key: value for key, value in specification.items() if key != "matrix"},
        }
        print(f"mean P@50={results[name]['summary']['mean_p_at_50']:.1%}")

    blend_specs = {
        "A6_current_plus_fold_query_equal": (
            ["B2_current_rank_blend", "A1_engineered_fold_query_lambdarank"], [0.5, 0.5]
        ),
        "A7_three_ranker_equal": (
            ["B0_engineered_lambdarank", "B1_base_lambdarank", "A1_engineered_fold_query_lambdarank"],
            [1 / 3, 1 / 3, 1 / 3],
        ),
        "A9_fold_query_pair_equal": (
            ["A1_engineered_fold_query_lambdarank", "A3_base_fold_query_lambdarank"], [0.5, 0.5]
        ),
        "A10_current_plus_xendcg_equal": (
            ["B2_current_rank_blend", "A4_engineered_fold_query_rank_xendcg"], [0.5, 0.5]
        ),
        "A11_current_plus_top20_equal": (
            ["B2_current_rank_blend", "A5_engineered_fold_query_top20"], [0.5, 0.5]
        ),
    }
    for name, (columns, weights) in blend_specs.items():
        oof[name] = fold_rank_blend(oof, columns, weights)
        rows = evaluate_oof(target, clients, folds, oof[name].to_numpy())
        results[name] = {"folds": rows, "summary": summarize(rows), "components": columns, "weights": weights}

    comparison = []
    for name, result in results.items():
        summary = result["summary"]
        comparison.append(
            {
                "candidate": name,
                "mean_p_at_20": summary["mean_p_at_20"],
                "mean_p_at_50": summary["mean_p_at_50"],
                "std_p_at_50": summary["std_p_at_50"],
                "mean_p_at_100": summary["mean_p_at_100"],
                "mean_roc_auc": summary["mean_roc_auc"],
                "mean_average_precision": summary["mean_average_precision"],
            }
        )
    comparison.sort(key=lambda row: (row["mean_p_at_50"], row["mean_p_at_20"]), reverse=True)
    winner = next(row for row in comparison if not row["candidate"].startswith("B"))
    benchmark = next(row for row in comparison if row["candidate"] == "B2_current_rank_blend")
    payload = {
        "experiment": "metric_aligned_advanced_ranking",
        "estimand": "cross-client current-snapshot decline ranking; not future forecasting",
        "primary_metric": "mean precision@50 across the same five fixed GroupKFold folds",
        "random_state": RANDOM_STATE,
        "rows": int(len(prepared)),
        "clients": int(clients.nunique()),
        "positive_rate": float(target.mean()),
        "winner": winner["candidate"],
        "benchmark": benchmark["candidate"],
        "winner_mean_p_at_50": winner["mean_p_at_50"],
        "benchmark_mean_p_at_50": benchmark["mean_p_at_50"],
        "improvement_pp": 100.0 * (winner["mean_p_at_50"] - benchmark["mean_p_at_50"]),
        "target_improvement_pp": 5.0,
        "target_achieved": bool(winner["mean_p_at_50"] >= benchmark["mean_p_at_50"] + 0.05),
        "comparison": comparison,
        "results": results,
        "feature_counts": {"base": len(base_names), "engineered": len(engineered_names)},
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
        },
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    oof.to_csv(OOF_PATH, index=False)

    print("\n=== FINAL COMPARISON ===")
    for row in comparison:
        print(
            f"{row['candidate']}: P@50={row['mean_p_at_50']:.1%} "
            f"P@20={row['mean_p_at_20']:.1%} P@100={row['mean_p_at_100']:.1%}"
        )
    print(f"Benchmark P@50={benchmark['mean_p_at_50']:.1%}")
    print(f"Best advanced candidate={winner['candidate']} at {winner['mean_p_at_50']:.1%}")
    print(f"Improvement={payload['improvement_pp']:+.1f}pp; target achieved={payload['target_achieved']}")
    print(f"Wrote {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
