"""Cross-fitted probe for leakage-safe stacking candidates.

This script is deliberately a cheap screening stage.  It uses only previously
saved out-of-fold predictions and evaluates every level-2 candidate by holding
out one of the same five client-grouped folds.  Because the level-1 models for
the four meta-training folds were not nested inside the outer split, these
numbers are diagnostic rather than the final validation receipt.  Promising
methods must be rerun with inner OOF predictions in ``stacking_experiment.py``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
V1_PATH = ROOT / "work" / "outputs" / "feature_engineering_oof.csv"
V2_PATH = ROOT / "work" / "outputs" / "feature_engineering_v2_oof.csv"
OUTPUT_PATH = ROOT / "work" / "outputs" / "stacking_probe_results.json"
RANDOM_STATE = 42


def precision_at_50(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-np.asarray(scores), kind="mergesort")[:50]
    return float(np.asarray(y_true, dtype=int)[order].mean())


def fold_rank(values: pd.Series, folds: pd.Series) -> pd.Series:
    return values.groupby(folds).rank(method="average", pct=True)


def build_meta_matrix(oof: pd.DataFrame, model_columns: list[str]) -> pd.DataFrame:
    ranked = pd.DataFrame(index=oof.index)
    for column in model_columns:
        ranked[f"rank__{column}"] = fold_rank(oof[column], oof["fold"])

    matrix = ranked.copy()
    matrix["consensus_mean"] = ranked.mean(axis=1)
    matrix["consensus_median"] = ranked.median(axis=1)
    matrix["consensus_min"] = ranked.min(axis=1)
    matrix["consensus_max"] = ranked.max(axis=1)
    matrix["disagreement_std"] = ranked.std(axis=1)
    matrix["top_005_votes"] = (ranked >= 0.995).sum(axis=1)
    matrix["top_010_votes"] = (ranked >= 0.990).sum(axis=1)
    matrix["top_020_votes"] = (ranked >= 0.980).sum(axis=1)
    matrix["top_050_votes"] = (ranked >= 0.950).sum(axis=1)
    return matrix.astype(float)


def make_meta_models() -> dict[str, object]:
    return {
        "logistic_l2": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2_000, random_state=RANDOM_STATE),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=250,
            max_leaf_nodes=7,
            min_samples_leaf=100,
            l2_regularization=2.0,
            random_state=RANDOM_STATE,
        ),
        "extra_trees_meta": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=50,
            max_features=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "lightgbm_meta": lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.025,
            n_estimators=350,
            num_leaves=7,
            max_depth=4,
            min_child_samples=100,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=3.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }


def make_boundary_models() -> dict[str, object]:
    """Small-data rerankers for the union of the base models' top candidates."""
    return {
        "boundary_logistic_l2": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2_000, random_state=RANDOM_STATE),
        ),
        "boundary_extra_trees": ExtraTreesClassifier(
            n_estimators=600,
            max_depth=7,
            min_samples_leaf=5,
            max_features=0.9,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "boundary_lightgbm": lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.02,
            n_estimators=300,
            num_leaves=7,
            max_depth=4,
            min_child_samples=15,
            colsample_bytree=0.9,
            reg_alpha=0.5,
            reg_lambda=3.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }


def candidate_pool_mask(rank_matrix: np.ndarray, folds: np.ndarray, top_n: int = 50) -> np.ndarray:
    """Union of every base model's top-N rows, constructed separately by fold."""
    mask = np.zeros(len(folds), dtype=bool)
    for fold in np.unique(folds):
        fold_indices = np.flatnonzero(folds == fold)
        fold_ranks = rank_matrix[fold_indices]
        for model_index in range(fold_ranks.shape[1]):
            local_top = np.argsort(-fold_ranks[:, model_index], kind="mergesort")[:top_n]
            mask[fold_indices[local_top]] = True
    return mask


def mean_fold_p50(y: np.ndarray, scores: np.ndarray, folds: np.ndarray) -> float:
    return float(
        np.mean([precision_at_50(y[folds == fold], scores[folds == fold]) for fold in np.unique(folds)])
    )


def greedy_ensemble_weights(
    train_ranks: np.ndarray,
    y_train: np.ndarray,
    train_folds: np.ndarray,
    model_columns: list[str],
    iterations: int = 50,
) -> tuple[np.ndarray, float]:
    """Caruana-style forward selection with replacement on training folds."""
    selected: list[int] = []
    running_sum = np.zeros(len(y_train), dtype=float)
    best_selected: list[int] = []
    best_score = -np.inf

    for _ in range(iterations):
        candidate_scores = []
        for model_index in range(train_ranks.shape[1]):
            blend = (running_sum + train_ranks[:, model_index]) / (len(selected) + 1)
            candidate_scores.append(mean_fold_p50(y_train, blend, train_folds))
        chosen = int(np.argmax(candidate_scores))
        selected.append(chosen)
        running_sum += train_ranks[:, chosen]
        iteration_score = float(candidate_scores[chosen])
        if iteration_score > best_score:
            best_score = iteration_score
            best_selected = selected.copy()

    counts = Counter(best_selected)
    weights = np.asarray([counts.get(index, 0) for index in range(len(model_columns))], dtype=float)
    weights /= weights.sum()
    return weights, best_score


def main() -> None:
    v1 = pd.read_csv(V1_PATH)
    v2 = pd.read_csv(V2_PATH)
    if not v1["content_id"].equals(v2["content_id"]):
        raise AssertionError("V1 and V2 OOF rows do not align")
    if not v1["is_declining_label"].equals(v2["is_declining_label"]):
        raise AssertionError("V1 and V2 labels do not align")

    oof = v2[["content_id", "client_id", "is_declining_label", "fold"]].copy()
    v1_models = [
        "E0_current_lgbm",
        "E1_engineered_same_lgbm",
        "E2_engineered_regularized_lgbm",
        "E3_base_lambdarank",
        "E4_engineered_lambdarank",
        "E5_engineered_rank_blend",
    ]
    v2_models = [
        "F1_fev2_same_lgbm",
        "F2_fev2_regularized_lgbm",
        "F3_fev2_target_encoded_lgbm",
        "F4_fev2_xgboost",
        "F5_fev2_extra_trees",
        "F6_fev2_lambdarank_k50",
    ]
    for column in v1_models:
        oof[column] = v1[column].to_numpy()
    for column in v2_models:
        oof[column] = v2[column].to_numpy()
    model_columns = v1_models + v2_models

    x_meta = build_meta_matrix(oof, model_columns)
    y = oof["is_declining_label"].to_numpy(dtype=int)
    folds = oof["fold"].to_numpy(dtype=int)
    rank_columns = [f"rank__{column}" for column in model_columns]
    rank_matrix = x_meta[rank_columns].to_numpy()

    # Client-local ranks add the within-domain signal used by the level-1
    # LambdaRank models while retaining fold-global ranks for queue calibration.
    for column in model_columns:
        x_meta[f"client_rank__{column}"] = oof[column].groupby(oof["client_id"]).rank(
            method="average", pct=True
        )
    pool_mask = candidate_pool_mask(rank_matrix, folds, top_n=50)

    results: dict[str, list[dict]] = {name: [] for name in make_meta_models()}
    results["greedy_ensemble_selection"] = []
    for name in make_boundary_models():
        results[name] = []
    predictions = {name: np.full(len(oof), np.nan) for name in results}
    greedy_weights: list[dict] = []

    for held_fold in sorted(np.unique(folds)):
        train_mask = folds != held_fold
        test_mask = folds == held_fold
        for name, model in make_meta_models().items():
            model.fit(x_meta.loc[train_mask], y[train_mask])
            scores = model.predict_proba(x_meta.loc[test_mask])[:, 1]
            predictions[name][test_mask] = scores
            results[name].append(
                {"fold": int(held_fold), "p_at_50": precision_at_50(y[test_mask], scores)}
            )

        weights, training_score = greedy_ensemble_weights(
            rank_matrix[train_mask], y[train_mask], folds[train_mask], model_columns
        )
        greedy_scores = rank_matrix[test_mask] @ weights
        predictions["greedy_ensemble_selection"][test_mask] = greedy_scores
        results["greedy_ensemble_selection"].append(
            {"fold": int(held_fold), "p_at_50": precision_at_50(y[test_mask], greedy_scores)}
        )
        greedy_weights.append(
            {
                "fold": int(held_fold),
                "training_mean_p_at_50": training_score,
                "nonzero_weights": {
                    model_columns[index]: float(weight)
                    for index, weight in enumerate(weights)
                    if weight > 0
                },
            }
        )

        boundary_train = train_mask & pool_mask
        boundary_test = test_mask & pool_mask
        for name, model in make_boundary_models().items():
            model.fit(x_meta.loc[boundary_train], y[boundary_train])
            scores = np.full(int(test_mask.sum()), -1.0, dtype=float)
            test_positions = np.flatnonzero(test_mask)
            boundary_positions = np.flatnonzero(boundary_test)
            local_boundary_positions = np.searchsorted(test_positions, boundary_positions)
            scores[local_boundary_positions] = model.predict_proba(x_meta.loc[boundary_test])[:, 1]
            predictions[name][test_mask] = scores
            results[name].append(
                {"fold": int(held_fold), "p_at_50": precision_at_50(y[test_mask], scores)}
            )

    summaries = []
    for name, rows in results.items():
        values = np.asarray([row["p_at_50"] for row in rows])
        summaries.append(
            {
                "candidate": name,
                "mean_p_at_50": float(values.mean()),
                "std_p_at_50": float(values.std(ddof=1)),
                "fold_p_at_50": [float(value) for value in values],
            }
        )
    summaries.sort(key=lambda row: row["mean_p_at_50"], reverse=True)

    payload = {
        "stage": "cross_fitted_diagnostic_probe_not_final_nested_validation",
        "rows": int(len(oof)),
        "folds": int(len(np.unique(folds))),
        "base_models": model_columns,
        "candidate_pool_rows": int(pool_mask.sum()),
        "winner": summaries[0]["candidate"],
        "winner_mean_p_at_50": summaries[0]["mean_p_at_50"],
        "summaries": summaries,
        "greedy_weights": greedy_weights,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Cross-fitted diagnostic probe (not final nested validation)")
    for row in summaries:
        fold_text = ", ".join(f"{value:.1%}" for value in row["fold_p_at_50"])
        print(f"{row['candidate']}: mean P@50={row['mean_p_at_50']:.1%}; folds=[{fold_text}]")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
