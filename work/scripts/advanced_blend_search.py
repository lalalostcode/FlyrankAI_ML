"""Cross-fitted rank-blend selection over advanced ranking OOF predictions.

For each held-out fold, blend weights are selected only from the other four
folds.  This is an inexpensive diagnostic of whether adaptive weighting is
worth a fully nested retraining run.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OOF_PATH = ROOT / "work" / "outputs" / "advanced_ranking_oof.csv"
RESULTS_PATH = ROOT / "work" / "outputs" / "advanced_blend_search_results.json"


def precision_at_50(y_true: np.ndarray, scores: np.ndarray) -> float:
    top = np.argsort(-np.asarray(scores), kind="mergesort")[:50]
    return float(np.asarray(y_true, dtype=int)[top].mean())


def mean_p50(y: np.ndarray, scores: np.ndarray, folds: np.ndarray, selected_folds: np.ndarray) -> float:
    return float(
        np.mean(
            [precision_at_50(y[folds == fold], scores[folds == fold]) for fold in selected_folds]
        )
    )


def main() -> None:
    oof = pd.read_csv(OOF_PATH)
    models = [
        "B0_engineered_lambdarank",
        "B1_base_lambdarank",
        "B2_current_rank_blend",
        "A1_engineered_fold_query_lambdarank",
        "A3_base_fold_query_lambdarank",
        "A4_engineered_fold_query_rank_xendcg",
        "A5_engineered_fold_query_top20",
    ]
    folds = oof["fold"].to_numpy(dtype=int)
    y = oof["is_declining_label"].to_numpy(dtype=int)
    ranks = np.column_stack(
        [oof[column].groupby(oof["fold"]).rank(method="average", pct=True).to_numpy() for column in models]
    )

    candidates: list[dict] = []
    # Pairwise grids include the strong current blend and every diverse ranker.
    for left, right in itertools.combinations(range(len(models)), 2):
        for left_weight in np.linspace(0.0, 1.0, 21):
            weights = np.zeros(len(models), dtype=float)
            weights[left] = left_weight
            weights[right] = 1.0 - left_weight
            candidates.append(
                {
                    "description": f"{models[left]} + {models[right]}",
                    "weights": weights,
                    "scores": ranks @ weights,
                }
            )
    # A compact simplex grid around the three strongest client-query rankers.
    strong = [0, 1, 6]
    for first in np.linspace(0.0, 1.0, 11):
        for second in np.linspace(0.0, 1.0 - first, int(round((1.0 - first) * 10)) + 1):
            third = 1.0 - first - second
            weights = np.zeros(len(models), dtype=float)
            weights[strong] = [first, second, third]
            candidates.append(
                {
                    "description": "three_strong_rankers",
                    "weights": weights,
                    "scores": ranks @ weights,
                }
            )

    rows = []
    selected_scores = np.full(len(oof), np.nan, dtype=float)
    unique_folds = np.unique(folds)
    for held_fold in unique_folds:
        training_folds = unique_folds[unique_folds != held_fold]
        training_scores = np.asarray(
            [mean_p50(y, candidate["scores"], folds, training_folds) for candidate in candidates]
        )
        best_training_score = float(training_scores.max())
        best_indices = np.flatnonzero(np.isclose(training_scores, best_training_score))
        # Average tied candidates to reduce the instability of a 50-row cutoff.
        held_mask = folds == held_fold
        tied_prediction = np.mean(
            np.column_stack([candidates[index]["scores"][held_mask] for index in best_indices]), axis=1
        )
        selected_scores[held_mask] = tied_prediction
        mean_weights = np.mean(
            np.vstack([candidates[index]["weights"] for index in best_indices]), axis=0
        )
        rows.append(
            {
                "fold": int(held_fold),
                "training_mean_p_at_50": best_training_score,
                "number_of_tied_candidates": int(len(best_indices)),
                "test_p_at_50": precision_at_50(y[held_mask], tied_prediction),
                "mean_nonzero_weights": {
                    models[index]: float(weight)
                    for index, weight in enumerate(mean_weights)
                    if weight > 1e-12
                },
            }
        )

    current = ranks[:, models.index("B2_current_rank_blend")]
    current_p50 = mean_p50(y, current, folds, unique_folds)
    selected_p50 = mean_p50(y, selected_scores, folds, unique_folds)

    # Oracle is explicitly diagnostic: it chooses the candidate on the scored fold.
    oracle_rows = []
    for held_fold in unique_folds:
        held_mask = folds == held_fold
        scores = np.asarray(
            [precision_at_50(y[held_mask], candidate["scores"][held_mask]) for candidate in candidates]
        )
        oracle_rows.append(float(scores.max()))

    payload = {
        "stage": "cross_fitted_weight_selection_diagnostic",
        "models": models,
        "candidate_blends": int(len(candidates)),
        "benchmark_mean_p_at_50": current_p50,
        "cross_fitted_mean_p_at_50": selected_p50,
        "improvement_pp": 100.0 * (selected_p50 - current_p50),
        "folds": rows,
        "oracle_foldwise_mean_p_at_50_not_valid": float(np.mean(oracle_rows)),
        "target_achieved": bool(selected_p50 >= current_p50 + 0.05),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Benchmark mean P@50: {current_p50:.1%}")
    print(f"Cross-fitted selected blend mean P@50: {selected_p50:.1%}")
    print(f"Improvement: {payload['improvement_pp']:+.1f}pp")
    fold_text = [f"{row['test_p_at_50']:.1%}" for row in rows]
    print(f"Fold P@50: {fold_text}")
    print(f"Oracle diagnostic ceiling: {payload['oracle_foldwise_mean_p_at_50_not_valid']:.1%}")
    print(f"Wrote {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
