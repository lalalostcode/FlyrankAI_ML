"""Cross-fitted hierarchical calibration probe for the global top-50 queue.

The level-1 rankers learn within-client ordering, while evaluation pools several
clients into one queue.  This diagnostic predicts each held-out client's label
prevalence from label-free client summaries and uses that prediction as a small
offset to the current row-level rank.  Blend strength is selected only on the
other four folds.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
V1_SCRIPT = ROOT / "work" / "scripts" / "feature_engineering_experiment.py"
FEATURE_PATH = ROOT / "data" / "processed" / "refresh_feature_vector.csv"
V2_OOF_PATH = ROOT / "work" / "outputs" / "feature_engineering_v2_oof.csv"
RESULTS_PATH = ROOT / "work" / "outputs" / "hierarchical_calibration_probe_results.json"
RANDOM_STATE = 42


def load_v1_module():
    spec = importlib.util.spec_from_file_location("flyrank_fe_v1", V1_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {V1_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1 = load_v1_module()


def precision_at_50(y_true: np.ndarray, scores: np.ndarray) -> float:
    top = np.argsort(-np.asarray(scores), kind="mergesort")[:50]
    return float(np.asarray(y_true, dtype=int)[top].mean())


def fit_ridge(x: pd.DataFrame, y: pd.Series) -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(x, y)


def client_risk_rank(
    row_clients: pd.Series,
    client_predictions: pd.Series,
    row_folds: pd.Series,
) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "client": row_clients.to_numpy(),
            "fold": row_folds.to_numpy(),
            "risk": row_clients.map(client_predictions).to_numpy(),
        }
    )
    client_frame = frame.drop_duplicates(["fold", "client"]).copy()
    client_frame["risk_rank"] = client_frame.groupby("fold")["risk"].rank(method="average", pct=True)
    lookup = client_frame.set_index(["fold", "client"])["risk_rank"]
    return np.asarray([lookup.loc[(fold, client)] for fold, client in zip(frame["fold"], frame["client"])])


def main() -> None:
    prepared = pd.read_csv(FEATURE_PATH)
    oof = pd.read_csv(V2_OOF_PATH)
    if not prepared["content_id"].equals(oof["content_id"]):
        raise AssertionError("Feature rows and OOF rows do not align")

    matrix, _ = v1.build_matrix(prepared, v1.BASE_NUMERIC_FEATURES, v1.BASE_CATEGORICAL_FEATURES)
    clients = prepared["client_id"].fillna("unknown").astype(str)
    folds = oof["fold"].astype(int)
    y = oof["is_declining_label"].astype(int)
    base_rank = oof["F7_lambda_pair_equal_blend"].groupby(folds).rank(method="average", pct=True)

    client_features = matrix.assign(_client=clients.to_numpy()).groupby("_client").mean()
    client_features["log_inventory_size"] = np.log1p(clients.value_counts().reindex(client_features.index))
    client_target = y.groupby(clients).mean().reindex(client_features.index)
    client_fold = folds.groupby(clients).first().reindex(client_features.index)

    # Each client's diagnostic prediction is generated without that client.
    loo_predictions = pd.Series(index=client_features.index, dtype=float)
    for held_client in client_features.index:
        train_clients = client_features.index != held_client
        model = fit_ridge(client_features.loc[train_clients], client_target.loc[train_clients])
        loo_predictions.loc[held_client] = float(model.predict(client_features.loc[[held_client]])[0])
    loo_row_risk = client_risk_rank(clients, loo_predictions, folds)

    unique_folds = np.sort(folds.unique())
    alpha_grid = np.linspace(0.0, 0.5, 21)
    final_scores = np.full(len(oof), np.nan, dtype=float)
    rows = []

    for held_fold in unique_folds:
        training_folds = unique_folds[unique_folds != held_fold]
        alpha_scores = []
        for alpha in alpha_grid:
            blend = (1.0 - alpha) * base_rank.to_numpy() + alpha * loo_row_risk
            alpha_scores.append(
                np.mean(
                    [
                        precision_at_50(y[folds == fold].to_numpy(), blend[folds == fold])
                        for fold in training_folds
                    ]
                )
            )
        best_score = float(np.max(alpha_scores))
        tied_alphas = alpha_grid[np.isclose(alpha_scores, best_score)]
        chosen_alpha = float(tied_alphas.mean())

        outer_train_clients = client_features.index[client_fold != held_fold]
        outer_test_clients = client_features.index[client_fold == held_fold]
        model = fit_ridge(client_features.loc[outer_train_clients], client_target.loc[outer_train_clients])
        test_client_prediction = pd.Series(
            model.predict(client_features.loc[outer_test_clients]), index=outer_test_clients
        )
        test_mask = folds == held_fold
        test_risk_rank = client_risk_rank(
            clients.loc[test_mask],
            test_client_prediction,
            folds.loc[test_mask],
        )
        scores = (1.0 - chosen_alpha) * base_rank.loc[test_mask].to_numpy() + chosen_alpha * test_risk_rank
        final_scores[test_mask] = scores
        rows.append(
            {
                "fold": int(held_fold),
                "selected_alpha": chosen_alpha,
                "training_mean_p_at_50": best_score,
                "test_p_at_50": precision_at_50(y.loc[test_mask].to_numpy(), scores),
            }
        )

    benchmark = float(
        np.mean(
            [precision_at_50(y[folds == fold].to_numpy(), base_rank[folds == fold].to_numpy()) for fold in unique_folds]
        )
    )
    calibrated = float(
        np.mean(
            [precision_at_50(y[folds == fold].to_numpy(), final_scores[folds == fold]) for fold in unique_folds]
        )
    )
    payload = {
        "stage": "cross_fitted_hierarchical_calibration_diagnostic",
        "clients": int(client_features.shape[0]),
        "client_predictor_features": int(client_features.shape[1]),
        "benchmark_mean_p_at_50": benchmark,
        "calibrated_mean_p_at_50": calibrated,
        "improvement_pp": 100.0 * (calibrated - benchmark),
        "folds": rows,
        "target_achieved": bool(calibrated >= benchmark + 0.05),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Benchmark mean P@50: {benchmark:.1%}")
    print(f"Hierarchically calibrated mean P@50: {calibrated:.1%}")
    print(f"Improvement: {payload['improvement_pp']:+.1f}pp")
    print(f"Selected alphas: {[row['selected_alpha'] for row in rows]}")
    print(f"Fold P@50: {[round(row['test_p_at_50'], 3) for row in rows]}")
    print(f"Wrote {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
