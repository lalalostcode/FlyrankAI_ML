"""Paired feature-engineering experiments for the FlyRank decline ranker.

This is a development experiment on the 30k starter snapshot. It preserves the
exact five client-grouped folds used in W06 and compares every candidate on the
same rows and metrics. It does not establish future forecasting validity.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = ROOT / "data" / "processed" / "refresh_feature_vector.csv"
RAW_PATH = ROOT / "data" / "raw" / "content_refresh_anonymized.csv"
BASELINE_PATH = ROOT / "work" / "outputs" / "baseline_action_score.csv"
RESULTS_PATH = ROOT / "work" / "outputs" / "feature_engineering_results.json"
OOF_PATH = ROOT / "work" / "outputs" / "feature_engineering_oof.csv"

RANDOM_STATE = 42
N_SPLITS = 5

BASE_NUMERIC_FEATURES = [
    "search_volume",
    "competition",
    "cpc",
    "word_count",
    "char_count",
    "log_impressions_90d",
    "log_clicks_90d",
    "log_sessions_90d",
    "log_ai_sessions_90d",
    "days_with_impressions",
    "days_with_sessions",
    "content_age_days",
    "days_since_last_update",
    "ctr",
    "avg_position",
    "engagement_rate",
    "scroll_rate",
    "ai_traffic_pct",
]

BASE_CATEGORICAL_FEATURES = [
    "competition_level",
    "content_type",
    "main_intent",
    "age_tier",
    "freshness_tier",
    "word_count_tier",
    "impression_tier",
    "position_tier",
]

FORBIDDEN_MODEL_COLUMNS = {
    "trend_direction",
    "trend_pct",
    "is_declining_label",
    "content_id",
    "client_id",
    "impressions_last_30d",
    "clicks_last_30d",
    "sessions_last_30d",
    "health_score",
    "quick_win",
    "needs_attention",
    "baseline_refresh_score",
    "action_score",
}


def precision_at_k(y_true: pd.Series | np.ndarray, scores: np.ndarray, k: int) -> float:
    y_array = np.asarray(y_true, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    cutoff = min(k, len(y_array))
    if cutoff == 0:
        return 0.0
    top = np.argsort(-score_array, kind="mergesort")[:cutoff]
    return float(y_array[top].mean())


def evaluate_scores(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_array = np.asarray(y_true, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    base_rate = float(y_array.mean())
    p50 = precision_at_k(y_array, score_array, 50)
    return {
        "base_rate": base_rate,
        "p_at_20": precision_at_k(y_array, score_array, 20),
        "p_at_50": p50,
        "p_at_100": precision_at_k(y_array, score_array, 100),
        "lift_at_50": p50 / base_rate if base_rate > 0 else 0.0,
        "roc_auc": float(roc_auc_score(y_array, score_array)),
        "average_precision": float(average_precision_score(y_array, score_array)),
    }


def add_engineered_features(prepared: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Create label-free features available for batch scoring at snapshot time.

    Client-relative features use only predictor values from the target client's
    current inventory. This is valid for the intended batch-ranking workflow but
    should not be used for single-row inference without stored client statistics.
    """
    raw_indexed = raw.set_index("content_id", drop=False)
    aligned_raw = raw_indexed.reindex(prepared["content_id"]).reset_index(drop=True)
    if aligned_raw["content_id"].isna().any():
        raise ValueError("Raw/prepared content_id alignment failed")

    frame = prepared.copy()

    # Missingness is systematic in this dataset; preserve it before numeric fill.
    frame["missing_position_flag"] = aligned_raw["avg_position"].fillna(0).eq(0).astype(int)
    frame["has_keyword_data"] = aligned_raw["search_volume"].notna().astype(int)
    frame["has_word_count"] = aligned_raw["word_count"].notna().astype(int)
    frame["has_intent"] = aligned_raw["main_intent"].notna().astype(int)
    frame["has_scroll_rate"] = aligned_raw["scroll_rate"].notna().astype(int)

    # Do not add previous-30-day comparator columns while the inherited E0
    # feature set still contains 90-day totals. The totals include the latest
    # outcome month, and pairing them with the comparator can partially
    # reconstruct the threshold label. A first-pass ablation exposed this by
    # producing implausible ~100% P@50, so that feature family is rejected.

    impressions = pd.to_numeric(frame["impressions_90d"], errors="coerce").fillna(0)
    clicks = pd.to_numeric(frame["clicks_90d"], errors="coerce").fillna(0)
    sessions = pd.to_numeric(frame["sessions_90d"], errors="coerce").fillna(0)
    imp_days = pd.to_numeric(frame["days_with_impressions"], errors="coerce").fillna(0)
    session_days = pd.to_numeric(frame["days_with_sessions"], errors="coerce").fillna(0)
    staleness = pd.to_numeric(frame["days_since_last_update"], errors="coerce").fillna(0).clip(lower=0)
    position = pd.to_numeric(frame["avg_position"], errors="coerce").replace(0, np.nan)

    frame["log_impressions_per_active_day"] = np.log1p(impressions / np.maximum(imp_days, 1))
    frame["log_clicks_per_active_day"] = np.log1p(clicks / np.maximum(imp_days, 1))
    frame["log_sessions_per_active_day"] = np.log1p(sessions / np.maximum(session_days, 1))
    frame["impression_day_coverage"] = imp_days / 90.0
    frame["session_day_coverage"] = session_days / 90.0
    frame["visibility_x_staleness"] = frame["log_impressions_90d"] * np.log1p(staleness)
    frame["position_x_visibility"] = position.fillna(position.median()) * frame["log_impressions_90d"]
    frame["engagement_x_sessions"] = frame["engagement_rate"] * frame["log_sessions_90d"]
    frame["content_depth_per_visibility"] = np.log1p(frame["word_count"].clip(lower=0)) / (
        1.0 + frame["log_impressions_90d"]
    )

    # Within-client ranks remove scale differences. No labels are used.
    rank_sources = {
        "impressions": "log_impressions_90d",
        "clicks": "log_clicks_90d",
        "sessions": "log_sessions_90d",
        "ctr": "ctr",
        "position": "avg_position",
        "staleness": "days_since_last_update",
        "engagement": "engagement_rate",
    }
    for short_name, source in rank_sources.items():
        values = pd.to_numeric(frame[source], errors="coerce")
        if source == "avg_position":
            values = values.replace(0, np.nan)
        frame[f"client_{short_name}_percentile"] = values.groupby(frame["client_id"]).rank(
            method="average", pct=True, na_option="keep"
        )
        client_median = values.groupby(frame["client_id"]).transform("median")
        frame[f"client_{short_name}_delta"] = values - client_median

    return frame.replace([np.inf, -np.inf], np.nan)


ENGINEERED_NUMERIC_FEATURES = [
    "missing_position_flag",
    "has_keyword_data",
    "has_word_count",
    "has_intent",
    "has_scroll_rate",
    "log_impressions_per_active_day",
    "log_clicks_per_active_day",
    "log_sessions_per_active_day",
    "impression_day_coverage",
    "session_day_coverage",
    "visibility_x_staleness",
    "position_x_visibility",
    "engagement_x_sessions",
    "content_depth_per_visibility",
    "client_impressions_percentile",
    "client_impressions_delta",
    "client_clicks_percentile",
    "client_clicks_delta",
    "client_sessions_percentile",
    "client_sessions_delta",
    "client_ctr_percentile",
    "client_ctr_delta",
    "client_position_percentile",
    "client_position_delta",
    "client_staleness_percentile",
    "client_staleness_delta",
    "client_engagement_percentile",
    "client_engagement_delta",
]


def build_matrix(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    requested = set(numeric_features) | set(categorical_features)
    forbidden_found = sorted(requested & FORBIDDEN_MODEL_COLUMNS)
    if forbidden_found:
        raise AssertionError(f"Forbidden feature request: {forbidden_found}")

    missing = sorted(requested - set(frame.columns))
    if missing:
        raise ValueError(f"Requested features missing from frame: {missing}")

    numeric = frame[numeric_features].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    categorical = frame[categorical_features].fillna("unknown").astype(str)
    encoded = pd.get_dummies(
        categorical,
        prefix=categorical_features,
        dummy_na=False,
        dtype=float,
    )
    matrix = pd.concat([numeric.reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)
    matrix.columns = [
        name.replace("[", "_").replace("]", "_").replace("<", "_").replace(">", "_")
        for name in matrix.columns
    ]
    leaked = sorted(set(matrix.columns) & FORBIDDEN_MODEL_COLUMNS)
    if leaked:
        raise AssertionError(f"Forbidden columns reached model matrix: {leaked}")
    return matrix, list(matrix.columns)


def classifier_factory(kind: str):
    if kind == "current_lgbm":
        return lgb.LGBMClassifier(
            class_weight="balanced",
            max_depth=6,
            n_estimators=200,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    if kind == "engineered_same_lgbm":
        return lgb.LGBMClassifier(
            class_weight="balanced",
            max_depth=6,
            n_estimators=200,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    if kind == "engineered_regularized_lgbm":
        return lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.035,
            n_estimators=650,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.15,
            reg_lambda=1.5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    raise KeyError(kind)


def fit_lambdarank(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    train_groups: pd.Series,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, lgb.LGBMRanker]:
    """Optimize within-client editorial queues without crossing group boundaries."""
    order = np.argsort(train_groups.astype(str).to_numpy(), kind="mergesort")
    sorted_x = x_train.iloc[order]
    sorted_y = y_train.iloc[order]
    sorted_groups = train_groups.iloc[order]
    group_sizes = sorted_groups.groupby(sorted_groups, sort=False).size().to_list()
    if max(group_sizes) > 10_000:
        raise ValueError("A LambdaRank client group exceeds LightGBM's 10k-row query limit")

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.035,
        n_estimators=500,
        num_leaves=31,
        min_child_samples=40,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=1.5,
        lambdarank_truncation_level=101,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    ranker.fit(sorted_x, sorted_y, group=group_sizes, eval_at=[20, 50, 100])
    return ranker.predict(x_test), ranker


def summarize_fold_metrics(fold_rows: list[dict[str, float | int]]) -> dict[str, float]:
    metrics = [
        "base_rate",
        "p_at_20",
        "p_at_50",
        "p_at_100",
        "lift_at_50",
        "roc_auc",
        "average_precision",
    ]
    summary: dict[str, float] = {}
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in fold_rows], dtype=float)
        summary[f"mean_{metric}"] = float(values.mean())
        summary[f"std_{metric}"] = float(values.std(ddof=1))
    return summary


def main() -> None:
    np.random.seed(RANDOM_STATE)
    prepared = pd.read_csv(FEATURE_PATH)
    raw = pd.read_csv(RAW_PATH)
    baseline = pd.read_csv(BASELINE_PATH)

    if len(prepared) != 30_000 or prepared["client_id"].nunique() != 32:
        raise AssertionError("Unexpected starter dataset shape")
    if not prepared["content_id"].is_unique:
        raise AssertionError("content_id must be unique")

    baseline_scores = (
        baseline.set_index("content_id")["baseline_refresh_score"]
        .reindex(prepared["content_id"])
        .to_numpy(dtype=float)
    )
    if np.isnan(baseline_scores).any():
        raise ValueError("Baseline queue does not align to feature vector")

    engineered = add_engineered_features(prepared, raw)
    base_matrix, base_names = build_matrix(
        prepared,
        BASE_NUMERIC_FEATURES,
        BASE_CATEGORICAL_FEATURES,
    )
    engineered_matrix, engineered_names = build_matrix(
        engineered,
        BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES,
        BASE_CATEGORICAL_FEATURES,
    )

    target = prepared["is_declining_label"].astype(int)
    groups = prepared["client_id"].fillna("unknown").astype(str)
    folds = list(GroupKFold(n_splits=N_SPLITS).split(base_matrix, target, groups=groups))

    candidates = {
        "R0_rule_baseline": {"matrix": None, "kind": "rule"},
        "E0_current_lgbm": {"matrix": base_matrix, "kind": "current_lgbm"},
        "E1_engineered_same_lgbm": {"matrix": engineered_matrix, "kind": "engineered_same_lgbm"},
        "E2_engineered_regularized_lgbm": {
            "matrix": engineered_matrix,
            "kind": "engineered_regularized_lgbm",
        },
        "E3_base_lambdarank": {"matrix": base_matrix, "kind": "lambdarank"},
        "E4_engineered_lambdarank": {"matrix": engineered_matrix, "kind": "lambdarank"},
    }

    results: dict[str, dict] = {}
    oof = pd.DataFrame(
        {
            "content_id": prepared["content_id"],
            "client_id": prepared["client_id"],
            "is_declining_label": target,
            "fold": np.zeros(len(prepared), dtype=int),
        }
    )
    feature_importance_accumulator: dict[str, np.ndarray] = {}

    print(f"Rows: {len(prepared):,}; clients: {groups.nunique()}; positive rate: {target.mean():.2%}")
    print(f"Base matrix: {base_matrix.shape}; engineered matrix: {engineered_matrix.shape}")
    print(f"Added engineered numeric features: {len(ENGINEERED_NUMERIC_FEATURES)}")
    print("Forbidden-column guard: PASS")

    for fold_number, (_, test_index) in enumerate(folds, start=1):
        oof.loc[test_index, "fold"] = fold_number

    for candidate_name, spec in candidates.items():
        print(f"\n=== {candidate_name} ===")
        fold_rows: list[dict[str, float | int]] = []
        oof_scores = np.full(len(prepared), np.nan, dtype=float)

        for fold_number, (train_index, test_index) in enumerate(folds, start=1):
            y_train = target.iloc[train_index]
            y_test = target.iloc[test_index]

            if spec["kind"] == "rule":
                scores = baseline_scores[test_index]
                fitted_model = None
            else:
                matrix = spec["matrix"]
                x_train = matrix.iloc[train_index]
                x_test = matrix.iloc[test_index]
                if spec["kind"] == "lambdarank":
                    scores, fitted_model = fit_lambdarank(
                        x_train,
                        y_train,
                        groups.iloc[train_index],
                        x_test,
                    )
                else:
                    fitted_model = classifier_factory(str(spec["kind"]))
                    fitted_model.fit(x_train, y_train)
                    scores = fitted_model.predict_proba(x_test)[:, 1]

                importance = np.asarray(fitted_model.feature_importances_, dtype=float)
                feature_importance_accumulator.setdefault(
                    candidate_name,
                    np.zeros_like(importance, dtype=float),
                )
                feature_importance_accumulator[candidate_name] += importance / N_SPLITS

            oof_scores[test_index] = scores
            metrics = evaluate_scores(y_test, scores)
            row: dict[str, float | int] = {
                "fold": fold_number,
                "test_rows": int(len(test_index)),
                "test_clients": int(groups.iloc[test_index].nunique()),
                **metrics,
            }
            fold_rows.append(row)
            print(
                f"fold={fold_number} rows={len(test_index):,} clients={row['test_clients']} "
                f"base={metrics['base_rate']:.1%} P@20={metrics['p_at_20']:.1%} "
                f"P@50={metrics['p_at_50']:.1%} P@100={metrics['p_at_100']:.1%} "
                f"AUC={metrics['roc_auc']:.4f} AP={metrics['average_precision']:.4f}"
            )

        if np.isnan(oof_scores).any():
            raise AssertionError(f"Missing OOF scores for {candidate_name}")
        oof[candidate_name] = oof_scores
        results[candidate_name] = {
            "folds": fold_rows,
            "summary": summarize_fold_metrics(fold_rows),
        }
        summary = results[candidate_name]["summary"]
        print(
            f"MEAN P@50={summary['mean_p_at_50']:.1%} +/- {summary['std_p_at_50']:.1%}; "
            f"AUC={summary['mean_roc_auc']:.4f}; AP={summary['mean_average_precision']:.4f}"
        )

    # Fixed, equal-weight rank blend of the two strongest classifier families.
    blend_components = ["E1_engineered_same_lgbm", "E2_engineered_regularized_lgbm"]
    rank_columns = []
    for component in blend_components:
        rank_columns.append(oof.groupby("fold")[component].rank(pct=True, method="average"))
    oof["E5_engineered_rank_blend"] = np.mean(np.vstack(rank_columns), axis=0)
    blend_fold_rows: list[dict[str, float | int]] = []
    print("\n=== E5_engineered_rank_blend ===")
    for fold_number, (_, test_index) in enumerate(folds, start=1):
        metrics = evaluate_scores(target.iloc[test_index], oof.loc[test_index, "E5_engineered_rank_blend"])
        row = {
            "fold": fold_number,
            "test_rows": int(len(test_index)),
            "test_clients": int(groups.iloc[test_index].nunique()),
            **metrics,
        }
        blend_fold_rows.append(row)
        print(
            f"fold={fold_number} P@20={metrics['p_at_20']:.1%} "
            f"P@50={metrics['p_at_50']:.1%} P@100={metrics['p_at_100']:.1%} "
            f"AUC={metrics['roc_auc']:.4f} AP={metrics['average_precision']:.4f}"
        )
    results["E5_engineered_rank_blend"] = {
        "folds": blend_fold_rows,
        "summary": summarize_fold_metrics(blend_fold_rows),
    }

    winner_name = max(
        (name for name in results if name != "R0_rule_baseline"),
        key=lambda name: (
            results[name]["summary"]["mean_p_at_50"],
            results[name]["summary"]["mean_p_at_20"],
            results[name]["summary"]["mean_average_precision"],
        ),
    )
    current_p50 = results["E0_current_lgbm"]["summary"]["mean_p_at_50"]
    winner_p50 = results[winner_name]["summary"]["mean_p_at_50"]

    comparison = []
    for name, payload in results.items():
        summary = payload["summary"]
        comparison.append(
            {
                "candidate": name,
                "mean_p_at_20": summary["mean_p_at_20"],
                "std_p_at_20": summary["std_p_at_20"],
                "mean_p_at_50": summary["mean_p_at_50"],
                "std_p_at_50": summary["std_p_at_50"],
                "mean_p_at_100": summary["mean_p_at_100"],
                "std_p_at_100": summary["std_p_at_100"],
                "mean_lift_at_50": summary["mean_lift_at_50"],
                "mean_roc_auc": summary["mean_roc_auc"],
                "std_roc_auc": summary["std_roc_auc"],
                "mean_average_precision": summary["mean_average_precision"],
                "std_average_precision": summary["std_average_precision"],
            }
        )

    importance_payload: dict[str, list[dict[str, float]]] = {}
    for candidate_name, importances in feature_importance_accumulator.items():
        names = (
            base_names
            if candidate_name in {"E0_current_lgbm", "E3_base_lambdarank"}
            else engineered_names
        )
        order = np.argsort(-importances)[:20]
        importance_payload[candidate_name] = [
            {"feature": names[index], "mean_importance": float(importances[index])}
            for index in order
        ]

    payload = {
        "experiment": "paired_grouped_feature_engineering",
        "estimand": "cross-client current-snapshot decline ranking; not future forecasting",
        "primary_metric": "mean precision@50 across five fixed GroupKFold client folds",
        "random_state": RANDOM_STATE,
        "n_rows": int(len(prepared)),
        "n_clients": int(groups.nunique()),
        "positive_rate": float(target.mean()),
        "base_feature_count": int(base_matrix.shape[1]),
        "engineered_feature_count": int(engineered_matrix.shape[1]),
        "engineered_numeric_features": ENGINEERED_NUMERIC_FEATURES,
        "forbidden_columns": sorted(FORBIDDEN_MODEL_COLUMNS),
        "winner": winner_name,
        "current_mean_p_at_50": float(current_p50),
        "winner_mean_p_at_50": float(winner_p50),
        "paired_improvement_pp": float((winner_p50 - current_p50) * 100.0),
        "comparison": comparison,
        "results": results,
        "top_feature_importance": importance_payload,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    oof.to_csv(OOF_PATH, index=False)

    print("\n=== FINAL COMPARISON ===")
    table = pd.DataFrame(comparison).sort_values("mean_p_at_50", ascending=False)
    print(
        table[
            [
                "candidate",
                "mean_p_at_20",
                "mean_p_at_50",
                "std_p_at_50",
                "mean_p_at_100",
                "mean_roc_auc",
                "mean_average_precision",
            ]
        ].to_string(index=False)
    )
    print(f"\nWinner: {winner_name}")
    print(f"Current P@50: {current_p50:.1%}")
    print(f"Winner P@50:  {winner_p50:.1%}")
    print(f"Paired improvement: {(winner_p50 - current_p50) * 100:+.1f}pp")
    print(f"Wrote metrics: {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Wrote ignored OOF diagnostics: {OOF_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
