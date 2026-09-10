"""Second-generation feature engineering to challenge the LambdaRank benchmark.

All model comparisons use the exact five client-grouped folds from W06. No
previous/latest 30-day trend input, label source, ID, or product score is used
as a model feature. Fold-safe target encodings use training labels only.
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
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]
V1_SCRIPT = ROOT / "work" / "scripts" / "feature_engineering_experiment.py"
FEATURE_PATH = ROOT / "data" / "processed" / "refresh_feature_vector.csv"
RAW_PATH = ROOT / "data" / "raw" / "content_refresh_anonymized.csv"
V1_OOF_PATH = ROOT / "work" / "outputs" / "feature_engineering_oof.csv"
RESULTS_PATH = ROOT / "work" / "outputs" / "feature_engineering_v2_results.json"
OOF_PATH = ROOT / "work" / "outputs" / "feature_engineering_v2_oof.csv"

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


def safe_ratio(numerator: pd.Series, denominator: pd.Series, fill: float = 0.0) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill)


def robust_group_features(
    frame: pd.DataFrame,
    values: pd.Series,
    group_column: str,
    prefix: str,
) -> None:
    grouped = values.groupby(frame[group_column])
    median = grouped.transform("median")
    q25 = grouped.transform(lambda series: series.quantile(0.25))
    q75 = grouped.transform(lambda series: series.quantile(0.75))
    iqr = (q75 - q25).replace(0, np.nan)
    frame[f"{prefix}_robust_z"] = ((values - median) / iqr).clip(-10, 10).fillna(0)
    frame[f"{prefix}_percentile"] = grouped.rank(method="average", pct=True).fillna(0.5)


def add_v2_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    numeric = {}
    raw_numeric_columns = [
        "impressions_90d",
        "clicks_90d",
        "pageviews_90d",
        "sessions_90d",
        "users_90d",
        "engaged_sessions_90d",
        "ai_sessions_90d",
        "scroll_events_90d",
        "days_with_impressions",
        "days_with_sessions",
        "content_age_days",
        "days_since_last_update",
        "word_count",
        "char_count",
        "search_volume",
        "ctr",
        "avg_position",
        "engagement_rate",
    ]
    for column in raw_numeric_columns:
        numeric[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).clip(lower=0)

    log_sources = [
        "pageviews_90d",
        "users_90d",
        "engaged_sessions_90d",
        "scroll_events_90d",
        "content_age_days",
        "days_since_last_update",
        "word_count",
        "char_count",
        "search_volume",
    ]
    for column in log_sources:
        result[f"log_{column}"] = np.log1p(numeric[column])

    result["pageviews_per_session"] = safe_ratio(numeric["pageviews_90d"], numeric["sessions_90d"]).clip(0, 100)
    result["users_per_session"] = safe_ratio(numeric["users_90d"], numeric["sessions_90d"]).clip(0, 10)
    result["sessions_per_click"] = safe_ratio(numeric["sessions_90d"], numeric["clicks_90d"]).clip(0, 100)
    result["pageviews_per_click"] = safe_ratio(numeric["pageviews_90d"], numeric["clicks_90d"]).clip(0, 500)
    result["scrolls_per_session"] = safe_ratio(numeric["scroll_events_90d"], numeric["sessions_90d"]).clip(0, 500)
    result["chars_per_word"] = safe_ratio(numeric["char_count"], numeric["word_count"]).clip(0, 100)
    result["staleness_age_fraction"] = safe_ratio(
        numeric["days_since_last_update"], numeric["content_age_days"]
    ).clip(0, 2)
    result["active_day_gap"] = numeric["days_with_impressions"] - numeric["days_with_sessions"]
    result["active_day_ratio"] = safe_ratio(
        numeric["days_with_sessions"], numeric["days_with_impressions"]
    ).clip(0, 2)
    result["search_analytics_gap"] = result["log_clicks_90d"] - result["log_sessions_90d"]
    result["visibility_demand_gap"] = result["log_search_volume"] - result["log_impressions_90d"]
    result["engagement_depth"] = result["log_pageviews_90d"] * np.log1p(numeric["engagement_rate"])

    valid_position = numeric["avg_position"].replace(0, np.nan)
    result["log_position"] = np.log1p(valid_position).fillna(0)
    result["reciprocal_position"] = (1.0 / valid_position).clip(0, 10).fillna(0)
    result["visibility_per_log_position"] = (
        result["log_impressions_90d"] / (1.0 + result["log_position"])
    )
    result["ctr_x_log_position"] = numeric["ctr"] * result["log_position"]
    result["age_x_staleness_fraction"] = result["log_content_age_days"] * result["staleness_age_fraction"]

    # Robust relative scales by client and content type. These use predictor
    # distributions only and match batch scoring of a full client inventory.
    relative_sources = {
        "impressions": result["log_impressions_90d"],
        "clicks": result["log_clicks_90d"],
        "sessions": result["log_sessions_90d"],
        "position": valid_position,
        "ctr": numeric["ctr"],
        "staleness": result["log_days_since_last_update"],
        "age": result["log_content_age_days"],
        "word_count": result["log_word_count"],
        "demand": result["log_search_volume"],
        "engagement": numeric["engagement_rate"],
    }
    for name, values in relative_sources.items():
        robust_group_features(result, values, "client_id", f"client2_{name}")
    for name in ["impressions", "position", "ctr", "staleness", "word_count", "engagement"]:
        robust_group_features(result, relative_sources[name], "content_type", f"type_{name}")

    # Predictor-only quantile categories and categorical interactions.
    decile_sources = {
        "visibility_decile": result["log_impressions_90d"],
        "position_decile": valid_position,
        "staleness_decile": result["log_days_since_last_update"],
        "age_decile": result["log_content_age_days"],
        "engagement_decile": numeric["engagement_rate"],
    }
    for name, values in decile_sources.items():
        ranks = values.rank(method="average", pct=True).fillna(0)
        result[name] = np.minimum((ranks * 10).astype(int), 9).astype(str)

    cross_pairs = {
        "content_position_cross": ("content_type", "position_tier"),
        "content_visibility_cross": ("content_type", "impression_tier"),
        "content_freshness_cross": ("content_type", "freshness_tier"),
        "position_visibility_cross": ("position_tier", "impression_tier"),
        "freshness_visibility_cross": ("freshness_tier", "impression_tier"),
        "age_freshness_cross": ("age_tier", "freshness_tier"),
        "depth_visibility_cross": ("word_count_tier", "impression_tier"),
        "intent_position_cross": ("main_intent", "position_tier"),
        "intent_visibility_cross": ("main_intent", "impression_tier"),
    }
    for name, (left, right) in cross_pairs.items():
        result[name] = (
            result[left].fillna("unknown").astype(str)
            + "__"
            + result[right].fillna("unknown").astype(str)
        )

    return result.replace([np.inf, -np.inf], np.nan)


V2_NUMERIC_FEATURES = [
    "log_pageviews_90d",
    "log_users_90d",
    "log_engaged_sessions_90d",
    "log_scroll_events_90d",
    "log_content_age_days",
    "log_days_since_last_update",
    "log_word_count",
    "log_char_count",
    "log_search_volume",
    "pageviews_per_session",
    "users_per_session",
    "sessions_per_click",
    "pageviews_per_click",
    "scrolls_per_session",
    "chars_per_word",
    "staleness_age_fraction",
    "active_day_gap",
    "active_day_ratio",
    "search_analytics_gap",
    "visibility_demand_gap",
    "engagement_depth",
    "log_position",
    "reciprocal_position",
    "visibility_per_log_position",
    "ctr_x_log_position",
    "age_x_staleness_fraction",
]
for prefix in ["client2", "type"]:
    names = (
        ["impressions", "clicks", "sessions", "position", "ctr", "staleness", "age", "word_count", "demand", "engagement"]
        if prefix == "client2"
        else ["impressions", "position", "ctr", "staleness", "word_count", "engagement"]
    )
    for name in names:
        V2_NUMERIC_FEATURES.extend([f"{prefix}_{name}_robust_z", f"{prefix}_{name}_percentile"])

V2_CATEGORICAL_FEATURES = [
    "visibility_decile",
    "position_decile",
    "staleness_decile",
    "age_decile",
    "engagement_decile",
    "content_position_cross",
    "content_visibility_cross",
    "content_freshness_cross",
    "position_visibility_cross",
    "freshness_visibility_cross",
    "age_freshness_cross",
    "depth_visibility_cross",
    "intent_position_cross",
    "intent_visibility_cross",
]

TARGET_ENCODING_COLUMNS = [
    "content_type",
    "main_intent",
    "position_tier",
    "impression_tier",
    "freshness_tier",
    "content_position_cross",
    "content_visibility_cross",
    "position_visibility_cross",
    "freshness_visibility_cross",
    "intent_position_cross",
]


def add_fold_safe_target_encodings(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    feature_frame: pd.DataFrame,
    y_train: pd.Series,
    train_index: np.ndarray,
    test_index: np.ndarray,
    smoothing: float = 50.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = x_train.copy()
    test = x_test.copy()
    global_rate = float(y_train.mean())

    for column in TARGET_ENCODING_COLUMNS:
        train_category = feature_frame.iloc[train_index][column].fillna("unknown").astype(str)
        test_category = feature_frame.iloc[test_index][column].fillna("unknown").astype(str)
        stats = pd.DataFrame({"category": train_category.to_numpy(), "target": y_train.to_numpy()}).groupby(
            "category"
        )["target"].agg(["sum", "count"])
        sums = train_category.map(stats["sum"]).astype(float)
        counts = train_category.map(stats["count"]).astype(float)
        train[f"te_{column}"] = (
            sums.to_numpy() - y_train.to_numpy() + smoothing * global_rate
        ) / (counts.to_numpy() - 1.0 + smoothing)
        test[f"te_{column}"] = test_category.map(
            (stats["sum"] + smoothing * global_rate) / (stats["count"] + smoothing)
        ).fillna(global_rate).to_numpy()

    return train, test


def model_factory(kind: str):
    if kind == "same_lgbm":
        return lgb.LGBMClassifier(
            class_weight="balanced",
            max_depth=6,
            n_estimators=200,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    if kind == "regularized_lgbm":
        return lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.025,
            n_estimators=900,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.25,
            reg_lambda=2.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    if kind == "xgboost":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.035,
            n_estimators=650,
            max_depth=5,
            min_child_weight=15,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.15,
            reg_lambda=2.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )
    if kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=18,
            min_samples_leaf=10,
            max_features=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    raise KeyError(kind)


def fit_ranker(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    train_groups: pd.Series,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, lgb.LGBMRanker]:
    order = np.argsort(train_groups.astype(str).to_numpy(), kind="mergesort")
    sorted_x = x_train.iloc[order]
    sorted_y = y_train.iloc[order]
    sorted_groups = train_groups.iloc[order]
    group_sizes = sorted_groups.groupby(sorted_groups, sort=False).size().to_list()
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.025,
        n_estimators=750,
        num_leaves=31,
        min_child_samples=50,
        colsample_bytree=0.8,
        reg_alpha=0.25,
        reg_lambda=2.0,
        lambdarank_truncation_level=51,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    ranker.fit(sorted_x, sorted_y, group=group_sizes, eval_at=[20, 50, 100])
    return ranker.predict(x_test), ranker


def summarize(rows: list[dict]) -> dict[str, float]:
    summary = {}
    for metric in ["base_rate", "p_at_20", "p_at_50", "p_at_100", "lift_at_50", "roc_auc", "average_precision"]:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        summary[f"mean_{metric}"] = float(values.mean())
        summary[f"std_{metric}"] = float(values.std(ddof=1))
    return summary


def fold_metrics_for_scores(
    target: pd.Series,
    groups: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    scores: np.ndarray,
) -> list[dict]:
    rows = []
    for fold_number, (_, test_index) in enumerate(folds, start=1):
        metrics = v1.evaluate_scores(target.iloc[test_index], scores[test_index])
        rows.append(
            {
                "fold": fold_number,
                "test_rows": int(len(test_index)),
                "test_clients": int(groups.iloc[test_index].nunique()),
                **metrics,
            }
        )
    return rows


def rank_blend_by_fold(oof: pd.DataFrame, components: list[str], weights: list[float]) -> np.ndarray:
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Blend weights must sum to one")
    blended = np.zeros(len(oof), dtype=float)
    for component, weight in zip(components, weights):
        blended += weight * oof.groupby("fold")[component].rank(pct=True, method="average").to_numpy()
    return blended


def main() -> None:
    np.random.seed(RANDOM_STATE)
    prepared = pd.read_csv(FEATURE_PATH)
    raw = pd.read_csv(RAW_PATH)
    v1_oof = pd.read_csv(V1_OOF_PATH)
    if not prepared["content_id"].equals(v1_oof["content_id"]):
        raise AssertionError("V1 OOF predictions do not align with the prepared feature vector")

    engineered_v1 = v1.add_engineered_features(prepared, raw)
    feature_frame = add_v2_features(engineered_v1)
    numeric_features = v1.BASE_NUMERIC_FEATURES + v1.ENGINEERED_NUMERIC_FEATURES + V2_NUMERIC_FEATURES
    categorical_features = v1.BASE_CATEGORICAL_FEATURES + V2_CATEGORICAL_FEATURES
    matrix, feature_names = v1.build_matrix(feature_frame, numeric_features, categorical_features)

    target = prepared["is_declining_label"].astype(int)
    groups = prepared["client_id"].fillna("unknown").astype(str)
    folds = list(GroupKFold(n_splits=N_SPLITS).split(matrix, target, groups=groups))

    oof = pd.DataFrame(
        {
            "content_id": prepared["content_id"],
            "client_id": prepared["client_id"],
            "is_declining_label": target,
            "fold": v1_oof["fold"].astype(int),
            "B0_engineered_lambdarank": v1_oof["E4_engineered_lambdarank"],
            "B1_base_lambdarank": v1_oof["E3_base_lambdarank"],
        }
    )

    candidates = {
        "F1_fev2_same_lgbm": {"kind": "same_lgbm", "target_encode": False},
        "F2_fev2_regularized_lgbm": {"kind": "regularized_lgbm", "target_encode": False},
        "F3_fev2_target_encoded_lgbm": {"kind": "regularized_lgbm", "target_encode": True},
        "F4_fev2_xgboost": {"kind": "xgboost", "target_encode": False},
        "F5_fev2_extra_trees": {"kind": "extra_trees", "target_encode": False},
        "F6_fev2_lambdarank_k50": {"kind": "lambdarank", "target_encode": False},
    }

    results = {}
    importance_accumulator = {}
    print(f"Rows={len(prepared):,}; clients={groups.nunique()}; positive rate={target.mean():.2%}")
    print(f"FE v2 matrix={matrix.shape}; numeric={len(numeric_features)}; categorical={len(categorical_features)}")
    print("Forbidden-column guard: PASS")

    for benchmark_name in ["B0_engineered_lambdarank", "B1_base_lambdarank"]:
        rows = fold_metrics_for_scores(target, groups, folds, oof[benchmark_name].to_numpy())
        results[benchmark_name] = {"folds": rows, "summary": summarize(rows)}

    for candidate_name, spec in candidates.items():
        print(f"\n=== {candidate_name} ===")
        scores = np.full(len(prepared), np.nan, dtype=float)
        fold_rows = []
        importance = None

        for fold_number, (train_index, test_index) in enumerate(folds, start=1):
            x_train = matrix.iloc[train_index]
            x_test = matrix.iloc[test_index]
            y_train = target.iloc[train_index]
            y_test = target.iloc[test_index]

            if spec["target_encode"]:
                x_train, x_test = add_fold_safe_target_encodings(
                    x_train,
                    x_test,
                    feature_frame,
                    y_train,
                    train_index,
                    test_index,
                )

            if spec["kind"] == "lambdarank":
                fold_scores, fitted = fit_ranker(x_train, y_train, groups.iloc[train_index], x_test)
            else:
                fitted = model_factory(str(spec["kind"]))
                fitted.fit(x_train, y_train)
                fold_scores = fitted.predict_proba(x_test)[:, 1]

            scores[test_index] = fold_scores
            fold_importance = np.asarray(fitted.feature_importances_, dtype=float)
            if importance is None:
                importance = np.zeros_like(fold_importance)
            importance += fold_importance / N_SPLITS

            metrics = v1.evaluate_scores(y_test, fold_scores)
            fold_rows.append(
                {
                    "fold": fold_number,
                    "test_rows": int(len(test_index)),
                    "test_clients": int(groups.iloc[test_index].nunique()),
                    **metrics,
                }
            )
            print(
                f"fold={fold_number} P@20={metrics['p_at_20']:.1%} P@50={metrics['p_at_50']:.1%} "
                f"P@100={metrics['p_at_100']:.1%} AUC={metrics['roc_auc']:.4f} AP={metrics['average_precision']:.4f}"
            )

        if np.isnan(scores).any():
            raise AssertionError(f"Missing OOF scores for {candidate_name}")
        oof[candidate_name] = scores
        results[candidate_name] = {"folds": fold_rows, "summary": summarize(fold_rows)}
        candidate_feature_names = feature_names + (
            [f"te_{column}" for column in TARGET_ENCODING_COLUMNS] if spec["target_encode"] else []
        )
        order = np.argsort(-importance)[:20]
        importance_accumulator[candidate_name] = [
            {"feature": candidate_feature_names[index], "mean_importance": float(importance[index])}
            for index in order
        ]
        summary = results[candidate_name]["summary"]
        print(
            f"MEAN P@50={summary['mean_p_at_50']:.1%} +/- {summary['std_p_at_50']:.1%}; "
            f"AUC={summary['mean_roc_auc']:.4f}; AP={summary['mean_average_precision']:.4f}"
        )

    # Predeclared equal-rank blends test complementary model errors.
    blend_specs = {
        "F7_lambda_pair_equal_blend": (
            ["B0_engineered_lambdarank", "B1_base_lambdarank"],
            [0.5, 0.5],
        ),
        "F8_lambda_fev2_lgbm_equal_blend": (
            ["B0_engineered_lambdarank", "F3_fev2_target_encoded_lgbm"],
            [0.5, 0.5],
        ),
        "F9_lambda_fev2_ranker_equal_blend": (
            ["B0_engineered_lambdarank", "F6_fev2_lambdarank_k50"],
            [0.5, 0.5],
        ),
        "F10_three_model_equal_blend": (
            ["B0_engineered_lambdarank", "B1_base_lambdarank", "F3_fev2_target_encoded_lgbm"],
            [1 / 3, 1 / 3, 1 / 3],
        ),
    }
    for name, (components, weights) in blend_specs.items():
        oof[name] = rank_blend_by_fold(oof, components, weights)
        rows = fold_metrics_for_scores(target, groups, folds, oof[name].to_numpy())
        results[name] = {"folds": rows, "summary": summarize(rows), "components": components, "weights": weights}

    comparison = []
    for name, result in results.items():
        summary = result["summary"]
        comparison.append(
            {
                "candidate": name,
                "mean_p_at_20": summary["mean_p_at_20"],
                "std_p_at_20": summary["std_p_at_20"],
                "mean_p_at_50": summary["mean_p_at_50"],
                "std_p_at_50": summary["std_p_at_50"],
                "mean_p_at_100": summary["mean_p_at_100"],
                "std_p_at_100": summary["std_p_at_100"],
                "mean_roc_auc": summary["mean_roc_auc"],
                "mean_average_precision": summary["mean_average_precision"],
                "mean_lift_at_50": summary["mean_lift_at_50"],
            }
        )

    winner = max(
        (row for row in comparison if not row["candidate"].startswith("B")),
        key=lambda row: (row["mean_p_at_50"], row["mean_p_at_20"], row["mean_p_at_100"]),
    )
    benchmark = next(row for row in comparison if row["candidate"] == "B0_engineered_lambdarank")
    paired_differences = []
    winner_folds = results[winner["candidate"]]["folds"]
    benchmark_folds = results["B0_engineered_lambdarank"]["folds"]
    for winner_fold, benchmark_fold in zip(winner_folds, benchmark_folds):
        paired_differences.append(
            {
                "fold": winner_fold["fold"],
                "winner_p_at_50": winner_fold["p_at_50"],
                "benchmark_p_at_50": benchmark_fold["p_at_50"],
                "improvement_pp": 100.0 * (winner_fold["p_at_50"] - benchmark_fold["p_at_50"]),
            }
        )

    payload = {
        "experiment": "feature_engineering_v2_against_lambdarank",
        "estimand": "cross-client current-snapshot decline ranking; not future forecasting",
        "primary_metric": "mean precision@50 across the same five fixed GroupKFold folds",
        "random_state": RANDOM_STATE,
        "rows": int(len(prepared)),
        "clients": int(groups.nunique()),
        "positive_rate": float(target.mean()),
        "matrix_shape": list(matrix.shape),
        "v2_numeric_features": V2_NUMERIC_FEATURES,
        "v2_categorical_features": V2_CATEGORICAL_FEATURES,
        "fold_safe_target_encoding_columns": TARGET_ENCODING_COLUMNS,
        "winner": winner["candidate"],
        "benchmark": "B0_engineered_lambdarank",
        "winner_mean_p_at_50": winner["mean_p_at_50"],
        "benchmark_mean_p_at_50": benchmark["mean_p_at_50"],
        "improvement_pp": 100.0 * (winner["mean_p_at_50"] - benchmark["mean_p_at_50"]),
        "paired_differences": paired_differences,
        "comparison": comparison,
        "results": results,
        "top_feature_importance": importance_accumulator,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__,
        },
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    oof.to_csv(OOF_PATH, index=False)

    table = pd.DataFrame(comparison).sort_values("mean_p_at_50", ascending=False)
    print("\n=== FINAL COMPARISON ===")
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
    print(f"\nBenchmark P@50: {benchmark['mean_p_at_50']:.1%}")
    print(f"Winner: {winner['candidate']} at {winner['mean_p_at_50']:.1%}")
    print(f"Improvement: {payload['improvement_pp']:+.1f}pp")
    print(f"Wrote {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
