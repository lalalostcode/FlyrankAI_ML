"""Build the ML-10 human-reviewed content action queue.

The module consumes cross-fitted W06 scores and the W04 observable context.
It never exports target, trend, fold, or shadow-model fields in the operational
queue. Aggregate JSON and PNG receipts are safe to use in the capstone paper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_STATE = 42
SELECTED_MODEL = "B2_current_rank_blend"
SHADOW_MODEL = "A11_current_plus_top20_equal"
FORBIDDEN_EXPORT_COLUMNS = {
    "is_declining_label",
    "trend_direction",
    "trend_pct",
    "shadow_rank_score",
    "fold",
}
EXPORT_COLUMNS = [
    "content_id",
    "client_id",
    "client_rank",
    "priority_tier",
    "model_rank_score",
    "reason_codes",
    "suggested_action",
    "impressions_90d",
    "sessions_90d",
    "avg_position",
    "ctr",
    "engagement_rate",
    "scroll_rate",
    "content_age_days",
    "days_since_last_update",
    "word_count",
]


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository without assuming the notebook working directory."""
    origin = (start or Path.cwd()).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / "skills" / "README.md").exists() and (
            candidate / "work" / "outputs"
        ).exists():
            return candidate
    raise FileNotFoundError("Run this notebook from inside the FlyRank repository.")


def load_inputs(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate the earlier homework/experiment artifacts."""
    output_dir = root / "work" / "outputs"
    oof_path = output_dir / "advanced_ranking_oof.csv"
    baseline_path = output_dir / "baseline_action_score.csv"
    results_path = output_dir / "advanced_ranking_results.json"
    paths = (oof_path, baseline_path, results_path)
    missing = [str(path.relative_to(root)) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prerequisite W04/W06 artifacts: "
            + ", ".join(missing)
            + ". Run the earlier experiment notebooks/scripts first."
        )

    oof = pd.read_csv(oof_path)
    baseline = pd.read_csv(baseline_path)
    experiment = json.loads(results_path.read_text(encoding="utf-8"))

    required_oof = {
        "content_id",
        "client_id",
        "fold",
        "is_declining_label",
        SELECTED_MODEL,
        SHADOW_MODEL,
    }
    required_context = {
        "content_id",
        "client_id",
        "impressions_90d",
        "sessions_90d",
        "avg_position",
        "ctr",
        "engagement_rate",
        "scroll_rate",
        "content_age_days",
        "days_since_last_update",
        "word_count",
    }
    assert required_oof.issubset(oof.columns), required_oof - set(oof.columns)
    assert required_context.issubset(baseline.columns), required_context - set(
        baseline.columns
    )
    assert oof["content_id"].is_unique and baseline["content_id"].is_unique
    assert len(oof) == experiment["rows"] == 30_000
    assert oof["client_id"].nunique() == experiment["clients"] == 32

    context_columns = sorted(required_context - {"client_id"})
    work = oof[
        [
            "content_id",
            "client_id",
            "fold",
            "is_declining_label",
            SELECTED_MODEL,
            SHADOW_MODEL,
        ]
    ].merge(
        baseline[context_columns],
        on="content_id",
        how="left",
        validate="one_to_one",
    )
    assert len(work) == len(oof) and work["avg_position"].notna().any()
    return work.rename(
        columns={
            SELECTED_MODEL: "model_rank_score",
            SHADOW_MODEL: "shadow_rank_score",
        }
    ), experiment


def build_queue(work: pd.DataFrame) -> pd.DataFrame:
    """Create reason codes, review prompts, and a top-50 queue per client."""
    valid_position = work["avg_position"].gt(0)
    reason_flags = pd.DataFrame(
        {
            "low_ctr_visible": (
                work["impressions_90d"].ge(500)
                & work["avg_position"].between(1, 20)
                & work["ctr"].between(0, 0.5, inclusive="left")
            ),
            "thin_visible": work["impressions_90d"].ge(250)
            & work["word_count"].between(1, 1199),
            "stale_visible": work["impressions_90d"].ge(500)
            & work["days_since_last_update"].ge(180),
            "page_one_aging": valid_position
            & work["avg_position"].le(10)
            & work["content_age_days"].ge(180),
            "weak_engagement_visible": work["sessions_90d"].ge(30)
            & (
                work["engagement_rate"].between(0, 30, inclusive="neither")
                | work["scroll_rate"].between(0, 30, inclusive="neither")
            ),
        },
        index=work.index,
    )
    ranked = work.copy()
    ranked["reason_codes"] = reason_flags.apply(
        lambda row: "|".join(
            name for name, active in row.items() if bool(active)
        )
        or "model_pattern_needs_diagnosis",
        axis=1,
    )
    ranked["suggested_action"] = np.select(
        [
            reason_flags["low_ctr_visible"],
            reason_flags["thin_visible"],
            reason_flags["stale_visible"] | reason_flags["page_one_aging"],
            reason_flags["weak_engagement_visible"],
        ],
        [
            "inspect_search_snippet",
            "review_depth_and_relevance",
            "review_facts_and_freshness",
            "review_intent_and_experience",
        ],
        default="diagnose_before_action",
    )
    ranked["client_rank"] = (
        ranked.groupby("client_id")["model_rank_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    ranked["priority_tier"] = np.select(
        [
            ranked["client_rank"].le(10),
            ranked["client_rank"].le(25),
            ranked["client_rank"].le(50),
        ],
        ["review_now", "review_next", "monitor"],
        default="outside_current_queue",
    )
    return (
        ranked.loc[ranked["client_rank"].le(50)]
        .sort_values(["client_id", "client_rank"], kind="stable")
        .reset_index(drop=True)
    )


def validate_scope(experiment: dict[str, Any]) -> tuple[dict[str, float], str]:
    """Return validated metrics and a claim constrained to the estimand."""
    benchmark = experiment["results"][SELECTED_MODEL]["summary"]
    assert (
        experiment["estimand"]
        == "cross-client current-snapshot decline ranking; not future forecasting"
    )
    assert experiment["target_achieved"] is False
    safe_claim = (
        "On five client-grouped folds in this single-snapshot dataset, the "
        f"selected model ranked the top 50 items at mean precision "
        f"{benchmark['mean_p_at_50']:.1%} ± "
        f"{benchmark['std_p_at_50']:.1%}, compared with a "
        f"{benchmark['mean_base_rate']:.1%} mean fold base rate. The queue is "
        "decision support for manual review, not a future forecast."
    )
    banned_claims = [
        "proves",
        "causes",
        "will increase",
        "predicts future",
        "guarantees",
    ]
    assert not any(term in safe_claim.lower() for term in banned_claims)
    return benchmark, safe_claim


def validate_queue(queue: pd.DataFrame) -> pd.DataFrame:
    """Enforce the operational-export and human-review policy."""
    allowed_actions = {
        "inspect_search_snippet",
        "review_depth_and_relevance",
        "review_facts_and_freshness",
        "review_intent_and_experience",
        "diagnose_before_action",
    }
    assert set(queue["suggested_action"]).issubset(allowed_actions)
    assert FORBIDDEN_EXPORT_COLUMNS.isdisjoint(EXPORT_COLUMNS)
    assert queue.groupby("client_id")["client_rank"].max().le(50).all()
    assert queue.groupby("client_id")["client_rank"].min().eq(1).all()
    assert queue["reason_codes"].notna().all()
    assert queue["suggested_action"].notna().all()
    return (
        queue.sort_values(
            ["suggested_action", "model_rank_score"],
            ascending=[True, False],
        )
        .groupby("suggested_action", group_keys=False)
        .head(2)
        .copy()
    )


def make_monitoring_plan(
    work: pd.DataFrame, benchmark: dict[str, float]
) -> tuple[pd.DataFrame, dict[str, float], float]:
    """Create reference profiles and explicit pause/revalidation triggers."""
    key_features = [
        "impressions_90d",
        "sessions_90d",
        "avg_position",
        "ctr",
        "engagement_rate",
        "days_since_last_update",
        "word_count",
    ]
    reference_missingness = {
        column: float(work[column].isna().mean()) for column in key_features
    }
    p50_pause_floor = max(0.0, benchmark["mean_p_at_50"] - 0.10)
    plan = pd.DataFrame(
        [
            {
                "signal": "schema / required fields",
                "check": "every scoring run",
                "alert": "any missing or renamed required field",
                "response": "stop scoring and repair upstream contract",
            },
            {
                "signal": "feature missingness",
                "check": "every scoring run",
                "alert": ">5 percentage-point increase vs reference",
                "response": "investigate source; do not zero-fill blindly",
            },
            {
                "signal": "score or action mix",
                "check": "every scoring run",
                "alert": ">15 percentage-point mix shift",
                "response": "review drift and thresholds before release",
            },
            {
                "signal": "mature-label P@50",
                "check": "each labeled cycle",
                "alert": f"<{p50_pause_floor:.1%} or >10pp below reference",
                "response": "diagnose; pause after two consecutive alerts",
            },
            {
                "signal": "label base rate",
                "check": "each labeled cycle",
                "alert": ">10 percentage-point shift",
                "response": "report recalibrated context; revalidate ranking",
            },
            {
                "signal": "new client / future period",
                "check": "before deployment",
                "alert": "outside validated snapshot/population",
                "response": "shadow mode plus grouped/temporal evaluation",
            },
        ]
    )
    return plan, reference_missingness, p50_pause_floor


def export_artifacts(
    root: Path,
    work: pd.DataFrame,
    queue: pd.DataFrame,
    experiment: dict[str, Any],
    benchmark: dict[str, float],
    reference_missingness: dict[str, float],
    p50_pause_floor: float,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Write, reload, and validate the operational and paper artifacts."""
    output_dir = root / "work" / "outputs"
    queue_path = output_dir / "action_playbook_queue.csv"
    metrics_path = output_dir / "action_playbook_metrics.json"
    figure_path = output_dir / "action_playbook_summary.png"

    queue_export = queue[EXPORT_COLUMNS].copy()
    queue_export.to_csv(queue_path, index=False)
    action_counts = queue_export["suggested_action"].value_counts()
    reason_counts = (
        queue_export["reason_codes"].str.split("|").explode().value_counts()
    )
    priority_counts = queue_export["priority_tier"].value_counts()

    receipts = {
        "artifact": "content_action_playbook_v1",
        "scope": experiment["estimand"],
        "selected_model": SELECTED_MODEL,
        "selection_status": "conservative benchmark retained",
        "shadow_model": SHADOW_MODEL,
        "shadow_status": (
            f"not promoted; same-fold gain {experiment['improvement_pp']:+.1f}pp, "
            f"below +{experiment['target_improvement_pp']:.1f}pp target"
        ),
        "random_state": RANDOM_STATE,
        "source_rows": int(len(work)),
        "client_groups": int(work["client_id"].nunique()),
        "queue_rows": int(len(queue_export)),
        "queue_policy": {
            "max_per_client": 50,
            "review_now": "ranks 1-10",
            "review_next": "ranks 11-25",
            "monitor": "ranks 26-50",
        },
        "validation": {
            "folds": 5,
            "mean_fold_base_rate": float(benchmark["mean_base_rate"]),
            "mean_precision_at_20": float(benchmark["mean_p_at_20"]),
            "mean_precision_at_50": float(benchmark["mean_p_at_50"]),
            "std_precision_at_50": float(benchmark["std_p_at_50"]),
            "mean_precision_at_100": float(benchmark["mean_p_at_100"]),
            "mean_lift_at_50": float(benchmark["mean_lift_at_50"]),
            "mean_roc_auc": float(benchmark["mean_roc_auc"]),
            "mean_average_precision": float(benchmark["mean_average_precision"]),
            "fold_precision_at_50": [
                float(row["p_at_50"])
                for row in experiment["results"][SELECTED_MODEL]["folds"]
            ],
        },
        "priority_counts": {str(k): int(v) for k, v in priority_counts.items()},
        "action_counts": {str(k): int(v) for k, v in action_counts.items()},
        "reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
        "reference_missingness": reference_missingness,
        "monitoring": {
            "p50_pause_floor": float(p50_pause_floor),
            "performance_alerts_before_pause": 2,
            "missingness_shift_pp": 5.0,
            "mix_shift_pp": 15.0,
            "base_rate_shift_pp": 10.0,
        },
        "operational_export_excludes": sorted(FORBIDDEN_EXPORT_COLUMNS),
    }
    metrics_path.write_text(json.dumps(receipts, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    action_counts.sort_values().plot.barh(ax=axes[0], color="#386cb0")
    axes[0].set_title("Suggested review actions")
    axes[0].set_xlabel("Queue rows")
    axes[0].set_ylabel("")
    reason_counts.sort_values().plot.barh(ax=axes[1], color="#7fc97f")
    axes[1].set_title("Observable reason codes")
    axes[1].set_xlabel("Queue rows (codes can overlap)")
    axes[1].set_ylabel("")
    fig.suptitle(
        "Content action playbook — aggregate queue composition",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    verified_queue = pd.read_csv(queue_path)
    verified_receipts = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(verified_queue) == len(queue_export) == verified_receipts["queue_rows"]
    assert FORBIDDEN_EXPORT_COLUMNS.isdisjoint(verified_queue.columns)
    assert verified_queue["client_rank"].between(1, 50).all()
    assert figure_path.exists() and figure_path.stat().st_size > 10_000
    return queue_path, metrics_path, figure_path, receipts
