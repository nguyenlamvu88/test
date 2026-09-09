from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODEL_FEATURES = {
    "A — Market only": [
        "return_1d", "return_5d_lag", "rvol_20d", "dollar_volume", "realized_vol_20d"
    ],
    "B — Social only": [
        "mentions_1d", "mentions_3d", "unique_authors_3d", "communities_3d", "attention_accel"
    ],
    "D — Combined": [
        "return_1d", "return_5d_lag", "rvol_20d", "dollar_volume", "realized_vol_20d",
        "mentions_1d", "mentions_3d", "unique_authors_3d", "communities_3d", "attention_accel",
    ],
}


def walk_forward_validate(
    dataset: pd.DataFrame,
    holdout_year: int = 2026,
    min_train_rows: int = 200,
    top_fraction: float = 0.05,
    embargo_days: int = 10,
) -> pd.DataFrame:
    """Train on prior years and score each later year, leaving holdout untouched."""
    if dataset.empty:
        return pd.DataFrame()
    data = dataset.copy()
    data["asof_date"] = pd.to_datetime(data["asof_date"])
    data["year"] = data["asof_date"].dt.year
    data["target"] = (data["outcome_class"] == "clean_50").astype(int)
    data = data[data["outcome_class"] != "ambiguous_50"].copy()
    years = sorted(year for year in data["year"].unique() if year < holdout_year)
    rows = []
    for test_year in years[1:]:
        test_start = pd.Timestamp(year=int(test_year), month=1, day=1)
        train = data[data["asof_date"] < test_start - pd.Timedelta(days=embargo_days)]
        test = data[data["year"] == test_year]
        if len(train) < min_train_rows or test.empty or train["target"].nunique() < 2:
            continue
        for name, features in MODEL_FEATURES.items():
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            )
            model.fit(train[features], train["target"])
            probability = model.predict_proba(test[features])[:, 1]
            y_true = test["target"].to_numpy()
            k = max(1, int(np.ceil(len(test) * top_fraction)))
            top = np.argsort(probability)[::-1][:k]
            precision_k = float(y_true[top].mean())
            base_rate = float(y_true.mean())
            rows.append({
                "test_year": int(test_year),
                "model_name": name,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "positives": int(y_true.sum()),
                "roc_auc": _safe_auc(y_true, probability),
                "average_precision": _safe_ap(y_true, probability),
                "precision_at_k": precision_k,
                "lift_at_k": precision_k / base_rate if base_rate > 0 else None,
                "brier_score": float(brier_score_loss(y_true, probability)),
            })
    return pd.DataFrame(rows)


def validation_decision(metrics: pd.DataFrame) -> dict[str, str | float | None]:
    if metrics.empty:
        return {"decision": "INSUFFICIENT DATA", "reason": "No eligible walk-forward folds."}
    pivot = metrics.groupby("model_name")["average_precision"].mean()
    combined = pivot.get("D — Combined")
    baseline = pivot.get("A — Market only")
    if combined is None or baseline is None:
        return {"decision": "INSUFFICIENT DATA", "reason": "Baseline comparison is incomplete."}
    lift = float(combined / baseline) if baseline > 0 else None
    if lift is not None and lift >= 1.20:
        return {"decision": "ADVANCE", "ap_lift": lift, "reason": "Combined out-of-time AP is at least 20% above market-only."}
    return {"decision": "NO-GO / REVISE", "ap_lift": lift, "reason": "Combined model has not cleared the predeclared 20% AP-lift gate."}


def _safe_auc(y_true: np.ndarray, probability: np.ndarray) -> float | None:
    return float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None


def _safe_ap(y_true: np.ndarray, probability: np.ndarray) -> float | None:
    return float(average_precision_score(y_true, probability)) if y_true.sum() else None
