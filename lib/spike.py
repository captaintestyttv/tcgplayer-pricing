import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

from lib.config import (
    N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, VALIDATION_SPLIT, RANDOM_SEED,
    SAMPLE_WEIGHT_FEATURE, get_logger,
)

log = get_logger(__name__)

FEATURE_COLS = [
    # Original
    "rarity_rank",
    "num_printings",
    "set_age_days",
    "formats_legal_count",
    "price_momentum_7d",
    "price_volatility_30d",
    "current_price",
    # Phase 1: card metadata
    "edhrec_rank",
    "edhrec_saltiness",
    "is_reserved_list",
    "is_legendary",
    "is_creature",
    "color_count",
    "keyword_count",
    "mana_value",
    "subtype_count",
    # Phase 2: foil & buylist
    "foil_to_normal_ratio",
    "buylist_ratio",
    "buylist_momentum_7d",
    # Phase 3: cluster
    "cluster_momentum_7d",
    # Phase 4: change detection
    "recently_reprinted",
    "legality_changed",
    # Phase 5: set release signals
    "set_release_proximity",
    "spoiler_season",
    # Phase 6: derived signals
    "price_range_30d",
    "days_since_last_price_change",
    "set_card_count",
    "price_percentile",
]


def train(rows: list[dict], model_path: str, device: str = "cpu") -> None:
    """Train XGBoost spike classifier and save to model_path."""
    if not rows:
        raise ValueError("No training data provided")

    df = pd.DataFrame(rows)
    X = df[FEATURE_COLS].fillna(0)
    y = df["spike"]

    # Sample weights: sqrt(price) so high-value cards matter more
    raw_weights = df[SAMPLE_WEIGHT_FEATURE].clip(lower=0.01)
    sample_weights = np.sqrt(raw_weights)

    # Class imbalance: scale_pos_weight = #negative / #positive
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    spw = n_neg / max(n_pos, 1)

    # Validation split for metrics (stratified to preserve spike ratio)
    val_metrics = {}
    if len(y) >= 20 and n_pos >= 2:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_SPLIT,
                                     random_state=RANDOM_SEED)
        train_idx, val_idx = next(sss.split(X, y))
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        val_model = xgb.XGBClassifier(
            device=device,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            scale_pos_weight=spw,
            eval_metric="logloss",
            verbosity=0,
        )
        val_model.fit(X_train, y_train, sample_weight=sample_weights.iloc[train_idx])
        y_pred = val_model.predict(X_val)
        y_prob = val_model.predict_proba(X_val)[:, 1]

        val_metrics = {
            "val_accuracy": round(float(accuracy_score(y_val, y_pred)), 4),
            "val_auc": round(float(roc_auc_score(y_val, y_prob)), 4) if len(set(y_val)) > 1 else None,
            "val_precision": round(float(precision_score(y_val, y_pred, zero_division=0)), 4),
            "val_recall": round(float(recall_score(y_val, y_pred, zero_division=0)), 4),
            "val_samples": len(y_val),
        }

    # Train final model on full data
    model = xgb.XGBClassifier(
        device=device,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        scale_pos_weight=spw,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X, y, sample_weight=sample_weights)
    model.save_model(model_path)

    # Feature importance
    importances = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in model.feature_importances_]))
    sorted_imp = dict(sorted(importances.items(), key=lambda x: -x[1]))
    top3 = list(sorted_imp.items())[:3]

    meta = {
        "trained_at": datetime.now().isoformat(),
        "num_samples": len(rows),
        "device": device,
        "hyperparameters": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "scale_pos_weight": round(spw, 4),
        },
        "sample_weighting": "sqrt_current_price",
        "spike_rate": float(y.mean()),
        "feature_cols": FEATURE_COLS,
        "feature_importance": sorted_imp,
        **val_metrics,
    }
    meta_path = model_path.replace(".json", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info(
        "Model saved to %s (%d samples, device=%s, spike_rate=%.3f)",
        model_path, len(rows), device, meta["spike_rate"],
    )
    if top3:
        log.info("Top features: %s", ", ".join(f"{k}={v:.3f}" for k, v in top3))
    if val_metrics:
        log.info(
            "Validation: accuracy=%.3f auc=%s precision=%.3f recall=%.3f",
            val_metrics["val_accuracy"],
            f"{val_metrics['val_auc']:.3f}" if val_metrics["val_auc"] is not None else "N/A",
            val_metrics["val_precision"],
            val_metrics["val_recall"],
        )


def load_model_meta(model_path: str) -> dict | None:
    """Load model metadata if available."""
    meta_path = model_path.replace(".json", "_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def check_model_compatibility(model_path: str) -> bool:
    """Check if the trained model's features match current FEATURE_COLS.

    Returns True if compatible or if meta has no feature_cols (old model).
    """
    meta = load_model_meta(model_path)
    if meta is None or "feature_cols" not in meta:
        return True
    return meta["feature_cols"] == FEATURE_COLS


# TODO: use CUDA for scoring when available (match training device detection)
def score(features: list[dict], model_path: str) -> list[float]:
    """Return spike probability (0-1) for each feature dict.

    Raises ValueError if model was trained with incompatible features.
    """
    if not check_model_compatibility(model_path):
        meta = load_model_meta(model_path)
        raise ValueError(
            f"Model feature mismatch: model has {len(meta['feature_cols'])} features, "
            f"current code has {len(FEATURE_COLS)}. Retrain required."
        )
    df = pd.DataFrame(features)[FEATURE_COLS].fillna(0)
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model.predict_proba(df)[:, 1].tolist()
