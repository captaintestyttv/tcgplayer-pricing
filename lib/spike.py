import json
import os
from datetime import datetime

import pandas as pd
import xgboost as xgb

from lib.config import N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, get_logger

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
]


def train(rows: list[dict], model_path: str, device: str = "cpu") -> None:
    """Train XGBoost spike classifier and save to model_path."""
    if not rows:
        raise ValueError("No training data provided")

    df = pd.DataFrame(rows)
    X = df[FEATURE_COLS].fillna(0)
    y = df["spike"]

    model = xgb.XGBClassifier(
        device=device,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X, y)
    model.save_model(model_path)

    meta = {
        "trained_at": datetime.now().isoformat(),
        "num_samples": len(rows),
        "device": device,
        "hyperparameters": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
        },
        "spike_rate": float(y.mean()),
    }
    meta_path = model_path.replace(".json", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info(
        "Model saved to %s (%d samples, device=%s, spike_rate=%.3f)",
        model_path, len(rows), device, meta["spike_rate"],
    )


def load_model_meta(model_path: str) -> dict | None:
    """Load model metadata if available."""
    meta_path = model_path.replace(".json", "_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def score(features: list[dict], model_path: str) -> list[float]:
    """Return spike probability (0-1) for each feature dict."""
    df = pd.DataFrame(features)[FEATURE_COLS].fillna(0)
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model.predict_proba(df)[:, 1].tolist()
