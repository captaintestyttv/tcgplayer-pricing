import pandas as pd
import xgboost as xgb

FEATURE_COLS = [
    "rarity_rank",
    "num_printings",
    "set_age_days",
    "formats_legal_count",
    "price_momentum_7d",
    "price_volatility_30d",
    "current_price",
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
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X, y)
    model.save_model(model_path)


def score(features: list[dict], model_path: str) -> list[float]:
    """Return spike probability (0-1) for each feature dict."""
    df = pd.DataFrame(features)[FEATURE_COLS].fillna(0)
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model.predict_proba(df)[:, 1].tolist()
