"""FastAPI ML microservice for TCGPlayer spike prediction.

Exposes endpoints for training, prediction, and backtesting using the
existing lib/ feature pipeline and XGBoost classifier. Card data is
loaded from PostgreSQL instead of local JSON files.
"""

import io
import os
import sys
import tempfile
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add the parent project root to sys.path so we can import from lib/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.config import (
    SPIKE_HOLD_THRESHOLD, COMMISSION_FEE, TRANSACTION_FEE, TRANSACTION_FLAT,
    SHIPPING_REVENUE, POSTAGE_STANDARD, POSTAGE_MEDIA_MAIL,
    HIGH_VALUE_THRESHOLD, MIN_MARGIN, MARKET_UP_PCT, MARKET_DOWN_PCT,
    COMPETITIVE_PCT, SUGGESTED_DISCOUNT, MIN_PRICE, get_logger,
)
from lib.features import extract_features, generate_training_data, compute_cluster_features
from lib.forecast import forecast_with_confidence, trend_direction
from lib.spike import FEATURE_COLS, train as train_model, score as score_model

from db import (
    get_cards_with_prices,
    get_user_inventory,
    save_model,
    load_latest_model,
    save_predictions,
)

log = get_logger(__name__)

app = FastAPI(
    title="TCGPlayer ML Service",
    description="Spike prediction and pricing recommendations for MTG card inventory",
    version="1.0.0",
)

# Track whether a model is currently loaded in memory
_model_loaded = False


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    user_id: str


class TrainResponse(BaseModel):
    model_id: int
    num_samples: int
    spike_rate: float
    val_accuracy: float | None = None
    val_auc: float | None = None
    val_precision: float | None = None
    val_recall: float | None = None
    feature_importance: dict[str, float] = {}


class PredictResponse(BaseModel):
    user_id: str
    cards_scored: int
    hold_count: int
    sell_now_count: int
    raise_count: int
    lower_count: int
    predictions_saved: int


class BacktestResponse(BaseModel):
    total_samples: int
    spike_count: int
    spike_rate: float
    threshold: float
    confusion_matrix: dict[str, int]
    accuracy: float
    precision: float
    recall: float
    f1: float
    calibration_bins: list[dict]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calc_margin(market: float) -> float:
    """Calculate net margin after fees and shipping."""
    revenue = market + SHIPPING_REVENUE if market < HIGH_VALUE_THRESHOLD else market
    fees = revenue * (COMMISSION_FEE + TRANSACTION_FEE) + TRANSACTION_FLAT
    postage = POSTAGE_STANDARD if market < HIGH_VALUE_THRESHOLD else POSTAGE_MEDIA_MAIL
    return round(revenue - fees - postage, 2)


def _pricing_action(market: float, current: float, net: float) -> tuple[str, str]:
    """Determine pricing action and reason."""
    if net < MIN_MARGIN:
        return "RAISE", f"Low margin (${net:.2f})"
    if market > current * (1 + MARKET_UP_PCT):
        return "RAISE", "Market up 10%+, current underpriced"
    if market < current * (1 - MARKET_DOWN_PCT):
        return "LOWER", "Market down 10%+, overpriced"
    if market >= HIGH_VALUE_THRESHOLD and current < market * COMPETITIVE_PCT:
        return "RAISE", "Competitive adjustment for high-value"
    return "", ""


def _save_model_to_db(model_path: str, meta: dict) -> int:
    """Read the trained model file and store it in the database."""
    with open(model_path, "rb") as f:
        model_bytes = f.read()
    return save_model(model_bytes, meta)


def _load_model_from_db_to_tempfile() -> tuple[str, dict] | None:
    """Load the latest model from DB and write to a temp file for XGBoost.

    Returns (temp_file_path, metadata) or None.
    """
    result = load_latest_model()
    if result is None:
        return None

    model_bytes, metadata = result
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.write(model_bytes)
    tmp.close()
    return tmp.name, metadata


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check endpoint."""
    global _model_loaded
    # Check if a model exists in the DB
    result = load_latest_model()
    _model_loaded = result is not None
    return {"status": "ok", "model_loaded": _model_loaded}


@app.post("/train", response_model=TrainResponse)
def train_endpoint():
    """Train the XGBoost spike classifier on all card data from the database.

    Reads card metadata and price history from PostgreSQL, generates training
    data using sliding windows (lib/features.py), trains the model
    (lib/spike.py), and stores the model blob + metadata in the models table.
    """
    global _model_loaded

    log.info("Loading card data from database for training...")
    cards = get_cards_with_prices()
    if not cards:
        raise HTTPException(status_code=400, detail="No card data found in database.")

    log.info("Generating training data from %d cards...", len(cards))
    rows = generate_training_data(cards)
    if not rows:
        raise HTTPException(
            status_code=400,
            detail="No training data generated. Cards may have insufficient price history (need 31+ days).",
        )

    # Train to a temp file (lib/spike.py writes to disk)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "spike_classifier.json")
        train_model(rows, model_path, device="cpu")

        # Read back the metadata that train() wrote
        import json
        meta_path = model_path.replace(".json", "_meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        # Store model blob + metadata in DB
        model_id = _save_model_to_db(model_path, meta)

    _model_loaded = True

    log.info("Model trained and saved to DB with id=%d", model_id)

    return TrainResponse(
        model_id=model_id,
        num_samples=meta["num_samples"],
        spike_rate=meta["spike_rate"],
        val_accuracy=meta.get("val_accuracy"),
        val_auc=meta.get("val_auc"),
        val_precision=meta.get("val_precision"),
        val_recall=meta.get("val_recall"),
        feature_importance=meta.get("feature_importance", {}),
    )


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    """Score all cards in a user's inventory and save predictions.

    Loads the user's inventory from the database, scores each card using
    the latest trained model, computes pricing recommendations, and writes
    results to the predictions table.
    """
    user_id = request.user_id

    # Load model from DB
    model_result = _load_model_from_db_to_tempfile()
    if model_result is None:
        raise HTTPException(
            status_code=400,
            detail="No trained model found. Call POST /train first.",
        )
    model_path, model_meta = model_result

    try:
        # Verify feature compatibility
        model_features = model_meta.get("feature_cols", [])
        if model_features and model_features != FEATURE_COLS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model feature mismatch: model has {len(model_features)} features, "
                    f"current code has {len(FEATURE_COLS)}. Retrain required (POST /train)."
                ),
            )

        # Load user inventory
        log.info("Loading inventory for user %s...", user_id)
        inventory = get_user_inventory(user_id)
        if not inventory:
            raise HTTPException(
                status_code=404,
                detail=f"No inventory found for user_id={user_id}.",
            )

        # Extract features and score
        features_list = [
            extract_features(tcg_id, card) for tcg_id, card in inventory.items()
        ]
        compute_cluster_features(features_list, inventory)
        scores = score_model(features_list, model_path)
        spike_scores = {f["tcgplayer_id"]: s for f, s in zip(features_list, scores)}

        # Build predictions
        predictions = []
        hold_count = 0
        sell_now_count = 0
        raise_count = 0
        lower_count = 0

        for tcg_id, card in inventory.items():
            market = card.get("market_price", 0.0)
            current = card.get("marketplace_price", 0.0)
            name = card.get("product_name", "")
            margin = _calc_margin(market)
            action, reason = _pricing_action(market, current, margin)
            suggested = round(market * SUGGESTED_DISCOUNT, 2) if action == "RAISE" else round(market, 2)

            # Forecast
            ph = card.get("price_history", {})
            pred_7d = pred_30d = trend = ""
            low7 = high7 = low30 = high30 = r_sq = ""
            spike_prob = ""

            fc7 = forecast_with_confidence(ph, 7)
            fc30 = forecast_with_confidence(ph, 30)
            if fc7:
                pred_7d = f"{fc7['predicted']:.4f}"
                low7 = f"{fc7['lower']:.4f}"
                high7 = f"{fc7['upper']:.4f}"
                r_sq = f"{fc7['r_squared']:.4f}"
            else:
                pred_7d = "insufficient_data"
            if fc30:
                pred_30d = f"{fc30['predicted']:.4f}"
                low30 = f"{fc30['lower']:.4f}"
                high30 = f"{fc30['upper']:.4f}"
            else:
                pred_30d = "insufficient_data"

            trend = trend_direction(ph)
            sp = spike_scores.get(tcg_id, 0.0)
            spike_prob = f"{sp:.4f}"

            signal = ""
            if sp >= SPIKE_HOLD_THRESHOLD:
                signal = "HOLD"
                hold_count += 1
            elif trend == "down" and current <= market * 1.05:
                signal = "SELL_NOW"
                sell_now_count += 1

            if action == "RAISE":
                raise_count += 1
            elif action == "LOWER":
                lower_count += 1

            predictions.append({
                "TCGplayer Id": tcg_id,
                "Product Name": name,
                "Current Price": current,
                "Market Price": market,
                "Suggested Price": max(MIN_PRICE, suggested),
                "Action": action,
                "Reason": reason,
                "Margin": margin,
                "Predicted 7d": pred_7d,
                "7d Lower": low7,
                "7d Upper": high7,
                "Predicted 30d": pred_30d,
                "30d Lower": low30,
                "30d Upper": high30,
                "R-Squared": r_sq,
                "Trend": trend,
                "Spike Probability": spike_prob,
                "Signal": signal,
            })

        # Save predictions to DB
        saved_count = save_predictions(user_id, predictions)

        return PredictResponse(
            user_id=user_id,
            cards_scored=len(predictions),
            hold_count=hold_count,
            sell_now_count=sell_now_count,
            raise_count=raise_count,
            lower_count=lower_count,
            predictions_saved=saved_count,
        )

    finally:
        # Clean up temp model file
        if os.path.exists(model_path):
            os.unlink(model_path)


@app.post("/backtest", response_model=BacktestResponse)
def backtest_endpoint():
    """Evaluate the latest model against training data.

    Loads all card data from the database, generates training rows,
    scores them with the latest model, and returns performance metrics.
    """
    # Load model from DB
    model_result = _load_model_from_db_to_tempfile()
    if model_result is None:
        raise HTTPException(
            status_code=400,
            detail="No trained model found. Call POST /train first.",
        )
    model_path, model_meta = model_result

    try:
        # Feature compatibility check
        model_features = model_meta.get("feature_cols", [])
        if model_features and model_features != FEATURE_COLS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model feature mismatch: model has {len(model_features)} features, "
                    f"current code has {len(FEATURE_COLS)}. Retrain required (POST /train)."
                ),
            )

        # Load cards and generate training data
        log.info("Loading card data for backtest...")
        cards = get_cards_with_prices()
        if not cards:
            raise HTTPException(status_code=400, detail="No card data found in database.")

        rows = generate_training_data(cards)
        if not rows:
            raise HTTPException(
                status_code=400,
                detail="No training data generated. Insufficient price history.",
            )

        # Score all rows
        probabilities = score_model(rows, model_path)

        # Compute metrics
        actuals = [r["spike"] for r in rows]
        predicted = [1 if p >= SPIKE_HOLD_THRESHOLD else 0 for p in probabilities]

        tp = sum(1 for a, p in zip(actuals, predicted) if a == 1 and p == 1)
        fp = sum(1 for a, p in zip(actuals, predicted) if a == 0 and p == 1)
        fn = sum(1 for a, p in zip(actuals, predicted) if a == 1 and p == 0)
        tn = sum(1 for a, p in zip(actuals, predicted) if a == 0 and p == 0)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        accuracy = (tp + tn) / max(len(actuals), 1)

        # Probability calibration: 10 bins
        bins = []
        for i in range(10):
            lo = i * 0.1
            hi = (i + 1) * 0.1
            in_bin = [(a, p) for a, p in zip(actuals, probabilities) if lo <= p < hi]
            if in_bin:
                actual_rate = sum(a for a, _ in in_bin) / len(in_bin)
                avg_prob = sum(p for _, p in in_bin) / len(in_bin)
                bins.append({
                    "range": f"{lo:.1f}-{hi:.1f}",
                    "count": len(in_bin),
                    "actual_spike_rate": round(actual_rate, 4),
                    "avg_predicted_prob": round(avg_prob, 4),
                })

        return BacktestResponse(
            total_samples=len(rows),
            spike_count=sum(actuals),
            spike_rate=round(sum(actuals) / max(len(actuals), 1), 4),
            threshold=SPIKE_HOLD_THRESHOLD,
            confusion_matrix={"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            calibration_bins=bins,
        )

    finally:
        # Clean up temp model file
        if os.path.exists(model_path):
            os.unlink(model_path)
