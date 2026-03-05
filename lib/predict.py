import csv
import json
import os
from datetime import datetime

from lib.config import (
    SPIKE_HOLD_THRESHOLD, COMMISSION_FEE, TRANSACTION_FEE, TRANSACTION_FLAT,
    SHIPPING_REVENUE, POSTAGE_STANDARD, POSTAGE_MEDIA_MAIL,
    HIGH_VALUE_THRESHOLD, MIN_MARGIN, MARKET_UP_PCT, MARKET_DOWN_PCT,
    COMPETITIVE_PCT, SUGGESTED_DISCOUNT, MIN_PRICE, get_logger,
)
from lib.features import extract_features, generate_training_data, compute_cluster_features, enrich_with_accumulated_history
from lib.forecast import forecast_card, forecast_with_confidence, trend_direction
from lib.mtgjson import load_inventory_cache
from lib.spike import FEATURE_COLS, score, train, load_model_meta, check_model_compatibility

log = get_logger(__name__)


def _calc_margin(market: float) -> float:
    revenue = market + SHIPPING_REVENUE if market < HIGH_VALUE_THRESHOLD else market
    fees = revenue * (COMMISSION_FEE + TRANSACTION_FEE) + TRANSACTION_FLAT
    postage = POSTAGE_STANDARD if market < HIGH_VALUE_THRESHOLD else POSTAGE_MEDIA_MAIL
    return round(revenue - fees - postage, 2)


def _pricing_action(market: float, current: float, net: float):
    if net < MIN_MARGIN:
        return "RAISE", f"Low margin (${net:.2f})"
    if market > current * (1 + MARKET_UP_PCT):
        return "RAISE", "Market up 10%+, current underpriced"
    if market < current * (1 - MARKET_DOWN_PCT):
        return "LOWER", "Market down 10%+, overpriced"
    if market >= HIGH_VALUE_THRESHOLD and current < market * COMPETITIVE_PCT:
        return "RAISE", "Competitive adjustment for high-value"
    return "", ""


def run_predict(
    history_dir: str,
    data_dir: str,
    models_dir: str,
    output_dir: str,
    dry_run: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    latest_path = os.path.join(history_dir, "latest.csv")
    with open(latest_path, newline="") as f:
        inventory = {row["TCGplayer Id"]: row for row in csv.DictReader(f)}

    cache = load_inventory_cache(data_dir)
    if not cache:
        print("No MTGJson cache found. Run 'monitor.sh sync' first. Falling back to report mode.")
        if not dry_run:
            _write_empty_outputs(output_dir)
        return

    model_path = os.path.join(models_dir, "spike_classifier.json")
    need_train = not os.path.exists(model_path)
    if not need_train and not check_model_compatibility(model_path):
        print("Model feature mismatch detected — retraining...")
        need_train = True

    if need_train:
        print("Training locally (CPU)...")
        from lib.mtgjson import load_training_cache
        training_cache = load_training_cache(data_dir)
        train_source = training_cache if training_cache else cache
        if training_cache:
            print(f"Using full training cache ({len(training_cache)} cards)")
        else:
            print(f"No training cache found, using inventory cache ({len(cache)} cards)")
        rows = generate_training_data(train_source)
        if rows:
            train(rows, model_path, device="cpu")
        else:
            print("Insufficient history for training. Spike scores will be 0.")
            model_path = None

    # Enrich inventory cache with accumulated price history for scoring
    enrich_with_accumulated_history(cache)

    spike_scores = {}
    if model_path and os.path.exists(model_path):
        features_list = [extract_features(tid, card) for tid, card in cache.items()]
        compute_cluster_features(features_list, cache)
        scores = score(features_list, model_path)
        spike_scores = {f["tcgplayer_id"]: s for f, s in zip(features_list, scores)}

    PRED_FIELDS = [
        "TCGplayer Id", "Product Name", "Current Price", "Market Price",
        "Suggested Price", "Action", "Reason", "Margin",
        "Predicted 7d", "7d Lower", "7d Upper",
        "Predicted 30d", "30d Lower", "30d Upper",
        "R-Squared", "Trend", "Spike Probability", "Signal",
    ]
    WATCH_FIELDS = ["TCGplayer Id", "Product Name", "Current Price", "Spike Probability", "Trend"]

    predictions = []
    watchlist = []

    for tcg_id, item in inventory.items():
        qty = int(item.get("Total Quantity", 0) or 0)
        if qty == 0:
            continue

        market = float(item.get("TCG Market Price") or 0)
        current = float(item.get("TCG Marketplace Price") or 0)
        name = item.get("Product Name", "")
        margin = _calc_margin(market)
        action, reason = _pricing_action(market, current, margin)
        suggested = round(market * SUGGESTED_DISCOUNT, 2) if action == "RAISE" else round(market, 2)

        card = cache.get(tcg_id)
        pred_7d = pred_30d = trend = spike_prob = signal = ""
        low7 = high7 = low30 = high30 = r_sq = ""

        if card:
            ph = card["price_history"]
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

            if sp >= SPIKE_HOLD_THRESHOLD:
                signal = "HOLD"
            elif trend == "down" and current <= market * 1.05:
                signal = "SELL_NOW"

            if signal == "HOLD":
                watchlist.append({
                    "TCGplayer Id": tcg_id,
                    "Product Name": name,
                    "Current Price": current,
                    "Spike Probability": spike_prob,
                    "Trend": trend,
                })

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

    if dry_run:
        hold_count = sum(1 for p in predictions if p["Signal"] == "HOLD")
        sell_now_count = sum(1 for p in predictions if p["Signal"] == "SELL_NOW")
        raise_count = sum(1 for p in predictions if p["Action"] == "RAISE")
        lower_count = sum(1 for p in predictions if p["Action"] == "LOWER")
        print(f"\n--- Dry Run Summary ---")
        print(f"Cards scored: {len(predictions)}")
        print(f"Actions: RAISE={raise_count}, LOWER={lower_count}")
        print(f"Signals: HOLD={hold_count}, SELL_NOW={sell_now_count}")
        print(f"Watchlist: {len(watchlist)} cards")
        if watchlist:
            print(f"\nTop watchlist entries:")
            watchlist.sort(key=lambda x: float(x["Spike Probability"]), reverse=True)
            for w in watchlist[:10]:
                print(f"  {w['Product Name']:<40} spike={w['Spike Probability']}  trend={w['Trend']}")
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    predictions_path = os.path.join(output_dir, f"predictions-{timestamp}.csv")
    watchlist_path = os.path.join(output_dir, f"watchlist-{timestamp}.csv")

    with open(predictions_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PRED_FIELDS)
        writer.writeheader()
        writer.writerows(predictions)

    watchlist.sort(key=lambda x: float(x["Spike Probability"]), reverse=True)
    with open(watchlist_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=WATCH_FIELDS)
        writer.writeheader()
        writer.writerows(watchlist)

    print(f"Predictions: {predictions_path}")
    print(f"Watchlist ({len(watchlist)} cards): {watchlist_path}")


def _write_empty_outputs(output_dir: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pred_fields = ["TCGplayer Id", "Product Name", "Current Price", "Market Price",
                   "Suggested Price", "Action", "Reason", "Margin",
                   "Predicted 7d", "7d Lower", "7d Upper",
                   "Predicted 30d", "30d Lower", "30d Upper",
                   "R-Squared", "Trend", "Spike Probability", "Signal"]
    watch_fields = ["TCGplayer Id", "Product Name", "Current Price", "Spike Probability", "Trend"]
    for path, fields in [
        (os.path.join(output_dir, f"predictions-{timestamp}.csv"), pred_fields),
        (os.path.join(output_dir, f"watchlist-{timestamp}.csv"), watch_fields),
    ]:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
