"""Data access layer for the web UI.

Wraps lib/ functions and reads output CSVs/JSON. Routes never touch files directly.
"""

import csv
import json
import os
from datetime import datetime

# Resolve project root (parent of web/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HISTORY_DIR = os.path.join(PROJECT_ROOT, "history")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "mtgjson")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EXPORTS_DIR = os.path.join(PROJECT_ROOT, "tcgplayer-exports")

# In-memory cache for inventory_cards.json
_inventory_cache = None
_inventory_mtime = 0


def _cache_path():
    return os.path.join(DATA_DIR, "inventory_cards.json")


def get_inventory_cache() -> dict:
    """Load inventory cache, refreshing if file changed on disk."""
    global _inventory_cache, _inventory_mtime
    path = _cache_path()
    if not os.path.exists(path):
        return {}
    mtime = os.path.getmtime(path)
    if _inventory_cache is None or mtime > _inventory_mtime:
        with open(path) as f:
            _inventory_cache = json.load(f)
        _inventory_mtime = mtime
    return _inventory_cache


def invalidate_inventory_cache():
    """Force reload on next access."""
    global _inventory_cache, _inventory_mtime
    _inventory_cache = None
    _inventory_mtime = 0


def get_latest_csv() -> list[dict]:
    """Read history/latest.csv into a list of dicts."""
    path = os.path.join(HISTORY_DIR, "latest.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _find_latest_file(directory: str, prefix: str, ext: str) -> str | None:
    """Find the most recent file matching prefix-*.ext in directory."""
    if not os.path.isdir(directory):
        return None
    matches = sorted(
        f for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith(ext)
    )
    return os.path.join(directory, matches[-1]) if matches else None


def get_predictions() -> list[dict]:
    """Read the latest predictions CSV."""
    path = _find_latest_file(OUTPUT_DIR, "predictions-", ".csv")
    if not path:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def get_watchlist() -> list[dict]:
    """Read the latest watchlist CSV."""
    path = _find_latest_file(OUTPUT_DIR, "watchlist-", ".csv")
    if not path:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def get_analysis() -> dict:
    """Read analysis-latest.json."""
    path = os.path.join(OUTPUT_DIR, "analysis-latest.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def get_backtest() -> dict:
    """Read the latest backtest JSON."""
    path = _find_latest_file(OUTPUT_DIR, "backtest-", ".json")
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


def get_model_meta() -> dict:
    """Read spike_classifier_meta.json."""
    path = os.path.join(MODELS_DIR, "spike_classifier_meta.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def get_card(tcg_id: str) -> dict | None:
    """Get combined card data: inventory row + cache metadata."""
    inventory = get_latest_csv()
    row = None
    for r in inventory:
        if r.get("TCGplayer Id") == tcg_id:
            row = r
            break

    cache = get_inventory_cache()
    card_cache = cache.get(tcg_id, {})

    if not row and not card_cache:
        return None

    result = {}
    if row:
        result["tcgplayer_id"] = tcg_id
        result["name"] = row.get("Product Name", "")
        result["set_name"] = row.get("Set Name", "")
        result["rarity"] = row.get("Rarity", "")
        result["condition"] = row.get("Condition", "")
        result["market_price"] = float(row.get("TCG Market Price") or 0)
        result["list_price"] = float(row.get("TCG Marketplace Price") or 0)
        result["quantity"] = int(row.get("Total Quantity", 0) or 0)

    if card_cache:
        result["name"] = result.get("name") or card_cache.get("name", "")
        result["rarity"] = result.get("rarity") or card_cache.get("rarity", "")
        result["set_code"] = card_cache.get("setCode", "")
        result["legalities"] = card_cache.get("legalities", {})
        result["printings"] = card_cache.get("printings", [])
        result["edhrec_rank"] = card_cache.get("edhrecRank")
        result["is_reserved"] = card_cache.get("isReserved", False)
        result["mana_value"] = card_cache.get("manaValue", 0)
        result["text"] = card_cache.get("text", "")
        result["types"] = card_cache.get("types", [])
        result["subtypes"] = card_cache.get("subtypes", [])
        result["color_identity"] = card_cache.get("colorIdentity", [])
        result["keywords"] = card_cache.get("keywords", [])

    # Attach prediction data if available
    predictions = get_predictions()
    for p in predictions:
        if p.get("TCGplayer Id") == tcg_id:
            result["prediction"] = p
            break

    return result


def get_card_prices(tcg_id: str) -> dict:
    """Get price history arrays for charting."""
    cache = get_inventory_cache()
    card = cache.get(tcg_id, {})
    return {
        "normal": card.get("price_history", {}),
        "foil": card.get("foil_price_history", {}),
        "buylist": card.get("buylist_price_history", {}),
    }


def get_dashboard_stats() -> dict:
    """Compute summary stats for the dashboard."""
    inventory = get_latest_csv()
    active = [r for r in inventory if int(r.get("Total Quantity", 0) or 0) > 0]

    total_market = sum(float(r.get("TCG Market Price") or 0) for r in active)
    total_list = sum(float(r.get("TCG Marketplace Price") or 0) for r in active)

    predictions = get_predictions()
    actions = {}
    signals = {}
    for p in predictions:
        a = p.get("Action", "")
        s = p.get("Signal", "")
        if a:
            actions[a] = actions.get(a, 0) + 1
        if s:
            signals[s] = signals.get(s, 0) + 1

    model_meta = get_model_meta()
    cache = get_inventory_cache()

    # History file count
    history_files = []
    if os.path.isdir(HISTORY_DIR):
        history_files = [
            f for f in os.listdir(HISTORY_DIR)
            if f.startswith("export-") and f.endswith(".csv")
        ]

    return {
        "total_cards": len(active),
        "total_market_value": round(total_market, 2),
        "total_list_value": round(total_list, 2),
        "predictions_count": len(predictions),
        "actions": actions,
        "signals": signals,
        "model_trained": model_meta.get("trained_at", ""),
        "model_device": model_meta.get("device", ""),
        "model_auc": model_meta.get("validation", {}).get("auc", ""),
        "cache_cards": len(cache),
        "history_exports": len(history_files),
    }


def get_export_files() -> list[dict]:
    """List history export files with timestamps."""
    if not os.path.isdir(HISTORY_DIR):
        return []
    files = sorted(
        (f for f in os.listdir(HISTORY_DIR) if f.startswith("export-") and f.endswith(".csv")),
        reverse=True,
    )
    result = []
    for f in files:
        path = os.path.join(HISTORY_DIR, f)
        stat = os.stat(path)
        result.append({
            "filename": f,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    return result
