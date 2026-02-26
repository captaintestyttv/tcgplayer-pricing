import numpy as np
from datetime import datetime

RARITY_RANK = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3}
SPIKE_THRESHOLD = 0.20  # >20% in 30 days = spike


def extract_features(tcgplayer_id: str, card: dict) -> dict:
    prices = card.get("price_history", {})
    sorted_entries = sorted(prices.items())
    price_vals = [float(v) for _, v in sorted_entries]

    current_price = price_vals[-1] if price_vals else 0.0

    if len(price_vals) >= 7 and price_vals[-7] > 0:
        momentum_7d = (price_vals[-1] - price_vals[-7]) / price_vals[-7]
    else:
        momentum_7d = 0.0

    volatility_30d = float(np.std(price_vals[-30:])) if len(price_vals) >= 2 else 0.0

    if sorted_entries:
        first_date = datetime.fromisoformat(sorted_entries[0][0])
        set_age_days = (datetime.now() - first_date).days
    else:
        set_age_days = 0

    return {
        "tcgplayer_id": tcgplayer_id,
        "rarity_rank": RARITY_RANK.get(card.get("rarity", ""), 0),
        "num_printings": len(card.get("printings", [])),
        "set_age_days": set_age_days,
        "formats_legal_count": sum(
            1 for v in card.get("legalities", {}).values() if v == "legal"
        ),
        "price_momentum_7d": momentum_7d,
        "price_volatility_30d": volatility_30d,
        "current_price": current_price,
    }


def generate_training_data(cards: dict) -> list[dict]:
    """Generate (features, spike_label) rows from historical windows."""
    rows = []
    for tcgplayer_id, card in cards.items():
        prices = sorted(card.get("price_history", {}).items())
        price_vals = [float(v) for _, v in prices]

        if len(price_vals) < 31:
            continue

        for i in range(len(price_vals) - 30):
            window = price_vals[i : i + 31]
            spike = int(
                window[0] > 0 and (max(window[1:]) - window[0]) / window[0] > SPIKE_THRESHOLD
            )
            snapshot = dict(card)
            snapshot["price_history"] = dict(prices[:i+1])
            snapshot["rarity"] = card.get("rarity", "")
            feat = extract_features(tcgplayer_id, snapshot)
            feat["spike"] = spike
            rows.append(feat)

    return rows
