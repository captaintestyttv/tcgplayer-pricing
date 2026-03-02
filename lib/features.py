from collections import defaultdict

import numpy as np
from datetime import datetime

from lib.config import RARITY_RANK, SPIKE_THRESHOLD, MIN_PRICE, get_logger

log = get_logger(__name__)


def extract_features(tcgplayer_id: str, card: dict) -> dict:
    prices = card.get("price_history", {})
    sorted_entries = sorted(prices.items())
    price_vals = [float(v) for _, v in sorted_entries]

    current_price = price_vals[-1] if price_vals else 0.0

    if len(price_vals) >= 7 and price_vals[-7] > 0:
        momentum_7d = (price_vals[-1] - price_vals[-7]) / max(price_vals[-7], MIN_PRICE)
    else:
        momentum_7d = 0.0

    volatility_30d = float(np.std(price_vals[-30:])) if len(price_vals) >= 2 else 0.0

    if sorted_entries:
        first_date = datetime.fromisoformat(sorted_entries[0][0])
        set_age_days = (datetime.now() - first_date).days
    else:
        set_age_days = 0

    # Phase 2: foil & buylist signals
    foil_prices = card.get("foil_price_history", {})
    foil_vals = [float(v) for _, v in sorted(foil_prices.items())]
    if foil_vals and current_price > 0:
        foil_to_normal_ratio = foil_vals[-1] / current_price
    else:
        foil_to_normal_ratio = 0.0

    buylist_prices = card.get("buylist_price_history", {})
    buylist_vals = [float(v) for _, v in sorted(buylist_prices.items())]
    if buylist_vals and current_price > 0:
        buylist_ratio = buylist_vals[-1] / current_price
    else:
        buylist_ratio = 0.0

    if len(buylist_vals) >= 7 and buylist_vals[-7] > 0:
        buylist_momentum_7d = (buylist_vals[-1] - buylist_vals[-7]) / max(buylist_vals[-7], MIN_PRICE)
    else:
        buylist_momentum_7d = 0.0

    edhrec_rank = card.get("edhrecRank")

    return {
        "tcgplayer_id": tcgplayer_id,
        # Original 7
        "rarity_rank": RARITY_RANK.get(card.get("rarity", ""), 0),
        "num_printings": len(card.get("printings", [])),
        "set_age_days": set_age_days,
        "formats_legal_count": sum(
            1 for v in card.get("legalities", {}).values() if v == "legal"
        ),
        "price_momentum_7d": momentum_7d,
        "price_volatility_30d": volatility_30d,
        "current_price": current_price,
        # Phase 1: card metadata (9 features)
        "edhrec_rank": edhrec_rank if edhrec_rank is not None else 99999,
        "edhrec_saltiness": card.get("edhrecSaltiness") or 0.0,
        "is_reserved_list": int(card.get("isReserved", False)),
        "is_legendary": int("Legendary" in card.get("supertypes", [])),
        "is_creature": int("Creature" in card.get("types", [])),
        "color_count": len(card.get("colorIdentity", [])),
        "keyword_count": len(card.get("keywords", [])),
        "mana_value": float(card.get("manaValue", 0) or 0),
        "subtype_count": len(card.get("subtypes", [])),
        # Phase 2: foil & buylist (3 features)
        "foil_to_normal_ratio": foil_to_normal_ratio,
        "buylist_ratio": buylist_ratio,
        "buylist_momentum_7d": buylist_momentum_7d,
        # Phase 3: cluster (1 feature, computed post-hoc)
        "cluster_momentum_7d": 0.0,
        # Phase 4: change detection (2 features)
        "recently_reprinted": int(card.get("recently_reprinted", 0)),
        "legality_changed": int(card.get("legality_changed", 0)),
    }


def compute_cluster_features(features_list: list[dict], cards: dict) -> None:
    """Mutate feature dicts in-place to add cluster_momentum_7d."""
    subtype_momentum = defaultdict(list)
    for feat in features_list:
        card = cards.get(feat["tcgplayer_id"], {})
        for st in card.get("subtypes", []):
            subtype_momentum[st].append(feat["price_momentum_7d"])

    subtype_avg = {st: sum(v) / len(v) for st, v in subtype_momentum.items() if v}

    for feat in features_list:
        card = cards.get(feat["tcgplayer_id"], {})
        subtypes = card.get("subtypes", [])
        if subtypes and subtype_avg:
            feat["cluster_momentum_7d"] = max(
                subtype_avg.get(st, 0.0) for st in subtypes
            )
        else:
            feat["cluster_momentum_7d"] = 0.0


def generate_training_data(cards: dict) -> list[dict]:
    """Generate (features, spike_label) rows from historical windows."""
    rows = []
    for tcgplayer_id, card in cards.items():
        prices = sorted(card.get("price_history", {}).items())
        price_vals = [float(v) for _, v in prices]

        if len(price_vals) < 31:
            log.debug("Skipping %s: only %d days of history", tcgplayer_id, len(price_vals))
            continue

        foil = sorted(card.get("foil_price_history", {}).items())
        buylist = sorted(card.get("buylist_price_history", {}).items())

        for i in range(len(price_vals) - 30):
            window = price_vals[i : i + 31]
            spike = int(
                window[0] > 0 and (max(window[1:]) - window[0]) / window[0] > SPIKE_THRESHOLD
            )
            snapshot = dict(card)
            snapshot["price_history"] = dict(prices[:i+1])

            cutoff_date = prices[i][0]
            snapshot["foil_price_history"] = {d: v for d, v in foil if d <= cutoff_date}
            snapshot["buylist_price_history"] = {d: v for d, v in buylist if d <= cutoff_date}

            feat = extract_features(tcgplayer_id, snapshot)
            feat["spike"] = spike
            rows.append(feat)

    compute_cluster_features(rows, cards)

    log.info("Generated %d training rows from %d cards", len(rows), len(cards))
    return rows
