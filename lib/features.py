import bisect
from collections import defaultdict

import numpy as np
from datetime import datetime

from lib.config import (
    RARITY_RANK, SPIKE_THRESHOLD, SPIKE_MIN_PRICE, MIN_PRICE,
    SPOILER_WINDOW_DAYS, RELEASE_PROXIMITY_MAX, get_logger,
)
from lib.price_store import load_prices, merge_price_dicts
from lib.progress import ProgressBar, status

log = get_logger(__name__)


def _extract_card_metadata(tcgplayer_id: str, card: dict) -> dict:
    """Extract features that are constant across sliding windows for a card."""
    return {
        "tcgplayer_id": tcgplayer_id,
        "rarity_rank": RARITY_RANK.get(card.get("rarity", ""), 0),
        "num_printings": len(card.get("printings", [])),
        "formats_legal_count": sum(
            1 for v in card.get("legalities", {}).values() if v == "legal"
        ),
        "edhrec_rank": card.get("edhrecRank") if card.get("edhrecRank") is not None else 99999,
        "edhrec_saltiness": card.get("edhrecSaltiness") or 0.0,
        "is_reserved_list": int(card.get("isReserved", False)),
        "is_legendary": int("Legendary" in card.get("supertypes", [])),
        "is_creature": int("Creature" in card.get("types", [])),
        "color_count": len(card.get("colorIdentity", [])),
        "keyword_count": len(card.get("keywords", [])),
        "mana_value": float(card.get("manaValue", 0) or 0),
        "subtype_count": len(card.get("subtypes", [])),
        "recently_reprinted": int(card.get("recently_reprinted", 0)),
        "legality_changed": int(card.get("legality_changed", 0)),
        # Post-hoc features (defaults, computed later)
        "cluster_momentum_7d": 0.0,
        "spoiler_tribal_overlap": 0.0,
        "spoiler_keyword_overlap": 0.0,
        "spoiler_color_overlap": 0.0,
    }


def _extract_window_features(
    price_vals: list[float],
    price_dates: list[str],
    foil_vals: list[float],
    buylist_vals: list[float],
    end_idx: int,
    reference_date: datetime,
    card: dict,
) -> dict:
    """Extract price-derived features for a specific window endpoint.

    Args:
        price_vals: All price values, pre-sorted by date.
        price_dates: All date strings, pre-sorted.
        foil_vals: Foil price values up to cutoff, pre-sorted.
        buylist_vals: Buylist price values up to cutoff, pre-sorted.
        end_idx: Index into price_vals for the window's current day (inclusive).
        reference_date: Reference date for time-based features.
        card: Full card dict for set release date lookups.
    """
    pv = price_vals[:end_idx + 1]
    current_price = pv[-1] if pv else 0.0

    if len(pv) >= 7 and pv[-7] > 0:
        momentum_7d = (pv[-1] - pv[-7]) / max(pv[-7], MIN_PRICE)
    else:
        momentum_7d = 0.0

    volatility_30d = float(np.std(pv[-30:])) if len(pv) >= 2 else 0.0

    if pv:
        first_date = datetime.fromisoformat(price_dates[0])
        set_age_days = (reference_date - first_date).days
    else:
        set_age_days = 0

    release_str = card.get("setReleaseDate", "")
    if release_str:
        try:
            release_date = datetime.fromisoformat(release_str)
            days_to_release = (release_date - reference_date).days
            set_release_proximity = max(0, min(days_to_release, RELEASE_PROXIMITY_MAX))
        except ValueError:
            set_release_proximity = RELEASE_PROXIMITY_MAX
    else:
        set_release_proximity = RELEASE_PROXIMITY_MAX

    spoiler_season = int(
        card.get("setIsPartialPreview", False)
        or (0 < set_release_proximity <= SPOILER_WINDOW_DAYS)
    )

    fv = foil_vals
    bv = buylist_vals

    if fv and current_price > 0:
        foil_to_normal_ratio = fv[-1] / current_price
    else:
        foil_to_normal_ratio = 0.0

    if bv and current_price > 0:
        buylist_ratio = bv[-1] / current_price
    else:
        buylist_ratio = 0.0

    if len(bv) >= 7 and bv[-7] > 0:
        buylist_momentum_7d = (bv[-1] - bv[-7]) / max(bv[-7], MIN_PRICE)
    else:
        buylist_momentum_7d = 0.0

    if len(pv) >= 2:
        last_30 = pv[-30:]
        price_mean = sum(last_30) / len(last_30)
        price_range_30d = (max(last_30) - min(last_30)) / max(price_mean, MIN_PRICE)
    else:
        price_range_30d = 0.0

    if len(pv) >= 14 and pv[-14] > 0:
        momentum_14d = (pv[-1] - pv[-14]) / max(pv[-14], MIN_PRICE)
    else:
        momentum_14d = 0.0

    if len(pv) >= 14:
        mom_now = (pv[-1] - pv[-7]) / max(pv[-7], MIN_PRICE)
        mom_prev = (pv[-7] - pv[-14]) / max(pv[-14], MIN_PRICE)
        price_acceleration_7d = mom_now - mom_prev
    else:
        price_acceleration_7d = 0.0

    if pv:
        peak = max(pv)
        drawdown_from_peak = (peak - pv[-1]) / max(peak, MIN_PRICE)
    else:
        drawdown_from_peak = 0.0

    if len(pv) >= 2:
        lo, hi = min(pv), max(pv)
        spread = hi - lo
        price_relative_to_range = (pv[-1] - lo) / spread if spread > 1e-6 else 0.5
    else:
        price_relative_to_range = 0.5

    if len(fv) >= 7 and fv[-7] > 0:
        foil_momentum_7d = (fv[-1] - fv[-7]) / max(fv[-7], MIN_PRICE)
    else:
        foil_momentum_7d = 0.0

    if len(pv) >= 14:
        window = pv[-30:] if len(pv) >= 30 else pv
        y = np.array(window)
        n = len(y)
        x = np.arange(n, dtype=np.float64)
        x_mean = (n - 1) / 2.0
        y_mean = y.mean()
        y_centered = y - y_mean
        x_centered = x - x_mean
        ss_tot = y_centered.dot(y_centered)
        if ss_tot > 1e-12:
            slope = x_centered.dot(y_centered) / x_centered.dot(x_centered)
            residuals = y_centered - slope * x_centered
            ss_res = residuals.dot(residuals)
            trend_strength = max(0.0, float(1.0 - ss_res / ss_tot))
        else:
            trend_strength = 0.0
    else:
        trend_strength = 0.0

    return {
        "current_price": current_price,
        "price_momentum_7d": momentum_7d,
        "price_volatility_30d": volatility_30d,
        "set_age_days": set_age_days,
        "set_release_proximity": set_release_proximity,
        "spoiler_season": spoiler_season,
        "foil_to_normal_ratio": foil_to_normal_ratio,
        "buylist_ratio": buylist_ratio,
        "buylist_momentum_7d": buylist_momentum_7d,
        "price_range_30d": price_range_30d,
        "momentum_14d": momentum_14d,
        "price_acceleration_7d": price_acceleration_7d,
        "drawdown_from_peak": drawdown_from_peak,
        "price_relative_to_range": price_relative_to_range,
        "foil_momentum_7d": foil_momentum_7d,
        "trend_strength": trend_strength,
    }


def extract_features(tcgplayer_id: str, card: dict, reference_date: datetime | None = None) -> dict:
    if reference_date is None:
        reference_date = datetime.now()

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
        set_age_days = (reference_date - first_date).days
    else:
        set_age_days = 0

    # Phase 5: set release signals
    release_str = card.get("setReleaseDate", "")
    if release_str:
        try:
            release_date = datetime.fromisoformat(release_str)
            days_to_release = (release_date - reference_date).days
            set_release_proximity = max(0, min(days_to_release, RELEASE_PROXIMITY_MAX))
        except ValueError:
            set_release_proximity = RELEASE_PROXIMITY_MAX
    else:
        set_release_proximity = RELEASE_PROXIMITY_MAX

    spoiler_season = int(
        card.get("setIsPartialPreview", False)
        or (0 < set_release_proximity <= SPOILER_WINDOW_DAYS)
    )

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

    # Phase 6: derived price signals
    if len(price_vals) >= 2:
        last_30 = price_vals[-30:]
        price_mean = sum(last_30) / len(last_30)
        price_range_30d = (max(last_30) - min(last_30)) / max(price_mean, MIN_PRICE)
    else:
        price_range_30d = 0.0

    # Phase 7: price dynamics
    if len(price_vals) >= 14 and price_vals[-14] > 0:
        momentum_14d = (price_vals[-1] - price_vals[-14]) / max(price_vals[-14], MIN_PRICE)
    else:
        momentum_14d = 0.0

    if len(price_vals) >= 14:
        mom_now = (price_vals[-1] - price_vals[-7]) / max(price_vals[-7], MIN_PRICE)
        mom_prev = (price_vals[-7] - price_vals[-14]) / max(price_vals[-14], MIN_PRICE)
        price_acceleration_7d = mom_now - mom_prev
    else:
        price_acceleration_7d = 0.0

    if price_vals:
        peak = max(price_vals)
        drawdown_from_peak = (peak - price_vals[-1]) / max(peak, MIN_PRICE)
    else:
        drawdown_from_peak = 0.0

    if len(price_vals) >= 2:
        lo, hi = min(price_vals), max(price_vals)
        spread = hi - lo
        price_relative_to_range = (price_vals[-1] - lo) / spread if spread > 1e-6 else 0.5
    else:
        price_relative_to_range = 0.5

    if len(foil_vals) >= 7 and foil_vals[-7] > 0:
        foil_momentum_7d = (foil_vals[-1] - foil_vals[-7]) / max(foil_vals[-7], MIN_PRICE)
    else:
        foil_momentum_7d = 0.0

    if len(price_vals) >= 14:
        window = price_vals[-30:] if len(price_vals) >= 30 else price_vals
        y = np.array(window)
        n = len(y)
        x = np.arange(n, dtype=np.float64)
        x_mean = (n - 1) / 2.0
        y_mean = y.mean()
        y_centered = y - y_mean
        x_centered = x - x_mean
        ss_tot = y_centered.dot(y_centered)
        if ss_tot > 1e-12:
            slope = x_centered.dot(y_centered) / x_centered.dot(x_centered)
            residuals = y_centered - slope * x_centered
            ss_res = residuals.dot(residuals)
            trend_strength = max(0.0, float(1.0 - ss_res / ss_tot))
        else:
            trend_strength = 0.0
    else:
        trend_strength = 0.0

    set_card_count = card.get("set_card_count", 0)
    price_percentile = card.get("price_percentile", 0.5)

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
        # Phase 5: set release signals (2 features)
        "set_release_proximity": set_release_proximity,
        "spoiler_season": spoiler_season,
        # Phase 6: derived signals (3 features)
        "price_range_30d": price_range_30d,
        "set_card_count": set_card_count,
        "price_percentile": price_percentile,
        # Phase 7: price dynamics (6 features)
        "momentum_14d": momentum_14d,
        "price_acceleration_7d": price_acceleration_7d,
        "drawdown_from_peak": drawdown_from_peak,
        "price_relative_to_range": price_relative_to_range,
        "foil_momentum_7d": foil_momentum_7d,
        "trend_strength": trend_strength,
        # Phase 8: spoiler synergy (3 features, computed post-hoc)
        "spoiler_tribal_overlap": 0.0,
        "spoiler_keyword_overlap": 0.0,
        "spoiler_color_overlap": 0.0,
    }


def compute_spoiler_synergy_features(
    features_list: list[dict],
    cards: dict,
    reference_date: datetime | None = None,
) -> None:
    """Mutate feature dicts in-place to add spoiler synergy features.

    Computes overlap between each card's subtypes/keywords/colors and cards
    from sets releasing within RELEASE_PROXIMITY_MAX days of reference_date.
    """
    from datetime import timedelta

    def _build_upcoming_index(ref_dt: datetime) -> tuple[dict, set, list, int]:
        """Return (subtype_counts, keyword_set, color_sets, total_upcoming) for a ref date."""
        subtype_counts: dict[str, int] = defaultdict(int)
        keyword_set: set[str] = set()
        color_sets: list[set] = []
        for card in cards.values():
            release_str = card.get("setReleaseDate", "")
            if not release_str:
                continue
            try:
                release_dt = datetime.fromisoformat(release_str)
            except ValueError:
                continue
            if release_dt > ref_dt and (release_dt - ref_dt).days <= RELEASE_PROXIMITY_MAX:
                for st in card.get("subtypes", []):
                    subtype_counts[st] += 1
                keyword_set.update(card.get("keywords", []))
                ci = set(card.get("colorIdentity", []))
                if ci:
                    color_sets.append(ci)
        return subtype_counts, keyword_set, color_sets, len(color_sets)

    def _apply(feat: dict, card: dict, subtype_counts: dict, keyword_set: set,
               color_sets: list, total_upcoming: int) -> None:
        card_subtypes = card.get("subtypes", [])
        if card_subtypes and subtype_counts:
            feat["spoiler_tribal_overlap"] = float(max(
                subtype_counts.get(st, 0) for st in card_subtypes
            ))
        else:
            feat["spoiler_tribal_overlap"] = 0.0

        card_keywords = set(card.get("keywords", []))
        feat["spoiler_keyword_overlap"] = float(len(card_keywords & keyword_set))

        card_colors = set(card.get("colorIdentity", []))
        if card_colors and total_upcoming > 0:
            shared = sum(1 for cs in color_sets if card_colors & cs)
            feat["spoiler_color_overlap"] = shared / total_upcoming
        else:
            feat["spoiler_color_overlap"] = 0.0

    if reference_date is not None:
        # Live scoring: single reference date for all rows
        idx = _build_upcoming_index(reference_date)
        for feat in features_list:
            card = cards.get(feat["tcgplayer_id"], {})
            _apply(feat, card, *idx)
    else:
        # Training path: group by per-row _reference_date
        by_date: dict[str, list[dict]] = defaultdict(list)
        for feat in features_list:
            rd = feat.get("_reference_date", "")
            by_date[rd].append(feat)

        for date_str, feats in by_date.items():
            if date_str:
                ref_dt = datetime.fromisoformat(date_str)
            else:
                ref_dt = datetime.now()
            idx = _build_upcoming_index(ref_dt)
            for feat in feats:
                card = cards.get(feat["tcgplayer_id"], {})
                _apply(feat, card, *idx)

    # Clean up temporary keys
    for feat in features_list:
        feat.pop("_reference_date", None)


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


def enrich_with_accumulated_history(cards: dict) -> None:
    """Replace each card's price histories with accumulated data from Parquet store (in-place).

    Merges the cache's price_history with any deeper history stored in Parquet.
    This gives training more sliding windows when history exceeds MTGJson's 90-day window.
    """
    price_fields = [
        ("price_history", "normal"),
        ("foil_price_history", "foil"),
        ("buylist_price_history", "buylist"),
    ]
    enriched = 0
    for card_id, card in ProgressBar.iter(cards.items(), "Enriching history", total=len(cards)):
        for field, ptype in price_fields:
            cache_prices = card.get(field, {})
            stored_prices = load_prices(card_id, price_type=ptype)
            if stored_prices:
                merged = merge_price_dicts(stored_prices, cache_prices)
                if len(merged) > len(cache_prices):
                    card[field] = merged
                    enriched += 1
    if enriched:
        status(f"Enriched {enriched} price histories from Parquet store")


def generate_training_data(cards: dict) -> list[dict]:
    """Generate (features, spike_label) rows from historical windows."""
    rows = []

    enrich_with_accumulated_history(cards)

    # Compute per-set context
    set_cards_map = {}
    for tid, c in cards.items():
        ps = sorted(c.get("price_history", {}).items())
        if ps:
            sc = c.get("setCode", "")
            set_cards_map.setdefault(sc, []).append(float(ps[-1][1]))
    set_card_counts = {s: len(ps) for s, ps in set_cards_map.items()}

    with ProgressBar("Generating windows", total=len(cards)) as bar:
        for tcgplayer_id, card in cards.items():
            prices = sorted(card.get("price_history", {}).items())
            price_dates = [d for d, _ in prices]
            price_vals = [float(v) for _, v in prices]

            if len(price_vals) < 31:
                log.debug("Skipping %s: only %d days of history", tcgplayer_id, len(price_vals))
                bar.advance()
                continue

            foil_sorted = sorted(card.get("foil_price_history", {}).items())
            foil_dates = [d for d, _ in foil_sorted]
            foil_vals = [float(v) for _, v in foil_sorted]
            buylist_sorted = sorted(card.get("buylist_price_history", {}).items())
            buylist_dates = [d for d, _ in buylist_sorted]
            buylist_vals = [float(v) for _, v in buylist_sorted]

            # Card-level metadata (constant across all windows)
            meta = _extract_card_metadata(tcgplayer_id, card)

            set_code = card.get("setCode", "")
            meta["set_card_count"] = set_card_counts.get(set_code, 0)
            set_prices = set_cards_map.get(set_code, [])

            for i in range(len(price_vals) - 30):
                window = price_vals[i : i + 31]
                spike = int(
                    window[0] >= SPIKE_MIN_PRICE
                    and (max(window[1:]) - window[0]) / window[0] > SPIKE_THRESHOLD
                )

                cutoff_date = price_dates[i]
                ref_date = datetime.fromisoformat(cutoff_date)

                # Slice foil/buylist to cutoff_date using bisect
                foil_end = bisect.bisect_right(foil_dates, cutoff_date)
                buy_end = bisect.bisect_right(buylist_dates, cutoff_date)

                wf = _extract_window_features(
                    price_vals, price_dates,
                    foil_vals[:foil_end], buylist_vals[:buy_end],
                    i, ref_date, card,
                )

                # Price percentile
                if set_prices and window[0] > 0:
                    wf["price_percentile"] = sum(
                        1 for p in set_prices if p <= window[0]
                    ) / len(set_prices)
                else:
                    wf["price_percentile"] = 0.5

                # Merge: metadata (constant) + window features (variable)
                feat = {**meta, **wf}
                feat["spike"] = spike
                feat["_reference_date"] = cutoff_date
                rows.append(feat)

            bar.advance()

    status(f"Computing cluster features for {len(rows)} rows...")
    compute_cluster_features(rows, cards)

    status(f"Computing spoiler synergy features for {len(rows)} rows...")
    compute_spoiler_synergy_features(rows, cards)

    status(f"Generated {len(rows):,} training rows from {len(cards):,} cards")
    return rows
