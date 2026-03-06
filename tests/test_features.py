# tests/test_features.py
import json
import pytest
from pathlib import Path
from lib.features import extract_features, generate_training_data, compute_cluster_features, compute_spoiler_synergy_features

FIXTURES = Path(__file__).parent / "fixtures" / "inventory_cards.json"

@pytest.fixture
def cards():
    return json.loads(FIXTURES.read_text())

def test_extract_features_keys(cards):
    feat = extract_features("111111", cards["111111"])
    expected_keys = {
        "tcgplayer_id", "rarity_rank", "num_printings", "set_age_days",
        "formats_legal_count", "price_momentum_7d", "price_volatility_30d",
        "current_price",
        # Phase 1
        "edhrec_rank", "edhrec_saltiness", "is_reserved_list",
        "is_legendary", "is_creature", "color_count", "keyword_count",
        "mana_value", "subtype_count",
        # Phase 2
        "foil_to_normal_ratio", "buylist_ratio", "buylist_momentum_7d",
        # Phase 3
        "cluster_momentum_7d",
        # Phase 4
        "recently_reprinted", "legality_changed",
        # Phase 5
        "set_release_proximity", "spoiler_season",
        # Phase 6: derived signals
        "price_range_30d",
        "set_card_count", "price_percentile",
        # Phase 7: price dynamics
        "momentum_14d", "price_acceleration_7d", "drawdown_from_peak",
        "price_relative_to_range", "foil_momentum_7d", "trend_strength",
        # Phase 8: spoiler synergy
        "spoiler_tribal_overlap", "spoiler_keyword_overlap", "spoiler_color_overlap",
    }
    assert expected_keys == set(feat.keys())

def test_rarity_rank(cards):
    feat = extract_features("111111", cards["111111"])
    assert feat["rarity_rank"] == 2  # rare = 2

def test_num_printings(cards):
    feat = extract_features("222222", cards["222222"])
    assert feat["num_printings"] == 3

def test_formats_legal_count(cards):
    feat = extract_features("111111", cards["111111"])
    assert feat["formats_legal_count"] == 2  # standard + pioneer

def test_current_price_is_last_history_entry(cards):
    feat = extract_features("111111", cards["111111"])
    assert feat["current_price"] == pytest.approx(1.91)

def test_price_momentum_positive_for_rising_card(cards):
    feat = extract_features("111111", cards["111111"])
    assert feat["price_momentum_7d"] > 0

def test_generate_training_data_structure(cards):
    rows = generate_training_data(cards)
    assert len(rows) > 0
    assert "spike" in rows[0]
    assert "rarity_rank" in rows[0]
    assert "edhrec_rank" in rows[0]
    assert "cluster_momentum_7d" in rows[0]

def test_generate_training_data_labels_spike(cards):
    rows = generate_training_data(cards)
    card_rows = [r for r in rows if r["tcgplayer_id"] == "111111"]
    assert any(r["spike"] == 1 for r in card_rows)


def test_extract_features_empty_price_history():
    card = {"rarity": "common", "printings": [], "legalities": {}, "price_history": {}}
    feat = extract_features("0", card)
    assert feat["current_price"] == 0.0
    assert feat["price_momentum_7d"] == 0.0
    assert feat["price_volatility_30d"] == 0.0
    assert feat["set_age_days"] == 0
    # Phase 1 defaults
    assert feat["edhrec_rank"] == 99999
    assert feat["edhrec_saltiness"] == 0.0
    assert feat["is_reserved_list"] == 0
    assert feat["is_legendary"] == 0
    assert feat["is_creature"] == 0
    assert feat["color_count"] == 0
    assert feat["keyword_count"] == 0
    assert feat["mana_value"] == 0.0
    assert feat["subtype_count"] == 0
    # Phase 2 defaults
    assert feat["foil_to_normal_ratio"] == 0.0
    assert feat["buylist_ratio"] == 0.0
    assert feat["buylist_momentum_7d"] == 0.0
    # Phase 4 defaults
    assert feat["recently_reprinted"] == 0
    assert feat["legality_changed"] == 0
    # Phase 5 defaults
    assert feat["set_release_proximity"] == 90  # RELEASE_PROXIMITY_MAX
    assert feat["spoiler_season"] == 0
    # Phase 6 defaults
    assert feat["price_range_30d"] == 0.0
    assert feat["set_card_count"] == 0
    assert feat["price_percentile"] == pytest.approx(0.5)
    # Phase 8 defaults
    assert feat["spoiler_tribal_overlap"] == 0.0
    assert feat["spoiler_keyword_overlap"] == 0.0
    assert feat["spoiler_color_overlap"] == 0.0


def test_extract_features_unknown_rarity():
    card = {"rarity": "special", "printings": [], "legalities": {}, "price_history": {"2026-01-01": 1.0}}
    feat = extract_features("0", card)
    assert feat["rarity_rank"] == 0


def test_momentum_safe_with_zero_price():
    """Regression: division by zero when 7-day-ago price is 0."""
    history = {f"2026-01-{i:02d}": (0.0 if i <= 10 else 1.0) for i in range(1, 18)}
    card = {"rarity": "rare", "printings": [], "legalities": {}, "price_history": history}
    feat = extract_features("0", card)
    assert isinstance(feat["price_momentum_7d"], float)


def test_generate_training_data_skips_short_history():
    cards = {"0": {"rarity": "common", "printings": [], "legalities": {}, "price_history": {"2026-01-01": 1.0}}}
    rows = generate_training_data(cards)
    assert rows == []


# Phase 1 tests
def test_metadata_features(cards):
    feat = extract_features("111111", cards["111111"])
    assert feat["edhrec_rank"] == 1500
    assert feat["edhrec_saltiness"] == pytest.approx(1.2)
    assert feat["is_reserved_list"] == 0
    assert feat["is_legendary"] == 1
    assert feat["is_creature"] == 1
    assert feat["color_count"] == 1  # ["G"]
    assert feat["keyword_count"] == 1  # ["Trample"]
    assert feat["mana_value"] == pytest.approx(3.0)
    assert feat["subtype_count"] == 2  # ["Elf", "Warrior"]


def test_metadata_defaults_when_missing(cards):
    feat = extract_features("222222", cards["222222"])
    assert feat["edhrec_rank"] == 99999  # null -> default
    assert feat["edhrec_saltiness"] == 0.0  # null -> default
    assert feat["is_legendary"] == 0
    assert feat["is_creature"] == 0  # Instant, not Creature
    assert feat["subtype_count"] == 0


# Phase 2 tests
def test_foil_buylist_features(cards):
    feat = extract_features("111111", cards["111111"])
    # foil_to_normal_ratio: 3.85 / 1.91
    assert feat["foil_to_normal_ratio"] == pytest.approx(3.85 / 1.91, rel=1e-3)
    # buylist_ratio: 0.80 / 1.91
    assert feat["buylist_ratio"] == pytest.approx(0.80 / 1.91, rel=1e-3)
    # buylist_momentum_7d: (0.80 - 0.73) / 0.73
    assert feat["buylist_momentum_7d"] == pytest.approx((0.80 - 0.73) / 0.73, rel=1e-3)


def test_foil_buylist_defaults_no_data(cards):
    feat = extract_features("222222", cards["222222"])
    assert feat["foil_to_normal_ratio"] == 0.0
    assert feat["buylist_ratio"] == 0.0
    assert feat["buylist_momentum_7d"] == 0.0


# Phase 3 tests
def test_compute_cluster_features(cards):
    features = [extract_features(tid, card) for tid, card in cards.items()]
    compute_cluster_features(features, cards)
    # Card 111111 has subtypes ["Elf", "Warrior"], should have non-zero cluster momentum
    feat_111 = next(f for f in features if f["tcgplayer_id"] == "111111")
    assert feat_111["cluster_momentum_7d"] != 0.0  # has subtypes + momentum
    # Card 222222 has no subtypes, should be 0.0
    feat_222 = next(f for f in features if f["tcgplayer_id"] == "222222")
    assert feat_222["cluster_momentum_7d"] == 0.0


def test_cluster_features_single_card():
    """Cluster momentum for a lone subtype equals the card's own momentum."""
    cards = {
        "1": {
            "rarity": "rare", "printings": ["A"], "legalities": {},
            "subtypes": ["Dragon"],
            "price_history": {f"2026-01-{i:02d}": float(i) for i in range(1, 18)},
        },
    }
    features = [extract_features("1", cards["1"])]
    compute_cluster_features(features, cards)
    assert features[0]["cluster_momentum_7d"] == pytest.approx(features[0]["price_momentum_7d"])


# Phase 5 tests
def test_set_release_proximity_upcoming():
    from datetime import datetime
    card = {
        "rarity": "rare", "printings": ["A"], "legalities": {},
        "price_history": {"2026-01-01": 1.0},
        "setReleaseDate": "2026-02-15",
    }
    feat = extract_features("1", card, reference_date=datetime(2026, 2, 1))
    assert feat["set_release_proximity"] == 14


def test_set_release_proximity_past():
    from datetime import datetime
    card = {
        "rarity": "rare", "printings": ["A"], "legalities": {},
        "price_history": {"2026-01-01": 1.0},
        "setReleaseDate": "2025-10-01",
    }
    feat = extract_features("1", card, reference_date=datetime(2026, 2, 1))
    assert feat["set_release_proximity"] == 0  # clamped to 0 for past dates


def test_spoiler_season_active():
    from datetime import datetime
    card = {
        "rarity": "rare", "printings": ["A"], "legalities": {},
        "price_history": {"2026-01-01": 1.0},
        "setIsPartialPreview": True,
    }
    feat = extract_features("1", card, reference_date=datetime(2026, 2, 1))
    assert feat["spoiler_season"] == 1


def test_spoiler_season_from_proximity():
    from datetime import datetime
    card = {
        "rarity": "rare", "printings": ["A"], "legalities": {},
        "price_history": {"2026-01-01": 1.0},
        "setReleaseDate": "2026-02-20",
        "setIsPartialPreview": False,
    }
    feat = extract_features("1", card, reference_date=datetime(2026, 2, 1))
    assert feat["spoiler_season"] == 1  # 19 days out, within SPOILER_WINDOW_DAYS


def test_spike_label_ignores_cheap_cards():
    """Cards below SPIKE_MIN_PRICE should never be labeled as spikes."""
    card = {
        "rarity": "common", "printings": ["A"], "legalities": {},
        "price_history": {f"2026-01-{i:02d}": 0.05 * (1.5 if i > 15 else 1.0) for i in range(1, 32)},
        "foil_price_history": {}, "buylist_price_history": {},
        "subtypes": [],
    }
    rows = generate_training_data({"1": card})
    assert all(r["spike"] == 0 for r in rows), "Cheap card spikes should be filtered out"


def test_spike_label_works_above_floor():
    """Cards above SPIKE_MIN_PRICE should still get spike labels normally."""
    card = {
        "rarity": "rare", "printings": ["A"], "legalities": {},
        "price_history": {f"2026-01-{i:02d}": (1.0 if i <= 15 else 1.5) for i in range(1, 32)},
        "foil_price_history": {}, "buylist_price_history": {},
        "subtypes": [],
    }
    rows = generate_training_data({"1": card})
    assert any(r["spike"] == 1 for r in rows), "Above-floor spikes should be labeled"


def test_reference_date_affects_set_age(cards):
    from datetime import datetime
    ref = datetime(2026, 3, 1)
    feat = extract_features("111111", cards["111111"], reference_date=ref)
    # set_age_days = ref - first price date (2025-11-27)
    expected = (ref - datetime(2025, 11, 27)).days
    assert feat["set_age_days"] == expected


# Phase 6 tests
def test_price_range_30d_positive_for_volatile_card(cards):
    feat = extract_features("111111", cards["111111"])
    assert feat["price_range_30d"] > 0


def test_price_range_30d_zero_for_flat_card():
    card = {
        "rarity": "common", "printings": ["A"], "legalities": {},
        "price_history": {f"2026-01-{i:02d}": 1.0 for i in range(1, 32)},
    }
    feat = extract_features("1", card)
    assert feat["price_range_30d"] == pytest.approx(0.0)


def test_set_card_count(cards):
    feat = extract_features("111111", cards["111111"])
    assert isinstance(feat["set_card_count"], int)


def test_price_percentile_requires_context():
    """Without set context, price_percentile defaults to 0.5."""
    card = {
        "rarity": "rare", "printings": ["A"], "legalities": {},
        "price_history": {"2026-01-01": 5.0},
    }
    feat = extract_features("1", card)
    assert feat["price_percentile"] == pytest.approx(0.5)


# Phase 7 tests
def test_momentum_14d():
    history = {f"2026-01-{i:02d}": float(i) for i in range(1, 18)}
    card = {"rarity": "rare", "printings": [], "legalities": {}, "price_history": history}
    feat = extract_features("1", card)
    # 17 days of data, price goes 1..17, momentum_14d = (17 - 4) / 4 = 3.25
    assert feat["momentum_14d"] == pytest.approx((17 - 4) / 4)


def test_momentum_14d_insufficient_data():
    history = {f"2026-01-{i:02d}": float(i) for i in range(1, 10)}
    card = {"rarity": "rare", "printings": [], "legalities": {}, "price_history": history}
    feat = extract_features("1", card)
    assert feat["momentum_14d"] == 0.0


def test_price_acceleration_7d():
    # Rising then flat: acceleration should be negative
    history = {f"2026-01-{i:02d}": float(min(i, 10)) for i in range(1, 18)}
    card = {"rarity": "rare", "printings": [], "legalities": {}, "price_history": history}
    feat = extract_features("1", card)
    assert feat["price_acceleration_7d"] < 0  # decelerating


def test_drawdown_from_peak():
    history = {f"2026-01-{i:02d}": (10.0 if i <= 5 else 5.0) for i in range(1, 18)}
    card = {"rarity": "rare", "printings": [], "legalities": {}, "price_history": history}
    feat = extract_features("1", card)
    assert feat["drawdown_from_peak"] == pytest.approx(0.5)


def test_drawdown_from_peak_at_peak():
    history = {f"2026-01-{i:02d}": float(i) for i in range(1, 18)}
    card = {"rarity": "rare", "printings": [], "legalities": {}, "price_history": history}
    feat = extract_features("1", card)
    assert feat["drawdown_from_peak"] == pytest.approx(0.0)


def test_price_relative_to_range():
    # Price at bottom of range
    history = {f"2026-01-{i:02d}": (10.0 if i <= 5 else 1.0) for i in range(1, 18)}
    card = {"rarity": "rare", "printings": [], "legalities": {}, "price_history": history}
    feat = extract_features("1", card)
    assert feat["price_relative_to_range"] == pytest.approx(0.0)


def test_price_relative_to_range_flat():
    history = {f"2026-01-{i:02d}": 5.0 for i in range(1, 18)}
    card = {"rarity": "rare", "printings": [], "legalities": {}, "price_history": history}
    feat = extract_features("1", card)
    assert feat["price_relative_to_range"] == pytest.approx(0.5)  # flat -> 0.5


def test_foil_momentum_7d():
    foil_history = {f"2026-01-{i:02d}": float(i) for i in range(1, 18)}
    card = {
        "rarity": "rare", "printings": [], "legalities": {},
        "price_history": {"2026-01-01": 1.0},
        "foil_price_history": foil_history,
    }
    feat = extract_features("1", card)
    assert feat["foil_momentum_7d"] == pytest.approx((17 - 11) / 11)


def test_foil_momentum_7d_no_data():
    card = {"rarity": "rare", "printings": [], "legalities": {}, "price_history": {"2026-01-01": 1.0}}
    feat = extract_features("1", card)
    assert feat["foil_momentum_7d"] == 0.0


def test_trend_strength_linear():
    # Perfectly linear prices should have R^2 = 1.0
    history = {f"2026-01-{i:02d}": float(i) for i in range(1, 18)}
    card = {"rarity": "rare", "printings": [], "legalities": {}, "price_history": history}
    feat = extract_features("1", card)
    assert feat["trend_strength"] == pytest.approx(1.0, abs=1e-6)


def test_trend_strength_insufficient_data():
    history = {f"2026-01-{i:02d}": float(i) for i in range(1, 10)}
    card = {"rarity": "rare", "printings": [], "legalities": {}, "price_history": history}
    feat = extract_features("1", card)
    assert feat["trend_strength"] == 0.0


def test_empty_history_phase7_defaults():
    card = {"rarity": "common", "printings": [], "legalities": {}, "price_history": {}}
    feat = extract_features("0", card)
    assert feat["momentum_14d"] == 0.0
    assert feat["price_acceleration_7d"] == 0.0
    assert feat["drawdown_from_peak"] == 0.0
    assert feat["price_relative_to_range"] == 0.5
    assert feat["foil_momentum_7d"] == 0.0
    assert feat["trend_strength"] == 0.0


# Phase 8: spoiler synergy tests
def test_spoiler_tribal_overlap(cards):
    from datetime import datetime
    ref = datetime(2026, 3, 6)  # 333333 releases 2026-04-01, within 90 days
    features = [extract_features(tid, card, reference_date=ref) for tid, card in cards.items()]
    compute_spoiler_synergy_features(features, cards, reference_date=ref)
    feat_111 = next(f for f in features if f["tcgplayer_id"] == "111111")
    assert feat_111["spoiler_tribal_overlap"] > 0  # shares "Elf" with upcoming 333333


def test_spoiler_keyword_overlap(cards):
    from datetime import datetime
    ref = datetime(2026, 3, 6)
    features = [extract_features(tid, card, reference_date=ref) for tid, card in cards.items()]
    compute_spoiler_synergy_features(features, cards, reference_date=ref)
    feat_111 = next(f for f in features if f["tcgplayer_id"] == "111111")
    assert feat_111["spoiler_keyword_overlap"] >= 1  # shares "Trample" with upcoming 333333


def test_spoiler_color_overlap(cards):
    from datetime import datetime
    ref = datetime(2026, 3, 6)
    features = [extract_features(tid, card, reference_date=ref) for tid, card in cards.items()]
    compute_spoiler_synergy_features(features, cards, reference_date=ref)
    feat_111 = next(f for f in features if f["tcgplayer_id"] == "111111")
    assert feat_111["spoiler_color_overlap"] > 0  # shares "G" with upcoming 333333


def test_spoiler_synergy_no_upcoming(cards):
    from datetime import datetime
    ref = datetime(2027, 1, 1)  # far future, all sets already released
    features = [extract_features(tid, card, reference_date=ref) for tid, card in cards.items()]
    compute_spoiler_synergy_features(features, cards, reference_date=ref)
    for feat in features:
        assert feat["spoiler_tribal_overlap"] == 0.0
        assert feat["spoiler_keyword_overlap"] == 0.0
        assert feat["spoiler_color_overlap"] == 0.0


def test_spoiler_synergy_no_subtypes(cards):
    from datetime import datetime
    ref = datetime(2026, 3, 6)
    features = [extract_features(tid, card, reference_date=ref) for tid, card in cards.items()]
    compute_spoiler_synergy_features(features, cards, reference_date=ref)
    feat_222 = next(f for f in features if f["tcgplayer_id"] == "222222")
    assert feat_222["spoiler_tribal_overlap"] == 0.0  # no subtypes


def test_spoiler_synergy_reference_date_matters(cards):
    from datetime import datetime
    # Before the upcoming set window
    ref_before = datetime(2025, 12, 1)  # 333333 releases 2026-04-01, >90 days away
    features_before = [extract_features(tid, card, reference_date=ref_before) for tid, card in cards.items()]
    compute_spoiler_synergy_features(features_before, cards, reference_date=ref_before)
    feat_before = next(f for f in features_before if f["tcgplayer_id"] == "111111")

    # During the upcoming set window
    ref_during = datetime(2026, 3, 6)
    features_during = [extract_features(tid, card, reference_date=ref_during) for tid, card in cards.items()]
    compute_spoiler_synergy_features(features_during, cards, reference_date=ref_during)
    feat_during = next(f for f in features_during if f["tcgplayer_id"] == "111111")

    # Before window: 333333 is >90 days out, so no overlap
    assert feat_before["spoiler_tribal_overlap"] == 0.0
    # During window: 333333 is within 90 days, so overlap detected
    assert feat_during["spoiler_tribal_overlap"] > 0


def test_trend_strength_matches_after_optimization():
    """Verify trend_strength gives same results with numpy lstsq as sklearn."""
    import numpy as np
    # Perfectly linear
    history1 = {f"2026-01-{i:02d}": float(i) for i in range(1, 18)}
    card1 = {"rarity": "rare", "printings": [], "legalities": {}, "price_history": history1}
    feat1 = extract_features("1", card1)
    assert feat1["trend_strength"] == pytest.approx(1.0, abs=1e-6)

    # Noisy data
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, 17)
    history2 = {f"2026-01-{i:02d}": float(i) + noise[i-1] for i in range(1, 18)}
    card2 = {"rarity": "rare", "printings": [], "legalities": {}, "price_history": history2}
    feat2 = extract_features("1", card2)
    assert 0 < feat2["trend_strength"] < 1

    # Flat data
    history3 = {f"2026-01-{i:02d}": 5.0 for i in range(1, 18)}
    card3 = {"rarity": "rare", "printings": [], "legalities": {}, "price_history": history3}
    feat3 = extract_features("1", card3)
    assert feat3["trend_strength"] == pytest.approx(0.0, abs=1e-6)
