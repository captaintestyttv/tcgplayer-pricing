# tests/test_features.py
import json
import pytest
from pathlib import Path
from lib.features import extract_features, generate_training_data

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
