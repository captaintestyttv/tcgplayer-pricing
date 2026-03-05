# tests/test_training_cache.py
"""Tests for full-universe training cache builder."""
import json
import pytest
from lib.mtgjson import build_training_cache, load_training_cache


@pytest.fixture
def sample_identifiers():
    return {
        "uuid-001": {
            "name": "Lightning Bolt", "rarity": "common", "setCode": "M10",
            "printings": ["M10", "2ED"],
            "legalities": {"standard": "Legal", "modern": "Legal"},
            "edhrecRank": 5, "edhrecSaltiness": 0.3, "isReserved": False,
            "supertypes": [], "types": ["Instant"], "subtypes": [],
            "colorIdentity": ["R"], "keywords": [], "manaValue": 1.0,
            "text": "Deal 3 damage.",
        },
        "uuid-002": {
            "name": "Black Lotus", "rarity": "rare", "setCode": "LEA",
            "printings": ["LEA"],
            "legalities": {"vintage": "Restricted"},
            "edhrecRank": None, "edhrecSaltiness": None, "isReserved": True,
            "supertypes": [], "types": ["Artifact"], "subtypes": [],
            "colorIdentity": [], "keywords": [], "manaValue": 0.0,
            "text": "Add three mana.",
        },
        "uuid-003": {
            "name": "No Prices Card", "rarity": "common", "setCode": "TST",
            "printings": ["TST"], "legalities": {}, "isReserved": False,
            "supertypes": [], "types": ["Creature"], "subtypes": [],
            "colorIdentity": [], "keywords": [], "manaValue": 2.0, "text": "",
        },
    }


@pytest.fixture
def sample_prices():
    return {
        "uuid-001": {
            "paper": {"tcgplayer": {
                "retail": {
                    "normal": {f"2026-01-{i:02d}": 1.0 + i * 0.01 for i in range(1, 32)},
                    "foil": {f"2026-01-{i:02d}": 3.0 for i in range(1, 11)},
                },
                "buylist": {"normal": {f"2026-01-{i:02d}": 0.5 for i in range(1, 11)}},
            }}
        },
        "uuid-002": {
            "paper": {"tcgplayer": {
                "retail": {"normal": {f"2026-01-{i:02d}": 5000.0 for i in range(1, 32)}},
            }}
        },
        # uuid-003 has no tcgplayer prices at all
    }


@pytest.fixture
def sample_set_data():
    return {"M10": {"releaseDate": "2009-07-17", "isPartialPreview": False}}


def test_build_training_cache_includes_all_priced_cards(sample_identifiers, sample_prices, sample_set_data):
    cache = build_training_cache(sample_identifiers, sample_prices, sample_set_data)
    assert len(cache) == 2
    assert "uuid-001" in cache
    assert "uuid-002" in cache
    assert "uuid-003" not in cache


def test_build_training_cache_card_structure(sample_identifiers, sample_prices, sample_set_data):
    cache = build_training_cache(sample_identifiers, sample_prices, sample_set_data)
    card = cache["uuid-001"]
    assert card["name"] == "Lightning Bolt"
    assert card["rarity"] == "common"
    assert "price_history" in card
    assert len(card["price_history"]) == 31
    assert "foil_price_history" in card
    assert "buylist_price_history" in card
    assert "setReleaseDate" in card


def test_build_training_cache_skips_no_prices(sample_identifiers, sample_prices, sample_set_data):
    cache = build_training_cache(sample_identifiers, sample_prices, sample_set_data)
    assert "uuid-003" not in cache


def test_build_training_cache_respects_max_cards(sample_identifiers, sample_prices, sample_set_data):
    cache = build_training_cache(sample_identifiers, sample_prices, sample_set_data, max_cards=1)
    assert len(cache) == 1


def test_load_training_cache_returns_empty_when_missing(tmp_path):
    assert load_training_cache(str(tmp_path)) == {}


def test_load_training_cache_reads_file(tmp_path):
    data = {"uuid-001": {"name": "Test"}}
    (tmp_path / "training_cards.json").write_text(json.dumps(data))
    assert load_training_cache(str(tmp_path)) == data
