# tests/test_mtgjson.py
import json
import pytest
from pathlib import Path
from lib.mtgjson import build_inventory_cache, load_inventory_cache

MOCK_IDENTIFIERS = {
    "data": {
        "aaaa-1111": {
            "identifiers": {"tcgplayerProductId": "8553809"},
            "name": "Alacrian Jaguar",
            "rarity": "common",
            "setCode": "ATD",
            "printings": ["ATD"],
            "legalities": {"standard": "Legal"},
        }
    }
}

MOCK_PRICES = {
    "data": {
        "aaaa-1111": {
            "paper": {
                "tcgplayer": {
                    "retail": {
                        "normal": {"2026-02-01": 0.05, "2026-02-02": 0.06}
                    }
                }
            }
        }
    }
}

SAMPLE_INVENTORY_IDS = {"8553809", "9999999"}


def test_build_inventory_cache_maps_known_card():
    cache = build_inventory_cache(
        SAMPLE_INVENTORY_IDS, MOCK_IDENTIFIERS["data"], MOCK_PRICES["data"]
    )
    assert "8553809" in cache
    card = cache["8553809"]
    assert card["name"] == "Alacrian Jaguar"
    assert card["rarity"] == "common"
    assert "2026-02-01" in card["price_history"]


def test_build_inventory_cache_skips_unknown_ids():
    cache = build_inventory_cache(
        SAMPLE_INVENTORY_IDS, MOCK_IDENTIFIERS["data"], MOCK_PRICES["data"]
    )
    assert "9999999" not in cache


def test_build_inventory_cache_empty_history_when_no_prices():
    prices_no_tcg = {"aaaa-1111": {"paper": {}}}
    cache = build_inventory_cache(
        {"8553809"}, MOCK_IDENTIFIERS["data"], prices_no_tcg
    )
    assert cache["8553809"]["price_history"] == {}


def test_load_inventory_cache_roundtrip(tmp_path):
    cache = {"8553809": {"name": "Test", "price_history": {}}}
    cache_path = tmp_path / "inventory_cards.json"
    cache_path.write_text(json.dumps(cache))
    loaded = load_inventory_cache(str(tmp_path))
    assert loaded == cache


def test_load_inventory_cache_returns_empty_when_missing(tmp_path):
    result = load_inventory_cache(str(tmp_path))
    assert result == {}
