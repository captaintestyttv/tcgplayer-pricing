# tests/test_mtgjson.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from lib.mtgjson import (
    build_inventory_cache, build_sku_to_uuid, load_inventory_cache,
    download_json, detect_changes,
)

MOCK_IDENTIFIERS = {
    "data": {
        "aaaa-1111": {
            "identifiers": {"tcgplayerProductId": "8553809"},
            "name": "Alacrian Jaguar",
            "rarity": "common",
            "setCode": "ATD",
            "printings": ["ATD"],
            "legalities": {"standard": "Legal"},
            "edhrecRank": 5000,
            "edhrecSaltiness": 0.5,
            "isReserved": False,
            "supertypes": [],
            "types": ["Creature"],
            "subtypes": ["Cat"],
            "colorIdentity": ["G"],
            "keywords": ["Vigilance"],
            "manaValue": 2.0,
            "text": "Vigilance",
        }
    }
}

MOCK_PRICES = {
    "data": {
        "aaaa-1111": {
            "paper": {
                "tcgplayer": {
                    "retail": {
                        "normal": {"2026-02-01": 0.05, "2026-02-02": 0.06},
                        "foil": {"2026-02-01": 0.15, "2026-02-02": 0.18},
                    },
                    "buylist": {
                        "normal": {"2026-02-01": 0.02, "2026-02-02": 0.03},
                    },
                }
            }
        }
    }
}

MOCK_SKUS = {
    "aaaa-1111": [{"skuId": 8553809}]
}

SAMPLE_INVENTORY_IDS = {"8553809", "9999999"}
SAMPLE_SKU_TO_UUID = build_sku_to_uuid(MOCK_SKUS)


def test_build_inventory_cache_maps_known_card():
    cache = build_inventory_cache(
        SAMPLE_INVENTORY_IDS, MOCK_IDENTIFIERS["data"], MOCK_PRICES["data"],
        SAMPLE_SKU_TO_UUID,
    )
    assert "8553809" in cache
    card = cache["8553809"]
    assert card["name"] == "Alacrian Jaguar"
    assert card["rarity"] == "common"
    assert "2026-02-01" in card["price_history"]


def test_build_inventory_cache_skips_unknown_ids():
    cache = build_inventory_cache(
        SAMPLE_INVENTORY_IDS, MOCK_IDENTIFIERS["data"], MOCK_PRICES["data"],
        SAMPLE_SKU_TO_UUID,
    )
    assert "9999999" not in cache


def test_build_inventory_cache_empty_history_when_no_prices():
    prices_no_tcg = {"aaaa-1111": {"paper": {}}}
    cache = build_inventory_cache(
        {"8553809"}, MOCK_IDENTIFIERS["data"], prices_no_tcg,
        SAMPLE_SKU_TO_UUID,
    )
    assert cache["8553809"]["price_history"] == {}
    assert cache["8553809"]["foil_price_history"] == {}
    assert cache["8553809"]["buylist_price_history"] == {}


def test_load_inventory_cache_roundtrip(tmp_path):
    cache = {"8553809": {"name": "Test", "price_history": {}}}
    cache_path = tmp_path / "inventory_cards.json"
    cache_path.write_text(json.dumps(cache))
    loaded = load_inventory_cache(str(tmp_path))
    assert loaded == cache


def test_load_inventory_cache_returns_empty_when_missing(tmp_path):
    result = load_inventory_cache(str(tmp_path))
    assert result == {}


def test_download_retries_on_failure(tmp_path):
    import requests as req
    dest = str(tmp_path / "test.json")
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b'{"data": {}}']
    mock_response.raise_for_status.return_value = None
    with patch("lib.mtgjson.requests.get") as mock_get, \
         patch("lib.mtgjson.time.sleep"):
        mock_get.side_effect = [
            req.ConnectionError("Connection reset"),
            mock_response,
        ]
        download_json("https://example.com/test.json", dest, force=True)
    assert Path(dest).exists()


def test_download_atomic_write_cleans_tmp_on_failure(tmp_path):
    import requests as req
    dest = str(tmp_path / "test.json")
    with patch("lib.mtgjson.requests.get") as mock_get, \
         patch("lib.mtgjson.time.sleep"), \
         patch("lib.mtgjson.DOWNLOAD_MAX_RETRIES", 1):
        mock_get.side_effect = req.ConnectionError("Network error")
        with pytest.raises(req.ConnectionError):
            download_json("https://example.com/test.json", dest, force=True)
    assert not Path(dest).exists()
    assert not Path(dest + ".tmp").exists()


# Phase 1: metadata fields in cache
def test_build_inventory_cache_includes_metadata():
    cache = build_inventory_cache(
        {"8553809"}, MOCK_IDENTIFIERS["data"], MOCK_PRICES["data"],
        SAMPLE_SKU_TO_UUID,
    )
    card = cache["8553809"]
    assert card["edhrecRank"] == 5000
    assert card["edhrecSaltiness"] == 0.5
    assert card["isReserved"] is False
    assert card["supertypes"] == []
    assert card["types"] == ["Creature"]
    assert card["subtypes"] == ["Cat"]
    assert card["colorIdentity"] == ["G"]
    assert card["keywords"] == ["Vigilance"]
    assert card["manaValue"] == 2.0
    assert card["text"] == "Vigilance"


# Phase 2: foil & buylist price channels
def test_build_inventory_cache_includes_foil_buylist():
    cache = build_inventory_cache(
        {"8553809"}, MOCK_IDENTIFIERS["data"], MOCK_PRICES["data"],
        SAMPLE_SKU_TO_UUID,
    )
    card = cache["8553809"]
    assert card["foil_price_history"] == {"2026-02-01": 0.15, "2026-02-02": 0.18}
    assert card["buylist_price_history"] == {"2026-02-01": 0.02, "2026-02-02": 0.03}


# Phase 4: change detection
def test_detect_changes_reprint():
    old_cache = {"1": {"printings": ["A"], "legalities": {"standard": "legal"}}}
    new_cache = {"1": {"printings": ["A", "B"], "legalities": {"standard": "legal"},
                       "recently_reprinted": 0, "legality_changed": 0}}
    detect_changes(old_cache, new_cache)
    assert new_cache["1"]["recently_reprinted"] == 1
    assert new_cache["1"]["legality_changed"] == 0


def test_detect_changes_legality():
    old_cache = {"1": {"printings": ["A"], "legalities": {"standard": "legal"}}}
    new_cache = {"1": {"printings": ["A"], "legalities": {"standard": "banned"},
                       "recently_reprinted": 0, "legality_changed": 0}}
    detect_changes(old_cache, new_cache)
    assert new_cache["1"]["recently_reprinted"] == 0
    assert new_cache["1"]["legality_changed"] == 1


def test_detect_changes_new_card():
    old_cache = {}
    new_cache = {"1": {"printings": ["A"], "legalities": {"standard": "legal"},
                       "recently_reprinted": 0, "legality_changed": 0}}
    detect_changes(old_cache, new_cache)
    # New card has no old entry: printings > 0 > 0 is False, legalities != {} is True
    assert new_cache["1"]["recently_reprinted"] == 1
    assert new_cache["1"]["legality_changed"] == 1
