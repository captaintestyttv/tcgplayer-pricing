# tests/test_mtgjson.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from lib.mtgjson import build_inventory_cache, build_sku_to_uuid, load_inventory_cache, download_json

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
