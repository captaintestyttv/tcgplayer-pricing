import os
import pytest
from lib.price_store import save_prices, load_prices, merge_price_dicts


def test_merge_price_dicts_non_overlapping():
    old = {"2026-01-01": 1.0, "2026-01-02": 2.0}
    new = {"2026-01-03": 3.0, "2026-01-04": 4.0}
    merged = merge_price_dicts(old, new)
    assert merged == {"2026-01-01": 1.0, "2026-01-02": 2.0,
                       "2026-01-03": 3.0, "2026-01-04": 4.0}


def test_merge_price_dicts_overlapping():
    old = {"2026-01-01": 1.0, "2026-01-02": 2.0}
    new = {"2026-01-02": 2.5, "2026-01-03": 3.0}
    merged = merge_price_dicts(old, new)
    assert merged == {"2026-01-01": 1.0, "2026-01-02": 2.5, "2026-01-03": 3.0}


def test_merge_prefers_new_on_conflict():
    old = {"2026-01-01": 1.0}
    new = {"2026-01-01": 9.99}
    merged = merge_price_dicts(old, new)
    assert merged["2026-01-01"] == 9.99


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("lib.price_store.PRICE_HISTORY_DIR", str(tmp_path))
    prices = {"2026-01-01": 1.5, "2026-01-02": 2.0}
    save_prices("ABC123", prices, source="mtgjson")
    loaded = load_prices("ABC123")
    assert loaded == prices


def test_history_grows_across_saves(tmp_path, monkeypatch):
    monkeypatch.setattr("lib.price_store.PRICE_HISTORY_DIR", str(tmp_path))
    save_prices("ABC123", {"2026-01-01": 1.0, "2026-01-02": 2.0}, source="mtgjson")
    save_prices("ABC123", {"2026-01-03": 3.0, "2026-01-04": 4.0}, source="mtgjson")
    loaded = load_prices("ABC123")
    assert len(loaded) == 4
    assert loaded["2026-01-01"] == 1.0
    assert loaded["2026-01-04"] == 4.0


def test_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("lib.price_store.PRICE_HISTORY_DIR", str(tmp_path))
    loaded = load_prices("NONEXISTENT")
    assert loaded == {}


def test_save_empty_prices_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr("lib.price_store.PRICE_HISTORY_DIR", str(tmp_path))
    save_prices("ABC123", {}, source="mtgjson")
    loaded = load_prices("ABC123")
    assert loaded == {}


def test_different_price_types(tmp_path, monkeypatch):
    monkeypatch.setattr("lib.price_store.PRICE_HISTORY_DIR", str(tmp_path))
    save_prices("ABC123", {"2026-01-01": 1.0}, source="mtgjson", price_type="normal")
    save_prices("ABC123", {"2026-01-01": 5.0}, source="mtgjson", price_type="foil")
    assert load_prices("ABC123", "normal") == {"2026-01-01": 1.0}
    assert load_prices("ABC123", "foil") == {"2026-01-01": 5.0}


def test_second_save_overwrites_on_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr("lib.price_store.PRICE_HISTORY_DIR", str(tmp_path))
    save_prices("ABC123", {"2026-01-01": 1.0}, source="old_source")
    save_prices("ABC123", {"2026-01-01": 9.99}, source="new_source")
    loaded = load_prices("ABC123")
    assert loaded["2026-01-01"] == 9.99
