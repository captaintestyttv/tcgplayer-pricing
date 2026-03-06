# tests/test_perf.py
"""Performance benchmarks for training data generation.

Run with: python -m pytest tests/test_perf.py -v -s
These tests are NOT about correctness — they measure wall time.
"""
import time
import pytest
from lib.features import extract_features, generate_training_data


def _make_card(n_days: int, card_id: str = "1") -> dict:
    """Create a synthetic card with n_days of price history."""
    return {
        "rarity": "rare",
        "setCode": "TST",
        "printings": ["TST", "TST2"],
        "legalities": {"standard": "legal", "pioneer": "legal"},
        "edhrecRank": 1500,
        "edhrecSaltiness": 1.2,
        "isReserved": False,
        "supertypes": ["Legendary"],
        "types": ["Creature"],
        "subtypes": ["Elf", "Warrior"],
        "colorIdentity": ["G"],
        "keywords": ["Trample"],
        "manaValue": 3.0,
        "text": "Trample",
        "foil_price_history": {f"2026-01-{i:02d}": 3.0 + i * 0.05 for i in range(1, min(n_days + 1, 32))},
        "buylist_price_history": {f"2026-01-{i:02d}": 0.5 + i * 0.01 for i in range(1, min(n_days + 1, 32))},
        "recently_reprinted": 0,
        "legality_changed": 0,
        "setReleaseDate": "2025-10-01",
        "setIsPartialPreview": False,
        "price_history": _make_price_history(n_days),
    }


def _make_price_history(n_days: int) -> dict:
    """Generate n_days of rising price history starting 2025-06-01."""
    from datetime import datetime, timedelta
    base = datetime(2025, 6, 1)
    return {
        (base + timedelta(days=i)).strftime("%Y-%m-%d"): 1.0 + i * 0.02
        for i in range(n_days)
    }


@pytest.mark.slow
def test_generate_training_data_perf_100_cards():
    """Benchmark: 100 cards x 90 days = ~6000 windows."""
    cards = {str(i): _make_card(90, str(i)) for i in range(100)}
    start = time.time()
    rows = generate_training_data(cards)
    elapsed = time.time() - start
    print(f"\n  100 cards x 90 days: {len(rows)} rows in {elapsed:.2f}s")
    assert len(rows) > 0


@pytest.mark.slow
def test_generate_training_data_perf_500_cards():
    """Benchmark: 500 cards x 90 days = ~30000 windows."""
    cards = {str(i): _make_card(90, str(i)) for i in range(500)}
    start = time.time()
    rows = generate_training_data(cards)
    elapsed = time.time() - start
    print(f"\n  500 cards x 90 days: {len(rows)} rows in {elapsed:.2f}s")
    assert len(rows) > 0


@pytest.mark.slow
def test_extract_features_perf_10k_calls():
    """Benchmark: 10,000 extract_features calls (simulates inner loop)."""
    card = _make_card(90)
    from datetime import datetime
    ref = datetime(2026, 3, 1)
    start = time.time()
    for _ in range(10_000):
        extract_features("1", card, reference_date=ref)
    elapsed = time.time() - start
    print(f"\n  10,000 extract_features calls: {elapsed:.2f}s ({elapsed/10_000*1000:.2f}ms/call)")
