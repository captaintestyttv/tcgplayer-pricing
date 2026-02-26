# tests/test_forecast.py
import json
import pytest
from pathlib import Path
from lib.forecast import forecast_card, trend_direction

FIXTURES = Path(__file__).parent / "fixtures" / "inventory_cards.json"

@pytest.fixture
def cards():
    return json.loads(FIXTURES.read_text())

def test_forecast_returns_float_for_sufficient_history(cards):
    price_history = cards["111111"]["price_history"]
    result = forecast_card(price_history, days_ahead=7)
    assert isinstance(result, float)
    assert result > 0

def test_forecast_returns_none_for_insufficient_history(cards):
    price_history = cards["222222"]["price_history"]  # only 5 days
    result = forecast_card(price_history, days_ahead=7)
    assert result is None

def test_forecast_predicts_higher_for_rising_card(cards):
    price_history = cards["111111"]["price_history"]
    current = list(price_history.values())[-1]
    pred_7d = forecast_card(price_history, days_ahead=7)
    assert pred_7d > current

def test_forecast_30d_higher_than_7d_for_rising_card(cards):
    price_history = cards["111111"]["price_history"]
    pred_7d = forecast_card(price_history, days_ahead=7)
    pred_30d = forecast_card(price_history, days_ahead=30)
    assert pred_30d > pred_7d

def test_trend_up_for_rising_card(cards):
    assert trend_direction(cards["111111"]["price_history"]) == "up"

def test_trend_flat_for_insufficient_history(cards):
    assert trend_direction(cards["222222"]["price_history"]) == "flat"

def test_forecast_never_below_minimum(cards):
    tiny_history = {f"2026-02-{i:02d}": max(0.01, 1.0 - i * 0.05) for i in range(1, 20)}
    result = forecast_card(tiny_history, days_ahead=30)
    assert result is None or result >= 0.01
