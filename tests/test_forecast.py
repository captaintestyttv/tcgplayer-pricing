# tests/test_forecast.py
import json
import pytest
from pathlib import Path
from lib.forecast import forecast_card, forecast_with_confidence, trend_direction

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


def test_forecast_exactly_14_days_boundary():
    history = {f"2026-01-{i:02d}": 1.0 + i * 0.01 for i in range(1, 15)}
    assert len(history) == 14
    result = forecast_card(history, days_ahead=7)
    assert result is not None
    assert isinstance(result, float)


def test_forecast_13_days_returns_none():
    history = {f"2026-01-{i:02d}": 1.0 for i in range(1, 14)}
    assert len(history) == 13
    assert forecast_card(history, days_ahead=7) is None


def test_forecast_empty_history():
    assert forecast_card({}, days_ahead=7) is None


def test_trend_direction_with_zero_start():
    """trend_direction should not crash when first price is 0."""
    history = {f"2026-01-{i:02d}": 0.0 if i == 1 else 1.0 for i in range(1, 10)}
    result = trend_direction(history)
    assert result in ("up", "down", "flat")


# Improvement 6: forecast with confidence
def test_forecast_with_confidence_returns_dict(cards):
    ph = cards["111111"]["price_history"]
    result = forecast_with_confidence(ph, 7)
    assert result is not None
    assert "predicted" in result
    assert "lower" in result
    assert "upper" in result
    assert "std_error" in result
    assert "r_squared" in result


def test_confidence_interval_bounds(cards):
    ph = cards["111111"]["price_history"]
    result = forecast_with_confidence(ph, 7)
    assert result["lower"] <= result["predicted"] <= result["upper"]


def test_confidence_30d_wider_than_7d(cards):
    ph = cards["111111"]["price_history"]
    r7 = forecast_with_confidence(ph, 7)
    r30 = forecast_with_confidence(ph, 30)
    width_7 = r7["upper"] - r7["lower"]
    width_30 = r30["upper"] - r30["lower"]
    assert width_30 >= width_7


def test_confidence_r_squared_between_0_and_1(cards):
    ph = cards["111111"]["price_history"]
    result = forecast_with_confidence(ph, 7)
    assert 0 <= result["r_squared"] <= 1


def test_confidence_returns_none_for_insufficient_data():
    assert forecast_with_confidence({}, 7) is None
    short = {f"2026-01-{i:02d}": 1.0 for i in range(1, 10)}
    assert forecast_with_confidence(short, 7) is None


def test_confidence_lower_at_least_min_price():
    # Declining prices should still floor at MIN_PRICE
    history = {f"2026-01-{i:02d}": max(0.02, 1.0 - i * 0.05) for i in range(1, 20)}
    result = forecast_with_confidence(history, 30)
    if result:
        assert result["lower"] >= 0.01
        assert result["predicted"] >= 0.01


def test_confidence_matches_point_forecast(cards):
    """Point forecast and confidence forecast should agree on predicted value."""
    ph = cards["111111"]["price_history"]
    point = forecast_card(ph, 7)
    conf = forecast_with_confidence(ph, 7)
    assert point == pytest.approx(conf["predicted"], abs=0.001)
