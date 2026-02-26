import numpy as np
from sklearn.linear_model import LinearRegression

MIN_HISTORY_DAYS = 14
MIN_PRICE = 0.01


def forecast_card(price_history: dict, days_ahead: int = 7) -> float | None:
    """Predict price N days ahead using linear regression. Returns None if insufficient data."""
    sorted_entries = sorted(price_history.items())
    price_vals = np.array([float(v) for _, v in sorted_entries])

    if len(price_vals) < MIN_HISTORY_DAYS:
        return None

    # Use last 90 days at most
    price_vals = price_vals[-90:]
    x = np.arange(len(price_vals)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, price_vals)

    future_x = np.array([[len(price_vals) + days_ahead - 1]])
    predicted = float(model.predict(future_x)[0])
    return max(MIN_PRICE, round(predicted, 4))


def trend_direction(price_history: dict) -> str:
    """Return 'up', 'down', or 'flat' based on 7-day slope."""
    sorted_entries = sorted(price_history.items())
    price_vals = [float(v) for _, v in sorted_entries]

    if len(price_vals) < 7:
        return "flat"

    recent = price_vals[-7:]
    slope = (recent[-1] - recent[0]) / max(recent[0], MIN_PRICE)

    if slope > 0.03:
        return "up"
    if slope < -0.03:
        return "down"
    return "flat"
