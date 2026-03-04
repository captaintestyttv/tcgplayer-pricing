import numpy as np
from sklearn.linear_model import LinearRegression

from lib.config import MIN_HISTORY_DAYS, MAX_HISTORY_DAYS, MIN_PRICE, TREND_THRESHOLD


def forecast_card(price_history: dict, days_ahead: int = 7) -> float | None:
    """Predict price N days ahead using linear regression. Returns None if insufficient data."""
    sorted_entries = sorted(price_history.items())
    price_vals = np.array([float(v) for _, v in sorted_entries])

    if len(price_vals) < MIN_HISTORY_DAYS:
        return None

    price_vals = price_vals[-MAX_HISTORY_DAYS:]
    x = np.arange(len(price_vals)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, price_vals)

    future_x = np.array([[len(price_vals) + days_ahead - 1]])
    predicted = float(model.predict(future_x)[0])
    return max(MIN_PRICE, round(predicted, 4))


def forecast_with_confidence(price_history: dict, days_ahead: int = 7) -> dict | None:
    """Predict price with confidence interval. Returns None if insufficient data.

    Returns dict with: predicted, lower, upper, std_error, r_squared
    """
    sorted_entries = sorted(price_history.items())
    price_vals = np.array([float(v) for _, v in sorted_entries])

    if len(price_vals) < MIN_HISTORY_DAYS:
        return None

    price_vals = price_vals[-MAX_HISTORY_DAYS:]
    n = len(price_vals)
    x = np.arange(n).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, price_vals)

    # R-squared
    r_squared = float(model.score(x, price_vals))

    # Residual standard error
    y_pred_train = model.predict(x)
    residuals = price_vals - y_pred_train
    # degrees of freedom = n - 2 (slope + intercept)
    std_error = float(np.sqrt(np.sum(residuals ** 2) / max(n - 2, 1)))

    # Prediction at future point
    x_future = len(price_vals) + days_ahead - 1
    predicted = float(model.predict(np.array([[x_future]]))[0])

    # Prediction interval: wider for points further from the mean
    x_mean = float(np.mean(x))
    x_var = float(np.sum((x.flatten() - x_mean) ** 2))
    # Leverage for the prediction point
    leverage = 1.0 + 1.0 / n + (x_future - x_mean) ** 2 / max(x_var, 1e-10)
    margin = 1.96 * std_error * np.sqrt(leverage)

    lower = max(MIN_PRICE, round(predicted - margin, 4))
    upper = max(MIN_PRICE, round(predicted + margin, 4))
    predicted = max(MIN_PRICE, round(predicted, 4))

    return {
        "predicted": predicted,
        "lower": lower,
        "upper": upper,
        "std_error": round(std_error, 4),
        "r_squared": round(r_squared, 4),
    }


def trend_direction(price_history: dict) -> str:
    """Return 'up', 'down', or 'flat' based on 7-day slope."""
    sorted_entries = sorted(price_history.items())
    price_vals = [float(v) for _, v in sorted_entries]

    if len(price_vals) < 7:
        return "flat"

    recent = price_vals[-7:]
    slope = (recent[-1] - recent[0]) / max(recent[0], MIN_PRICE)

    if slope > TREND_THRESHOLD:
        return "up"
    if slope < -TREND_THRESHOLD:
        return "down"
    return "flat"
