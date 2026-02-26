# Predictive Pricing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add XGBoost spike classification and linear regression price forecasting to the TCGPlayer monitor, backed by MTGJson data and optional remote GPU training via Tailscale SSH.

**Architecture:** Python lib modules (`mtgjson`, `features`, `forecast`, `spike`, `predict`) invoked from `monitor.sh` via new `sync`, `train`, and `predict` commands. MTGJson data is downloaded once and cached as a lean per-inventory JSON file. Training can run locally (XGBoost CPU) or remotely (XGBoost GPU) via SSH + rsync — the model file is identical either way.

**Tech Stack:** Python 3, `pandas`, `numpy`, `scikit-learn` (LinearRegression), `xgboost`, `requests`, `pytest`

---

### Task 1: Project scaffolding

**Files:**
- Create: `lib/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/fixtures/inventory_cards.json`
- Create: `requirements.txt`
- Modify: `.gitignore`

**Step 1: Create directory structure**

```bash
mkdir -p lib tests/fixtures data/mtgjson models
touch lib/__init__.py tests/__init__.py
```

**Step 2: Create `requirements.txt`**

```
pandas
numpy
scikit-learn
xgboost
requests
pytest
```

**Step 3: Create `tests/fixtures/inventory_cards.json`**

This fixture is used by all subsequent tests. It has two cards: one with 90 days of history (enough for forecasting + training), one with only 5 days (insufficient for forecasting).

```json
{
  "111111": {
    "uuid": "aaaa-1111",
    "name": "Test Rare",
    "rarity": "rare",
    "setCode": "TST",
    "printings": ["TST"],
    "legalities": { "standard": "legal", "pioneer": "legal" },
    "price_history": {
      "2025-11-27": 1.00, "2025-11-28": 1.02, "2025-11-29": 1.01,
      "2025-11-30": 1.03, "2025-12-01": 1.05, "2025-12-02": 1.04,
      "2025-12-03": 1.06, "2025-12-04": 1.08, "2025-12-05": 1.07,
      "2025-12-06": 1.09, "2025-12-07": 1.11, "2025-12-08": 1.10,
      "2025-12-09": 1.12, "2025-12-10": 1.14, "2025-12-11": 1.13,
      "2025-12-12": 1.15, "2025-12-13": 1.17, "2025-12-14": 1.16,
      "2025-12-15": 1.18, "2025-12-16": 1.20, "2025-12-17": 1.22,
      "2025-12-18": 1.21, "2025-12-19": 1.23, "2025-12-20": 1.25,
      "2025-12-21": 1.24, "2025-12-22": 1.26, "2025-12-23": 1.28,
      "2025-12-24": 1.27, "2025-12-25": 1.29, "2025-12-26": 1.31,
      "2025-12-27": 1.30, "2025-12-28": 1.32, "2025-12-29": 1.34,
      "2025-12-30": 1.33, "2025-12-31": 1.35, "2026-01-01": 1.37,
      "2026-01-02": 1.36, "2026-01-03": 1.38, "2026-01-04": 1.40,
      "2026-01-05": 1.39, "2026-01-06": 1.41, "2026-01-07": 1.43,
      "2026-01-08": 1.42, "2026-01-09": 1.44, "2026-01-10": 1.46,
      "2026-01-11": 1.45, "2026-01-12": 1.47, "2026-01-13": 1.49,
      "2026-01-14": 1.48, "2026-01-15": 1.50, "2026-01-16": 1.52,
      "2026-01-17": 1.51, "2026-01-18": 1.53, "2026-01-19": 1.55,
      "2026-01-20": 1.54, "2026-01-21": 1.56, "2026-01-22": 1.58,
      "2026-01-23": 1.57, "2026-01-24": 1.59, "2026-01-25": 1.61,
      "2026-01-26": 1.60, "2026-01-27": 1.62, "2026-01-28": 1.64,
      "2026-01-29": 1.63, "2026-01-30": 1.65, "2026-01-31": 1.67,
      "2026-02-01": 1.66, "2026-02-02": 1.68, "2026-02-03": 1.70,
      "2026-02-04": 1.69, "2026-02-05": 1.71, "2026-02-06": 1.73,
      "2026-02-07": 1.72, "2026-02-08": 1.74, "2026-02-09": 1.76,
      "2026-02-10": 1.75, "2026-02-11": 1.77, "2026-02-12": 1.79,
      "2026-02-13": 1.78, "2026-02-14": 1.80, "2026-02-15": 1.82,
      "2026-02-16": 1.81, "2026-02-17": 1.83, "2026-02-18": 1.85,
      "2026-02-19": 1.84, "2026-02-20": 1.86, "2026-02-21": 1.88,
      "2026-02-22": 1.87, "2026-02-23": 1.89, "2026-02-24": 1.91
    }
  },
  "222222": {
    "uuid": "bbbb-2222",
    "name": "Test Common",
    "rarity": "common",
    "setCode": "TST",
    "printings": ["TST", "TST2", "TST3"],
    "legalities": { "standard": "legal" },
    "price_history": {
      "2026-02-20": 0.05,
      "2026-02-21": 0.05,
      "2026-02-22": 0.05,
      "2026-02-23": 0.06,
      "2026-02-24": 0.06
    }
  }
}
```

**Step 4: Update `.gitignore`**

Add:
```
data/mtgjson/AllPrices.json
data/mtgjson/AllIdentifiers.json
models/
```

Keep `data/mtgjson/inventory_cards.json` tracked (it's the lean cache, not the raw downloads).

**Step 5: Install dependencies on Pi**

```bash
pip install -r requirements.txt
```

**Step 6: Commit**

```bash
git add lib/__init__.py tests/__init__.py tests/fixtures/inventory_cards.json requirements.txt .gitignore
git commit -m "feat: scaffold lib/, tests/, and requirements for predictive pricing"
```

---

### Task 2: Feature extraction (`lib/features.py`)

**Files:**
- Create: `lib/features.py`
- Create: `tests/test_features.py`

**Step 1: Write the failing tests**

```python
# tests/test_features.py
import json
import pytest
from pathlib import Path
from lib.features import extract_features, generate_training_data

FIXTURES = Path(__file__).parent / "fixtures" / "inventory_cards.json"

@pytest.fixture
def cards():
    return json.loads(FIXTURES.read_text())

def test_extract_features_keys(cards):
    feat = extract_features("111111", cards["111111"])
    expected_keys = {
        "tcgplayer_id", "rarity_rank", "num_printings", "set_age_days",
        "formats_legal_count", "price_momentum_7d", "price_volatility_30d",
        "current_price",
    }
    assert expected_keys == set(feat.keys())

def test_rarity_rank(cards):
    feat = extract_features("111111", cards["111111"])
    assert feat["rarity_rank"] == 2  # rare = 2

def test_num_printings(cards):
    feat = extract_features("222222", cards["222222"])
    assert feat["num_printings"] == 3

def test_formats_legal_count(cards):
    feat = extract_features("111111", cards["111111"])
    assert feat["formats_legal_count"] == 2  # standard + pioneer

def test_current_price_is_last_history_entry(cards):
    feat = extract_features("111111", cards["111111"])
    assert feat["current_price"] == pytest.approx(1.91)

def test_price_momentum_positive_for_rising_card(cards):
    feat = extract_features("111111", cards["111111"])
    assert feat["price_momentum_7d"] > 0

def test_generate_training_data_structure(cards):
    rows = generate_training_data(cards)
    assert len(rows) > 0
    assert "spike" in rows[0]
    assert "rarity_rank" in rows[0]

def test_generate_training_data_labels_spike(cards):
    # 111111 has steady upward trend — some windows should be labeled spike=1
    rows = generate_training_data(cards)
    card_rows = [r for r in rows if r["tcgplayer_id"] == "111111"]
    assert any(r["spike"] == 1 for r in card_rows)
```

**Step 2: Run to verify they fail**

```bash
pytest tests/test_features.py -v
```
Expected: `ModuleNotFoundError: No module named 'lib.features'`

**Step 3: Implement `lib/features.py`**

```python
import numpy as np
from datetime import datetime

RARITY_RANK = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3}
SPIKE_THRESHOLD = 0.20  # >20% in 30 days = spike


def extract_features(tcgplayer_id: str, card: dict) -> dict:
    prices = card.get("price_history", {})
    sorted_entries = sorted(prices.items())
    price_vals = [float(v) for _, v in sorted_entries]

    current_price = price_vals[-1] if price_vals else 0.0

    if len(price_vals) >= 7 and price_vals[-7] > 0:
        momentum_7d = (price_vals[-1] - price_vals[-7]) / price_vals[-7]
    else:
        momentum_7d = 0.0

    volatility_30d = float(np.std(price_vals[-30:])) if len(price_vals) >= 2 else 0.0

    if sorted_entries:
        first_date = datetime.fromisoformat(sorted_entries[0][0])
        set_age_days = (datetime.now() - first_date).days
    else:
        set_age_days = 0

    return {
        "tcgplayer_id": tcgplayer_id,
        "rarity_rank": RARITY_RANK.get(card.get("rarity", ""), 0),
        "num_printings": len(card.get("printings", [])),
        "set_age_days": set_age_days,
        "formats_legal_count": sum(
            1 for v in card.get("legalities", {}).values() if v == "legal"
        ),
        "price_momentum_7d": momentum_7d,
        "price_volatility_30d": volatility_30d,
        "current_price": current_price,
    }


def generate_training_data(cards: dict) -> list[dict]:
    """Generate (features, spike_label) rows from historical windows."""
    rows = []
    for tcgplayer_id, card in cards.items():
        prices = sorted(card.get("price_history", {}).items())
        price_vals = [float(v) for _, v in prices]

        if len(price_vals) < 31:
            continue

        for i in range(len(price_vals) - 30):
            window = price_vals[i : i + 31]
            spike = int(
                window[0] > 0 and (max(window[1:]) - window[0]) / window[0] > SPIKE_THRESHOLD
            )
            # Snapshot card at position i for features
            snapshot = dict(card)
            snapshot["price_history"] = dict(prices[:i+1])
            snapshot["rarity"] = card.get("rarity", "")
            feat = extract_features(tcgplayer_id, snapshot)
            feat["spike"] = spike
            rows.append(feat)

    return rows
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_features.py -v
```
Expected: all 8 tests PASS

**Step 5: Commit**

```bash
git add lib/features.py tests/test_features.py
git commit -m "feat: add feature extraction with spike labeling"
```

---

### Task 3: Price forecasting (`lib/forecast.py`)

**Files:**
- Create: `lib/forecast.py`
- Create: `tests/test_forecast.py`

**Step 1: Write the failing tests**

```python
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
    # Even if regression extrapolates negative, floor at 0.01
    tiny_history = {f"2026-02-{i:02d}": max(0.01, 1.0 - i * 0.05) for i in range(1, 20)}
    result = forecast_card(tiny_history, days_ahead=30)
    assert result is None or result >= 0.01
```

**Step 2: Run to verify they fail**

```bash
pytest tests/test_forecast.py -v
```
Expected: `ModuleNotFoundError: No module named 'lib.forecast'`

**Step 3: Implement `lib/forecast.py`**

```python
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
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_forecast.py -v
```
Expected: all 7 tests PASS

**Step 5: Commit**

```bash
git add lib/forecast.py tests/test_forecast.py
git commit -m "feat: add per-card linear regression price forecasting"
```

---

### Task 4: Spike classifier (`lib/spike.py`)

**Files:**
- Create: `lib/spike.py`
- Create: `tests/test_spike.py`

**Step 1: Write the failing tests**

```python
# tests/test_spike.py
import json
import pytest
import tempfile
from pathlib import Path
from lib.features import generate_training_data
from lib.spike import train, score, FEATURE_COLS

FIXTURES = Path(__file__).parent / "fixtures" / "inventory_cards.json"

@pytest.fixture
def cards():
    return json.loads(FIXTURES.read_text())

@pytest.fixture
def trained_model(cards, tmp_path):
    """Train a model on fixture data and return path to saved model."""
    model_path = str(tmp_path / "test_model.json")
    rows = generate_training_data(cards)
    train(rows, model_path, device="cpu")
    return model_path

def test_train_creates_model_file(cards, tmp_path):
    model_path = str(tmp_path / "model.json")
    rows = generate_training_data(cards)
    train(rows, model_path, device="cpu")
    assert Path(model_path).exists()

def test_score_returns_list_of_floats(cards, trained_model):
    from lib.features import extract_features
    features = [extract_features(tid, card) for tid, card in cards.items()]
    scores = score(features, trained_model)
    assert len(scores) == len(features)
    assert all(0.0 <= s <= 1.0 for s in scores)

def test_score_requires_only_feature_cols(cards, trained_model):
    # score() should handle cards with insufficient price history (no spike label needed)
    from lib.features import extract_features
    features = [extract_features("222222", cards["222222"])]
    scores = score(features, trained_model)
    assert len(scores) == 1

def test_train_raises_on_empty_rows(tmp_path):
    model_path = str(tmp_path / "model.json")
    with pytest.raises(ValueError, match="No training data"):
        train([], model_path, device="cpu")
```

**Step 2: Run to verify they fail**

```bash
pytest tests/test_spike.py -v
```
Expected: `ModuleNotFoundError: No module named 'lib.spike'`

**Step 3: Implement `lib/spike.py`**

```python
import pandas as pd
import xgboost as xgb

FEATURE_COLS = [
    "rarity_rank",
    "num_printings",
    "set_age_days",
    "formats_legal_count",
    "price_momentum_7d",
    "price_volatility_30d",
    "current_price",
]


def train(rows: list[dict], model_path: str, device: str = "cpu") -> None:
    """Train XGBoost spike classifier and save to model_path."""
    if not rows:
        raise ValueError("No training data provided")

    df = pd.DataFrame(rows)
    X = df[FEATURE_COLS].fillna(0)
    y = df["spike"]

    model = xgb.XGBClassifier(
        device=device,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X, y)
    model.save_model(model_path)


def score(features: list[dict], model_path: str) -> list[float]:
    """Return spike probability (0-1) for each feature dict."""
    df = pd.DataFrame(features)[FEATURE_COLS].fillna(0)
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model.predict_proba(df)[:, 1].tolist()
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_spike.py -v
```
Expected: all 4 tests PASS

**Step 5: Commit**

```bash
git add lib/spike.py tests/test_spike.py
git commit -m "feat: add XGBoost spike classifier (train + score)"
```

---

### Task 5: MTGJson sync (`lib/mtgjson.py`)

**Files:**
- Create: `lib/mtgjson.py`
- Create: `tests/test_mtgjson.py`

**Step 1: Write the failing tests**

```python
# tests/test_mtgjson.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from lib.mtgjson import build_inventory_cache, load_inventory_cache

# Minimal MTGJson-shaped responses for mocking
MOCK_IDENTIFIERS = {
    "data": {
        "aaaa-1111": {
            "identifiers": {"tcgplayerProductId": "8553809"},
            "name": "Alacrian Jaguar",
            "rarity": "common",
            "setCode": "ATD",
            "printings": ["ATD"],
            "legalities": {"standard": "legal"},
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

SAMPLE_INVENTORY_IDS = {"8553809", "9999999"}  # 9999999 has no MTGJson match


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
    prices_no_tcg = {
        "aaaa-1111": {"paper": {}}  # no tcgplayer key
    }
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
```

**Step 2: Run to verify they fail**

```bash
pytest tests/test_mtgjson.py -v
```
Expected: `ModuleNotFoundError: No module named 'lib.mtgjson'`

**Step 3: Implement `lib/mtgjson.py`**

```python
import csv
import gzip
import json
import os
import sys
from pathlib import Path

import requests

MTGJSON_BASE = "https://mtgjson.com/api/v5"
CACHE_FILENAME = "inventory_cards.json"


def download_json(url: str, dest_path: str) -> None:
    """Download a (possibly gzip-compressed) JSON file."""
    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    print(f"  → saved to {dest_path}")


def load_json_file(path: str) -> dict:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_inventory_ids(history_dir: str) -> set[str]:
    """Extract TCGplayer IDs from latest.csv."""
    latest = os.path.join(history_dir, "latest.csv")
    if not os.path.exists(latest):
        return set()
    ids = set()
    with open(latest, newline="") as f:
        for row in csv.DictReader(f):
            tid = row.get("TCGplayer Id", "").strip()
            if tid:
                ids.add(tid)
    return ids


def build_inventory_cache(
    inventory_ids: set[str],
    identifiers_data: dict,
    prices_data: dict,
) -> dict:
    """Build lean cache from MTGJson data scoped to inventory IDs."""
    # Build reverse map: tcgplayerProductId → uuid + card metadata
    tcg_to_card = {}
    for uuid, card in identifiers_data.items():
        tcg_id = card.get("identifiers", {}).get("tcgplayerProductId")
        if tcg_id:
            tcg_to_card[tcg_id] = (uuid, card)

    cache = {}
    for tcg_id in inventory_ids:
        if tcg_id not in tcg_to_card:
            continue
        uuid, card = tcg_to_card[tcg_id]

        # Extract price history
        price_history = {}
        try:
            normal = prices_data[uuid]["paper"]["tcgplayer"]["retail"]["normal"]
            price_history = {k: float(v) for k, v in normal.items()}
        except (KeyError, TypeError):
            pass

        cache[tcg_id] = {
            "uuid": uuid,
            "name": card.get("name", ""),
            "rarity": card.get("rarity", ""),
            "setCode": card.get("setCode", ""),
            "printings": card.get("printings", []),
            "legalities": {
                k: v.lower() for k, v in card.get("legalities", {}).items()
            },
            "price_history": price_history,
        }
    return cache


def load_inventory_cache(data_dir: str) -> dict:
    path = os.path.join(data_dir, CACHE_FILENAME)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def sync(history_dir: str, data_dir: str) -> None:
    """Download MTGJson data and build inventory_cards.json."""
    os.makedirs(data_dir, exist_ok=True)

    identifiers_path = os.path.join(data_dir, "AllIdentifiers.json")
    prices_path = os.path.join(data_dir, "AllPrices.json")

    download_json(f"{MTGJSON_BASE}/AllIdentifiers.json", identifiers_path)
    download_json(f"{MTGJSON_BASE}/AllPrices.json", prices_path)

    print("Building inventory cache...")
    inventory_ids = read_inventory_ids(history_dir)
    if not inventory_ids:
        print("No inventory IDs found in latest.csv — run import first.")
        sys.exit(1)

    identifiers_data = load_json_file(identifiers_path)["data"]
    prices_data = load_json_file(prices_path)["data"]

    cache = build_inventory_cache(inventory_ids, identifiers_data, prices_data)
    cache_path = os.path.join(data_dir, CACHE_FILENAME)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"✅ Cached {len(cache)} cards → {cache_path}")
    skipped = len(inventory_ids) - len(cache)
    if skipped:
        print(f"   {skipped} inventory IDs had no MTGJson match (sealed product, etc.)")
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_mtgjson.py -v
```
Expected: all 5 tests PASS

**Step 5: Commit**

```bash
git add lib/mtgjson.py tests/test_mtgjson.py
git commit -m "feat: add MTGJson sync and inventory cache builder"
```

---

### Task 6: Prediction orchestration (`lib/predict.py`)

**Files:**
- Create: `lib/predict.py`
- Create: `tests/test_predict.py`

**Step 1: Write the failing tests**

```python
# tests/test_predict.py
import csv
import json
import pytest
from pathlib import Path
from lib.predict import run_predict

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def setup_dirs(tmp_path):
    """Create directory layout and minimal fixture files."""
    history_dir = tmp_path / "history"
    data_dir = tmp_path / "data" / "mtgjson"
    models_dir = tmp_path / "models"
    output_dir = tmp_path / "output"
    for d in [history_dir, data_dir, models_dir, output_dir]:
        d.mkdir(parents=True)

    # Copy fixture inventory cache
    src = FIXTURES / "inventory_cards.json"
    (data_dir / "inventory_cards.json").write_text(src.read_text())

    # Minimal latest.csv
    latest_csv = history_dir / "latest.csv"
    latest_csv.write_text(
        "TCGplayer Id,Product Name,TCG Market Price,TCG Marketplace Price,Total Quantity\n"
        "111111,Test Rare,1.91,2.00,3\n"
        "222222,Test Common,0.06,0.05,1\n"
        "333333,Not In MTGJson,0.50,0.50,2\n"
    )
    return {
        "history_dir": str(history_dir),
        "data_dir": str(data_dir),
        "models_dir": str(models_dir),
        "output_dir": str(output_dir),
    }


def test_run_predict_creates_predictions_csv(setup_dirs):
    run_predict(**setup_dirs)
    output = Path(setup_dirs["output_dir"])
    predictions_files = list(output.glob("predictions-*.csv"))
    assert len(predictions_files) == 1


def test_run_predict_creates_watchlist_csv(setup_dirs):
    run_predict(**setup_dirs)
    output = Path(setup_dirs["output_dir"])
    watchlist_files = list(output.glob("watchlist-*.csv"))
    assert len(watchlist_files) == 1


def test_predictions_csv_has_required_columns(setup_dirs):
    run_predict(**setup_dirs)
    output = Path(setup_dirs["output_dir"])
    pred_file = list(output.glob("predictions-*.csv"))[0]
    with open(pred_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    expected_cols = {
        "TCGplayer Id", "Product Name", "Current Price", "Market Price",
        "Suggested Price", "Action", "Reason", "Margin",
        "Predicted 7d", "Predicted 30d", "Trend", "Spike Probability", "Signal",
    }
    assert expected_cols.issubset(set(reader.fieldnames))


def test_unmatched_card_still_appears_in_output(setup_dirs):
    run_predict(**setup_dirs)
    output = Path(setup_dirs["output_dir"])
    pred_file = list(output.glob("predictions-*.csv"))[0]
    with open(pred_file) as f:
        rows = list(csv.DictReader(f))
    ids = {r["TCGplayer Id"] for r in rows}
    # 333333 has no MTGJson match — it should still get standard pricing
    assert "333333" in ids


def test_watchlist_only_contains_high_spike_probability(setup_dirs):
    run_predict(**setup_dirs)
    output = Path(setup_dirs["output_dir"])
    watchlist_file = list(output.glob("watchlist-*.csv"))[0]
    with open(watchlist_file) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        assert float(row["Spike Probability"]) >= 0.6
```

**Step 2: Run to verify they fail**

```bash
pytest tests/test_predict.py -v
```
Expected: `ModuleNotFoundError: No module named 'lib.predict'`

**Step 3: Implement `lib/predict.py`**

```python
import csv
import json
import os
from datetime import datetime

from lib.features import extract_features, generate_training_data
from lib.forecast import forecast_card, trend_direction
from lib.mtgjson import load_inventory_cache
from lib.spike import FEATURE_COLS, score, train

SPIKE_HOLD_THRESHOLD = 0.6
COMMISSION_FEE = 0.1075
TRANSACTION_FEE = 0.025
TRANSACTION_FLAT = 0.30
SHIPPING_REVENUE = 1.31


def _calc_margin(market: float) -> float:
    revenue = market + SHIPPING_REVENUE if market < 5 else market
    fees = revenue * (COMMISSION_FEE + TRANSACTION_FEE) + TRANSACTION_FLAT
    postage = 0.73 if market < 5 else 1.50
    return round(revenue - fees - postage, 2)


def _pricing_action(market: float, current: float, net: float):
    if net < 0.10:
        return "RAISE", f"Low margin (${net:.2f})"
    if market > current * 1.1:
        return "RAISE", "Market up 10%+, current underpriced"
    if market < current * 0.9:
        return "LOWER", "Market down 10%+, overpriced"
    if market >= 5 and current < market * 0.95:
        return "RAISE", "Competitive adjustment for high-value"
    return "", ""


def run_predict(history_dir: str, data_dir: str, models_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Load inventory CSV
    latest_path = os.path.join(history_dir, "latest.csv")
    with open(latest_path, newline="") as f:
        inventory = {row["TCGplayer Id"]: row for row in csv.DictReader(f)}

    # Load MTGJson cache
    cache = load_inventory_cache(data_dir)
    if not cache:
        print("⚠️  No MTGJson cache found. Run 'monitor.sh sync' first. Falling back to report mode.")
        _write_basic_report(inventory, output_dir)
        return

    # Ensure model exists — auto-train if missing
    model_path = os.path.join(models_dir, "spike_classifier.json")
    if not os.path.exists(model_path):
        print("Model not found — training locally (CPU)...")
        rows = generate_training_data(cache)
        if rows:
            train(rows, model_path, device="cpu")
        else:
            print("⚠️  Insufficient history for training. Spike scores will be 0.")
            model_path = None

    # Score spike probability for all cached cards
    spike_scores = {}
    if model_path and os.path.exists(model_path):
        features_list = [
            extract_features(tid, card) for tid, card in cache.items()
        ]
        scores = score(features_list, model_path)
        spike_scores = {f["tcgplayer_id"]: s for f, s in zip(features_list, scores)}

    # Build output rows
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    predictions_path = os.path.join(output_dir, f"predictions-{timestamp}.csv")
    watchlist_path = os.path.join(output_dir, f"watchlist-{timestamp}.csv")

    PRED_FIELDS = [
        "TCGplayer Id", "Product Name", "Current Price", "Market Price",
        "Suggested Price", "Action", "Reason", "Margin",
        "Predicted 7d", "Predicted 30d", "Trend", "Spike Probability", "Signal",
    ]
    WATCH_FIELDS = ["TCGplayer Id", "Product Name", "Current Price", "Spike Probability", "Trend"]

    predictions = []
    watchlist = []

    for tcg_id, item in inventory.items():
        qty = int(item.get("Total Quantity", 0) or 0)
        if qty == 0:
            continue

        market = float(item.get("TCG Market Price") or 0)
        current = float(item.get("TCG Marketplace Price") or 0)
        name = item.get("Product Name", "")
        margin = _calc_margin(market)
        action, reason = _pricing_action(market, current, margin)
        suggested = round(market * 0.98, 2) if action == "RAISE" else round(market, 2)

        # Prediction columns (blank if card not in MTGJson cache)
        card = cache.get(tcg_id)
        pred_7d = pred_30d = trend = spike_prob = signal = ""

        if card:
            ph = card["price_history"]
            p7 = forecast_card(ph, 7)
            p30 = forecast_card(ph, 30)
            pred_7d = f"{p7:.4f}" if p7 is not None else "insufficient_data"
            pred_30d = f"{p30:.4f}" if p30 is not None else "insufficient_data"
            trend = trend_direction(ph)
            sp = spike_scores.get(tcg_id, 0.0)
            spike_prob = f"{sp:.4f}"

            if sp >= SPIKE_HOLD_THRESHOLD:
                signal = "HOLD"
            elif trend == "down" and current <= market * 1.05:
                signal = "SELL_NOW"

            if signal == "HOLD":
                watchlist.append({
                    "TCGplayer Id": tcg_id,
                    "Product Name": name,
                    "Current Price": current,
                    "Spike Probability": spike_prob,
                    "Trend": trend,
                })

        predictions.append({
            "TCGplayer Id": tcg_id,
            "Product Name": name,
            "Current Price": current,
            "Market Price": market,
            "Suggested Price": max(0.01, suggested),
            "Action": action,
            "Reason": reason,
            "Margin": margin,
            "Predicted 7d": pred_7d,
            "Predicted 30d": pred_30d,
            "Trend": trend,
            "Spike Probability": spike_prob,
            "Signal": signal,
        })

    with open(predictions_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PRED_FIELDS)
        writer.writeheader()
        writer.writerows(predictions)

    watchlist.sort(key=lambda x: float(x["Spike Probability"]), reverse=True)
    with open(watchlist_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=WATCH_FIELDS)
        writer.writeheader()
        writer.writerows(watchlist)

    print(f"✅ Predictions: {predictions_path}")
    print(f"✅ Watchlist ({len(watchlist)} cards): {watchlist_path}")


def _write_basic_report(inventory: dict, output_dir: str) -> None:
    """Minimal fallback when no MTGJson cache exists."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(output_dir, f"predictions-{timestamp}.csv")
    fields = ["TCGplayer Id", "Product Name", "Current Price", "Market Price",
              "Suggested Price", "Action", "Reason", "Margin",
              "Predicted 7d", "Predicted 30d", "Trend", "Spike Probability", "Signal"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
    # Also write empty watchlist
    wpath = os.path.join(output_dir, f"watchlist-{timestamp}.csv")
    with open(wpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["TCGplayer Id", "Product Name",
                                               "Current Price", "Spike Probability", "Trend"])
        writer.writeheader()
```

**Step 4: Run all tests**

```bash
pytest tests/ -v
```
Expected: all tests PASS

**Step 5: Commit**

```bash
git add lib/predict.py tests/test_predict.py
git commit -m "feat: add prediction orchestration pipeline"
```

---

### Task 7: Remote training script (`scripts/train_remote.py`)

This script runs **on the PC** (no tests — it's a thin CLI wrapper around `lib/spike.train`).

**Files:**
- Create: `scripts/train_remote.py`

**Step 1: Create `scripts/train_remote.py`**

```python
#!/usr/bin/env python3
"""
Remote training script — runs on the PC with NVIDIA GPU.
Invoked by monitor.sh via SSH.

Usage:
    python3 train_remote.py --features /tmp/tcgplayer/features.json \
                            --output /tmp/tcgplayer/spike_classifier.json
"""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to features JSON")
    parser.add_argument("--output", required=True, help="Path to save trained model")
    args = parser.parse_args()

    try:
        import xgboost  # noqa: F401
    except ImportError:
        print("ERROR: xgboost not installed. Run: pip install xgboost")
        sys.exit(1)

    with open(args.features) as f:
        rows = json.load(f)

    if not rows:
        print("ERROR: features file is empty")
        sys.exit(1)

    # Import spike locally — this script is rsynced to the PC alongside lib/
    from lib.spike import train
    print(f"Training on {len(rows)} examples with device=cuda ...")
    train(rows, args.output, device="cuda")
    print(f"✅ Model saved to {args.output}")


if __name__ == "__main__":
    main()
```

**Step 2: Make executable**

```bash
chmod +x scripts/train_remote.py
```

**Step 3: Commit**

```bash
git add scripts/train_remote.py
git commit -m "feat: add remote GPU training script for PC"
```

---

### Task 8: Wire up `monitor.sh`

**Files:**
- Modify: `scripts/monitor.sh`

**Step 1: Add path constants and new commands to the usage block**

At the top of `monitor.sh`, after the existing path definitions, add:

```bash
DATA_DIR="${PRICING_DIR}/data/mtgjson"
MODELS_DIR="${PRICING_DIR}/models"
LIB_DIR="${PRICING_DIR}/lib"
```

Update the usage block comments at the top of the file:

```bash
#   monitor.sh sync                        # Download MTGJson data
#   monitor.sh train [--remote <host>]     # Train spike classifier
#   monitor.sh predict                     # Run predictions + recommendations
```

**Step 2: Add `sync_data()` function**

```bash
sync_data() {
    python3 - <<PYEOF
import sys
sys.path.insert(0, "${PRICING_DIR}")
from lib.mtgjson import sync
sync("${HISTORY_DIR}", "${DATA_DIR}")
PYEOF
}
```

**Step 3: Add `train_model()` function**

```bash
train_model() {
    local remote_host="${1:-}"

    if [[ -z "$remote_host" ]]; then
        echo "Training locally (CPU)..."
        python3 - <<PYEOF
import sys, json
sys.path.insert(0, "${PRICING_DIR}")
from lib.mtgjson import load_inventory_cache
from lib.features import generate_training_data
from lib.spike import train

cache = load_inventory_cache("${DATA_DIR}")
if not cache:
    print("No MTGJson cache. Run 'monitor.sh sync' first.")
    sys.exit(1)

rows = generate_training_data(cache)
if not rows:
    print("Insufficient price history for training.")
    sys.exit(1)

train(rows, "${MODELS_DIR}/spike_classifier.json", device="cpu")
PYEOF
    else
        echo "Training remotely on ${remote_host} (GPU)..."
        REMOTE_TMP="/tmp/tcgplayer_train"

        # Extract features locally
        python3 - <<PYEOF
import sys, json
sys.path.insert(0, "${PRICING_DIR}")
from lib.mtgjson import load_inventory_cache
from lib.features import generate_training_data
import os

cache = load_inventory_cache("${DATA_DIR}")
rows = generate_training_data(cache)
os.makedirs("/tmp/tcgplayer_train", exist_ok=True)
with open("/tmp/tcgplayer_train/features.json", "w") as f:
    json.dump(rows, f)
print(f"Extracted {len(rows)} training rows")
PYEOF

        # Sync lib + script + features to remote
        ssh "${remote_host}" "mkdir -p ${REMOTE_TMP}/lib"
        rsync -az "${PRICING_DIR}/lib/" "${remote_host}:${REMOTE_TMP}/lib/"
        rsync -az "${PRICING_DIR}/scripts/train_remote.py" "${remote_host}:${REMOTE_TMP}/"
        rsync -az "/tmp/tcgplayer_train/features.json" "${remote_host}:${REMOTE_TMP}/"

        # Train remotely
        ssh "${remote_host}" "cd ${REMOTE_TMP} && python3 train_remote.py \
            --features ${REMOTE_TMP}/features.json \
            --output ${REMOTE_TMP}/spike_classifier.json"

        if [[ $? -ne 0 ]]; then
            echo "⚠️  Remote training failed. Falling back to local CPU..."
            train_model
            return
        fi

        # Retrieve model
        mkdir -p "${MODELS_DIR}"
        rsync -az "${remote_host}:${REMOTE_TMP}/spike_classifier.json" "${MODELS_DIR}/"
        echo "✅ Model retrieved from ${remote_host}"
    fi
}
```

**Step 4: Add `run_predict()` function**

```bash
run_predict() {
    python3 - <<PYEOF
import sys
sys.path.insert(0, "${PRICING_DIR}")
from lib.predict import run_predict
run_predict(
    history_dir="${HISTORY_DIR}",
    data_dir="${DATA_DIR}",
    models_dir="${MODELS_DIR}",
    output_dir="${OUTPUT_DIR}",
)
PYEOF
}
```

**Step 5: Add new cases to the `case` statement**

```bash
    sync)
        sync_data
        ;;
    train)
        if [[ "${2:-}" == "--remote" ]]; then
            train_model "${3:-}"
        else
            train_model
        fi
        ;;
    predict)
        run_predict
        ;;
```

**Step 6: Manual smoke test**

```bash
# Verify sync command is wired (will actually download — only run when ready)
bash scripts/monitor.sh
# Should show updated usage including sync/train/predict
```

**Step 7: Run full test suite one final time**

```bash
pytest tests/ -v
```
Expected: all tests PASS

**Step 8: Commit**

```bash
git add scripts/monitor.sh
git commit -m "feat: wire sync, train, and predict commands into monitor.sh"
```

---

### Task 9: Push and verify

**Step 1: Push to GitHub**

```bash
git push origin main
```

**Step 2: Confirm on PC (one-time setup)**

On the PC (via SSH or directly):
```bash
pip install xgboost pandas numpy
# Verify CUDA is available:
python3 -c "import xgboost as xgb; print(xgb.__version__)"
```

**Step 3: End-to-end walkthrough**

```bash
# On Pi:
bash scripts/monitor.sh sync
bash scripts/monitor.sh train --remote <tailscale-hostname>
bash scripts/monitor.sh predict
# Check output/predictions-*.csv and output/watchlist-*.csv
```
