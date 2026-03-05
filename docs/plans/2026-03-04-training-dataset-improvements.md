# Training Dataset Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve spike classifier training by expanding the dataset from ~589 inventory cards to the full MTGJson universe, adding external data features, filtering out noise via price floors, and weighting samples by economic relevance.

**Architecture:** Four independent improvements to the training pipeline, each backward-compatible. Changes touch `lib/config.py` (new constants), `lib/mtgjson.py` (full-universe cache builder), `lib/features.py` (price floor + external features), `lib/spike.py` (sample weights), and corresponding tests. The existing `inventory_cards.json` remains for scoring/prediction; a new `training_cards.json` cache provides the expanded training set.

**Tech Stack:** Python, XGBoost (sample_weight parameter), MTGJson bulk data (AllIdentifiers.json, AllPrices.json), EDHREC popularity data (via AllIdentifiers.json fields already downloaded).

---

## Current State

- **Training data:** `generate_training_data(cache)` takes `inventory_cards.json` (~589 cards from user's inventory). Cards with 31+ days of price history produce sliding-window rows. The fixture `inventory_cards.json` has 2 cards (111111 with 90 days, 222222 with 5 days).
- **Spike label:** `int(window[0] > 0 and (max(window[1:]) - window[0]) / window[0] > 0.20)` — any >20% increase in a 30-day window, regardless of starting price.
- **Training:** XGBoost with `scale_pos_weight` for class imbalance, but no per-sample weights.
- **External data:** EDHREC rank and saltiness are already in the cache from AllIdentifiers.json. No other external sources.
- **Test suite:** 80 tests across 6 files. Key fixtures in `tests/fixtures/inventory_cards.json`.

## Improvement Summary

| # | Improvement | What Changes |
|---|---|---|
| 1 | Price floor on spike labels | `lib/config.py`, `lib/features.py`, `tests/test_features.py` |
| 2 | Sample weighting by card value | `lib/spike.py`, `lib/config.py`, `tests/test_spike.py` |
| 3 | Train on all MTGJson cards | `lib/mtgjson.py`, `lib/config.py`, `lib/predict.py`, `scripts/monitor.sh`, `tests/test_mtgjson.py` |
| 4 | External data features | `lib/features.py`, `lib/spike.py`, `lib/config.py`, `tests/test_features.py`, `tests/test_spike.py`, `tests/fixtures/inventory_cards.json` |

---

### Task 1: Price Floor on Spike Labels

Spike labels on sub-$0.25 cards are noise. A card going from $0.03 to $0.04 (+33%) teaches the model patterns that aren't economically actionable.

**Files:**
- Modify: `lib/config.py` (add constant)
- Modify: `lib/features.py:148-152` (spike label logic in `generate_training_data`)
- Modify: `tests/test_features.py` (add tests, update existing)

**Step 1: Add config constant**

In `lib/config.py`, add under the Feature Extraction section:

```python
SPIKE_MIN_PRICE = 0.25            # Ignore spikes below this starting price
```

**Step 2: Write failing tests**

Add to `tests/test_features.py`:

```python
def test_spike_label_ignores_cheap_cards():
    """Cards below SPIKE_MIN_PRICE should never be labeled as spikes."""
    from lib.config import SPIKE_MIN_PRICE
    card = {
        "rarity": "common", "printings": ["A"], "legalities": {},
        "price_history": {f"2026-01-{i:02d}": 0.05 * (1.5 if i > 15 else 1.0) for i in range(1, 32)},
        "foil_price_history": {}, "buylist_price_history": {},
        "subtypes": [],
    }
    # Price goes from $0.05 to $0.075 (50% increase) — but below SPIKE_MIN_PRICE
    rows = generate_training_data({"1": card})
    assert all(r["spike"] == 0 for r in rows), "Cheap card spikes should be filtered out"


def test_spike_label_works_above_floor():
    """Cards above SPIKE_MIN_PRICE should still get spike labels normally."""
    card = {
        "rarity": "rare", "printings": ["A"], "legalities": {},
        "price_history": {f"2026-01-{i:02d}": (1.0 if i <= 15 else 1.5) for i in range(1, 32)},
        "foil_price_history": {}, "buylist_price_history": {},
        "subtypes": [],
    }
    # Price goes from $1.00 to $1.50 (50% increase) — above floor
    rows = generate_training_data({"1": card})
    assert any(r["spike"] == 1 for r in rows), "Above-floor spikes should be labeled"
```

**Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_features.py::test_spike_label_ignores_cheap_cards tests/test_features.py::test_spike_label_works_above_floor -v`
Expected: `test_spike_label_ignores_cheap_cards` FAILS (cheap cards currently get spike=1)

**Step 4: Implement the price floor**

In `lib/features.py`, update the import to include `SPIKE_MIN_PRICE`:

```python
from lib.config import (
    RARITY_RANK, SPIKE_THRESHOLD, SPIKE_MIN_PRICE, MIN_PRICE,
    SPOILER_WINDOW_DAYS, RELEASE_PROXIMITY_MAX, get_logger,
)
```

In `generate_training_data()`, change the spike label logic (line ~150):

```python
            spike = int(
                window[0] >= SPIKE_MIN_PRICE
                and (max(window[1:]) - window[0]) / window[0] > SPIKE_THRESHOLD
            )
```

Note: changed `window[0] > 0` to `window[0] >= SPIKE_MIN_PRICE`. This is the only line that changes.

**Step 5: Run all tests**

Run: `python -m pytest tests/test_features.py -v`
Expected: All pass (including existing `test_generate_training_data_labels_spike` — fixture card 111111 starts at $1.00, well above $0.25)

**Step 6: Commit**

```bash
git add lib/config.py lib/features.py tests/test_features.py
git commit -m "feat: add $0.25 price floor for spike labels

Cards below SPIKE_MIN_PRICE no longer labeled as spikes during training.
Prevents model from learning noise patterns on penny cards."
```

---

### Task 2: Sample Weighting by Card Value

Cards worth $5+ matter more to the user's bottom line. Weight training samples by `sqrt(current_price)` so the model prioritizes accuracy on economically relevant cards without completely ignoring cheap ones.

**Files:**
- Modify: `lib/config.py` (add constant)
- Modify: `lib/spike.py:52-106` (`train()` function)
- Modify: `tests/test_spike.py` (add tests)

**Step 1: Add config constant**

In `lib/config.py`, add under Spike Classifier section:

```python
SAMPLE_WEIGHT_FEATURE = "current_price"  # Feature used for sample weighting
```

**Step 2: Write failing tests**

Add to `tests/test_spike.py`:

```python
def test_train_records_sample_weighting(cards, tmp_path):
    model_path = str(tmp_path / "model.json")
    rows = generate_training_data(cards)
    train(rows, model_path, device="cpu")
    meta = load_model_meta(model_path)
    assert "sample_weighting" in meta
    assert meta["sample_weighting"] == "sqrt_current_price"


def test_train_sample_weights_affect_importance(tmp_path):
    """Higher-priced samples should influence the model more."""
    # Create two sets of rows: one with weights, verify meta records it
    from lib.spike import train
    rows = [
        {col: 0.0 for col in FEATURE_COLS}
        | {"spike": 0, "current_price": 10.0}
        for _ in range(20)
    ] + [
        {col: 0.0 for col in FEATURE_COLS}
        | {"spike": 1, "current_price": 10.0}
        for _ in range(5)
    ]
    model_path = str(tmp_path / "model.json")
    train(rows, model_path, device="cpu")
    meta = load_model_meta(model_path)
    assert meta["sample_weighting"] == "sqrt_current_price"
```

**Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_spike.py::test_train_records_sample_weighting tests/test_spike.py::test_train_sample_weights_affect_importance -v`
Expected: FAIL with `KeyError: 'sample_weighting'`

**Step 4: Implement sample weighting**

In `lib/spike.py`, update the import:

```python
from lib.config import (
    N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, VALIDATION_SPLIT, RANDOM_SEED,
    SAMPLE_WEIGHT_FEATURE, get_logger,
)
```

In the `train()` function, after `y = df["spike"]` (line ~59), add sample weight computation:

```python
    # Sample weights: sqrt(price) so high-value cards matter more
    raw_weights = df[SAMPLE_WEIGHT_FEATURE].clip(lower=0.01)
    sample_weights = np.sqrt(raw_weights)
```

Then pass `sample_weight=` to both `.fit()` calls. For the validation model (line ~84):

```python
        val_model.fit(X_train, y_train, sample_weight=sample_weights.iloc[train_idx])
```

For the final model (line ~106):

```python
    model.fit(X, y, sample_weight=sample_weights)
```

In the metadata dict (line ~114), add:

```python
        "sample_weighting": "sqrt_current_price",
```

**Step 5: Run all tests**

Run: `python -m pytest tests/test_spike.py -v`
Expected: All pass

**Step 6: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All 80+ tests pass

**Step 7: Commit**

```bash
git add lib/config.py lib/spike.py tests/test_spike.py
git commit -m "feat: add sqrt(price) sample weighting to training

Higher-value cards now have more influence on model training via
sample_weight parameter. Records weighting method in model metadata."
```

---

### Task 3: Train on All MTGJson Cards (Full Universe)

Currently training on ~589 inventory cards. AllPrices.json contains price history for ~70,000+ cards with TCGPlayer retail data. Building a separate `training_cards.json` cache with all cards provides a massively larger and more diverse training set.

**Files:**
- Modify: `lib/config.py` (add constant)
- Modify: `lib/mtgjson.py` (add `build_training_cache()` function, modify `sync()`)
- Modify: `lib/predict.py:66-72` (training data source)
- Modify: `scripts/monitor.sh` (train command uses training cache)
- Create: `tests/test_training_cache.py` (new test file)

**Step 1: Add config constants**

In `lib/config.py`, add under Feature Extraction section:

```python
TRAINING_CACHE_FILENAME = "training_cards.json"  # Full-universe training cache
TRAINING_CACHE_MAX_CARDS = 0       # 0 = no limit; set to e.g. 5000 for faster builds
```

**Step 2: Write failing tests**

Create `tests/test_training_cache.py`:

```python
# tests/test_training_cache.py
"""Tests for full-universe training cache builder."""
import json
import pytest
from lib.mtgjson import build_training_cache, load_training_cache


@pytest.fixture
def sample_identifiers():
    """Minimal AllIdentifiers-like data."""
    return {
        "uuid-001": {
            "name": "Lightning Bolt",
            "rarity": "common",
            "setCode": "M10",
            "printings": ["M10", "2ED"],
            "legalities": {"standard": "Legal", "modern": "Legal"},
            "edhrecRank": 5,
            "edhrecSaltiness": 0.3,
            "isReserved": False,
            "supertypes": [],
            "types": ["Instant"],
            "subtypes": [],
            "colorIdentity": ["R"],
            "keywords": [],
            "manaValue": 1.0,
            "text": "Deal 3 damage.",
        },
        "uuid-002": {
            "name": "Black Lotus",
            "rarity": "rare",
            "setCode": "LEA",
            "printings": ["LEA"],
            "legalities": {"vintage": "Restricted"},
            "edhrecRank": None,
            "edhrecSaltiness": None,
            "isReserved": True,
            "supertypes": [],
            "types": ["Artifact"],
            "subtypes": [],
            "colorIdentity": [],
            "keywords": [],
            "manaValue": 0.0,
            "text": "Add three mana.",
        },
        "uuid-003": {
            "name": "No Prices Card",
            "rarity": "common",
            "setCode": "TST",
            "printings": ["TST"],
            "legalities": {},
            "isReserved": False,
            "supertypes": [],
            "types": ["Creature"],
            "subtypes": [],
            "colorIdentity": [],
            "keywords": [],
            "manaValue": 2.0,
            "text": "",
        },
    }


@pytest.fixture
def sample_prices():
    """Minimal AllPrices-like data."""
    return {
        "uuid-001": {
            "paper": {
                "tcgplayer": {
                    "retail": {
                        "normal": {f"2026-01-{i:02d}": 1.0 + i * 0.01 for i in range(1, 32)},
                        "foil": {f"2026-01-{i:02d}": 3.0 for i in range(1, 11)},
                    },
                    "buylist": {
                        "normal": {f"2026-01-{i:02d}": 0.5 for i in range(1, 11)},
                    },
                }
            }
        },
        "uuid-002": {
            "paper": {
                "tcgplayer": {
                    "retail": {
                        "normal": {f"2026-01-{i:02d}": 5000.0 for i in range(1, 32)},
                    },
                }
            }
        },
        # uuid-003 has no tcgplayer prices
    }


@pytest.fixture
def sample_set_data():
    return {"M10": {"releaseDate": "2009-07-17", "isPartialPreview": False}}


def test_build_training_cache_includes_all_priced_cards(
    sample_identifiers, sample_prices, sample_set_data
):
    cache = build_training_cache(sample_identifiers, sample_prices, sample_set_data)
    # uuid-001 and uuid-002 have prices, uuid-003 does not
    assert len(cache) == 2
    assert "uuid-001" in cache
    assert "uuid-002" in cache
    assert "uuid-003" not in cache


def test_build_training_cache_card_structure(
    sample_identifiers, sample_prices, sample_set_data
):
    cache = build_training_cache(sample_identifiers, sample_prices, sample_set_data)
    card = cache["uuid-001"]
    assert card["name"] == "Lightning Bolt"
    assert card["rarity"] == "common"
    assert "price_history" in card
    assert len(card["price_history"]) == 31
    assert "foil_price_history" in card
    assert "buylist_price_history" in card
    assert "setReleaseDate" in card


def test_build_training_cache_skips_no_prices(
    sample_identifiers, sample_prices, sample_set_data
):
    cache = build_training_cache(sample_identifiers, sample_prices, sample_set_data)
    # uuid-003 has no TCGPlayer prices
    uuids = set(cache.keys())
    assert "uuid-003" not in uuids


def test_build_training_cache_respects_max_cards(
    sample_identifiers, sample_prices, sample_set_data
):
    cache = build_training_cache(
        sample_identifiers, sample_prices, sample_set_data, max_cards=1
    )
    assert len(cache) == 1


def test_load_training_cache_returns_empty_when_missing(tmp_path):
    result = load_training_cache(str(tmp_path))
    assert result == {}


def test_load_training_cache_reads_file(tmp_path):
    data = {"uuid-001": {"name": "Test"}}
    (tmp_path / "training_cards.json").write_text(json.dumps(data))
    result = load_training_cache(str(tmp_path))
    assert result == data
```

**Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_training_cache.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_training_cache'`

**Step 4: Implement `build_training_cache` and `load_training_cache`**

In `lib/mtgjson.py`, add the import for the new config constant:

```python
from lib.config import (
    DOWNLOAD_TIMEOUT, DOWNLOAD_MAX_RETRIES, DOWNLOAD_BACKOFF_BASE,
    SPOILER_WINDOW_DAYS, TRAINING_CACHE_FILENAME, get_logger,
)
```

Add these functions after `build_inventory_cache()`:

```python
def build_training_cache(
    identifiers_data: dict,
    prices_data: dict,
    set_data: dict | None = None,
    max_cards: int = 0,
) -> dict:
    """Build training cache from ALL cards with TCGPlayer retail prices.

    Unlike build_inventory_cache (scoped to user's inventory via SKU mapping),
    this iterates all UUIDs in AllPrices.json that have paper/tcgplayer/retail
    data. Keyed by UUID (not TCGPlayer ID) since we don't have SKUs for
    non-inventory cards.

    Args:
        identifiers_data: AllIdentifiers["data"] dict (uuid -> card)
        prices_data: AllPrices["data"] dict (uuid -> price channels)
        set_data: {setCode: {releaseDate, isPartialPreview}} from SetList.json
        max_cards: limit output size (0 = no limit, for testing/debugging)
    """
    if set_data is None:
        set_data = {}
    cache = {}
    count = 0

    for uuid, price_channels in prices_data.items():
        # Only include cards with TCGPlayer retail normal prices
        try:
            normal = price_channels["paper"]["tcgplayer"]["retail"]["normal"]
            price_history = {k: float(v) for k, v in normal.items()}
        except (KeyError, TypeError):
            continue

        if not price_history:
            continue

        card = identifiers_data.get(uuid)
        if not card:
            continue

        foil_price_history = {}
        try:
            foil = price_channels["paper"]["tcgplayer"]["retail"]["foil"]
            foil_price_history = {k: float(v) for k, v in foil.items()}
        except (KeyError, TypeError):
            pass

        buylist_price_history = {}
        try:
            buylist = price_channels["paper"]["tcgplayer"]["buylist"]["normal"]
            buylist_price_history = {k: float(v) for k, v in buylist.items()}
        except (KeyError, TypeError):
            pass

        set_code = card.get("setCode", "").upper()
        cache[uuid] = {
            "uuid": uuid,
            "name": card.get("name", ""),
            "rarity": card.get("rarity", ""),
            "setCode": card.get("setCode", ""),
            "printings": card.get("printings", []),
            "legalities": {
                k: v.lower() for k, v in card.get("legalities", {}).items()
            },
            "price_history": price_history,
            "edhrecRank": card.get("edhrecRank"),
            "edhrecSaltiness": card.get("edhrecSaltiness"),
            "isReserved": card.get("isReserved", False),
            "supertypes": card.get("supertypes", []),
            "types": card.get("types", []),
            "subtypes": card.get("subtypes", []),
            "colorIdentity": card.get("colorIdentity", []),
            "keywords": card.get("keywords", []),
            "manaValue": card.get("manaValue", 0),
            "text": card.get("text", ""),
            "foil_price_history": foil_price_history,
            "buylist_price_history": buylist_price_history,
            "recently_reprinted": 0,
            "legality_changed": 0,
            "setReleaseDate": set_data.get(set_code, {}).get("releaseDate", ""),
            "setIsPartialPreview": set_data.get(set_code, {}).get("isPartialPreview", False),
        }

        count += 1
        if max_cards and count >= max_cards:
            break

    return cache


def load_training_cache(data_dir: str) -> dict:
    """Load the full-universe training cache."""
    path = os.path.join(data_dir, TRAINING_CACHE_FILENAME)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_training_cache.py -v`
Expected: All pass

**Step 6: Integrate into sync**

In `lib/mtgjson.py`, modify the `sync()` function. After the existing cache build (after `detect_changes`), add the training cache build:

```python
    # Build full-universe training cache
    from lib.config import TRAINING_CACHE_MAX_CARDS
    print("Building training cache (all cards with TCGPlayer prices)...")
    training_cache = build_training_cache(
        identifiers_data, prices_data, set_data,
        max_cards=TRAINING_CACHE_MAX_CARDS,
    )
    training_cache_path = os.path.join(data_dir, TRAINING_CACHE_FILENAME)
    with open(training_cache_path, "w") as f:
        json.dump(training_cache, f)
    print(f"Training cache: {len(training_cache)} cards -> {training_cache_path}")
```

**Step 7: Update training to prefer training cache**

In `lib/predict.py`, update the training data generation section (around line 67-72). Replace:

```python
    if need_train:
        print("Training locally (CPU)...")
        rows = generate_training_data(cache)
```

With:

```python
    if need_train:
        print("Training locally (CPU)...")
        from lib.mtgjson import load_training_cache
        training_cache = load_training_cache(data_dir)
        train_source = training_cache if training_cache else cache
        if training_cache:
            print(f"Using full training cache ({len(training_cache)} cards)")
        else:
            print(f"No training cache found, using inventory cache ({len(cache)} cards)")
        rows = generate_training_data(train_source)
```

Also update `scripts/monitor.sh` `train_model()` function similarly. After loading the inventory cache (around line 407), add before `generate_training_data`:

```python
from lib.mtgjson import load_training_cache
training_cache = load_training_cache(DATA_DIR)
if training_cache:
    print(f"Using full training cache ({len(training_cache)} cards)")
    cache = training_cache
```

**Step 8: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass (training cache is optional — inventory cache fallback)

**Step 9: Commit**

```bash
git add lib/config.py lib/mtgjson.py lib/predict.py scripts/monitor.sh tests/test_training_cache.py
git commit -m "feat: train on all MTGJson cards, not just inventory

New build_training_cache() iterates AllPrices.json for all cards with
TCGPlayer retail data (~70k cards). Built during sync alongside the
existing inventory cache. Training prefers training_cards.json when
available, falls back to inventory_cards.json."
```

---

### Task 4: External Data Features — Supply/Demand Signals

MTGJson's AllIdentifiers.json already contains EDHREC rank and saltiness (used today). We can extract additional signals from existing data without new downloads:

- **`price_percentile`** — Where does this card's price sit relative to all cards in its set? A $2 common in a set where most commons are $0.10 may be underpriced.
- **`set_card_count`** — Total cards in the set (proxy for set complexity/dilution).
- **`days_since_last_price_change`** — Stale pricing may precede sudden corrections.
- **`price_range_30d`** — (max - min) / mean over 30 days. Different from volatility (std dev) — captures amplitude of swings.

These 4 features require zero external API calls — they're derived from data we already have.

**Files:**
- Modify: `lib/features.py` (add 4 new features to `extract_features`)
- Modify: `lib/spike.py:18-49` (add 4 entries to `FEATURE_COLS`)
- Modify: `lib/config.py` (no new constants needed)
- Modify: `tests/test_features.py` (update key checks, add tests)
- Modify: `tests/test_spike.py` (update any FEATURE_COLS length checks)
- Modify: `tests/fixtures/inventory_cards.json` (ensure fields exist for tests)

**Step 1: Write failing tests**

Add to `tests/test_features.py`:

```python
def test_extract_features_has_new_derived_features(cards):
    feat = extract_features("111111", cards["111111"])
    assert "price_range_30d" in feat
    assert "days_since_last_price_change" in feat
    assert "set_card_count" in feat
    assert "price_percentile" in feat


def test_price_range_30d_positive_for_volatile_card(cards):
    feat = extract_features("111111", cards["111111"])
    # Card 111111 has steadily rising prices — range should be positive
    assert feat["price_range_30d"] > 0


def test_price_range_30d_zero_for_flat_card():
    card = {
        "rarity": "common", "printings": ["A"], "legalities": {},
        "price_history": {f"2026-01-{i:02d}": 1.0 for i in range(1, 32)},
    }
    feat = extract_features("1", card)
    assert feat["price_range_30d"] == pytest.approx(0.0)


def test_days_since_last_price_change(cards):
    feat = extract_features("111111", cards["111111"])
    # Card 111111 has daily changes, so days_since should be small
    assert feat["days_since_last_price_change"] <= 2


def test_days_since_last_price_change_flat_card():
    card = {
        "rarity": "common", "printings": ["A"], "legalities": {},
        "price_history": {f"2026-01-{i:02d}": 1.0 for i in range(1, 32)},
    }
    feat = extract_features("1", card)
    # All prices identical, days_since = total days of history
    assert feat["days_since_last_price_change"] == 30


def test_set_card_count(cards):
    feat = extract_features("111111", cards["111111"])
    # Card 111111 has setCode TST — set_card_count should come from set_cards context
    assert isinstance(feat["set_card_count"], int)


def test_price_percentile_requires_context():
    """Without set context, price_percentile defaults to 0.5."""
    card = {
        "rarity": "rare", "printings": ["A"], "legalities": {},
        "price_history": {"2026-01-01": 5.0},
    }
    feat = extract_features("1", card)
    assert feat["price_percentile"] == pytest.approx(0.5)
```

Update `test_extract_features_keys` to include the 4 new keys:

```python
def test_extract_features_keys(cards):
    feat = extract_features("111111", cards["111111"])
    expected_keys = {
        "tcgplayer_id", "rarity_rank", "num_printings", "set_age_days",
        "formats_legal_count", "price_momentum_7d", "price_volatility_30d",
        "current_price",
        # Phase 1
        "edhrec_rank", "edhrec_saltiness", "is_reserved_list",
        "is_legendary", "is_creature", "color_count", "keyword_count",
        "mana_value", "subtype_count",
        # Phase 2
        "foil_to_normal_ratio", "buylist_ratio", "buylist_momentum_7d",
        # Phase 3
        "cluster_momentum_7d",
        # Phase 4
        "recently_reprinted", "legality_changed",
        # Phase 5
        "set_release_proximity", "spoiler_season",
        # Phase 6: derived signals
        "price_range_30d", "days_since_last_price_change",
        "set_card_count", "price_percentile",
    }
    assert expected_keys == set(feat.keys())
```

Also update `test_extract_features_empty_price_history` to assert defaults:

```python
    # Phase 6 defaults
    assert feat["price_range_30d"] == 0.0
    assert feat["days_since_last_price_change"] == 0
    assert feat["set_card_count"] == 0
    assert feat["price_percentile"] == pytest.approx(0.5)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_features.py::test_extract_features_has_new_derived_features -v`
Expected: FAIL with `KeyError`

**Step 3: Implement the 4 new features**

In `lib/features.py`, update `extract_features()`. Add these computations after the existing feature calculations, before the return dict:

```python
    # Phase 6: derived price signals
    if len(price_vals) >= 2:
        last_30 = price_vals[-30:]
        price_mean = sum(last_30) / len(last_30)
        price_range_30d = (max(last_30) - min(last_30)) / max(price_mean, MIN_PRICE)
    else:
        price_range_30d = 0.0

    # Days since price last changed
    days_since_last_change = 0
    if len(price_vals) >= 2:
        for j in range(len(price_vals) - 1, 0, -1):
            if abs(price_vals[j] - price_vals[j - 1]) > 1e-6:
                break
            days_since_last_change += 1

    # Set card count — from printings-in-set context passed via card dict
    set_card_count = card.get("set_card_count", 0)

    # Price percentile within set — passed via card dict or defaults to 0.5
    price_percentile = card.get("price_percentile", 0.5)
```

Add these 4 fields to the return dict:

```python
        # Phase 6: derived signals
        "price_range_30d": price_range_30d,
        "days_since_last_price_change": days_since_last_change,
        "set_card_count": set_card_count,
        "price_percentile": price_percentile,
```

**Step 4: Compute set-level context in `generate_training_data`**

In `generate_training_data()`, before the main loop, compute per-set statistics. Add after `rows = []`:

```python
    # Compute per-set context: card counts and price percentiles
    set_cards_map = {}  # setCode -> list of current prices
    for tcgplayer_id, card in cards.items():
        prices = sorted(card.get("price_history", {}).items())
        if prices:
            set_code = card.get("setCode", "")
            set_cards_map.setdefault(set_code, []).append(float(prices[-1][1]))

    set_card_counts = {s: len(ps) for s, ps in set_cards_map.items()}
```

Then in the inner loop, before calling `extract_features`, set the context on the snapshot:

```python
            set_code = card.get("setCode", "")
            snapshot["set_card_count"] = set_card_counts.get(set_code, 0)
            # Price percentile: fraction of set cards with lower price
            set_prices = set_cards_map.get(set_code, [])
            if set_prices and window[0] > 0:
                snapshot["price_percentile"] = sum(
                    1 for p in set_prices if p <= window[0]
                ) / len(set_prices)
            else:
                snapshot["price_percentile"] = 0.5
```

Also update `compute_cluster_features` signature if needed — but since it mutates in-place, no change required.

**Step 5: Update `FEATURE_COLS` in `lib/spike.py`**

Add the 4 new features to the end of `FEATURE_COLS`:

```python
FEATURE_COLS = [
    # ... existing 24 features ...
    # Phase 6: derived signals
    "price_range_30d",
    "days_since_last_price_change",
    "set_card_count",
    "price_percentile",
]
```

**Step 6: Update the fixture**

In `tests/fixtures/inventory_cards.json`, add `set_card_count` to both cards:

For card 111111, add: `"set_card_count": 2`
For card 222222, add: `"set_card_count": 2`

(Both are in set "TST" which has 2 cards in the fixture.)

**Step 7: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All pass. Existing tests that check `FEATURE_COLS` length or model compatibility will need retraining (the trained_model fixture creates fresh models each test run, so version mismatches resolve automatically).

**Step 8: Commit**

```bash
git add lib/features.py lib/spike.py tests/test_features.py tests/test_spike.py tests/fixtures/inventory_cards.json
git commit -m "feat: add 4 derived features (price range, staleness, set context)

New features: price_range_30d, days_since_last_price_change,
set_card_count, price_percentile. All derived from existing data,
no new downloads required. Total features: 24 -> 28."
```

---

## Verification

After all 4 tasks are complete:

1. **Run full test suite:**
   ```bash
   python -m pytest tests/ -v
   ```
   Expected: All tests pass (80 existing + ~15 new)

2. **Run sync to build training cache** (requires MTGJson data on disk):
   ```bash
   bash scripts/monitor.sh sync --cache
   ```
   Expected: Builds both `inventory_cards.json` and `training_cards.json`

3. **Run training:**
   ```bash
   bash scripts/monitor.sh train
   ```
   Expected: Uses training cache, reports sample count (should be much larger than before). Model metadata should include `sample_weighting: sqrt_current_price`.

4. **Run predictions:**
   ```bash
   bash scripts/monitor.sh predict --dry-run
   ```
   Expected: Model auto-retrains (feature mismatch from 24→28 features), then scores inventory cards.

5. **Run backtest:**
   ```bash
   bash scripts/monitor.sh backtest
   ```
   Expected: Metrics should generally improve (larger dataset, better signal-to-noise).

## Task Dependencies

```
Task 1 (price floor) ──────┐
                            ├── Independent, can be done in any order
Task 2 (sample weights) ───┤
                            │
Task 3 (full universe) ────┤   (but run full test suite after each)
                            │
Task 4 (new features) ─────┘   (must update FEATURE_COLS last or alongside)
```

All 4 tasks are independent in code, but Task 4 changes `FEATURE_COLS` which triggers model version incompatibility. If implementing in parallel, do Task 4 last since it changes the feature contract. Tasks 1-3 don't change the feature list.
