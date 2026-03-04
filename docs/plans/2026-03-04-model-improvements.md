# Model Improvements — Design Doc

## Status
Proposed — 2026-03-04

## Overview

Six improvements to the spike classifier, forecasting pipeline, and operational tooling. Grouped by improvement number; cross-cutting concerns noted at the end.

---

## Improvement 1 — Model Validation & Class Imbalance

### Current State

`lib/spike.py:train()` fits XGBoost on the full dataset with no train/validation split. The `_meta.json` records `spike_rate` but no accuracy, AUC, or validation metrics. The model uses default `scale_pos_weight=1`, which means the majority class (non-spikes, typically 95%+ of rows) dominates the loss function. There is no way to know if the model is actually learning or just predicting "no spike" for everything.

### Problem

- No validation split means we cannot detect overfitting.
- No recorded metrics means we cannot compare model quality across retrains.
- Class imbalance (spike_rate often < 5%) biases the classifier toward always predicting non-spike, reducing recall for the spikes we actually care about.

### Code Changes

#### `lib/config.py`

Add new constants:

```python
# ---------------------------------------------------------------------------
# Model Validation
# ---------------------------------------------------------------------------
VALIDATION_SPLIT = 0.2            # Fraction held out for validation
RANDOM_SEED = 42                  # Reproducible splits
```

#### `lib/spike.py`

Replace the current `train()` function body. The function signature stays the same.

```python
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

from lib.config import (
    N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
    VALIDATION_SPLIT, RANDOM_SEED, get_logger,
)
```

Inside `train()`:

1. Build DataFrame as before.
2. Compute `scale_pos_weight`:
   ```python
   neg_count = int((y == 0).sum())
   pos_count = int((y == 1).sum())
   spw = neg_count / max(pos_count, 1)
   ```
3. Create stratified split:
   ```python
   splitter = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
   train_idx, val_idx = next(splitter.split(X, y))
   X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
   y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
   ```
4. Pass `scale_pos_weight=spw` to `XGBClassifier`.
5. Fit on `X_train, y_train`.
6. Evaluate on `X_val, y_val`:
   ```python
   y_pred = model.predict(X_val)
   y_proba = model.predict_proba(X_val)[:, 1]
   val_accuracy = float(accuracy_score(y_val, y_pred))
   val_auc = float(roc_auc_score(y_val, y_proba)) if pos_count > 0 else None
   val_precision = float(precision_score(y_val, y_pred, zero_division=0))
   val_recall = float(recall_score(y_val, y_pred, zero_division=0))
   ```
7. **Re-fit on full data** for the saved model (we want validation metrics for monitoring but the production model should use all available data):
   ```python
   model_full = xgb.XGBClassifier(
       device=device, n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
       learning_rate=LEARNING_RATE, scale_pos_weight=spw,
       eval_metric="logloss", verbosity=0,
   )
   model_full.fit(X, y)
   model_full.save_model(model_path)
   ```
8. Write expanded `_meta.json`:
   ```python
   meta = {
       "trained_at": datetime.now().isoformat(),
       "num_samples": len(rows),
       "num_train": len(train_idx),
       "num_val": len(val_idx),
       "device": device,
       "hyperparameters": {
           "n_estimators": N_ESTIMATORS,
           "max_depth": MAX_DEPTH,
           "learning_rate": LEARNING_RATE,
           "scale_pos_weight": round(spw, 4),
       },
       "spike_rate": float(y.mean()),
       "validation": {
           "accuracy": val_accuracy,
           "auc": val_auc,
           "precision": val_precision,
           "recall": val_recall,
       },
   }
   ```
9. Log validation metrics:
   ```python
   log.info(
       "Validation: accuracy=%.3f, AUC=%.3f, precision=%.3f, recall=%.3f",
       val_accuracy, val_auc or 0, val_precision, val_recall,
   )
   ```

#### `scripts/train_remote.py`

No changes needed — it calls `lib/spike.train()` which handles everything internally.

### Test Cases

Add to `tests/test_spike.py`:

1. **`test_train_meta_contains_validation_metrics`** — Train on fixture data, load meta, assert `meta["validation"]["accuracy"]` is a float between 0 and 1, assert `meta["validation"]["auc"]` is a float or None, assert `meta["validation"]["precision"]` and `meta["validation"]["recall"]` exist.

2. **`test_train_meta_contains_scale_pos_weight`** — Assert `meta["hyperparameters"]["scale_pos_weight"]` exists and is > 1.0 (since spikes are the minority class).

3. **`test_train_meta_contains_split_counts`** — Assert `meta["num_train"] + meta["num_val"] == meta["num_samples"]`.

4. **`test_train_stratified_split_preserves_spike_ratio`** — With a known spike rate, assert that both train and val splits have approximately the same spike rate (within tolerance). This can be verified indirectly through the meta or by patching/inspecting the split.

### Backward Compatibility

- `_meta.json` gains new keys (`validation`, `num_train`, `num_val`, updated `hyperparameters`). Old meta files without these keys still load fine via `load_model_meta()`.
- The saved model format (XGBoost `.json`) is unchanged. Old models still load in `score()`.
- `scale_pos_weight` changes model behavior — **retrain required** after deploying this change. Old models without `scale_pos_weight` remain valid but may have worse recall.

---

## Improvement 2 — Feature Importance Tracking

### Current State

`_meta.json` records hyperparameters and spike rate but nothing about which features the model relies on. After training, there is no way to understand which of the 22 features are most predictive without loading the model in a separate notebook.

### Problem

- Cannot detect feature drift or identify useless features without manual inspection.
- Cannot compare feature importance across model versions.
- No visibility into whether newly added features (e.g., from the signal expansion plan) actually contribute.

### Code Changes

#### `lib/spike.py`

In `train()`, after fitting `model_full` (the final model trained on all data, from Improvement 1), extract importances:

```python
importance_dict = {}
for feat_name, imp_val in zip(FEATURE_COLS, model_full.feature_importances_):
    importance_dict[feat_name] = round(float(imp_val), 6)

# Sort descending for readability
importance_sorted = dict(sorted(importance_dict.items(), key=lambda x: -x[1]))
```

Add to the meta dict:

```python
meta["feature_importances"] = importance_sorted
```

This uses XGBoost's default `importance_type="weight"` (number of times a feature is used in splits). This is the most informative default for tree-based models with mixed feature types.

#### Log output

Add a summary log line:

```python
top_3 = list(importance_sorted.items())[:3]
log.info("Top features: %s", ", ".join(f"{k}={v:.4f}" for k, v in top_3))
```

### Test Cases

Add to `tests/test_spike.py`:

1. **`test_train_meta_contains_feature_importances`** — Train on fixture data, load meta, assert `meta["feature_importances"]` is a dict with exactly `len(FEATURE_COLS)` entries.

2. **`test_feature_importances_keys_match_feature_cols`** — Assert `set(meta["feature_importances"].keys()) == set(FEATURE_COLS)`.

3. **`test_feature_importances_values_are_nonnegative`** — Assert all values in the importance dict are >= 0.0.

### Backward Compatibility

- Additive change to `_meta.json`. Old meta files without `feature_importances` are still valid — `load_model_meta()` returns whatever JSON is in the file.

---

## Improvement 3 — Model Versioning

### Current State

`FEATURE_COLS` is defined at the top of `lib/spike.py` and used both in `train()` and `score()`. If the feature list changes between training and scoring (e.g., a new feature is added but an old model is still on disk), `score()` will either crash (if the model expects columns that don't exist) or silently produce wrong predictions (if the model was trained on fewer features and extra columns are ignored by XGBoost). There is no version check.

### Problem

- Adding features requires retraining, but nothing enforces this — stale models silently produce bad predictions.
- Remote-trained models may have been trained with a different version of the code than the scoring machine.
- No way to verify model compatibility without trial-and-error.

### Code Changes

#### `lib/spike.py`

**In `train()`**, record the feature list in meta:

```python
meta["feature_cols"] = FEATURE_COLS.copy()
```

This goes in the same meta dict built in `train()` (alongside the validation metrics from Improvement 1 and the importances from Improvement 2).

**In `score()`**, add compatibility check before prediction:

```python
def score(features: list[dict], model_path: str) -> list[float]:
    """Return spike probability (0-1) for each feature dict."""
    # Validate model compatibility
    meta = load_model_meta(model_path)
    if meta and "feature_cols" in meta:
        model_features = meta["feature_cols"]
        if model_features != FEATURE_COLS:
            missing_in_model = set(FEATURE_COLS) - set(model_features)
            missing_in_code = set(model_features) - set(FEATURE_COLS)
            msg_parts = []
            if missing_in_model:
                msg_parts.append(f"features in code but not model: {missing_in_model}")
            if missing_in_code:
                msg_parts.append(f"features in model but not code: {missing_in_code}")
            raise ValueError(
                f"Model/code feature mismatch — retrain required. {'; '.join(msg_parts)}"
            )

    df = pd.DataFrame(features)[FEATURE_COLS].fillna(0)
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model.predict_proba(df)[:, 1].tolist()
```

**Graceful degradation**: If `meta` is None (old model without meta file) or `feature_cols` is not in meta (model trained before this improvement), skip the check and proceed as before. This means old models still work but new models get the safety check.

#### `lib/predict.py`

In `run_predict()`, catch the `ValueError` from `score()` and provide a clear user message:

```python
try:
    scores = score(features_list, model_path)
    spike_scores = {f["tcgplayer_id"]: s for f, s in zip(features_list, scores)}
except ValueError as e:
    print(f"Model incompatible: {e}")
    print("Re-training model with current features...")
    rows = generate_training_data(cache)
    if rows:
        train(rows, model_path, device="cpu")
        scores = score(features_list, model_path)
        spike_scores = {f["tcgplayer_id"]: s for f, s in zip(features_list, scores)}
    else:
        print("Insufficient history for retraining. Spike scores will be 0.")
```

### Test Cases

Add to `tests/test_spike.py`:

1. **`test_train_meta_contains_feature_cols`** — Train on fixture data, load meta, assert `meta["feature_cols"] == FEATURE_COLS`.

2. **`test_score_raises_on_feature_mismatch`** — Train a model, then monkeypatch `FEATURE_COLS` to add a fake feature, call `score()`, expect `ValueError` with "retrain required" message.

3. **`test_score_succeeds_when_features_match`** — Train and score with the same `FEATURE_COLS`, no error raised.

4. **`test_score_skips_check_when_no_meta`** — Score against a model that has no `_meta.json` file (delete it after training), verify no error is raised.

5. **`test_score_skips_check_when_meta_lacks_feature_cols`** — Write a meta file without `feature_cols` key, verify score proceeds without error.

Add to `tests/test_predict.py`:

6. **`test_predict_auto_retrains_on_feature_mismatch`** — Set up a model with mismatched `feature_cols` in meta, run `run_predict()`, verify it completes successfully (auto-retrains).

### Backward Compatibility

- Models trained before this change have no `feature_cols` in meta. The check is skipped for these models — they work exactly as before.
- After retraining with the new code, the check is active. Any future feature list change will require explicit retraining.

---

## Improvement 4 — Backtest Command

### Current State

There is no way to evaluate the model's predictive accuracy on historical data. The only feedback loop is manual observation of the watchlist over time. The `generate_training_data()` function already implements sliding-window spike labeling, but its output is only used for training — never for evaluation against actual outcomes.

### Problem

- Cannot measure precision/recall of spike predictions.
- Cannot compare model versions objectively.
- Cannot tune `SPIKE_HOLD_THRESHOLD` based on actual performance.
- No way to answer "how often does a HOLD signal actually result in a price increase?"

### Design

The backtest command replays history: for each day `t` in a configurable window, it trains a model on data up to day `t`, generates predictions at day `t`, then checks outcomes at day `t+30`. This produces precision, recall, and F1 for the spike classifier, plus mean absolute error for the linear forecast.

#### Simplified approach (recommended for v1)

Rather than literally retraining a model at each time step (expensive), the backtest uses the **existing trained model** and replays the labeling logic:

1. Load the inventory cache.
2. For each card with sufficient history (>= 61 days), create evaluation windows:
   - Feature snapshot at day `t` (using `extract_features()` with truncated price history)
   - Actual spike label from the 30-day window starting at day `t`
3. Score all snapshots with the current model.
4. Compare predictions against actual labels.
5. Report precision, recall, F1, and confusion matrix.

This is essentially what `generate_training_data()` already does for producing labeled rows, but instead of using those rows for training, we score them with the model and compare.

### Code Changes

#### `lib/backtest.py` (new file)

```python
"""Backtest the spike classifier against historical data."""

import csv
import json
import os
from datetime import datetime

import numpy as np

from lib.config import SPIKE_HOLD_THRESHOLD, get_logger
from lib.features import generate_training_data, extract_features, compute_cluster_features
from lib.mtgjson import load_inventory_cache
from lib.spike import FEATURE_COLS, score, load_model_meta

log = get_logger(__name__)


def run_backtest(
    data_dir: str,
    models_dir: str,
    output_dir: str,
) -> dict:
    """Run backtest and return metrics dict."""
    cache = load_inventory_cache(data_dir)
    if not cache:
        print("No MTGJson cache found. Run 'monitor.sh sync' first.")
        return {}

    model_path = os.path.join(models_dir, "spike_classifier.json")
    if not os.path.exists(model_path):
        print("No trained model found. Run 'monitor.sh train' first.")
        return {}

    # Generate labeled evaluation data using the same sliding-window logic as training
    rows = generate_training_data(cache)
    if not rows:
        print("Insufficient price history for backtesting.")
        return {}

    # Score all rows with the trained model
    probabilities = score(rows, model_path)

    # Compare predictions to actual labels
    actuals = [r["spike"] for r in rows]
    predicted = [1 if p >= SPIKE_HOLD_THRESHOLD else 0 for p in probabilities]

    tp = sum(1 for a, p in zip(actuals, predicted) if a == 1 and p == 1)
    fp = sum(1 for a, p in zip(actuals, predicted) if a == 0 and p == 1)
    fn = sum(1 for a, p in zip(actuals, predicted) if a == 1 and p == 0)
    tn = sum(1 for a, p in zip(actuals, predicted) if a == 0 and p == 0)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    # Probability calibration: average predicted probability vs actual spike rate per bin
    bins = np.linspace(0, 1, 11)  # 10 bins
    calibration = []
    for i in range(len(bins) - 1):
        mask = [(bins[i] <= p < bins[i+1]) for p in probabilities]
        bin_actual = [a for a, m in zip(actuals, mask) if m]
        if bin_actual:
            calibration.append({
                "bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                "count": len(bin_actual),
                "actual_spike_rate": round(sum(bin_actual) / len(bin_actual), 4),
                "avg_predicted": round(
                    sum(p for p, m in zip(probabilities, mask) if m) / len(bin_actual), 4
                ),
            })

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "model_meta": load_model_meta(model_path),
        "threshold": SPIKE_HOLD_THRESHOLD,
        "num_samples": len(rows),
        "num_spikes": sum(actuals),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "calibration": calibration,
    }

    # Print report
    print(f"\n--- Backtest Results ---")
    print(f"Samples: {len(rows)} ({sum(actuals)} spikes, {len(rows) - sum(actuals)} non-spikes)")
    print(f"Threshold: {SPIKE_HOLD_THRESHOLD}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP={tp}  FP={fp}")
    print(f"  FN={fn}  TN={tn}")

    # Write results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_path = os.path.join(output_dir, f"backtest-{timestamp}.json")
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {results_path}")

    return metrics
```

#### `scripts/monitor.sh`

Add a new command case and function:

```bash
# =============================================================================
# Backtest spike classifier
# =============================================================================
run_backtest() {
    python3 - <<PYEOF
import sys
sys.path.insert(0, "${PRICING_DIR}")
from lib.backtest import run_backtest
run_backtest(
    data_dir="${DATA_DIR}",
    models_dir="${MODELS_DIR}",
    output_dir="${OUTPUT_DIR}",
)
PYEOF
}
```

In the `case` block, add before the `*` catch-all:

```bash
    backtest)
        run_backtest
        ;;
```

Update the help text to include:

```
echo "  backtest                   Backtest model against historical data"
```

### Test Cases

Create `tests/test_backtest.py`:

1. **`test_run_backtest_returns_metrics`** — Set up fixture data + trained model, call `run_backtest()`, assert result dict has keys `precision`, `recall`, `f1`, `confusion_matrix`.

2. **`test_backtest_confusion_matrix_sums_to_total`** — Assert `tp + fp + fn + tn == num_samples`.

3. **`test_backtest_no_cache_returns_empty`** — Call with empty data dir, expect empty dict.

4. **`test_backtest_no_model_returns_empty`** — Call with data but no model file, expect empty dict.

5. **`test_backtest_writes_json_output`** — Verify a `backtest-*.json` file is written to output dir.

6. **`test_backtest_precision_recall_bounds`** — Assert 0 <= precision <= 1, 0 <= recall <= 1.

7. **`test_backtest_calibration_bins`** — Assert calibration list entries have expected keys.

### Backward Compatibility

- New command, no existing behavior changes.
- Requires a trained model to exist (uses `score()`), so model versioning (Improvement 3) applies automatically.

### Note on data leakage

**Important caveat**: The v1 backtest described above evaluates the model on data it was likely trained on (since `generate_training_data()` produces the same rows). This gives an **optimistic** estimate. A future v2 should implement temporal holdout: train on data before date X, evaluate on data after date X. This is noted in the output for transparency. The `_meta.json` and backtest output should include a `"warning": "in-sample evaluation"` field.

For a proper out-of-sample backtest, a future enhancement would split the card cache by date, train on the first N% of time windows, and evaluate on the remaining. This requires refactoring `generate_training_data()` to accept a date range parameter.

---

## Improvement 5 — SetList.json Integration

### Current State

The system has no knowledge of upcoming set releases or spoiler seasons. `lib/mtgjson.py` downloads `AllIdentifiers.json`, `AllPrices.json`, and `TcgplayerSkus.json` but not `SetList.json`. The signal expansion plan (2026-03-01) noted SetList.json as a future data source but deferred it.

### Problem

- Set release dates are strong price signals: cards in Standard-legal sets often drop on reprint, while Commander staples spike during spoiler season when synergistic new cards are revealed.
- No `set_release_proximity` feature means the model cannot learn seasonal price patterns tied to release cycles.
- No `spoiler_season` flag means the model cannot account for the increased volatility window around new set previews.

### Data Source

`SetList.json` (~200KB) from `https://mtgjson.com/api/v5/SetList.json`. Structure:

```json
{
  "data": [
    {
      "code": "DSK",
      "name": "Duskmourn: House of Horror",
      "releaseDate": "2025-09-27",
      "isPartialPreview": false,
      "type": "expansion",
      ...
    },
    ...
  ]
}
```

Key fields per set: `code` (string), `releaseDate` (YYYY-MM-DD string), `isPartialPreview` (bool), `type` (string — "expansion", "commander", "masters", etc.).

### Code Changes

#### `lib/config.py`

Add constants:

```python
# ---------------------------------------------------------------------------
# Set Release Features
# ---------------------------------------------------------------------------
SPOILER_WINDOW_DAYS = 30          # Days before release considered "spoiler season"
RELEASE_PROXIMITY_MAX_DAYS = 90   # Cap for set_release_proximity feature
```

#### `lib/mtgjson.py`

**Download**: Add `SetList.json` to the sync function. Since it is small (~200KB), always download it alongside the other files.

In `sync()`, add after the TcgplayerSkus download:

```python
setlist_path = os.path.join(data_dir, "SetList.json")

# In the download block:
download_json(f"{MTGJSON_BASE}/SetList.json", setlist_path,
              force=should_force("SetList.json"))
```

Update the existence check loop to include SetList.json. Since SetList.json is new and optional (the system should degrade gracefully without it), do NOT add it to the hard requirement loop — only check in a soft manner:

```python
if not os.path.exists(setlist_path):
    log.warning("SetList.json not found — set release features will be unavailable")
```

**New function** to load and index the set list:

```python
def load_set_list(data_dir: str) -> dict:
    """Load SetList.json and return dict keyed by set code.

    Returns:
        {set_code: {"releaseDate": "YYYY-MM-DD", "isPartialPreview": bool, "type": str}}
    """
    path = os.path.join(data_dir, "SetList.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    sets = {}
    for s in data.get("data", []):
        code = s.get("code", "").upper()
        if code:
            sets[code] = {
                "releaseDate": s.get("releaseDate", ""),
                "isPartialPreview": s.get("isPartialPreview", False),
                "type": s.get("type", ""),
            }
    return sets
```

**Cache integration**: Add set release data to inventory cache during `build_inventory_cache()`. The function signature gains an optional `set_list` parameter:

```python
def build_inventory_cache(
    inventory_ids: set[str],
    identifiers_data: dict,
    prices_data: dict,
    sku_to_uuid: dict,
    set_list: dict | None = None,  # NEW
) -> dict:
```

Inside the per-card loop, add:

```python
set_code = card.get("setCode", "").upper()
set_info = (set_list or {}).get(set_code, {})
# Store set release date for feature extraction
cache[sku_id]["setReleaseDate"] = set_info.get("releaseDate", "")
cache[sku_id]["setIsPartialPreview"] = set_info.get("isPartialPreview", False)
```

In `sync()`, load the set list and pass it:

```python
set_list = load_set_list(data_dir)
cache = build_inventory_cache(inventory_ids, identifiers_data, prices_data, sku_to_uuid, set_list)
```

#### `lib/features.py`

Import the new config constants:

```python
from lib.config import (
    RARITY_RANK, SPIKE_THRESHOLD, MIN_PRICE,
    SPOILER_WINDOW_DAYS, RELEASE_PROXIMITY_MAX_DAYS,
    get_logger,
)
```

Add two new features to `extract_features()`:

```python
# Phase 5: set release signals
set_release_date_str = card.get("setReleaseDate", "")
if set_release_date_str:
    try:
        release_date = datetime.fromisoformat(set_release_date_str)
        days_to_release = (release_date - datetime.now()).days
        # Positive = future release, negative = already released
        # Clamp to [-RELEASE_PROXIMITY_MAX_DAYS, RELEASE_PROXIMITY_MAX_DAYS]
        set_release_proximity = max(
            -RELEASE_PROXIMITY_MAX_DAYS,
            min(days_to_release, RELEASE_PROXIMITY_MAX_DAYS)
        )
    except ValueError:
        set_release_proximity = 0
else:
    set_release_proximity = 0

spoiler_season = int(card.get("setIsPartialPreview", False))
```

Return dict additions:

```python
# Phase 5: set release signals (2 features)
"set_release_proximity": set_release_proximity,
"spoiler_season": spoiler_season,
```

**Training data**: In `generate_training_data()`, the `extract_features()` call already passes the full card dict (as `snapshot`), so the new fields will be picked up automatically as long as the cache includes them. The `setReleaseDate` is static per card (does not change across sliding windows), which is correct — the feature represents set proximity relative to the snapshot time. However, we need to adjust: for training windows, `set_release_proximity` should be computed relative to the window date, not `datetime.now()`.

Update `extract_features()` to accept an optional `reference_date` parameter:

```python
def extract_features(tcgplayer_id: str, card: dict, reference_date: datetime | None = None) -> dict:
    ref_date = reference_date or datetime.now()
```

Use `ref_date` instead of `datetime.now()` for:
- `set_age_days` calculation (already uses `datetime.now()`)
- `set_release_proximity` calculation (new)

In `generate_training_data()`, pass the window's cutoff date as `reference_date`:

```python
cutoff_date_dt = datetime.fromisoformat(prices[i][0])
feat = extract_features(tcgplayer_id, snapshot, reference_date=cutoff_date_dt)
```

#### `lib/spike.py`

Add the two new features to `FEATURE_COLS`:

```python
FEATURE_COLS = [
    # ... existing 22 features ...
    # Phase 5: set release
    "set_release_proximity",
    "spoiler_season",
]
```

This brings the total from 22 to 24 features.

### Test Cases

Add to `tests/test_mtgjson.py`:

1. **`test_load_set_list_returns_dict_by_code`** — Write a mock SetList.json, call `load_set_list()`, assert it returns sets keyed by uppercase code.

2. **`test_load_set_list_returns_empty_when_missing`** — Call with a path that has no SetList.json, expect empty dict.

3. **`test_build_inventory_cache_includes_set_release_date`** — Pass a set_list to `build_inventory_cache()`, assert `setReleaseDate` is in the output card dict.

4. **`test_sync_downloads_setlist`** — (Integration-style, with mocked HTTP) Verify that sync attempts to download SetList.json.

Add to `tests/test_features.py`:

5. **`test_set_release_proximity_future_set`** — Card with `setReleaseDate` 30 days in the future, assert `set_release_proximity` is approximately 30.

6. **`test_set_release_proximity_past_set`** — Card with `setReleaseDate` 60 days ago, assert `set_release_proximity` is approximately -60.

7. **`test_set_release_proximity_capped`** — Card with `setReleaseDate` 200 days ago, assert `set_release_proximity` == -RELEASE_PROXIMITY_MAX_DAYS.

8. **`test_set_release_proximity_missing_date`** — Card with no `setReleaseDate`, assert `set_release_proximity` == 0.

9. **`test_spoiler_season_flag`** — Card with `setIsPartialPreview: true`, assert `spoiler_season` == 1.

10. **`test_extract_features_with_reference_date`** — Pass a `reference_date` to `extract_features()`, assert `set_age_days` and `set_release_proximity` are computed relative to that date instead of now.

Update `tests/test_features.py` existing tests:

11. **`test_extract_features_keys`** — Update `expected_keys` to include `set_release_proximity` and `spoiler_season`.

Update `tests/fixtures/inventory_cards.json`:

12. Add `setReleaseDate` and `setIsPartialPreview` fields to both test cards:
    ```json
    "setReleaseDate": "2025-10-15",
    "setIsPartialPreview": false
    ```

### Backward Compatibility

- `SetList.json` download is soft-required: if missing, features default to 0. Old caches without `setReleaseDate`/`setIsPartialPreview` produce `set_release_proximity=0` and `spoiler_season=0`.
- `build_inventory_cache()` gains an optional `set_list` parameter with default `None`, so existing callers (including tests) continue to work without changes.
- `extract_features()` gains an optional `reference_date` parameter with default `None`, so existing callers are unaffected.
- Adding features to `FEATURE_COLS` triggers the version check from Improvement 3, requiring a retrain. This is correct and expected.
- The `sync` command in `monitor.sh` may need its `--setlist` flag or SetList.json can just always download alongside others (recommended since it is only ~200KB).

### Migration

1. Run `monitor.sh sync` to download SetList.json and rebuild cache with new fields.
2. Run `monitor.sh train` to retrain with 24 features.
3. Old models will fail the version check (Improvement 3) and auto-retrain in `predict`.

---

## Improvement 6 — Prediction Confidence (Forecast Intervals)

### Current State

`lib/forecast.py:forecast_card()` returns a single point prediction from linear regression. There is no confidence interval, so all predictions appear equally reliable regardless of how noisy the underlying price history is. A card with stable, linear price growth and a card with wild oscillations both produce a bare point estimate.

### Problem

- Users cannot distinguish high-confidence predictions from noisy guesses.
- The prediction pipeline treats all forecasts equally — a SELL_NOW signal based on a noisy forecast is as weighted as one based on clean data.
- No way to filter the watchlist by prediction reliability.

### Design

Use the standard error of the linear regression residuals to compute prediction intervals. For a linear regression with `n` data points, the standard error of a prediction at point `x_0` is:

```
se = s * sqrt(1 + 1/n + (x_0 - x_mean)^2 / sum((x_i - x_mean)^2))
```

where `s` is the residual standard error. The 95% prediction interval is `predicted +/- 1.96 * se`.

### Code Changes

#### `lib/forecast.py`

Add a new function (keep `forecast_card()` unchanged for backward compatibility):

```python
def forecast_card_with_interval(
    price_history: dict, days_ahead: int = 7, confidence: float = 0.95
) -> dict | None:
    """Predict price N days ahead with prediction interval.

    Returns:
        {
            "predicted": float,
            "lower": float,
            "upper": float,
            "std_error": float,
            "r_squared": float,
        }
        or None if insufficient data.
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

    # Point prediction
    future_x = np.array([[n + days_ahead - 1]])
    predicted = float(model.predict(future_x)[0])
    predicted = max(MIN_PRICE, round(predicted, 4))

    # Residuals and R-squared
    y_pred_train = model.predict(x)
    residuals = price_vals - y_pred_train.flatten()
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((price_vals - np.mean(price_vals)) ** 2))
    r_squared = 1 - ss_res / max(ss_tot, 1e-9)

    # Standard error of prediction
    s = np.sqrt(ss_res / max(n - 2, 1))  # residual standard error
    x_mean = np.mean(x)
    x_0 = n + days_ahead - 1
    sum_sq_dev = float(np.sum((x.flatten() - x_mean) ** 2))
    se = float(s * np.sqrt(1 + 1/n + (x_0 - x_mean)**2 / max(sum_sq_dev, 1e-9)))

    # Z-score for confidence level (use scipy if available, otherwise hardcode common values)
    z_scores = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_scores.get(confidence, 1.960)

    lower = max(MIN_PRICE, round(predicted - z * se, 4))
    upper = round(predicted + z * se, 4)

    return {
        "predicted": predicted,
        "lower": lower,
        "upper": upper,
        "std_error": round(se, 4),
        "r_squared": round(r_squared, 4),
    }
```

#### `lib/predict.py`

Import the new function:

```python
from lib.forecast import forecast_card, forecast_card_with_interval, trend_direction
```

In `run_predict()`, replace the forecast block for each card:

```python
if card:
    ph = card["price_history"]
    forecast_7d = forecast_card_with_interval(ph, 7)
    forecast_30d = forecast_card_with_interval(ph, 30)

    if forecast_7d:
        pred_7d = f"{forecast_7d['predicted']:.4f}"
        conf_7d_lower = f"{forecast_7d['lower']:.4f}"
        conf_7d_upper = f"{forecast_7d['upper']:.4f}"
        r_squared = f"{forecast_7d['r_squared']:.4f}"
    else:
        pred_7d = conf_7d_lower = conf_7d_upper = r_squared = "insufficient_data"

    if forecast_30d:
        pred_30d = f"{forecast_30d['predicted']:.4f}"
        conf_30d_lower = f"{forecast_30d['lower']:.4f}"
        conf_30d_upper = f"{forecast_30d['upper']:.4f}"
    else:
        pred_30d = conf_30d_lower = conf_30d_upper = "insufficient_data"

    trend = trend_direction(ph)
    # ... spike scoring unchanged ...
```

Add new columns to `PRED_FIELDS`:

```python
PRED_FIELDS = [
    "TCGplayer Id", "Product Name", "Current Price", "Market Price",
    "Suggested Price", "Action", "Reason", "Margin",
    "Predicted 7d", "7d Lower", "7d Upper",
    "Predicted 30d", "30d Lower", "30d Upper",
    "R-Squared",
    "Trend", "Spike Probability", "Signal",
]
```

Add the new fields to the prediction row dict:

```python
predictions.append({
    # ... existing fields ...
    "7d Lower": conf_7d_lower,
    "7d Upper": conf_7d_upper,
    "30d Lower": conf_30d_lower,
    "30d Upper": conf_30d_upper,
    "R-Squared": r_squared,
    # ... existing fields ...
})
```

Also update `_write_empty_outputs()` to include the new column names in `pred_fields`.

#### Optional: confidence-weighted signal

As a future enhancement (not required for v1), the `R-Squared` value could modulate the SELL_NOW signal — only trigger SELL_NOW when R-Squared > 0.5 (indicating the linear trend is a reasonable fit). This avoids acting on noisy forecasts.

### Test Cases

Add to `tests/test_forecast.py`:

1. **`test_forecast_with_interval_returns_dict`** — Call `forecast_card_with_interval()` on rising card fixture, assert result is a dict with keys `predicted`, `lower`, `upper`, `std_error`, `r_squared`.

2. **`test_forecast_interval_lower_less_than_upper`** — Assert `result["lower"] <= result["predicted"] <= result["upper"]`.

3. **`test_forecast_interval_returns_none_for_insufficient_data`** — Assert returns None for < 14 days of history.

4. **`test_forecast_interval_lower_bound_floored`** — For a declining card where the lower bound would go negative, assert `result["lower"] >= 0.01`.

5. **`test_forecast_interval_r_squared_between_0_and_1`** — For a linearly rising card, assert `result["r_squared"]` is close to 1.0.

6. **`test_forecast_interval_noisy_data_wider_interval`** — Compare interval width for a stable card vs a noisy card, assert noisy card has wider interval.

7. **`test_forecast_interval_consistent_with_point_forecast`** — Assert `forecast_card_with_interval(ph, 7)["predicted"]` equals `forecast_card(ph, 7)` (same point estimate).

8. **`test_forecast_interval_30d_wider_than_7d`** — For the same card, assert the 30-day interval is wider than the 7-day interval.

Update `tests/test_predict.py`:

9. **`test_predictions_csv_has_confidence_columns`** — Assert the output CSV has `7d Lower`, `7d Upper`, `30d Lower`, `30d Upper`, `R-Squared` columns.

### Backward Compatibility

- `forecast_card()` is unchanged — existing code that calls it continues to work.
- `forecast_card_with_interval()` is a new function; no existing callers are affected.
- The predictions CSV gains 5 new columns. Any downstream consumers that parse by column name (not position) will continue to work. Consumers that parse by position will need updating.
- `_write_empty_outputs()` must be updated to include the new columns in its field list.

---

## Cross-Cutting Concerns

### Files touched by multiple improvements

| File | Improvements |
|---|---|
| `lib/spike.py` | 1, 2, 3, 5 |
| `lib/config.py` | 1, 5 |
| `lib/features.py` | 5 |
| `lib/forecast.py` | 6 |
| `lib/predict.py` | 3, 6 |
| `lib/mtgjson.py` | 5 |
| `scripts/monitor.sh` | 4 |
| `tests/test_spike.py` | 1, 2, 3 |
| `tests/test_features.py` | 5 |
| `tests/test_forecast.py` | 6 |
| `tests/test_predict.py` | 3, 6 |
| `tests/test_mtgjson.py` | 5 |
| `tests/test_backtest.py` | 4 (new file) |
| `lib/backtest.py` | 4 (new file) |
| `tests/fixtures/inventory_cards.json` | 5 |

### Implementation Order

The recommended implementation order minimizes merge conflicts and maximizes safety:

1. **Improvement 1** (validation + class imbalance) — Foundation. Changes `train()` internals but not its signature. All downstream code is unaffected.
2. **Improvement 2** (feature importances) — Small additive change to `train()` that builds on the Improvement 1 meta structure.
3. **Improvement 3** (model versioning) — Depends on 1 and 2 (since `train()` now writes `feature_cols` to the same meta dict). Changes `score()` which is used everywhere.
4. **Improvement 6** (forecast intervals) — Independent of 1-3. Touches `forecast.py` and `predict.py` only. Can be done in parallel with 1-3.
5. **Improvement 5** (SetList.json) — Adds new features, which triggers the versioning check from Improvement 3. Should be done after 3 is in place so the mismatch check works correctly.
6. **Improvement 4** (backtest) — Depends on all others being stable. Uses `score()` and `generate_training_data()`, so benefits from the versioning and validation improvements.

### New dependencies

- `scikit-learn` is already in `requirements.txt` (used by `forecast.py` for `LinearRegression`). Improvement 1 adds `StratifiedShuffleSplit`, `accuracy_score`, `roc_auc_score`, `precision_score`, `recall_score` — all from scikit-learn, no new pip dependency.
- No new pip dependencies are introduced by any improvement.

### Config constants summary

| Constant | Value | Improvement | File |
|---|---|---|---|
| `VALIDATION_SPLIT` | 0.2 | 1 | `lib/config.py` |
| `RANDOM_SEED` | 42 | 1 | `lib/config.py` |
| `SPOILER_WINDOW_DAYS` | 30 | 5 | `lib/config.py` |
| `RELEASE_PROXIMITY_MAX_DAYS` | 90 | 5 | `lib/config.py` |

### Total feature count progression

| State | Count |
|---|---|
| Current | 22 |
| After Improvement 5 | 24 |

### Total test count progression

| State | Count |
|---|---|
| Current | 54 |
| After Improvement 1 | 58 (+4) |
| After Improvement 2 | 61 (+3) |
| After Improvement 3 | 66 (+5, across spike + predict) |
| After Improvement 4 | 73 (+7, new test file) |
| After Improvement 5 | 84 (+11, across features + mtgjson + fixture update) |
| After Improvement 6 | 93 (+9, across forecast + predict) |

### Migration checklist

After all improvements are implemented:

1. `pip install -r requirements.txt` — no new deps, but verify scikit-learn is current enough for the metrics imports.
2. `bash scripts/monitor.sh sync` — downloads SetList.json, rebuilds cache with `setReleaseDate` and `setIsPartialPreview`.
3. `bash scripts/monitor.sh train` — retrains with 24 features, stratified split, class weighting, writes full meta.
4. `bash scripts/monitor.sh predict --dry-run` — verify predictions include confidence columns.
5. `bash scripts/monitor.sh backtest` — baseline precision/recall metrics.
6. `python3 -m pytest tests/ -v` — all tests pass (target: ~93 tests).
