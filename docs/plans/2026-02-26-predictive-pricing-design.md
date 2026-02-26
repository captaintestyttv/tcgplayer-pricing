# Predictive Pricing Design

**Date:** 2026-02-26
**Status:** Approved

## Goal

Add predictive modeling to the TCGPlayer price monitor to:
- Forecast each card's price at +7 days and +30 days (short and medium term)
- Anticipate price spikes before they happen
- Produce forward-looking pricing recommendations and a hold/sell watchlist

## Approach

Statistical per-card linear regression for price forecasting, combined with an XGBoost spike classifier trained on card metadata and price features. Training runs on a remote PC with NVIDIA GPU via Tailscale SSH; inference runs locally on the Pi.

## Architecture

```
scripts/monitor.sh          ← CLI entry point, gains sync/train/predict commands
lib/
  mtgjson.py               ← download/cache MTGJson data, build inventory cache
  features.py              ← extract feature vectors from price history + metadata
  forecast.py              ← per-card linear regression (7d, 30d)
  spike.py                 ← XGBoost spike classifier (train + score)
  predict.py               ← orchestrates pipeline, writes output files
data/
  mtgjson/                 ← cached MTGJson files (gitignored)
    inventory_cards.json   ← lean per-card cache for inventory only
models/
  spike_classifier.json    ← trained XGBoost model (XGBoost native format)
```

Existing commands (`import`, `analyze`, `report`, `baseline`) are unchanged. `predict` is additive and replaces `report` in the ongoing workflow once the model is trained.

## Data

### MTGJson files downloaded

| File | Size | Purpose |
|---|---|---|
| `AllPrices.json` | ~600MB | Daily TCGPlayer price history per card, keyed by MTGJson UUID |
| `AllIdentifiers.json` | ~50MB | Maps UUIDs to third-party IDs including `tcgplayerProductId` |

After `sync`, both files are processed into a lean `inventory_cards.json` containing only cards present in the current inventory. Subsequent runs use this cache — no re-downloading unless `sync` is explicitly called.

### ID mapping

`AllIdentifiers.json` → `identifiers.tcgplayerProductId` matches `TCGplayer Id` in exports exactly. Cards without a match (sealed product, non-singles) are excluded from predictions and fall through to standard pricing logic.

### Inventory card cache schema

```json
{
  "8553809": {
    "uuid": "...",
    "name": "Alacrian Jaguar",
    "rarity": "common",
    "setCode": "ATD",
    "printings": ["ATD"],
    "legalities": { "standard": "legal", "modern": "legal" },
    "price_history": { "2026-01-01": 0.05, "2026-01-02": 0.06 }
  }
}
```

## Models

### Price Forecast — per-card linear regression (`forecast.py`)

Fits a linear regression on the last 90 days of MTGJson price history for each card. Outputs:
- `predicted_price_7d`
- `predicted_price_30d`
- `trend`: `up` / `flat` / `down`

Cards with fewer than 14 days of history are flagged `insufficient_data` and fall back to existing market-price logic. Spike scoring still runs on metadata features alone for these cards.

### Spike Classifier — XGBoost (`spike.py`)

Trained across all inventory cards. Outputs `spike_probability` (0–1).

**Spike definition:** >20% price increase within any 30-day window — consistent with the existing `spikes_up` threshold in `analyze_changes`.

**Features:**

| Feature | Rationale |
|---|---|
| `rarity_rank` | Mythics spike more than commons |
| `num_printings` | Fewer printings → higher spike risk |
| `set_age_days` | Older cards with low supply spike on rediscovery |
| `formats_legal_count` | More format inclusions → more demand exposure |
| `price_momentum_7d` | Recent upward movement predicts continuation |
| `price_volatility_30d` | High volatility cards are already in motion |
| `current_price` | Cheap cards spike differently than expensive ones |

Training labels are derived from price history: any card crossing the +20% threshold in a 30-day window is a positive example.

**Model format:** XGBoost native `.json` — version-stable, loads identically on Pi (CPU) and PC (GPU).

## Remote Training

```bash
monitor.sh train                           # local XGBoost CPU (slow fallback)
monitor.sh train --remote <tailscale-host> # GPU training on PC
```

### Data flow

```
Pi                               PC (via Tailscale SSH)
────────────────────             ──────────────────────
lib/features.py
  → features.json
rsync features.json ───────────→ receives features.json
ssh: python train.py ──────────→ XGBoost trains (device='cuda')
                                   → spike_classifier.json
rsync model ←──────────────────  spike_classifier.json
models/ updated
```

No persistent service on the PC — it just needs to be awake and reachable on Tailscale.

### Requirements

| Machine | Packages |
|---|---|
| Pi | `pandas numpy xgboost requests` |
| PC | `pandas numpy xgboost` + CUDA toolkit |

## Commands (updated)

| Command | Description |
|---|---|
| `monitor.sh sync` | Download MTGJson data, build `inventory_cards.json` |
| `monitor.sh train [--remote host]` | Extract features, train XGBoost spike classifier |
| `monitor.sh predict` | Run forecast + spike scoring, write enriched output |
| `monitor.sh import <file>` | Unchanged |
| `monitor.sh analyze` | Unchanged |
| `monitor.sh report` | Unchanged (fallback if predict not yet set up) |
| `monitor.sh baseline` | Unchanged |

## Output

### `output/predictions-<timestamp>.csv`

Full forward-looking recommendations, extending the existing report format:

```
TCGplayer Id, Product Name, Current Price, Market Price, Suggested Price,
Action, Reason, Margin, Predicted 7d, Predicted 30d, Trend,
Spike Probability, Signal
```

`Signal` values: `HOLD` (spike_probability > 0.6), `SELL_NOW` (trend=down + listed near market), blank (defer to existing pricing logic).

### `output/watchlist-<timestamp>.csv`

Hold candidates only — cards with `spike_probability > 0.6`, sorted descending. Intended for a quick daily review.

## Degradation Handling

| Condition | Behavior |
|---|---|
| `sync` not yet run | `predict` warns, falls back to `report` |
| Card not in MTGJson | Standard pricing recommendation only, no prediction columns |
| < 14 days price history | Forecast skipped; spike scoring runs on metadata features only |
| Model file missing | `predict` auto-runs `train` before proceeding |
| Remote PC unreachable | `train` falls back to local CPU with a warning |
