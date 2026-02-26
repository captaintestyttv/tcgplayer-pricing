# TCGPlayer Price Monitor

A personal tool for monitoring Magic: The Gathering card prices on TCGPlayer, with predictive modeling to forecast price movements and flag hold/sell opportunities.

## What it does

- **Import** TCGPlayer CSV exports and track price history over time
- **Analyze** price changes between exports (spikes, drops, stable cards)
- **Report** pricing recommendations based on current market vs. your listed price
- **Predict** future prices using linear regression + XGBoost spike classification backed by MTGJson historical data
- **Watchlist** cards with high spike probability so you know what to hold

## Prerequisites

- Python 3.10+
- A TCGPlayer seller account with inventory

```bash
pip3 install -r requirements.txt
```

## Setup

### First time

```bash
# Clone and install
git clone https://github.com/captaintestyttv/tcgplayer-pricing
cd tcgplayer-pricing
pip3 install -r requirements.txt

# Import your first export (download from TCGPlayer → Inventory → Export)
bash scripts/monitor.sh import tcgplayer-exports/TCGplayer__MyPricing_*.csv

# Set it as your baseline for comparison
bash scripts/monitor.sh baseline
```

### Enable predictive pricing (one-time, ~15 minutes)

```bash
# Download MTGJson data and build your inventory cache
bash scripts/monitor.sh sync

# Train the spike classifier
# Option A: locally on Pi (slow but works)
bash scripts/monitor.sh train

# Option B: remotely on a machine with NVIDIA GPU (fast)
bash scripts/monitor.sh train --remote <tailscale-hostname>
```

See [docs/remote-training.md](docs/remote-training.md) for GPU setup details.

## Daily workflow

```bash
# 1. Download a fresh export from TCGPlayer → Inventory → Export Pricing
# 2. Import it
bash scripts/monitor.sh import tcgplayer-exports/<new-file>.csv

# 3. Run predictive recommendations
bash scripts/monitor.sh predict
```

Output files appear in `output/`:
- `predictions-<timestamp>.csv` — full pricing recommendations with forecasts
- `watchlist-<timestamp>.csv` — high spike-probability cards to consider holding

## Commands

| Command | Description |
|---|---|
| `import <file.csv>` | Import a TCGPlayer pricing export |
| `analyze` | Compare last two exports — show spikes, drops, stable |
| `report` | Generate pricing recommendations (no predictions) |
| `baseline` | Set current data as your comparison baseline |
| `sync` | Download/refresh MTGJson data (~650MB, run monthly) |
| `train [--remote host]` | Train the spike classifier |
| `predict` | Full predictive recommendations + watchlist |

## Output format

### `predictions-<timestamp>.csv`

| Column | Description |
|---|---|
| TCGplayer Id | Card ID |
| Product Name | Card name |
| Current Price | Your current listed price |
| Market Price | TCGPlayer market price |
| Suggested Price | Recommended price |
| Action | `RAISE`, `LOWER`, or blank |
| Reason | Why the action is recommended |
| Margin | Estimated net profit after all fees |
| Predicted 7d | Forecasted price in 7 days |
| Predicted 30d | Forecasted price in 30 days |
| Trend | `up`, `down`, or `flat` |
| Spike Probability | 0–1 score from XGBoost classifier |
| Signal | `HOLD`, `SELL_NOW`, or blank |

### `watchlist-<timestamp>.csv`

Cards with `Spike Probability >= 0.6`, sorted descending. Intended for a quick daily review before adjusting prices.

## Fee structure

Margin calculations use these constants (editable in `lib/predict.py`):

| Fee | Value |
|---|---|
| TCGPlayer commission | 10.75% |
| Payment processing | 2.5% + $0.30 |
| Shipping revenue (cards < $5) | +$1.31 |
| Postage (cards < $5) | $0.73 |
| Postage (cards ≥ $5) | $1.50 |

## Project structure

```
scripts/monitor.sh          Main CLI entry point
scripts/train_remote.py     Runs on remote GPU machine for training
lib/
  mtgjson.py               MTGJson download + inventory cache
  features.py              Feature extraction for ML models
  forecast.py              Per-card linear regression (7d/30d)
  spike.py                 XGBoost spike classifier
  predict.py               Full prediction pipeline
history/                   Timestamped price export archives
data/mtgjson/              MTGJson cache (gitignored for raw files)
models/                    Trained model files (gitignored)
output/                    Generated CSVs (gitignored)
tests/                     pytest test suite (29 tests)
```

## Running tests

```bash
pytest tests/ -v
```
