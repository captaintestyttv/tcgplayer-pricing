# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TCGPlayer price monitoring and predictive pricing tool for Magic: The Gathering card inventory. It imports TCGPlayer CSV exports, tracks price history, generates pricing recommendations, and uses machine learning (XGBoost) to predict price spikes.

## Commands

```bash
# Import a new TCGPlayer export (auto-detects CSV in tcgplayer-exports/)
bash scripts/monitor.sh import
bash scripts/monitor.sh import tcgplayer-exports/<file>.csv  # or specify explicitly

# Analyze price changes between last two exports
bash scripts/monitor.sh analyze

# Generate pricing recommendations CSV
bash scripts/monitor.sh report

# Set current data as baseline
bash scripts/monitor.sh baseline

# Download/refresh MTGJson data and rebuild inventory cache
bash scripts/monitor.sh sync                # download missing files + rebuild cache
bash scripts/monitor.sh sync --force        # re-download all files + rebuild cache
bash scripts/monitor.sh sync --cache        # rebuild cache only (no downloads)
bash scripts/monitor.sh sync --prices       # re-download AllPrices.json only
bash scripts/monitor.sh sync --identifiers  # re-download AllIdentifiers.json only
bash scripts/monitor.sh sync --skus         # re-download TcgplayerSkus.json only

# Train the spike classifier model
bash scripts/monitor.sh train               # local (auto-detects CPU/CUDA)
bash scripts/monitor.sh train --remote <host>  # remote GPU via SSH

# Run full predictive pipeline (forecast + spike detection + recommendations)
bash scripts/monitor.sh predict
bash scripts/monitor.sh predict --dry-run  # preview without writing files

# Evaluate model performance against historical data
bash scripts/monitor.sh backtest
```

## Running Tests

```bash
# Run the full test suite (80 tests)
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_features.py -v
```

Dependencies: `pip install -r requirements.txt` (pandas, numpy, scikit-learn, xgboost, requests, pytest)

## Platform Notes

Developed on Windows 11 with Git Bash (MSYS2) and Python via conda. The conda environment is named `tcgplayer`. Activate it with:

```bash
conda activate tcgplayer
```

`monitor.sh` auto-detects the Python command (`python3` on Linux/macOS, `python` on Windows/conda). All bash commands should be run through Git Bash on Windows.

## Architecture

Entry point is `scripts/monitor.sh` (bash with embedded Python heredocs). The predictive pricing pipeline lives in `lib/` as standalone Python modules.

### Directory Layout

| Path | Purpose |
|---|---|
| `scripts/monitor.sh` | Main CLI — all commands route through here |
| `scripts/train_remote.py` | Entry point for remote GPU training jobs |
| `lib/` | Python modules for predictive pricing pipeline |
| `lib/config.py` | Centralized constants (fees, thresholds, hyperparameters) |
| `lib/mtgjson.py` | MTGJson download, caching, and inventory builder |
| `lib/features.py` | Feature extraction + spike labeling for training |
| `lib/forecast.py` | Per-card linear regression price forecasting |
| `lib/spike.py` | XGBoost spike classifier (train + score) |
| `lib/predict.py` | Prediction orchestration — ties everything together |
| `lib/backtest.py` | Model evaluation against historical training data |
| `tests/` | pytest test suite (80 tests across 6 files) |
| `tests/fixtures/` | Test fixture data (inventory_cards.json) |
| `docs/plans/` | Design docs and implementation plans |
| `history/` | Timestamped export archives + `latest.csv` + `baseline.csv` |
| `output/` | Generated CSVs and analysis JSON (gitignored) |
| `data/mtgjson/` | Cached MTGJson files (~650MB, gitignored) |
| `models/` | Trained model files (gitignored) |
| `tcgplayer-exports/` | Drop zone for raw TCGPlayer downloads (gitignored) |
| `.github/workflows/` | CI — runs tests on push/PR to main |

### Data Pipeline

1. **Import** — TCGPlayer CSV exports go into `history/` with timestamps
2. **Sync** — Downloads MTGJson files (AllPrices.json, AllIdentifiers.json, TcgplayerSkus.json, SetList.json), builds `inventory_cards.json` cache joining card metadata with price history scoped to the user's inventory
3. **Train** — Generates training data via 31-day sliding windows, labels spikes (>20% increase in 30 days), trains XGBoost classifier with stratified validation split. Records validation metrics, feature importance, and feature version in companion metadata file
4. **Predict** — Runs forecasting + spike scoring + fee-based margin calc, outputs `predictions-*.csv` (with confidence intervals) and `watchlist-*.csv`. Auto-retrains if model version mismatches current features
5. **Backtest** — Evaluates trained model against historical training data, computing accuracy, precision, recall, F1, confusion matrix, and probability calibration bins. Outputs `backtest-*.json`

### Python Modules

**`lib/config.py`** — Single source of truth for all tunable constants: fee values, pricing thresholds, model hyperparameters, network settings, and CSV schema. Also provides `get_logger(name)` for structured logging (level controlled by `LOG_LEVEL` env var).

**`lib/features.py`** — Extracts 24 features per card across 6 signal categories. Original 7: `rarity_rank`, `num_printings`, `set_age_days`, `formats_legal_count`, `price_momentum_7d`, `price_volatility_30d`, `current_price`. Card metadata (9): `edhrec_rank`, `edhrec_saltiness`, `is_reserved_list`, `is_legendary`, `is_creature`, `color_count`, `keyword_count`, `mana_value`, `subtype_count`. Foil/buylist (3): `foil_to_normal_ratio`, `buylist_ratio`, `buylist_momentum_7d`. Cluster (1): `cluster_momentum_7d` (max avg momentum across card's subtypes, computed post-hoc via `compute_cluster_features()`). Change detection (2): `recently_reprinted`, `legality_changed`. Set timing (2): `set_release_proximity` (days until set release, 0–90), `spoiler_season` (binary, from partial preview or proximity ≤30 days). Accepts optional `reference_date` for historically-correct feature computation in training windows. Spike threshold: >20% price increase in 30 days.

**`lib/forecast.py`** — Linear regression on last 90 days of price history. Requires minimum 14 days of data. Returns 7-day and 30-day price predictions plus trend direction (up/down/flat with 3% threshold). Floor at $0.01. Also provides `forecast_with_confidence()` returning prediction intervals (lower/upper bounds at 95% confidence) and R-squared goodness of fit, using residual standard error with leverage-based intervals.

**`lib/spike.py`** — XGBoost classifier (200 estimators, max_depth=4, learning_rate=0.1). Uses 24 features (all except `tcgplayer_id` and `spike` label). Outputs probability 0–1. Supports CPU and CUDA devices. Training uses stratified 80/20 validation split with `scale_pos_weight` for class imbalance. Model saved as `.json` with companion `_meta.json` recording: training timestamp, sample count, device, spike rate, hyperparameters (including scale_pos_weight), validation metrics (accuracy, AUC, precision, recall), feature importance (sorted descending), and feature column list for version compatibility. `check_model_compatibility()` validates feature list against current code; `score()` raises on mismatch.

**`lib/predict.py`** — Orchestrates the full pipeline. Loads latest.csv + inventory cache, auto-trains if model missing or version-incompatible, scores all cards, applies pricing rules. Spike probability >= 0.6 triggers HOLD signal. Outputs 18-column predictions CSV (with confidence intervals and R-squared) and filtered watchlist CSV. Supports `--dry-run` for preview without file output.

**`lib/backtest.py`** — Evaluates model quality by scoring training data and comparing predicted probabilities to actual spike labels. Computes confusion matrix (TP/FP/FN/TN), accuracy, precision, recall, F1, and 10-bin probability calibration. Writes results to `output/backtest-*.json`. Requires trained model and inventory cache.

**`lib/mtgjson.py`** — Downloads and caches MTGJson bulk files with retry logic (exponential backoff) and atomic writes (temp file + rename). Builds SKU-to-UUID mapping from TcgplayerSkus.json to match TCGPlayer inventory IDs to card UUIDs in AllIdentifiers.json. Also downloads SetList.json for set release dates and spoiler status. Cache includes card metadata (edhrecRank, isReserved, supertypes, types, subtypes, colorIdentity, keywords, manaValue, text), foil and buylist price histories, set timing data (setReleaseDate, setIsPartialPreview), and change detection flags (recently_reprinted, legality_changed) computed by comparing with the previous cache on rebuild.

### Fee Constants

| Variable | Value | Meaning |
|---|---|---|
| `COMMISSION_FEE` | 10.75% | TCGPlayer seller commission |
| `TRANSACTION_FEE` | 2.5% | Payment processing percentage |
| `TRANSACTION_FLAT` | $0.30 | Payment processing flat fee |
| `SHIPPING_REVENUE` | $1.31 | Shipping charged to buyer (cards < $5) |
| `POSTAGE_STANDARD` | $0.73 | Actual postage for cards < $5 |

Cards >= $5 use no shipping revenue and $1.50 postage (media mail assumed).

### Pricing Recommendation Logic

- Net margin < $0.10 → RAISE
- Market > current listing by 10%+ → RAISE
- Market < current listing by 10%+ → LOWER
- Competitive adjustment for high-value cards (>= $5, current < 95% of market) → RAISE
- Suggested price = `market * 0.98` for RAISE, `market` for LOWER

### Predictive Signals

- `HOLD` — Spike probability >= 0.6 (card likely to increase, don't lower price)
- `SELL_NOW` — Downtrend + price near market (sell before further decline)
- Predictions CSV includes: Predicted 7d, 7d Lower, 7d Upper, Predicted 30d, 30d Lower, 30d Upper, R-Squared, Trend, Spike Probability, Signal

### Degradation Behavior

The system degrades gracefully when components are unavailable:

- No MTGJson cache → pricing-only output, no predictions
- Missing or version-mismatched model → auto-retrains before predict
- Insufficient price history (<14 days) → skips forecast, uses metadata features only
- Remote GPU unreachable → falls back to local CPU training

## Conventions

- **Bash + Python hybrid**: `monitor.sh` uses heredocs (`<<PYEOF`) for inline Python. The `lib/` modules are standard Python imported via `sys.path.insert`.
- **Configuration**: All tunable constants live in `lib/config.py`. Import from there — never duplicate values.
- **Logging**: Use `from lib.config import get_logger; log = get_logger(__name__)`. Set `LOG_LEVEL=DEBUG` env var for verbose output.
- **Path handling**: All paths derive from `SCRIPT_DIR`/`PRICING_DIR` — no hardcoded absolute paths.
- **Timestamps**: `YYYYMMDD-HHMMSS` format for filenames.
- **Card IDs**: Column is `TCGplayer Id` (string). Matching to MTGJson uses SKU-to-UUID mapping via TcgplayerSkus.json.
- **CSV validation**: Import validates required columns (`TCGplayer Id`, `Product Name`, `TCG Market Price`, `TCG Marketplace Price`, `Total Quantity`) before accepting a file.
- **Testing**: pytest with temporary directories for isolation. Fixtures in `tests/fixtures/`. Each module has its own test file. CI runs via GitHub Actions on push/PR to main.
- **Output CSVs**: Written to `output/` with timestamped filenames. The `output/` directory is gitignored and created on demand.
