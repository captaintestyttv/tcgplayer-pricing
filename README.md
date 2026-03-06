# TCGPlayer Price Monitor

A personal tool for monitoring Magic: The Gathering card prices on TCGPlayer, with predictive modeling to forecast price movements and flag hold/sell opportunities.

## What it does

- **Import** TCGPlayer CSV exports and track price history over time
- **Accumulate** persistent price history in Parquet storage across syncs
- **Analyze** price changes between exports (spikes, drops, stable cards)
- **Report** pricing recommendations based on current market vs. your listed price
- **Predict** future prices using linear regression + XGBoost spike classification backed by MTGJson historical data
- **Watchlist** cards with high spike probability so you know what to hold
- **Browse** results in a local Flask web UI or the SaaS scaffold

## Prerequisites

- Python 3.10+
- A TCGPlayer seller account with inventory

```bash
pip install -r requirements.txt
```

## Setup

### First time

```bash
# Clone and install
git clone https://github.com/captaintestyttv/tcgplayer-pricing
cd tcgplayer-pricing
pip install -r requirements.txt

# Import your first export (download from TCGPlayer -> Inventory -> Export)
bash scripts/monitor.sh import tcgplayer-exports/TCGplayer__MyPricing_*.csv

# Set it as your baseline for comparison
bash scripts/monitor.sh baseline
```

### Enable predictive pricing (one-time, ~15 minutes)

```bash
# Download MTGJson data and build your inventory cache
# Also persists price history to Parquet store for accumulation
bash scripts/monitor.sh sync

# Train the spike classifier
# Option A: locally (auto-detects CPU/CUDA)
bash scripts/monitor.sh train

# Option B: remotely on a machine with NVIDIA GPU (fast)
bash scripts/monitor.sh train --remote <tailscale-hostname>
```

See [docs/remote-training.md](docs/remote-training.md) for GPU setup details.

### Deepen price history (optional, requires MTGGoldfish Premium)

```bash
# Download historical CSVs for all inventory cards
python scripts/goldfish_import.py --cookie "SESSION=..." --cards inventory

# Or import already-downloaded CSVs
python scripts/goldfish_import.py --import-only data/goldfish_raw/
```

This adds years of daily price data to the Parquet store, improving model training depth. Requires an active MTGGoldfish Premium subscription. The session cookie can be obtained from browser dev tools after logging in. Downloads are rate-limited (2s between requests) with automatic resume support.

## Daily workflow

```bash
# 1. Download a fresh export from TCGPlayer -> Inventory -> Export Pricing
# 2. Import it
bash scripts/monitor.sh import tcgplayer-exports/<new-file>.csv

# 3. Run predictive recommendations
bash scripts/monitor.sh predict
```

Output files appear in `output/`:
- `predictions-<timestamp>.csv` -- full pricing recommendations with forecasts
- `watchlist-<timestamp>.csv` -- high spike-probability cards to consider holding

### Local web UI

```bash
python -m web.app
```

Opens a Flask dashboard at `http://localhost:5000` with pages for predictions, watchlist, analysis, backtest results, individual card details, and background job management.

## Commands

| Command | Description |
|---|---|
| `import <file.csv>` | Import a TCGPlayer pricing export |
| `analyze` | Compare last two exports -- show spikes, drops, stable |
| `report` | Generate pricing recommendations (no predictions) |
| `baseline` | Set current data as your comparison baseline |
| `sync [flags]` | Download/refresh MTGJson data + persist to Parquet (~650MB, run monthly) |
| `train [--remote host]` | Train the spike classifier |
| `predict [--dry-run]` | Full predictive recommendations + watchlist |
| `backtest` | Evaluate model against historical training data |

### Sync flags

| Flag | Effect |
|---|---|
| *(none)* | Download missing files + rebuild cache |
| `--force` | Re-download all files + rebuild cache |
| `--cache` | Rebuild cache only (no downloads) |
| `--prices` | Re-download AllPrices.json only |
| `--identifiers` | Re-download AllIdentifiers.json only |
| `--skus` | Re-download TcgplayerSkus.json only |

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
| 7d Lower / 7d Upper | 95% confidence interval bounds |
| Predicted 30d | Forecasted price in 30 days |
| 30d Lower / 30d Upper | 95% confidence interval bounds |
| R-Squared | Forecast model goodness of fit (0-1) |
| Trend | `up`, `down`, or `flat` |
| Spike Probability | 0-1 score from XGBoost classifier |
| Signal | `HOLD`, `SELL_NOW`, or blank |

### `watchlist-<timestamp>.csv`

Cards with `Spike Probability >= 0.6`, sorted descending. Intended for a quick daily review before adjusting prices.

## Fee structure

Margin calculations use these constants (defined in `lib/config.py`):

| Fee | Value |
|---|---|
| TCGPlayer commission | 10.75% |
| Payment processing | 2.5% + $0.30 |
| Shipping revenue (cards < $5) | +$1.31 |
| Postage (cards < $5) | $0.73 |
| Postage (cards >= $5) | $1.50 |

## Data pipeline

```
TCGPlayer CSV  -->  Import  -->  history/latest.csv
                                      |
MTGJson files  -->  Sync    -->  data/mtgjson/inventory_cards.json
                        |             |
                        +-------> data/price_history/**/*.parquet  (accumulated)
                                      |
MTGGoldfish    -->  Goldfish ----+    |
                    Import            |
                                      v
                              Train (sliding windows)
                                      |
                                      v
                              models/spike_model.json
                                      |
                                      v
                              Predict (forecast + spike + pricing)
                                      |
                                      v
                              output/predictions-*.csv
                              output/watchlist-*.csv
```

## Project structure

```
scripts/
  monitor.sh              Main CLI entry point (bash + Python heredocs)
  train_remote.py         Runs on remote GPU machine for training
  goldfish_import.py      MTGGoldfish Premium CSV downloader/importer

lib/
  config.py               Centralized constants and logging
  mtgjson.py              MTGJson download + inventory/training cache builder
  features.py             24-feature extraction for ML (6 signal categories)
  forecast.py             Per-card linear regression (7d/30d with confidence)
  spike.py                XGBoost spike classifier (train + score)
  predict.py              Full prediction pipeline orchestration
  backtest.py             Model evaluation (confusion matrix, calibration)
  analysis.py             Price change analysis between exports
  price_store.py          Parquet-based persistent price history store
  goldfish.py             MTGGoldfish CSV parser + fuzzy card matcher

web/
  app.py                  Flask web UI routes
  data.py                 Data loading layer for web UI
  jobs.py                 Background job runner
  templates/              Jinja2 HTML templates
  static/                 CSS + JavaScript

saas/                     Next.js SaaS rewrite scaffold
  prisma/schema.prisma    Database schema (10 models, TimescaleDB)
  docker-compose.yml      TimescaleDB + ML service containers
  scripts/migrate-data.ts Data migration from JSON/Parquet to PostgreSQL
  src/server/             tRPC routers + NextAuth + pg-boss jobs
  src/app/                7 pages (dashboard, predictions, watchlist, etc.)
  src/components/         shadcn/ui components + price chart

ml-service/               FastAPI ML microservice for SaaS backend
  main.py                 /train, /predict, /backtest, /health endpoints
  db.py                   SQLAlchemy queries matching Prisma schema

tests/                    pytest test suite (117 tests, 10 files)
  fixtures/               Test fixture data
history/                  Timestamped price export archives
data/mtgjson/             MTGJson cache (~650MB, gitignored)
data/price_history/       Parquet price store (gitignored)
data/goldfish_raw/        Downloaded MTGGoldfish CSVs (gitignored)
models/                   Trained model files (gitignored)
output/                   Generated CSVs + analysis JSON (gitignored)
tcgplayer-exports/        Drop zone for raw TCGPlayer downloads (gitignored)
docs/                     Design docs, plans, remote training guide
.github/workflows/        CI -- runs tests on push/PR to main
```

## Running tests

```bash
# Full suite
python -m pytest tests/ -v

# Single file
python -m pytest tests/test_features.py -v
```

CI runs automatically via GitHub Actions on push/PR to main.

## SaaS deployment (scaffold)

The `saas/` directory contains a Next.js scaffold for a production SaaS version. It is not yet production-ready but provides the full architecture.

### Local development

```bash
# Start TimescaleDB + ML service
cd saas && docker compose up -d

# Apply Prisma schema
cd saas && npx prisma db push

# Apply TimescaleDB extensions
docker exec -i saas-db-1 psql -U tcgplayer < saas/prisma/timescaledb-setup.sql

# Migrate existing data to PostgreSQL
cd saas && npx tsx scripts/migrate-data.ts

# Start Next.js dev server
cd saas && npm run dev
```

### Tech stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, shadcn/ui, Recharts |
| API | tRPC v11 with superjson |
| Database | PostgreSQL 16 + TimescaleDB (hypertables, compression) |
| ORM | Prisma v7 with `@prisma/adapter-pg` |
| Auth | NextAuth v4 (JWT, credentials provider) |
| Jobs | pg-boss (PostgreSQL-backed task queue) |
| ML | FastAPI microservice wrapping existing lib/ pipeline |
| Deploy | Vercel (frontend) + Docker (DB + ML service) |

## License

Private / personal use.
