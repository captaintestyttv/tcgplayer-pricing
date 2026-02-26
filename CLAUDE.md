# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a TCGPlayer price monitoring tool for Magic: The Gathering card inventory. It imports TCGPlayer CSV exports, tracks price history, and generates pricing recommendations.

## Commands

```bash
# Import a new TCGPlayer export
bash scripts/monitor.sh import tcgplayer-exports/<file>.csv

# Analyze price changes between last two exports
bash scripts/monitor.sh analyze

# Generate pricing recommendations CSV
bash scripts/monitor.sh report

# Set current data as baseline
bash scripts/monitor.sh baseline
```

## Architecture

Single bash script (`scripts/monitor.sh`) with embedded Python heredocs for data processing:

- **`import`** — copies the CSV to `history/export-<timestamp>.csv` and `history/latest.csv`, then runs analysis automatically
- **`analyze`** — Python compares the two most recent history exports; saves `output/analysis-latest.json`
- **`report`** — Python reads the latest export, applies fee math, and saves a `output/price-adjustments-<timestamp>.csv` ready for review

### Directory layout

| Path | Purpose |
|---|---|
| `history/` | Timestamped export archives + `latest.csv` + `baseline.csv` |
| `output/` | Generated analysis JSON and price-adjustment CSVs |
| `tcgplayer-exports/` | Drop zone for raw TCGPlayer downloads |

### Fee constants (encoded in the script)

| Variable | Value | Meaning |
|---|---|---|
| `COMMISSION_FEE` | 10.75% | TCGPlayer seller commission |
| `TRANSACTION_FEE` | 2.5% | Payment processing percentage |
| `TRANSACTION_FLAT` | $0.30 | Payment processing flat fee |
| `SHIPPING_REVENUE` | $1.31 | Shipping charged to buyer |
| `POSTAGE_STANDARD` | $0.73 | Actual postage for cards < $5 |

Cards ≥ $5 use no shipping revenue and $1.50 postage (media mail assumed).

### Pricing recommendation logic

- Net margin < $0.10 → RAISE
- Market > current listing by 10%+ → RAISE
- Market < current listing by 10%+ → LOWER
- Suggested price = `market * 0.98` for RAISE, `market` for LOWER