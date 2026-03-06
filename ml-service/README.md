# ML Service

FastAPI microservice that bridges the Python ML pipeline (`lib/`) with the PostgreSQL database used by the SaaS frontend.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check + model status |
| POST | `/train` | Train XGBoost spike classifier from DB data |
| POST | `/predict` | Run predictions for all cards, write results to DB |
| POST | `/backtest` | Evaluate model against historical training data |

## How it works

1. Reads card data and price history from PostgreSQL (matching the Prisma schema)
2. Converts to the dict format expected by `lib/features.py`, `lib/spike.py`, etc.
3. Runs the existing training/prediction/backtesting pipeline
4. Writes results (model blobs, predictions, metrics) back to PostgreSQL

Model files are stored as binary blobs in the `Model` table rather than on the filesystem.

## Running

### Via Docker (recommended)

The service is included in `saas/docker-compose.yml`:

```bash
cd saas && docker compose up ml-service
```

### Standalone

```bash
cd ml-service
pip install -r requirements.txt
DATABASE_URL="postgresql://tcgplayer:tcgplayer@localhost:5432/tcgplayer" uvicorn main:app --host 0.0.0.0 --port 8000
```

## Dependencies

- fastapi, uvicorn -- web framework
- psycopg2-binary, sqlalchemy -- database access
- pandas, numpy, scikit-learn, xgboost, pyarrow -- ML pipeline (shared with `lib/`)

## Docker

The Dockerfile builds from the project root to access both `ml-service/` and `lib/`:

```
COPY ml-service/ /app/ml-service/
COPY lib/ /app/lib/
```
