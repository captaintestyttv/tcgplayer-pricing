# TCGPlayer Pricing -- SaaS Frontend

Next.js SaaS scaffold for the TCGPlayer pricing tool. Replaces the local Flask UI with a production-oriented stack.

## Architecture

| Layer | Technology |
|---|---|
| Frontend | Next.js 14 (App Router), TypeScript, Tailwind CSS, shadcn/ui |
| API | tRPC v11 with superjson transformer |
| Database | PostgreSQL 16 + TimescaleDB (hypertables with compression) |
| ORM | Prisma v7 with `@prisma/adapter-pg` driver adapter |
| Auth | NextAuth v4 (JWT strategy, credentials provider) |
| Jobs | pg-boss v10 (PostgreSQL-backed task queue) |
| ML | FastAPI microservice (see `../ml-service/`) |
| Charts | Recharts (LineChart for price history) |

## Pages

| Route | Description |
|---|---|
| `/` | Dashboard with 4 stat cards (inventory, avg price, predictions, watchlist) |
| `/predictions` | Predictions table with search and signal filter |
| `/watchlist` | HOLD signal cards sorted by spike probability |
| `/cards` | Card search with pagination |
| `/cards/[uuid]` | Card detail with tabbed price charts (normal, foil, buylist) |
| `/backtest` | Model evaluation metrics and confusion matrix |
| `/import` | CSV file upload for inventory import |
| `/settings` | ML pipeline controls (train, predict, backtest triggers) |

## tRPC Routers

| Router | Endpoints |
|---|---|
| `cards` | `search`, `getById`, `list` |
| `prices` | `getHistory`, `getLatest`, `getBulk` |
| `predictions` | `list`, `watchlist`, `getByCard` |
| `inventory` | `list`, `import` |
| `ml` | `status`, `triggerTrain`, `triggerPredict`, `triggerBacktest` |

## Database Schema

10 Prisma models: `Card`, `CardTcgplayerId`, `PriceHistory` (TimescaleDB hypertable), `UserInventory`, `Prediction`, `Model`, `Account`, `Session`, `User`, `VerificationToken`.

See `prisma/schema.prisma` for full schema and `prisma/timescaledb-setup.sql` for hypertable/index setup.

## Local Development

### Prerequisites

- Docker (for TimescaleDB)
- Node.js 18+
- The parent project's Python environment (for the ML service)

### Setup

```bash
# Start database + ML service
docker compose up -d

# Install dependencies
npm install

# Generate Prisma client
npx prisma generate

# Push schema to database
npx prisma db push

# Apply TimescaleDB extensions
docker exec -i saas-db-1 psql -U tcgplayer < prisma/timescaledb-setup.sql

# Migrate existing data from JSON/Parquet to PostgreSQL
npx tsx scripts/migrate-data.ts

# Start dev server
npm run dev
```

The app runs at `http://localhost:3000`.

### Environment Variables

Defined in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql://tcgplayer:tcgplayer@localhost:5432/tcgplayer` | PostgreSQL connection |
| `NEXTAUTH_SECRET` | `dev-secret-change-in-production` | JWT signing secret |
| `NEXTAUTH_URL` | `http://localhost:3000` | Auth callback URL |
| `ML_SERVICE_URL` | `http://localhost:8000` | FastAPI ML service URL |

### Docker Services

- **db** -- TimescaleDB (PostgreSQL 16) on port 5432
- **ml-service** -- FastAPI ML microservice on port 8000 (builds from `../ml-service/Dockerfile`)

## Deployment

`vercel.json` is included for Vercel deployment. The database and ML service need separate hosting (e.g., Railway, Render, or a VPS with Docker).

## Status

This is a scaffold / proof-of-concept. The core architecture is in place but not yet production-tested. The local CLI + Flask UI remains the primary interface.
