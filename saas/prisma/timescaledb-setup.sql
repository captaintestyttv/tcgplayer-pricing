-- Run after Prisma migration to set up TimescaleDB extensions and hypertables.
-- Usage: psql -U tcgplayer -d tcgplayer -f prisma/timescaledb-setup.sql

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Convert price_history to a hypertable partitioned by date
SELECT create_hypertable('price_history', 'date', if_not_exists => TRUE);

-- Compression policy: compress chunks older than 7 days
ALTER TABLE price_history SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'card_uuid,price_type,source'
);

SELECT add_compression_policy('price_history', INTERVAL '7 days', if_not_exists => TRUE);

-- Useful indexes
CREATE INDEX IF NOT EXISTS idx_price_history_card_date
  ON price_history (card_uuid, date DESC);

CREATE INDEX IF NOT EXISTS idx_cards_set_code
  ON cards (set_code);

CREATE INDEX IF NOT EXISTS idx_predictions_user_run
  ON predictions (user_id, run_at DESC);
