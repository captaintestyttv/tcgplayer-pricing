"""Parquet-based persistent price history store.

Accumulates price data across syncs instead of replacing it.
Storage layout: {PRICE_HISTORY_DIR}/{price_type}/{id_prefix}/{card_id}.parquet
where id_prefix = first 2 chars of card_id for filesystem sharding.
"""

import os

import pyarrow as pa
import pyarrow.parquet as pq

from lib.config import PRICE_HISTORY_DIR, get_logger

log = get_logger(__name__)

SCHEMA = pa.schema([
    ("date", pa.string()),
    ("price", pa.float64()),
    ("source", pa.string()),
])


def _parquet_path(card_id: str, price_type: str = "normal") -> str:
    prefix = card_id[:2] if len(card_id) >= 2 else card_id
    return os.path.join(PRICE_HISTORY_DIR, price_type, prefix, f"{card_id}.parquet")


def merge_price_dicts(old: dict[str, float], new: dict[str, float]) -> dict[str, float]:
    """Combine two {date: price} dicts. New wins on conflict."""
    merged = dict(old)
    merged.update(new)
    return merged


def save_prices(
    card_id: str,
    prices: dict[str, float],
    source: str,
    price_type: str = "normal",
) -> None:
    """Append/merge new prices into the Parquet file for this card."""
    if not prices:
        return

    path = _parquet_path(card_id, price_type)

    existing = _read_parquet(path)

    # Merge: new prices override existing for same date
    merged = dict(existing)
    for date_str, price in prices.items():
        merged[date_str] = (price, source)

    # For existing entries that weren't overridden, keep their source
    for date_str, val in existing.items():
        if date_str not in prices:
            merged[date_str] = val

    dates = sorted(merged.keys())
    rows = {
        "date": dates,
        "price": [merged[d][0] if isinstance(merged[d], tuple) else merged[d] for d in dates],
        "source": [merged[d][1] if isinstance(merged[d], tuple) else source for d in dates],
    }

    table = pa.table(rows, schema=SCHEMA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path)
    log.debug("Saved %d price entries for %s/%s", len(dates), price_type, card_id)


def load_prices(card_id: str, price_type: str = "normal") -> dict[str, float]:
    """Load merged price history for a card. Returns {date: price}."""
    path = _parquet_path(card_id, price_type)
    data = _read_parquet(path)
    # Strip source info, return just {date: price}
    result = {}
    for date_str, val in data.items():
        if isinstance(val, tuple):
            result[date_str] = val[0]
        else:
            result[date_str] = val
    return result


def _read_parquet(path: str) -> dict:
    """Read parquet file into {date: (price, source)} dict."""
    if not os.path.exists(path):
        return {}
    try:
        table = pq.read_table(path)
        dates = table.column("date").to_pylist()
        prices = table.column("price").to_pylist()
        sources = table.column("source").to_pylist()
        return {d: (p, s) for d, p, s in zip(dates, prices, sources)}
    except Exception:
        log.warning("Failed to read %s, returning empty", path)
        return {}
