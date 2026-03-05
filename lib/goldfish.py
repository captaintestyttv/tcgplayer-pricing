"""MTGGoldfish CSV parsing and card matching.

Parses price history CSVs downloaded from MTGGoldfish Premium
and matches them to MTGJson UUIDs for import into the Parquet price store.
"""

import csv
import os
from difflib import SequenceMatcher

from lib.config import get_logger

log = get_logger(__name__)

FUZZY_MATCH_THRESHOLD = 0.9


def parse_goldfish_csv(filepath: str) -> list[tuple[str, float]]:
    """Parse a MTGGoldfish price CSV into (date, price) pairs.

    MTGGoldfish CSVs have format:
        Date,Price
        2024-01-15,3.50
        2024-01-14,3.45
        ...
    """
    entries = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return entries
        for row in reader:
            if len(row) < 2:
                continue
            date_str = row[0].strip()
            try:
                price = float(row[1].strip())
            except (ValueError, IndexError):
                continue
            if price > 0:
                entries.append((date_str, price))
    return entries


def match_goldfish_to_uuid(
    card_name: str,
    set_name: str,
    identifiers: dict,
) -> str | None:
    """Match a MTGGoldfish card name + set to a MTGJson UUID.

    Tries exact match first (name + setCode), then fuzzy fallback.
    identifiers is the AllIdentifiers 'data' dict: {uuid: card_dict}.
    """
    card_name_lower = card_name.lower().strip()
    set_name_lower = set_name.lower().strip()

    best_uuid = None
    best_score = 0.0

    for uuid, card in identifiers.items():
        name = card.get("name", "").lower()
        set_code = card.get("setCode", "").lower()
        set_full = card.get("setName", "").lower() if "setName" in card else ""

        # Exact match on name + (set code or set name)
        if name == card_name_lower:
            if set_code == set_name_lower or set_full == set_name_lower:
                return uuid

        # Track best fuzzy match
        name_score = SequenceMatcher(None, card_name_lower, name).ratio()
        if name_score >= FUZZY_MATCH_THRESHOLD:
            set_score = max(
                SequenceMatcher(None, set_name_lower, set_code).ratio(),
                SequenceMatcher(None, set_name_lower, set_full).ratio(),
            )
            combined = name_score * 0.7 + set_score * 0.3
            if combined > best_score:
                best_score = combined
                best_uuid = uuid

    if best_uuid and best_score >= FUZZY_MATCH_THRESHOLD:
        log.info("Fuzzy matched '%s' [%s] -> %s (score=%.3f)",
                 card_name, set_name, best_uuid, best_score)
        return best_uuid

    log.warning("No match for '%s' [%s]", card_name, set_name)
    return None


def import_goldfish_dir(
    dir_path: str,
    identifiers: dict,
) -> dict:
    """Batch-import all MTGGoldfish CSVs from a directory.

    Expects filenames like: CardName_SetName.csv or CardName.csv
    Returns {filename: uuid_or_None} mapping for reporting.
    """
    from lib.price_store import save_prices

    results = {}
    if not os.path.isdir(dir_path):
        log.error("Directory not found: %s", dir_path)
        return results

    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".csv"):
            continue

        base = fname[:-4]  # strip .csv
        # Try to split CardName_SetName
        if "_" in base:
            parts = base.rsplit("_", 1)
            card_name, set_name = parts[0], parts[1]
        else:
            card_name = base
            set_name = ""

        filepath = os.path.join(dir_path, fname)
        entries = parse_goldfish_csv(filepath)
        if not entries:
            results[fname] = None
            continue

        uuid = match_goldfish_to_uuid(card_name, set_name, identifiers)
        if uuid:
            prices = {date: price for date, price in entries}
            save_prices(uuid, prices, source="mtggoldfish", price_type="normal")
            log.info("Imported %d prices for %s -> %s", len(prices), fname, uuid)

        results[fname] = uuid

    return results
