#!/usr/bin/env python
"""Download MTGGoldfish Premium price history CSVs and import into the Parquet price store.

Usage:
    python scripts/goldfish_import.py --cookie "SESSION=..." --cards inventory
    python scripts/goldfish_import.py --cookie "SESSION=..." --cards all
    python scripts/goldfish_import.py --import-only DIR  # skip download, just import CSVs

Requires an active MTGGoldfish Premium subscription.
The session cookie can be obtained from browser dev tools after logging in.
"""

import argparse
import json
import os
import sys
import time
import urllib.parse

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRICING_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PRICING_DIR)

from lib.config import get_logger
from lib.goldfish import parse_goldfish_csv, match_goldfish_to_uuid, import_goldfish_dir
from lib.mtgjson import load_json_file, load_inventory_cache, load_training_cache
from lib.price_store import save_prices

log = get_logger(__name__)

GOLDFISH_BASE = "https://www.mtggoldfish.com"
GOLDFISH_RAW_DIR = os.path.join(PRICING_DIR, "data", "goldfish_raw")
PROGRESS_FILE = os.path.join(PRICING_DIR, "data", "goldfish_progress.json")
RATE_LIMIT_SECS = 2


def load_progress() -> set:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return set(json.load(f))
    return set()


def save_progress(completed: set) -> None:
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(sorted(completed), f)


def build_card_list(mode: str, data_dir: str) -> list[dict]:
    """Build list of cards to download from MTGGoldfish.

    Each entry: {"name": str, "set_code": str, "card_id": str}
    """
    if mode == "inventory":
        cache = load_inventory_cache(data_dir)
    else:
        cache = load_training_cache(data_dir)

    cards = []
    for card_id, card in cache.items():
        name = card.get("name", "")
        set_code = card.get("setCode", "")
        if name and set_code:
            cards.append({
                "name": name,
                "set_code": set_code,
                "card_id": card_id,
            })
    return cards


def download_goldfish_csv(
    card_name: str,
    set_name: str,
    session_cookie: str,
    dest_dir: str,
) -> str | None:
    """Download a single card's price history CSV from MTGGoldfish.

    Returns the path to the saved CSV, or None on failure.
    URL format: /price-download/paper/{CardName+[SET]}  (quote_plus encoded)
    """
    encoded = urllib.parse.quote_plus(f"{card_name} [{set_name}]")
    url = f"{GOLDFISH_BASE}/price-download/paper/{encoded}"

    headers = {
        "Cookie": session_cookie,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        if resp.status_code == 404:
            log.warning("Not found: %s [%s]", card_name, set_name)
            return None
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "text/csv" not in content_type and "text/plain" not in content_type:
            # Check if we got HTML (redirect to login, etc.)
            if "text/html" in content_type or "<html" in resp.text[:200].lower():
                log.warning("Got HTML instead of CSV for %s [%s] — session may be expired",
                            card_name, set_name)
                return None
            log.warning("Unexpected content type for %s: %s",
                        card_name, content_type)
            return None

        safe_name = card_name.replace("/", "_").replace("\\", "_")
        filename = f"{safe_name}_{set_name}.csv"
        filepath = os.path.join(dest_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(resp.text)
        return filepath

    except requests.RequestException as e:
        log.error("Download failed for %s [%s]: %s", card_name, set_name, e)
        return None


def main():
    parser = argparse.ArgumentParser(description="MTGGoldfish price history importer")
    parser.add_argument("--cookie", help="MTGGoldfish session cookie string")
    parser.add_argument("--cards", choices=["inventory", "all"], default="inventory",
                        help="Which cards to download (default: inventory)")
    parser.add_argument("--import-only", metavar="DIR",
                        help="Skip download, just import CSVs from this directory")
    parser.add_argument("--data-dir", default=os.path.join(PRICING_DIR, "data", "mtgjson"),
                        help="Path to MTGJson data directory")
    args = parser.parse_args()

    if args.import_only:
        print(f"Importing CSVs from {args.import_only}...")
        identifiers_path = os.path.join(args.data_dir, "AllIdentifiers.json")
        if not os.path.exists(identifiers_path):
            print("Error: AllIdentifiers.json not found. Run sync first.")
            sys.exit(1)
        identifiers = load_json_file(identifiers_path)["data"]
        results = import_goldfish_dir(args.import_only, identifiers)
        matched = sum(1 for v in results.values() if v is not None)
        print(f"Imported {matched}/{len(results)} files successfully")
        return

    if not args.cookie:
        print("Error: --cookie is required for downloading. Use --import-only to skip.")
        sys.exit(1)

    data_dir = args.data_dir
    cards = build_card_list(args.cards, data_dir)
    if not cards:
        print("No cards found. Run sync first.")
        sys.exit(1)

    os.makedirs(GOLDFISH_RAW_DIR, exist_ok=True)
    completed = load_progress()
    remaining = [c for c in cards if c["card_id"] not in completed]

    print(f"Cards to download: {len(remaining)} (skipping {len(completed)} already done)")

    identifiers_path = os.path.join(data_dir, "AllIdentifiers.json")
    identifiers = load_json_file(identifiers_path)["data"]

    for i, card in enumerate(remaining):
        print(f"[{i+1}/{len(remaining)}] {card['name']} [{card['set_code']}]")

        filepath = download_goldfish_csv(
            card["name"], card["set_code"],
            args.cookie, GOLDFISH_RAW_DIR,
        )

        if filepath:
            entries = parse_goldfish_csv(filepath)
            if entries:
                uuid = match_goldfish_to_uuid(card["name"], card["set_code"], identifiers)
                if uuid:
                    prices = {date: price for date, price in entries}
                    save_prices(uuid, prices, source="mtggoldfish", price_type="normal")
                    print(f"  -> Imported {len(entries)} prices")
                else:
                    print(f"  -> Downloaded but no UUID match")
            else:
                print(f"  -> Empty CSV")
        else:
            print(f"  -> Download failed")

        completed.add(card["card_id"])
        save_progress(completed)
        time.sleep(RATE_LIMIT_SECS)

    print(f"\nDone! {len(completed)} cards processed.")


if __name__ == "__main__":
    main()
