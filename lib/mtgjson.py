import csv
import gzip
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from pathlib import Path

import requests

from lib.config import (
    DOWNLOAD_TIMEOUT, DOWNLOAD_MAX_RETRIES, DOWNLOAD_BACKOFF_BASE,
    SPOILER_WINDOW_DAYS, TRAINING_CACHE_FILENAME, get_logger,
)

log = get_logger(__name__)

MTGJSON_BASE = "https://mtgjson.com/api/v5"
CACHE_FILENAME = "inventory_cards.json"


def download_json(url: str, dest_path: str, force: bool = False) -> None:
    """Download a JSON file with retry and atomic write."""
    if not force and os.path.exists(dest_path):
        print(f"  -> {os.path.basename(dest_path)} already exists, skipping")
        return

    tmp_path = dest_path + ".tmp"
    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        try:
            print(f"Downloading {url} ...")
            response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            os.rename(tmp_path, dest_path)
            print(f"  -> saved to {dest_path}")
            return
        except (requests.RequestException, OSError) as exc:
            log.warning("Download attempt %d/%d failed: %s", attempt, DOWNLOAD_MAX_RETRIES, exc)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if attempt < DOWNLOAD_MAX_RETRIES:
                delay = DOWNLOAD_BACKOFF_BASE ** attempt
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise


def load_json_file(path: str) -> dict:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_inventory_ids(history_dir: str) -> set[str]:
    """Extract TCGplayer IDs from latest.csv."""
    latest = os.path.join(history_dir, "latest.csv")
    if not os.path.exists(latest):
        return set()
    ids = set()
    with open(latest, newline="") as f:
        for row in csv.DictReader(f):
            tid = row.get("TCGplayer Id", "").strip()
            if tid:
                ids.add(tid)
    return ids


def build_sku_to_uuid(skus_data: dict) -> dict:
    """Build skuId -> uuid mapping from TcgplayerSkus data."""
    mapping = {}
    for uuid, skus in skus_data.items():
        for sku in skus:
            sku_id = sku.get("skuId")
            if sku_id:
                mapping[str(sku_id)] = uuid
    return mapping


def load_set_list(data_dir: str) -> dict:
    """Load SetList.json and return {setCode: {releaseDate, isPartialPreview}}."""
    path = os.path.join(data_dir, "SetList.json")
    if not os.path.exists(path):
        return {}
    raw = load_json_file(path)
    sets = raw.get("data") or raw  # handle both {data: [...]} and [...]
    if isinstance(sets, dict):
        sets = list(sets.values()) if not isinstance(list(sets.values())[0] if sets else None, dict) else list(sets.values())
    result = {}
    for s in (sets if isinstance(sets, list) else []):
        code = s.get("code", "")
        if code:
            result[code.upper()] = {
                "releaseDate": s.get("releaseDate", ""),
                "isPartialPreview": s.get("isPartialPreview", False),
            }
    return result


def build_inventory_cache(
    inventory_ids: set[str],
    identifiers_data: dict,
    prices_data: dict,
    sku_to_uuid: dict,
    set_data: dict | None = None,
) -> dict:
    """Build lean cache from MTGJson data scoped to inventory IDs."""
    if set_data is None:
        set_data = {}
    cache = {}
    for sku_id in inventory_ids:
        uuid = sku_to_uuid.get(sku_id)
        if not uuid:
            continue
        card = identifiers_data.get(uuid)
        if not card:
            continue

        price_history = {}
        try:
            normal = prices_data[uuid]["paper"]["tcgplayer"]["retail"]["normal"]
            price_history = {k: float(v) for k, v in normal.items()}
        except (KeyError, TypeError):
            log.debug("No TCGPlayer retail prices for sku %s (uuid %s)", sku_id, uuid)

        foil_price_history = {}
        try:
            foil = prices_data[uuid]["paper"]["tcgplayer"]["retail"]["foil"]
            foil_price_history = {k: float(v) for k, v in foil.items()}
        except (KeyError, TypeError):
            pass

        buylist_price_history = {}
        try:
            buylist = prices_data[uuid]["paper"]["tcgplayer"]["buylist"]["normal"]
            buylist_price_history = {k: float(v) for k, v in buylist.items()}
        except (KeyError, TypeError):
            pass

        cache[sku_id] = {
            "uuid": uuid,
            "name": card.get("name", ""),
            "rarity": card.get("rarity", ""),
            "setCode": card.get("setCode", ""),
            "printings": card.get("printings", []),
            "legalities": {
                k: v.lower() for k, v in card.get("legalities", {}).items()
            },
            "price_history": price_history,
            # Phase 1: card metadata
            "edhrecRank": card.get("edhrecRank"),
            "edhrecSaltiness": card.get("edhrecSaltiness"),
            "isReserved": card.get("isReserved", False),
            "supertypes": card.get("supertypes", []),
            "types": card.get("types", []),
            "subtypes": card.get("subtypes", []),
            "colorIdentity": card.get("colorIdentity", []),
            "keywords": card.get("keywords", []),
            "manaValue": card.get("manaValue", 0),
            "text": card.get("text", ""),
            # Phase 2: foil & buylist price channels
            "foil_price_history": foil_price_history,
            "buylist_price_history": buylist_price_history,
            # Phase 4: change flags (set by detect_changes after build)
            "recently_reprinted": 0,
            "legality_changed": 0,
            # Phase 5: set release data
            "setReleaseDate": set_data.get(card.get("setCode", "").upper(), {}).get("releaseDate", ""),
            "setIsPartialPreview": set_data.get(card.get("setCode", "").upper(), {}).get("isPartialPreview", False),
        }
    return cache


def build_training_cache(
    identifiers_data: dict,
    prices_data: dict,
    set_data: dict | None = None,
    max_cards: int = 0,
) -> dict:
    """Build training cache from ALL cards with TCGPlayer retail prices.

    Unlike build_inventory_cache (scoped to user's inventory via SKU mapping),
    this iterates all UUIDs in AllPrices.json that have paper/tcgplayer/retail
    data. Keyed by UUID (not TCGPlayer ID).
    """
    if set_data is None:
        set_data = {}
    cache = {}
    count = 0

    for uuid, price_channels in prices_data.items():
        try:
            normal = price_channels["paper"]["tcgplayer"]["retail"]["normal"]
            price_history = {k: float(v) for k, v in normal.items()}
        except (KeyError, TypeError):
            continue

        if not price_history:
            continue

        card = identifiers_data.get(uuid)
        if not card:
            continue

        foil_price_history = {}
        try:
            foil = price_channels["paper"]["tcgplayer"]["retail"]["foil"]
            foil_price_history = {k: float(v) for k, v in foil.items()}
        except (KeyError, TypeError):
            pass

        buylist_price_history = {}
        try:
            buylist = price_channels["paper"]["tcgplayer"]["buylist"]["normal"]
            buylist_price_history = {k: float(v) for k, v in buylist.items()}
        except (KeyError, TypeError):
            pass

        set_code = card.get("setCode", "").upper()
        cache[uuid] = {
            "uuid": uuid,
            "name": card.get("name", ""),
            "rarity": card.get("rarity", ""),
            "setCode": card.get("setCode", ""),
            "printings": card.get("printings", []),
            "legalities": {
                k: v.lower() for k, v in card.get("legalities", {}).items()
            },
            "price_history": price_history,
            "edhrecRank": card.get("edhrecRank"),
            "edhrecSaltiness": card.get("edhrecSaltiness"),
            "isReserved": card.get("isReserved", False),
            "supertypes": card.get("supertypes", []),
            "types": card.get("types", []),
            "subtypes": card.get("subtypes", []),
            "colorIdentity": card.get("colorIdentity", []),
            "keywords": card.get("keywords", []),
            "manaValue": card.get("manaValue", 0),
            "text": card.get("text", ""),
            "foil_price_history": foil_price_history,
            "buylist_price_history": buylist_price_history,
            "recently_reprinted": 0,
            "legality_changed": 0,
            "setReleaseDate": set_data.get(set_code, {}).get("releaseDate", ""),
            "setIsPartialPreview": set_data.get(set_code, {}).get("isPartialPreview", False),
        }

        count += 1
        if max_cards and count >= max_cards:
            break

    return cache


def load_training_cache(data_dir: str) -> dict:
    """Load the full-universe training cache."""
    path = os.path.join(data_dir, TRAINING_CACHE_FILENAME)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def detect_changes(old_cache: dict, new_cache: dict) -> None:
    """Compare old and new caches, set change flags on new_cache in-place."""
    for tid, new_card in new_cache.items():
        old_card = old_cache.get(tid, {})
        new_card["recently_reprinted"] = int(
            len(new_card.get("printings", [])) > len(old_card.get("printings", []))
        )
        new_card["legality_changed"] = int(
            new_card.get("legalities", {}) != old_card.get("legalities", {})
        )


def load_inventory_cache(data_dir: str) -> dict:
    path = os.path.join(data_dir, CACHE_FILENAME)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def sync(
    history_dir: str,
    data_dir: str,
    force: bool = False,
    cache_only: bool = False,
    files: list[str] | None = None,
) -> None:
    """Download MTGJson data and build inventory_cards.json.

    flags:
      force      -- re-download even if files already exist
      cache_only -- skip all downloads, just rebuild the cache
      files      -- list of filenames to (re-)download, e.g. ["AllPrices.json"]
                    overrides force; other files are skipped if they exist
    """
    os.makedirs(data_dir, exist_ok=True)

    identifiers_path = os.path.join(data_dir, "AllIdentifiers.json")
    prices_path = os.path.join(data_dir, "AllPrices.json")
    skus_path = os.path.join(data_dir, "TcgplayerSkus.json")
    setlist_path = os.path.join(data_dir, "SetList.json")

    if not cache_only:
        # Determine per-file force flag
        def should_force(filename: str) -> bool:
            if files is not None:
                return filename in files
            return force

        downloads = [
            (f"{MTGJSON_BASE}/AllIdentifiers.json", identifiers_path, should_force("AllIdentifiers.json")),
            (f"{MTGJSON_BASE}/AllPrices.json", prices_path, should_force("AllPrices.json")),
            (f"{MTGJSON_BASE}/TcgplayerSkus.json", skus_path, should_force("TcgplayerSkus.json")),
            (f"{MTGJSON_BASE}/SetList.json", setlist_path, should_force("SetList.json")),
        ]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(download_json, url, dest, f): dest for url, dest, f in downloads}
            for future in as_completed(futures):
                future.result()  # re-raises any download exception

    for path, label in [(identifiers_path, "AllIdentifiers.json"),
                        (prices_path, "AllPrices.json"),
                        (skus_path, "TcgplayerSkus.json")]:
        if not os.path.exists(path):
            print(f"Error: {label} not found. Run sync without --cache to download it.")
            sys.exit(1)

    print("Building inventory cache...")
    inventory_ids = read_inventory_ids(history_dir)
    if not inventory_ids:
        print("No inventory IDs found in latest.csv -- run import first.")
        sys.exit(1)

    # Load JSON files in parallel (AllPrices + AllIdentifiers are ~650MB combined)
    with ThreadPoolExecutor(max_workers=3) as pool:
        id_future = pool.submit(lambda: load_json_file(identifiers_path)["data"])
        prices_future = pool.submit(lambda: load_json_file(prices_path)["data"])
        skus_future = pool.submit(lambda: load_json_file(skus_path)["data"])
        identifiers_data = id_future.result()
        prices_data = prices_future.result()
        skus_data = skus_future.result()
    sku_to_uuid = build_sku_to_uuid(skus_data)

    set_data = load_set_list(data_dir)

    old_cache = load_inventory_cache(data_dir)
    cache = build_inventory_cache(inventory_ids, identifiers_data, prices_data, sku_to_uuid, set_data)
    if old_cache:
        detect_changes(old_cache, cache)

    cache_path = os.path.join(data_dir, CACHE_FILENAME)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"Cached {len(cache)} cards -> {cache_path}")
    skipped = len(inventory_ids) - len(cache)
    if skipped:
        print(f"   {skipped} inventory IDs had no MTGJson match (sealed product, etc.)")

    # Build full-universe training cache
    from lib.config import TRAINING_CACHE_MAX_CARDS
    print("Building training cache (all cards with TCGPlayer prices)...")
    training_cache = build_training_cache(
        identifiers_data, prices_data, set_data,
        max_cards=TRAINING_CACHE_MAX_CARDS,
    )
    training_cache_path = os.path.join(data_dir, TRAINING_CACHE_FILENAME)
    with open(training_cache_path, "w") as f:
        json.dump(training_cache, f)
    print(f"Training cache: {len(training_cache)} cards -> {training_cache_path}")
