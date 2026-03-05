# Signal Expansion — Design Doc

## Status
Proposed — 2026-03-01

## Problem

The spike classifier uses 7 features derived from basic card metadata and price history. It has no visibility into the dominant demand drivers for Magic singles:

- **Commander popularity** — EDH is the #1 singles demand driver, currently invisible
- **Supply constraints** — Reserved List cards have hard supply caps, not modeled
- **Card mechanics** — keyword/tribal synergy cascades cause correlated spikes
- **Buylist signals** — stores raising buylist prices is a leading indicator of retail spikes
- **Event-driven shocks** — bans, reprints, and new set releases cause immediate price swings

## Data Sources

### Already Downloaded (AllIdentifiers.json)

These fields exist in the data but are not extracted into the inventory cache:

| Field | Type | Example |
|---|---|---|
| `edhrecRank` | int | 1523 (lower = more popular) |
| `edhrecSaltiness` | float | 2.31 (0-4 scale) |
| `isReserved` | bool | true/false |
| `supertypes` | string[] | ["Legendary"] |
| `types` | string[] | ["Creature", "Enchantment"] |
| `subtypes` | string[] | ["Elf", "Warrior"] |
| `colorIdentity` | string[] | ["G", "U"] |
| `keywords` | string[] | ["Flying", "Trample"] |
| `manaValue` | float | 3.0 |
| `text` | string | "When ~ enters..." |

### Already Downloaded (AllPrices.json)

Price channels available but unused:

| Channel | Path | Use |
|---|---|---|
| Foil retail | `paper.tcgplayer.retail.foil` | Collector demand signal |
| Normal buylist | `paper.tcgplayer.buylist.normal` | Store demand leading indicator |

### New Download Required

| File | Size | URL | Use |
|---|---|---|---|
| `SetList.json` | ~200KB | `mtgjson.com/api/v5/SetList.json` | Set release dates, spoiler season detection (`isPartialPreview`) |

### External (Not Recommended for Now)

- **EDHREC** — no public API; `edhrecRank` and `edhrecSaltiness` are already in MTGJson
- **Scryfall** — bulk data available (oracle text, related cards) but redundant with MTGJson for our needs
- **WotC News** — no RSS/API; ban detection better done by diffing `legalities` between syncs

## Implementation Phases

### Phase 1 — MTGJson Metadata Features

Expand the inventory cache and feature extractor to use fields already in AllIdentifiers.json.

#### Cache Changes (`lib/mtgjson.py`)

Add to `build_inventory_cache()` output per card:

```python
"edhrecRank": card.get("edhrecRank"),           # int or None
"edhrecSaltiness": card.get("edhrecSaltiness"),  # float or None
"isReserved": card.get("isReserved", False),     # bool
"supertypes": card.get("supertypes", []),        # list
"types": card.get("types", []),                  # list
"subtypes": card.get("subtypes", []),            # list
"colorIdentity": card.get("colorIdentity", []),  # list
"keywords": card.get("keywords", []),            # list
"manaValue": card.get("manaValue", 0),           # float
"text": card.get("text", ""),                    # str
```

#### New Features (`lib/features.py`)

| Feature | Derivation | Default |
|---|---|---|
| `edhrec_rank` | `card["edhrecRank"]` — lower = more popular. Normalize: use raw int, XGBoost handles scale. | 99999 if None |
| `edhrec_saltiness` | `card["edhrecSaltiness"]` | 0.0 if None |
| `is_reserved_list` | `int(card["isReserved"])` | 0 |
| `is_legendary` | `int("Legendary" in card["supertypes"])` | 0 |
| `is_creature` | `int("Creature" in card["types"])` | 0 |
| `color_count` | `len(card["colorIdentity"])` | 0 |
| `keyword_count` | `len(card["keywords"])` | 0 |
| `mana_value` | `card["manaValue"]` | 0.0 |
| `subtype_count` | `len(card["subtypes"])` | 0 |

**Feature count: 7 → 16**

#### Model Changes (`lib/spike.py`)

Update `FEATURE_COLS` to include the 9 new features. Retrain required.

#### Test Changes

- Update `tests/fixtures/inventory_cards.json` to include the new fields
- Add tests for new features (defaults, edge cases)
- Verify existing tests still pass with expanded cache schema

### Phase 2 — Foil & Buylist Price Signals

#### Cache Changes (`lib/mtgjson.py`)

Extract additional price channels in `build_inventory_cache()`:

```python
# Foil prices
foil_history = {}
try:
    foil = prices_data[uuid]["paper"]["tcgplayer"]["retail"]["foil"]
    foil_history = {k: float(v) for k, v in foil.items()}
except (KeyError, TypeError):
    pass

# Buylist prices
buylist_history = {}
try:
    buylist = prices_data[uuid]["paper"]["tcgplayer"]["buylist"]["normal"]
    buylist_history = {k: float(v) for k, v in buylist.items()}
except (KeyError, TypeError):
    pass
```

#### New Features (`lib/features.py`)

| Feature | Derivation |
|---|---|
| `foil_to_normal_ratio` | Latest foil price / latest normal price. Default 0.0 if no foil data. |
| `buylist_ratio` | Latest buylist price / latest normal price. Default 0.0 if no buylist. |
| `buylist_momentum_7d` | 7-day momentum of buylist prices (same calc as `price_momentum_7d`). |

**Feature count: 16 → 19**

### Phase 3 — Cross-Card Cluster Momentum

Detect when a tribal or mechanical group is moving together.

#### New Logic (`lib/features.py`)

After extracting individual card features, compute group-level aggregates:

```python
def compute_cluster_features(all_features: list[dict], cards: dict) -> None:
    """Mutate feature dicts in-place to add cluster momentum."""
    # Group by subtypes
    subtype_momentum = defaultdict(list)
    for feat, (tid, card) in zip(all_features, cards.items()):
        for st in card.get("subtypes", []):
            subtype_momentum[st].append(feat["price_momentum_7d"])

    subtype_avg = {st: sum(vals)/len(vals) for st, vals in subtype_momentum.items()}

    for feat, (tid, card) in zip(all_features, cards.items()):
        subtypes = card.get("subtypes", [])
        if subtypes:
            feat["cluster_momentum_7d"] = max(subtype_avg.get(st, 0) for st in subtypes)
        else:
            feat["cluster_momentum_7d"] = 0.0
```

| Feature | Derivation |
|---|---|
| `cluster_momentum_7d` | Max average 7-day momentum across all subtypes the card belongs to. Captures "Elves are spiking" signal. |

**Feature count: 19 → 20**

### Phase 4 — Reprint & Legality Change Detection

Detect reprints and bans by diffing successive cache builds.

#### New Logic (`lib/mtgjson.py`)

On cache rebuild, load previous cache and compare:

```python
def detect_changes(old_cache: dict, new_cache: dict) -> dict:
    """Return per-card change flags."""
    changes = {}
    for tid in new_cache:
        old = old_cache.get(tid, {})
        new = new_cache[tid]
        changes[tid] = {
            "reprinted": len(new.get("printings", [])) > len(old.get("printings", [])),
            "legality_changed": new.get("legalities", {}) != old.get("legalities", {}),
        }
    return changes
```

Save changes as `data/mtgjson/cache_changes.json`. Feature extractor reads this file if present.

| Feature | Derivation |
|---|---|
| `recently_reprinted` | 1 if printings count increased since last sync, 0 otherwise |
| `legality_changed` | 1 if any format legality changed since last sync, 0 otherwise |

**Feature count: 20 → 22**

### Phase 5 — Oracle Text Synergy (Future)

The most ambitious signal. Two approaches:

**Approach A — Keyword matching (simpler):**
- Extract action keywords from oracle text: "destroy", "exile", "draw", "token", "counter", "sacrifice", "mill"
- Store as a set per card
- When a new card appears in SetList with `isPartialPreview`, compute Jaccard similarity against all inventory cards
- Cards with similarity > 0.3 get a `synergy_score` boost

**Approach B — Embedding similarity (more powerful, requires `sentence-transformers`):**
- Encode oracle text with `all-MiniLM-L6-v2`
- Pre-compute embeddings for all cards in cache
- Cosine similarity between new spoiled cards and inventory
- Requires GPU for initial embedding, but inference is fast

This phase is deferred until Phases 1-4 prove value. It also requires downloading `SetList.json` to detect spoiler season.

## Migration

Each phase is backward-compatible:

1. New cache fields default gracefully (None/empty/0)
2. Models trained on old features still load and score (XGBoost handles missing columns)
3. New features require `sync --cache` to rebuild cache, then `train` to retrain model
4. Old models can be kept as fallback until new model is validated

## Risks

| Risk | Mitigation |
|---|---|
| Too many features cause overfitting | XGBoost handles high-dimensional sparse data well. Monitor train/validation accuracy gap in `_meta.json`. |
| `edhrecRank` is None for many cards | Default to 99999 (lowest popularity). XGBoost handles this as a feature split. |
| Buylist data sparse for cheap cards | Default ratio to 0.0. Only helps for cards where buylist data exists. |
| Cluster momentum is noisy for large subtypes | Use max (not mean) of subtype averages to amplify strong signals. |
| Cache size increases | Additional fields add ~30% to cache JSON size. Still under 20MB for typical inventory. |

## Verification

After each phase:
1. `python3 -m pytest tests/ -v` — all tests pass
2. `bash scripts/monitor.sh sync --cache` — cache rebuilds with new fields
3. `bash scripts/monitor.sh train` — model trains on expanded features
4. `bash scripts/monitor.sh predict --dry-run` — predictions produce reasonable output
5. Compare watchlist before/after — new signals should surface cards the old model missed
