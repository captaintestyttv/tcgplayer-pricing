"""Database helper functions for the ML microservice.

Provides functions to load card data, inventory, models, and predictions
from PostgreSQL using SQLAlchemy. Connection string is read from the
DATABASE_URL environment variable.

Table structure matches the Prisma schema in saas/prisma/schema.prisma.
"""

import json
import os
from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

_engine: Engine | None = None


def get_engine() -> Engine:
    """Get or create a SQLAlchemy engine from DATABASE_URL."""
    global _engine
    if _engine is None:
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL environment variable is not set. "
                "Expected format: postgresql://user:pass@host:port/dbname"
            )
        _engine = create_engine(database_url, pool_pre_ping=True)
    return _engine


def get_cards_with_prices() -> dict[str, dict]:
    """Load all cards with their price history from the database.

    Returns a dict keyed by card UUID, with each value matching the format
    expected by lib/features.py extract_features().
    """
    engine = get_engine()
    cards: dict[str, dict] = {}

    with engine.connect() as conn:
        card_rows = conn.execute(text("""
            SELECT uuid, name, set_code, rarity, mana_value,
                   color_identity, types, subtypes, supertypes, keywords,
                   legalities, edhrec_rank, edhrec_saltiness, is_reserved,
                   printings, set_release_date
            FROM cards
        """)).fetchall()

        for row in card_rows:
            cards[row.uuid] = {
                "uuid": row.uuid,
                "name": row.name or "",
                "price_history": {},
                "foil_price_history": {},
                "buylist_price_history": {},
                "rarity": row.rarity or "",
                "printings": row.printings or [],
                "legalities": row.legalities or {},
                "edhrecRank": row.edhrec_rank,
                "edhrecSaltiness": row.edhrec_saltiness or 0.0,
                "isReserved": bool(row.is_reserved),
                "supertypes": row.supertypes or [],
                "types": row.types or [],
                "subtypes": row.subtypes or [],
                "colorIdentity": row.color_identity or [],
                "keywords": row.keywords or [],
                "manaValue": float(row.mana_value or 0),
                "setCode": row.set_code or "",
                "setReleaseDate": str(row.set_release_date) if row.set_release_date else "",
                "setIsPartialPreview": False,
                "recently_reprinted": 0,
                "legality_changed": 0,
            }

        # Load price histories in bulk
        price_rows = conn.execute(text("""
            SELECT card_uuid, date, price_type, price
            FROM price_history
            ORDER BY card_uuid, date
        """)).fetchall()

        field_map = {
            "normal": "price_history",
            "foil": "foil_price_history",
            "buylist": "buylist_price_history",
        }

        for row in price_rows:
            if row.card_uuid not in cards:
                continue
            field = field_map.get(row.price_type, "price_history")
            date_str = str(row.date)
            cards[row.card_uuid][field][date_str] = float(row.price)

    return cards


def get_user_inventory(user_id: str) -> dict[str, dict]:
    """Load a user's inventory items with card data.

    Returns a dict keyed by tcgplayer_id with inventory and card metadata merged.
    """
    engine = get_engine()
    cards: dict[str, dict] = {}

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                ui.tcgplayer_id,
                ui.quantity,
                ui.listed_price,
                ct.card_uuid,
                c.name, c.rarity, c.printings, c.legalities,
                c.edhrec_rank, c.edhrec_saltiness, c.is_reserved,
                c.supertypes, c.types, c.subtypes, c.color_identity,
                c.keywords, c.mana_value, c.set_code, c.set_release_date
            FROM user_inventory ui
            JOIN card_tcgplayer_ids ct ON ct.tcgplayer_id = ui.tcgplayer_id
            JOIN cards c ON c.uuid = ct.card_uuid
            WHERE ui.user_id = :user_id
              AND ui.quantity > 0
        """), {"user_id": user_id}).fetchall()

        card_uuids = []
        for row in rows:
            tcg_id = str(row.tcgplayer_id)
            card_uuids.append(row.card_uuid)
            cards[tcg_id] = {
                "uuid": row.card_uuid,
                "product_name": row.name or "",
                "quantity": int(row.quantity or 0),
                "marketplace_price": float(row.listed_price or 0),
                "market_price": 0.0,
                "price_history": {},
                "foil_price_history": {},
                "buylist_price_history": {},
                "rarity": row.rarity or "",
                "printings": row.printings or [],
                "legalities": row.legalities or {},
                "edhrecRank": row.edhrec_rank,
                "edhrecSaltiness": row.edhrec_saltiness or 0.0,
                "isReserved": bool(row.is_reserved),
                "supertypes": row.supertypes or [],
                "types": row.types or [],
                "subtypes": row.subtypes or [],
                "colorIdentity": row.color_identity or [],
                "keywords": row.keywords or [],
                "manaValue": float(row.mana_value or 0),
                "setCode": row.set_code or "",
                "setReleaseDate": str(row.set_release_date) if row.set_release_date else "",
                "setIsPartialPreview": False,
                "recently_reprinted": 0,
                "legality_changed": 0,
            }

        # Load price histories
        if card_uuids:
            # Build tcg_id -> card_uuid mapping for price lookup
            uuid_to_tcg = {row.card_uuid: str(row.tcgplayer_id) for row in rows}
            price_rows = conn.execute(text("""
                SELECT card_uuid, date, price_type, price
                FROM price_history
                WHERE card_uuid = ANY(:uuids)
                ORDER BY card_uuid, date
            """), {"uuids": card_uuids}).fetchall()

            field_map = {
                "normal": "price_history",
                "foil": "foil_price_history",
                "buylist": "buylist_price_history",
            }

            for row in price_rows:
                tcg_id = uuid_to_tcg.get(row.card_uuid)
                if tcg_id and tcg_id in cards:
                    field = field_map.get(row.price_type, "price_history")
                    cards[tcg_id][field][str(row.date)] = float(row.price)

    return cards


def save_model(model_bytes: bytes, metadata: dict) -> int:
    """Insert a trained model blob and metadata into the models table."""
    engine = get_engine()

    val_metrics = metadata.get("validation_metrics", {})

    with engine.connect() as conn:
        result = conn.execute(text("""
            INSERT INTO models (
                trained_at, num_samples, spike_rate, feature_cols,
                val_accuracy, val_auc, val_precision, val_recall,
                hyperparameters, feature_importance, model_blob
            ) VALUES (
                :trained_at, :num_samples, :spike_rate, :feature_cols,
                :val_accuracy, :val_auc, :val_precision, :val_recall,
                :hyperparameters, :feature_importance, :model_blob
            )
            RETURNING id
        """), {
            "trained_at": metadata.get("trained_at", datetime.utcnow().isoformat()),
            "num_samples": metadata.get("num_samples", 0),
            "spike_rate": metadata.get("spike_rate", 0.0),
            "feature_cols": metadata.get("feature_cols", []),
            "val_accuracy": val_metrics.get("accuracy"),
            "val_auc": val_metrics.get("auc"),
            "val_precision": val_metrics.get("precision"),
            "val_recall": val_metrics.get("recall"),
            "hyperparameters": json.dumps(metadata.get("hyperparameters", {})),
            "feature_importance": json.dumps(metadata.get("feature_importance", {})),
            "model_blob": model_bytes,
        })
        model_id = result.scalar_one()
        conn.commit()
    return model_id


def load_latest_model() -> tuple[bytes, dict] | None:
    """Load the most recent model from the models table."""
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id, trained_at, num_samples, spike_rate, feature_cols,
                   val_accuracy, val_auc, val_precision, val_recall,
                   hyperparameters, feature_importance, model_blob
            FROM models
            ORDER BY trained_at DESC
            LIMIT 1
        """)).fetchone()

    if row is None:
        return None

    metadata = {
        "model_id": row.id,
        "trained_at": str(row.trained_at),
        "num_samples": row.num_samples,
        "spike_rate": row.spike_rate,
        "feature_cols": row.feature_cols or [],
        "validation_metrics": {
            "accuracy": row.val_accuracy,
            "auc": row.val_auc,
            "precision": row.val_precision,
            "recall": row.val_recall,
        },
        "hyperparameters": row.hyperparameters or {},
        "feature_importance": row.feature_importance or {},
    }
    return bytes(row.model_blob), metadata


def save_predictions(user_id: str, predictions: list[dict]) -> int:
    """Batch insert predictions for a user."""
    if not predictions:
        return 0

    engine = get_engine()
    now = datetime.utcnow()

    with engine.connect() as conn:
        for pred in predictions:
            conn.execute(text("""
                INSERT INTO predictions (
                    user_id, tcgplayer_id, run_at,
                    spike_probability, predicted_7d, predicted_30d,
                    trend, signal, action, suggested_price
                ) VALUES (
                    :user_id, :tcgplayer_id, :run_at,
                    :spike_prob, :pred_7d, :pred_30d,
                    :trend, :signal, :action, :suggested_price
                )
            """), {
                "user_id": user_id,
                "tcgplayer_id": pred.get("TCGplayer Id", ""),
                "run_at": now,
                "spike_prob": _to_float(pred.get("Spike Probability")),
                "pred_7d": _to_float(pred.get("Predicted 7d")),
                "pred_30d": _to_float(pred.get("Predicted 30d")),
                "trend": pred.get("Trend", ""),
                "signal": pred.get("Signal", ""),
                "action": pred.get("Action", ""),
                "suggested_price": _to_float(pred.get("Suggested Price")),
            })
        conn.commit()

    return len(predictions)


def _to_float(value: Any) -> float | None:
    """Safely convert a value to float, returning None for non-numeric strings."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
