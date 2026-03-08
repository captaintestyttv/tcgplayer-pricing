"""Centralized configuration constants for the TCGPlayer pricing pipeline."""

import logging
import os

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING").upper()

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))
    return logger

# ---------------------------------------------------------------------------
# TCGPlayer Fee Constants
# ---------------------------------------------------------------------------
COMMISSION_FEE = 0.1075        # TCGPlayer seller commission
TRANSACTION_FEE = 0.025        # Payment processing percentage
TRANSACTION_FLAT = 0.30        # Payment processing flat fee ($)
SHIPPING_REVENUE = 1.31        # Shipping charged to buyer (cards < $5)
POSTAGE_STANDARD = 0.73        # Actual postage for cards < $5 ($)
POSTAGE_MEDIA_MAIL = 1.50      # Postage for cards >= $5 ($)
HIGH_VALUE_THRESHOLD = 5.00    # Cards >= this use media mail, no shipping revenue

# ---------------------------------------------------------------------------
# Pricing Logic Thresholds
# ---------------------------------------------------------------------------
MIN_MARGIN = 0.10              # Below this net margin -> RAISE
MARKET_UP_PCT = 0.10           # Market > current by this % -> RAISE
MARKET_DOWN_PCT = 0.10         # Market < current by this % -> LOWER
COMPETITIVE_PCT = 0.95         # High-value card below this % of market -> RAISE
SUGGESTED_DISCOUNT = 0.98      # Suggested price = market * this for RAISE

# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------
RARITY_RANK = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3}
SPIKE_THRESHOLD = 0.20         # >20% increase in 30 days = spike
SPIKE_MIN_PRICE = 0.25            # Ignore spikes below this starting price
MIN_PRICE = 0.01               # Floor for all price values
SPOILER_WINDOW_DAYS = 30       # Days before release to flag spoiler season
RELEASE_PROXIMITY_MAX = 90     # Clamp set_release_proximity at this many days
TRAINING_CACHE_FILENAME = "training_cards.json"  # Full-universe training cache
TRAINING_CACHE_MAX_CARDS = 0       # 0 = no limit; set to e.g. 5000 for faster builds

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PRICE_HISTORY_DIR = os.path.join(DATA_DIR, "price_history")

# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
MIN_HISTORY_DAYS = 14          # Minimum days of history for forecasting
MAX_HISTORY_DAYS = 90          # Max days to use for linear regression
TREND_THRESHOLD = 0.03         # Slope threshold for up/down vs flat

# ---------------------------------------------------------------------------
# Spike Classifier (XGBoost)
# ---------------------------------------------------------------------------
N_ESTIMATORS = 200
MAX_DEPTH = 4
LEARNING_RATE = 0.1
MIN_CHILD_WEIGHT = 10              # Minimum sum of instance weight in a child
REG_ALPHA = 0.1                    # L1 regularization
REG_LAMBDA = 1.0                   # L2 regularization
SUBSAMPLE = 0.8                    # Row subsampling per tree
COLSAMPLE_BYTREE = 0.8            # Feature subsampling per tree
VALIDATION_SPLIT = 0.2             # Fraction of data held out for validation
RANDOM_SEED = 42                   # Reproducible train/val split
SAMPLE_WEIGHT_FEATURE = "current_price"  # Feature used for sample weighting

# ---------------------------------------------------------------------------
# Prediction Pipeline
# ---------------------------------------------------------------------------
SPIKE_HOLD_THRESHOLD = 0.6     # Spike probability >= this -> HOLD signal

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
DOWNLOAD_TIMEOUT = 120         # Seconds for HTTP requests
DOWNLOAD_MAX_RETRIES = 3       # Retry attempts for downloads
DOWNLOAD_BACKOFF_BASE = 2      # Exponential backoff base (seconds)

# ---------------------------------------------------------------------------
# CSV Schema
# ---------------------------------------------------------------------------
REQUIRED_CSV_COLUMNS = [
    "TCGplayer Id",
    "Product Name",
    "TCG Market Price",
    "TCG Marketplace Price",
    "Total Quantity",
]
