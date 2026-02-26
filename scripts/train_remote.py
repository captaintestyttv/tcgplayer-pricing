#!/usr/bin/env python3
"""
Remote training script — runs on the PC with NVIDIA GPU.
Invoked by monitor.sh via SSH.

Usage:
    python3 train_remote.py --features /tmp/tcgplayer/features.json \
                            --output /tmp/tcgplayer/spike_classifier.json
"""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost spike classifier on GPU")
    parser.add_argument("--features", required=True, help="Path to features JSON")
    parser.add_argument("--output", required=True, help="Path to save trained model")
    args = parser.parse_args()

    try:
        import xgboost  # noqa: F401
    except ImportError:
        print("ERROR: xgboost not installed. Run: pip install xgboost")
        sys.exit(1)

    with open(args.features) as f:
        rows = json.load(f)

    if not rows:
        print("ERROR: features file is empty")
        sys.exit(1)

    # lib/ is rsynced alongside this script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from lib.spike import train

    print(f"Training on {len(rows)} examples with device=cuda ...")
    train(rows, args.output, device="cuda")
    print(f"✅ Model saved to {args.output}")


if __name__ == "__main__":
    main()
