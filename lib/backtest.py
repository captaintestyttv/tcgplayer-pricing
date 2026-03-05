"""Backtest spike classifier against historical data."""

import json
import os
from datetime import datetime

from lib.config import SPIKE_HOLD_THRESHOLD, get_logger
from lib.features import generate_training_data, compute_cluster_features
from lib.spike import FEATURE_COLS, score, load_model_meta

log = get_logger(__name__)


def run_backtest(
    data_dir: str,
    models_dir: str,
    output_dir: str,
) -> dict:
    """Score training data with the trained model to evaluate performance.

    Returns a results dict with confusion matrix, precision, recall, F1,
    and probability calibration bins.
    """
    from lib.mtgjson import load_inventory_cache

    cache = load_inventory_cache(data_dir)
    if not cache:
        raise ValueError("No MTGJson cache found. Run 'monitor.sh sync' first.")

    model_path = os.path.join(models_dir, "spike_classifier.json")
    if not os.path.exists(model_path):
        raise ValueError("No trained model found. Run 'monitor.sh train' first.")

    rows = generate_training_data(cache)
    if not rows:
        raise ValueError("No training data generated. Insufficient price history.")

    # Score all rows
    probabilities = score(rows, model_path)

    # Compute metrics
    actuals = [r["spike"] for r in rows]
    predicted = [1 if p >= SPIKE_HOLD_THRESHOLD else 0 for p in probabilities]

    tp = sum(1 for a, p in zip(actuals, predicted) if a == 1 and p == 1)
    fp = sum(1 for a, p in zip(actuals, predicted) if a == 0 and p == 1)
    fn = sum(1 for a, p in zip(actuals, predicted) if a == 1 and p == 0)
    tn = sum(1 for a, p in zip(actuals, predicted) if a == 0 and p == 0)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    accuracy = (tp + tn) / max(len(actuals), 1)

    # Probability calibration: 10 bins
    bins = []
    for i in range(10):
        lo = i * 0.1
        hi = (i + 1) * 0.1
        in_bin = [(a, p) for a, p in zip(actuals, probabilities) if lo <= p < hi]
        if in_bin:
            actual_rate = sum(a for a, _ in in_bin) / len(in_bin)
            avg_prob = sum(p for _, p in in_bin) / len(in_bin)
            bins.append({
                "range": f"{lo:.1f}-{hi:.1f}",
                "count": len(in_bin),
                "actual_spike_rate": round(actual_rate, 4),
                "avg_predicted_prob": round(avg_prob, 4),
            })

    meta = load_model_meta(model_path)

    results = {
        "timestamp": datetime.now().isoformat(),
        "model_trained_at": meta.get("trained_at", "unknown") if meta else "unknown",
        "total_samples": len(rows),
        "spike_count": sum(actuals),
        "spike_rate": round(sum(actuals) / max(len(actuals), 1), 4),
        "threshold": SPIKE_HOLD_THRESHOLD,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "calibration_bins": bins,
    }

    # Write results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"backtest-{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n--- Backtest Results ---")
    print(f"Samples: {results['total_samples']} ({results['spike_count']} spikes, rate={results['spike_rate']:.1%})")
    print(f"Threshold: {SPIKE_HOLD_THRESHOLD}")
    print(f"Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"\nResults saved to {output_path}")

    return results
