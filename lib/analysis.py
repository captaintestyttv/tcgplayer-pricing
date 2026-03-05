"""Price change analysis — extracted from monitor.sh analyze_changes heredoc."""

import csv
import json
import os
from datetime import datetime

from lib.config import get_logger

log = get_logger(__name__)


def run_analysis(history_dir: str, output_dir: str) -> dict:
    """Compare the two most recent exports and return an analysis dict.

    Also writes analysis-latest.json to *output_dir*.
    """
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(history_dir)
        if f.startswith("export-") and f.endswith(".csv")
    )

    if len(files) < 2:
        return {
            "error": f"Need at least 2 exports to compare. Currently have {len(files)}.",
            "spikes_up": 0, "spikes_down": 0,
            "increases": 0, "decreases": 0, "stable": 0,
            "spikes_up_details": [], "spikes_down_details": [],
        }

    latest_file = files[-1]
    with open(os.path.join(history_dir, latest_file), newline="") as f:
        latest = {row["TCGplayer Id"]: row for row in csv.DictReader(f)}

    prev_file = files[-2]
    with open(os.path.join(history_dir, prev_file), newline="") as f:
        previous = {row["TCGplayer Id"]: row for row in csv.DictReader(f)}

    spikes_up = []
    spikes_down = []
    increases = []
    decreases = []
    stable = []

    for pid, item in latest.items():
        current_price = float(item.get("TCG Market Price") or 0)
        current_list = float(item.get("TCG Marketplace Price") or 0)
        qty = int(item.get("Total Quantity", 0) or 0)

        if qty == 0:
            continue

        if pid in previous:
            prev_item = previous[pid]
            prev_price = float(prev_item.get("TCG Market Price") or 0)
            prev_list = float(prev_item.get("TCG Marketplace Price") or 0)

            if prev_price > 0:
                change_pct = ((current_price - prev_price) / prev_price) * 100

                entry = {
                    "name": item.get("Product Name", "")[:40],
                    "id": pid,
                    "current": current_price,
                    "previous": prev_price,
                    "change_pct": round(change_pct, 2),
                    "current_list": current_list,
                    "change_amt": round(current_list - prev_list, 2),
                }

                if change_pct > 20:
                    spikes_up.append(entry)
                elif change_pct < -20:
                    spikes_down.append(entry)
                elif change_pct > 5:
                    increases.append(entry)
                elif change_pct < -5:
                    decreases.append(entry)
                else:
                    stable.append(entry)

    spikes_up.sort(key=lambda x: -x["change_pct"])
    spikes_down.sort(key=lambda x: x["change_pct"])

    analysis = {
        "timestamp": datetime.now().isoformat(),
        "files_compared": [prev_file, latest_file],
        "spikes_up": len(spikes_up),
        "spikes_down": len(spikes_down),
        "increases": len(increases),
        "decreases": len(decreases),
        "stable": len(stable),
        "spikes_up_details": spikes_up[:20],
        "spikes_down_details": spikes_down[:20],
        "increases_details": increases[:20],
        "decreases_details": decreases[:20],
    }

    with open(os.path.join(output_dir, "analysis-latest.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    return analysis
