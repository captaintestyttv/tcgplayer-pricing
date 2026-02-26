#!/usr/bin/env bash
# =============================================================================
# TCGPlayer Price Monitor & Predictive Pricing
# =============================================================================
# Usage:
#   monitor.sh import <export.csv>   # Import new price data
#   monitor.sh analyze                # Analyze changes vs history
#   monitor.sh report                 # Generate pricing recommendations
#   monitor.sh baseline               # Set current as baseline
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRICING_DIR="/home/cap/.openclaw/workspace/projects/tcgplayer-pricing"
HISTORY_DIR="${PRICING_DIR}/history"
OUTPUT_DIR="${PRICING_DIR}/output"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Fees
COMMISSION_FEE=0.1075
TRANSACTION_FEE=0.025
TRANSACTION_FLAT=0.30
SHIPPING_REVENUE=1.31
POSTAGE_STANDARD=0.73

# =============================================================================
# Import new price data
# =============================================================================
import_data() {
    local input_file="$1"
    if [[ -z "$input_file" ]]; then
        echo "Usage: $0 import <export.csv>"
        return 1
    fi
    
    if [[ ! -f "$input_file" ]]; then
        echo "Error: File not found: $input_file"
        return 1
    fi
    
    # Copy to history with timestamp
    cp "$input_file" "${HISTORY_DIR}/export-${TIMESTAMP}.csv"
    
    echo "✅ Imported: $input_file"
    echo "   Saved to: ${HISTORY_DIR}/export-${TIMESTAMP}.csv"
    
    # Also save as "latest.csv" for quick access
    cp "$input_file" "${HISTORY_DIR}/latest.csv"
    
    # Run analysis
    analyze_changes
}

# =============================================================================
# Analyze price changes vs history
# =============================================================================
analyze_changes() {
    echo ""
    echo "=== Price Analysis ==="
    
    python3 << 'PYEOF'
import csv
import os
import json
from datetime import datetime

HISTORY_DIR = "/home/cap/.openclaw/workspace/projects/tcgplayer-pricing/history"
OUTPUT_DIR = "/home/cap/.openclaw/workspace/projects/tcgplayer-pricing/output"

# Find all history files
files = sorted([f for f in os.listdir(HISTORY_DIR) if f.startswith('export-') and f.endswith('.csv')])

if len(files) < 2:
    print(f"Need at least 2 exports to compare. Currently have {len(files)}.")
    exit(0)

# Load latest
latest_file = files[-1]
with open(f"{HISTORY_DIR}/{latest_file}", 'r') as f:
    reader = csv.DictReader(f)
    latest = {row['TCGplayer Id']: row for row in reader}

# Load previous
prev_file = files[-2]
with open(f"{HISTORY_DIR}/{prev_file}", 'r') as f:
    reader = csv.DictReader(f)
    previous = {row['TCGplayer Id']: row for row in reader}

print(f"Comparing {prev_file} → {latest_file}")
print()

# Categorize changes
spikes_up = []    # >20% increase
spikes_down = []  # >20% decrease  
increases = []
decreases = []
stable = []

for pid, item in latest.items():
    current_price = float(item.get('TCG Market Price') or 0)
    current_list = float(item.get('TCG Marketplace Price') or 0)
    qty = int(item.get('Total Quantity', 0) or 0)
    
    if qty == 0:
        continue
        
    if pid in previous:
        prev_item = previous[pid]
        prev_price = float(prev_item.get('TCG Market Price') or 0)
        prev_list = float(prev_item.get('TCG Marketplace Price') or 0)
        
        if prev_price > 0:
            change_pct = ((current_price - prev_price) / prev_price) * 100
            
            entry = {
                'name': item.get('Product Name', '')[:40],
                'id': pid,
                'current': current_price,
                'previous': prev_price,
                'change_pct': change_pct,
                'current_list': current_list,
                'change_amt': current_list - prev_list
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

# Print report
print(f"📈 SPIKES (>20% up): {len(spikes_up)}")
for e in sorted(spikes_up, key=lambda x: -x['change_pct'])[:10]:
    print(f"   {e['name']:<40} ${e['previous']:.2f} → ${e['current']:.2f} (+{e['change_pct']:.0f}%)")

print(f"\n📉 DROPS (>20% down): {len(spikes_down)}")
for e in sorted(spikes_down, key=lambda x: x['change_pct'])[:10]:
    print(f"   {e['name']:<40} ${e['previous']:.2f} → ${e['current']:.2f} ({e['change_pct']:.0f}%)")

print(f"\n⬆️ MODERATE INCREASES (>5%): {len(increases)}")
print(f"⬇️ MODERATE DECREASES (<-5%): {len(decreases)}")
print(f"➡️ STABLE (<5% change): {len(stable)}")

# Save analysis
analysis = {
    'timestamp': datetime.now().isoformat(),
    'files_compared': [prev_file, latest_file],
    'spikes_up': len(spikes_up),
    'spikes_down': len(spikes_down),
    'increases': len(increases),
    'decreases': len(decreases),
    'stable': len(stable),
    'spikes_up_details': spikes_up[:20],
    'spikes_down_details': spikes_down[:20]
}

with open(f"{OUTPUT_DIR}/analysis-latest.json", 'w') as f:
    json.dump(analysis, f, indent=2)

print(f"\n✅ Analysis saved to {OUTPUT_DIR}/analysis-latest.json")
PYEOF
}

# =============================================================================
# Generate pricing recommendations
# =============================================================================
generate_recommendations() {
    echo ""
    echo "=== Pricing Recommendations ==="
    
    python3 << 'PYEOF'
import csv
import json
import os
from datetime import datetime

HISTORY_DIR = "/home/cap/.openclaw/workspace/projects/tcgplayer-pricing/history"
OUTPUT_DIR = "/home/cap/.openclaw/workspace/projects/tcgplayer-pricing/output"

# Find history files
files = sorted([f for f in os.listdir(HISTORY_DIR) if f.startswith('export-') and f.endswith('.csv')])
if not files:
    print("No data found. Import an export first.")
    exit(0)

# Load latest
latest_file = files[-1]
with open(f"{HISTORY_DIR}/{latest_file}", 'r') as f:
    reader = csv.DictReader(f)
    items = [row for row in reader]

# Pricing rules
COMMISSION_FEE = 0.1075
TRANSACTION_FEE = 0.025
TRANSACTION_FLAT = 0.30
SHIPPING_REVENUE = 1.31

recommendations = []

for item in items:
    qty = int(item.get('Total Quantity', 0) or 0)
    if qty == 0:
        continue
    
    market = float(item.get('TCG Market Price') or 0)
    current = float(item.get('TCG Marketplace Price') or 0)
    
    # Calculate profit
    if market < 5:
        revenue = market + SHIPPING_REVENUE
    else:
        revenue = market
    
    fees = revenue * (COMMISSION_FEE + TRANSACTION_FEE) + TRANSACTION_FLAT
    postage = 0.73 if market < 5 else 1.50
    net = revenue - fees - postage
    
    # Recommendation logic
    action = None
    reason = ""
    
    if net < 0.10:
        action = "RAISE"
        reason = f"Low margin (${net:.2f})"
    elif market > current * 1.1:
        action = "RAISE"
        reason = f"Market up 10%+, current underpriced"
    elif market < current * 0.9:
        action = "LOWER"
        reason = f"Market down 10%+, over priced"
    elif market >= 5 and current < market * 0.95:
        action = "RAISE"
        reason = "Competitive adjustment for high-value"
    
    if action:
        suggested = round(market * 0.98, 2) if action == "RAISE" else round(market, 2)
        recommendations.append({
            'name': item.get('Product Name', '')[:40],
            'id': item.get('TCGplayer Id'),
            'current': current,
            'market': market,
            'suggested': max(0.01, suggested),
            'action': action,
            'reason': reason,
            'margin': round(net, 2)
        })

# Sort by margin impact
recommendations.sort(key=lambda x: x['margin'])

print(f"Total recommendations: {len(recommendations)}")
print()

# Group by action
raise_count = sum(1 for r in recommendations if r['action'] == 'RAISE')
lower_count = sum(1 for r in recommendations if r['action'] == 'LOWER')

print(f"🔺 RAISE PRICES: {raise_count}")
print(f"🔻 LOWER PRICES: {lower_count}")
print()

# Show top recommendations
print("Top recommendations by urgency:")
for r in recommendations[:15]:
    print(f"  {r['action']:<6} {r['name']:<40} ${r['current']:.2f} → ${r['suggested']:.2f} ({r['reason']})")

# Generate import CSV
if recommendations:
    output_file = f"{OUTPUT_DIR}/price-adjustments-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['TCGplayer Id', 'Product Name', 'Current Price', 'Market Price', 'Suggested Price', 'Action', 'Reason', 'Margin'])
        writer.writeheader()
        for r in recommendations:
            writer.writerow({
                'TCGplayer Id': r['id'],
                'Product Name': r['name'],
                'Current Price': r['current'],
                'Market Price': r['market'],
                'Suggested Price': r['suggested'],
                'Action': r['action'],
                'Reason': r['reason'],
                'Margin': r['margin']
            })
    print(f"\n✅ Recommendations saved to: {output_file}")

PYEOF
}

# =============================================================================
# Main
# =============================================================================
case "${1:-}" in
    import)
        import_data "$2"
        ;;
    analyze)
        analyze_changes
        ;;
    report|recommendations)
        generate_recommendations
        ;;
    baseline)
        echo "Baseline set at $TIMESTAMP"
        cp "${HISTORY_DIR}/latest.csv" "${HISTORY_DIR}/baseline.csv" 2>/dev/null || echo "No data to set as baseline"
        ;;
    *)
        echo "TCGPlayer Price Monitor"
        echo ""
        echo "Usage: $0 <command> [args]"
        echo ""
        echo "Commands:"
        echo "  import <file.csv>   Import new price export"
        echo "  analyze             Analyze changes vs history"
        echo "  report              Generate pricing recommendations"
        echo "  baseline            Set current data as baseline for comparison"
        ;;
esac