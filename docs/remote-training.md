# Remote GPU Training Setup

The spike classifier trains faster on an NVIDIA GPU. `monitor.sh train --remote` handles the data transfer and model retrieval automatically over your Tailscale network — the PC just needs to be awake and reachable.

## What happens under the hood

```
Pi                                  PC (via Tailscale SSH)
────────────────────────            ──────────────────────────
Extract features from cache
  → /tmp/tcgplayer_train/
    features.json

rsync lib/ + train_remote.py ─────→ /tmp/tcgplayer_train/
rsync features.json ──────────────→ /tmp/tcgplayer_train/

ssh: python3 train_remote.py ─────→ XGBoost trains (device=cuda)
                                      → spike_classifier.json

rsync model back ←────────────────  /tmp/tcgplayer_train/
                                      spike_classifier.json
models/spike_classifier.json saved
```

No persistent service runs on the PC. It only does work when you explicitly run `train --remote`.

## PC setup (one-time)

### 1. Install Python dependencies

```bash
pip install xgboost pandas numpy
```

CUDA toolkit does not need to be installed separately — XGBoost bundles its own CUDA runtime as of v2.0.

Verify GPU is detected:
```bash
python3 -c "import xgboost as xgb; print(xgb.__version__)"
# Should print without error
```

### 2. Verify Tailscale connectivity

From the Pi, confirm you can reach the PC:
```bash
tailscale ping <pc-hostname>
```

### 3. Set up SSH key authentication

Remote training uses SSH without a password prompt. If you haven't already:

```bash
# On the Pi — generate a key if needed
ssh-keygen -t ed25519 -C "pi-tcgplayer"

# Copy it to the PC (replace <user> and <pc-hostname>)
ssh-copy-id <user>@<pc-hostname>

# Test it
ssh <user>@<pc-hostname> echo "ok"
```

### 4. Run a training job

```bash
bash scripts/monitor.sh train --remote <pc-tailscale-hostname>
```

Expected output:
```
Training remotely on <hostname> (GPU)...
Extracted 1842 training rows
Training on 1842 examples with device=cuda ...
✅ Model saved to /tmp/tcgplayer_train/spike_classifier.json
✅ Model retrieved from <hostname>
```

## Retraining

Retrain whenever you've accumulated more price history (monthly is plenty):

```bash
bash scripts/monitor.sh sync    # optional: refresh MTGJson data first
bash scripts/monitor.sh train --remote <hostname>
```

The new model overwrites `models/spike_classifier.json`. If training fails, the previous model is left in place.

## Fallback behaviour

| Condition | What happens |
|---|---|
| PC unreachable | Falls back to local CPU training with a warning |
| `models/` has no model | `predict` auto-trains locally before scoring |
| Training data too sparse | `predict` skips spike scoring, runs forecast only |

## Troubleshooting

**SSH asks for a password**
Set up key authentication (Step 3 above).

**`xgboost` not found on PC**
```bash
pip install xgboost
```

**`device=cuda` error on PC**
Your XGBoost version may be older than 2.0. Update it:
```bash
pip install --upgrade xgboost
```

**Features file is empty**
Run `sync` first to build the MTGJson inventory cache, then `train`:
```bash
bash scripts/monitor.sh sync
bash scripts/monitor.sh train --remote <hostname>
```
