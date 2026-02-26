# Remote GPU Training Setup

The spike classifier trains faster on an NVIDIA GPU. `monitor.sh train --remote` handles data transfer and model retrieval automatically — you run one command on the Pi and it does the rest.

## Windows requirement: WSL2

The transfer script uses `scp` and `mkdir` over SSH, and the remote training script uses Unix paths (`/tmp/`). These require a Linux shell on the PC side. **WSL2 must be configured as the default SSH shell on your Windows machine** — once that's done, everything else works as documented.

### One-time WSL2 setup — on the PC

**1. Install WSL2 with Ubuntu** (in PowerShell as Administrator):
```powershell
wsl --install
# Reboot when prompted, then complete Ubuntu first-run setup
```

**2. Enable the OpenSSH Server** (in PowerShell as Administrator):
```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd
```

**3. Set WSL2 bash as the default SSH shell** (in PowerShell as Administrator):
```powershell
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" `
  -Name DefaultShell `
  -Value "C:\Windows\System32\bash.exe" `
  -PropertyType String -Force
```

**4. Install Python dependencies inside WSL2** (open the Ubuntu app or run `wsl`):
```bash
pip3 install xgboost pandas numpy
```

After this, SSH connections from the Pi will land in a Linux bash session inside WSL2, and all the Linux commands (`mkdir`, `python3`, `/tmp/` paths) work normally.

## How it works

The Pi prepares training data locally, ships it to the PC over Tailscale SSH, the PC trains with CUDA and writes the model, then the Pi retrieves it. No persistent service runs on the PC.

```
Pi                                  PC (via Tailscale SSH)
────────────────────────            ──────────────────────────
1. Extract features from
   MTGJson cache
   → /tmp/tcgplayer_train/
     features.json

2. rsync lib/ + train_remote.py ──→ /tmp/tcgplayer_train/
   rsync features.json ───────────→ /tmp/tcgplayer_train/

3. ssh: python3 train_remote.py ──→ XGBoost trains (device=cuda)
                                      → spike_classifier.json

4. rsync model back ←─────────────  /tmp/tcgplayer_train/
                                      spike_classifier.json

5. Save to models/
   spike_classifier.json
```

---

## One-time setup

### Step 1 — On the PC: install Python dependencies

**Windows (in WSL2 — open the Ubuntu app or run `wsl` in a terminal):**
```bash
pip3 install xgboost pandas numpy
```

XGBoost v2.0+ bundles its own CUDA runtime — no separate CUDA toolkit install needed. Verify it works:
```bash
python3 -c "import xgboost as xgb; print(xgb.__version__)"
```

### Step 2 — On the Pi: verify Tailscale connectivity

```bash
tailscale ping <pc-hostname>
```

Replace `<pc-hostname>` with the Tailscale machine name for your PC (visible in the Tailscale admin panel or `tailscale status`).

### Step 3 — On the Pi: set up SSH key authentication

Remote training SSHes into the PC without a password prompt. Run these commands **on the Pi**:

```bash
# Generate a key (skip if you already have one at ~/.ssh/id_ed25519)
ssh-keygen -t ed25519 -C "pi-tcgplayer"

# Authorise the Pi's key on the PC (you'll be prompted for your PC password once)
ssh-copy-id <user>@<pc-hostname>

# Confirm passwordless login works
ssh <user>@<pc-hostname> echo "connection ok"
```

Replace `<user>` with your PC login username and `<pc-hostname>` with the Tailscale hostname from Step 2.

---

## Running a training job

Everything from here runs **on the Pi**.

### First run (if you haven't synced MTGJson data yet)

```bash
bash scripts/monitor.sh sync
```

This downloads MTGJson data and builds your inventory cache (~15 minutes, ~650MB).

### Train on the PC's GPU

```bash
bash scripts/monitor.sh train --remote <user>@<pc-hostname>
```

Expected output:

```
Training remotely on <hostname> (GPU)...
Extracted 1842 training rows
Training on 1842 examples with device=cuda ...
✅ Model saved to /tmp/tcgplayer_train/spike_classifier.json
✅ Model retrieved from <hostname>
```

The trained model is saved to `models/spike_classifier.json` on the Pi. The next `predict` run will use it automatically.

---

## Retraining

Retrain monthly as your price history grows. Run **on the Pi**:

```bash
bash scripts/monitor.sh sync                          # optional: refresh MTGJson data
bash scripts/monitor.sh train --remote <user>@<pc-hostname>
```

The new model overwrites the previous one. If training fails for any reason, the existing model is left in place.

---

## Fallback behaviour

| Condition | What happens |
|---|---|
| PC unreachable over Tailscale | Falls back to local CPU training on the Pi with a warning |
| `models/spike_classifier.json` missing | `predict` auto-trains locally (CPU) before scoring |
| Training data too sparse | `predict` skips spike scoring, still runs price forecasts |

---

## Troubleshooting

**SSH asks for a password**
Key authentication isn't set up. Follow Step 3 above.

**`xgboost` not found on PC**
Run on the PC:
```bash
pip install xgboost
```

**`device=cuda` error on PC**
XGBoost is older than v2.0. Update it on the PC:
```bash
pip install --upgrade xgboost
```

**"Features file is empty" error**
The MTGJson cache hasn't been built yet. Run on the Pi:
```bash
bash scripts/monitor.sh sync
bash scripts/monitor.sh train --remote <user>@<pc-hostname>
```
