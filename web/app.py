"""Flask web UI for TCGPlayer Pricing Tool."""

import csv
import io
import os
import sys

# Ensure project root is on sys.path so lib/ imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, render_template, jsonify, request, redirect, url_for

from web.data import (
    get_dashboard_stats, get_predictions, get_watchlist, get_analysis,
    get_backtest, get_model_meta, get_card, get_card_prices,
    get_export_files, invalidate_inventory_cache,
    HISTORY_DIR, OUTPUT_DIR, DATA_DIR, MODELS_DIR, EXPORTS_DIR, PROJECT_ROOT,
)
from web.jobs import runner

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    stats = get_dashboard_stats()
    return render_template("dashboard.html", stats=stats)


@app.route("/predictions")
def predictions_page():
    preds = get_predictions()
    return render_template("predictions.html", predictions=preds)


@app.route("/watchlist")
def watchlist_page():
    items = get_watchlist()
    return render_template("watchlist.html", watchlist=items)


@app.route("/analysis")
def analysis_page():
    data = get_analysis()
    return render_template("analysis.html", analysis=data)


@app.route("/backtest")
def backtest_page():
    data = get_backtest()
    meta = get_model_meta()
    return render_template("backtest.html", backtest=data, model_meta=meta)


@app.route("/card/<tcg_id>")
def card_page(tcg_id):
    card = get_card(tcg_id)
    if not card:
        return render_template("card.html", card=None, tcg_id=tcg_id), 404
    return render_template("card.html", card=card, tcg_id=tcg_id)


@app.route("/jobs")
def jobs_page():
    jobs = runner.all_jobs()
    return render_template("jobs.html", jobs=jobs)


# ---------------------------------------------------------------------------
# API routes (JSON)
# ---------------------------------------------------------------------------

@app.route("/api/dashboard")
def api_dashboard():
    return jsonify(get_dashboard_stats())


@app.route("/api/predictions")
def api_predictions():
    return jsonify(get_predictions())


@app.route("/api/watchlist")
def api_watchlist():
    return jsonify(get_watchlist())


@app.route("/api/card/<tcg_id>/prices")
def api_card_prices(tcg_id):
    return jsonify(get_card_prices(tcg_id))


@app.route("/api/jobs")
def api_jobs():
    return jsonify(runner.all_jobs())


@app.route("/api/jobs/<job_id>")
def api_job(job_id):
    job = runner.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job.to_dict())


# ---------------------------------------------------------------------------
# Action endpoints
# ---------------------------------------------------------------------------

@app.route("/api/import", methods=["POST"])
def api_import():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename or not f.filename.endswith(".csv"):
        return jsonify({"error": "File must be a .csv"}), 400

    # Validate CSV columns before saving
    content = f.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    if reader.fieldnames is None:
        return jsonify({"error": "CSV is empty or has no header"}), 400

    from lib.config import REQUIRED_CSV_COLUMNS
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in reader.fieldnames]
    if missing:
        return jsonify({"error": f"Missing columns: {', '.join(missing)}"}), 400

    # Save to tcgplayer-exports/
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    save_path = os.path.join(EXPORTS_DIR, f.filename)
    with open(save_path, "w", newline="", encoding="utf-8") as out:
        out.write(content)

    def _import_job():
        import shutil
        from datetime import datetime
        from lib.analysis import run_analysis

        os.makedirs(HISTORY_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = os.path.join(HISTORY_DIR, f"export-{timestamp}.csv")
        shutil.copy2(save_path, dest)
        shutil.copy2(save_path, os.path.join(HISTORY_DIR, "latest.csv"))
        print(f"Imported: {save_path}")
        print(f"Saved to: {dest}")

        run_analysis(HISTORY_DIR, OUTPUT_DIR)
        print("Analysis complete.")

    job = runner.start("import", _import_job)
    if not job:
        return jsonify({"error": "An import job is already running"}), 409
    return jsonify(job.to_dict()), 202


@app.route("/api/sync", methods=["POST"])
def api_sync():
    body = request.get_json(silent=True) or {}
    force = body.get("force", False)
    cache_only = body.get("cache_only", False)

    def _sync_job():
        from lib.mtgjson import sync
        sync(HISTORY_DIR, DATA_DIR, force=force, cache_only=cache_only)
        invalidate_inventory_cache()
        print("Sync complete.")

    job = runner.start("sync", _sync_job)
    if not job:
        return jsonify({"error": "A sync job is already running"}), 409
    return jsonify(job.to_dict()), 202


@app.route("/api/train", methods=["POST"])
def api_train():
    def _train_job():
        from lib.mtgjson import load_inventory_cache
        from lib.features import generate_training_data
        from lib.spike import train

        cache = load_inventory_cache(DATA_DIR)
        if not cache:
            raise ValueError("No MTGJson cache found. Run sync first.")

        rows = generate_training_data(cache)
        if not rows:
            raise ValueError("Insufficient price history for training.")

        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, "spike_classifier.json")
        train(rows, model_path, device="cpu")
        print("Training complete.")

    job = runner.start("train", _train_job)
    if not job:
        return jsonify({"error": "A train job is already running"}), 409
    return jsonify(job.to_dict()), 202


@app.route("/api/predict", methods=["POST"])
def api_predict():
    def _predict_job():
        from lib.predict import run_predict
        run_predict(
            history_dir=HISTORY_DIR,
            data_dir=DATA_DIR,
            models_dir=MODELS_DIR,
            output_dir=OUTPUT_DIR,
        )
        print("Predict complete.")

    job = runner.start("predict", _predict_job)
    if not job:
        return jsonify({"error": "A predict job is already running"}), 409
    return jsonify(job.to_dict()), 202


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    def _backtest_job():
        from lib.backtest import run_backtest
        run_backtest(
            data_dir=DATA_DIR,
            models_dir=MODELS_DIR,
            output_dir=OUTPUT_DIR,
        )
        print("Backtest complete.")

    job = runner.start("backtest", _backtest_job)
    if not job:
        return jsonify({"error": "A backtest job is already running"}), 409
    return jsonify(job.to_dict()), 202


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=True)
