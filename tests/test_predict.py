# tests/test_predict.py
import csv
import json
import pytest
from pathlib import Path
from lib.predict import run_predict

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def setup_dirs(tmp_path):
    history_dir = tmp_path / "history"
    data_dir = tmp_path / "data" / "mtgjson"
    models_dir = tmp_path / "models"
    output_dir = tmp_path / "output"
    for d in [history_dir, data_dir, models_dir, output_dir]:
        d.mkdir(parents=True)

    src = FIXTURES / "inventory_cards.json"
    (data_dir / "inventory_cards.json").write_text(src.read_text())

    latest_csv = history_dir / "latest.csv"
    latest_csv.write_text(
        "TCGplayer Id,Product Name,TCG Market Price,TCG Marketplace Price,Total Quantity\n"
        "111111,Test Rare,1.91,2.00,3\n"
        "222222,Test Common,0.06,0.05,1\n"
        "333333,Not In MTGJson,0.50,0.50,2\n"
    )
    return {
        "history_dir": str(history_dir),
        "data_dir": str(data_dir),
        "models_dir": str(models_dir),
        "output_dir": str(output_dir),
    }


def test_run_predict_creates_predictions_csv(setup_dirs):
    run_predict(**setup_dirs)
    output = Path(setup_dirs["output_dir"])
    assert len(list(output.glob("predictions-*.csv"))) == 1


def test_run_predict_creates_watchlist_csv(setup_dirs):
    run_predict(**setup_dirs)
    output = Path(setup_dirs["output_dir"])
    assert len(list(output.glob("watchlist-*.csv"))) == 1


def test_predictions_csv_has_required_columns(setup_dirs):
    run_predict(**setup_dirs)
    output = Path(setup_dirs["output_dir"])
    pred_file = list(output.glob("predictions-*.csv"))[0]
    with open(pred_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    expected_cols = {
        "TCGplayer Id", "Product Name", "Current Price", "Market Price",
        "Suggested Price", "Action", "Reason", "Margin",
        "Predicted 7d", "Predicted 30d", "Trend", "Spike Probability", "Signal",
    }
    assert expected_cols.issubset(set(reader.fieldnames))


def test_unmatched_card_still_appears_in_output(setup_dirs):
    run_predict(**setup_dirs)
    output = Path(setup_dirs["output_dir"])
    pred_file = list(output.glob("predictions-*.csv"))[0]
    with open(pred_file) as f:
        rows = list(csv.DictReader(f))
    ids = {r["TCGplayer Id"] for r in rows}
    assert "333333" in ids


def test_watchlist_only_contains_high_spike_probability(setup_dirs):
    run_predict(**setup_dirs)
    output = Path(setup_dirs["output_dir"])
    watchlist_file = list(output.glob("watchlist-*.csv"))[0]
    with open(watchlist_file) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        assert float(row["Spike Probability"]) >= 0.6


def test_empty_cache_writes_empty_outputs(tmp_path):
    """When no MTGJson cache exists, predict writes header-only CSVs."""
    history_dir = tmp_path / "history"
    data_dir = tmp_path / "data" / "mtgjson"
    models_dir = tmp_path / "models"
    output_dir = tmp_path / "output"
    for d in [history_dir, data_dir, models_dir, output_dir]:
        d.mkdir(parents=True)
    (history_dir / "latest.csv").write_text(
        "TCGplayer Id,Product Name,TCG Market Price,TCG Marketplace Price,Total Quantity\n"
        "111111,Test,1.00,1.00,1\n"
    )
    run_predict(str(history_dir), str(data_dir), str(models_dir), str(output_dir))
    pred_files = list(output_dir.glob("predictions-*.csv"))
    assert len(pred_files) == 1
    with open(pred_files[0]) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 0  # header only


def test_dry_run_does_not_write_files(setup_dirs):
    run_predict(**setup_dirs, dry_run=True)
    output = Path(setup_dirs["output_dir"])
    assert len(list(output.glob("predictions-*.csv"))) == 0
    assert len(list(output.glob("watchlist-*.csv"))) == 0
