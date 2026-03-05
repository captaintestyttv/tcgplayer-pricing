# tests/test_backtest.py
import json
import pytest
from pathlib import Path
from lib.backtest import run_backtest

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
    )
    return {
        "data_dir": str(data_dir),
        "models_dir": str(models_dir),
        "output_dir": str(output_dir),
    }


@pytest.fixture
def trained_dirs(setup_dirs):
    """Setup with a pre-trained model."""
    from lib.features import generate_training_data
    from lib.spike import train
    from lib.mtgjson import load_inventory_cache
    import os

    cache = load_inventory_cache(setup_dirs["data_dir"])
    rows = generate_training_data(cache)
    model_path = os.path.join(setup_dirs["models_dir"], "spike_classifier.json")
    train(rows, model_path, device="cpu")
    return setup_dirs


def test_backtest_returns_results(trained_dirs):
    results = run_backtest(**trained_dirs)
    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
    assert "confusion_matrix" in results


def test_backtest_writes_output_file(trained_dirs):
    run_backtest(**trained_dirs)
    output = Path(trained_dirs["output_dir"])
    assert len(list(output.glob("backtest-*.json"))) == 1


def test_backtest_confusion_matrix_sums(trained_dirs):
    results = run_backtest(**trained_dirs)
    cm = results["confusion_matrix"]
    assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == results["total_samples"]


def test_backtest_raises_without_cache(tmp_path):
    with pytest.raises(ValueError, match="cache"):
        run_backtest(str(tmp_path), str(tmp_path), str(tmp_path))


def test_backtest_raises_without_model(setup_dirs):
    with pytest.raises(ValueError, match="model"):
        run_backtest(**setup_dirs)


def test_backtest_calibration_bins(trained_dirs):
    results = run_backtest(**trained_dirs)
    assert "calibration_bins" in results
    for b in results["calibration_bins"]:
        assert "range" in b
        assert "count" in b
        assert "actual_spike_rate" in b
