# tests/test_spike.py
import json
import pytest
from pathlib import Path
from lib.features import generate_training_data
from lib.spike import train, score, FEATURE_COLS

FIXTURES = Path(__file__).parent / "fixtures" / "inventory_cards.json"

@pytest.fixture
def cards():
    return json.loads(FIXTURES.read_text())

@pytest.fixture
def trained_model(cards, tmp_path):
    model_path = str(tmp_path / "test_model.json")
    rows = generate_training_data(cards)
    train(rows, model_path, device="cpu")
    return model_path

def test_train_creates_model_file(cards, tmp_path):
    model_path = str(tmp_path / "model.json")
    rows = generate_training_data(cards)
    train(rows, model_path, device="cpu")
    assert Path(model_path).exists()

def test_score_returns_list_of_floats(cards, trained_model):
    from lib.features import extract_features
    features = [extract_features(tid, card) for tid, card in cards.items()]
    scores = score(features, trained_model)
    assert len(scores) == len(features)
    assert all(0.0 <= s <= 1.0 for s in scores)

def test_score_requires_only_feature_cols(cards, trained_model):
    from lib.features import extract_features
    features = [extract_features("222222", cards["222222"])]
    scores = score(features, trained_model)
    assert len(scores) == 1

def test_train_raises_on_empty_rows(tmp_path):
    model_path = str(tmp_path / "model.json")
    with pytest.raises(ValueError, match="No training data"):
        train([], model_path, device="cpu")


def test_train_creates_metadata_file(cards, tmp_path):
    from lib.spike import load_model_meta
    model_path = str(tmp_path / "model.json")
    rows = generate_training_data(cards)
    train(rows, model_path, device="cpu")
    meta = load_model_meta(model_path)
    assert meta is not None
    assert meta["num_samples"] == len(rows)
    assert meta["device"] == "cpu"
    assert "trained_at" in meta
    assert "hyperparameters" in meta
    assert "spike_rate" in meta


def test_load_model_meta_returns_none_when_missing(tmp_path):
    from lib.spike import load_model_meta
    assert load_model_meta(str(tmp_path / "nonexistent.json")) is None
