# tests/test_spike.py
import json
import pytest
from pathlib import Path
from lib.features import generate_training_data
from lib.spike import train, score, FEATURE_COLS, load_model_meta, check_model_compatibility

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
    assert load_model_meta(str(tmp_path / "nonexistent.json")) is None


# Improvement 1: validation metrics in metadata
def test_train_includes_validation_metrics(cards, tmp_path):
    model_path = str(tmp_path / "model.json")
    rows = generate_training_data(cards)
    train(rows, model_path, device="cpu")
    meta = load_model_meta(model_path)
    assert "val_accuracy" in meta
    assert 0 <= meta["val_accuracy"] <= 1
    assert "val_precision" in meta
    assert "val_recall" in meta
    assert "val_samples" in meta


def test_train_uses_scale_pos_weight(cards, tmp_path):
    model_path = str(tmp_path / "model.json")
    rows = generate_training_data(cards)
    train(rows, model_path, device="cpu")
    meta = load_model_meta(model_path)
    assert "scale_pos_weight" in meta["hyperparameters"]
    assert meta["hyperparameters"]["scale_pos_weight"] > 0


# Improvement 2: feature importance
def test_train_records_feature_importance(cards, tmp_path):
    model_path = str(tmp_path / "model.json")
    rows = generate_training_data(cards)
    train(rows, model_path, device="cpu")
    meta = load_model_meta(model_path)
    assert "feature_importance" in meta
    assert len(meta["feature_importance"]) == len(FEATURE_COLS)
    # Sorted descending
    values = list(meta["feature_importance"].values())
    assert values == sorted(values, reverse=True)


# Improvement 3: model versioning
def test_train_records_feature_cols(cards, tmp_path):
    model_path = str(tmp_path / "model.json")
    rows = generate_training_data(cards)
    train(rows, model_path, device="cpu")
    meta = load_model_meta(model_path)
    assert "feature_cols" in meta
    assert meta["feature_cols"] == FEATURE_COLS


def test_check_model_compatibility_passes(cards, tmp_path):
    model_path = str(tmp_path / "model.json")
    rows = generate_training_data(cards)
    train(rows, model_path, device="cpu")
    assert check_model_compatibility(model_path) is True


def test_check_model_compatibility_fails_on_mismatch(cards, tmp_path):
    model_path = str(tmp_path / "model.json")
    rows = generate_training_data(cards)
    train(rows, model_path, device="cpu")
    # Tamper with meta to simulate old model
    meta_path = model_path.replace(".json", "_meta.json")
    meta = json.loads(Path(meta_path).read_text())
    meta["feature_cols"] = ["rarity_rank", "current_price"]
    Path(meta_path).write_text(json.dumps(meta))
    assert check_model_compatibility(model_path) is False


def test_score_raises_on_incompatible_model(cards, tmp_path):
    from lib.features import extract_features
    model_path = str(tmp_path / "model.json")
    rows = generate_training_data(cards)
    train(rows, model_path, device="cpu")
    # Tamper with meta
    meta_path = model_path.replace(".json", "_meta.json")
    meta = json.loads(Path(meta_path).read_text())
    meta["feature_cols"] = ["rarity_rank"]
    Path(meta_path).write_text(json.dumps(meta))
    features = [extract_features(tid, card) for tid, card in cards.items()]
    with pytest.raises(ValueError, match="feature mismatch"):
        score(features, model_path)


def test_check_compatibility_no_meta(tmp_path):
    """Old models without meta should be considered compatible."""
    assert check_model_compatibility(str(tmp_path / "no_such.json")) is True
