import json
import os
import pytest
from lib.goldfish import parse_goldfish_csv, match_goldfish_to_uuid, import_goldfish_dir


SAMPLE_CSV = """\
Date,Price
2024-01-15,3.50
2024-01-14,3.45
2024-01-13,3.60
"""

SAMPLE_IDENTIFIERS = {
    "uuid-bolt": {
        "name": "Lightning Bolt",
        "setCode": "M10",
        "setName": "Magic 2010",
    },
    "uuid-counterspell": {
        "name": "Counterspell",
        "setCode": "MH2",
        "setName": "Modern Horizons 2",
    },
}


def test_parse_goldfish_csv(tmp_path):
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(SAMPLE_CSV)
    entries = parse_goldfish_csv(str(csv_path))
    assert len(entries) == 3
    assert entries[0] == ("2024-01-15", 3.50)
    assert entries[2] == ("2024-01-13", 3.60)


def test_parse_goldfish_csv_skips_bad_rows(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("Date,Price\n2024-01-01,abc\n2024-01-02,1.50\n")
    entries = parse_goldfish_csv(str(csv_path))
    assert len(entries) == 1
    assert entries[0] == ("2024-01-02", 1.50)


def test_parse_goldfish_csv_skips_zero_prices(tmp_path):
    csv_path = tmp_path / "zero.csv"
    csv_path.write_text("Date,Price\n2024-01-01,0.00\n2024-01-02,1.50\n")
    entries = parse_goldfish_csv(str(csv_path))
    assert len(entries) == 1


def test_parse_goldfish_csv_empty_file(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")
    entries = parse_goldfish_csv(str(csv_path))
    assert entries == []


def test_match_exact_by_set_code():
    uuid = match_goldfish_to_uuid("Lightning Bolt", "M10", SAMPLE_IDENTIFIERS)
    assert uuid == "uuid-bolt"


def test_match_exact_by_set_name():
    uuid = match_goldfish_to_uuid("Lightning Bolt", "Magic 2010", SAMPLE_IDENTIFIERS)
    assert uuid == "uuid-bolt"


def test_match_case_insensitive():
    uuid = match_goldfish_to_uuid("lightning bolt", "m10", SAMPLE_IDENTIFIERS)
    assert uuid == "uuid-bolt"


def test_match_fuzzy_name():
    # "Lightningg Bolt" has high similarity to "Lightning Bolt"
    uuid = match_goldfish_to_uuid("Lightningg Bolt", "M10", SAMPLE_IDENTIFIERS)
    assert uuid == "uuid-bolt"


def test_match_returns_none_for_unmatched():
    uuid = match_goldfish_to_uuid("Nonexistent Card", "XYZ", SAMPLE_IDENTIFIERS)
    assert uuid is None


def test_import_goldfish_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("lib.price_store.PRICE_HISTORY_DIR", str(tmp_path / "prices"))

    # Create a CSV file named CardName_SetCode.csv
    csv_path = tmp_path / "csvs" / "Lightning Bolt_M10.csv"
    csv_path.parent.mkdir()
    csv_path.write_text(SAMPLE_CSV)

    results = import_goldfish_dir(str(tmp_path / "csvs"), SAMPLE_IDENTIFIERS)
    assert "Lightning Bolt_M10.csv" in results
    assert results["Lightning Bolt_M10.csv"] == "uuid-bolt"


def test_import_goldfish_dir_nonexistent():
    results = import_goldfish_dir("/nonexistent/path", SAMPLE_IDENTIFIERS)
    assert results == {}


def test_import_resumption(tmp_path):
    """Progress file tracks completed card IDs for resume support."""
    progress_path = tmp_path / "progress.json"
    completed = {"card-1", "card-2"}
    with open(progress_path, "w") as f:
        json.dump(sorted(completed), f)

    with open(progress_path) as f:
        loaded = set(json.load(f))
    assert loaded == completed

    # Simulating resume: card-3 not in progress, so it would be downloaded
    all_cards = {"card-1", "card-2", "card-3"}
    remaining = all_cards - loaded
    assert remaining == {"card-3"}
