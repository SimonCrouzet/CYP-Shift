"""Provenance recording and dataset loading."""

import json

import pandas as pd
import pytest

from cypshift.data.challenge import FILENAMES, load
from cypshift.data.manifest import MANIFEST_SUFFIX, Manifest, sha256_of, verify


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(path, index=False)
    return path


def test_manifest_records_the_source_and_shape(csv_file):
    manifest = Manifest.describe(csv_file, "https://example.org/sample.csv", n_rows=3, n_columns=2)
    assert manifest.source_url == "https://example.org/sample.csv"
    assert manifest.n_rows == 3
    assert manifest.n_bytes == csv_file.stat().st_size
    assert len(manifest.sha256) == 64


def test_manifest_round_trips_through_disk(csv_file):
    original = Manifest.describe(csv_file, "https://example.org/sample.csv")
    written = original.write(csv_file.parent)
    assert json.loads(written.read_text())["filename"] == "sample.csv"
    assert Manifest.read(written) == original


def test_verify_detects_a_changed_file(csv_file):
    Manifest.describe(csv_file, "https://example.org/sample.csv").write(csv_file.parent)
    assert verify(csv_file)
    csv_file.write_text("a,b\n9,9\n")
    assert not verify(csv_file)


def test_verify_is_false_without_a_manifest(csv_file):
    assert not verify(csv_file)


def test_sha256_matches_a_second_reading(csv_file):
    assert sha256_of(csv_file) == sha256_of(csv_file)


def test_unknown_dataset_name_is_rejected():
    with pytest.raises(KeyError, match="unknown dataset"):
        load("not_a_dataset")


def test_every_declared_file_is_a_csv():
    assert set(FILENAMES) == {"inhibition", "screen", "tdi", "emax", "test"}
    assert all(name.endswith(".csv") for name in FILENAMES.values())


def test_manifest_suffix_is_appended_to_the_filename(csv_file):
    manifest = Manifest.describe(csv_file, "https://example.org/sample.csv")
    assert manifest.path_for == "sample.csv" + MANIFEST_SUFFIX
