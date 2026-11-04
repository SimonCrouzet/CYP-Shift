"""The learning-curve export merges the bundle's held-out errors with the log's training losses."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_learning_curves as script  # noqa: E402 -- after the scripts path insert

LOG = """\
10:00:00 [cell_a] fold 1/2 | 10 train, 5 held out
10:00:01 [cell_a] member 1/1 stage 1/1 epoch 1/2  train 0.50  0.1 min elapsed, 0.1 min left
10:00:02 [cell_a] member 1/1 stage 1/1 epoch 2/2  train 0.40  0.2 min elapsed, 0.0 min left
10:00:03 [cell_a] fold 2/2 | 10 train, 5 held out
10:00:04 [cell_a] member 1/1 stage 1/1 epoch 1/2  train 0.60  0.1 min elapsed, 0.1 min left
10:00:05 [cell_a] member 1/1 stage 1/1 epoch 2/2  train 0.30  0.2 min elapsed, 0.0 min left
10:00:06 [other] member 1/1 stage 1/1 epoch 1/2  train 9.99  0.1 min elapsed, 0.1 min left
"""


def test_curves_merge_held_out_errors_with_training_losses_by_fold_and_epoch(tmp_path, monkeypatch):
    bundle = tmp_path / "rev" / "reg_cell_a_seed0"
    bundle.mkdir(parents=True)
    entry = {"member": 1, "stage": 1, "epochs": 2}
    curves = {"0": [{**entry, "error": [0.9, 0.7]}], "1": [{**entry, "error": [0.8, 0.75]}]}
    (bundle / "curves.json").write_text(json.dumps(curves))
    monkeypatch.setattr(script, "BUNDLES", tmp_path)
    table = script.curves("rev", ["cell_a"], LOG)
    assert table[["fold", "epoch", "fold_mae", "train_loss"]].values.tolist() == [
        [0, 1, 0.9, 0.5],
        [0, 2, 0.7, 0.4],
        [1, 1, 0.8, 0.6],
        [1, 2, 0.75, 0.3],
    ]
    summary = script.summary(table).iloc[0]
    assert summary["lowest_fold_mae"] == (0.7 + 0.75) / 2
    assert summary["epoch_of_lowest"] == 2
    assert summary["last_train_loss"] == (0.4 + 0.3) / 2


def test_a_cell_without_a_curves_file_still_gets_its_training_losses(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "BUNDLES", tmp_path)
    table = script.curves("rev", ["cell_a"], LOG)
    assert table["fold_mae"].isna().all() and len(table) == 4


def test_an_untagged_single_cell_log_is_assigned_to_the_named_cell(tmp_path, monkeypatch):
    untagged = "\n".join(line.replace("[cell_a] ", "") for line in LOG.splitlines()[:-1])
    monkeypatch.setattr(script, "BUNDLES", tmp_path)
    table = script.curves("rev", ["cell_a"], [(untagged, "cell_a")])
    assert table["train_loss"].tolist() == [0.5, 0.4, 0.6, 0.3]


def test_a_running_cell_takes_its_held_out_errors_from_the_log_for_finished_folds_only(
    tmp_path, monkeypatch
):
    running = """\
10:00:00 fold 1/2 | 10 train, 5 held out
10:00:01 member 1/1 stage 1/1 epoch 1/2  train 0.50  fold MAE 0.90  0.1 min elapsed, 0.1 min left
10:00:02 member 1/1 stage 1/1 epoch 2/2  train 0.40  fold MAE 0.70  0.2 min elapsed, 0.0 min left
10:00:03 fold 2/2 | 10 train, 5 held out
10:00:04 member 1/1 stage 1/1 epoch 1/2  train 0.60  fold MAE 0.80  0.1 min elapsed, 0.1 min left
"""
    monkeypatch.setattr(script, "BUNDLES", tmp_path)
    table = script.curves("rev", ["cell_a"], [(running, "cell_a")])
    assert table[["fold", "epoch", "fold_mae", "train_loss"]].values.tolist() == [
        [0, 1, 0.9, 0.5],
        [0, 2, 0.7, 0.4],
    ]
    assert list(table.columns) == [
        "name",
        "fold",
        "member",
        "stage",
        "epoch",
        "fold_mae",
        "train_loss",
    ]


def test_the_epochs_so_far_of_an_unfinished_fold_are_kept_on_request(tmp_path, monkeypatch):
    running = "\n".join(
        [
            "10:00:00 fold 1/2 | 10 train, 5 held out",
            "10:00:01 member 1/1 stage 1/1 epoch 1/3  train 0.50  fold MAE 0.90",
            "10:00:02 member 1/1 stage 1/1 epoch 2/3  train 0.40  fold MAE 0.70",
        ]
    )
    monkeypatch.setattr(script, "BUNDLES", tmp_path)
    assert script.curves("rev", ["cell_a"], [(running, "cell_a")]).empty
    kept = script.curves("rev", ["cell_a"], [(running, "cell_a")], keep_partial=True)
    assert kept["epoch"].tolist() == [1, 2] and kept["fold_mae"].tolist() == [0.9, 0.7]
