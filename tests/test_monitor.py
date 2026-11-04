import json

import pytest

from cypshift.models.monitor import (
    CURVES_FILE,
    PrunedError,
    patience_epochs,
    prune_epoch,
    reference_error,
    write_curves,
)


def test_the_pruning_epoch_is_the_half_way_epoch_rounded_up():
    assert [prune_epoch(n) for n in (1, 2, 5, 30)] == [1, 1, 3, 15]


def test_patience_is_a_share_of_the_epochs_and_at_least_one_epoch():
    assert [patience_epochs(0.1, n) for n in (5, 10, 30, 31)] == [1, 1, 3, 4]
    assert patience_epochs(0.0, 30) == 0


def _curve(member, stage, epochs, error):
    return {"member": member, "stage": stage, "epochs": epochs, "error": error}


def test_the_reference_error_is_the_mean_lowest_first_stage_error_of_the_fold(tmp_path):
    write_curves(
        tmp_path,
        {
            0: [
                _curve(1, 1, 4, [1.0, 0.8, 0.6, 0.5]),
                _curve(2, 1, 4, [1.0, 0.7, 0.4, 0.45]),
                _curve(1, 2, 4, [9, 9, 9, 9]),
            ],
            1: [_curve(1, 1, 4, [2.0, 1.0, 1.2, 1.1])],
        },
    )
    assert reference_error(str(tmp_path), 0) == pytest.approx((0.5 + 0.4) / 2)
    assert reference_error(str(tmp_path / CURVES_FILE), 1) == pytest.approx(1.0)


def test_there_is_no_reference_for_a_fold_without_a_first_stage_curve(tmp_path):
    write_curves(tmp_path, {0: [_curve(1, 2, 10, [1.0, 0.9, 0.8])]})
    assert reference_error(str(tmp_path), 0) is None
    assert reference_error(str(tmp_path), 3) is None
    assert json.loads((tmp_path / CURVES_FILE).read_text())["0"][0]["epochs"] == 10


def test_a_pruned_cell_is_an_exception():
    assert issubclass(PrunedError, Exception)
