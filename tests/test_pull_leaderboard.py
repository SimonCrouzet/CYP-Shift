"""Pulling the leaderboards stores each board twice: as the latest pull and in a stamped history
directory, so no pull is lost."""

import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pull_leaderboard  # noqa: E402 -- after the scripts path insert


def _config() -> dict:
    def board(cid: int, rows: list[list]) -> dict:
        return {
            "id": cid,
            "type": "leaderboard",
            "props": {"value": {"headers": ["Rank", "Username"], "data": rows}},
        }

    return {
        "components": [
            {"id": 1, "type": "tabitem", "props": {"label": "Live"}},
            {"id": 2, "type": "tabitem", "props": {"label": "CYP2D6"}},
            board(3, [[1, "a"], [2, "b"]]),
            board(4, []),
        ],
        "layout": {
            "id": 0,
            "children": [
                {"id": 1, "children": [{"id": 2, "children": [{"id": 3}, {"id": 4}]}]},
            ],
        },
    }


def test_snapshot_keeps_the_latest_and_a_stamped_copy(tmp_path):
    now = datetime(2026, 10, 29, 9, 5, tzinfo=UTC)
    history, frames = pull_leaderboard.snapshot(_config(), out_dir=tmp_path, now=now)
    assert list(frames) == ["live__cyp2d6"]  # the empty board is skipped
    assert history == tmp_path / "history" / "2026-10-29_09h05Z"
    assert (tmp_path / "live__cyp2d6.csv").read_text() == (history / "live__cyp2d6.csv").read_text()
    assert len(frames["live__cyp2d6"]) == 2


def test_a_second_pull_adds_a_directory_and_replaces_the_latest(tmp_path):
    first = datetime(2026, 10, 29, 9, 5, tzinfo=UTC)
    second = datetime(2026, 10, 29, 21, 10, tzinfo=UTC)
    config = _config()
    pull_leaderboard.snapshot(config, out_dir=tmp_path, now=first)
    config["components"][2]["props"]["value"]["data"].append([3, "c"])
    pull_leaderboard.snapshot(config, out_dir=tmp_path, now=second)
    stamps = sorted(path.name for path in (tmp_path / "history").iterdir())
    assert stamps == ["2026-10-29_09h05Z", "2026-10-29_21h10Z"]
    assert len((tmp_path / "live__cyp2d6.csv").read_text().splitlines()) == 4
    first_copy = tmp_path / "history" / stamps[0] / "live__cyp2d6.csv"
    assert len(first_copy.read_text().splitlines()) == 3


def _config_with_downloads() -> dict:
    config = _config()
    config["components"] += [
        {"id": 5, "type": "downloadbutton", "props": {}},
        {"id": 6, "type": "downloadbutton", "props": {}},
        {"id": 7, "type": "tabitem", "props": {"label": "CYP3A4"}},
    ]
    # button 5 comes right before board 3; button 6, in another tab, before the empty board 4
    config["layout"]["children"] = [
        {
            "id": 1,
            "children": [
                {"id": 2, "children": [{"id": 5}, {"id": 3}]},
                {"id": 7, "children": [{"id": 6}, {"id": 4}]},
            ],
        }
    ]
    config["dependencies"] = [
        {"targets": [[5, "click"]], "api_name": "partial"},
        {"targets": [[6, "click"]], "api_name": "partial_1"},
        {"targets": [[3, "change"]], "api_name": "other"},
    ]
    return config


def test_each_board_is_paired_with_the_download_button_before_it():
    assert pull_leaderboard.download_api_names(_config_with_downloads()) == {
        3: "partial",
        4: "partial_1",
    }


def test_the_download_replaces_the_table_and_a_failed_one_keeps_it(tmp_path, monkeypatch):
    import pandas as pd

    def fake_download(api_name):
        if api_name != "partial":
            raise OSError("no network")
        return pd.DataFrame({"Rank": [1, 2], "Username": ["a", "b"], "ST-RAE std": [0.1, 0.2]})

    monkeypatch.setattr(pull_leaderboard, "download_csv", fake_download)
    _, frames = pull_leaderboard.snapshot(_config_with_downloads(), out_dir=tmp_path)
    assert "ST-RAE std" in frames["live__cyp2d6"].columns
    assert "ST-RAE std" in (tmp_path / "live__cyp2d6.csv").read_text().splitlines()[0]
