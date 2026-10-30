"""One user's rows over the stored leaderboard pulls."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import leaderboard_history as script  # noqa: E402 -- after the scripts path insert


def _pull(history: Path, stamp: str, rank: int, score: float) -> None:
    folder = history / stamp
    folder.mkdir(parents=True)
    pd.DataFrame(
        {
            "Rank": [1, rank],
            "Username": ['<a href="x">other</a>', '<a href="y">SC38</a>'],
            "Model Report Link": ["a", "b"],
            "MCC": [0.5, score],
        }
    ).to_csv(folder / "live__overall.csv", index=False)


def test_the_users_rows_are_listed_per_pull_and_board_with_html_stripped(tmp_path):
    _pull(tmp_path, "2026-10-29_10h12Z", 97, 0.26)
    _pull(tmp_path, "2026-10-30_00h07Z", 81, 0.28)
    rows = script.user_rows(tmp_path, "SC38")
    assert rows["pull"].tolist() == ["2026-10-29_10h12Z", "2026-10-30_00h07Z"]
    assert rows["Rank"].tolist() == [97, 81] and rows["MCC"].tolist() == [0.26, 0.28]
    assert set(rows["board"]) == {"live__overall"} and "Model Report Link" not in rows
    assert script.user_rows(tmp_path, "nobody").empty
