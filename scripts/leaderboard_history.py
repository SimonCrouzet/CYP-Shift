"""List one user's rows in every stored leaderboard pull.

    python scripts/leaderboard_history.py [--user SC38] [--output data/leaderboard/sc38_history.csv]

Reads ``data/leaderboard/history/<UTC stamp>/<board>.csv`` (written by ``pull_leaderboard.py``,
overall and per isoform, live and interim) and writes one row per pull, board and entry of the user,
so the board's view of a submission can be followed across pulls even though the live board keeps
only a user's latest entry.
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger("leaderboard_history")

HISTORY = Path("data/leaderboard/history")


def user_rows(history: Path, user: str) -> pd.DataFrame:
    """The user's rows over every pull in ``history``, with the pull stamp and board name."""
    frames = []
    for pull in sorted(p for p in history.iterdir() if p.is_dir()):
        for path in sorted(pull.glob("*.csv")):
            board = pd.read_csv(path)
            if "Username" not in board.columns:
                continue
            names = board["Username"].astype(str).map(lambda s: re.sub(r"<[^>]+>", "", s))
            rows = board[names == user].drop(columns=["Model Report Link"], errors="ignore")
            if len(rows):
                frames.append(rows.assign(pull=pull.name, board=path.stem, entries=len(board)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--user", default="SC38")
    parser.add_argument("--history", type=Path, default=HISTORY)
    parser.add_argument("--output", type=Path, default=Path("data/leaderboard/sc38_history.csv"))
    args = parser.parse_args()
    rows = user_rows(args.history, args.user)
    rows.to_csv(args.output, index=False)
    logger.info(
        "%d rows of %s over %d pulls -> %s",
        len(rows),
        args.user,
        rows["pull"].nunique(),
        args.output,
    )


if __name__ == "__main__":
    main()
