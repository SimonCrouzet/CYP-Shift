"""Pull the live and interim challenge leaderboards from the HuggingFace Space.

The Space (``openadmet/cyp-challenge``) is a Gradio app, and its ``/config`` endpoint
carries the initial value of every ``leaderboard`` component -- the full table, not
just what one screen shows. This reads that config, walks the layout tree to name each
board by its tab path (e.g. ``live/regression/CYP2D6``), and writes one CSV per board to
``data/leaderboard/``. No Gradio client needed, nothing is submitted, and the output
goes under the gitignored data directory.

    /mnt/wsl/ext/envs/cypshift/bin/python scripts/pull_leaderboard.py
"""

import json
import logging
import re
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger("leaderboard")

CONFIG_URL = "https://openadmet-cyp-challenge.hf.space/config"
OUT_DIR = Path("data/leaderboard")


def fetch_config(url: str = CONFIG_URL) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 -- fixed https Space URL
        return json.load(response)


def label_by_id(config: dict) -> dict[int, str]:
    """Map each leaderboard component id to a slug built from its tab-label path.

    The layout is a tree of component ids; a ``tabitem``'s label names the section it
    opens, so the stack of tab labels above a leaderboard is its address on the page.
    """
    components = {c["id"]: c for c in config["components"]}
    names: dict[int, str] = {}

    def walk(node: dict, trail: list[str]) -> None:
        cid = node.get("id")
        component = components.get(cid, {})
        kind = component.get("type")
        here = trail
        if kind == "tabitem":
            label = component.get("props", {}).get("label", "")
            here = [*trail, re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_").lower() or str(cid)]
        if kind == "leaderboard":
            # The full tab path, so the live and interim boards (which share subtab
            # labels like "regression/overall") get distinct names.
            names[cid] = "__".join(trail) if trail else str(cid)
        for child in node.get("children", []) or []:
            walk(child, here)

    walk(config["layout"], [])
    return names


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = fetch_config()
    names = label_by_id(config)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for component in config["components"]:
        if component.get("type") != "leaderboard":
            continue
        value = component.get("props", {}).get("value") or {}
        headers, data = value.get("headers"), value.get("data")
        if not headers or not data:
            continue
        name = names.get(component["id"], str(component["id"]))
        frame = pd.DataFrame(data, columns=headers)
        path = OUT_DIR / f"{name}.csv"
        frame.to_csv(path, index=False)
        logger.info("wrote %3d rows to %s", len(frame), path)
        written += 1
    logger.info("wrote %d leaderboards to %s", written, OUT_DIR)


if __name__ == "__main__":
    main()
