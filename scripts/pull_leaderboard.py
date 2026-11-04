"""Pull the live and interim challenge leaderboards from the HuggingFace Space.

The Space (``openadmet/cyp-challenge``) is a Gradio app, and its ``/config`` endpoint
carries the initial value of every ``leaderboard`` component -- the full table, not
just what one screen shows. This reads that config, walks the layout tree to name each
board by its tab path (e.g. ``live/regression/CYP2D6``), and writes one CSV per board to
``data/leaderboard/`` (the latest pull) and a copy of every board, overall and per isoform, under
``data/leaderboard/history/<UTC stamp>/`` (every pull is kept). Anything that reads the boards
should go through ``snapshot`` so the pull is stored. No Gradio client needed, nothing is
submitted, and the output goes under the gitignored data directory.

    /mnt/wsl/ext/envs/cypshift/bin/python scripts/pull_leaderboard.py
"""

import io
import json
import logging
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger("leaderboard")

CONFIG_URL = "https://openadmet-cyp-challenge.hf.space/config"
API_ROOT = "https://openadmet-cyp-challenge.hf.space/gradio_api"
OUT_DIR = Path("data/leaderboard")
HISTORY_DIR = OUT_DIR / "history"


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


def download_api_names(config: dict) -> dict[int, str]:
    """Map each leaderboard component id to the API name of its "Download CSV" button.

    The Space puts a download button right before the board it exports, so the layout order pairs
    them. The download carries the bootstrap standard deviations that the page's table leaves out.
    """
    kinds = {c["id"]: c.get("type") for c in config["components"]}
    order: list[int] = []

    def walk(node: dict) -> None:
        if kinds.get(node.get("id")) in ("leaderboard", "downloadbutton"):
            order.append(node["id"])
        for child in node.get("children", []) or []:
            walk(child)

    walk(config["layout"])
    api_by_button = {
        target: dependency["api_name"]
        for dependency in config.get("dependencies", [])
        for target, event in dependency.get("targets", [])
        if event == "click" and kinds.get(target) == "downloadbutton"
    }
    names: dict[int, str] = {}
    button = None
    for cid in order:
        if kinds[cid] == "downloadbutton":
            button = cid if cid in api_by_button else None
        elif button is not None:
            names[cid] = api_by_button[button]
            button = None
    return names


def download_csv(api_name: str, root: str = API_ROOT) -> pd.DataFrame:
    """What the "Download CSV" button of a board returns (the call the page itself makes)."""
    request = urllib.request.Request(  # noqa: S310 -- fixed https Space URL
        f"{root}/call/{api_name}",
        data=b'{"data": []}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        event_id = json.load(response)["event_id"]
    with urllib.request.urlopen(f"{root}/call/{api_name}/{event_id}", timeout=120) as response:  # noqa: S310
        lines = response.read().decode().splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "event: complete":
            url = json.loads(lines[index + 1].removeprefix("data:").strip())[0]["url"]
            break
    else:
        raise ValueError(f"the download {api_name} did not complete: {' | '.join(lines)[:200]}")
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        return pd.read_csv(io.BytesIO(response.read()))


def boards(config: dict) -> dict[str, pd.DataFrame]:
    """Every non-empty leaderboard of the config as a frame, by tab-path slug."""
    names = label_by_id(config)
    frames: dict[str, pd.DataFrame] = {}
    for component in config["components"]:
        if component.get("type") != "leaderboard":
            continue
        value = component.get("props", {}).get("value") or {}
        headers, data = value.get("headers"), value.get("data")
        if not headers or not data:
            continue
        frames[names.get(component["id"], str(component["id"]))] = pd.DataFrame(
            data, columns=headers
        )
    return frames


def snapshot(
    config: dict | None = None,
    out_dir: Path = OUT_DIR,
    now: datetime | None = None,
) -> tuple[Path, dict[str, pd.DataFrame]]:
    """Pull the boards (unless ``config`` is given), store them and return them.

    Each board is the page's "Download CSV" file (it has the standard-deviation columns), or the
    table in the page's config when the download fails. It is written to ``out_dir`` as the latest
    pull and to ``out_dir/history/<UTC stamp>/`` so no pull is lost; the stamp directory and the
    frames come back for the caller to read.
    """
    config = fetch_config() if config is None else config
    frames = boards(config)
    ids = {name: cid for cid, name in label_by_id(config).items()}
    api_names = download_api_names(config)
    for name, frame in list(frames.items()):
        api_name = api_names.get(ids.get(name))
        if api_name is None:
            continue
        try:
            full = download_csv(api_name)
        except (OSError, ValueError, KeyError, IndexError) as error:
            logger.warning("%s: the CSV download failed (%s); keeping the table", name, error)
            continue
        if len(full) != len(frame):
            logger.warning("%s: download has %d rows, the table %d", name, len(full), len(frame))
        frames[name] = full
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d_%Hh%MZ")
    history = out_dir / "history" / stamp
    history.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
        frame.to_csv(history / f"{name}.csv", index=False)
    logger.info("stored %d leaderboards in %s and %s", len(frames), out_dir, history)
    return history, frames


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot()


if __name__ == "__main__":
    main()
