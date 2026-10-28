"""Compute the per-compound feature blocks once and cache them for every run.

Each fold of each run featurises its whole training table again, and a table that pools the
auxiliary sources holds tens of thousands of rows for a few thousand compounds. This writes the
blocks for every compound the pipeline can see -- the challenge files and the named auxiliary
sources -- to ``data/external/features/<block>.npz``, keyed by the SMILES string as written, which
:mod:`cypshift.features.molecular` reads in place of computing (a compound the cache lacks is
computed as before).

    python scripts/compute_features.py --aux octant tox21 novartis
    python scripts/compute_features.py --aux octant tox21 novartis --update  # only new compounds

Blocks: ``physchem``, ``mechanistic`` and ``morgan_r<radius>_b<bits>`` (count fingerprints, stored
as 16-bit integers).
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from compute_embeddings import gather_aux_smiles, gather_smiles

from cypshift.features import molecular

logger = logging.getLogger("compute_features")
CHUNK = 500


def _block(name: str, chunk: list[str], radius: int, n_bits: int) -> tuple[np.ndarray, list | None]:
    """One block for one chunk of SMILES: the values and the column names (None if unnamed)."""
    if name == "physchem":
        frame = molecular.compute_physicochemical(chunk)
    elif name == "mechanistic":
        frame = molecular.compute_mechanistic(chunk)
    else:
        counts = molecular.compute_morgan_counts(chunk, radius, n_bits)
        return np.clip(counts, 0, 65535).astype(np.uint16), None
    return frame.to_numpy(np.float32), list(frame.columns)


def _existing(name: str) -> tuple[list[str], np.ndarray | None, list | None]:
    """The SMILES, values and columns already cached for a block (empty when there is none)."""
    path = molecular.FEATURE_CACHE_DIR / f"{name}.npz"
    if not path.exists():
        return [], None, None
    with np.load(path, allow_pickle=False) as stored:
        columns = stored["columns"].tolist() if "columns" in stored else None
        return stored["smiles"].tolist(), stored["values"], columns


def _write(name: str, smiles: list[str], values: np.ndarray, columns: list | None) -> None:
    molecular.FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = molecular.FEATURE_CACHE_DIR / f"{name}.npz"
    temporary = path.with_name(path.name + ".tmp.npz")
    arrays = {"smiles": np.array(smiles, dtype=str), "values": values}
    if columns is not None:
        arrays["columns"] = np.array(columns, dtype=str)
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)  # a reader never sees a half-written cache
    logger.info("wrote %s: %d rows x %d columns", path, *values.shape)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--aux", nargs="*", default=[], help="auxiliary sources to cover")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--bits", type=int, default=4096)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--update",
        action="store_true",
        help="compute only the compounds a block's cache lacks and append them",
    )
    args = parser.parse_args()

    smiles = sorted(
        {
            *gather_smiles(args.data_dir, [], raw=True),
            *gather_aux_smiles(args.data_dir, args.aux, raw=True),
        }
    )
    for name in ("physchem", "mechanistic", f"morgan_r{args.radius}_b{args.bits}"):
        known, values, columns = _existing(name) if args.update else ([], None, None)
        seen = set(known)
        todo = [s for s in smiles if s not in seen]
        logger.info("%s: %d compounds, %d to compute", name, len(smiles), len(todo))
        if not todo:
            continue
        chunks = [todo[i : i + CHUNK] for i in range(0, len(todo), CHUNK)]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            parts = list(
                pool.map(
                    _block,
                    [name] * len(chunks),
                    chunks,
                    [args.radius] * len(chunks),
                    [args.bits] * len(chunks),
                )
            )
        fresh = np.vstack([values for values, _ in parts])
        columns = columns if columns is not None else parts[0][1]
        stacked = fresh if values is None else np.vstack([values, fresh.astype(values.dtype)])
        _write(name, known + todo, stacked, columns)


if __name__ == "__main__":
    main()
