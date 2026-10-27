"""Compute molecular foundation-model embeddings and cache them for the feature block.

``--source mammal`` runs in the **dedicated MAMMAL environment**, not the main one -- MAMMAL
pulls transformers 5 and a full torch/TF stack, which is why it is isolated. It reads the
challenge SMILES (plus any extra CSVs given), canonicalises them with RDKit -- the same
canonicalisation :mod:`cypshift.features.embeddings` uses to look them up -- embeds each with
MAMMAL's encoder (masked-mean over the last hidden state), and writes
``data/external/embeddings/<source>.parquet`` keyed by canonical SMILES, which the main
environment then reads with no MAMMAL dependency.

    HF_HOME=/mnt/wsl/ext/hf_cache \\
      /mnt/wsl/ext/envs/cypshift-mammal/bin/python scripts/compute_embeddings.py \\
      --source mammal --data-dir data/raw

``--source chemeleon`` runs in the main environment: the frozen CheMeleon message-passing
encoder (the checkpoint the graph model initialises from), mean-aggregated over atoms to a
2048-wide vector.

    python scripts/compute_embeddings.py --source chemeleon --aux octant tox21

Only the challenge compounds are embedded by default; pass ``--extra`` CSVs (with a SMILES
column), or ``--aux`` auxiliary sources (main environment only), to cover the data a run turns
on, since an uncached compound is an error at feature time, not a zero-fill.
"""

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

MAMMAL_MODEL = "ibm/biomed.omics.bl.sm.ma-ted-458m"
CACHE_DIR = Path("data/external/embeddings")
CHALLENGE_FILES = (
    "cyp-challenge-TRAIN_inhibition.csv",
    "cyp-challenge-TEST-BLINDED.csv",
    "cyp-challenge-single-concentration-TRAIN.csv",
    "cyp-challenge-TRAIN_TDI.csv",
    "cyp-challenge-TRAIN_Emax.csv",
)
SMILES_PROMPT = (
    "<@TOKENIZER-TYPE=SMILES@MAX-LEN={maxlen}><MOLECULAR_ENTITY>"
    "<MOLECULAR_ENTITY_SMALL_MOLECULE><SEQUENCE_NATURAL_START>{smiles}<SEQUENCE_NATURAL_END><EOS>"
)


def canonical(smiles: str) -> str | None:
    """RDKit canonical SMILES, matching cypshift.data.chembl.canonical."""
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule) if molecule else None


def gather_smiles(data_dir: Path, extra: list[Path]) -> list[str]:
    """Unique canonical SMILES from the challenge files and any extra CSVs."""
    series = []
    for name in CHALLENGE_FILES:
        path = data_dir / name
        if path.exists():
            series.append(pd.read_csv(path)["SMILES"])
    for path in extra:
        series.append(pd.read_csv(path)["SMILES"])
    raw = pd.concat(series, ignore_index=True) if series else pd.Series([], dtype=str)
    canon = {c for c in (canonical(s) for s in raw.dropna().unique()) if c}
    logger.info("gathered %d unique canonical SMILES", len(canon))
    return sorted(canon)


def gather_aux_smiles(data_dir: Path, sources: list[str]) -> list[str]:
    """Unique canonical SMILES of the named auxiliary sources, as the pipeline's loaders return
    them for every enzyme it can pull rows for, the scored ones and their relatives (both qHTS
    quantities, inactives included)."""
    import importlib

    from cypshift.data.sequences import ACCESSIONS

    challenge = pd.read_csv(data_dir / CHALLENGE_FILES[0])["SMILES"]
    found: set[str] = set()
    for source in sources:
        loader = importlib.import_module(f"cypshift.data.{source}").load_auxiliary
        variants = [{}]
        if source in ("tox21", "veith"):
            variants = [{"include_inactives": True}, {"representation": "max_response"}]
        for enzyme in ACCESSIONS:
            for kwargs in variants:
                try:
                    rows = loader(enzyme, challenge, **kwargs)
                except Exception as error:  # noqa: BLE001 -- a source without data for this enzyme
                    logger.info("%s %s %s: %s", source, enzyme, kwargs, error)
                    continue
                found.update(c for c in (canonical(s) for s in rows["SMILES"].unique()) if c)
    logger.info("gathered %d unique canonical SMILES from %s", len(found), sources)
    return sorted(found)


def chemeleon_vectors(smiles: list[str], device: str, batch_size: int = 256) -> np.ndarray:
    """The frozen CheMeleon encoder's mean-aggregated atom states, one row per SMILES."""
    import torch
    from chemprop import data, featurizers
    from chemprop import nn as cpnn

    from cypshift.models.graph import chemeleon_message_passing

    encoder = chemeleon_message_passing(Path("data/external/chemeleon/chemeleon_mp.pt"))
    encoder = encoder.eval().to(device)
    aggregate = cpnn.MeanAggregation()
    featuriser = featurizers.SimpleMoleculeMolGraphFeaturizer()
    out = []
    for start in range(0, len(smiles), batch_size):
        chunk = smiles[start : start + batch_size]
        batch = data.BatchMolGraph([featuriser(Chem.MolFromSmiles(s)) for s in chunk])
        batch.to(device)
        with torch.no_grad():
            out.append(aggregate(encoder(batch), batch.batch).cpu().numpy())
        if (start // batch_size) % 20 == 0:
            logger.info("  embedded %d / %d", start, len(smiles))
    return np.vstack(out)


def mammal_embedder(device: str, maxlen: int) -> Callable[[str], np.ndarray]:
    """A ``smiles -> vector`` callable: MAMMAL's encoder, masked-mean-pooled."""
    import torch
    from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
    from mammal.keys import (
        ENCODER_INPUTS_ATTENTION_MASK,
        ENCODER_INPUTS_STR,
        ENCODER_INPUTS_TOKENS,
        ENCODER_LAST_HIDDEN_STATE,
    )
    from mammal.model import Mammal

    model = Mammal.from_pretrained(MAMMAL_MODEL).eval().to(device)
    tokenizer = ModularTokenizerOp.from_pretrained(MAMMAL_MODEL)

    def embed(smiles: str) -> np.ndarray:
        sample = {ENCODER_INPUTS_STR: SMILES_PROMPT.format(maxlen=maxlen, smiles=smiles)}
        tokenizer(
            sample,
            key_in=ENCODER_INPUTS_STR,
            key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
            key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK,
            max_seq_len=maxlen,
        )
        sample[ENCODER_INPUTS_TOKENS] = torch.tensor(sample[ENCODER_INPUTS_TOKENS], device=device)
        sample[ENCODER_INPUTS_ATTENTION_MASK] = torch.tensor(
            sample[ENCODER_INPUTS_ATTENTION_MASK], device=device
        )
        with torch.no_grad():
            out = model.forward_encoder_only([sample])
        hidden = out[ENCODER_LAST_HIDDEN_STATE][0]
        mask = sample[ENCODER_INPUTS_ATTENTION_MASK].bool()
        return hidden[mask].mean(0).cpu().numpy()

    return embed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="mammal", help="cache name under the embeddings dir")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--extra", type=Path, nargs="*", default=[], help="extra SMILES-column CSVs"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maxlen", type=int, default=512)
    parser.add_argument(
        "--aux", nargs="*", default=[], help="auxiliary sources to embed too (chemeleon only)"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="embed only the compounds missing from the existing cache and append them",
    )
    args = parser.parse_args()

    smiles = gather_smiles(args.data_dir, args.extra)
    existing = None
    if args.source == "chemeleon":
        smiles = sorted({*smiles, *gather_aux_smiles(args.data_dir, args.aux)})
        cached = CACHE_DIR / f"{args.source}.parquet"
        if args.update and cached.exists():
            existing = pd.read_parquet(cached)
            known = set(existing["SMILES"])
            smiles = [s for s in smiles if s not in known]
            logger.info("%d compounds missing from the cache", len(smiles))
            if not smiles:
                return
        vectors = chemeleon_vectors(smiles, args.device)
        model, prompt = "CheMeleon message passing (frozen), mean aggregation", None
    else:
        embed = mammal_embedder(args.device, args.maxlen)
        vectors = []
        for i, one in enumerate(smiles):
            vectors.append(embed(one))
            if i % 200 == 0:
                logger.info("  embedded %d / %d", i, len(smiles))
        vectors = np.vstack(vectors)
        model, prompt = MAMMAL_MODEL, SMILES_PROMPT
    frame = pd.DataFrame(vectors, columns=[f"e{i}" for i in range(vectors.shape[1])])
    frame.insert(0, "SMILES", smiles)
    if existing is not None:
        frame = pd.concat([existing, frame], ignore_index=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{args.source}.parquet"
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary)
    temporary.replace(path)  # a reader never sees a half-written cache
    # A provenance manifest: which model and prompt produced these vectors, and the RDKit
    # canonicalisation the reader keys on, so a cache from a different model is identifiable.
    manifest = {
        "source": args.source,
        "model": model,
        "prompt": prompt,
        "auxiliary_sources": args.aux,
        "max_len": args.maxlen,
        "width": int(vectors.shape[1]),
        "n_compounds": len(frame),
        "canonicalisation": "rdkit MolToSmiles",
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("wrote %d embeddings of width %d to %s", len(frame), vectors.shape[1], path)


if __name__ == "__main__":
    main()
