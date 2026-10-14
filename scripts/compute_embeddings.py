"""Compute molecular foundation-model embeddings and cache them for the feature block.

Runs in the **dedicated MAMMAL environment**, not the main one -- MAMMAL pulls transformers
5 and a full torch/TF stack, which is why it is isolated. It reads the challenge SMILES
(plus any extra CSVs given), canonicalises them with RDKit -- the same canonicalisation
:mod:`cypshift.features.embeddings` uses to look them up -- embeds each with MAMMAL's
encoder (masked-mean over the last hidden state), and writes
``data/external/embeddings/<source>.parquet`` keyed by canonical SMILES, which the main
environment then reads with no MAMMAL dependency.

    HF_HOME=/mnt/wsl/ext/hf_cache \\
      /mnt/wsl/ext/envs/cypshift-mammal/bin/python scripts/compute_embeddings.py \\
      --source mammal --data-dir data

Only the challenge compounds are embedded by default; pass ``--extra`` CSVs (with a SMILES
column) to cover the auxiliary sources a run turns on, since an uncached compound is an
error at feature time, not a zero-fill.
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
CHALLENGE_FILES = ("cyp-challenge-TRAIN_inhibition.csv", "cyp-challenge-TEST-BLINDED.csv")
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
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--extra", type=Path, nargs="*", default=[], help="extra SMILES-column CSVs"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maxlen", type=int, default=512)
    args = parser.parse_args()

    smiles = gather_smiles(args.data_dir, args.extra)
    embed = mammal_embedder(args.device, args.maxlen)
    vectors = []
    for i, one in enumerate(smiles):
        vectors.append(embed(one))
        if i % 200 == 0:
            logger.info("  embedded %d / %d", i, len(smiles))
    vectors = np.vstack(vectors)
    frame = pd.DataFrame(vectors, columns=[f"e{i}" for i in range(vectors.shape[1])])
    frame.insert(0, "SMILES", smiles)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{args.source}.parquet"
    frame.to_parquet(path)
    # A provenance manifest: which model and prompt produced these vectors, and the RDKit
    # canonicalisation the reader keys on, so a cache from a different model is identifiable.
    manifest = {
        "source": args.source,
        "model": MAMMAL_MODEL,
        "prompt": SMILES_PROMPT,
        "max_len": args.maxlen,
        "width": int(vectors.shape[1]),
        "n_compounds": len(frame),
        "canonicalisation": "rdkit MolToSmiles",
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("wrote %d embeddings of width %d to %s", len(frame), vectors.shape[1], path)


if __name__ == "__main__":
    main()
