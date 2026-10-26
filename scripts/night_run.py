"""Sweep the model space overnight on cross-validation, then write submissions.

The boosted source/hyperparameter axis is exhausted (its holdout gains do not touch the
leaderboard, which is ranking- and calibration-limited), so this run goes wide instead:
every model family and its levers, scored on the stratified scaffold 5-fold CV, where the
ranking (Spearman) transfers to the leaderboard even though the ST-RAE level does not.

* Regression -- all five families and their levers: boosted (with the mammal embedding,
  ionization geometry, the established octant+tox21 aux, and deeper trees), neural, the
  D-MPNN graph (CheMeleon and MAMMAL pretraining, bagging, the qHTS response head,
  ionization, and aux), and the per-target ensemble. Compared on the challenge data so the
  families are apples-to-apples; a few aux-stack variants check whether octant+tox21 helps
  the stronger models too. The mammal embedding stays challenge-only because its cache does
  not cover the aux compounds; CheMeleon carries the pretrained-encoder-with-aux combination.
* TDI -- the dose-response curve model under each conditioning strategy (detached
per-assay_condition
  heads, a shared head on an enzyme descriptor, FiLM), scored by MCC out of fold.
* Submissions -- the best regression config (lowest mean macro ST-RAE) and the best TDI
  config (highest mean macro MCC) are each refit on all training data and written to
  ``submissions/``, with their configs saved alongside. These are candidates to inspect,
  not auto-uploaded: the ST-RAE level is optimistic, so judge the bundles on Spearman first.

Resilient: every (phase, name, seed) result is appended to ``data/night_results.csv`` as it
lands under a fixed schema, tagged with the code revision. A re-run skips only rows
completed at the *current* revision; failures (an OOM graph fit, an invalid combination) are
recorded and retried, never treated as done, so one bad config cannot sink the night. Each
config is one full cross-validation; the GPU families make this long, hence a single repeat.

    /mnt/wsl/ext/envs/cypshift/bin/python scripts/night_run.py
"""

import concurrent.futures
import dataclasses
import hashlib
import itertools
import json
import logging
import multiprocessing
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

from cypshift.cli import submit
from cypshift.config import CLASSIFICATION_ENDPOINTS
from cypshift.data.conditioning import DESCRIPTOR_SOURCES
from cypshift.data.sources import supplying
from cypshift.eval.scoring import MACRO_LABEL
from cypshift.experiment import ExperimentConfig
from cypshift.local_env import load_local_env
from cypshift.models.monitor import PrunedError
from cypshift.pipeline import run_cv

logger = logging.getLogger("night")

RESULTS = Path("data/night_results.csv")
BEST_CONFIG = Path("configs/night_best.json")
SUBMISSION = Path("submissions/night_submission_regression.csv")
TDI_BEST_CONFIG = Path("configs/night_best_tdi.json")
TDI_SUBMISSION = Path("submissions/night_submission_classification.csv")
#: Per-run out-of-fold bundles, namespaced by revision, for later blending/calibration.
BUNDLE_DIR = Path("data/run_bundles")
#: Folds for the stratified, scaffold-grouped cross-validation each config is scored on.
N_FOLDS = 5
#: Graph dataloader worker processes, read from the environment so it can be tuned per run
#: without editing tracked code (which would change the results namespace and force a fresh
#: run). Set ``CYP_WORKERS`` before launching: ~4 on a free machine, 2 when sharing it, 0-1
#: when the machine is in use. Only the graph model uses it; it does not affect results.
NUM_WORKERS = int(os.environ.get("CYP_WORKERS", "0"))
#: Ensemble members per config, from the environment: 0 keeps each family's own default.
#: Set ``CYP_MEMBERS=1`` to screen architectures fast (one model, not an ensemble) when the
#: GPU families would otherwise be too slow to cover in a window; ensemble the winners after.
#: Unlike workers it does change results, so a screening run is for ranking, not final scores.
NUM_MEMBERS = int(os.environ.get("CYP_MEMBERS", "0"))
#: Configs trained at the same time, from the environment. Each runs in its own process on the one
#: GPU (a graph fit is bound by kernel-launch latency, not by the GPU), so the cells of one
#: comparison overlap. Does not affect results; ``1`` runs them one after another in this process.
JOBS = int(os.environ.get("CYP_JOBS", "1"))
#: Fold-error monitoring of graph cross-validation, from the environment like the settings above
#: (not part of a recipe's name). ``CYP_MONITOR=1`` evaluates each fold's held-out compounds after
#: every epoch, records the error curves in the run bundle and stops a stage after
#: ``CYP_EARLY_STOP`` (a share of its epochs, default 0.1; 0 never stops) epochs without a lower
#: error. ``CYP_PRUNE_REFERENCE`` (a run bundle directory) abandons a configuration whose fold
#: error at half its epochs is more than ``CYP_PRUNE_TOLERANCE`` (a fraction, default 0.3) above
#: the reference's lowest error for that fold.
MONITOR = os.environ.get("CYP_MONITOR") == "1"
EARLY_STOP = os.environ.get("CYP_EARLY_STOP")
PRUNE_REFERENCE = os.environ.get("CYP_PRUNE_REFERENCE", "")
PRUNE_TOLERANCE = os.environ.get("CYP_PRUNE_TOLERANCE")
#: The queue ``main`` runs: ``full`` is every phase below, ``focus`` is the staged plan of
#: :func:`run_focus`, ``submit`` refits the focus plan's final recipe on all training data
#: (:func:`run_focus_submission`), ``wide`` and ``finetune`` cross-validate the wide members and
#: the staged-training cells (:func:`run_wide`, :func:`run_finetune`).
PLAN = os.environ.get("CYP_PLAN", "full")

#: The established auxiliary set (octant + tox21), which beats the plain baseline on the
#: boosted model; used here to check whether it also helps the stronger families.
OCT_TOX = {"sources.octant": True, "sources.tox21": True}
#: Deeper boosted trees, the tuned recipe from the earlier sweep.
DEEP = {
    "model.boosted.num_leaves": 127,
    "model.boosted.n_estimators": 1000,
    "model.boosted.learning_rate": 0.03,
}

#: A fixed measurement panel for comparing readouts: the screen as a response head, and Veith
#: potency and max-response together.
PANEL = {
    "sources.veith": True,
    "sources.qhts_representation": "both",
    "multifidelity.screen_strategy": "response_head",
    # Pinned so a shared-versus-independent readout comparison is not also a width comparison:
    # chemprop's shared FFN defaults to 300 while the independent heads default to 128 when
    # ``ffn_hidden`` is 0. 300 is the shared default, so shared-readout cells are unchanged;
    # hidden width equal still leaves the independent heads with more parameters (one MLP per task).
    "model.graph.ffn_hidden": 300,
}
#: Initialise the graph encoder from CheMeleon (the checkpoint fixes the message-passing width).
CHEMELEON = {"model.graph.pretraining": "chemeleon"}
#: Independent output heads per (target, assay) potency pair and per-target response heads.
MEASUREMENT_HEADS = {"model.graph.per_measurement_heads": True}
#: The graph-backbone dose-response curve model, the base of the graph-curve variants.
GRAPH_CURVE = {"model.kind": "curve", "model.curve.backbone": "graph"}
#: Every auxiliary source at once -- the big-pool check (known to dilute, re-tested here).
ALL_AUX = {
    "sources.octant": True,
    "sources.tox21": True,
    "sources.veith": True,
    "sources.sair": True,
    "sources.novartis": True,
    "sources.chembl": True,
}

#: Every model family and its levers, scored on the CV. Challenge-only unless the name says
#: an aux source. The mammal cache now covers the octant+tox21 compounds (generated by
#: compute_embeddings.py --extra), so mammal-with-aux is included alongside CheMeleon.
REGRESSION: dict[str, dict] = {
    # The exact recipe we submitted to the leaderboard (octant+tox21, 127-leaf boosted;
    # live MA-ST-RAE 0.859). Scored first and on the same CV as everything else, so every
    # config below is read as a delta against our real reference point.
    "ref_submitted": {**OCT_TOX, "model.boosted.num_leaves": 127},
    # --- Boosted (fast, CPU): objective, features, conditioning, screen, sources ---
    "boosted": {},
    "boosted_absolute": {"model.objective": "absolute"},
    "boosted_squared": {"model.objective": "squared"},
    "boosted_expectile_t60": {"model.objective": "expectile", "model.objective_tau": 0.6},
    "boosted_expectile_t70": {"model.objective": "expectile", "model.objective_tau": 0.7},
    "boosted_pinball_t60": {"model.objective": "pinball", "model.objective_tau": 0.6},
    "boosted_octant_tox21_expectile_t60": {
        **OCT_TOX,
        "model.objective": "expectile",
        "model.objective_tau": 0.6,
    },
    "boosted_mammal": {"features.molecular_embedding": "mammal"},
    "boosted_ionization": {"features.use_ionization_geometry": True},
    "boosted_deep": DEEP,
    "boosted_fp2048": {"features.fingerprint_bits": 2048},
    "boosted_no_physchem": {"features.use_physicochemical": False},
    "boosted_no_mechanistic": {"features.use_mechanistic": False},
    # (boosted + descriptor conditioning is rejected at build time -- a tree has no trunk to
    # modulate -- so it is not a sweepable config; only the one-hot encodings apply.)
    "boosted_cond_metadata": {"conditioning.strategy": "metadata"},
    "boosted_screen_predicted_label": {"multifidelity.screen_strategy": "predicted_label"},
    "boosted_screen_embedding": {"multifidelity.screen_strategy": "embedding"},
    "boosted_macro_balance": {"training.macro_balance": True},
    "boosted_octant_tox21": OCT_TOX,
    "boosted_octant_tox21_deep": {**OCT_TOX, **DEEP},
    "boosted_octant_tox21_absolute": {**OCT_TOX, "model.objective": "absolute"},
    "boosted_octant_tox21_ionization": {**OCT_TOX, "features.use_ionization_geometry": True},
    "boosted_mammal_octant_tox21": {**OCT_TOX, "features.molecular_embedding": "mammal"},
    "boosted_veith": {"sources.veith": True},
    "boosted_veith_no_qhts": {"sources.veith": True, "auxiliary.include_qhts_inactives": False},
    "boosted_octant_tox21_veith": {**OCT_TOX, "sources.veith": True},
    "boosted_sair": {"sources.sair": True},
    "boosted_all_aux": ALL_AUX,
    # --- D-MPNN graph (GPU): the ranking candidates, run early ---
    "graph": {"model.kind": "graph"},
    "graph_absolute": {"model.kind": "graph", "model.objective": "absolute"},
    "graph_squared": {"model.kind": "graph", "model.objective": "squared"},
    "graph_chemeleon": {"model.kind": "graph", "model.graph.pretraining": "chemeleon"},
    "graph_chemeleon_frozen": {
        "model.kind": "graph",
        "model.graph.pretraining": "chemeleon",
        "model.graph.pretraining_finetune": False,
    },
    "graph_mammal": {"model.kind": "graph", "model.graph.pretraining": "mammal"},
    "graph_mammal_bagging": {
        "model.kind": "graph",
        "model.graph.pretraining": "mammal",
        "model.graph.ensemble": "bagging",
    },
    "graph_separated": {"model.kind": "graph", "model.graph.separate_trunks": True},
    "graph_bagging": {"model.kind": "graph", "model.graph.ensemble": "bagging"},
    "graph_chemeleon_bagging": {
        "model.kind": "graph",
        "model.graph.pretraining": "chemeleon",
        "model.graph.ensemble": "bagging",
    },
    "graph_ionization": {"model.kind": "graph", "features.use_ionization_geometry": True},
    "graph_chemeleon_ionization": {
        "model.kind": "graph",
        "model.graph.pretraining": "chemeleon",
        "features.use_ionization_geometry": True,
    },
    "graph_response": {
        "model.kind": "graph",
        "sources.veith": True,
        "sources.qhts_representation": "max_response",
    },
    # Veith potency AND max-response in one run (the reference recipe's coexisting qHTS tasks),
    # so the panel carries both quantities rather than one or the other.
    "graph_veith_both": {
        "model.kind": "graph",
        "sources.veith": True,
        "sources.qhts_representation": "both",
    },
    # Veith qHTS with vs without its inactives, to finally sweep include_qhts_inactives.
    "graph_veith": {"model.kind": "graph", "sources.veith": True},
    "graph_veith_active_only": {
        "model.kind": "graph",
        "sources.veith": True,
        "auxiliary.include_qhts_inactives": False,
    },
    "graph_chemeleon_response": {
        "model.kind": "graph",
        "model.graph.pretraining": "chemeleon",
        "sources.veith": True,
        "sources.qhts_representation": "max_response",
    },
    # The graph only implements factorised/metadata conditioning and the extra_rows screen
    # strategy; descriptor/film/predicted_label are neural-only, so those configs are dropped
    # here rather than failing every run.
    "graph_cond_metadata": {"model.kind": "graph", "conditioning.strategy": "metadata"},
    "graph_macro_balance": {"model.kind": "graph", "training.macro_balance": True},
    "graph_octant_tox21": {"model.kind": "graph", **OCT_TOX},
    # Same aux, but matched against the challenge by InChIKey connectivity block rather than
    # exact canonical SMILES -- a coarser, more aggressive overlap removal.
    "graph_octant_tox21_inchikey": {
        "model.kind": "graph",
        **OCT_TOX,
        "data.dedup_mode": "inchikey_block",
    },
    # Several sources that pool by chemistry (octant/tox21 luminescent, chembl/sair literature),
    # with vs without source-stratified conditioning that keeps them apart.
    "graph_all_aux": {
        "model.kind": "graph",
        **OCT_TOX,
        "sources.chembl": True,
        "sources.sair": True,
        "sources.veith": True,
    },
    "graph_all_aux_source_heads": {
        "model.kind": "graph",
        **OCT_TOX,
        "sources.chembl": True,
        "sources.sair": True,
        "sources.veith": True,
        "model.graph.source_heads": True,
    },
    "graph_chemeleon_octant_tox21": {
        "model.kind": "graph",
        "model.graph.pretraining": "chemeleon",
        **OCT_TOX,
    },
    "graph_mammal_octant_tox21": {
        "model.kind": "graph",
        "model.graph.pretraining": "mammal",
        **OCT_TOX,
    },
    "graph_chemeleon_veith": {
        "model.kind": "graph",
        "model.graph.pretraining": "chemeleon",
        "sources.veith": True,
    },
    "graph_expectile_t60": {
        "model.kind": "graph",
        "model.objective": "expectile",
        "model.objective_tau": 0.6,
    },
    "graph_chemeleon_pinball_t60": {
        "model.kind": "graph",
        "model.graph.pretraining": "chemeleon",
        "model.objective": "pinball",
        "model.objective_tau": 0.6,
    },
    "graph_chemeleon_expectile_t60": {
        "model.kind": "graph",
        "model.graph.pretraining": "chemeleon",
        "model.objective": "expectile",
        "model.objective_tau": 0.6,
    },
    # Trains the preincubated assay_condition so the graph serves the TDI track too (scored by MCC),
    # the regression+classification link over one architecture.
    "graph_chemeleon_tdi": {
        "model.kind": "graph",
        "model.graph.pretraining": "chemeleon",
        "sources.tdi_assay_condition": True,
    },
    # --- Neural MLP (GPU) ---
    "neural": {"model.kind": "neural"},
    "neural_absolute": {"model.kind": "neural", "model.objective": "absolute"},
    "neural_squared": {"model.kind": "neural", "model.objective": "squared"},
    "neural_mammal": {"model.kind": "neural", "features.molecular_embedding": "mammal"},
    "neural_ionization": {"model.kind": "neural", "features.use_ionization_geometry": True},
    "neural_cond_descriptor": {"model.kind": "neural", "conditioning.strategy": "descriptor"},
    "neural_cond_metadata": {"model.kind": "neural", "conditioning.strategy": "metadata"},
    "neural_cond_film": {"model.kind": "neural", "conditioning.strategy": "film"},
    "neural_screen_predicted_label": {
        "model.kind": "neural",
        "multifidelity.screen_strategy": "predicted_label",
    },
    "neural_macro_balance": {"model.kind": "neural", "training.macro_balance": True},
    "neural_octant_tox21": {"model.kind": "neural", **OCT_TOX},
    "neural_octant_tox21_absolute": {
        "model.kind": "neural",
        **OCT_TOX,
        "model.objective": "absolute",
    },
    "neural_mammal_octant_tox21": {
        "model.kind": "neural",
        **OCT_TOX,
        "features.molecular_embedding": "mammal",
    },
    "neural_veith": {"model.kind": "neural", "sources.veith": True},
    "neural_expectile_t60": {
        "model.kind": "neural",
        "model.objective": "expectile",
        "model.objective_tau": 0.6,
    },
    # Per-target ensemble (one learner per endpoint) has no builder in pipeline.build_model;
    # its per-isoform hypothesis is already carried by graph_separated (separate_trunks), so
    # its configs are dropped rather than failing every run.
    # --- Tabular foundation model (TabPFN, per-endpoint over dense features) ---
    # Needs TABPFN_TOKEN in the environment; without it these fail (recorded) and are skipped.
    "tfm": {"model.kind": "tfm"},
    "tfm_mammal": {"model.kind": "tfm", "features.molecular_embedding": "mammal"},
    # (tfm + auxiliary sources is rejected: the tabular model trains only on reference-assay
    # rows and would drop them, so there is no tfm_*_octant_tox21 config.)
    # TabPFN's PCA width. The feature-set sweep varies which blocks TabPFN sees but leaves
    # the compression fixed at 500; vary it on the default (fingerprint) set, where PCA bites.
    "tfm_pca250": {"model.kind": "tfm", "model.tfm.pca_components": 250},
    "tfm_pca1000": {"model.kind": "tfm", "model.tfm.pca_components": 1000},
    # --- Auxiliary selection: how far out the binding-site distance reaches, and how a
    #     relative's weight falls off. Never swept before. These bite only with a cross-enzyme
    #     source on, so they sit on graph+ChEMBL (ChEMBL spans the CYP family): distance 0.0
    #     admits only the scored enzymes' own rows, 1.5 adds seven CYPs, 3.0 adds seventeen. ---
    "graph_chembl": {"model.kind": "graph", "sources.chembl": True},
    "graph_chembl_dist0": {
        "model.kind": "graph",
        "sources.chembl": True,
        "auxiliary.max_distance": 0.0,
    },
    "graph_chembl_dist3": {
        "model.kind": "graph",
        "sources.chembl": True,
        "auxiliary.max_distance": 3.0,
    },
    "graph_chembl_dist3_target_lo": {
        "model.kind": "graph",
        "sources.chembl": True,
        "auxiliary.max_distance": 3.0,
        "auxiliary.target_weight": 0.3,
    },
    "graph_chembl_uniform": {
        "model.kind": "graph",
        "sources.chembl": True,
        "auxiliary.weighting": "uniform",
    },
    # Down-weight the scored enzymes' own outside-assay rows on the best aux config.
    "graph_octant_tox21_scored_lo": {
        "model.kind": "graph",
        **OCT_TOX,
        "auxiliary.scored_weight": 0.5,
    },
    # --- Screen weight: on by default at 0.3; vary it. ---
    "graph_screen_weight_lo": {"model.kind": "graph", "multifidelity.screen_weight": 0.1},
    "graph_screen_weight_hi": {"model.kind": "graph", "multifidelity.screen_weight": 0.6},
    # --- Graph levers on the plain-graph and octant+tox21 baselines; all graph-only or
    #     graph+aux. See notes/performance_analysis_and_next_steps_2026-10-22.md. ---
    # Screen routed to the response head instead of the isotonic pIC50 pseudo-label.
    "graph_screen_response": {
        "model.kind": "graph",
        "multifidelity.screen_strategy": "response_head",
    },
    "graph_octant_tox21_screen_response": {
        "model.kind": "graph",
        **OCT_TOX,
        "multifidelity.screen_strategy": "response_head",
    },
    # Gated CYP2D6-specific adaptation block on the shared trunk.
    "graph_adapt_2d6": {"model.kind": "graph", "model.graph.adapt_target": "CYP2D6"},
    "graph_octant_tox21_adapt_2d6": {
        "model.kind": "graph",
        **OCT_TOX,
        "model.graph.adapt_target": "CYP2D6",
    },
    # Point-pull and interval-aware ranking terms on the soft-threshold loss, two weights each.
    "graph_point_loss_lo": {"model.kind": "graph", "model.point_loss_weight": 0.1},
    "graph_point_loss_hi": {"model.kind": "graph", "model.point_loss_weight": 0.5},
    "graph_ranking_loss_lo": {"model.kind": "graph", "model.ranking_loss_weight": 0.1},
    "graph_ranking_loss_hi": {"model.kind": "graph", "model.ranking_loss_weight": 0.5},
    "graph_octant_tox21_point_ranking": {
        "model.kind": "graph",
        **OCT_TOX,
        "model.point_loss_weight": 0.3,
        "model.ranking_loss_weight": 0.3,
    },
    # Tox21 censored inactives, and a per-endpoint auxiliary weight budget, on octant+tox21.
    "graph_octant_tox21_inactives": {
        "model.kind": "graph",
        **OCT_TOX,
        "auxiliary.tox21_include_inactives": True,
    },
    "graph_octant_tox21_budget_half": {
        "model.kind": "graph",
        **OCT_TOX,
        "auxiliary.per_endpoint_budget": 0.5,
    },
    "graph_octant_tox21_budget_one": {
        "model.kind": "graph",
        **OCT_TOX,
        "auxiliary.per_endpoint_budget": 1.0,
    },
    # --- 2026-10-22 updated-review Tier C levers (graph), all graph-only or graph+aux. ---
    # Raw measured log2 fold-change as its own screen response task (vs the inhibited fraction).
    "graph_screen_log2fc": {
        "model.kind": "graph",
        "multifidelity.screen_strategy": "log2fc_response",
    },
    "graph_octant_tox21_screen_log2fc": {
        "model.kind": "graph",
        **OCT_TOX,
        "multifidelity.screen_strategy": "log2fc_response",
    },
    # Screen fraction and qHTS max-response as simultaneous response tasks (the N-head panel).
    "graph_screen_and_qhts_response": {
        "model.kind": "graph",
        "sources.veith": True,
        "sources.qhts_representation": "max_response",
        "multifidelity.screen_strategy": "response_head",
    },
    # Shared encoder, one value-task output head per target.
    "graph_per_target_heads": {"model.kind": "graph", "model.graph.per_target_heads": True},
    "graph_octant_tox21_per_target_heads": {
        "model.kind": "graph",
        **OCT_TOX,
        "model.graph.per_target_heads": True,
    },
    # One independent output head per (target, assay) potency pair and per-target response heads
    # on the shared encoder (the reference recipes' masked output table).
    "graph_measurement_heads": {"model.kind": "graph", "model.graph.per_measurement_heads": True},
    "graph_octant_tox21_measurement_heads": {
        "model.kind": "graph",
        **OCT_TOX,
        "model.graph.per_measurement_heads": True,
    },
    # The readout-architecture comparison on one fixed, full measurement panel (screen response
    # + Veith potency and max-response), so a difference is the readout and not the data: the
    # conditioned shared readout (control), per-target potency heads, and the full table.
    "graph_panel": {"model.kind": "graph", **PANEL},
    "graph_panel_per_target_heads": {
        "model.kind": "graph",
        **PANEL,
        "model.graph.per_target_heads": True,
    },
    "graph_panel_measurement_heads": {
        "model.kind": "graph",
        **PANEL,
        "model.graph.per_measurement_heads": True,
    },
    # Wide readout FFN over the default encoder (the reference recipes' wide-FFN shape).
    "graph_ffn1024": {"model.kind": "graph", "model.graph.ffn_hidden": 1024},
    # Inverse-variance measurement weighting (precision-weighted labels).
    "graph_inverse_variance": {"model.kind": "graph", "training.inverse_variance_weighting": True},
    "graph_octant_tox21_inverse_variance": {
        "model.kind": "graph",
        **OCT_TOX,
        "training.inverse_variance_weighting": True,
    },
    # --- ECFP6 (radius 3) instead of the default ECFP4, on the fingerprint-reading families. ---
    "boosted_octant_tox21_ecfp6": {**OCT_TOX, "features.fingerprint_radius": 3},
    "neural_ecfp6": {"model.kind": "neural", "features.fingerprint_radius": 3},
    # --- Conditioning strategies only the neural trunk implements (a learned (enzyme, assay)
    #     table, and a per-condition affine calibration layer). The graph takes only one-hot. ---
    "neural_cond_embedding": {"model.kind": "neural", "conditioning.strategy": "embedding"},
    "neural_cond_calibration": {"model.kind": "neural", "conditioning.strategy": "calibration"},
    # --- Cross-model: the dose-response curve head on a chemprop D-MPNN backbone (the graph
    #     family's learned encoder in place of the curve model's fingerprint MLP). The curve's
    #     regression midpoint, scored against the plain graph and the fingerprint-MLP curve. ---
    "curve_graph": {"model.kind": "curve", "model.curve.backbone": "graph"},
    "curve_graph_chemeleon": {
        "model.kind": "curve",
        "model.curve.backbone": "graph",
        "model.curve.pretraining": "chemeleon",
    },
}

#: The TDI track: the dose-response curve model under each conditioning strategy, with the
#: preincubated assay_condition on so its two assay_conditions can be compared into an is_TDI call.
TDI: dict[str, dict] = {
    "curve_heads": {
        "model.kind": "curve",
        "sources.tdi_assay_condition": True,
        "model.curve.conditioning": "heads",
    },
    "curve_descriptor": {
        "model.kind": "curve",
        "sources.tdi_assay_condition": True,
        "model.curve.conditioning": "descriptor",
    },
    "curve_film": {
        "model.kind": "curve",
        "sources.tdi_assay_condition": True,
        "model.curve.conditioning": "film",
    },
    "curve_heads_ionization": {
        "model.kind": "curve",
        "sources.tdi_assay_condition": True,
        "model.curve.conditioning": "heads",
        "features.use_ionization_geometry": True,
    },
    "curve_heads_tdishift_15": {
        "model.kind": "curve",
        "sources.tdi_assay_condition": True,
        "model.curve.conditioning": "heads",
        "model.curve.tdi_shift_threshold": 0.15,
    },
    "curve_heads_tdishift_50": {
        "model.kind": "curve",
        "sources.tdi_assay_condition": True,
        "model.curve.conditioning": "heads",
        "model.curve.tdi_shift_threshold": 0.5,
    },
    "curve_heads_losstau_60": {
        "model.kind": "curve",
        "sources.tdi_assay_condition": True,
        "model.curve.conditioning": "heads",
        "model.curve.loss_tau": 0.6,
    },
    # Deeper curve backbone under the same dose-response head (pluggable-backbone lever).
    "curve_heads_deep": {
        "model.kind": "curve",
        "sources.tdi_assay_condition": True,
        "model.curve.conditioning": "heads",
        "model.curve.trunk_depth": 4,
    },
    # Cross-model: the curve head on a chemprop D-MPNN backbone -- the curve's two reads
    # (direct for regression, direct-vs-preincubated for TDI) on the graph family's learned
    # encoder rather than the fingerprint MLP, optionally CheMeleon-initialised.
    "curve_graph_heads": {
        "model.kind": "curve",
        "sources.tdi_assay_condition": True,
        "model.curve.conditioning": "heads",
        "model.curve.backbone": "graph",
    },
    "curve_graph_chemeleon_heads": {
        "model.kind": "curve",
        "sources.tdi_assay_condition": True,
        "model.curve.conditioning": "heads",
        "model.curve.backbone": "graph",
        "model.curve.pretraining": "chemeleon",
    },
    "curve_graph_descriptor": {
        "model.kind": "curve",
        "sources.tdi_assay_condition": True,
        "model.curve.conditioning": "descriptor",
        "model.curve.backbone": "graph",
    },
}


def _ofat(base_name: str, base: dict, options: dict[str, list]) -> dict[str, dict]:
    """One config per (hyperparameter, value): the base plus that single override.

    One-factor-at-a-time around each family's defaults, rather than a full grid (which would
    be combinatorially impossible). Each generated config is named ``<base>_<field><value>``.
    """
    configs = {}
    for key, values in options.items():
        field = key.split(".")[-1]
        for value in values:
            tag = str(value).replace(".", "p").replace("-", "m")
            configs[f"{base_name}_{field}{tag}"] = {**base, key: value}
    return configs


#: Per-family hyperparameters, each swept one-factor-at-a-time around the family default.
BOOSTED_HP = {
    "model.boosted.num_leaves": [31, 127],
    "model.boosted.learning_rate": [0.03, 0.1],
    "model.boosted.n_estimators": [300, 1000],
    "model.boosted.min_child_samples": [5, 50],
    "model.boosted.subsample": [0.6, 1.0],
    "model.boosted.colsample_bytree": [0.4, 0.8],
}
NEURAL_HP = {
    "model.neural.hidden": [128, 512],
    "model.neural.dropout": [0.1, 0.3],
    "model.neural.learning_rate": [0.0003, 0.003],
    "model.neural.epochs": [100],
}
GRAPH_HP = {
    "model.graph.hidden": [200, 600],
    "model.graph.depth": [2, 5],
    "model.graph.dropout": [0.0, 0.2],
    "model.graph.epochs": [60],
    "model.graph.batch_size": [32],
}
CURVE_HP = {
    "model.curve.hidden": [128, 512],
    "model.curve.dropout": [0.1, 0.3],
    "model.curve.epochs": [100],
    "model.curve.screen_noise": [0.05, 0.2],
}

REGRESSION.update(_ofat("boosted", {}, BOOSTED_HP))
REGRESSION.update(_ofat("neural", {"model.kind": "neural"}, NEURAL_HP))
REGRESSION.update(_ofat("graph", {"model.kind": "graph"}, GRAPH_HP))


def _feature_set(**on: object) -> dict:
    """A fully-specified feature set: every block stated on/off, so the set is the swept
    unit (not a single toggle layered on defaults)."""
    blocks = {
        "features.use_fingerprint": False,
        "features.use_physicochemical": False,
        "features.use_mechanistic": False,
        "features.use_ionization_geometry": False,
        "features.molecular_embedding": "none",
    }
    return {**blocks, **on}


#: Feature sets swept as a first-class axis. The feature set is a lever, not a fixed choice,
#: so each model is run over the same panel of block combinations.
FEATURE_SETS = {
    "fp": _feature_set(**{"features.use_fingerprint": True}),
    "dense": _feature_set(
        **{"features.use_physicochemical": True, "features.use_mechanistic": True}
    ),
    "fp_dense": _feature_set(
        **{
            "features.use_fingerprint": True,
            "features.use_physicochemical": True,
            "features.use_mechanistic": True,
        }
    ),
    "dense_ion": _feature_set(
        **{
            "features.use_physicochemical": True,
            "features.use_mechanistic": True,
            "features.use_ionization_geometry": True,
        }
    ),
    "dense_emb": _feature_set(
        **{
            "features.use_physicochemical": True,
            "features.use_mechanistic": True,
            "features.molecular_embedding": "mammal",
        }
    ),
    "emb": _feature_set(**{"features.molecular_embedding": "mammal"}),
    "full": _feature_set(
        **{
            "features.use_fingerprint": True,
            "features.use_physicochemical": True,
            "features.use_mechanistic": True,
            "features.use_ionization_geometry": True,
            "features.molecular_embedding": "mammal",
        }
    ),
}


def _feature_sweep(prefix: str, base: dict) -> dict[str, dict]:
    """One config per feature set, on top of a base model config."""
    return {f"{prefix}_{name}": {**base, **overrides} for name, overrides in FEATURE_SETS.items()}


# Sweep the feature set for the tabular models and the neural network, where the input
# representation is the lever (TabPFN especially; the boosted baseline for comparison).
REGRESSION.update(_feature_sweep("tfm_fs", {"model.kind": "tfm"}))
REGRESSION.update(_feature_sweep("boosted_fs", {}))


#: Graph settings that only matter once fold monitoring is on.
MONITOR_FIELDS = (
    "early_stopping_patience",
    "prune_reference",
    "prune_tolerance",
)


def _resolved_key(overrides: dict) -> str:
    """The fully resolved settings of one entry, for comparing entries under different names."""
    config = ExperimentConfig()
    experiment_logger = logging.getLogger("cypshift.experiment")
    previous = experiment_logger.level
    experiment_logger.setLevel(logging.WARNING)  # ``override`` logs every key at INFO
    try:
        for key, value in overrides.items():
            config.override(key, value)
    finally:
        experiment_logger.setLevel(previous)
    resolved = dataclasses.asdict(config)
    graph = resolved["model"]["graph"]
    del graph["matmul_precision"]  # a speed setting: the same recipe under either precision
    for field, default in (("ffn_layers", 1), ("batch_norm", False), ("aggregation", "mean")):
        if graph[field] == default:
            del graph[field]  # recorded names predate these settings
    auxiliary = resolved["auxiliary"]
    if auxiliary["row_selection"] == "none":
        for field in ("row_selection", "max_rows_per_group", "row_share", "row_selection_assays"):
            del auxiliary[field]
    else:
        if auxiliary["row_share"] == 0.0:
            del auxiliary["row_share"]
        if not auxiliary["row_selection_assays"]:
            del auxiliary["row_selection_assays"]
    if not graph["monitor_fold"]:
        # An inactive opt-in setting is not part of a recipe's identity, so names recorded before
        # the setting existed still match.
        for field in ("monitor_fold", *MONITOR_FIELDS):
            del graph[field]
    return repr(resolved)


def _add_distinct(pool: dict[str, dict], new: dict[str, dict]) -> list[str]:
    """Add the entries whose resolved settings no declared entry already has.

    Returns, for each entry of ``new`` in order, the name under which its settings are declared:
    its own, or the earlier entry it duplicates.
    """
    declared = {_resolved_key(overrides): name for name, overrides in pool.items()}
    names = []
    for name, overrides in new.items():
        key = _resolved_key(overrides)
        if key not in declared:
            pool[name] = overrides
            declared[key] = name
        names.append(declared[key])
    return names


#: The dense blocks a graph encoder reads beside the molecular graph (as ``x_d``). The Morgan
#: fingerprint is not an input there, and the MAMMAL embedding enters through
#: ``model.graph.pretraining``.
GRAPH_DENSE_BLOCKS = {
    "physchem": "features.use_physicochemical",
    "mechanistic": "features.use_mechanistic",
    "ionization": "features.use_ionization_geometry",
}


def _dense_block_sweep(prefix: str, base: dict) -> dict[str, dict]:
    """Every on/off combination of :data:`GRAPH_DENSE_BLOCKS` on top of a base config."""
    sweep = {}
    for flags in itertools.product((False, True), repeat=len(GRAPH_DENSE_BLOCKS)):
        on = [name for name, flag in zip(GRAPH_DENSE_BLOCKS, flags, strict=True) if flag]
        blocks = dict(zip(GRAPH_DENSE_BLOCKS.values(), flags, strict=True))
        sweep[f"{prefix}_fs_{'_'.join(on) or 'none'}"] = {**base, **blocks}
    return sweep


#: Enzyme descriptor sources for the descriptor-conditioned models, each alone and all together.
#: ``curated`` alone is the existing ``neural_cond_descriptor`` / ``curve_descriptor`` entry.
DESCRIPTOR_SETS = {
    **{source: (source,) for source in DESCRIPTOR_SOURCES if source != "curated"},
    "all": tuple(DESCRIPTOR_SOURCES),
}

_add_distinct(REGRESSION, _feature_sweep("neural_fs", {"model.kind": "neural"}))
_add_distinct(REGRESSION, _dense_block_sweep("graph", {"model.kind": "graph"}))
_add_distinct(
    REGRESSION, _dense_block_sweep("graph_octant_tox21", {"model.kind": "graph", **OCT_TOX})
)
_add_distinct(TDI, _feature_sweep("curve_fs", TDI["curve_heads"]))
_add_distinct(TDI, _dense_block_sweep("curve_graph_heads", TDI["curve_graph_heads"]))
for _set, _sources in DESCRIPTOR_SETS.items():
    REGRESSION[f"neural_cond_descriptor_{_set}"] = {
        "model.kind": "neural",
        "conditioning.strategy": "descriptor",
        "conditioning.descriptor_sources": _sources,
    }
    TDI[f"curve_descriptor_{_set}"] = {
        **TDI["curve_descriptor"],
        "model.curve.descriptor_sources": _sources,
    }
    TDI[f"curve_graph_descriptor_{_set}"] = {
        **TDI["curve_graph_descriptor"],
        "model.curve.descriptor_sources": _sources,
    }


#: The qHTS panel as a data-side axis, crossed with the model families that can read it: Veith
#: potency with its inactives kept as censored rows (the default), potency from the active
#: compounds only, the 0-1 max-response alone, and both quantities (with and without the
#: inactives). The graph reads the max-response through response heads; boosted, neural and the
#: fingerprint curve read it through a helper model's predicted response (see
#: ``cypshift.data.screen_model.fit_response_model``).
def _qhts_variants() -> dict[str, dict]:
    """The qHTS cells of one base: every source set that can supply a max-response (each such
    source alone, then all of them together) read as the max-response, as both quantities, and as
    the potency control, then Veith's potency and both-quantity cells with its inactives dropped.
    The source sets come from ``cypshift.data.sources``, not from a list of names here."""
    suppliers = supplying("max_response")
    source_sets = [(name,) for name in suppliers]
    if len(suppliers) > 1:
        source_sets.append(suppliers)
    variants: dict[str, dict] = {}
    for representation in ("max_response", "both", "pic50"):
        for names in source_sets:
            settings = {f"sources.{name}": True for name in names}
            if representation != "pic50":
                settings["sources.qhts_representation"] = representation
            variants["_".join(names) + f"_{representation}"] = settings
    variants["veith_pic50_active_only"] = {
        "sources.veith": True,
        "auxiliary.include_qhts_inactives": False,
    }
    variants["veith_both_active_only"] = {
        "sources.veith": True,
        "sources.qhts_representation": "both",
        "auxiliary.include_qhts_inactives": False,
    }
    return variants


QHTS_VARIANTS = _qhts_variants()


def _qhts_grid(pool: dict[str, dict], bases: tuple[str, ...]) -> list[str]:
    """Every (base, qHTS variant) cell of ``pool``, as the declared names, in the order of
    ``bases`` and, within a base, of ``QHTS_VARIANTS`` (the max-response cells first)."""
    names: list[str] = []
    for base in bases:
        cells = {
            f"{base}_{tag}": {**pool[base], **variant} for tag, variant in QHTS_VARIANTS.items()
        }
        names += _add_distinct(pool, cells)
    return list(dict.fromkeys(names))


#: The recipe with the lowest macro ST-RAE so far (0.6415): the graph with Octant, Tox21 and
#: Novartis. The qHTS grid starts from it, then goes through the other bases.
(QHTS_BEST_BASE,) = _add_distinct(
    REGRESSION,
    {
        "graph_src_octant_tox21_novartis": {
            **REGRESSION["graph_octant_tox21"],
            "sources.novartis": True,
        }
    },
)

QHTS_REGRESSION = tuple(
    _qhts_grid(
        REGRESSION,
        (
            QHTS_BEST_BASE,
            "graph_octant_tox21",
            "graph",
            "graph_chemeleon",
            "neural_octant_tox21",
            "neural",
            "boosted_octant_tox21",
            "boosted",
        ),
    )
)
QHTS_TDI = tuple(_qhts_grid(TDI, ("curve_heads",)))

#: Data-side options swept around the best recipe measured so far, with the model held fixed:
#: the graph with Octant and active-only Tox21 auxiliaries. Three groups: (1) the source SET, every
#: subset of the six auxiliary sources of size one and two, the best recipe plus each remaining
#: source, and all six; (2) one-factor passes on the screen, the TDI assay condition, dedup,
#: Tox21 inactives, per-endpoint budget and the row weighting; (3) one-factor passes on the
#: auxiliary-selection fields, which act only when ChEMBL is on, around the best recipe plus ChEMBL.
#: The qHTS representation and its inactives are the grid above (``QHTS_REGRESSION``).
DATA_BASE = "graph_octant_tox21"
AUXILIARY_SOURCES = ("octant", "tox21", "chembl", "sair", "veith", "novartis")
HEAVY_SOURCES = ("chembl", "sair", "veith")


def _source_set_name(subset: tuple[str, ...]) -> str:
    return "graph_src_" + ("_".join(subset) if subset else "none")


def _source_sets() -> list[tuple[str, ...]]:
    sets = [()]
    for size in (1, 2):
        sets += list(itertools.combinations(AUXILIARY_SOURCES, size))
    best = ("octant", "tox21")
    sets += [best + (source,) for source in AUXILIARY_SOURCES if source not in best]
    sets.append(AUXILIARY_SOURCES)
    return sets


def _data_side_regression() -> tuple[str, ...]:
    base = REGRESSION[DATA_BASE]
    base_chembl = {**base, "sources.chembl": True}
    panels: dict[str, dict] = {}
    for subset in _source_sets():
        panels[_source_set_name(subset)] = {
            "model.kind": "graph",
            **{f"sources.{source}": True for source in subset},
        }
    panels.update(
        _ofat(
            DATA_BASE,
            base,
            {
                "sources.screen": [False],
                "multifidelity.screen_strategy": ["response_head", "log2fc_response"],
                "multifidelity.screen_weight": [0.1, 0.6, 1.0],
                "data.dedup_mode": ["inchikey_block"],
                "auxiliary.tox21_include_inactives": [True],
                "auxiliary.per_endpoint_budget": [0.25, 0.5, 1.0],
                "training.macro_balance": [True],
                "training.inverse_variance_weighting": [True],
            },
        )
    )
    with_tdi = {**base, "sources.tdi_assay_condition": True}
    panels[f"{DATA_BASE}_tdi"] = with_tdi
    panels.update(_ofat(f"{DATA_BASE}_tdi", with_tdi, {"multifidelity.tdi_weight": [0.25, 1.0]}))
    panels[f"{DATA_BASE}_chembl"] = base_chembl
    panels.update(
        _ofat(
            f"{DATA_BASE}_chembl",
            base_chembl,
            {
                "auxiliary.max_distance": [0.0, 3.0],
                "auxiliary.weighting": ["uniform"],
                "auxiliary.scale": [0.5, 2.0],
                "auxiliary.target_weight": [0.3, 3.0],
                "auxiliary.scored_weight": [0.5, 2.0],
                "auxiliary.per_endpoint_budget": [0.5],
            },
        )
    )
    names = list(dict.fromkeys(_add_distinct(REGRESSION, panels)))
    # the qHTS representation cells of the best recipe belong to the same group
    names += [n for n in QHTS_REGRESSION if n.startswith("graph_octant_tox21_veith")]

    def cost(name: str) -> int:
        config = REGRESSION[name]
        return sum(bool(config.get(f"sources.{source}")) for source in HEAVY_SOURCES)

    return tuple(sorted(dict.fromkeys(names), key=cost))


DATA_SIDE_REGRESSION = _data_side_regression()


@dataclasses.dataclass(frozen=True)
class GreedyStep:
    """One data-side field and the values tried for it, when the current recipe satisfies
    ``requires`` (pairs of field and value that must already hold). ``implies`` are fields every
    candidate of the step sets itself, so the step does not wait on another step adopting them."""

    key: str
    values: tuple
    requires: tuple[tuple[str, object], ...] = ()
    implies: tuple[tuple[str, object], ...] = ()


_VEITH = (("sources.veith", True),)
_CHEMBL = (("sources.chembl", True),)
_SCREEN = (("sources.screen", True),)

#: Greedy search over the data side: each step tries its values on the current recipe, the best
#: one replaces the current recipe if it beats it by more than ``ADOPT_MARGIN``, and the next
#: step runs on the result. Sources first (what is trained on), then how the rows are
#: represented and weighted, then the fields that only act once their source is on. The qHTS
#: representation runs twice: on the qHTS sources the recipe already holds (Tox21), then with
#: Veith switched on as well.
GREEDY_STEPS = (
    GreedyStep("sources.chembl", (True,)),
    GreedyStep("sources.sair", (True,)),
    GreedyStep("sources.veith", (True,)),
    GreedyStep("sources.novartis", (True,)),
    GreedyStep("sources.octant", (False,)),
    GreedyStep("sources.tox21", (False,)),
    GreedyStep("sources.qhts_representation", ("max_response", "both")),
    GreedyStep("sources.qhts_representation", ("max_response", "both"), implies=_VEITH),
    GreedyStep("auxiliary.include_qhts_inactives", (False,), implies=_VEITH),
    GreedyStep("sources.screen", (False,)),
    GreedyStep("multifidelity.screen_strategy", ("response_head", "log2fc_response"), _SCREEN),
    GreedyStep("multifidelity.screen_weight", (0.1, 0.6, 1.0), _SCREEN),
    GreedyStep("sources.tdi_assay_condition", (True,)),
    GreedyStep("multifidelity.tdi_weight", (0.25, 1.0), (("sources.tdi_assay_condition", True),)),
    GreedyStep("auxiliary.tox21_include_inactives", (True,), (("sources.tox21", True),)),
    GreedyStep("auxiliary.per_endpoint_budget", (0.25, 0.5, 1.0)),
    GreedyStep("data.dedup_mode", ("inchikey_block",)),
    GreedyStep("training.macro_balance", (True,)),
    GreedyStep("training.inverse_variance_weighting", (True,)),
    GreedyStep("auxiliary.max_distance", (0.0, 3.0), _CHEMBL),
    GreedyStep("auxiliary.weighting", ("uniform",), _CHEMBL),
    GreedyStep("auxiliary.scale", (0.5, 2.0), _CHEMBL),
    GreedyStep("auxiliary.target_weight", (0.3, 3.0), _CHEMBL),
    GreedyStep("auxiliary.scored_weight", (0.5, 2.0), _CHEMBL),
)

#: A value replaces the current recipe only if its macro ST-RAE is lower by more than this. Repeats
#: of one configuration across revisions of this sweep differ by 0.001-0.007 (``graph_octant_tox21``
#: 0.6553-0.6580 over three revisions, ``graph`` 0.6564-0.6748 over six), so a smaller gap is noise.
ADOPT_MARGIN = float(os.environ.get("CYP_ADOPT_MARGIN", "0.005"))


def _setting(overrides: dict, key: str) -> object:
    """The value a recipe gives a field: its override, else the field's default."""
    if key in overrides:
        return overrides[key]
    target: object = ExperimentConfig()
    for part in key.split("."):
        target = getattr(target, part)
    return target


def run_greedy(base_name: str, steps: tuple[GreedyStep, ...], margin: float = ADOPT_MARGIN) -> dict:
    """Run ``steps`` as a greedy search from ``base_name`` and return the recipe it ends on.

    Decisions are read back from the results table, so a resumed run repeats them exactly. A
    candidate that did not score (failed) is never adopted. New recipes are registered in
    ``REGRESSION`` under a name carrying a hash of their resolved settings, so the same recipe
    keeps the same name and an already-measured one is not run again.
    """
    declared = {_resolved_key(overrides): name for name, overrides in REGRESSION.items()}
    current, current_name = dict(REGRESSION[base_name]), base_name
    run_phase("reg", {current_name: current}, SEEDS)
    path = [{"step": "base", "name": current_name}]
    for number, step in enumerate(steps, start=1):
        if any(_setting(current, key) != value for key, value in step.requires):
            logger.info("greedy %02d %s: skipped, precondition not met", number, step.key)
            continue
        candidates: dict[str, dict] = {}
        current_key = _resolved_key(current)
        for value in step.values:
            overrides = {**current, **dict(step.implies), step.key: value}
            key = _resolved_key(overrides)
            if key == current_key:
                continue
            if key not in declared:
                digest = hashlib.sha1(key.encode()).hexdigest()[:6]
                tag = str(value).replace(".", "p").replace("-", "m")
                name = f"greedy_{step.key.split('.')[-1]}{tag}_{digest}"
                REGRESSION[name] = overrides
                declared[key] = name
            candidates[declared[key]] = overrides
        if not candidates:
            continue
        run_phase("reg", candidates, SEEDS)
        scores = ranked_means("reg")["mean"]

        def score_of(name: str, scores: pd.Series = scores) -> float:
            value = scores.get(name, float("nan"))
            return float("inf") if pd.isna(value) else float(value)

        base_score = score_of(current_name)
        best = min(candidates, key=score_of)
        best_score = score_of(best)
        listing = ", ".join(f"{n}={score_of(n):.4f}" for n in candidates)
        if best_score <= base_score - margin:
            adopted = ", ".join(
                f"{key}={candidates[best][key]!r}" for key in (*dict(step.implies), step.key)
            )
            logger.info(
                "greedy %02d %s: ADOPT %s (%.4f -> %.4f); tried %s",
                number, step.key, adopted, base_score, best_score, listing,
            )  # fmt: skip
            current, current_name = candidates[best], best
            path.append({"step": step.key, "name": best, "score": best_score})
        else:
            logger.info(
                "greedy %02d %s: keep %s (%.4f); tried %s", number, step.key, current_name,
                base_score, listing,
            )  # fmt: skip
    logger.info("greedy search ended on %s: %s", current_name, json.dumps(current, sort_keys=True))
    RESULTS.parent.joinpath("greedy_path.json").write_text(
        json.dumps({"path": path, "recipe": current, "margin": margin}, indent=2, default=list)
    )
    return current


#: Fields no entry above varies: input width and optimiser batch of the neural network and the
#: curve, the learned enzyme-embedding width, the graph warmup, and the graph-curve encoder depth.
REGRESSION.update(
    _ofat(
        "neural",
        {"model.kind": "neural"},
        {"model.neural.batch_size": [128, 512], "features.fingerprint_bits": [1024, 2048]},
    )
)
REGRESSION.update(
    _ofat(
        "neural_cond_embedding",
        {"model.kind": "neural", "conditioning.strategy": "embedding"},
        {"conditioning.embedding_dim": [4, 16]},
    )
)
REGRESSION.update(_ofat("tfm", {"model.kind": "tfm"}, {"features.fingerprint_bits": [1024, 2048]}))
REGRESSION.update(_ofat("graph", {"model.kind": "graph"}, {"model.graph.warmup_epochs": [0, 5]}))
TDI.update(_ofat("curve_heads", TDI["curve_heads"], {"model.curve.batch_size": [128, 512]}))
TDI.update(
    _ofat("curve_graph_heads", TDI["curve_graph_heads"], {"model.curve.graph_depth": [2, 4]})
)


def _panel_grid() -> dict[str, dict]:
    """The full-panel comparison grid: {random, CheMeleon} x {shared, per-measurement readout} x
    {inhibited fraction, raw log2FC screen}, always with the screen and both Veith quantities.

    Two of the eight cells are declared by hand above (``graph_panel`` and
    ``graph_panel_measurement_heads``, random + fraction); this adds the other six under the same
    naming, so every cell has the same folds, objective, sources and budgets and a difference is one
    factor. ``graph_panel_per_target_heads`` stays as the intermediate readout control.
    """
    grid = {}
    for init, pretraining in (("", {}), ("_chemeleon", CHEMELEON)):
        for screen, strategy in (("", "response_head"), ("_log2fc", "log2fc_response")):
            for readout, heads in (("", {}), ("_measurement_heads", MEASUREMENT_HEADS)):
                name = f"graph_panel{init}{screen}{readout}"
                if name not in REGRESSION:
                    grid[name] = {
                        "model.kind": "graph",
                        **PANEL,
                        "multifidelity.screen_strategy": strategy,
                        **pretraining,
                        **heads,
                    }
    return grid


REGRESSION.update(_panel_grid())
REGRESSION.update(
    {
        # Matched screen-off controls for the random and the pretrained shared-readout panel cell:
        # the same Veith potency + max-response and readout width, with the screen removed, so the
        # difference to ``graph_panel`` / ``graph_panel_chemeleon`` is what the screen adds.
        "graph_panel_no_screen": {
            "model.kind": "graph",
            **PANEL,
            "sources.screen": False,
            "multifidelity.screen_strategy": "extra_rows",
        },
        "graph_panel_chemeleon_no_screen": {
            "model.kind": "graph",
            **PANEL,
            **CHEMELEON,
            "sources.screen": False,
            "multifidelity.screen_strategy": "extra_rows",
        },
        # Capacity control for the readout comparison: at width 300 the per-measurement heads hold
        # ~29x the shared readout's parameters (an MLP per task), so a win there could be capacity.
        # 128 is the width the independent heads default to.
        "graph_panel_measurement_heads_ffn128": {
            "model.kind": "graph",
            **{**PANEL, "model.graph.ffn_hidden": 128},
            **MEASUREMENT_HEADS,
        },
        # The loss on the panel: MAE in place of the soft-threshold interval loss.
        "graph_panel_mae": {"model.kind": "graph", **PANEL, "model.objective": "absolute"},
        "graph_panel_chemeleon_mae": {
            "model.kind": "graph",
            **PANEL,
            **CHEMELEON,
            "model.objective": "absolute",
        },
        # CheMeleon + raw screen + both Veith quantities + measurement heads, with ChEMBL added
        # (default weighting, and with the near-auxiliary rows capped at half the challenge weight).
        # The no-ChEMBL cell is ``graph_panel_chemeleon_log2fc_measurement_heads`` in the grid, so
        # the ingredient is read off that pair, not off the combined recipe alone.
        "graph_panel_chemeleon_log2fc_measurement_heads_chembl": {
            "model.kind": "graph",
            **PANEL,
            **CHEMELEON,
            **MEASUREMENT_HEADS,
            "multifidelity.screen_strategy": "log2fc_response",
            "sources.chembl": True,
        },
        "graph_panel_chemeleon_log2fc_measurement_heads_chembl_budget_half": {
            "model.kind": "graph",
            **PANEL,
            **CHEMELEON,
            **MEASUREMENT_HEADS,
            "multifidelity.screen_strategy": "log2fc_response",
            "sources.chembl": True,
            "auxiliary.per_endpoint_budget": 0.5,
        },
        # Graph-backbone curve variants (``CURVE_HP`` only moves the fingerprint-MLP backbone):
        # FiLM conditioning, a lower learning rate, longer training, and a frozen CheMeleon encoder.
        "curve_graph_film": {**GRAPH_CURVE, "model.curve.conditioning": "film"},
        "curve_graph_learning_rate0p0003": {**GRAPH_CURVE, "model.curve.learning_rate": 0.0003},
        "curve_graph_epochs100": {**GRAPH_CURVE, "model.curve.epochs": 100},
        "curve_graph_chemeleon_frozen": {
            **GRAPH_CURVE,
            "model.curve.pretraining": "chemeleon",
            "model.curve.pretraining_finetune": False,
        },
    }
)
# Task budgets on the panel itself. The realized table (see ``log_realized_inventory``) shows the
# auxiliary Veith rows carrying ~10x the challenge labels' total weight, so the auxiliary budget
# and the screen weight are first-order questions here, not tuning: cap the scored enzymes'
# near-auxiliary rows at a fraction of their challenge weight, and vary the screen weight.
REGRESSION.update(
    _ofat(
        "graph_panel",
        {"model.kind": "graph", **PANEL},
        {
            "auxiliary.per_endpoint_budget": [0.25, 0.5],
            "multifidelity.screen_weight": [0.1, 0.6],
        },
    )
)
#: Fine-tuning learning rate: the peak of chemprop's warmup/decay schedule (default 1e-3, which
#: ``graph`` / ``graph_chemeleon`` already run), for a random and a pretrained encoder.
GRAPH_LR = {"model.graph.learning_rate": [0.0001, 0.0003]}
REGRESSION.update(_ofat("graph", {"model.kind": "graph"}, GRAPH_LR))
REGRESSION.update(_ofat("graph_chemeleon", {"model.kind": "graph", **CHEMELEON}, GRAPH_LR))
# ... and on the best recipe measured so far (Octant + active-only Tox21 auxiliaries), so the
# unexplored lever is applied to the incumbent and not only to the plain references.
REGRESSION.update(_ofat("graph_octant_tox21", {"model.kind": "graph", **OCT_TOX}, GRAPH_LR))


#: Name markers for the genuinely-untested config axes this run was reordered to measure
#: first: auxiliary selection (ChEMBL max_distance / weighting / scored_weight), screen weight,
#: the feature-set sweep, TabPFN PCA width, ECFP6, and the neural-only conditioning strategies.
_UNTESTED_MARKERS = (
    "chembl", "screen_weight", "scored_lo", "_fs_", "_pca", "ecfp6",
    "cond_embedding", "cond_calibration",
)  # fmt: skip


def _gpu_first(item: tuple[str, dict]) -> int:
    """Run order key. Untested axes first (so a reordered run returns their answers before the
    known baselines), then the quick reference, then GPU families before CPU boosted -- the GPU
    is the scarce resource in a full-machine window and boosted can run any time on CPU."""
    name, overrides = item
    if any(marker in name for marker in _UNTESTED_MARKERS):
        return 0
    if name == "ref_submitted":
        return 1
    gpu = ("graph", "neural", "per_target_ensemble", "tfm")
    kind = overrides.get("model.kind")
    if kind in gpu:
        return 2
    # The graph-backbone curve trains a D-MPNN, so it is a GPU config too; the fingerprint-MLP
    # curve and the boosted trees stay in the CPU group.
    if kind == "curve" and overrides.get("model.curve.backbone") == "graph":
        return 2
    return 3


#: Reordered GPU-first so a full-machine run uses the GPU rather than spending hours on the
#: CPU boosted configs first (sort is stable, so within each group the curated order holds).
REGRESSION = dict(sorted(REGRESSION.items(), key=_gpu_first))

TDI.update(
    _ofat(
        "curve",
        {
            "model.kind": "curve",
            "sources.tdi_assay_condition": True,
            "model.curve.conditioning": "heads",
        },
        CURVE_HP,
    )
)

#: Configs with no completed row (``status == "ok"``) at any revision in the results table of
#: 2026-10-25, in cost order: tabular foundation model and neural (minutes), then graph (20-35
#: min on an L4), then graph with CheMeleon or a graph-backbone curve (about 50 min). They run
#: before any config that already has a measurement, so a window that ends early has measured new
#: options and not repeated old ones. Configs measured earlier on this revision are skipped by
#: ``run_phase`` and are not listed.
UNMEASURED_REGRESSION = (
    "neural_fs_fp",
    "neural_fs_dense",
    "neural_fs_dense_ion",
    "neural_fs_dense_emb",
    "neural_fs_emb",
    "neural_fs_full",
    "neural_cond_embedding_embedding_dim4",
    "neural_cond_embedding_embedding_dim16",
    "neural_cond_descriptor_pocket",
    "neural_cond_descriptor_zscales",
    "neural_cond_descriptor_srs",
    "neural_cond_descriptor_site",
    "neural_cond_descriptor_ctd",
    "neural_cond_descriptor_all",
    "neural_batch_size128",
    "neural_batch_size512",
    "neural_fingerprint_bits1024",
    "neural_fingerprint_bits2048",
    "tfm_fingerprint_bits1024",
    "tfm_fingerprint_bits2048",
    "tfm_fs_fp",
    "tfm_fs_dense",
    "tfm_fs_fp_dense",
    "tfm_fs_dense_ion",
    "tfm_fs_dense_emb",
    "tfm_fs_emb",
    "tfm_fs_full",
    "neural_hidden128",
    "neural_hidden512",
    "neural_dropout0p1",
    "neural_dropout0p3",
    "neural_learning_rate0p0003",
    "neural_learning_rate0p003",
    "neural_epochs100",
    "graph_fs_none",
    "graph_fs_ionization",
    "graph_fs_mechanistic",
    "graph_fs_mechanistic_ionization",
    "graph_fs_physchem",
    "graph_fs_physchem_ionization",
    "graph_octant_tox21_fs_none",
    "graph_octant_tox21_fs_ionization",
    "graph_octant_tox21_fs_mechanistic",
    "graph_octant_tox21_fs_mechanistic_ionization",
    "graph_octant_tox21_fs_physchem",
    "graph_octant_tox21_fs_physchem_ionization",
    "graph_octant_tox21_fs_physchem_mechanistic_ionization",
    "graph_warmup_epochs0",
    "graph_warmup_epochs5",
    "graph_chembl_dist0",
    "graph_chembl_dist3",
    "graph_chembl_dist3_target_lo",
    "graph_chembl_uniform",
    "graph_octant_tox21_scored_lo",
    "graph_screen_weight_lo",
    "graph_screen_weight_hi",
    "graph_screen_response",
    "graph_octant_tox21_screen_response",
    "graph_adapt_2d6",
    "graph_octant_tox21_adapt_2d6",
    "graph_point_loss_lo",
    "graph_point_loss_hi",
    "graph_ranking_loss_lo",
    "graph_ranking_loss_hi",
    "graph_octant_tox21_point_ranking",
    "graph_octant_tox21_inactives",
    "graph_octant_tox21_budget_half",
    "graph_octant_tox21_budget_one",
    "graph_screen_log2fc",
    "graph_octant_tox21_screen_log2fc",
    "graph_per_target_heads",
    "graph_octant_tox21_per_target_heads",
    "graph_measurement_heads",
    "graph_octant_tox21_measurement_heads",
    "graph_ffn1024",
    "graph_inverse_variance",
    "graph_octant_tox21_inverse_variance",
    "graph_hidden200",
    "graph_hidden600",
    "graph_depth2",
    "graph_depth5",
    "graph_dropout0p0",
    "graph_dropout0p2",
    "graph_epochs60",
    "graph_batch_size32",
    "graph_learning_rate0p0001",
    "graph_learning_rate0p0003",
    "graph_octant_tox21_learning_rate0p0001",
    "graph_octant_tox21_learning_rate0p0003",
    "curve_graph",
    "curve_graph_chemeleon",
    "curve_graph_film",
    "curve_graph_learning_rate0p0003",
    "curve_graph_epochs100",
    "curve_graph_chemeleon_frozen",
    "graph_chemeleon_learning_rate0p0001",
    "graph_chemeleon_learning_rate0p0003",
)
#: The whole TDI track: no TDI config has a completed row at any revision. MLP-backbone curves
#: first, then the graph-backbone curves.
UNMEASURED_TDI = (
    "curve_fs_fp",
    "curve_fs_dense",
    "curve_fs_dense_ion",
    "curve_fs_dense_emb",
    "curve_fs_emb",
    "curve_fs_full",
    "curve_descriptor_pocket",
    "curve_descriptor_zscales",
    "curve_descriptor_srs",
    "curve_descriptor_site",
    "curve_descriptor_ctd",
    "curve_descriptor_all",
    "curve_heads_batch_size128",
    "curve_heads_batch_size512",
    "curve_heads",
    "curve_descriptor",
    "curve_film",
    "curve_heads_ionization",
    "curve_heads_tdishift_15",
    "curve_heads_tdishift_50",
    "curve_heads_losstau_60",
    "curve_heads_deep",
    "curve_hidden128",
    "curve_hidden512",
    "curve_dropout0p1",
    "curve_dropout0p3",
    "curve_epochs100",
    "curve_screen_noise0p05",
    "curve_screen_noise0p2",
    "curve_graph_heads_fs_none",
    "curve_graph_heads_fs_ionization",
    "curve_graph_heads_fs_mechanistic",
    "curve_graph_heads_fs_mechanistic_ionization",
    "curve_graph_heads_fs_physchem",
    "curve_graph_heads_fs_physchem_ionization",
    "curve_graph_heads_fs_physchem_mechanistic_ionization",
    "curve_graph_descriptor_pocket",
    "curve_graph_descriptor_zscales",
    "curve_graph_descriptor_srs",
    "curve_graph_descriptor_site",
    "curve_graph_descriptor_ctd",
    "curve_graph_descriptor_all",
    "curve_graph_heads_graph_depth2",
    "curve_graph_heads_graph_depth4",
    "curve_graph_heads",
    "curve_graph_chemeleon_heads",
    "curve_graph_descriptor",
)
#: Unmeasured configs that are slow: the Veith-panel graphs (90-380 min on an L4) and the boosted
#: hyperparameter passes (CPU-bound, so the GPU idles during them).
UNMEASURED_REGRESSION_SLOW = (
    "graph_panel_chemeleon_log2fc_measurement_heads_chembl",
    "graph_panel_chemeleon_log2fc_measurement_heads_chembl_budget_half",
    "graph_panel_screen_weight0p1",
    "graph_panel_screen_weight0p6",
    "graph_veith_both",
    "graph_all_aux_source_heads",
    "graph_screen_and_qhts_response",
    "graph_panel_per_target_heads",
    "graph_panel_chemeleon_log2fc",
    "graph_panel_chemeleon_log2fc_measurement_heads",
    "graph_panel_no_screen",
    "graph_panel_chemeleon_no_screen",
    "graph_panel_measurement_heads_ffn128",
    "graph_panel_mae",
    "graph_panel_chemeleon_mae",
    "graph_panel_per_endpoint_budget0p25",
    "graph_panel_per_endpoint_budget0p5",
    "boosted_num_leaves31",
    "boosted_num_leaves127",
    "boosted_learning_rate0p03",
    "boosted_learning_rate0p1",
    "boosted_n_estimators300",
    "boosted_n_estimators1000",
    "boosted_min_child_samples5",
    "boosted_min_child_samples50",
    "boosted_subsample0p6",
    "boosted_subsample1p0",
    "boosted_colsample_bytree0p4",
    "boosted_colsample_bytree0p8",
)
#: The six full-panel cells measured at ``744efbd-m1`` while the max-response carried the wrong sign
#: (rows relabelled ``744efbd-m1-signbug``). The sign is fixed, so these rerun right after the qHTS
#: grid; cheapest first.
PANEL_RERUN_REGRESSION = (
    "graph_panel",
    "graph_panel_log2fc",
    "graph_panel_measurement_heads",
    "graph_panel_log2fc_measurement_heads",
    "graph_panel_chemeleon",
    "graph_panel_chemeleon_measurement_heads",
)
for _name, _pool in (
    *((n, REGRESSION) for n in (*UNMEASURED_REGRESSION, *UNMEASURED_REGRESSION_SLOW)),
    *((n, REGRESSION) for n in PANEL_RERUN_REGRESSION),
    *((n, TDI) for n in UNMEASURED_TDI),
):
    if _name not in _pool:  # fail at import, not hours into a run
        raise KeyError(f"unmeasured-first config {_name!r} is not declared")


#: One CV repeat (each seed is a full N_FOLDS cross-validation). The sweep is large and the
#: GPU families are slow, so one repeat; the CV already predicts every compound once, and the
#: driver is resumable, so a re-run continues where an interrupted night left off.
SEEDS = [0]


#: One fixed row schema, so a failure row cannot have fewer fields than a success row
#: and push later rows under the wrong columns.
RESULT_COLUMNS = [
    "rev", "phase", "name", "seed", "status", "macro_ST_RAE", "minutes",
    "CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4",
    "macro_MCC", "CYP2D6_MCC", "CYP3A4_MCC",
]  # fmt: skip


def run_identity() -> str:
    """A code-revision tag (short SHA, ``-dirty`` when uncommitted), or ``nogit``.

    Results are namespaced by it so a code change (e.g. the boosted weight fix) does not
    silently reuse results trained under the old semantics.

    ``CYP_REV`` overrides it. A tree shipped without its ``.git`` (a ``git archive`` uploaded to a
    remote Studio) has no revision of its own and would otherwise collapse every code version into
    ``nogit``; the launcher passes the archived commit's SHA so the namespace stays meaningful.
    """
    override = os.environ.get("CYP_REV")
    if override:
        return override
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        clean = subprocess.call(["git", "diff", "--quiet"]) == 0
        clean = clean and subprocess.call(["git", "diff", "--cached", "--quiet"]) == 0
        return sha if clean else f"{sha}-dirty"
    except Exception:  # noqa: BLE001 -- no git is not fatal, just a weaker identity
        return "nogit"


REV = run_identity()
if NUM_MEMBERS:
    # The ensemble size changes the trained model and its scores, so it namespaces the results:
    # a CYP_MEMBERS=1 screening run and a full-member run never share a bundle or a completion
    # key, and neither silently reuses the other's results under the same git revision.
    REV = f"{REV}-m{NUM_MEMBERS}"

#: Earlier revisions whose completed results would remain valid under this one. Empty: the
#: 9102533 bundles were trained before the current graph defaults (five members, not one) and
#: before the point objectives masked censored rows, so a matching config *name* no longer
#: proves a matching recipe -- carrying them would rank a stale single-member score beside the
#: current five-member ones and then export a five-member submission. A carried revision is
#: eligible only when its resolved recipe and the relevant training semantics match this one;
#: until that check exists, carry nothing and run a clean single-revision sweep.
CARRY_REVS: tuple[str, ...] = ()
#: Carried-over results feed resume, ranking and final selection alike. A member-count override
#: already renames REV, so a screening run never mixes with a family-default one either way.
CARRIED: tuple[str, ...] = () if NUM_MEMBERS else CARRY_REVS


def already_done() -> set[tuple[str, str, int]]:
    """Completed or pruned (phase, name, seed) at this revision or a carried-over one; failures
    retried."""
    if not RESULTS.exists():
        return set()
    table = pd.read_csv(RESULTS)
    if "rev" not in table.columns or "status" not in table.columns:
        return set()  # pre-schema results from an earlier code era: re-run under this one
    ok = table[table["rev"].isin({REV, *CARRIED}) & table["status"].isin(["ok", "pruned"])]
    return {(r.phase, r.name, int(r.seed)) for r in ok.itertuples()}


#: Weights & Biases handles, set by ``wandb_start`` when ``CYP_WANDB`` is on. The sweep runs one
#: config at a time in this process, so a single run logged from ``record`` needs no cross-process
#: coordination. Left None (a no-op) unless opted in, so the default sweep has no W&B dependency.
_WANDB_RUN = None
_WANDB_ROWS: list[list] = []


def wandb_start() -> None:
    """Begin a W&B run for the sweep when ``CYP_WANDB`` is set, so it can be followed online.

    Off by default: only ``CYP_WANDB`` (any value) turns it on, and ``wandb`` is imported lazily,
    so neither the dependency nor an account is needed for an ordinary local run. Project and
    entity come from the standard ``WANDB_PROJECT`` / ``WANDB_ENTITY`` env; the run is named by the
    code revision so it lines up with the results namespace. A missing package or a failed init
    (no key, no network) warns and the sweep continues -- W&B is a monitor, never a gate on the run.
    """
    global _WANDB_RUN
    if not os.environ.get("CYP_WANDB"):
        return
    try:
        import wandb
    except ImportError:
        logger.warning("CYP_WANDB is set but wandb is not installed (pip install wandb); skipping")
        return
    try:
        _WANDB_RUN = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "cyp-shift"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=REV,
            job_type="sweep",
            config={
                "rev": REV,
                "n_folds": N_FOLDS,
                "members": NUM_MEMBERS or "family default",
                "n_regression": len(REGRESSION),
                "n_tdi": len(TDI),
            },
        )
    except Exception as error:  # noqa: BLE001 -- auth, network or settings: never fail the sweep
        logger.warning("W&B init failed (%s); continuing without it", type(error).__name__)
        return
    logger.info("W&B run live: %s", getattr(_WANDB_RUN, "url", "(no url)"))


def wandb_log_row(row: dict) -> None:
    """Log one completed config to the live W&B run: a running results table plus scalars.

    Each call rebuilds the table from the accumulated rows (a logged ``wandb.Table`` is immutable,
    so it cannot be mutated and re-logged), and logs the config's macro ST-RAE / MCC under
    phase-grouped keys so the dashboard shows a point per configuration as the night proceeds.
    """
    if _WANDB_RUN is None:
        return
    import wandb

    _WANDB_ROWS.append([row.get(column) for column in RESULT_COLUMNS])
    metrics: dict = {
        "results": wandb.Table(columns=RESULT_COLUMNS, data=[list(r) for r in _WANDB_ROWS]),
        "configs_done": len(_WANDB_ROWS),
    }
    if row.get("status") == "ok" and row.get("macro_ST_RAE") is not None:
        metrics[f"ST_RAE/{row['phase']}/{row['name']}"] = row["macro_ST_RAE"]
        if row.get("macro_MCC") is not None:
            metrics[f"MCC/{row['phase']}/{row['name']}"] = row["macro_MCC"]
    _WANDB_RUN.log(metrics)


def wandb_finish() -> None:
    """Close the W&B run (no-op when W&B was not started)."""
    if _WANDB_RUN is not None:
        _WANDB_RUN.finish()


def record(row: dict) -> None:
    full = pd.DataFrame([{column: row.get(column) for column in RESULT_COLUMNS}])
    full.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
    wandb_log_row(row)


def compute_row(phase: str, name: str, overrides: dict, seed: int) -> dict:
    """Cross-validate one configuration and return its result row (nothing is recorded here)."""
    config = ExperimentConfig.load()
    config.override("data.seed", seed)
    for key, value in overrides.items():
        config.override(key, value)
    # Graph dataloader workers, set from the environment (CYP_WORKERS), not swept and not a
    # tracked edit -- so tuning it to the machine does not change the git revision and a
    # resume still reuses completed configs. Harmless on non-graph configs (unused field).
    config.override("model.graph.num_workers", NUM_WORKERS)
    if NUM_MEMBERS:
        config.override("model.n_members", NUM_MEMBERS)
    if MONITOR or PRUNE_REFERENCE:
        config.override("model.graph.monitor_fold", True)
        config.override("model.graph.prune_reference", PRUNE_REFERENCE)
        if EARLY_STOP is not None:
            config.override("model.graph.early_stopping_patience", float(EARLY_STOP))
        if PRUNE_TOLERANCE is not None:
            config.override("model.graph.prune_tolerance", float(PRUNE_TOLERANCE))
    config.validate()
    started = time.time()
    bundle = BUNDLE_DIR / REV / f"{phase}_{name}_seed{seed}".replace(":", "_")
    # Each seed is a full stratified scaffold cross-validation (every compound predicted once
    # out of fold); the seed drives both the fold partition and the model's own randomness.
    scores, _, tdi = run_cv(config, n_folds=N_FOLDS, bundle_path=bundle)
    macro = float(scores.loc[MACRO_LABEL, "ST-RAE_mean"])
    row = {
        "rev": REV,
        "phase": phase,
        "name": name,
        "seed": seed,
        "status": "ok",
        "macro_ST_RAE": round(macro, 4),
        "minutes": round((time.time() - started) / 60, 1),
    }
    for endpoint, data in scores.iterrows():
        if endpoint != MACRO_LABEL:
            row[endpoint.split("_")[0]] = round(float(data["ST-RAE_mean"]), 4)
    # The TDI track, present only for a dose-response curve model. Macro MCC is the mean of
    # the two scored isoforms, matching the leaderboard's MA-MCC.
    if tdi is not None:
        mccs = {
            endpoint.split("_")[0]: round(float(data["MCC_mean"]), 4)
            for endpoint, data in tdi.iterrows()
            if endpoint in CLASSIFICATION_ENDPOINTS
        }
        for isoform, value in mccs.items():
            row[f"{isoform}_MCC"] = value
        if mccs:
            row["macro_MCC"] = round(sum(mccs.values()) / len(mccs), 4)
    return row


def record_row(row: dict) -> None:
    """Record a finished row and log the one-line summary of it."""
    record(row)
    tail = f", macro MCC {row['macro_MCC']}" if row.get("macro_MCC") is not None else ""
    logger.info(
        ">>> %s/%s seed=%d: macro ST-RAE %.4f%s (%.1f min)",
        row["phase"],
        row["name"],
        row["seed"],
        row["macro_ST_RAE"],
        tail,
        row["minutes"],
    )


def evaluate(phase: str, name: str, overrides: dict, seed: int) -> None:
    record_row(compute_row(phase, name, overrides, seed))


def _compute_row_in_child(phase: str, name: str, overrides: dict, seed: int) -> dict:
    """:func:`compute_row` in a spawned worker; its log lines carry the configuration name."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO)
    for handler in root.handlers:
        handler.setFormatter(logging.Formatter(f"%(asctime)s [{name}] %(message)s", "%H:%M:%S"))
    return compute_row(phase, name, overrides, seed)


def _run_parallel(phase: str, todo: list[tuple[str, dict, int]]) -> None:
    """Run ``todo`` as concurrent worker processes, at most ``JOBS`` at a time.

    Workers are spawned (CUDA does not survive a fork) and live for one configuration each, so a
    fit's GPU and host memory is released when it ends. The parent records every row, so the results
    file has a single writer. A configuration that raises is recorded as failed; if a worker dies
    outright (killed for memory) the pool is broken and every configuration it had not finished is
    recorded as failed.
    """
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(JOBS, len(todo)), mp_context=context, max_tasks_per_child=1
    ) as pool:
        futures = {
            pool.submit(_compute_row_in_child, phase, name, overrides, seed): (name, seed)
            for name, overrides, seed in todo
        }
        for future in concurrent.futures.as_completed(futures):
            name, seed = futures[future]
            try:
                record_row(future.result())
            except PrunedError as pruned:
                logger.info("%s/%s seed=%d PRUNED: %s", phase, name, seed, pruned)
                record({"rev": REV, "phase": phase, "name": name, "seed": seed, "status": "pruned"})
            except Exception as error:  # noqa: BLE001 -- one failure must not sink the night
                logger.error("%s/%s seed=%d FAILED: %r", phase, name, seed, error)
                record({"rev": REV, "phase": phase, "name": name, "seed": seed, "status": "failed"})


def run_phase(
    phase: str,
    configs: dict[str, dict],
    seeds: list[int],
    extra_seeds: dict[str, list[int]] | None = None,
) -> None:
    """Run every config at ``seeds``, and the configs named in ``extra_seeds`` at those seeds too,
    as one queue."""
    done = already_done()
    todo = []
    for name, overrides in configs.items():
        for seed in [*seeds, *(extra_seeds or {}).get(name, [])]:
            if (phase, name, seed) in done:
                logger.info("skip %s/%s seed=%d (already recorded)", phase, name, seed)
            else:
                todo.append((name, overrides, seed))
    if JOBS > 1 and len(todo) > 1:
        _run_parallel(phase, todo)
        return
    for name, overrides, seed in todo:
        try:
            evaluate(phase, name, overrides, seed)
        except PrunedError as pruned:
            logger.info("%s/%s seed=%d PRUNED: %s", phase, name, seed, pruned)
            record({"rev": REV, "phase": phase, "name": name, "seed": seed, "status": "pruned"})
        except Exception as error:  # noqa: BLE001 -- one failure must not sink the night
            logger.exception("%s/%s seed=%d FAILED: %s", phase, name, seed, error)
            record({"rev": REV, "phase": phase, "name": name, "seed": seed, "status": "failed"})


def _completed(phase: str | None = None) -> pd.DataFrame:
    """Successful rows at the current revision or a carried-over one, optionally for one phase.

    Carried revisions are included so ranking and the final automatic submission see the
    earlier winners that ``already_done`` skipped re-running, rather than choosing among only
    the configs that happened to run under the current revision.
    """
    table = pd.read_csv(RESULTS)
    table = table[table["rev"].isin({REV, *CARRIED}) & (table["status"] == "ok")].copy()
    table["macro_ST_RAE"] = table["macro_ST_RAE"].astype(float)
    return table if phase is None else table[table["phase"] == phase]


def ranked_means(phase: str | None = None) -> pd.DataFrame:
    """Mean and spread of macro ST-RAE per configuration, best first."""
    return (
        _completed(phase)
        .groupby("name")["macro_ST_RAE"]
        .agg(["mean", "std", "count"])
        .sort_values("mean")
    )


def ranked_mcc(phase: str) -> pd.DataFrame:
    """Mean macro MCC per configuration, best (highest) first; empty when none scored TDI."""
    table = _completed(phase)
    scored = table.dropna(subset=["macro_MCC"]) if "macro_MCC" in table.columns else table.iloc[0:0]
    if scored.empty:
        return scored
    return (
        scored.groupby("name")["macro_MCC"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )


def write_submission(track: str, overrides: dict, output: Path, config_path: Path) -> None:
    """Refit the chosen config on all training data and write its blinded-test submission.

    Applies the same ``CYP_MEMBERS`` / ``CYP_WORKERS`` the sweep used, so the written ensemble
    matches the ranked recipe rather than reverting to the family default.
    """
    config = ExperimentConfig.load()
    for key, value in overrides.items():
        config.override(key, value)
    config.override("model.graph.num_workers", NUM_WORKERS)
    if NUM_MEMBERS:
        config.override("model.n_members", NUM_MEMBERS)
    config.validate()
    config.save(config_path)
    submit(output=output, config_path=config_path, track=track)
    logger.info("%s submission written to %s (config %s)", track, output, config_path)


def _archive_stale_results() -> None:
    """Move a results file with a different schema aside so a fresh run starts clean.

    Triggers whenever the existing columns differ from the current schema at all -- a missing
    ``rev`` column (pre-rev era) or a changed column set (e.g. the added MCC columns). This
    matters because ``record`` appends in text mode, and appending rows with a different field
    count to an existing file corrupts it (pandas then cannot parse the mixed widths).
    """
    if RESULTS.exists() and list(pd.read_csv(RESULTS, nrows=0).columns) != RESULT_COLUMNS:
        archived = RESULTS.with_suffix(".pre_schema.csv")
        RESULTS.rename(archived)
        logger.info("archived results with a mismatched schema to %s", archived)


@dataclasses.dataclass(frozen=True)
class Cell:
    """One configuration of the focus plan: ``sets`` are applied on top of the incumbent recipe.
    ``group`` names the factor the cell varies; cells of one group are alternatives, and the groups
    that win are stacked in the final round. ``control`` is the tag of the cell that differs from
    this one only in the factor under test (``"base"`` for the incumbent): a cell that is the lowest
    of the round and beats its control by more than the margin is adopted even when its gain over
    the incumbent is smaller, because the incumbent differs from it in more than that factor."""

    tag: str
    sets: tuple[tuple[str, object], ...]
    group: str = ""
    control: str = ""


#: The recipe the focus plan starts from: the graph with Octant, Tox21 and Novartis, which the
#: greedy search adopted (macro ST-RAE 0.6415).
FOCUS_BASE = QHTS_BEST_BASE
#: Summed near-auxiliary weight per challenge label, for the normalised cells: the mean over the
#: four scored enzymes of that ratio in ``FOCUS_BASE``'s training table (1.7, 14.3, 12.1, 8.0).
FOCUS_BUDGET = 9.0
_BOTH = (("sources.qhts_representation", "both"),)
_BUDGET = (("auxiliary.per_endpoint_budget", FOCUS_BUDGET),)
#: Independent heads per (target, measurement); the readout width is pinned to the shared
#: readout's default so the comparison is not also one of width.
_MEASUREMENT_HEADS = (
    ("model.graph.per_measurement_heads", True),
    ("model.graph.per_target_heads", False),
    ("model.graph.ffn_hidden", 300),
)
_LOG2FC = (("multifidelity.screen_strategy", "log2fc_response"),)

#: Round 1, data side: the qHTS representation (potency only, or potency and the Tox21
#: max-response) crossed with the auxiliary weights (as they are, or normalised to
#: ``FOCUS_BUDGET`` labels' worth per enzyme). The fourth cell, potency only with the weights as
#: they are, is ``FOCUS_BASE``. The normalised potency-only cell is the control for the weight
#: change.
FOCUS_ROUND_1 = (
    Cell("pic50_budget", _BUDGET),
    Cell("both", _BOTH, control="base"),
    Cell("both_budget", _BOTH + _BUDGET, control="pic50_budget"),
)
#: Round 2, data side, on the round-1 winner: the screen as isotonic pseudo-potencies (the
#: incumbent) or as the raw log2 fold-change on its own response head, crossed with the shared
#: readout (the incumbent) or independent per-measurement heads.
FOCUS_ROUND_2 = (
    Cell("log2fc", _LOG2FC),
    Cell("mheads", _MEASUREMENT_HEADS),
    Cell("log2fc_mheads", _LOG2FC + _MEASUREMENT_HEADS),
)
#: Round 3, model side: each cell changes one factor of the round-2 winner.
FOCUS_ROUND_3 = (
    Cell(
        "chemeleon_lr3e4",
        (
            ("model.graph.pretraining", "chemeleon"),
            ("model.graph.learning_rate", 3e-4),
            ("model.graph.epochs", 60),
        ),
        "encoder",
    ),
    Cell(
        "capacity",
        (
            ("model.graph.hidden", 600),
            ("model.graph.depth", 5),
            ("model.graph.ffn_hidden", 1024),
            ("model.graph.epochs", 60),
        ),
        "capacity",
    ),
    Cell("epochs60", (("model.graph.epochs", 60),), "epochs"),
    Cell(
        "per_target_heads",
        (
            ("model.graph.per_target_heads", True),
            ("model.graph.per_measurement_heads", False),
            ("model.graph.ffn_hidden", 300),
        ),
        "readout",
    ),
    Cell("per_measurement_heads", _MEASUREMENT_HEADS, "readout"),
    Cell("point_loss0p1", (("model.point_loss_weight", 0.1),), "loss"),
    Cell("absolute", (("model.objective", "absolute"),), "loss"),
    Cell("adapter_cyp2d6", (("model.graph.adapt_target", "CYP2D6"),), "adapter"),
)
#: The order the winning groups are stacked in, so a field two groups set (epochs, readout width)
#: ends on the value of the later one.
FOCUS_STACK_ORDER = ("epochs", "readout", "loss", "adapter", "capacity", "encoder")
#: Groups that cannot be stacked: the pretrained encoder fixes the message-passing width and depth.
FOCUS_EXCLUSIVE = (("capacity", "encoder"),)
_CHEMELEON_CELL = FOCUS_ROUND_3[0]


def _apply(recipe: dict, sets: tuple[tuple[str, object], ...]) -> dict:
    return {**recipe, **dict(sets)}


def _is_valid(overrides: dict) -> bool:
    """Whether the settings form a configuration ``ExperimentConfig.validate`` accepts."""
    experiment_logger = logging.getLogger("cypshift.experiment")
    previous = experiment_logger.level
    experiment_logger.setLevel(logging.WARNING)
    try:
        config = ExperimentConfig()
        for key, value in overrides.items():
            config.override(key, value)
        config.validate()
    except ValueError:
        return False
    finally:
        experiment_logger.setLevel(previous)
    return True


def _recipe_name(tag: str, overrides: dict, declared: dict[str, str]) -> str:
    """The name a recipe is recorded under: the declared entry with the same resolved settings,
    else ``focus_<tag>_<digest>`` (registered, so the same recipe keeps its name across reruns)."""
    key = _resolved_key(overrides)
    if key not in declared:
        name = f"focus_{tag}_{hashlib.sha1(key.encode()).hexdigest()[:6]}"
        REGRESSION[name] = overrides
        declared[key] = name
    return declared[key]


def _score_of(scores: pd.Series, name: str) -> float:
    value = scores.get(name, float("nan"))
    return float("inf") if pd.isna(value) else float(value)


def _focus_round(
    tag: str,
    cells: tuple[Cell, ...],
    current_name: str,
    current: dict,
    declared: dict[str, str],
    margin: float,
) -> tuple[str, dict, dict[str, float]]:
    """Run ``cells`` on the incumbent as one parallel batch and adopt the best one if it beats the
    incumbent by more than ``margin``. Returns the recipe name, its settings and every score."""
    candidates: dict[str, dict] = {}
    name_of: dict[str, str] = {"base": current_name}
    for cell in cells:
        overrides = _apply(current, cell.sets)
        if not _is_valid(overrides):
            logger.info("focus %s: %s skipped, invalid on the incumbent", tag, cell.tag)
        elif _resolved_key(overrides) == _resolved_key(current):
            logger.info("focus %s: %s is the incumbent", tag, cell.tag)
        else:
            name_of[cell.tag] = _recipe_name(cell.tag, overrides, declared)
            candidates[name_of[cell.tag]] = overrides
    control_of = {
        name_of[cell.tag]: name_of[cell.control]
        for cell in cells
        if cell.tag in name_of and cell.control in name_of
    }
    run_phase("reg", {current_name: current, **candidates}, SEEDS)
    scores = ranked_means("reg")["mean"]
    base = _score_of(scores, current_name)
    results = {name: _score_of(scores, name) for name in (current_name, *candidates)}
    listing = ", ".join(f"{name}={score:.4f}" for name, score in results.items())
    best = min(candidates, key=lambda name: results[name], default=None)
    if best is not None and results[best] <= base - margin:
        logger.info(
            "focus %s: ADOPT %s (%.4f -> %.4f); %s", tag, best, base, results[best], listing
        )
        return best, candidates[best], results
    control = control_of.get(best, "")
    if best is not None and control in results and results[best] < base:
        gain = results[control] - results[best]
        if gain > margin:
            logger.info(
                "focus %s: ADOPT %s (%.4f; %.4f below its control %s, %.4f below %s); %s",
                *(tag, best, results[best], gain, control),
                *(base - results[best], current_name, listing),
            )
            return best, candidates[best], results
    logger.info("focus %s: keep %s (%.4f); %s", tag, current_name, base, listing)
    return current_name, current, results


def _stack(current: dict, groups: dict[str, tuple[tuple[str, object], ...]]) -> dict:
    """``current`` with each group's settings applied in ``FOCUS_STACK_ORDER``."""
    recipe = dict(current)
    for group in FOCUS_STACK_ORDER:
        if group in groups:
            recipe = _apply(recipe, groups[group])
    return recipe


def _stackable(current: dict, groups: dict[str, tuple[tuple[str, object], ...]]) -> bool:
    exclusive = any(a in groups and b in groups for a, b in FOCUS_EXCLUSIVE)
    return not exclusive and _is_valid(_stack(current, groups))


def _fit_cost_rank(overrides: dict) -> int:
    """0 for a pretrained-encoder fit (the slowest), 1 for a wider or deeper encoder, else 2."""
    if overrides.get("model.graph.pretraining") == "chemeleon":
        return 0
    if overrides.get("model.graph.hidden", 300) > 300 or overrides.get("model.graph.depth", 3) > 3:
        return 1
    return 2


def _focus_model_round(
    current_name: str,
    current: dict,
    encoder_bases: list[dict],
    declared: dict,
    margin: float,
) -> tuple[str, dict, dict[str, float]]:
    """Round 3: every ``FOCUS_ROUND_3`` cell on the incumbent in one parallel batch, then the groups
    that won are stacked into one recipe, which is measured and kept if it beats the best single
    group. The pretrained encoder is also tried, as separate candidates, on each of
    ``encoder_bases`` that differs from the incumbent."""
    group_of: dict[str, str] = {}
    sets_of: dict[str, tuple[tuple[str, object], ...]] = {}
    candidates: dict[str, dict] = {}
    for cell in FOCUS_ROUND_3:
        overrides = _apply(current, cell.sets)
        if not _is_valid(overrides) or _resolved_key(overrides) == _resolved_key(current):
            logger.info("focus model: %s skipped (invalid or already the incumbent)", cell.tag)
            continue
        name = _recipe_name(cell.tag, overrides, declared)
        candidates[name], group_of[name], sets_of[name] = overrides, cell.group, cell.sets
    for base_recipe in encoder_bases:
        if _resolved_key(base_recipe) != _resolved_key(current):
            overrides = _apply(base_recipe, _CHEMELEON_CELL.sets)
            candidates[_recipe_name("chemeleon_other", overrides, declared)] = overrides
    # Slowest first, so a pool smaller than the batch starts the long fits before the short ones.
    candidates = dict(sorted(candidates.items(), key=lambda item: _fit_cost_rank(item[1])))
    run_phase("reg", {current_name: current, **candidates}, SEEDS)
    scores = ranked_means("reg")["mean"]
    base = _score_of(scores, current_name)
    results = {name: _score_of(scores, name) for name in (current_name, *candidates)}
    logger.info(
        "focus model: %s", ", ".join(f"{name}={score:.4f}" for name, score in results.items())
    )

    winners: dict[str, tuple[float, str]] = {}  # group -> (gain, cell name)
    for name, group in group_of.items():
        gain = base - results[name]
        if gain > margin and gain > winners.get(group, (0.0, ""))[0]:
            winners[group] = (gain, name)
    chosen: dict[str, tuple[tuple[str, object], ...]] = {}
    for group in sorted(winners, key=lambda g: -winners[g][0]):
        trial = {**chosen, group: sets_of[winners[group][1]]}
        if _stackable(current, trial):
            chosen = trial
        else:
            logger.info("focus model: group %s does not stack with %s", group, sorted(chosen))
    ranked = {name: results[name] for name in (current_name, *candidates)}
    if len(chosen) > 1:
        stacked = _stack(current, chosen)
        name = _recipe_name("stack_" + "_".join(sorted(chosen)), stacked, declared)
        run_phase("reg", {name: stacked}, SEEDS)
        scores = ranked_means("reg")["mean"]
        ranked[name] = _score_of(scores, name)
        candidates[name] = stacked
        logger.info("focus model: stack of %s -> %s = %.4f", sorted(chosen), name, ranked[name])
    best = min((n for n in ranked if n != current_name), key=ranked.get, default=None)
    if best is not None and ranked[best] <= base - margin:
        logger.info("focus model: ADOPT %s (%.4f -> %.4f)", best, base, ranked[best])
        return best, candidates[best], ranked
    logger.info("focus model: keep %s (%.4f)", current_name, base)
    return current_name, current, ranked


def blend_report(final: str, names: list[str], output: str = "focus_blends.csv") -> pd.DataFrame:
    """Macro ST-RAE of the equal-weight blend of ``final`` with each other recipe of the plan, from
    the out-of-fold predictions of their bundles; written to ``data/<output>``."""
    import blend

    members = {
        name: frame for name, frame in blend.load_members(REV, SEEDS[0]).items() if name in names
    }
    if final not in members:
        return pd.DataFrame()
    rows = [{"with": "(alone)", "macro_ST_RAE": blend.rank_single(members)[final]}]
    for other in sorted(set(members) - {final}):
        pair = {final: members[final], other: members[other]}
        rows.append(
            {"with": other, "macro_ST_RAE": blend.macro_st_rae_on(pair, "equal", None, None)}
        )
    table = pd.DataFrame(rows).sort_values("macro_ST_RAE").reset_index(drop=True)
    logger.info("equal-weight blends with %s:\n%s", final, table.to_string())
    table.to_csv(RESULTS.parent / output, index=False)
    return table


def run_focus(margin: float = ADOPT_MARGIN) -> dict:
    """The staged plan: round 1 (qHTS representation x auxiliary weights), round 2 (screen
    representation x readout) on its winner, round 3 (model side) on that, each round one parallel
    batch whose winner becomes the incumbent. Writes ``data/focus_path.json`` and
    ``data/focus_best.json`` and returns the final recipe's settings."""
    declared = {_resolved_key(overrides): name for name, overrides in REGRESSION.items()}
    name, recipe = FOCUS_BASE, dict(REGRESSION[FOCUS_BASE])
    path = [{"round": "base", "name": name}]
    scores: dict[str, float] = {}

    name, recipe, results = _focus_round("1 data", FOCUS_ROUND_1, name, recipe, declared, margin)
    scores.update(results)
    path.append({"round": "1 data", "name": name, "scores": results})
    # The best round-1 recipe of each kind -- potency only, and potency with the max-response --
    # which the pretrained encoder is tried on in round 3 beside the round-2 winner.
    by_kind: dict[str, dict[str, float]] = {"pic50": {}, "both": {}}
    for cell_name, score in results.items():
        kind = REGRESSION[cell_name].get("sources.qhts_representation", "pic50")
        by_kind[kind][cell_name] = score
    encoder_bases = [REGRESSION[min(cells, key=cells.get)] for cells in by_kind.values() if cells]

    name, recipe, results = _focus_round("2 data", FOCUS_ROUND_2, name, recipe, declared, margin)
    scores.update(results)
    path.append({"round": "2 data", "name": name, "scores": results})

    name, recipe, results = _focus_model_round(name, recipe, encoder_bases, declared, margin)
    scores.update(results)
    path.append({"round": "3 model", "name": name, "scores": results})

    logger.info("focus plan ended on %s: %s", name, json.dumps(recipe, sort_keys=True))
    out = RESULTS.parent
    (out / "focus_path.json").write_text(
        json.dumps({"path": path, "recipe": recipe, "margin": margin}, indent=2, default=list)
    )
    (out / "focus_best.json").write_text(json.dumps({"name": name, **recipe}, indent=2) + "\n")
    blend_report(name, list(scores))
    return recipe


FOCUS_SUBMISSION = Path("submissions/focus_submission_regression.csv")


def run_focus_submission() -> None:
    """Refit the recipe in ``data/focus_best.json`` on all training data and write the blinded-test
    submission. Run with ``CYP_MEMBERS=0`` so the ensemble has the graph model's own member count
    (five), not the single member the screening runs use."""
    recipe = json.loads((RESULTS.parent / "focus_best.json").read_text())
    name = recipe.pop("name")
    logger.info("refitting %s on all training data (members: %s)", name, NUM_MEMBERS or "default")
    write_submission(
        "regression", recipe, FOCUS_SUBMISSION, RESULTS.parent / "focus_submission_config.json"
    )


#: Encoder and readout width of the wide members: message-passing width 2048, depth 4, readout
#: width 1024, dropout 0.2.
_WIDE = (
    ("model.graph.hidden", 2048),
    ("model.graph.depth", 4),
    ("model.graph.ffn_hidden", 1024),
    ("model.graph.dropout", 0.2),
)
_INVERSE_VARIANCE = (("training.inverse_variance_weighting", True),)
#: The wide plan's members, applied to the focus plan's data recipe (``FOCUS_BASE`` with the
#: max-response and the normalised auxiliary weights): interval loss with uniform weights, absolute
#: loss with inverse-variance weights, interval loss with inverse-variance weights, and the
#: narrow encoder with inverse-variance weights (the control for the weighting alone). Slowest
#: first.
WIDE_CELLS = (
    Cell("wide_interval", _WIDE),
    Cell("wide_absolute_invvar", _WIDE + (("model.objective", "absolute"),) + _INVERSE_VARIANCE),
    Cell("wide_interval_invvar", _WIDE + _INVERSE_VARIANCE),
    Cell("invvar", _INVERSE_VARIANCE),
)


#: Settings that choose the training data, which a recipe of another model family takes from the
#: graph recipe it is blended with.
DATA_SIDE_PREFIXES = ("sources.", "auxiliary.", "multifidelity.")


def _member_recipe(member: str, recipe: dict) -> dict:
    """The declared recipe ``member`` with the data settings of ``recipe``. When a model family
    rejects the whole set (the TabPFN backend fits challenge rows only), the settings it accepts
    one by one are kept."""
    data = {k: v for k, v in recipe.items() if k.startswith(DATA_SIDE_PREFIXES)}
    overrides = {**REGRESSION[member], **data}
    if _is_valid(overrides):
        return overrides
    overrides = dict(REGRESSION[member])
    for key, value in data.items():
        if _is_valid({**overrides, key: value}):
            overrides[key] = value
        else:
            logger.info("%s does not take %s=%r", member, key, value)
    return overrides


def _run_cells(
    cells: tuple[Cell, ...],
    blends: str,
    start: str = FOCUS_BASE,
    base_sets: tuple[tuple[str, object], ...] = (),
    base_tag: str = "both_budget",
    members: tuple[str, ...] = (),
    extra_seeds: tuple[int, ...] = (),
) -> dict[str, str]:
    """Cross-validate the cells on the focus plan's data recipe (``start`` with the max-response and
    the normalised auxiliary weights, then ``base_sets``) beside the recipe itself, then report the
    equal-weight blend of the recipe with each cell (``data/<blends>``). ``members`` are declared
    recipes of other model families, given the recipe's data settings and listed before the
    cells; ``extra_seeds`` repeat the recipe itself at those seeds."""
    declared = {_resolved_key(overrides): name for name, overrides in REGRESSION.items()}
    recipe = _apply(_apply(dict(REGRESSION[start]), _BOTH + _BUDGET), base_sets)
    base = _recipe_name(base_tag, recipe, declared)
    configs = {}
    names = {"base": base}
    for member in members:
        overrides = _member_recipe(member, recipe)
        names[member] = _recipe_name(member, overrides, declared)
        configs[names[member]] = overrides
    for cell in cells:
        overrides = _apply(recipe, cell.sets)
        if not _is_valid(overrides):
            raise ValueError(f"cell {cell.tag} is not a valid configuration")
        names[cell.tag] = _recipe_name(cell.tag, overrides, declared)
        configs[names[cell.tag]] = overrides
    run_phase("reg", {base: recipe, **configs}, SEEDS, {base: list(extra_seeds)})
    scores = ranked_means("reg")["mean"]
    logger.info("cells:\n%s", scores.reindex([base, *configs]).to_string())
    blend_report(base, [base, *configs], blends)
    return names


def run_wide() -> None:
    """The wide members (``WIDE_CELLS``); blends in ``data/wide_blends.csv``."""
    _run_cells(WIDE_CELLS, "wide_blends.csv")


#: Staged graph training on the focus plan's data recipe: the pooled fit followed by a phase on the
#: challenge dose-response alone, with the encoder trained or frozen, and the matched control of one
#: more pooled epoch at the same peak rate (``CURRICULUM_PRESETS``).
FINETUNE_CELLS = (
    Cell("direct_finetune", (("training.preset", "direct_finetune"),)),
    Cell("direct_finetune_frozen", (("training.preset", "direct_finetune_frozen"),)),
    Cell("pooled_continue", (("training.preset", "pooled_continue"),)),
)


def run_finetune() -> None:
    """The staged-training cells (``FINETUNE_CELLS``); blends in ``data/finetune_blends.csv``."""
    _run_cells(FINETUNE_CELLS, "finetune_blends.csv")


#: Auxiliary row selection on the Octant and Tox21 recipe (no Novartis) with the max-response and
#: the normalised auxiliary weights, which hold each endpoint's auxiliary weight total fixed
#: whatever the row count. The cap is 2000 rows per enzyme, assay and source group: Tox21
#: max-response has 6,000 to 6,800 per enzyme, AC50 1,500 to 2,800, Octant 1,017. Cells: random,
#: MaxMin and clusters at the cap, and the full table at 18 epochs, which gives it as many optimizer
#: updates as the 30-epoch fits on the reduced tables (34,800 of 58,400 rows).
SUBSET_BASE = "graph_octant_tox21"
SUBSET_CAP = 2000
SUBSET_CELLS = (
    Cell("full_18_epochs", (("model.graph.epochs", 18),)),
    *(
        Cell(
            f"{selection}_{SUBSET_CAP}",
            (("auxiliary.row_selection", selection), ("auxiliary.max_rows_per_group", SUBSET_CAP)),
        )
        for selection in ("random", "maxmin", "clusters")
    ),
)


def run_subset() -> None:
    """The row-selection cells (``SUBSET_CELLS``); blends in ``data/subset_blends.csv``."""
    _run_cells(SUBSET_CELLS, "subset_blends.csv", SUBSET_BASE)


#: Row selection by share on the Octant and Tox21 recipe, every cell at the recipe's 30 epochs. Each
#: cell differs from the full table (``base``) in its row selection: the method and the share of
#: every group's rows kept. ``_share_cell`` can also restrict the selection to some assays
#: (``SHARE_TOX21``, ``SHARE_OCTANT``).
SHARE_TOX21 = ("luminescent_ac50", "luminescent_max_response")
SHARE_OCTANT = ("fluorogenic_pic50",)


def _share_cell(
    tag: str, group: str, control: str, selection: str, share: float, assays: tuple[str, ...] = ()
) -> Cell:
    sets = (("auxiliary.row_selection", selection), ("auxiliary.row_share", share))
    if assays:
        sets += (("auxiliary.row_selection_assays", list(assays)),)
    return Cell(tag, sets, group, control)


SHARE_CELLS = (
    _share_cell("clusters_75", "row selection", "base", "clusters", 0.75),
    _share_cell("maxmin_75", "row selection", "base", "maxmin", 0.75),
    _share_cell("clusters_50", "row selection", "base", "clusters", 0.5),
    _share_cell("maxmin_50", "row selection", "base", "maxmin", 0.5),
)


def _report_factors(cells: tuple[Cell, ...], names: dict[str, str]) -> pd.DataFrame:
    """Each cell's score beside its reference's, and whether the change exceeds the adopt margin."""
    scores = ranked_means("reg")["mean"]
    rows = []
    for cell in cells:
        reference = cell.control or "base"
        score, other = _score_of(scores, names[cell.tag]), _score_of(scores, names[reference])
        rows.append(
            {
                "cell": cell.tag,
                "factor": cell.group,
                "reference": reference,
                "score": score,
                "reference_score": other,
                "change": score - other,
                "beyond_margin": abs(score - other) > ADOPT_MARGIN,
            }
        )
    table = pd.DataFrame(rows)
    logger.info(
        "factors (change is score minus the reference's, lower is better; beyond_margin is whether "
        "the change exceeds the adopt margin %.3f):\n%s",
        ADOPT_MARGIN,
        table.round(4).to_string(index=False),
    )
    return table


def run_share() -> None:
    """The row-share cells (``SHARE_CELLS``); blends in ``data/share_blends.csv``, each cell
    against its reference in ``data/share_factors.csv``."""
    names = _run_cells(SHARE_CELLS, "share_blends.csv", SUBSET_BASE)
    _report_factors(SHARE_CELLS, names).to_csv(RESULTS.parent / "share_factors.csv", index=False)


#: The random 75% cell of the share plan, recorded before the plan dropped its random cells; it is a
#: candidate for the night's row policy beside the share plan's cells.
RANDOM_75 = _share_cell("random_75", "row selection", "base", "random", 0.75)


def choose_row_policy() -> Cell | None:
    """The recorded row-selection cell the night runs on: the lowest-scoring of ``RANDOM_75`` and
    ``SHARE_CELLS`` if it beats the full table by more than the adopt margin, else ``None`` (the
    full table)."""
    declared = {_resolved_key(overrides): name for name, overrides in REGRESSION.items()}
    recipe = _apply(dict(REGRESSION[SUBSET_BASE]), _BOTH + _BUDGET)
    scores = ranked_means("reg")["mean"]
    full = _score_of(scores, _recipe_name("both_budget", recipe, declared))
    candidates = {cell.tag: cell for cell in (RANDOM_75, *SHARE_CELLS)}
    table = {
        tag: _score_of(scores, _recipe_name(tag, _apply(recipe, cell.sets), declared))
        for tag, cell in candidates.items()
    }
    logger.info("full table %.4f; row-selection cells %s", full, table)
    best = min(table, key=table.get)
    if table[best] < full - ADOPT_MARGIN:
        logger.info("row policy %s beats the full table by more than %.3f", best, ADOPT_MARGIN)
        return candidates[best]
    logger.info("no row-selection cell beats the full table by more than %.3f", ADOPT_MARGIN)
    return None


#: The model-side round on the Octant and Tox21 recipe, run on the row policy ``choose_row_policy``
#: returns. Each cell changes one setting from the base, or from the cell named in ``control``.
NIGHT_CELLS = (
    Cell("pooled_continue", (("training.preset", "pooled_continue"),), "fine-tuning"),
    Cell(
        "direct_finetune",
        (("training.preset", "direct_finetune"),),
        "fine-tuning",
        "pooled_continue",
    ),
    Cell(
        "direct_finetune_frozen",
        (("training.preset", "direct_finetune_frozen"),),
        "fine-tuning",
        "pooled_continue",
    ),
    Cell("adapter_cyp2d6", (("model.graph.adapt_target", "CYP2D6"),), "readout"),
    Cell("absolute_loss", (("model.objective", "absolute"),), "objective"),
    Cell("epochs_60", (("model.graph.epochs", 60),), "epochs"),
    Cell("ffn_layers_2", (("model.graph.ffn_layers", 2),), "readout"),
    Cell("batch_norm", (("model.graph.batch_norm", True),), "pooling"),
    Cell("norm_pooling", (("model.graph.aggregation", "norm"),), "pooling"),
)
#: Other model families, run for the blend on the base's data settings where the family takes them:
#: boosted trees, and the dense-feature TabPFN and neural models.
NIGHT_MEMBERS = (
    "boosted_octant_tox21_deep",
    "boosted_pinball_t60",
    "tfm_fs_dense",
    "neural_fs_dense",
)
#: Seeds the base recipe is repeated at: the spread between runs, and the gain of averaging two.
NIGHT_SEEDS = (1, 2)


def report_night(names: dict[str, str]) -> None:
    """The base recipe's score at each seed and its seeds' equal-weight blend
    (``data/night_reseed.csv``), and the greedy equal-weight blend path over the night's recipes
    (``data/night_blend_path.csv``), from their out-of-fold bundles."""
    import blend

    base = names["base"]
    frames = {}
    for seed in (SEEDS[0], *NIGHT_SEEDS):
        try:
            frames[seed] = blend.load_members(REV, seed)[base]
        except (SystemExit, KeyError):
            logger.warning("no out-of-fold bundle of %s at seed %d", base, seed)
    rows = [
        {"what": f"base, seed {seed}", "macro_ST_RAE": blend.rank_single({base: frame})[base]}
        for seed, frame in frames.items()
    ]
    if len(frames) > 1:
        rows.append(
            {
                "what": f"equal-weight blend of the base at seeds {sorted(frames)}",
                "macro_ST_RAE": blend.macro_st_rae_on(
                    {f"seed{seed}": frame for seed, frame in frames.items()}, "equal", None, None
                ),
            }
        )
    reseed = pd.DataFrame(rows)
    logger.info("base across seeds:\n%s", reseed.to_string(index=False))
    reseed.to_csv(RESULTS.parent / "night_reseed.csv", index=False)

    wanted = set(names.values())
    members = {n: f for n, f in blend.load_members(REV, SEEDS[0]).items() if n in wanted}
    single = {n: blend.rank_single({n: f})[n] for n, f in members.items()}
    chosen = [min(single, key=single.get)]
    path = [{"step": 1, "added": chosen[0], "macro_ST_RAE": single[chosen[0]]}]
    while len(chosen) < len(members):
        trial = {
            n: blend.macro_st_rae_on({m: members[m] for m in [*chosen, n]}, "equal", None, None)
            for n in members
            if n not in chosen
        }
        pick = min(trial, key=trial.get)
        chosen.append(pick)
        path.append({"step": len(chosen), "added": pick, "macro_ST_RAE": trial[pick]})
    table = pd.DataFrame(path)
    logger.info("greedy equal-weight blend path:\n%s", table.to_string(index=False))
    table.to_csv(RESULTS.parent / "night_blend_path.csv", index=False)


def run_night() -> None:
    """The model-side cells (``NIGHT_CELLS``) and other-family members (``NIGHT_MEMBERS``) on the
    chosen row policy, the base repeated at ``NIGHT_SEEDS``; each cell against its reference in
    ``data/night_factors.csv``."""
    policy = choose_row_policy()
    names = _run_cells(
        NIGHT_CELLS,
        "night_blends.csv",
        SUBSET_BASE,
        base_sets=policy.sets if policy else (),
        base_tag=policy.tag if policy else "both_budget",
        members=NIGHT_MEMBERS,
        extra_seeds=NIGHT_SEEDS,
    )
    _report_factors(NIGHT_CELLS, names).to_csv(RESULTS.parent / "night_factors.csv", index=False)
    report_night(names)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    load_local_env()  # tokens for a local run; a remote run is forwarded them by lightning_run
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    _archive_stale_results()
    wandb_start()
    members = NUM_MEMBERS or "family default"
    logger.info(
        "run identity (results namespace): %s | graph workers: %d | ensemble members: %s",
        REV,
        NUM_WORKERS,
        members,
    )

    if PLAN == "submit":
        run_focus_submission()
        return
    if PLAN == "finetune":
        logger.info("=== Staged graph training on the focus data recipe (jobs: %d) ===", JOBS)
        run_finetune()
        wandb_finish()
        return
    if PLAN == "night":
        logger.info("=== Model-side round on the Octant + Tox21 recipe (jobs: %d) ===", JOBS)
        run_night()
        wandb_finish()
        return
    if PLAN == "share":
        logger.info("=== Auxiliary row shares on the Octant + Tox21 recipe (jobs: %d) ===", JOBS)
        run_share()
        wandb_finish()
        return
    if PLAN == "subset":
        logger.info("=== Auxiliary row selection on the focus data recipe (jobs: %d) ===", JOBS)
        run_subset()
        wandb_finish()
        return
    if PLAN == "wide":
        logger.info("=== Wide members on the focus data recipe (jobs: %d) ===", JOBS)
        run_wide()
        wandb_finish()
        return
    if PLAN == "focus":
        logger.info("=== Focus plan from %s (jobs: %d) ===", FOCUS_BASE, JOBS)
        run_focus()
        wandb_finish()
        return

    logger.info(
        "=== qHTS max-response x source set x model family: %d regression, %d TDI ===",
        len(QHTS_REGRESSION),
        len(QHTS_TDI),
    )
    run_phase("reg", {name: REGRESSION[name] for name in QHTS_REGRESSION}, SEEDS)
    run_phase("tdi", {name: TDI[name] for name in QHTS_TDI}, SEEDS)

    logger.info(
        "=== Full-panel cells rerun after the max-response sign fix: %d regression ===",
        len(PANEL_RERUN_REGRESSION),
    )
    run_phase("reg", {name: REGRESSION[name] for name in PANEL_RERUN_REGRESSION}, SEEDS)

    logger.info(
        "=== Greedy data-side search from %s: %d steps, adopt margin %.4f ===",
        DATA_BASE,
        len(GREEDY_STEPS),
        ADOPT_MARGIN,
    )
    run_greedy(DATA_BASE, GREEDY_STEPS)

    logger.info(
        "=== Data side on %s: %d regression configs ===", DATA_BASE, len(DATA_SIDE_REGRESSION)
    )
    run_phase("reg", {name: REGRESSION[name] for name in DATA_SIDE_REGRESSION}, SEEDS)

    logger.info(
        "=== Never measured first: %d regression, %d TDI, then %d slow regression ===",
        len(UNMEASURED_REGRESSION),
        len(UNMEASURED_TDI),
        len(UNMEASURED_REGRESSION_SLOW),
    )
    run_phase("reg", {name: REGRESSION[name] for name in UNMEASURED_REGRESSION}, SEEDS)
    run_phase("tdi", {name: TDI[name] for name in UNMEASURED_TDI}, SEEDS)
    run_phase("reg", {name: REGRESSION[name] for name in UNMEASURED_REGRESSION_SLOW}, SEEDS)

    logger.info(
        "=== Regression: %d configs, %d-fold CV x %d ===", len(REGRESSION), N_FOLDS, len(SEEDS)
    )
    run_phase("reg", REGRESSION, SEEDS)  # the unmeasured entries are already recorded and skipped
    regression = ranked_means("reg")
    logger.info("\nRegression (mean macro ST-RAE, lower better):\n%s", regression.to_string())

    # Refit the best of each track on all data and write a candidate submission. Not
    # auto-uploaded: the ST-RAE level is optimistic, so judge the bundles on Spearman first. The
    # regression candidate is written before the TDI sweep, so a window that ends in the TDI
    # configs still leaves it.
    if not regression.empty:
        best = regression.index[0]
        logger.info("=== Best regression: %s (%.4f) ===", best, regression.loc[best, "mean"])
        write_submission("regression", REGRESSION[best], SUBMISSION, BEST_CONFIG)

    logger.info("=== TDI: %d curve configs, %d-fold CV x %d ===", len(TDI), N_FOLDS, len(SEEDS))
    run_phase("tdi", TDI, SEEDS)
    tdi = ranked_mcc("tdi")
    if not tdi.empty:
        logger.info("\nTDI (mean macro MCC, higher better):\n%s", tdi.to_string())
        best_tdi = tdi.index[0]
        logger.info("=== Best TDI: %s (MCC %.4f) ===", best_tdi, tdi.loc[best_tdi, "mean"])
        write_submission("classification", TDI[best_tdi], TDI_SUBMISSION, TDI_BEST_CONFIG)

    wandb_finish()


if __name__ == "__main__":
    main()
