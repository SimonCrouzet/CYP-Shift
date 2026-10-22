# Cluster run — sweep and submission

Branch `model-levers-and-p0-fix`. `git pull` on the cluster, then launch. Keep the git tree
clean so the revision namespace stays fixed and `--resume` reuses completed configs.

## Launch

```bash
export TABPFN_TOKEN=<token>     # the 12 TabPFN (tfm) configs need it; without it they fail+skip
export HF_HOME=<embedding-cache>  # MAMMAL / CheMeleon caches
CYP_WORKERS=8 CYP_MEMBERS=1 python scripts/night_run.py
```

- `CYP_WORKERS` — graph dataloader workers; set to the node's cores (8+ is fine, the laptop was the bottleneck at 2).
- `CYP_MEMBERS=1` — single-member screen, for ranking configs fast. Drop it (family default, 5) only when training the final submission config.
- Resumable: re-run the same command; configs whose bundle already exists at this revision are skipped.
- Writes `data/night_results.csv` (one row per config) and `data/run_bundles/<rev>/<config>/` (out-of-fold predictions, the input to calibration and blending).

143 configs, ordered so the untested axes run first. At 1 member and a real GPU this is hours, not the ~2 days the laptop implied.

## What the sweep covers

- **Feature-set sweep** — `boosted_fs_*`, `tfm_fs_*`: which blocks each model sees (fingerprint / physchem+mechanistic / ionization / embedding, and combinations).
- **Auxiliary selection** — `graph_chembl*`: `max_distance` reach into related CYPs (0 = scored only, 1.5 = 7 CYPs, 3.0 = 17), weighting, scored/target weight. The main untried CYP2D6 lever in the sweep.
- **source_heads** — source-stratified aux conditioning (bug fixed `48b48e3`).
- **qHTS** — `graph_veith*`, `graph_response`: pIC50 vs max_response, inactives on/off. Already answered on the prior run (pIC50 beats max_response, inactives ~neutral); they re-run here for one clean namespace.
- **2026-10-22 analysis-note avenues** (graph, layered on `graph` and `graph_octant_tox21`): a screen-response head (`graph*_screen_response`, avenue 4); a CYP2D6 adaptation block (`graph*_adapt_2d6`, avenue 6); the point-pull and ranking-hinge loss terms (`graph_point_loss_*`, `graph_ranking_loss_*`, `graph_octant_tox21_point_ranking`, avenue 7); Tox21 censored inactives and the per-endpoint auxiliary budget (`graph_octant_tox21_inactives`, `graph_octant_tox21_budget_*`, avenue 5). Avenues 1–3 (calibration, blend, analogue correction) are post-sweep steps below, since they act on the OOF bundles.
- Screen weight, dedup mode, ECFP6, TabPFN PCA width, neural-only conditioning.

## After the sweep

1. **Rank**: `python scripts/analyze_bundles.py --rev <rev>`
2. **Calibrate** (avenue 1) — worth testing, but treat it as one lever, not the whole deficit: R² improving can come from discrimination, offset or spread, so keep ST-RAE the selection objective. Fit on the submitted ensemble's OOF (`--members`, averaged), not one member, and do **not** assume de-shrinking — the restrained `affine` lets the slope fall below one. Fit on OOF, no GPU:
   ```bash
   python scripts/calibrate.py --rev <rev> --members <best-config>[,<member2>,...] \
       --submission <in.csv> --output <out.csv> \
       --method affine --endpoint-method CYP2D6=offset
   ```
   Methods: `affine` (default, restrained free slope on the whole OOF), `offset` (intercept only), and the older `fit_potent` / `variance_match` / `variance_match_rho` / `fit_full` for comparison. Treat CYP2D6 separately (`--endpoint-method CYP2D6=offset`): its labelled population was selected through CYP2D6 while its test compounds were not, so an enrichment-justified slope need not transfer. Avoid `fit_potent` as the default — selecting rows by their true potency manufactures an upward correction that may not generalise. Keep it fit from OOF, not tuned to the live board.
3. **Blend** (avenue 2; equal-weight beats stacking in the write-ups, but `st_rae` fits the interval error directly): `python scripts/blend.py --rev <rev> --top-k 5 --method equal`, or `--members graph_octant_tox21,boosted_octant_tox21_deep,curve_heads`. For endpoint specialists use `--per-endpoint` (best member per endpoint) or `--endpoint-members CYP2C9=graph_chemeleon,CYP2D6=graph_absolute`; assess weights off their fitting data with `--eval-holdout-frac 0.3`.
4. **Analogue-correct** (avenue 3, optional) — nudge each prediction toward the measured references the test analogues surround, shrinking where there is no analogue:
   ```bash
   python scripts/analogue_correct.py --rev <rev> --members <best-config> \
       --submission <in.csv> --output <out.csv> --estimator residual --shrinkage 1.0
   ```
5. Write the chosen submission and upload. Cap: 1 per 12 h, latest counts.

Validate a calibration or correction on the **analogue view** before trusting it — the regime it is meant to help: `python scripts/analogue_validation.py --set model.kind=graph --holdout-frac 0.1`.

## Not in the sweep (separate jobs if wanted)

- **CYP2D6 structure** — the Boltz cofold cache is near-empty. A GPU precompute to populate it gives the heme-coordination features real values. Note that `sources.structure` is currently wired (and validated) only for the boosted model — `validate()` rejects it for the graph — so there is no `model.graph.use_structure` field today; enabling a graph structure experiment would need both the cache and lifting that restriction. This is the untried structural lever for the weak endpoint.
- A heavy new family (Uni-Mol, graph transformer) is deprioritized: the research is consistent that family barely matters at this data scale, and the winners' edge was proprietary data.

## Reading the CV (from 4 submissions so far)

- scaffold-CV ranks configs in the right order (it correctly put graph below boosted, octant+tox21 below plain graph, both confirmed live), but is ~0.12–0.18 optimistic in absolute ST-RAE — never read the CV number as a live score.
- Track CV **R²** and the predicted-σ / true-σ ratio, not CV ST-RAE. ST-RAE on CV is nearly blind to the calibration gap that is actually costing us live.
