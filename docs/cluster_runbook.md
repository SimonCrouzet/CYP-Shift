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

131 configs, ordered so the untested axes run first. At 1 member and a real GPU this is hours, not the ~2 days the laptop implied.

## What the sweep covers

- **Feature-set sweep** — `boosted_fs_*`, `tfm_fs_*`: which blocks each model sees (fingerprint / physchem+mechanistic / ionization / embedding, and combinations).
- **Auxiliary selection** — `graph_chembl*`: `max_distance` reach into related CYPs (0 = scored only, 1.5 = 7 CYPs, 3.0 = 17), weighting, scored/target weight. The main untried CYP2D6 lever in the sweep.
- **source_heads** — source-stratified aux conditioning (bug fixed `48b48e3`).
- **qHTS** — `graph_veith*`, `graph_response`: pIC50 vs max_response, inactives on/off. Already answered on the prior run (pIC50 beats max_response, inactives ~neutral); they re-run here for one clean namespace.
- Screen weight, dedup mode, ECFP6, TabPFN PCA width, neural-only conditioning.

## After the sweep

1. **Rank**: `python scripts/analyze_bundles.py --rev <rev>`
2. **Calibrate** — the lever that matters most. Our ranking is competitive (Spearman ~0.70) but R² is 0.14 vs the leaders' ~0.37, so the deficit is magnitude/spread, not ordering. Fit on OOF, no GPU:
   ```bash
   python scripts/calibrate.py --rev <rev> --member <best-config> \
       --submission <in.csv> --output <out.csv> \
       --method variance_match_rho --potent-quantile 0.5
   ```
   Methods: `fit_potent`, `variance_match`, `variance_match_rho`, `fit_full`. Sweep the method and `--potent-quantile` — this is cheap. Keep it fit from OOF, not tuned to the live board (the final eval set differs from the live leaderboard).
3. **Blend** (equal-weight; research says it beats stacking): `python scripts/blend.py --rev <rev> --top-k 5 --method equal`, or `--members graph_octant_tox21,boosted_octant_tox21_deep,curve_heads`.
4. Write the chosen submission and upload. Cap: 1 per 12 h, latest counts.

## Not in the sweep (separate jobs if wanted)

- **CYP2D6 structure** — the Boltz cofold cache is near-empty. A GPU precompute to populate it unlocks `model.graph.use_structure` and gives the heme-coordination features real values. This is the untried structural lever for the weak endpoint.
- A heavy new family (Uni-Mol, graph transformer) is deprioritized: the research is consistent that family barely matters at this data scale, and the winners' edge was proprietary data.

## Reading the CV (from 4 submissions so far)

- scaffold-CV ranks configs in the right order (it correctly put graph below boosted, octant+tox21 below plain graph, both confirmed live), but is ~0.12–0.18 optimistic in absolute ST-RAE — never read the CV number as a live score.
- Track CV **R²** and the predicted-σ / true-σ ratio, not CV ST-RAE. ST-RAE on CV is nearly blind to the calibration gap that is actually costing us live.
