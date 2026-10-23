# Cluster run — sweep and submission

Branch `model-levers-and-p0-fix`. `git pull` on the cluster, then launch. Keep the git tree
clean so the revision namespace stays fixed and `--resume` reuses completed configs.

## Setup

```bash
./setup.sh                # create/update the cypshift conda env, install, run the preflight
ENV_PREFIX=/scratch/envs/cypshift ./setup.sh --fetch   # env on fast local disk + fetch core data
```

`setup.sh` creates or updates the `cypshift` env from `environment.yml`, installs the package with
every runtime extra, then runs the preflight. Run the preflight alone any time before launching:

```bash
python scripts/check_env.py          # interpreter, packages, GPU, data caches, tokens
python scripts/check_env.py --fetch  # also download the core challenge files into data/raw
```

It exits non-zero when the core (challenge data + always-required packages) is missing. Auxiliary
caches (`data/external/`: octant, tox21, veith, chembl, sair, embeddings, chemeleon weights,
ionization) are gitignored and not fetched by `check_env.py` — sync them to the node, or the
config families that name them fail and skip. `TABPFN_TOKEN` is needed for the tfm configs.

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

141 regression configs (plus 14 TDI), ordered so the untested axes run first. At 1 member and a real GPU this is hours, not the ~2 days the laptop implied.

## Follow the sweep online (W&B)

Optional, off by default. Set `CYP_WANDB=1` to log the sweep to Weights & Biases as it runs — a live results table plus a macro-ST-RAE (and macro-MCC) point per config, named by the code revision:

```bash
pip install 'cypshift[track]'   # wandb; already in the dev install
export WANDB_PROJECT=cyp-shift WANDB_ENTITY=<you>   # optional; defaults project=cyp-shift
CYP_WANDB=1 CYP_WORKERS=8 CYP_MEMBERS=1 python scripts/night_run.py
```

Without `CYP_WANDB` there is no W&B dependency or network call; a missing `wandb` with it set warns and the sweep continues (W&B is a monitor, never a gate).

## Run it on a persistent Lightning AI Studio

`scripts/lightning_run.py` treats a Lightning Studio as a long-lived workstation. A Studio keeps its filesystem across stop/start, so the env, the data and `data/night_results.csv` survive; the sweep is resumable, so a relaunch continues where it left off. The sweep runs **detached** on the Studio (`nohup` + a pid file), so closing the laptop does not kill it.

```bash
pip install 'cypshift[cloud]'                          # lightning-sdk (local orchestration only)
# credentials: fill in local.machine.env (below), or `lightning login`
python scripts/lightning_run.py launch --dry-run       # print the plan; no cloud, no SDK
python scripts/lightning_run.py launch --machine L4 --members 1 --wandb
python scripts/lightning_run.py status                 # Studio up? sweep alive? configs done? log tail
python scripts/lightning_run.py fetch                  # download + merge results (safe mid-run)
python scripts/lightning_run.py stop                   # stop the Studio; files are kept
```

- `launch` is idempotent: it reuses a running Studio (the SDK's `start` raises on one), refreshes the small code tarball each time, uploads the data **once** (`--refresh-data` resends), and runs the check-first `./setup.sh` every time (a few seconds once the env exists). The env lives in `cyp-shift/.env` so it persists with the project. It refuses to start a second sweep while one is running.
- `--machine` is any name the installed SDK offers (`L4` by default; `T4`, `L40S`, `A100`, `H100`, ...); it is ignored for a Studio that is already running. `--max-hours` sets the machine runtime cap where the machine type requires one.
- The sweep's run identity comes from the archived commit's SHA, passed as `CYP_REV` (a `git archive` carries no `.git`, which would otherwise put every code version in one `nogit` results namespace). `git archive` ships the **committed** tree, so commit before launching.
- `--data-external` also ships the local-only auxiliary caches; `--no-setup` skips `setup.sh`; `--remote-python` overrides the interpreter.
- `--wandb` sets `CYP_WANDB` remotely and forwards `WANDB_API_KEY` / `WANDB_PROJECT` / `WANDB_ENTITY`. `TABPFN_TOKEN` and `HF_TOKEN` are forwarded whenever they are set, with no flag (the `tfm_*` configs need the first; without it they fail and are skipped). Secret values are shown as `***` in `--dry-run` and never logged.
- `--wait` blocks (polling every `--poll` seconds) until the sweep ends, then fetches; add `--stop-after` to stop the Studio too. Without `--wait`, `launch` returns as soon as the sweep is running.
- `fetch` merges `data/night_results.csv` rather than overwriting it, so rows from local runs are kept and re-fetching mid-run only adds what is new. Bundles, best-config JSONs and candidate submissions are copied into the repo.
- A running Studio is billed until stopped. Remote paths are relative to the Studio root, where both the shell and the file API start; if they ever differ, the upload check fails with a message to set `--remote-dir` to an absolute path.

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

## Entrypoints use different validation schemes

Do not compare numbers across these — they are not the same split:

- `scripts/night_run.py` → `run_cv()`: stratified scaffold-grouped **5-fold CV** (every training compound predicted once out of fold). This is the cluster sweep and the numbers in `night_results.csv`.
- `scripts/run_experiment.py`, `scripts/overnight_sweep.py`, and `pipeline.run()`: a single **similarity-matched holdout**, not CV.
- `scripts/analogue_validation.py`: a single **analogue holdout** (retained anchors, their neighbours held out).
- `scripts/postprocess_validation.py`: a **nested** outer holdout with inner-only calibration/correction fits — the only path that scores the postprocessors independently.

## Reading the CV (from 4 submissions so far)

- scaffold-CV ranks configs in the right order (it correctly put graph below boosted, octant+tox21 below plain graph, both confirmed live), but is ~0.12–0.18 optimistic in absolute ST-RAE — never read the CV number as a live score.
- Track CV **R²** and the predicted-σ / true-σ ratio, not CV ST-RAE. ST-RAE on CV is nearly blind to the calibration gap that is actually costing us live.

## Tokens: `local.machine.env`

Per-machine tokens live in `local.machine.env` at the repo root. It is gitignored (and so absent from `git archive`, i.e. never shipped to the Studio), and `scripts/lightning_run.py`, `scripts/night_run.py`, `scripts/check_env.py` and the `cypshift` CLI load it at start (`cypshift.local_env`).

| Variable | Used for |
|---|---|
| `LIGHTNING_USER_ID`, `LIGHTNING_API_KEY` | the Lightning SDK (alternative: `lightning login`) |
| `CYP_LAI_STUDIO`, `CYP_LAI_TEAMSPACE` | optional defaults for `--studio` / `--teamspace` |
| `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY` | `CYP_WANDB=1` locally; forwarded by `launch --wandb` |
| `TABPFN_TOKEN` | the `tfm_*` configs; forwarded by `launch` when set |
| `HF_TOKEN` | gated HuggingFace assets (embedding generation); forwarded by `launch` when set |

Format is `NAME=value`, one per line (`export`, quotes and `#` comments are accepted). A variable already exported in the shell wins over the file, and a blank value is ignored, so the shipped template with empty values is inert. `CYP_LOCAL_ENV=/path/to/file` points the loader elsewhere, to keep tokens outside the checkout. Only variable names are ever logged; a malformed line is reported by number, never by content.
