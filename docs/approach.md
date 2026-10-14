# CYP-Shift: predicting CYP inhibition under an interval-censored loss

*Our entry for the [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge). Code is open ([MIT](../README.md#-licence)); this note walks through the approach while the challenge is still open, through 3 November 2026.*

## Inhibition across four CYP isoforms

Four cytochrome P450 enzymes — CYP1A2, CYP2C9, CYP2D6 and CYP3A4 — metabolise most small-molecule drugs, and a compound that inhibits one of them can raise the blood level of another drug to a toxic dose. Predicting that inhibition early, from structure alone, is what keeps a dangerous combination out of the clinic. The challenge asks for a pIC50 per compound for each of the four isoforms, on 750 blinded compounds, scored against dose-response curves the organisers hold back.

Two features of the task decide almost every design choice we made: the metric is unusual, and the labels are sparse in a way that is not random.

## Scoring inside the credible interval

The regression track is ranked on a soft-threshold relative absolute error (ST-RAE): a dead-zone L1 loss that is **exactly zero anywhere inside a compound's 95% credible interval**, and grows linearly once a prediction falls outside it. The denominator is the same loss committed by a constant predictor at the mean, so 1.0 is the score of guessing the mean and lower is better.

The credible band is narrow for potent compounds and wide for weak ones — roughly 0.3–0.5 log units for actives against 1.2–2.6 for inactives. The actives therefore carry nearly all of the score. The loss rewards a prediction for landing anywhere inside a compound's band — a narrow target for a potent compound, a wide one for a weak one — and is indifferent to where inside it sits. A model tuned for squared error shrinks its spread toward the global mean, which for a potent compound lands outside that narrow band and pays; the gap between squared-error tuning and this band-aware target is where the track is won or lost.

## Label coverage and the single-concentration screen

Only 4,905 compounds carry a fitted dose-response pIC50, and the coverage is uneven: 47.6% of compounds have a CYP3A4 label but only 26.2% a CYP2C9 one, because a compound was promoted to a full 12-point curve only when it hit that isoform in the primary screen. Train on these labels alone and the model learns "given that this compound is a hit, how strong a hit" — which is not the question the blinded test, full of near-inactive analogs, actually asks.

The single-concentration screen is the fix. It is dense and unbiased, covering 4,376 compounds at one concentration across all four isoforms, and its log fold-change ranks pIC50 almost perfectly (Spearman −0.94 on CYP3A4 down to −0.83 on CYP2D6). It supplies exactly the inactives the supervised labels lack. Using it well — rather than using more exotic data — is the single highest-value move in the pipeline, so it is on by default.

## A modular pipeline of swept options

CYP-Shift is deliberately not one model. Several model families sit over one shared conditioning vocabulary and a set of optional evidence sources, a config picks one combination, and the sweeps decide which combination wins. The families are a gradient-boosted tree over fingerprints and physicochemical descriptors, a neural MLP, a Chemprop D-MPNN, and a dose-response *curve* model that predicts the Hill curve itself and reads pIC50 off it. The curve model earns its place on the multi-fidelity data: a dose-response label constrains the curve's midpoint and a single-concentration reading constrains one point on the same curve, so two assays become two views of one object rather than two tasks to reconcile.

How a row is told *which enzyme and assay it came from* is its own swept axis, because a tree and an MLP want different answers. A tree can split on an assay-identity column and then on fingerprint bits inside that branch, which is an interaction for free; a linear first layer cannot, and needs the condition expressed as properties or as a learned per-assay offset instead. We verified the conditioning carries chemistry and not just a constant: predicting the same compounds under the screen and the dose-response arm, the per-compound spread of the difference (0.268) is as large as its mean (0.264), so the model is doing more than learning an intercept.

Auxiliary potencies — ChEMBL, SAIR, Veith, Tox21, Novartis, Octant — enter through independent on/off switches, each deduplicated against the whole challenge set, and a single binding-site distance knob decides which enzymes' rows to admit. Pulling a scored enzyme's own measurements from an outside source and pulling a relative's are the same operation at different distances, so one axis covers both. None of these is on by default; a plain run trains on the challenge data alone.

Finally, the enzyme can be described rather than named. A one-hot says CYP2C9 and CYP2D6 are two symbols; a descriptor says they recognise opposite charges, so a rule learned on one can transfer to the other by chemistry. The descriptors are computed from structure and sequence, not hand-set, and a leave-one-enzyme-out check keeps the set narrow until width earns its place — with so few enzymes, a descriptor wide enough to identify each is a one-hot in disguise.

## CYP2D6 and its ion-pair interaction

Three of the four isoforms respond to bulk physicochemistry; CYP2D6 does not. A five-fold cross-validated model on physicochemical descriptors reaches R² 0.375 on CYP3A4 and a negative R² on CYP2D6. What CYP2D6 tracks instead is a single substructural feature — a protonated aliphatic amine, worth +0.55 log units of potency there and negative on all three other isoforms, the signature of the Asp301/Glu216 ion pair in its crystal structures. CYP2D6 also correlates with the other three isoforms at 0.06 or below and is measured on a different assay technology. It needs a representation that sees that specific interaction, and because it decides the macro average, it is where the remaining gains most likely sit.

## Validation against an analog-series test

The split is the hardest unsolved problem here. The test set is 75 analog series expanded from training hits, while the training library is a diversity set with almost no analog series, so a like-for-like local split cannot be built from it. A random split is worse than merely optimistic: it is biased in favour of descriptors and against fingerprints, which inverts the test regime. We score on a similarity-matched holdout whose nearest-neighbour profile reproduces the blinded test's, which closes most of the gap a random split leaves, though it stays an imperfect proxy. We treat any single local number with suspicion and derive the decision rule from the loss function itself wherever we can.

## Pose and affinity from the cofold

The second and third tracks reuse the same cofold. A Boltz-2 model folds the enzyme with its heme and the inhibitor together, and we read two things off the complex: the predicted affinity, and the pose — how close the inhibitor's coordinating nitrogen sits to the heme iron. On a validation cofold, imidazole's nitrogen landed 2.19 Å from the iron, textbook type-II coordination, and exactly the mechanism the fingerprint model is measurably worst on. The structure features are a GPU precompute cached to disk, so a sweep trial reads them without ever invoking Boltz.

## Prediction scale and the CYP2D6 representation

The pipeline reproduces the official metric end to end and trains directly on the soft-threshold objective; the TDI and structure tracks reuse the same data and cofold, though the regression track is where our work is concentrated. The open problem we are working now is the scale of the point predictions. A regression model tends to predict a spread narrower than the truth, and under an interval-censored loss a prediction that is correctly ranked but too close to the centre still lands outside the narrow band of a potent compound and pays for it. The fix is to calibrate the predicted spread and offset per isoform so each point estimate falls inside its compound's band rather than sitting at the posterior mean. That calibration, and a representation built for the CYP2D6 ion-pair interaction, are the two levers we expect to matter most before the deadline.
