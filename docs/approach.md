# CYP-Shift: approach to the OpenADMET CYP inhibition challenge

*Methods and results as of 2 November 2026, before final evaluation. Code is
available under the [MIT licence](../LICENSE).*

Dose-response measurements in this challenge are concentrated among screening
hits, leaving many compound–isoform pairs without a measured potency. The test
set contains analogues of those hits measured across all four isoforms, so both
label coverage and compound selection affect prediction. We combined the
challenge data with public auxiliary data, compared molecular representations
under scaffold cross-validation, and calibrated the selected predictions using
public leaderboard feedback. We also investigated a joint dose-response model
to predict both direct inhibition and time-dependent inhibition (TDI).

## Challenge data and scoring

The direct-inhibition table contains 4,905 compounds, with 1,412 measured pIC50
values for CYP1A2, 1,285 for CYP2C9, 1,493 for CYP2D6 and 2,335 for CYP3A4. The
single-concentration table contains 17,504 measurements across 4,376 compounds and
all four isoforms. The TDI table contains 6,145 compounds. The three tables contain
overlapping sets of compounds.

The organizers selected compounds for dose-response measurements after a primary
screen. The 750 test compounds were purchased as analogues of 75 hits selected on
CYP1A2, CYP2C9 and CYP3A4; CYP2D6 did not guide that selection. The primary screen
used NADPH preincubation, so it can include time-dependent inhibition. It is
therefore an auxiliary assay for direct-inhibition prediction, rather than a
single point on the direct-inhibition curve. [OpenADMET challenge methods](https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/)

Regression is ranked by the mean soft-threshold relative absolute error
(MA-ST-RAE) across the four isoforms. For a prediction p and the supplied 95%
credible interval [L, U], the absolute error is:

```text
e(p; L, U) = max(0, L - p) + max(0, p - U)
ST-RAE    = sum(e(prediction; L, U)) / sum(e(mean(y); L, U))
```

Predictions inside the interval incur no error. The denominator uses a constant
prediction at the mean observed pIC50, so a score of 1 corresponds to that
baseline. Our implementation follows the
[official scoring code](https://github.com/OpenADMET/CYP-Challenge-Tutorial/blob/main/evaluation/custom_scoring_functions.py).

Weak inhibitors have wider credible intervals in the training data. Median
interval widths across isoforms are 1.21–2.59 pIC50 units for compounds below
pIC50 4, compared with 0.22–0.43 for compounds with pIC50 ≥ 4. The score therefore
tolerates larger deviations from the fitted potency for weak inhibitors, while
still penalizing predictions beyond their interval bounds.

## Combining screening and auxiliary data

The primary screen supplies activity information for compound–isoform pairs
without a fitted direct-inhibition curve. On compounds with both measurements,
Spearman correlations between screen log2 fold-change and direct pIC50 range from
−0.83 for CYP2D6 to −0.94 for CYP3A4. Because dose-response measurements followed
hit selection, the relationship outside this paired subset remains uncertain.

For the selected graph models, we fitted a decreasing isotonic regression per
isoform to map screen readings onto the pIC50 scale. We included the resulting
pseudo-labels only where a direct measurement was missing, with weight 0.3
relative to a direct measurement and a separate assay indicator. We fitted this
calibration within each training fold and excluded validation compounds' screen
readings.

Screen readings on the weak-activity side of the calibrated range provide upper
bounds on pIC50. For CYP2D6, 2,855 of the 2,883 screen readings without a direct
measurement lie beyond that range. We therefore used almost all of its additional
screen observations to constrain low activity without specifying individual
potencies.

The selected regression models also use three public auxiliary sources:

| Source | Information used |
|---|---|
| [Octant CYP release](https://huggingface.co/datasets/openadmet/Octant_CYP_inhibition_reactivity_blog_release) | CYP3A4 pIC50 and credible intervals after assay quality filtering |
| [Tox21 CYP inhibition panel, Sakamuru et al. (2025)](https://pubmed.ncbi.nlm.nih.gov/40507787/) | Biochemical P450-Glo AC50 values and maximum inhibition responses |
| [Novartis dataset on Polaris](https://polarishub.io/datasets/novartis/novartis-cyp3a4-v1) | Released predictions of pIC50 for CYP3A4, CYP2C9 and CYP2D6 |

The Novartis pIC50 values are model-generated surrogate labels. The
[Polaris processing notebook](https://github.com/polaris-hub/polaris-recipes/blob/main/org-Novartis/CYP/00_CYP3A4_data_processing.ipynb)
loads them from the public surrogate training set; our cached table contains
16,372 compounds with these values before filtering. The source paper describes
generating the surrogate data with models trained using in-house assays. We used
the public predictions, without access to those proprietary assay tables, so this
contribution is a form of knowledge distillation. [Fluetsch et al. (2024)](https://pubmed.ncbi.nlm.nih.gov/38501689/)

We selected auxiliary sources by testing their contribution to a fixed graph
model. An early comparison used a directed message-passing neural network
(D-MPNN) with width 300, depth 3 and 30 training epochs, with the screen and
Octant plus Tox21 potency data as the baseline. Adding
Novartis improved both MAE and MA-ST-RAE; adding ChEMBL, SAIR or Veith potency data
increased both errors. The following results use the same five-fold split at
seed 0, before the later architecture and auxiliary-weight changes.

| Additional source beyond Octant and Tox21 | Macro MAE | MA-ST-RAE |
|---|---:|---:|
| None | 0.5408 | 0.6553 |
| Novartis | 0.5327 | 0.6415 |
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | 0.5515 | 0.6743 |
| [SAIR](https://huggingface.co/datasets/SandboxAQ/SAIR), potency labels | 0.5485 | 0.6749 |
| [Veith et al. (2009)](https://pubmed.ncbi.nlm.nih.gov/19855396/), qHTS potency | 0.5501 | 0.6640 |

Removing Octant from the three-source model gave MA-ST-RAE 0.6386, an improvement
of 0.0029 over 0.6415. This was below the staged search's practical improvement
threshold of 0.005; subsequent architecture comparisons retained all three
sources.

We removed auxiliary compounds matching the direct-inhibition training set or
blinded test set by canonical SMILES. For each scored isoform, we rescaled the
total weight of the retained auxiliary observations to nine times the number of
retained direct-inhibition labels, giving each isoform the same ratio of
auxiliary to direct-label weight.

We also tested whether selecting a smaller, more chemically diverse auxiliary
set would improve prediction. Within each isoform, assay and source, we compared
random sampling, MaxMin selection using Morgan-fingerprint Tanimoto distances,
and representatives from fingerprint clusters. We tested a cap of 2,000 rows per
group and retention of 50% or 75% of the rows, holding the total auxiliary weight
per isoform fixed. On Octant plus Tox21, random retention of 75% initially
improved MA-ST-RAE from 0.6631 to 0.6577. Across three seeds, however, its mean was
0.6627 versus 0.6621 for all rows. MaxMin and cluster selection also failed to
improve the corresponding full-data control, so we retained all eligible rows.

The graph models distinguish isoform and assay through separate indicator
vectors concatenated with the molecular representation. Tox21 potency and maximum
response enter as separate outputs, trained with soft-threshold absolute error
and squared error, respectively. The TabPFN component uses uncensored potency
observations and assay indicators, without screen pseudo-labels, response tasks
or sample weighting.

## Molecular representations and task sharing

We compared fixed molecular features with graph representations learned from
the CYP data. LightGBM, multilayer perceptrons and TabPFN used combinations of
Morgan fingerprints, physicochemical descriptors, substructure descriptors and
pretrained molecular embeddings. The embedding comparisons included frozen
CheMeleon and [MAMMAL](https://arxiv.org/abs/2410.22367) representations; for
CheMeleon we also tested fine-tuning the graph encoder. Matched feature-block
comparisons helped distinguish the choice of representation from the choice of
predictor, although the model families differed in their support for censored
labels, response tasks and observation weights.

The selected graph architecture shares molecular features across isoforms and
assays. A D-MPNN updates directed bond representations, constructs atom
representations and averages them into a molecular vector. We concatenate this
vector with molecular descriptors and one-hot indicators for isoform and assay,
then use a shared feed-forward network to predict potency and assay responses.
The encoder receives the molecular graph; the assay and isoform affect the
prediction through the readout. This allows auxiliary assays to train the same
encoder while retaining distinct assay identities.

We varied how much of the network was shared between tasks. Alternatives
included separate encoders per isoform, a shared encoder with independent
isoform heads, independent heads for each isoform–assay pair, and shared hidden
layers ending in separate task outputs. On the 300-wide graph, independent
isoform–assay heads increased MA-ST-RAE from 0.6369 to 0.7168 at seed 0. In a later
paired comparison on the regularized 700-wide model, shared hidden layers with
separate task outputs gave MAE 0.5245 versus 0.5177 for the conditioned readout;
MA-ST-RAE was 0.6283 versus 0.6277. We retained the conditioned readout. We also
tested per-task target standardization and a CYP2D6-specific adaptation layer,
neither of which entered the selected model.

## Joint prediction of dose-response curves

We tested whether paired dose-response curves could combine potency estimates
and single-concentration responses to predict both direct inhibition and TDI.
For each molecule and isoform, the model predicts separate curves for direct
inhibition and NADPH preincubation. Each curve has a midpoint pIC50, a Hill slope h
and a maximum inhibited fraction Emax:

```text
inhibition(c) = Emax / (1 + 10^[h × (pC - pIC50)])
pC = -log10(c), with c in mol/L
```

A fitted potency constrains the midpoint, while a screen reading constrains the
preincubated curve at 49.5 µM. Direct-inhibition predictions are read from the
first curve; TDI calls are derived from the difference between the two midpoints,
with the assay-floor rules applied. The aim was to learn from measured responses
without first converting the screen into pseudo-pIC50 values. We implemented
independent output heads per isoform and condition, as well as alternatives using
protein descriptors. A multilayer perceptron (MLP) or a D-MPNN supplies the
molecular representation to the
[curve model](../src/cypshift/models/curve.py).

The six MLP curve configurations performed poorly in five-fold cross-validation
at seed 0: regression MA-ST-RAE ranged from 0.807 to 0.910 and TDI macro Matthews
correlation coefficient (MCC) from 0.013 to 0.057. The configuration using the
screen, all three selected auxiliary sources and the auxiliary weighting
described above gave MAE 0.6487, MA-ST-RAE 0.8447 and TDI MCC 0.0566. We excluded
these models from the submission. We also checked the D-MPNN curve implementation
in a short holdout experiment, without completing a comparable five-fold
evaluation.

The training targets did not fully constrain curve shape. We trained
on fitted potencies and single-concentration responses, without using the
released Emax measurements or full concentration-response observations. One
screen concentration cannot separately identify the midpoint, slope and plateau;
even a known midpoint leaves the other two parameters underdetermined. The
parameterization starts near a Hill slope of 1 and Emax of 0.9, but those initial
values do not supply additional experimental constraints.

TDI adds a further difficulty: a twofold IC50 change is only 0.301 pIC50 units.
Predicting two potencies accurately enough to classify that difference is a
different objective from minimizing their individual regression errors. Direct
supervision with the provided TDI labels performed better in our experiments.

The curve model also pooled external potency assays into one direct-inhibition
curve per isoform, whereas the selected graph model can learn separate assay
relationships. We did not isolate the contributions of assay pooling, curve
identifiability and model capacity to the observed errors. These explanations
could be tested by separating assays, adding slope or Emax supervision, and
comparing the same molecular encoder with and without the curve constraint.

## Validation and model selection

We used five-fold cross-validation grouped by Bemis–Murcko scaffold, with
stratification to balance label coverage. This does not reproduce the challenge's
hit-expansion process. Public and hidden test subsets are themselves separated by
analogue series, so a calibration that helps the public subset may not transfer
to the hidden series. [Challenge split description](https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/)

We prioritized macro MAE, Pearson correlation and R² when comparing regression
models, while also reporting MA-ST-RAE. Early experiments showed that lower local
ST-RAE could favor predictions with too little variation, and the preferred
calibration did not transfer to the public leaderboard. The graph models still
use a soft-threshold training loss; the model-selection criteria are a separate
choice.

The seed changes both initialization and fold assignments. For one baseline
configuration, the standard deviation across five runs was approximately 0.0035 in MAE
and 0.0047 in Pearson correlation. We repeated the main graph comparisons across
seeds, but TabPFN and LightGBM comparisons used one seed.

We used the same cross-validation results to select models and estimate their
performance. Some graph runs also used the validation fold for early stopping.
The reported scores are therefore development estimates; we did not perform an
independent nested evaluation of the complete selection procedure.

## Hyperparameter sweeps and training comparisons

We organized the later sweeps around specific data and modelling questions.
Source comparisons held the graph model fixed; architecture comparisons used
the selected source set. Most candidates first ran on one five-fold split, with
selected graph configurations repeated across seeds. We then compared their
saved out-of-fold predictions both individually and in ensembles. The
[experiment driver](../scripts/night_run.py) records the configurations and
controls for these comparisons.

Auxiliary representation and weighting interacted in the graph experiments.
With all three retained sources, adding Tox21 maximum-response observations to
potency observations left MA-ST-RAE at 0.6415 under the original weights. With
the auxiliary weight fixed at nine times the direct-label count per isoform,
adding those responses improved MA-ST-RAE from 0.6492 to 0.6369. We therefore
retained potency and maximum response together with that weighting. Replacing
screen pseudo-labels with a raw log2 fold-change response task gave 0.6434 on the
same split, compared with 0.6369 for the isotonic pseudo-label approach.

The main graph-capacity sweep tested width/depth pairs of 300/3, 500/4, 700/5 and
1,000/6, together with readout depth, dropout, learning rate and training duration.
Increasing capacity alone did not improve prediction. The following examples
show how regularization and fine-tuning affected the choices; each comparison
uses its corresponding control at seed 0.

| Question | Comparison | Result and choice |
|---|---|---|
| Dropout at greater capacity | Dropout 0.05, 0.10 and 0.20 on the 500-wide, depth-4 graph | MA-ST-RAE decreased from 0.6847 to 0.6682 to 0.6458; this motivated stronger regularization for the larger models. |
| Capacity with regularization | The 300-wide baseline versus the 700-wide graph with dropout 0.30, peak learning rate 10⁻⁴ and up to 100 epochs | MAE decreased from 0.5304 to 0.5229, although MA-ST-RAE was similar, 0.6353 versus 0.6373. |
| Further capacity increase | A 1,000-wide, depth-6 graph with the same dropout, learning rate and epoch limit | MAE was 0.5336 and MA-ST-RAE 0.6633; we retained the 700-wide model. |
| Per-isoform fine-tuning | 30 epochs at 10⁻⁴ versus 10 epochs at 10⁻⁵ after pooled training | The longer schedule gave MA-ST-RAE 0.6553; the shorter schedule gave 0.6335 and entered the final model. |
| Training objective | Soft-threshold, absolute, squared, quantile and expectile losses; an added point-error term for the graph | We retained soft-threshold loss for the graph components and squared error for LightGBM, whose predictions improved the ensemble. |

Training schedules also tested whether auxiliary learning should precede
adaptation to the challenge assay. We compared pooled training with subsequent
direct-potency fine-tuning, with either a trainable or frozen encoder, and with
per-isoform fine-tuning. The selected per-isoform stage keeps that isoform's
auxiliary assays; a separate direct-assay-only variant was also evaluated.

Pretrained weights required their own tuning: applying the larger graph's
readout and training settings to CheMeleon reduced seed-0 Pearson correlation
from approximately 0.635 to 0.621, so the selected CheMeleon component retained
its separate 30-epoch configuration.

## Selected regression ensemble

A directed message-passing neural network implemented in
[Chemprop v2](https://pubmed.ncbi.nlm.nih.gov/41453060/) was the strongest individual
model family in our completed comparisons. It combines a learned molecular graph
representation with physicochemical and substructure descriptors, including
basic amines and potential heme-coordinating groups. The selected model has hidden
width 700, message-passing depth 5, two readout layers of width 700, dropout 0.3,
a peak learning rate of 10⁻⁴ and a maximum of 100 epochs. After pooled training,
we fine-tuned one copy per isoform for 10 epochs at 10⁻⁵. This reduced MAE by
approximately 0.002 in the initial paired runs.

Representation diversity improved the ensemble beyond averaging graph seeds.
Fine-tuning the pretrained [CheMeleon encoder](https://arxiv.org/abs/2506.15792)
gave weaker standalone predictions, but its addition improved Pearson correlation
for every isoform across each of the three evaluated seeds. The largest increase
was on CYP2D6: adding 30% CheMeleon to the five-seed graph average increased Pearson
correlation by approximately 0.016.

The selected weights are shared across isoforms:

| Component | Weight | Representation and training |
|---|---:|---|
| Chemprop D-MPNN | 0.50 | Molecular graph and descriptors; soft-threshold loss; per-isoform fine-tuning |
| CheMeleon D-MPNN | 0.20 | Pretrained encoder, fine-tuned for 30 epochs with soft-threshold loss |
| TabPFN | 0.15 | Physicochemical and substructure descriptors plus CheMeleon embeddings; combined features reduced to 128 principal components |
| LightGBM | 0.15 | Radius-2, 4,096-bit Morgan fingerprints and descriptors; squared-error loss |

[TabPFN](https://doi.org/10.1038/s41586-024-08328-6) uses one fitted regressor per
isoform with its internal ensembling. We refitted the two graph components and
LightGBM on all available training data as five-member ensembles. LightGBM also
uses auxiliary maximum-response predictions as features. Its squared-error
version contributed more to the ensemble than its soft-threshold version.

| Model | Macro MAE | Macro Pearson r | Macro R² | MA-ST-RAE |
|---|---:|---:|---:|---:|
| Selected graph, five-seed average | 0.5135 | 0.6599 | 0.4446 | 0.6182 |
| Graph + TabPFN + LightGBM | 0.5082 | 0.6670 | 0.4525 | 0.6064 |
| Selected four-component ensemble | 0.5064 | 0.6726 | 0.4553 | 0.5980 |

These are out-of-fold results before leaderboard calibration. The four-component
estimate averages five graph seeds and three CheMeleon seeds, with TabPFN and
LightGBM at seed 0. It therefore differs from the final refit ensemble size.
Nearby ensemble weights gave similar cross-validation scores. The
[regression specification](../configs/best_regression_model.json) records the
configurations and comparisons, and [best_model.py](../scripts/best_model.py)
assembles and calibrates the predictions.

## Calibration from public leaderboard feedback

Differences in the mean and variance of predicted potencies accounted for a large
part of our initial public-test error. An earlier graph model using Octant and
Tox21 scored MA-ST-RAE 0.781 before calibration and 0.550 after changing its
per-isoform mean and standard deviation. We estimated calibration targets from
aggregate R² feedback on affine transformations of submitted predictions, then
tested additional shifts and scales against ST-RAE. These targets are approximate:
the public subset is unknown, its prediction mean and variance can differ from
those of the full submission, and the reported scores are bootstrap summaries.

For the selected four-component ensemble, the following settings gave the best
measured public score for each isoform. The standard-deviation multiplier and
mean shift are relative to the initial calibration targets.

| Isoform | Standard-deviation multiplier | Mean shift, pIC50 | Public ST-RAE |
|---|---:|---:|---:|
| CYP1A2 | 1.15 | +0.10 | 0.4941 |
| CYP2C9 | 1.00 | 0.00 | 0.3697 |
| CYP2D6 | 1.50 | 0.00 | 0.6514 |
| CYP3A4 | 1.00 | +0.10 | 0.4170 |

Combining these calibrated predictions improved public MA-ST-RAE from 0.5064 to 0.4831, with
macro Spearman correlation approximately 0.729. Calibration used public test
feedback, and its gains have not been validated on the hidden series. The tested
settings also showed that improving MAE or R² need not improve ST-RAE.

## Time-dependent inhibition classification

For classification, we trained the graph model directly on the supplied binary
TDI labels through an additional score output. The selected graph variants use
squared error on this output alongside the potency and response tasks. Training
on the labels gave better results than deriving calls from predicted potency
shifts. We used a separate five-fold scaffold split stratified on TDI labels and
evaluated model comparisons on compounds with a direct measurement: 1,495 for
CYP2D6 and 2,346 for CYP3A4.

Excluding TDI labels without a direct measurement increased graph macro MCC from
0.274 to 0.305 at seed 0, with thresholds chosen to match the evaluated subset's
positive fraction. These labels were all negative in the training subset we
examined, and including them at full weight reduced MCC on the measured subset.
We retained two graph variants in the ensemble, assigning these observations
weights of 0 and 0.25. The zero-weight variant averages two seeds. The comparison
above used one seed; the zero-weight model scored 0.270 on the second seed.

The classification ensemble averages percentile-transformed scores separately
for each isoform. CYP2D6 combines the two graph variants, LightGBM on CheMeleon
embeddings and a nearest-neighbour model, each with weight 0.25. The latter uses a
Tanimoto-weighted vote of the 15 nearest training compounds by Morgan fingerprint.
CYP3A4 combines the two graph variants and LightGBM on fingerprints and descriptors,
each with weight one third. The ensemble's out-of-fold ROC AUC is 0.661 for CYP2D6
and 0.782 for CYP3A4. Graph test predictions average the five fold models, rather
than using the regression refits. The procedure is implemented in
[tdi_submission.py](../scripts/tdi_submission.py).

Aggregate public precision, recall and accuracy imply a CYP2D6 positive fraction
of approximately 7.1%, compared with 21.7% in the measured training subset. We
therefore tested thresholds to account for this estimated change in prevalence
and selected them using public feedback. Their benefit on the hidden series
remains unmeasured.

The 2 November leaderboard snapshot records macro MCC 0.3232, with 0.2290 for
CYP2D6 and 0.4174 for CYP3A4. The prepared final candidate combines the per-isoform
predictions with the best observed MCC: 113 positive calls among 750 compounds for
CYP2D6 and 232 for CYP3A4. Their separately observed MCC values imply a public macro MCC of
approximately 0.3334; this combined file had not been confirmed on the leaderboard
at that snapshot.

## Chemical interpretation of CYP2D6 predictions

CYP2D6 showed a different association with molecular properties from CYP3A4.
In the measured training subsets, the Spearman correlation between calculated
logP and pIC50 was 0.034 for CYP2D6 and 0.602 for CYP3A4. Compounds matching our
SMARTS pattern for basic aliphatic amines had a mean CYP2D6 pIC50 0.554 units
higher than non-matching compounds; the corresponding difference was negative
for the other three isoforms. These comparisons use unadjusted associations in
the selected compounds, without matched molecular pairs.

The amine association is consistent with structural studies showing interactions
between cationic ligands and acidic residues including Asp301 and Glu216. Their
roles depend on the ligand and binding geometry; a substructure association alone
cannot identify a particular interaction. [Wang et al. (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4335244/)
These observations motivated explicit amine descriptors; the additional geometry
descriptors produced only small gains in early tests.

Biology and assay design are difficult to separate here. CYP2D6 was measured by
dextromethorphan depletion using mass spectrometry, while the other three
isoforms used fluorogenic probes. [Assay description](https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/)
Together with test selection based on the other isoforms, this offers plausible
explanations for its distinct distribution and weaker prediction accuracy. Our
experiments do not determine how much each contributes.

## Exploratory protein–ligand structure prediction

We also tested [Boltz-2](https://pubmed.ncbi.nlm.nih.gov/40667369/) with CYP1A2,
heme and imidazole, aiming to obtain features describing heme coordination. In
the predicted complex, an imidazole nitrogen was 2.19 Å from the heme iron. This
was a geometric plausibility check on one complex, not a validation of pose or
affinity accuracy across the challenge compounds. Structure-derived features
were not used in the selected regression or classification ensembles, and this
exploration did not produce a validated structure-track submission.
