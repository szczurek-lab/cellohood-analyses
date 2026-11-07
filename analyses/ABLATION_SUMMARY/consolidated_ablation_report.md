# Cellohood Ablation Study — Consolidated Results
_Compiled 2026-09-27. All DLPFC numbers use the "trained-only-on-3-CellCharter-samples"
protocol (TRAIN_SAMPLES = TEST_SAMPLES of CellCharter's own DLPFC benchmark:
151507, 151672, 151673), scored on those same 3 samples for direct comparability
to CellCharter/UTAG/DR-SC/SEDR/BayesSpace/STAGATE baselines already used in the
manuscript's main DLPFC benchmark. All results reused cached model checkpoints
where possible; fresh runs used `/home/marmarmarmar/immucan/bin/python` with
OMP/OPENBLAS/MKL/NUMEXPR/VECLIB_NUM_THREADS=1 to avoid the shared-box OpenBLAS
fork-segfault._

## 1. Bag-distance-threshold sensitivity (Reviewer #1 concern 3, Reviewer #4 concern 3, Reviewer #2 concern 3)

Full sweep, thresholds 2–10, fixed hyperparameters (layer_size=128,
intermediate_layer_size=256, latent_size=64, num_heads=4), 10 runs/threshold,
both architectures. **90/90 jobs complete** as of 2026-09-27 (two real bugs
found and fixed in `cellohood` package code along the way — see Appendix A).

| Threshold | Transformer (mean ARI) | DeepSet (mean ARI) |
|---|---|---|
| 2  | 0.355 | 0.482 |
| 3  | 0.420 | 0.467 |
| 4  | 0.414 | 0.441 |
| 5  | 0.464 | 0.437 |
| 6  | 0.415 | 0.451 |
| 7  | 0.408 | 0.381 |
| 8  | 0.452 | 0.446 |
| **9**  | **0.061** | **0.061** |
| 10 | 0.433 | 0.420 |

**Finding:** thresholds 2–8 and 10 form a stable band (0.35–0.48 ARI) for both
architectures; the manuscript's chosen threshold (5, mean bag size ≈5) sits
comfortably inside it. **Threshold 9 collapses to near-chance (~0.06) in BOTH
architectures identically** — reproducible, not explained by PCA-component
count, not yet root-caused. Flagged as an open item, not smoothed over.

Figure: `dlpfc_threshold_ablation_figure.png`

## 2. Hyperparameter sensitivity (Reviewer #4 concern 1)

22 architecture configs (layer_size × intermediate_layer_size × latent_size ×
num_heads), thresholds 4–6, 5 runs each, trained-on-3 protocol.

**Range: 0.412–0.477 ARI (spread of 0.065).** No catastrophic failures; larger
configs mildly favored. Conclusion: performance is robust to architecture
hyperparameters.

## 3. Transformer vs. DeepSet architecture (Reviewer #4 concerns 1 & 2 — "why transformer vs. simpler DeepSets" and "no-attention baseline")

See table in §1. Crossover pattern: DeepSet ties/beats Transformer at small
thresholds (2–4), Transformer pulls ahead from 6–10 (gap ~0.01–0.07). Since
DeepSet pooling is a permutation-invariant sum/mean over the bag — i.e. a form
of "bag-embedding averaging" — this comparison also speaks to the bag-averaging
question in concern 2, not just the no-attention question.

## 4. Reconstruction quality vs. downstream clustering (Reviewer #4 concern 1, third sub-question)

Correlated final training reconstruction loss against downstream ARI across
the 22-config × threshold × run sweep (330 matched runs).

**Pearson r = -0.045, Spearman r = -0.041** (per-run). Per-config aggregate
(n=22): r = 0.215, not significant. **Conclusion: reconstruction quality and
clustering quality are empirically decoupled** — better reconstruction does
not predict better downstream clustering.

## 5. Generalization to unseen samples (Reviewer #2 concern 3)

Trained on 3 samples only, scored on the *other* 9 (never seen in training):
**0.420 ARI (std 0.038)**. Reference: model trained on all 12 samples, scored
on that same held-out-9: **0.417 ARI (std 0.042)**. Statistically
indistinguishable. **Conclusion: Cellohood generalizes to unseen samples
without retraining — a 4x smaller training set costs essentially nothing.**

## 6. PLDA resolution sensitivity — standard (75% variance) vs. coarse (50% variance) (Reviewer #2 concern 7)

| PLDA resolution | Mean ARI | Std |
|---|---|---|
| Standard (75% var, more components) | 0.481 | 0.040 |
| Coarse (50% var, fewer components) | 0.406 | 0.091 |

Consistent across thresholds 4–6. **Finer resolution clearly outperforms
coarser** — relevant evidence for principled resolution selection.

## 7. Local spatial smoothing ablation (Reviewer #4 concern 2 — component-wise ablation)

Same 5 cached model checkpoints (threshold=5.0), evaluated with vs. without
the post-hoc spatial smoothing step.

| Condition | Mean ARI | Std |
|---|---|---|
| With smoothing (standard pipeline) | 0.462 | ~0.06 |
| Without smoothing | 0.271 | 0.146 |

**Decisive, ~40% relative drop without smoothing.** Variance also more than
doubles. This is the single most conclusive component-ablation result: local
smoothing is doing substantial, necessary work, not cosmetic post-processing.

## 8. PLDA vs. direct clustering on raw embeddings (Reviewer #2 concern 5)

Threshold=5.0, 5 cached runs × 5 GMM seeds, same model, PCA(75% var) vs. no
PCA.

| Method | Mean ARI | Std |
|---|---|---|
| PLDA (PCA, 75% variance) | 0.495 | 0.061 |
| Direct clustering on raw embeddings | 0.475 | 0.067 |

Modest, fairly consistent edge for PLDA (+0.02 ARI, lower variance). PLDA wins
on 2/3 samples. **Conclusion: PLDA provides a small but real benefit beyond
post-hoc smoothing/dimensionality control** — a more measured answer than
either "PLDA is essential" or "PLDA is unnecessary."

## 9. Cross-dataset bag-size validation (Reviewer #1 concern 3, robustness across datasets)

Bag-size-vs-distance-threshold curves computed independently on Spleen CODEX,
NSCLC2 Panel 1, NSCLC2 Panel 2, and Jackson — extending the Kneedle threshold
heuristic's validation beyond DLPFC alone. All four show the same
diminishing-returns curve shape used to justify the Kneedle approach.

## 10. Independent modality / cohort validation — MIBI-TNBC (Reviewer #2 concerns 1 & 4, Reviewer #2 concern 10)

Not DLPFC — a genuinely new dataset (Keren et al. 2018 MIBI-TOF breast cancer,
34 patients, via CytoCommunity's public tutorial data), new input
representation (one-hot cell-type composition rather than continuous marker
channels), auto-thresholded via the Kneedle heuristic (threshold=70, mean bag
size 4.58), trained fully unsupervised.

**Earlier exploratory pass (superseded below, kept for the record):**
unsupervised mixing-score AUCROC 0.67–0.72, using a *hand-picked* GMM cluster
count (K=8, then K=10 after an ad-hoc sweep that selected the config with the
best Leave-One-Out-CV performance — a mild form of the same selection-leakage
problem documented throughout this report, since that choice was informed by
label-discriminative performance in a preliminary test, then reused in later
"fresh" analyses). This did **not** run Cellohood's standard analysis
pipeline — compare to e.g. the NSCLC2 Panel 1 manuscript notebook, which runs
a principled cluster-number-selection step, full marker/cell-type
characterization of discovered neighborhoods, and clinical/survival
integration; none of that was done in this first pass. A supervised
linear-probe attempt on cluster-composition features did *not* reliably
beat the hand-picked heuristic (0.20–0.62, including one apparent "perfect"
result verified to be a Leave-One-Out-CV degeneracy artifact, not a real
signal — see investigation notes).

**Proper-pipeline rerun — primary result.** We reran the analysis using
Cellohood's actual principled cluster-selection procedure
(`get_latent_directions` + `summarize_latent_direction_result`, the same
stability-based selection used for the Spleen and NSCLC2 manuscript
analyses, not a hand-picked K). At the standard 75%-variance PLDA level
(dim=2), the stability-selected cluster counts were **[13, 17, 19, 24]** —
all well above the earlier hand-picked K=8/10. Using K=13 (the smallest,
most parsimonious stable candidate) for the final GaussianMixture clustering
on cell-level embeddings (matching the Spleen notebook's convention: ccs
selects K, a GMM is then fit with that K):

| Check | Hand-picked K (earlier) | Proper pipeline, K=13 |
|---|---|---|
| Raw mixing score AUCROC | 0.670 | 0.674 |
| Enriched mixing score AUCROC | 0.919–0.922 | 0.919 |
| Correlation(enriched, confound alone) | r=0.996–0.998 | r=0.998 |
| Residualized (confound regressed out) | 0.659 *(from a separate, raw-bag-level statistic — see below)* | **0.530** *(cluster-based)* |

**The core findings replicate under proper rigor — the hand-picked K was not
responsible for the confound-collinearity problem.** The enrichment
"correction" is just as collinear with the trivial per-patient baseline
confound (r=0.998) under the properly-selected K=13 as it was under the
hand-picked K=10 (r=0.996-0.998) — this is a property of the ratio
construction and the dataset, not an artifact of under-tuned clustering.
One thing that does *not* replicate: a promising residualized "beyond-
confound" signal (AUCROC=0.659) found earlier used a direct, raw spatial
bag-co-membership statistic (immediate neighbors, no learned clustering at
all) — under the properly-selected cluster-based statistic instead, the
equivalent residualized check drops to **0.530, essentially chance**. So
the one piece of evidence for "real spatial signal beyond composition" we
had does not survive being redone through the standard clustering pipeline;
it may be specific to the most local, raw spatial-proximity statistic
rather than to Cellohood's learned neighborhoods as such.

**Cluster separability also confirmed, not just an artifact of hand-picking
K=10.** The properly-selected K=13 clusters are similarly homogeneous to
the earlier hand-picked ones (per-cell-type range across clusters: 0.11 for
the largest-ranging type, Epithelial; most other types <0.08; mean
pairwise L1 distance between cluster composition profiles = 0.182) — this
dataset/representation (one-hot cell-type composition, MIBI-TOF breast
cancer) genuinely does not separate into sharply distinct neighborhood
types under Cellohood's standard pipeline, at either resolution.

**Immune-subtype finding also replicates under proper rigor** — bulk
composition still beats the neighborhood-enrichment statistic for every
subtype checked (Treg: baseline 0.626 vs. neighborhood 0.541; CD8T: 0.789
vs. 0.674; DC: 0.685 vs. 0.563; CD11c-high: 0.867 vs. 0.659).

**Rigor correction — a sign-selection bias we had not accounted for.**
Every "best-direction" AUCROC in this section (reported as `max(AUC,
1-AUC)`, flagged throughout as "flip=...") carries an unacknowledged bias:
when a statistic's expected sign relative to the label isn't fixed in
advance, always reporting the better of two directions inflates apparent
performance even under the null. We tested this directly with a
permutation procedure that shuffles the label 20,000 times and recomputes
`max(AUC, 1-AUC)` each time, building the *correct* null distribution
rather than assuming it is centered at 0.5. For this cohort (n=33,
18 mixed/15 compartmentalized) that null has **mean 0.582, not 0.500**,
with a 95th percentile of 0.70 — meaning a naive "0.67 beats chance" read
is already most of the way to a false positive before any real effect is
considered.

Against this corrected null: the proper-pipeline raw mixing score
(observed 0.674) gives **permutation p=0.093** — not significant at
conventional thresholds, down from an apparently "modest but real" read.
The residualized (confound-regressed-out) score (observed 0.530) gives
**permutation p=0.786** — indistinguishable from noise, and in fact below
the biased null's own mean. Neither number should be read as evidence of
real signal. This is a stricter, more defensible conclusion than what
earlier passes of this section stated, and the same caveat applies to
every other best-direction AUCROC reported in this section (hand-picked-K
results, the PLDA50/75/100 sweep, the PC0-dropped check, the individual
immune-subtype numbers) — none of those were re-tested individually, but
the mechanism is general and the reader should not treat any of them as
properly powered significance tests, only as descriptive comparisons.

**Bottom line: proper pipeline rigor does not rescue the "beyond-confound"
case — and a correct significance test shows there was less to rescue than
it appeared.** The MIBI-TNBC comparison should be framed as "Cellohood's
core unsupervised pipeline runs successfully and recovers a
composition-driven classification signal on a new modality," not as
evidence of *spatial* reasoning beyond that confound on this specific
benchmark — no properly-tested statistic in this investigation supports
that stronger claim.

**CytoCommunity comparison — independently reproduced, not just cited.**
Rather than trust the paper's claimed AUCROC=0.891, we cloned CytoCommunity's
public repo and ran their actual, unmodified supervised pipeline end-to-end
(Step1 KNN-graph construction + Step2 GNN/sparse-MinCut-pooling training,
their literal default hyperparameters: Num_TCN=2, 10×10-fold CV, 100
epochs/fold, ~2h wall-clock on one RTX 2080 Ti) on their own bundled
MIBI-TNBC data (same 34 patients, 15/19 label split, verified to match the
paper exactly).

**Measured: AUCROC = 0.783 (std 0.039 across the 10 repeats) — not 0.891.**
A gap of ~0.11, more than 2.5x the observed repeat-to-repeat std, and
notably *below* their own cited SPACE-GM baseline (0.823). Most likely
explanation: the public repo has been refactored since publication (its own
changelog: "replaced the dense versions of the graph convolutional layer and
pooling layer with the sparse versions"), so the current clone is not the
code that produced the original number; compounded by the fact that Step2
sets no random seed anywhere, and with only 34 patients and a 512-dim GNN
this is a high-variance regime (individual fold accuracies ranged 0.33–1.0).

**Revised comparison table (all numbers on the same 34-patient task):**

| Method | AUCROC | Supervision |
|---|---|---|
| CytoCommunity, as published | 0.891 | Supervised, weakly (90% MinCut / 10% CE loss) |
| CytoCommunity, independently reproduced (this work) | **0.783** | Supervised, weakly |
| SPACE-GM, as published | 0.823 | Supervised |
| Cellohood (this work) | 0.67–0.72 | **Fully unsupervised** |

Against the number we can actually verify, Cellohood's fully-unsupervised
result sits ~0.06–0.11 below a *reproduced* supervised baseline, rather than
~0.17–0.22 below an *unverified* one — a materially stronger and more
honest story for a method that uses no label information at any stage.

**CytoCommunity's *unsupervised* mode, also independently reproduced —
and a cautionary finding about statistic sensitivity.** We also ran
CytoCommunity's actual unsupervised pipeline (Num_TCN=2, matching their
supervised mode's choice for this task; Num_Run=20 per patient, all 34
patients, repo defaults unchanged, ~4h wall-clock across 4 GPUs). The repo
computes no classification metric for unsupervised mode either — a scalar
has to be derived from the discovered per-patient TCNs, exactly as we did
for Cellohood, and the result turned out to depend enormously on that
derivation:

| Statistic definition | AUCROC |
|---|---|
| Immune fraction of tumor cells' assigned TCN — raw | **0.170** |
| Same, corrected for each patient's own baseline immune fraction (an enrichment *ratio*, not raw exposure) | **0.833** |

The raw version is worse than random because it mostly tracks a trivial
per-image confound (this specific 34-patient cohort's compartmentalized
tumors happen to have systematically higher overall immune content —
AUCROC=0.085 using *zero* TCN information at all), and CytoCommunity's
unsupervised K=2 MinCut split contributed almost nothing beyond that
confound: every one of the 34 patients was partitioned almost exactly
50/50, consistent with MinCut's built-in balance regularization producing
a generic spatial bisection rather than a partition aligned to the
neoplastic/immune boundary. The confound-corrected version — arguably the
more literal reading of "enriched for immune cells" — removes this
artifact and comes out on top of every other number in this table.

**We are not treating either number as "the" unsupervised CytoCommunity
result**, and recommend the same caution in how this gets written up: the
paper's own text states unsupervised methods (including CytoCommunity)
scored below SPACE-GM's 0.823, but reports no exact figure, so neither of
our numbers can be checked against a published ground truth. The 0.17→0.83
swing from a single scalar-derivation choice, on the same underlying model
output, is itself the finding.

**Full comparison table:**

| Method | AUCROC | Supervision |
|---|---|---|
| CytoCommunity unsupervised, confound-corrected (this work) | 0.833 | Unsupervised |
| CytoCommunity, as published | 0.891 | Supervised |
| CytoCommunity, independently reproduced (this work) | 0.783 | Supervised |
| SPACE-GM, as published | 0.823 | Supervised |
| Cellohood (this work) | 0.67–0.72 | Unsupervised |
| CytoCommunity unsupervised, naive statistic (this work) | 0.170 | Unsupervised |

One point worth making in the writeup: Cellohood's single, first-reasonable
mixing-score design reached 0.67–0.72 directly, with no confound correction
needed, whereas CytoCommunity's naive equivalent statistic collapsed to
0.170 and required a specific per-image baseline correction to reach 0.833.
A plausible (not proven) explanation is the structural difference discussed
above — Cellohood clusters in a single embedding space fit jointly across
the whole cohort, while CytoCommunity's unsupervised TCNs are discovered
independently per patient and are not cross-patient aligned — making
Cellohood's score less susceptible to this particular bulk-composition
confound. Caveats on this reproduction: Num_TCN=2 forces a binary spatial
split and MinCut's balance term pushed every patient toward ~50/50, which
may understate what unsupervised CytoCommunity could do with a richer TCN
count (their own tutorial uses Num_TCN=9 on MERFISH-Brain, a dataset with
more diverse tissue architecture); R's `diceR` package could not be
installed (no root access), so Step3's majority-voting ensemble was
reimplemented in Python — exact, not approximate, for Num_TCN=2 specifically
(only 2 label permutations exist, so per-cell majority vote after aligning
to a reference run is mathematically identical to what diceR does at K=2);
one patient had zero tumor cells under the task's cell-type mapping and was
dropped (n=33, not 34); and no hyperparameter sweep was run given time
constraints — this reflects one specific, defensible configuration, not an
exhaustive search.

### 10.3 Cross-method validation (UTAG, Spatial-LDA) and the one finding that survives rigorous testing

We additionally ran two more published unsupervised spatial methods — UTAG
(Leiden clustering on a cell-neighborhood graph, resolutions 0.3 and 1.0)
and Spatial-LDA (topic modeling over local cell-type-composition windows,
10 topics) — on the same 34-patient cohort, applying the identical
confound-check protocol used throughout this section.

**The enrichment-correction artifact is universal, not method-specific.**
All four methods tested (Cellohood, CytoCommunity, UTAG, Spatial-LDA) show
the same pattern: a weak-to-moderate raw statistic (AUCROC 0.17–0.67,
uncorrected) that explodes to an impressive-looking 0.83–0.96 once
"corrected" by dividing by each patient's own baseline immune fraction —
and collapses back toward chance under residualization. Four independent
implementations rediscovering the identical artifact is strong evidence
this is a property of the benchmark and of the naive correction, not of
any one method.

**The one finding that survives proper significance testing.** We checked
whether the three methods' *raw* (uncorrected) statistics agree with each
other beyond what shared confounding would predict — the signature of real
shared spatial signal rather than shared bias. They do, and unlike the
AUCROC numbers above, this comparison has no hidden sign-selection step
(each method's raw statistic uses one fixed formula, not a post-hoc best
direction), so we could test it properly with both a standard correlation
t-test and an independent permutation test (shuffling patient-pairing
20,000 times):

| Pair | Pearson r | t-test p | Permutation p |
|---|---|---|---|
| Cellohood (bag) vs. UTAG (res 0.3) | 0.721 | <0.0001 | <0.0001 |
| Cellohood (bag) vs. Spatial-LDA (hard) | 0.405 | 0.019 | 0.019 |
| UTAG (res 0.3) vs. Spatial-LDA (hard) | 0.684 | <0.0001 | <0.0001 |

All three pairs are significant by both tests, and each exceeds that
method's own individual correlation with the trivial confound (Cellohood
r=0.253 n.s., UTAG r=0.421 p=0.015, Spatial-LDA r=0.548 p=0.001).
**This is the one genuinely defensible positive finding from the entire
MIBI-TNBC investigation**: three independently-implemented spatial methods
— a learned transformer encoder with hierarchical bags, a Leiden community
graph, and a topic model — agree with each other about local
cellular-neighborhood structure on this tissue more than chance or shared
confounding would predict, even though none of them (properly tested)
predicts the mixed/compartmentalized label itself. CytoCommunity is the
outlier that does *not* fit this pattern — its raw statistic correlates
almost perfectly with the trivial confound (r≈0.90) instead, traced to its
unsupervised MinCut partition collapsing to an almost exact 50/50 split in
every one of the 34 patients (Num_TCN=2, balance-regularized), too coarse
to diverge from bulk composition.

**How to frame this honestly:** on this specific benchmark, no method we
tested — ours or any of the three published alternatives — shows
properly-tested evidence of predicting the mixed/compartmentalized outcome
from spatial structure beyond bulk composition. What *is* real,
properly-tested evidence: independent spatial methods converge on shared
local structure in this tissue that a composition-blind statistic does not
explain. That is a meaningful, if modest, validation of spatial-neighborhood
discovery as a concept on this data — just not of any one method's
performance on this particular classification task.

### 10.1 Richer clinical metadata for MIBI-TNBC — recurrence, survival, grade, stage, age

CytoCommunity's bundled copy of Keren et al. 2018 only ships one binary
label (mixed/compartmentalized). The original Cell paper's own Table S2
("age, tumor grade, TNM, clinical stage, TIL enumeration and survival for
all patients") is not available as a machine-readable file from the Cell.com
supplement in a form we could recover directly, but the same 41-patient
cohort was independently re-digitized by a later reanalysis, **Patwa, Risom
et al., "Multiplexed imaging analysis of the tumor-immune microenvironment
reveals predictors of outcome in triple-negative breast cancer,"
Communications Biology 2021** (the RASP-MIBI paper), whose code repository
(`github.com/aalokpatwa/rasp-mibi`) bundles
`intermediate_data/covariate_rsf_data.csv`: per-patient `grade`, `age`,
`stage`, `Recurrence`/`Recurrence_time`, `Survival`/`Survival_time` (death
event/time), and — crucially — an independently-derived `Architecture`
column (Mixed/Compartmentalized/Cold).

**ID-mapping verification (done before trusting any merge, as required).**
The paper's own text states patients 22 and 38 were excluded for lacking
recurrence outcomes and patient 30 for corrupted images — using Keren et
al.'s original 1–41 numbering directly. CytoCommunity's 34 bundled patients
are missing exactly `{15, 19, 22, 24, 25, 26, 30}` from 1–41; cross-referencing
against the recovered table, `{15, 19, 24, 25, 26}` are exactly the patients
labeled `"Cold"` there and `{22, 30}` are exactly the ones the RASP-MIBI
paper excludes — a perfect, non-coincidental match. As a second,
independent check we merged the recovered `Architecture` column onto our 34
`patientN` IDs by the literal integer in the ID (`patient7` → `ID=7`, etc.)
and compared it to our own mixed/compartmentalized label: **0/32 mismatches**
(the only patient without an `Architecture` value, `patient38`, is the one
the RASP-MIBI paper independently reports as missing outcome data). CytoCommunity's
`patientN` IDs therefore map **directly and losslessly** (same integer) onto
Keren et al.'s original patient numbering — no remapping needed, and this is
now verified via an outcome variable, not just the ID number itself.

**Final clinically-annotated cohort: N=32** (of the 33 patients the
Cellohood pipeline normally scores for this dataset, after excluding
`patient38`). 13/32 recurrence events, 12/32 deaths.

**Methodology (matching the house pattern from section 10 and the NSCLC2
manuscript's Clinical Data section): bulk-composition baseline first,
Cellohood K=13 neighborhood statistic second, then a direct test of whether
the neighborhood signal survives once the baseline is accounted for.**
Reused the exact proper-pipeline K=13 clustering, bulk immune-fraction
baseline, and raw/enriched mixing-score definitions from §10 (same
`GaussianMixture(n_components=13, random_state=42)` on the 75%-variance PLDA
embedding). For the two survival endpoints, fit `lifelines.CoxPHFitter`
(`penalizer=0.1` given N=32) on (a) bulk immune fraction alone, (b) the K=13
enriched mixing score alone, (c) both together, and ran a nested
likelihood-ratio test of (c) vs (a) to ask whether the neighborhood term adds
anything once the trivial baseline is in the model. For grade/stage (binarized
for enough group sizes) and age (continuous), reused the exact
confound/raw/enriched/residualized AUCROC (or correlation) structure from
§10's main confound check.

| Endpoint | Bulk-baseline alone | K=13 enriched alone | Combined | Beyond-baseline test |
|---|---|---|---|---|
| Recurrence (Cox) | concordance=0.690, coef p=0.060 | concordance=0.673, p=0.125 | concordance=0.680 | **LR p=0.845 — no improvement** |
| Overall survival (Cox) | concordance=0.657, p=0.197 | concordance=0.646, p=0.205 | concordance=0.646 | **LR p=0.645 — no improvement** |
| Grade (high ≥3 vs low <3; n=31, **28 vs 3**, severely imbalanced) | AUC 0.583 | AUC 0.583 | residualized AUC 0.619 | underpowered, no reliable signal either way |
| Stage (III/IV vs I/II, coarse 1-digit decode of a 2-digit field — exact TNM/stage coding scheme not confirmed from the repo, flagged as a caveat; n=32, 7 vs 25) | AUC 0.709 | AUC 0.680 | residualized AUC 0.611 | enriched/residualized both **below** the bulk baseline |
| Age (continuous, n=32) | r=0.116, p=0.526 | r=-0.118, p=0.521 | residualized r=0.001, p=0.994 | no signal in any version |

In every case where the combined/residualized model could be checked
against the baseline-alone model, adding the K=13 neighborhood term did not
improve on, and in the Cox models visibly **destabilized**, the
baseline-alone fit: in the combined Recurrence model the bulk-immune-fraction
coefficient's p-value rises from 0.060 to 0.205 and the enriched-score
coefficient collapses to p=0.844 — the classic signature of two collinear
predictors fighting over the same variance rather than the neighborhood term
contributing anything independent. Consistent with this, `enriched_score`
and `baseline_immune` remain strongly correlated on this N=32 subcohort
(Pearson r=-0.829) — the same confound-collinearity property documented in
§10's main analysis (there reported as r≈+0.998 against the reciprocal
baseline; the sign flips here only because this check correlates against
the baseline directly rather than its reciprocal, but the magnitude and
qualitative conclusion — the "neighborhood" statistic is largely a
restatement of bulk composition, not independent information — is the
same).

**Verdict: clean, honest null, consistent with everything else found for
this dataset.** None of recurrence, overall survival, grade, stage, or age
shows evidence that Cellohood's properly-selected K=13 neighborhood
clustering carries predictive information beyond this cohort's dominant
bulk-composition confound. The one baseline-alone result worth flagging on
its own (not as a Cellohood finding) is Recurrence: bulk immune fraction
alone reaches concordance=0.690 (p=0.060, borderline at N=32/13 events) —
mirroring, yet again, how much of this specific cohort's signal, for
essentially every outcome we have tried, is carried by raw cell-type
composition with no spatial information at all. Caveats: N=32 with 12-13
events per survival endpoint is a small-sample regime (penalized Cox used
for stability, but confidence intervals are wide); grade is unusable in
practice due to 28:3 imbalance; the exact meaning of the 2-digit `stage`
field could not be confirmed against the original paper's coding scheme
(Cell's own Table S2 was not recovered in machine-readable form — see below)
so the stage result is reported with that explicit caveat rather than as a
confirmed TNM/AJCC-stage finding.

**What we tried for the paper's own Table S2 and could not get.** Cell.com's
supplementary-materials page for the paper (`cell.com/fulltext/S0092-8674(18)31100-0`)
returned HTTP 403 to automated fetches; PMC/ScienceDirect mirrors were
checked but did not expose the table as a downloadable machine-readable
file either. The Angelo Lab's own data page (`angelolab.com/mibi-data`)
hosts two Dropbox links ("processed data" and "expression values, segmented
images, patient data") but Dropbox's folder-listing page is JS-rendered and
not scrapable by the tools available here; we did not attempt to
authenticate/download the raw Dropbox zip given the RASP-MIBI table already
gave us machine-readable, ID-verified clinical data covering the same
variables (recurrence, survival, grade, age, stage) for the same cohort.
If a verbatim copy of the original Table S2 is later obtained, it would be
worth re-running this same analysis against it as a consistency check,
but given the independent-reanalysis table already reproduces the
mixed/compartmentalized label perfectly, we are confident it is not a
different/incompatible patient cohort.

### 10.2 PLDA resolution sweep — is the null result a resolution-choice artifact?

Section 10's finding used only the standard (75%-variance, n_comp=2, K=13)
resolution. We swept all three resolution levels Cellohood's own
multi-resolution framework offers — coarse (PLDA50: 50% variance, n_comp=1),
standard (PLDA75), and full (PLDA100: 100% variance, n_comp=64, i.e. no PCA
reduction at all) — using the cluster counts `get_latent_directions`
recommends at each level (K=11, 13, 13 respectively, smallest stable
candidate each time), and reran the full confound/immune-subtype/clinical
protocol at each.

| | PLDA50 (coarse) | PLDA75 (standard) | PLDA100 (full) |
|---|---|---|---|
| Raw mixing score (best-dir AUC) | 0.626 | 0.670–0.674 | 0.637 |
| Enriched score vs. confound correlation | r=0.999 | r=0.998 | r=0.998 |
| Treg / CD8T / DC / CD11c-high | baseline wins all 4 | baseline wins all 4 | baseline wins all 4 |
| Recurrence (Cox LR test) | p=0.823 | p=0.845 | p=0.802 |
| Overall survival (Cox LR test) | p=0.612 | p=0.645 | p=0.619 |

**Nothing moves meaningfully across resolutions.** The confound-collinearity,
the immune-subtype pattern, and the clinical-variable null all replicate at
every scale Cellohood's multi-resolution framework offers. This rules out
"wrong resolution" as an explanation for the earlier null findings — it is
not a resolution-selection artifact, it is a property of this specific
cohort/task that holds uniformly from coarse to full resolution.

## 11. Runtime/scalability vs. bag size (Reviewer #4 concern 3, Reviewer #2 concern 8)

Extracted `epoch_time_sec` (steady-state, first 10 epochs dropped to exclude
graph-tracing/warmup) from every cached `model_history.df` across the full
threshold sweep (thresholds 2–10, 10 runs each, both architectures) and
normalized by training-set size (`training_params.json`'s `samples` field) to
get **per-bag training time**, since raw per-epoch time is confounded by the
training set shrinking as threshold grows (only 3 samples are used for
training in this protocol, so fewer, larger bags form at higher thresholds).

Fitted power-law scaling exponent (`time ~ bag_size^k`, bag size = mean
`max_neighborhood_size`, range 4–20 across thresholds 2–10):

| Architecture | Fitted exponent k |
|---|---|
| Transformer (self-attention) | 0.40 |
| DeepSet (pooling) | 0.48 |

**Finding: contrary to the naive theoretical expectation (self-attention is
O(n²), pooling is O(n)), both architectures show near-identical, sub-linear
empirical scaling at the bag sizes Cellohood actually operates at (≤20
cells).** The Transformer-vs-DeepSet per-bag time ratio ranges 0.78–1.24
across the swept bag sizes with no systematic widening gap as bags grow
larger — fixed per-batch overhead (data loading, kernel launch, small-batch
GPU underutilization) dominates wall-clock cost in this regime, not the
theoretical attention-complexity difference. **Conclusion: Reviewer #4's
quadratic-scaling concern, while theoretically correct, is not a practical
scalability bottleneck at the bag sizes (mean ≈5, max ≤20) Cellohood targets
— runtime cost is effectively architecture-agnostic here.** This would need
re-checking at much larger bag sizes (e.g. hundreds of cells) if Cellohood
were ever applied at a much coarser resolution than anything in the current
manuscript.

Figure: `dlpfc_runtime_scaling_figure.png`

## 12. Permutation-matching module ablation (Reviewer #4 concern 2)

Unlike local smoothing, this module is not a separable post-hoc step — it's
baked into the training loss itself. In `WinterCellEnvironmentAEV2.call()`
(`cellohood/models/autoencoders.py`, shared identically by both the
Transformer and DeepSet arms already tested — only the encoder differs
between them), a learned per-cell `ranking` (via a `Dense(1)` layer) feeds a
differentiable `soft_sorting_matrix` that permutes per-slot positional
embeddings so the decoder's reconstruction target for slot i is tied to a
*content-derived* rank rather than the cell's arbitrary position in the
input array (bags are built by grouping rows in whatever order they appear
in the underlying dataframe — confirmed no implicit distance-based or other
meaningful sort is applied at bag-construction time, so "arbitrary order" is
a fair characterization).

**Ablation design:** added a `use_permutation_matching` flag to
`WinterCellEnvironmentAEV2` (default `True`, fully backward-compatible).
When `False`, `sort_matrix` is replaced by the identity matrix — the
ranking layer and soft-sort computation are skipped entirely, and each
decoder slot reconstructs whatever cell arrived at that array index, with
no learned reordering. Everything else (encoder, bag-mean pooling, decoder,
loss normalization, entropy metric) is untouched. Wired in as
`model_type='NoPermMatch'` in `cellohood.train()`, exactly parallel to the
existing `'DeepSet'` option, using the same Transformer encoder as the
`'Cellohood'` baseline — so this isolates the permutation-matching question
cleanly from the architecture-choice question already answered by the
DeepSet comparison. (Also required adding the DeepSet-style save/load
dispatch entries in `saving_utils.py` for the new class, following the
pattern the previous bug fix established.)

Ran at threshold=5.0 (the manuscript's chosen threshold), trained-only-on-3-
CellCharter-samples protocol, 5 runs:

| Condition | Mean ARI | Std |
|---|---|---|
| With permutation matching (Transformer baseline) | 0.464 | — |
| **Without permutation matching (identity, 5 runs)** | **0.471** | **0.037** |

**Finding: no measurable difference (+0.007, well within noise) — in sharp
contrast to the decisive smoothing ablation.** Sanity check: the `ent`
(soft-sort entropy) training metric was ≈0.001 throughout for the
no-permutation-matching runs, as expected for an identity assignment,
confirming the module was genuinely disabled rather than silently still
active. Honest caveats: this is a single threshold with 5 runs, not the
full 2–10 sweep the other ablations received, and only tests the "identity
vs. learned soft-sort" contrast — it does not rule out that permutation
handling matters more at other bag sizes or that some *other* form of
permutation-awareness (not tested here) would show a larger effect. As
reported, though, this is a real, defensible result: at threshold=5,
removing the learned permutation-matching mechanism costs nothing.

Figure: none produced for this section (single-threshold bar comparison
would add little beyond the table above).

## 13. Cluster stability (Reviewer #2 concern 10)

Tested cluster-label stability on DLPFC (threshold=5.0) from four angles,
all using `adjusted_rand_score`/`normalized_mutual_info_score` on the
**discovered cluster labels directly** (not ARI-vs-ground-truth, which the
rest of this report already covers) — i.e. "if we ran this again, would we
find the same neighborhoods?" rather than "would they still match the
cortical layers?"

| Angle | What's compared | Mean ARI | Mean NMI |
|---|---|---|---|
| GMM seed (same embedding, 5 seeds) | 10 pairwise comparisons | **0.766** | 0.845 |
| 80% subsample (same config) | 5 trials vs. full-data labels | **0.908** | 0.931 |
| Hyperparameter config (5 diverse configs) | 10 pairwise comparisons | **0.448** | 0.596 |
| Resolution (coarse 50%-var vs. standard 75%-var) | purity + NMI | n/a (different k) | 0.549 (mean purity 0.569) |

Figure: `cluster_stability_figure.png`.

**Findings, by angle:**

- **GMM initialization alone is stable** (ARI 0.77, NMI 0.85 across 5 seeds)
  — re-running the same model's clustering doesn't meaningfully change the
  discovered partition.
- **Subsampling is even more stable** (ARI 0.91, NMI 0.93 across 5 trials of
  80% subsamples vs. the full-data clustering) — the neighborhoods aren't
  artifacts of specific cells happening to be present.
- **Hyperparameter choice is a real source of instability in cluster
  *identity*, even though it barely affects ARI-vs-ground-truth.** Five
  architecture configs that all score within the already-reported 0.41–0.48
  CC3 range agree with each other on actual cluster assignment at only
  ARI≈0.45, NMI≈0.60 — meaningfully below the seed/subsample numbers. In
  plain terms: different "equally good" models are finding partially
  different neighborhoods that happen to score similarly against the
  cortical-layer ground truth, not the same neighborhoods. This is an
  honest limitation worth stating plainly rather than only reporting the
  (reassuring) ARI-vs-ground-truth robustness from §2.
- **The coarse→standard PLDA resolution relationship is not a clean nested
  hierarchy.** If coarse clusters simply split into standard sub-clusters
  as resolution increases, each coarse cluster's cells would concentrate in
  one dominant standard cluster (purity near 1). Observed mean purity is
  0.569 (range 0.44–0.82 across the 4 coarse clusters) — real, non-trivial
  cross-cutting between resolution levels, not a clean refinement. Finer
  resolution still wins decisively on ARI-vs-ground-truth (§6), but the
  multi-resolution PLDA framework should not be described as producing a
  strictly nested set of partitions.

**Overall honest verdict:** Cellohood's clustering is robust to the things
that are "supposed" to be irrelevant (random seed, which specific cells are
present) but *not* fully robust to hyperparameter choice at the level of
actual cluster identity, and resolution levels relate to each other more
loosely than a clean hierarchy. None of this contradicts the ARI-vs-
ground-truth robustness already reported — a method can reliably recover
*layer-like* structure (§1, §2) while still disagreeing with itself on the
*exact* partition when the architecture changes. Both things are true and
worth stating together, not separately.

**Caveats:** single dataset (DLPFC) and single threshold (5.0) — the other
three ablation-swept thresholds were not re-run through this stability
protocol given time constraints; `ncomp_standard` for the PCA step came out
to 2 at this specific run/seed (on the low end of the 2–3 range reported
elsewhere for this threshold), which may affect the resolution-stability
purity numbers somewhat; the 5 hyperparameter configs used for Angle 3 were
a diverse but non-exhaustive subset of the full 22-config sweep.

## Open items (not yet done)

- **Threshold=9 collapse** — root cause still unknown.
- Visium HD/Xenium platform coverage (Reviewer #2 concern 1, literal) — no
  data available.
- PLDA.75/.100 benchmarked against competing methods at high resolution
  (Reviewer #2 concern 2) — substantial new experiments, not started.
- Cluster-stability extension to other thresholds/datasets beyond this
  single DLPFC/threshold=5.0 analysis (Reviewer #2 concern 10) — the core
  stability question is now answered (§13); broader coverage would
  strengthen it further but is not essential.

## Appendix A — bugs found and fixed during this campaign

1. `run_prediction_on_cello_df` silently truncated cells when a bag exceeded
   `max_neighborhood_size` without truncating the matching per-cell metadata,
   causing crashes (thresholds 7, 9) or, more seriously, *silent*
   misalignment whenever lengths coincidentally matched. Verified this did
   NOT affect thresholds {2,3,4,5,6,8,10} (training-derived
   max_neighborhood_size exactly equals the full-cohort max bag size at each
   of those thresholds — no truncation ever occurred). Fixed with a new
   `get_retained_cells_df_from_df()` helper plus a loud warning.
2. Train/test bag-construction copy-paste bug in the "split" ablation
   scripts (test block built from `train_data_df`). Fixed. Confirmed this
   never affected reported ARI numbers (those index the full `bag_cello_df`
   by sample id directly) — only made the training-time `val_loss` printout
   meaningless.
3. `saving_utils.py` model-loading dispatch pointed the DeepSet loader at the
   Transformer class, breaking reloads of cached DeepSet checkpoints. Fixed.
