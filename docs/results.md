# Hierarchical priors for out-of-distribution codes: results

Task: 30-day readmission prediction on MIMIC-IV 3.1 from the ICD-10-CM diagnosis
codes of the index admission (`docs/idea.md`). Code: `hierprior/`. Raw
per-run results: `output/hier/runs/*.json`; tables: `output/hier/summary.md`.

**TL;DR**

* The hierarchical prior **as specified** (prior strength also inversely
  proportional to the code's own support, "beta = 1") is statistically
  indistinguishable from the plain L2 baseline on overall AUROC, and only
  slightly better (+0.000 to +0.003) on the admissions that carry held-out codes.
* The weights it **imputes** for never-seen codes are genuinely informative:
  zeroing them (which is what the baseline does) costs a small but very
  consistent 0.0015-0.005 AUROC on affected admissions in every setting
  except the literal protocol A. In the one-feature regime (no collinearity)
  they beat the prevalence prior and track the codes' true marginal effects
  (r = 0.6-0.8 against the effects estimated with the codes seen).
* Dropping the support term (**beta = 0**, a standard Gaussian hierarchy) is
  much better: +0.007 to +0.027 AUROC over the baseline in protocol B, largest
  at small N.
  Ablating beta in {0, 0.5, 1, 2} gives a monotone ordering: 0 best, 2 worst.
* A simple **ancestor-indicator** logistic regression (every code also switches
  on an indicator for each of its prefixes) is best or tied-best everywhere.
* The **literal hold-out protocol** (drop every training patient with one
  random sibling from every final-level group) removes 82% of training
  patients (90% of training admissions) and leaves a cohort with 6% readmission prevalence vs 20%
  elsewhere; every model is near chance (AUROC 0.50-0.57), so it cannot answer
  the question. Protocol B (hold out in 10% of groups) is the informative one.

![scaling](../output/hier/scaling_main.png)

With the exploratory models and protocol N added:

![scaling, all models](../output/hier/scaling_all_models.png)

## Setup

**Cohort** (`hierprior/data.py`). Adult (anchor_age >= 18) admissions whose
diagnoses are all ICD-10, excluding in-hospital deaths: 248,067 admissions,
119,740 patients, 19,284 distinct codes, 13.3 codes per admission.
Label: a later admission of the same patient starting <= 30 days after
discharge (any ICD version counts as the readmission); prevalence 19.9%.
Train / dev / test split by patient 70/10/20 (seed 0): 173,919 / 24,903 /
49,245 admissions. (This deliberately differs from PyHealth's
`ReadmissionPredictionMIMIC4`, which only keeps patients with >= 2
admissions and drops each patient's last admission, conditioning on the
future.)

**Hierarchy.** Character prefixes: root -> first letter -> 3-character
category -> 4 -> ... -> full code (14,026 internal nodes). A final-level group
is an internal node whose children are all leaves and that has >= 2 of them:
4,132 groups.

**Held-out codes and protocols** (`hierprior/experiment.py`). With seed 0 one
random sibling is chosen per final-level group. Every training patient with
any held-out code is dropped; dev and test are untouched.

| protocol | held out | train patients (admissions) dropped | train prevalence | affected test admissions |
|---|---|---|---|---|
| A (literal) | 4,132 codes, 1 per group | 82% (83,818 -> 14,910) (90%: 173,919 -> 17,199) | 6% | 39,583 / 49,245 (80%) |
| B | 427 codes, 1 in a random 10% of groups | 27% (-> 61,453) (42%: -> 100,227) | 13-15% | 9,323 (19%) |
| N (exploratory) | nothing; "OOD" = codes absent from the training subset | 0% | 20% | 3,253-39,397 |

Each protocol's remaining training patients are subsampled at several
fractions (log-spaced, 3 subsample seeds; one run at fraction 1).
"Affected" = dev/test admissions with >= 1 held-out (or, in N, unseen) code.

**Models** (`hierprior/models.py`), all fit by exact penalised maximum
likelihood (sum log-loss + penalty; unpenalised intercept):

* **Baseline LR**: one weight per code, L2 penalty. A held-out code has no
  training data, so its weight is 0: the model ignores it.
* **Hierarchical prior LR (spec)**: every tree node v has a weight theta_v;
  leaves are the codes' weights, internal nodes are latent means. Prior:
  `theta_v ~ N(theta_parent, sigma2_parent * (1 + n_v)^beta / alpha)`, i.e.
  precision `alpha / (sigma2_g (1 + n_v)^beta)` - inversely proportional to the
  variance sigma2_g of the related codes' Gaussian and, with beta = 1, to the
  code's own training support n_v. sigma2_g is estimated per parent by
  empirical Bayes (EM with a diagonal-Laplace posterior variance, shrunk to
  the pooled value for its depth); alpha is tuned on dev. A code with no
  support has no likelihood term, so its weight equals its parent's latent
  mean: the imputed weight. Fit in the non-centred parametrisation (leaf
  weight = sum of per-node deviations on its root path) with Newton-CG.
* **beta = 0 variant [exploratory]**: the same model without the support term
  (standard hierarchical Gaussian prior).
* **Ancestor-indicator LR [reference, exploratory]**: L2 LR on the code
  indicators plus a binary "any code under node v" indicator for every
  ancestor v. This is the common "roll-up" baseline; an unseen code still
  switches on its ancestors' indicators.

Hyperparameters (L2 strength / alpha, 9-point log grids) are chosen by dev
AUROC per run; test AUROC is reported. Mean +- sd over the 3 subsamples.

## Results

### Protocol B (main)

Test AUROC, all admissions / admissions with a held-out code:

| train adm | Baseline LR | Hier. prior (spec, beta=1) | Hier. prior beta=0 | Ancestor-indicator LR |
|---|---|---|---|---|
| 1,004 | 0.590±0.011 / 0.585±0.010 | 0.585±0.007 / 0.585±0.011 | 0.610±0.007 / 0.612±0.005 | 0.615±0.008 / 0.618±0.006 |
| 3,031 | 0.611±0.009 / 0.603±0.011 | 0.611±0.006 / 0.606±0.006 | 0.629±0.010 / 0.627±0.008 | 0.638±0.007 / 0.636±0.004 |
| 10,012 | 0.640±0.010 / 0.635±0.009 | 0.641±0.011 / 0.637±0.009 | 0.653±0.013 / 0.650±0.009 | 0.660±0.008 / 0.657±0.006 |
| 29,999 | 0.661±0.008 / 0.656±0.007 | 0.662±0.009 / 0.657±0.007 | 0.670±0.008 / 0.665±0.007 | 0.674±0.007 / 0.669±0.006 |
| 100,227 | 0.679 / 0.668 | 0.680 / 0.670 | 0.685 / 0.676 | 0.685 / 0.676 |

Paired differences vs the baseline (same subsample), affected admissions:
spec +0.000 / +0.003 / +0.002 / +0.001 / +0.002 (sd up to 0.02 at the smallest
size); beta=0 +0.027 / +0.024 / +0.015 / +0.009 / +0.008; ancestor-indicator
+0.033 / +0.033 / +0.022 / +0.013 / +0.008.

**Do the imputed weights help?** Holding everything else fixed, zeroing the
held-out codes' imputed weights in the fitted hierarchical model lowers
affected-admission test AUROC by:

| train adm | 1,004 | 3,031 | 10,012 | 29,999 | 100,227 |
|---|---|---|---|---|---|
| spec (beta=1) | 0.0038±0.0038 | 0.0053±0.0016 | 0.0022±0.0001 | 0.0026±0.0012 | 0.0033 |
| beta=0 | 0.0034±0.0017 | 0.0039±0.0008 | 0.0027±0.0007 | 0.0026±0.0004 | 0.0029 |

So the answer to "do we do better on AUC than a model that ignores the unseen
codes?" is yes, by a small, reliable margin. The larger gaps between model
families come from how the prior regularises the codes that *were* seen.

### Protocol A (literal)

| train adm | Baseline LR | spec (beta=1) | beta=0 | Ancestor-indicator |
|---|---|---|---|---|
| 519 | 0.523 / 0.515 | 0.506 / 0.500 | 0.502 / 0.496 | 0.516 / 0.509 |
| 1,726 | 0.527 / 0.519 | 0.519 / 0.513 | 0.509 / 0.503 | 0.517 / 0.510 |
| 5,176 | 0.533 / 0.525 | 0.527 / 0.522 | 0.520 / 0.515 | 0.533 / 0.527 |
| 17,199 | 0.552 / 0.540 | 0.547 / 0.538 | 0.552 / 0.544 | 0.569 / 0.561 |

(sd across subsamples 0.01-0.03.) With 82% of training patients (90% of admissions) gone, the
survivors are those with no code from 4,132 randomly chosen siblings - mostly
patients with few codes, 6% readmitted - so what is learned transfers poorly
to the full test population. Imputation gains are -0.003 to +0.003, i.e.
noise. The effect is the protocol, not the prior: the full-data baseline
reaches 0.695 dev AUROC on the same features.

### One-input-feature regime

Each held-out code gets its own model `logit P = b0 + w_c x_c` (no
collinearity: w_c is the code's marginal log-odds shift). The baseline has no
data, so w_c = 0 and it predicts the prevalence. The hierarchical models fit
the same tree prior to the codes' univariate estimates and impute w_c from
the relatives (`hierprior/univariate.py`). Metrics over held-out codes on
test: log-likelihood gain per code occurrence vs the prevalence prior;
AUROC over all (admission, held-out code) pairs scored by w_c (baseline 0.5);
occurrence-weighted sign accuracy for codes with >= 20 test occurrences
(w = 0 counts as wrong; always guessing "positive" scores 0.74 in B, 0.72 in A).
"Sibling mean" = plain mean of the siblings' own univariate estimates.

| protocol | train adm | model | LL gain / occ | pair AUROC | sign acc |
|---|---|---|---|---|---|
| B | 1,004 | sibling mean | +0.003 | 0.527 | 0.33 |
| B | 1,004 | spec (beta=1) | +0.004 | 0.549 | 0.69 |
| B | 1,004 | beta=0 | +0.005 | 0.559 | 0.72 |
| B | 10,012 | sibling mean | +0.004 | 0.553 | 0.44 |
| B | 10,012 | spec (beta=1) | +0.004 | 0.555 | 0.86 |
| B | 10,012 | beta=0 | +0.009 | 0.600 | 0.89 |
| B | 100,227 | sibling mean | +0.008 | 0.593 | 0.47 |
| B | 100,227 | spec (beta=1) | +0.012 | 0.603 | 0.94 |
| B | 100,227 | beta=0 | +0.012 | 0.605 | 0.88 |
| A | 17,199 | spec (beta=1) | -0.013 | 0.535 | 0.36 |
| A | 17,199 | beta=0 | -0.005 | 0.528 | 0.35 |

In B, imputed univariate weights beat the prevalence prior at every training
size, rank held-out codes' risk above chance, and (from ~10k admissions) get
the direction right for ~90% of occurrences; the tree prior beats the naive
sibling mean. In A the imputations are worse than the prevalence prior - the
biased training cohort teaches the wrong offsets. (295 held-out codes in B
and 2,620 in A occur in test; full table in `output/hier/summary.md`.)

![univariate](../output/hier/univariate.png)

## Exploratory analyses

All clearly exploratory: chosen after seeing the main results, not
pre-specified in `docs/idea.md`.

**E1. The support term (beta).** Protocol B, 3 subsamples each (`output/hier/runs_beta`):

| train adm | beta=0 | beta=0.5 | beta=1 (spec) | beta=2 |
|---|---|---|---|---|
| 3,031 | 0.629±0.010 | 0.619±0.007 | 0.611±0.006 | 0.600±0.007 |
| 10,012 | 0.653±0.013 | 0.647±0.013 | 0.641±0.011 | 0.633±0.011 |

(test AUROC, all admissions, mean±sd over 3 subsamples; the affected-admission
ordering is identical.) Making the prior weaker for well-supported codes hurts: with beta = 1
a code seen 1,000 times is almost unregularised, while the dev-optimal L2
strength says even common codes need shrinkage (collinearity among co-coded
diagnoses). Pooling rare codes *and* shrinking common ones (beta = 0) is
better. Tuned alpha grows accordingly: 300 for beta = 0.5, 1e3-3e3 for beta = 1, 1e5-1e6 for beta = 2.

**E2. Natural OOD (protocol N).** No hold-out; at small training sizes most
codes are simply unseen (17,603 of 19,284 at 500 admissions; 7,013 at 52k).
Test AUROC, all / affected:

| train adm | Baseline LR | spec (beta=1) | beta=0 | Ancestor-indicator |
|---|---|---|---|---|
| 514 | 0.565±0.007 / 0.557±0.006 | 0.572±0.010 / 0.567±0.013 | 0.586±0.010 / 0.581±0.014 | 0.594±0.007 / 0.589±0.011 |
| 1,690 | 0.610±0.008 / 0.609±0.010 | 0.608±0.010 / 0.607±0.012 | 0.625±0.014 / 0.625±0.015 | 0.633±0.014 / 0.633±0.015 |
| 5,229 | 0.643±0.005 / 0.631±0.008 | 0.639±0.009 / 0.632±0.007 | 0.654±0.007 / 0.646±0.008 | 0.660±0.006 / 0.653±0.006 |
| 17,406 | 0.663±0.002 / 0.656±0.011 | 0.665±0.002 / 0.662±0.010 | 0.671±0.002 / 0.668±0.009 | 0.675±0.001 / 0.671±0.007 |
| 52,347 | 0.682±0.001 / 0.669±0.010 | 0.683±0.001 / 0.672±0.008 | 0.688±0.001 / 0.678±0.009 | 0.689±0.001 / 0.679±0.009 |

(The full-data point, 174k admissions, is missing: that run was killed by
memory pressure after ~4 CPU-hours and not rerun.)

The same picture: the spec model is at most +0.01 over the baseline, and only
at the smallest size; beta = 0 and the ancestor indicators help most at small
N and the gap narrows with data. Imputed weights of unseen codes again add
+0.0015 to +0.004 AUROC on affected admissions.

**E3. Imputed vs "oracle" weights.** For the 109 protocol-B held-out codes with
>= 20 occurrences in the full training set, compare the weight imputed with
the code unseen (B, full retained data) to an "oracle" weight learned with
the code's patients kept (all training patients). Multivariate oracle: L2 LR
(lambda = 30, the dev-optimal value at >= 10k admissions throughout); the
intended oracle, the same hierarchical model on the full data, was the killed
run. One-feature oracle: the code's own univariate estimate.

| imputed by | oracle | Pearson r | support-weighted r | sign agreement | RMSE (imputed) | RMSE (baseline's 0) |
|---|---|---|---|---|---|---|
| hier. LR, spec | L2 LR | 0.19 | 0.42 | 0.55 | 0.109 | 0.120 |
| hier. LR, beta=0 | L2 LR | 0.22 | 0.41 | 0.55 | 0.111 | 0.120 |
| one-feature, spec | univariate | 0.60 | 0.77 | 0.71 | 0.341 | 0.540 |
| one-feature, beta=0 | univariate | 0.63 | 0.80 | 0.66 | 0.356 | 0.540 |
| one-feature, sibling mean | univariate | 0.38 | 0.56 | 0.59 | 0.455 | 0.540 |

This is the collinearity point in the idea doc made concrete. Marginal
(one-feature) effects of sibling codes are strongly shared, so the tree
recovers an unseen code's marginal effect well (r = 0.6-0.8, RMSE down a
third vs assuming 0). Conditional (multivariate) weights are small and
dominated by what co-occurring codes already explain, so the imputation is
only weakly correlated with the oracle (r ~ 0.2, 0.4 support-weighted) and
barely beats 0 in RMSE - consistent with the small (+0.002-0.005) AUROC gain
from imputation in the multivariate models. (Caveat: the L2 oracle is itself
shrunk and is a different model family from the imputer.)

![imputed vs oracle](../output/hier/explore_imputed_vs_oracle.png)

**E4. AUROC by number of held-out codes per test admission** (protocol B, full
data):

| held-out codes | test admissions | prevalence | Baseline LR | spec (beta=1) | beta=0 |
|---|---|---|---|---|---|
| 0 | 39,922 | 19.1% | 0.682 | 0.682 | 0.688 |
| 1 | 8,239 | 20.9% | 0.664 | 0.665 | 0.670 |
| 2+ | 1,084 | 24.4% | 0.692 | 0.701 | 0.707 |

The spec model's advantage over the baseline appears only where held-out
codes accumulate (+0.008 with 2+ codes, single draw, n = 1,084 so roughly
+-0.02 sampling noise); beta = 0's advantage is mostly general regularisation
(present even with 0 held-out codes).

## Caveats

* One held-out draw (seed 0 as specified) and one train/dev/test split; the
  +- are over training subsamples only.
* ICD-10-CM diagnoses only (no procedures / drugs), so absolute AUROC
  (~0.69 at full data) is modest; the comparison between priors is the point.
* The hierarchy is character prefixes, not the official ICD-10-CM tabular
  chapters/blocks; letters stand in for chapters.
* The empirical-Bayes variances use a diagonal Laplace approximation; alpha is
  then a single global multiplier tuned on dev.
* Protocol N's full-data point is missing (run killed by memory pressure).
* Protocol B deviates from the idea doc (10% of groups instead of all)
  because the literal protocol leaves an unrepresentative training cohort;
  both are reported.

## Reproduce

```
.venv/Scripts/python -m hierprior.experiment --jobs 10          # main grid (~hours; full-data runs dominate)
.venv/Scripts/python -m hierprior.experiment --protocols B --fracs 0.03,0.1 --models hier_b05,hier_b2 --sub runs_beta
.venv/Scripts/python -m hierprior.univariate
.venv/Scripts/python -m hierprior.analyze                       # figures + output/hier/summary.md
.venv/Scripts/python -m hierprior.explore
```
