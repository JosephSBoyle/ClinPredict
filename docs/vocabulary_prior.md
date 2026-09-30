# The vocabulary prior: a synthesis of studies 1 to 3

**Claim.** A diagnosis code's place in ICD-10-CM predicts its 30-day
readmission risk before any of the code's own admissions are seen. The
prediction is calibrated, and it is worth 20 to 30 admissions of a primary
diagnosis's own data (70 to 80 when every code on an admission is scored). In
a full multivariate model the cheapest way to use it is prefix features, which
are worth two to three times the training data.

Published page (interactive code explorer, all figures):
https://claude.ai/artifact/3QxAUwquXE4LYz1gQzPjt9 (private until shared).
Code for the new analyses: `hierprior/vocab.py`, on top of study 3's fits
(`hierprior/onecode.py`, `hierprior/onecode_curve.py`) and study 2's runs.
Outputs: `output/vocab/`. Studies: `docs/results.md`, `docs/results2.md`,
`docs/results3.md`.

## Metrics

* **AUROC** of the predicted readmission probability.
* **Log-loss reduction**: 1 − LL_model / LL_base, where LL is the mean binary
  cross-entropy of the predicted probabilities (natural logarithm) and the
  base model predicts the training base rate for everyone. This is McFadden's
  R², computed on held-out data. Negative values are worse than the base rate.
* **Break-even** (the exchange rate): the number of a code's own admissions m
  at which the generic-prior learner's pooled held-out log-loss equals the
  vocabulary prior's with m = 0, interpolated on log(1 + m).
* **Pseudo-admissions** of a Gaussian prior with variance V on a log-odds:
  n(V) = 1 / (V p0 (1 − p0)). One admission carries Fisher information
  p0 (1 − p0) at base rate p0, so n(V) is the number of admissions that would
  give the same precision. With p0 = 19.2%, a prior sd of 0.4 is worth about 40.

## One model

A code c is a string; p ⊑ c ranges over its prefixes (E, E1, E11, E116, E1165).

    logit P(readmit | c) = b0 + w_c,    w_c = Σ_{p ⊑ c} θ_p,    θ_p ~ N(0, τ²_|p|)

Equivalently, w is a Gaussian process on code strings with covariance

    Cov(w_c, w_c') = Σ_{k ≤ ℓ(c, c')} τ²_k,    ℓ = number of leading characters shared.

The **vocabulary prior** of a code is π_c = p(w_c | data on every code except
c) = N(m_c, P_c), computed exactly for all codes by one upward and one
downward pass of Gaussian message passing on the prefix tree (study 3).

The three studies are this sum under different constraints:

| study | form of the prefix sum | result |
|---|---|---|
| 1 | codes-only LR (proper-prefix terms fixed at 0); hierarchical prior with a support term; one L2 penalty over codes and all prefixes ("ancestor indicators") | prefix features best, +0.005 to +0.033 AUROC |
| 2 | one variance per prefix group by empirical Bayes, centred on the parent or on 0 | no reliable gain over one penalty, −0.006 to +0.004 |
| 3 | one code per prediction, so the leave-code-out prior can be scored directly | the prior is right (below) |

The model makes three predictions, each testable on held-out data.

## Prediction 1: shared characters, shared risk

Model-free correlogram: for every pair of primary-diagnosis codes with at least
30 training admissions (969 codes), the precision-weighted mean of the product
of their centred log-odds shifts, grouped by the number of leading characters
the pair shares, over the noise-corrected variance of a code's shift. With one
primary diagnosis per admission, different codes are estimated from disjoint
admissions, so the noise is independent between codes. 95% intervals from a
delete-one-first-letter jackknife.

![correlogram](../output/vocab/correlogram.png)

| shared leading characters | code pairs | model-free correlation (95%) | fitted model |
|---|---|---|---|
| 0 (different first letter) | 429,029 | −0.03 (−0.04 to −0.01) | 0.00 |
| 1 | 31,944 | 0.22 (0.01 to 0.43) | 0.22 |
| 2 | 6,039 | 0.34 (0.10 to 0.58) | 0.50 |
| 3 (same category) | 1,507 | 0.55 (0.34 to 0.76) | 0.65 |
| 4 | 368 | 0.65 (0.25 to 1.06) | 0.85 |
| 5 | 91 | 0.66 (0.47 to 0.84) | 0.89 |

The correlation rises with every shared character. A code's category alone
accounts for about half of the between-code variance in readmission log-odds,
and its closest siblings for about two thirds. The fitted model agrees at the
first levels and overstates close-sibling similarity (0.85 against 0.65): its
code-specific variance share is 22% for primary diagnoses and 23% for every
code, against roughly a third model-free.

## Prediction 2: honest uncertainty

For codes with at least 30 training admissions, the standardised error
z = (ŵ_c − m_c) / √(P_c + v_c), with ŵ_c the code's own training estimate and
v_c its sampling variance, should be N(0, 1).

![calibration](../output/vocab/calibration.png)

| | primary (960 codes) | every code (3,549) |
|---|---|---|
| inside the 50% interval | 58% | 56% |
| inside the 80% interval | 84% | 84% |
| inside the 95% interval | 95% | 95% |
| mean z² (1 is exact) | 0.92 | 0.96 |
| slope of own estimate on prior (n-weighted) | 1.15 | 0.98 |
| correlation, prior with own estimate | 0.66 | 0.66 |

The intervals hold their nominal coverage; the middle one is slightly wide. The
correlation is 0.76 for primary codes with at least 100 admissions, whose own
estimates are less noisy (study 3's scatter: `output/hier3/scatter.png`). The
stated prior variance matches the measured error of its mean (0.157 against
0.154 for the 328 curve codes).

## Prediction 3: the exchange rate

Learning curves of study 3 (328 primary codes and 1,784 every-code codes with
at least 100 training admissions; the learner gets m of a code's admissions,
30 random draws per m). The dev curves use the same draws as
`hierprior.onecode_curve`; the test curves use fresh draws. Break-even
intervals: 2,000 bootstrap resamples of codes.

![exchange rate](../output/vocab/exchange.png)

Predicted from the prior's measured error alone (the squared error of m_c
against the codes' own estimates, noise subtracted), the generic learner's
squared bias plus variance under N(0, s²), and nothing from the curves:

| | primary diagnosis | every code |
|---|---|---|
| vocabulary prior: measured error variance of m_c | 0.154 | 0.049 |
| worth, n(error variance) | 42 admissions | 115 |
| generic prior N(0, s²): s² | 0.353 | 0.233 |
| worth, n(s²) | 18 | 24 |
| **predicted break-even** | **24** | **79** |
| measured, dev (95%) | 21 (13–33) | 71 (51–93) |
| measured, test (95%) | 29 (19–48) | 75 (56–95) |

For primary diagnoses the break-even is close to the plain difference of
pseudo-admissions, 42 − 18. The prediction lies inside both measured 95%
intervals, for primary diagnoses and for every code. The price rises with the relatives' data: with
relatives trained on 5k, 17k, 52k and 174k admissions the prior is worth 9,
14, 20 and 21 primary-diagnosis admissions on dev (17, 34, 57 and 71 in
every-code mode).

## Every data size

Study 3's scaling table (`docs/results3.md`) holds on test (seed-0 subsets,
AUROC / log-loss reduction, primary diagnoses):

| training adm | vocabulary prior only | own data + generic prior | own data + vocabulary prior |
|---|---|---|---|
| 495 | 0.515 / −0.4% | 0.517 / +0.2% | 0.537 / −0.3% |
| 1,705 | 0.561 / +0.3% | 0.537 / +0.3% | 0.556 / +0.5% |
| 4,906 | 0.595 / +1.8% | 0.569 / +1.2% | 0.604 / +2.2% |
| 17,094 | 0.611 / +2.4% | 0.601 / +2.2% | 0.625 / +3.2% |
| 51,969 | 0.620 / +2.8% | 0.627 / +3.3% | 0.641 / +4.1% |
| 173,919 | 0.622 / +3.0% | 0.645 / +4.4% | 0.651 / +4.7% |

With 5,230 training admissions (mean of 3 subsets), 65% of dev patients have a
primary diagnosis seen fewer than 10 times in training and 20% one never seen;
at full data 10% are still below 10.

## Full multivariate models (studies 1 and 2)

Study 2, random setting, test AUROC (mean of 3 subsets; 1 at full data):

| training adm | codes only | codes + prefixes, one L2 penalty | learned group priors | codes-only needs |
|---|---|---|---|---|
| 514 | 0.565 | 0.594 | 0.592 | 1,110 (×2.2) |
| 1,690 | 0.610 | 0.633 | 0.626 | 3,685 (×2.2) |
| 5,230 | 0.643 | 0.660 | 0.657 | 14,147 (×2.7) |
| 17,406 | 0.663 | 0.675 | 0.678 | 34,475 (×2.0) |
| 52,348 | 0.682 | 0.689 | 0.691 | 108,376 (×2.1) |
| 173,919 | 0.694 | 0.699 | 0.700 | beyond the data |

![multivariate](../output/vocab/multivariate.png)

"Codes-only needs" is the training size at which the codes-only LR reaches the
prefix model's AUROC (interpolated in log N). Log-loss reduction at full data:
7.8% codes only, 8.2% with prefixes. Learned per-group variances add nothing
reliable, as expected when the variance decomposition is smooth across groups.
The vocabulary predicts a code's marginal effect well (r = 0.6 to 0.8, study
1) and its conditional effect in the multivariate model poorly (r ≈ 0.2).

## Where it fails

The prior assumes codes under a prefix are exchangeable. It misses when the
last characters carry acuity, severity, stage or setting, which is what drives
readmission: I50.21/.22/.23 acute, chronic and acute-on-chronic systolic heart
failure (acute-on-chronic 29% readmitted, prior 21%); K85.91 necrotising
pancreatitis (41%, prior 22%) beside K85.90 (22%); N18.6 end-stage renal
disease (42%, prior 20%); pregnancy against childbirth codes (O48.0, O99.824);
catch-all codes such as F32.9 (38%, prior 26%) and R07.89 (9%, prior 18%).
Drill-downs: `output/hier3/drilldown_primary.png`. The own data correct the
prior for common codes; the risk is for rare codes in such groups. The obvious
next prior adds the missing axis from description words ("acute", "with
necrosis", "end stage") or uses the official ICD-10-CM blocks.

## Test split, scored once

The test split (20% of patients) was scored once for this synthesis with
study 3's fitted models unchanged. Primary diagnoses, all 173,919 training
admissions, 95% patient-bootstrap intervals (500 resamples):

| | dev (24,410 adm) | test (48,101 adm) |
|---|---|---|
| AUROC, vocabulary prior only | 0.621 (0.612–0.632) | 0.622 (0.614–0.629) |
| AUROC, own data + generic prior | 0.644 (0.633–0.657) | 0.645 (0.638–0.654) |
| AUROC, own data + vocabulary prior | 0.653 (0.642–0.665) | 0.651 (0.644–0.660) |
| log-loss reduction, vocabulary prior only | 3.0% (2.5–3.4) | 3.0% (2.6–3.3) |
| log-loss reduction, own + generic prior | 4.4% (3.7–5.1) | 4.4% (3.9–4.9) |
| log-loss reduction, own + vocabulary prior | 4.8% (4.2–5.5) | 4.7% (4.2–5.2) |
| codes unseen in training: AUROC, prior only | 0.641 (0.563–0.719), 376 adm | 0.603 (0.548–0.655), 777 adm |
| break-even, own admissions (bootstrap over codes) | 21 (13–33) | 29 (19–48) |

## Exploratory: what a small hospital could have known

Codes seen at most twice in a 4,906-admission training subset (seed 0) but at
least 100 times in the full training split (48 primary codes): the prior from
the small subset is closer to the full-data log-odds than the base rate is
(noise-corrected RMSE 0.51 against 0.68) and than the codes' own small-sample
estimates (1.07), with correlation 0.63. Every-code mode (230 codes): 0.49
against 0.55 and 0.95, correlation 0.45.

## What is and is not shown

Supported: the prior predicts marginal readmission risk with no own data; it is
calibrated in scale and uncertainty; its worth in admissions matches its
measured error; prefix features carry the benefit into full models.

Not shown: transfer to other hospitals or years (one hospital system, one
random patient split; study 2's temporal split had almost no new codes); codes
new to ICD-10-CM itself; other outcomes and vocabularies; the official
chapter and block hierarchy (the two-character prefix level used here is not
an ICD-10-CM level).

## Reproduce

```
.venv/Scripts/python -m hierprior.onecode          # study 3 fits, ~7 min
.venv/Scripts/python -m hierprior.onecode_curve    # study 3 learning curves, ~1 min
.venv/Scripts/python -m hierprior.vocab            # this synthesis, ~6 min
.venv/Scripts/python -m hierprior.vocab --page     # figures and page data from saved results
```
