# One-code mode: a prior for a code built from its relatives

Study 3. Task, cohort, label and patient split as in studies 1 and 2
(`docs/results.md`): 30-day readmission after an adult MIMIC-IV 3.1 admission.
**Chapter Z is excluded throughout** (factors influencing health status,
aftercare encounters, weeks of gestation, ...). Z codes are never scored and
never used as another code's relatives. Code: `hierprior/onecode.py` (main
grid), `hierprior/onecode_curve.py` (learning curves), `hierprior/analyze3.py`.
Full tables: `output/hier3/summary.md`. The test split was not used. A shorter
version restricted to one code per admission is `docs/results3_onecode.md`.

## Units

Every number below is one of three standard metrics, computed on the 10% dev
patients:

* **AUROC** of the predicted readmission probability.
* **Log-loss**, the binary cross-entropy: the average of -ln(probability the
  model gave to what actually happened), in **nats per prediction** (natural
  log; divide by ln 2 = 0.693 for bits). Lower is better.
  A model that predicts the base rate p0 for everyone scores the entropy of
  p0: **0.4893 nats** for primary diagnoses (p0 = 19.2%) and **0.5366 nats**
  for code occurrences (p0 = 22.7%, higher because sicker admissions carry
  more codes).
* **Log-loss reduction**: the percentage of that prevalence-only log-loss a
  model removes. For example, 4.8% means 0.4893 falls to 0.4658 nats.

For scale: one diagnosis on its own carries little information about
readmission. The best one-code model here reaches AUROC 0.65 and removes 5% of
the log-loss. The multivariate ancestor-indicator model of study 2, which
uses all codes of an admission, reaches AUROC 0.70.

## Setup

**One-code model.** Each prediction uses exactly one ICD-10 code c:

    P(readmit | c) = sigmoid(b0 + w_c),    b0 = logit(training base rate)

* **Primary mode** (the pure version): the code is the admission's primary
  diagnosis, so there is one code and one prediction per admission (169,740
  training and 24,410 dev admissions at full data).
* **Every-code mode**: every diagnosis on an admission is a separate
  prediction made from that code alone (1.9M training and 272k dev
  predictions). This tests the codes in general, not only primary diagnoses.

**How w_c is learned** (all from the training subset):

| name | the code's own data | prior | what it is |
|---|---|---|---|
| prevalence only | not used | - | w = 0 |
| naive prior only | not used | raw average of the siblings | inverse-variance-weighted mean of the siblings' own estimates, unshrunk |
| hierarchical prior only | not used | relatives | leave-c-out prediction from a Gaussian hierarchy over the ICD prefix tree (below) |
| own data only | used | none | unshrunk estimate, log((k+.5)/(n-k+.5)) - b0 |
| own data + naive prior | used | N(0, s²) | posterior with a zero-centred prior; s² fitted on all codes (one-feature ridge) |
| own data + hierarchical prior | used | relatives | posterior with the hierarchical prior |

**Hierarchical prior.** A code's weight is its parent group's mean plus a
code-level deviation. Each group's mean is its own parent's mean plus a
deviation (E1165 under E116, under E11, under E1, under E). The 11 variances
(one per tree depth) are fitted by marginal likelihood. Each code's binomial
likelihood enters by a Laplace approximation. Gaussian message passing then
gives, for every code, the distribution of its weight given all other codes'
data but not its own. So the prior leans on the siblings when they are well
supported and falls back towards cousins and the chapter when they are not.
It was checked against brute-force Gaussian conditioning on a small tree.

**Two amounts of data are varied.**
1. *Training admissions* (0.5k to 174k, random patient subsets, 3 seeds).
   This changes both the relatives' data and the code's own data.
2. *The code's own data alone.* For the codes with at least 100 training
   admissions (1,784 codes in every-code mode, 328 in primary mode), the
   learner gets exactly m = 0, 1, 2, ..., 100 of the code's admissions
   (random draws, 30 repeats), while the hierarchical prior comes from
   relatives trained on 5k to 174k admissions.

## Results

![scaling](../output/hier3/scaling.png)

### Primary mode (one code per admission), dev

| training adm | naive prior only | hier. prior only | own data only | own + naive prior | own + hier. prior |
|---|---|---|---|---|---|
| 514 | 0.541 / -14.6% | 0.539 / +0.2% | 0.519 / -7.3% | 0.521 / -0.0% | 0.548 / -0.2% |
| 1,690 | 0.541 / -12.2% | 0.575 / +0.9% | 0.537 / -7.0% | 0.543 / +0.6% | 0.577 / +1.1% |
| 5,230 | 0.541 / -10.4% | 0.592 / +1.6% | 0.570 / -5.8% | 0.575 / +1.2% | **0.607 / +2.1%** |
| 17,406 | 0.565 / -6.0% | 0.611 / +2.4% | 0.601 / -2.3% | 0.608 / +2.5% | **0.631 / +3.4%** |
| 52,348 | 0.576 / -3.1% | 0.618 / +2.7% | 0.620 / +0.6% | 0.627 / +3.4% | **0.643 / +4.1%** |
| 173,919 | 0.588 / -0.5% | 0.621 / +3.0% | 0.639 / +3.0% | 0.644 / +4.4% | **0.653 / +4.8%** |

Each cell is AUROC / log-loss reduction vs prevalence only (0.4893-0.4899
nats), mean over 3 subsamples. Negative reductions mean the model is worse
than predicting the base rate.

* **The hierarchical prior with the code's own data is best at every size
  from 1.7k admissions up.** Over the naive-prior posterior it adds +0.009 to
  +0.034 AUROC and 0.4 to 0.9 percentage points of log-loss reduction. The
  gap is largest between 2k and 17k admissions.
* **The hierarchical prior alone, with no data on the code, beats learning the
  code with a naive prior up to 5k admissions.** It stays level up to 17k
  admissions. At full data it gives two thirds of the own-data log-loss
  reduction (3.0% of 4.4%) and AUROC 0.621 vs 0.644.
* **The naive sibling average is worse than the base rate at every size.**
  Its ranking is above chance (AUROC 0.54-0.59), but it is overconfident. One
  or two siblings are a noisy guess, and it is not shrunk.
* **Own data with no prior** is also worse than the base rate until about
  40k admissions, because rare codes get extreme estimates.

### Every-code mode, dev

| training adm | naive prior only | hier. prior only | own data only | own + naive prior | own + hier. prior |
|---|---|---|---|---|---|
| 514 | 0.508 / -7.9% | 0.522 / -0.5% | 0.514 / -6.9% | 0.515 / -0.4% | 0.528 / -0.7% |
| 1,690 | 0.519 / -6.7% | 0.539 / +0.2% | 0.533 / -3.8% | 0.533 / +0.2% | 0.550 / +0.4% |
| 5,230 | 0.524 / -4.3% | 0.545 / +0.4% | 0.547 / -1.9% | 0.549 / +0.5% | **0.561 / +0.8%** |
| 17,406 | 0.530 / -2.9% | 0.552 / +0.7% | 0.559 / -0.4% | 0.561 / +0.9% | **0.568 / +1.1%** |
| 52,348 | 0.537 / -1.9% | 0.557 / +0.8% | 0.569 / +0.6% | 0.571 / +1.2% | **0.575 / +1.4%** |
| 173,919 | 0.543 / -1.3% | 0.558 / +0.8% | 0.577 / +1.2% | 0.578 / +1.5% | **0.580 / +1.6%** |

Prevalence-only log-loss 0.5366-0.5376 nats per prediction. The ordering is
the same as in primary mode. The gains are a third as large, because most
secondary codes carry little signal.

### How much is the hierarchical prior worth, in admissions of the code's own data?

![learning curves](../output/hier3/curve.png)

For the codes with at least 100 training admissions, the learner gets m of
the code's own admissions:

| primary mode, relatives from all 174k admissions | m = 0 | 1 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| AUROC, own + hierarchical prior | **0.618** | 0.619 | 0.624 | 0.631 | 0.636 | 0.645 | 0.650 |
| AUROC, own + naive prior (= own only) | 0.500 | 0.523 | 0.569 | 0.599 | 0.618 | 0.640 | 0.649 |
| log-loss, own + hierarchical prior | **0.4845** | 0.4843 | 0.4832 | 0.4816 | 0.4799 | 0.4774 | 0.4756 |
| log-loss, own + naive prior | 0.4990 | 0.4980 | 0.4938 | 0.4891 | 0.4848 | 0.4792 | 0.4761 |
| log-loss, own data only | 0.4990 | 0.6199 | 0.5378 | 0.5120 | 0.4951 | 0.4812 | 0.4767 |

(The two naive learners rank codes identically, so their AUROCs are equal.
They differ in calibration.)

**Break-even.** A learner with the naive prior needs this many of the code's
own admissions to match the hierarchical prior with none:

| relatives trained on | ~5k adm | ~17k | ~52k | 174k |
|---|---|---|---|---|
| primary mode (log-loss / AUROC) | 9 / 9 | 14 / 17 | 20 / 24 | **21 / 20** |
| every-code mode (log-loss / AUROC) | 17 / 18 | 34 / 26 | 57 / 46 | **71 / 51** |

So in primary mode, knowing a code's relatives is worth about 20 admissions
with that code. It is worth more in every-code mode (50-70), where single
codes are weak and noisy on their own. The advantage shrinks as the code's
own data grow and is almost gone at 100 admissions. The hierarchical
posterior is never worse.

### By the code's own amount of training data (full data, dev)

| training adm with the code | primary: AUROC hier. prior / own + naive / own + hier. | primary: log-loss red. | every-code: log-loss red. |
|---|---|---|---|
| 0 (unseen) | 0.641 / 0.500 / 0.641 | +3.0% / 0 / +3.0% | +2.3% / 0 / +2.3% |
| 1-9 | 0.632 / 0.580 / 0.642 | +3.5% / +1.6% / +4.0% | +2.1% / +0.9% / +2.5% |
| 10-99 | 0.623 / 0.633 / 0.647 | +2.9% / +3.7% / +4.4% | +2.0% / +2.2% / +2.8% |
| 100-999 | 0.628 / 0.663 / 0.663 | +3.4% / +5.6% / +5.6% | +1.8% / +2.8% / +2.8% |
| >= 1000 | 0.592 / 0.640 / 0.639 | +1.4% / +4.0% / +4.0% | +0.4% / +1.0% / +1.0% |

The prior alone beats the code's own (naive-prior) data for codes with fewer
than 10 admissions. From about 100 admissions the prior adds nothing.

### Is the prior calibrated?

For well-supported codes, regress each code's own estimate on its
hierarchical prior. The slope is 0.91 (every code, >= 200 admissions) and
1.06 (primary, >= 100), and the correlation is 0.77 and 0.75. The prior
explains 52-54% of the squared weights (uncentred R²).

![prior vs own](../output/hier3/scatter.png)

## Drill-downs

For each code: its own training estimate, its hierarchical prior (built
without the code's data), and the independent dev estimate, each ±95%.
Weights are log-odds shifts from the base rate. The group's log-loss
reduction on its dev predictions is given for the prior alone and for the
code's own data (with the naive prior). The published page has interactive
versions at ~17k and 174k training admissions.

**Primary mode** (dev groups are small, 86-390 admissions, so group numbers
are noisy):

![drill-down primary](../output/hier3/drilldown_primary.png)

* **F32.x depression.** The prior removes 6.0% of the log-loss vs 7.2% for
  own data. The rare subtypes (F32.1-F32.3, F32.A, 7-141 admissions) all get
  about +0.5 from their relatives. The common unspecified code F32.9 is +0.95
  by itself, and its prior of +0.38 undershoots.
* **I61.x intracerebral haemorrhage.** Every location code is rarely
  readmitted (-0.5 logits; the patients die or go to rehabilitation). The
  prior (7.5%) beats own data (6.6%). At 17k admissions, when each location
  had 1-17 cases, it was 7.8% vs 1.7%.
* **C78.x secondary cancers.** All about +0.5. The prior gets 8.0% vs 10.2%
  for own data, and at 17k it beats own data (11.1% vs 10.4%).
* **K57.3x diverticular disease.** All mildly protective. The prior gets
  4.1% vs 5.3%.
* **I50.2x systolic heart failure: fails.** Acute-on-chronic (I50.23, 471
  admissions) is +0.52. The others are around 0 in training and mostly
  negative on dev. The prior averages them to +0.1-0.3, which gives 0.8% vs 9.1%.
* **K85.9x acute pancreatitis: fails.** Necrotising pancreatitis (K85.91-2)
  is +0.5 to +0.9, while plain pancreatitis is +0.15. The prior gives 0.3% vs 5.7%.

**Every-code mode:**

![drill-down every code](../output/hier3/drilldown_any.png)

* **E87.x electrolyte disorders** and **C78.x secondary cancers** are
  homogeneous groups. The prior is as good as the code's own data (0.3% vs
  0.4%, and 4.1% vs 3.7%).
* **D63.x anaemia in chronic disease** partly works (2.7% vs 4.3%). Anaemia in
  cancer (D63.0) is +0.89, and its prior (+0.27) is pulled down by the CKD and
  other-disease siblings.
* **N18.x chronic kidney disease fails** (0.0% vs 0.8%). End-stage renal
  disease (+0.40) is the extreme of an ordered series, and its prior (+0.01)
  averages the milder stages.
* **K76.x liver disease fails** (-0.2% vs 4.1%). Fatty liver is -0.18, but
  its severe siblings (portal hypertension, hepatorenal syndrome) give it a
  prior of +0.46.
* **I50.2x systolic heart failure fails** (-0.3% vs 0.2%).

**Largest misses** (every code, >= 200 admissions, ranked by n x (own -
prior)²): I10 hypertension, Y92.9 unspecified place, O99.824 and O99.344
(childbirth vs pregnancy), D64.9 anaemia unspecified, G89.3 cancer pain,
R45.851 suicidal ideation, D62 posthaemorrhagic anaemia, R55 syncope, I25.10
coronary disease. In primary mode: O48.0 post-term pregnancy, chest pain
R07.89/R07.9, syncope R55, depression F32.9, alcohol intoxication F10.129 and
F10.229. The pattern: severity or setting (acute vs chronic, childbirth vs
pregnancy, intoxication vs abuse) varies within a prefix group. Common
catch-all codes also differ from their specific siblings.

## Caveats

* One train/dev split. The ± are over training subsamples, and full-data
  points are single runs. Test was not used.
* The prior's width is pooled per tree depth, not fitted per group. That does
  not change the prior's point prediction, but it does change how much the
  posterior trusts it.
* In every-code mode, predictions from the same admission are correlated.
* Excluding chapter Z removes 2.4% of primary diagnoses and 18% of code
  occurrences. The previous version of this study, with Z included, is in
  `output/hier3_withZ/` (it reported its results in a different unit).

## Reproduce

```
.venv/Scripts/python -m hierprior.onecode          # ~7 min
.venv/Scripts/python -m hierprior.onecode_curve    # ~1 min
.venv/Scripts/python -m hierprior.analyze3
```
