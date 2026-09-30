# Predicting readmission from one code: what a hierarchical prior buys

The one-code-per-admission version of study 3 (full version:
`docs/results3.md`). Every admission is represented by exactly one input, its
primary ICD-10 diagnosis, and the model predicts 30-day readmission from that
code alone. Chapter Z is excluded. MIMIC-IV 3.1, adults: 169,740 training and
24,410 dev admissions (test untouched). Code: `hierprior/onecode.py`,
`hierprior/onecode_curve.py`, `hierprior/analyze3.py` (primary mode).

## The model and the metrics

    P(readmit | primary code c) = sigmoid(b0 + w_c),    b0 = logit(19.2% training base rate)

The only question is how to learn w_c:

| learner | uses the code's own admissions | prior |
|---|---|---|
| prevalence only | no | w = 0 |
| naive prior only | no | raw inverse-variance average of the sibling codes' estimates |
| hierarchical prior only | no | predicted from the code's relatives through the ICD tree (Gaussian hierarchy, variances by marginal likelihood, leave-this-code-out) |
| own data only | yes | none: log((k+.5)/(n-k+.5)) - b0 |
| own data + naive prior | yes | N(0, s²), the same for every code (ridge) |
| own data + hierarchical prior | yes | the hierarchical prior above |

Metrics on dev admissions:
* **AUROC** of the predicted probability (one prediction per admission).
* **Log-loss**: mean over admissions of -ln(predicted probability of the
  actual outcome), in nats per admission (÷ 0.693 for bits). Predicting the
  base rate for everyone gives 0.4893 nats, the entropy of a 19.2% event.
* **Log-loss reduction**: the % of that 0.4893 a learner removes. For example,
  +4.8% means 0.4658 nats per admission.

## 1. By total training data

| training adm | naive prior only | hier. prior only | own data only | own + naive prior | own + hier. prior |
|---|---|---|---|---|---|
| 514 | 0.541 / -14.6% | 0.539 / +0.2% | 0.519 / -7.3% | 0.521 / -0.0% | 0.548 / -0.2% |
| 1,690 | 0.541 / -12.2% | 0.575 / +0.9% | 0.537 / -7.0% | 0.543 / +0.6% | 0.577 / +1.1% |
| 5,230 | 0.541 / -10.4% | 0.592 / +1.6% | 0.570 / -5.8% | 0.575 / +1.2% | **0.607 / +2.1%** |
| 17,406 | 0.565 / -6.0% | 0.611 / +2.4% | 0.601 / -2.3% | 0.608 / +2.5% | **0.631 / +3.4%** |
| 52,348 | 0.576 / -3.1% | 0.618 / +2.7% | 0.620 / +0.6% | 0.627 / +3.4% | **0.643 / +4.1%** |
| 173,919 | 0.588 / -0.5% | 0.621 / +3.0% | 0.639 / +3.0% | 0.644 / +4.4% | **0.653 / +4.8%** |

AUROC / log-loss reduction, dev, mean of 3 training subsamples (1 at full
data).

* Own data + hierarchical prior is best from 1.7k admissions up. Over own
  data + naive prior it adds +0.009 to +0.034 AUROC and 0.4-0.9 points of
  log-loss reduction.
* The hierarchical prior alone, without a single admission of the code,
  matches or beats own data + naive prior up to 17k training admissions.
* The naive sibling average is worse than predicting the base rate at every
  size, and the no-prior estimate is worse below ~40k admissions. Both are
  overconfident.

![scaling](../output/hier3/onecode_scaling.png)

## 2. By the code's own data: the prior is worth ~20 admissions

For the 328 primary codes with at least 100 training admissions, the learner
sees exactly m of the code's admissions (random draws, 30 repeats). The prior
comes from all other codes' training data.

| m (code's own admissions) | 0 | 1 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| AUROC, own + hier. prior | **0.618** | 0.619 | 0.624 | 0.631 | 0.636 | 0.645 | 0.650 |
| AUROC, own + naive prior | 0.500 | 0.523 | 0.569 | 0.599 | 0.618 | 0.640 | 0.649 |
| log-loss reduction, own + hier. prior | **+2.9%** | +2.9% | +3.2% | +3.5% | +3.8% | +4.3% | +4.7% |
| log-loss reduction, own + naive prior | 0.0% | +0.2% | +1.1% | +2.0% | +2.9% | +4.0% | +4.6% |
| log-loss reduction, own data only | 0.0% | -24% | -7.8% | -2.6% | +0.8% | +3.6% | +4.5% |

(Base-rate log-loss for these codes' dev admissions: 0.4990 nats. Own data
only ranks the codes exactly like own + naive prior, so its AUROC is the
same.)

With the naive prior, a learner needs **about 20 of the code's own
admissions** to match what the hierarchical prior gives with none (21 by
log-loss, 20 by AUROC). If the relatives come from only ~5k training
admissions, the prior is worth about 9. The gap closes by about 100
admissions, and the hierarchical posterior is never worse.

![learning curve](../output/hier3/onecode_curve.png)

## 3. By how rare the code is (full data)

| code's training admissions | AUROC: hier. prior only / own + naive / own + hier. | log-loss reduction |
|---|---|---|
| 0 (unseen) | 0.641 / 0.500 / 0.641 | +3.0% / 0 / +3.0% |
| 1-9 | 0.632 / 0.580 / 0.642 | +3.5% / +1.6% / +4.0% |
| 10-99 | 0.623 / 0.633 / 0.647 | +2.9% / +3.7% / +4.4% |
| 100-999 | 0.628 / 0.663 / 0.663 | +3.4% / +5.6% / +5.6% |
| >= 1000 | 0.592 / 0.640 / 0.639 | +1.4% / +4.0% / +4.0% |

## 4. Drill-down

![drill-down](../output/hier3/drilldown_primary.png)

Group results are dev log-loss reductions on the group's admissions (86-390
per group, so they are noisy):

| group | prior only | own + naive prior | why |
|---|---|---|---|
| F32.x depression | 6.0% | 7.2% | rare subtypes all about +0.5; the common F32.9 is +0.95 and the prior undershoots it |
| I61.x intracerebral haemorrhage | **7.5%** | 6.6% | every site is rarely readmitted (-0.5); at 17k admissions the prior wins 7.8% vs 1.7% |
| C78.x secondary cancers | 8.0% | 10.2% | all about +0.5; at 17k the prior wins 11.1% vs 10.4% |
| K57.3x diverticular disease | 4.1% | 5.3% | all mildly protective |
| I50.2x systolic heart failure | 0.8% | 9.1% | acute-on-chronic is +0.52, the rest are about 0; the average fits neither |
| K85.9x acute pancreatitis | 0.3% | 5.7% | necrotising (+0.5 to +0.9) vs plain (+0.15) |

The prior fails where severity or setting varies inside a prefix group:
acute vs chronic, necrotising vs not, and pregnancy (O48.0 post-term,
O99.824 childbirth) vs other obstetric codes.

## Caveats

One train/dev split. Full-data points are single runs. The hierarchical
prior's width is pooled per tree depth rather than per group.
