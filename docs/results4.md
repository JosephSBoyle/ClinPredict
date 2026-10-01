# Study 4: 1-year mortality, and the official ICD-10-CM levels

Study 3's one-code model and vocabulary prior, unchanged, run on a second
outcome and on the official ICD-10-CM hierarchy. Same cohort, patient split,
training subsets and dev set; chapter Z excluded; the test split was not used.
Code: `hierprior/levels.py` (fits, curves, calibration, correlogram),
`hierprior/analyze4.py` (tables, figures). Full tables:
`output/hier4/summary.md`.

## Metrics

* **AUROC** of the predicted probability on dev.
* **Log-loss reduction**: the percentage of the base-rate model's log-loss that
  a model removes (McFadden's R² on held-out data). The base-rate model
  predicts the training rate of the outcome for everyone. Negative values are
  worse than the base rate.
* **Break-even**: the number of a code's own admissions at which the
  naive-prior learner's pooled dev log-loss equals the hierarchical prior's
  with none (codes with at least 100 training admissions; 95% bootstrap over
  codes).

## The model

One code per prediction, as in study 3:

    logit P(y | c) = b0 + w_c,    b0 = logit(training rate of y)

and the code's weight is a sum of one learnt coefficient per level of a tree:

    w_c = Σ_{a on c's path} θ_a,    θ_a ~ N(0, τ²_level(a))

Each level has one variance τ², fitted by marginal likelihood. Every θ has a
posterior, computed by Gaussian message passing on the tree. Study 3's
leave-code-out prior N(m_c, P_c) is the distribution of w_c given every other
code's data. The only change here is the tree:

| tree | levels on a code's path (θ terms) | variances |
|---|---|---|
| **prefix** (study 3) | E → E1 → E11 → E116 → E1165 | 11 |
| **official_block** | chapter (E00-E89) + block (E08-E13) + code | 3 |
| **official_category** | chapter + block + category (E11) + code | 4 |
| **official_prefix** | chapter + block + category + E116 + code | 10 |

official_block is the literal three-level model: the code's coefficient is its chapter's
coefficient plus its block's plus its own. Chapters and blocks come from
PyHealth's ICD10CM table (22 chapters, 273 blocks, 1,700 categories among the
observed codes).

**Outcomes.** 30-day readmission, as before (base rate 19.9% of cohort
admissions). **1-year mortality**: the patient's date of death is within 365
days of this discharge (13.2%). MIMIC-IV records out-of-hospital deaths up to
a year after a patient's last discharge, so every admission has a label. The
cohort already excludes in-hospital deaths. The two labels correlate at 0.14.

## 1. Mortality next to readmission (prefix tree)

![scaling](../output/hier4/scaling.png)

Dev AUROC / log-loss reduction, primary diagnosis only (one code per admission):

| training adm | readmission: own + naive prior | readmission: own + hier. prior | mortality: prior only | mortality: own + naive prior | mortality: own + hier. prior |
|---|---|---|---|---|---|
| 514 | 0.521 / −0.0% | 0.548 / −0.2% | 0.594 / +2.0% | 0.565 / +1.4% | **0.621 / +2.9%** |
| 1,690 | 0.543 / +0.6% | 0.577 / +1.1% | 0.669 / +5.8% | 0.623 / +3.6% | **0.688 / +7.0%** |
| 5,230 | 0.575 / +1.2% | 0.607 / +2.1% | 0.694 / +7.8% | 0.664 / +5.4% | **0.713 / +9.0%** |
| 17,406 | 0.608 / +2.5% | 0.631 / +3.4% | 0.713 / +9.1% | 0.716 / +9.2% | **0.745 / +12.0%** |
| 52,348 | 0.627 / +3.4% | 0.643 / +4.1% | 0.722 / +9.7% | 0.740 / +11.5% | **0.757 / +13.2%** |
| 173,919 | 0.644 / +4.4% | 0.653 / +4.8% | 0.726 / +10.0% | 0.759 / +13.4% | **0.766 / +14.2%** |

Mean over 3 training subsets (1 at full data). The readmission rows reproduce
study 3 exactly.

* **The same ordering holds for mortality.** The code's own data with the
  hierarchical prior is best at every size, in both modes. Over the naive
  prior it adds +0.056, +0.065, +0.049, +0.029, +0.017 and +0.007 AUROC from
  0.5k to 174k admissions (readmission: +0.027 to +0.034 at small sizes,
  +0.009 at full data).
* **The prior alone is worth more for mortality.** At full data it removes 10.0%
  of the base-rate log-loss (readmission 3.0%), three quarters of what the
  code's own data remove (13.4%). It beats learning the code with a naive
  prior up to about 17k admissions, as for readmission.
* **Every code occurrence** (full tables in `output/hier4/summary.md`):
  mortality 0.665 / 6.3% with the hierarchical prior against 0.663 / 6.2%
  with the naive one at full data, and 0.597 against 0.569 at 514 admissions.
* **Calibration holds** (codes with ≥ 30 training admissions): the 95%
  interval of the leave-code-out prior covers 97% of primary codes and 95% of
  every-code codes for mortality. Mean z² is 0.86 to 0.93, so the intervals are
  slightly wide. The prior correlates 0.80 to 0.82 with the codes' own
  estimates, against 0.66 for readmission.

### The exchange rate

![learning curves](../output/hier4/curve.png)

| break-even (own admissions) | primary diagnosis | every code |
|---|---|---|
| readmission, relatives from ~5k adm | 8 (5–14) | 17 (10–25) |
| readmission, relatives from all 174k | 23 (14–34) | 71 (51–93) |
| mortality, relatives from ~5k adm | 9 (5–16) | 14 (9–20) |
| mortality, relatives from all 174k | 16 (8–35) | 22 (15–33) |

The prior's advantage in log-loss reduction is three times larger for
mortality (9.6% against 2.9% at m = 0, primary), but it is worth fewer of a
code's own admissions. For mortality a code's own admissions are more
informative: codes differ more (naive prior sd 0.99 against 0.59 logits for
readmission), so the naive learner catches up faster. Both curves meet by
m = 100.

## 2. The official levels (chapter, block, category, code)

![trees](../output/hier4/trees.png)

AUROC difference from the prefix tree, paired by training subset. Full data:

| | readmission, primary | mortality, primary | readmission, every code | mortality, every code |
|---|---|---|---|---|
| prior only: prefix | 0.621 | 0.726 | 0.558 | 0.625 |
| prior only: official_block (chapter + block + code) | 0.611 | 0.704 | 0.543 | 0.601 |
| prior only: official_category (+ category) | 0.623 | 0.729 | 0.556 | 0.623 |
| prior only: official_prefix | 0.623 | 0.730 | 0.558 | 0.625 |
| own + prior: prefix | 0.653 | 0.766 | 0.580 | 0.665 |
| own + prior: official_block | 0.652 | 0.764 | 0.579 | 0.665 |
| own + prior: official_category | 0.654 | 0.766 | 0.580 | 0.665 |
| own + prior: official_prefix | 0.654 | 0.766 | 0.580 | 0.665 |

* **Chapter + block + code (official_block) is a weaker prior.** On its own it is
  0.010 to 0.024 AUROC below the prefix tree at full data, and the gap grows
  with data. It lacks the 3-character category, which carries a large share of
  the variance (in official_category the category sd is 0.28 for readmission and 0.53 for
  mortality). Its break-even is correspondingly lower (readmission: 16 primary
  and 30 every-code, against 23 and 71). With the code's own data the gap
  closes to 0 to 0.002 at full data and at most 0.004 from 1.7k
  admissions up. At 514 admissions it is the best tree for readmission in
  primary mode (+0.010), but not for mortality (−0.006).
* **Adding the category (official_category) matches the prefix tree.** Primary mode, prior
  alone: +0.002 to +0.005 AUROC for mortality from 1.7k admissions up, and up
  to +0.002 for readmission from 17k. Every-code mode: 0.001 to 0.003 lower.
  With own data it is within 0.006 of the prefix tree at every size and
  within 0.001 at full data. The prefix tree's first letter and two-character
  levels are not ICD levels; the official chapters and blocks do as well as
  them, no better.
* **Adding the subcategory prefixes (official_prefix)** changes nothing further.

### What each level carries

Fitted prior sds (log-odds) of the official_category tree at full data, primary diagnoses:

| | chapter | block | category | code |
|---|---|---|---|---|
| readmission | 0.27 | 0.35 | 0.28 | 0.37 |
| mortality | 1.03 | 0.67 | 0.53 | 0.54 |

For readmission the four levels contribute about equally. For mortality the
chapter dominates.

![chapters](../output/hier4/chapters.png)

The chapter coefficients of the official_block model, both outcomes on one axis.
Neoplasms, blood, respiratory and circulatory disease raise 1-year mortality
(+0.6 to +1.1); pregnancy lowers it by 3.4 (0.1% of such admissions die
within the year) and congenital conditions by −1.0. Mental and behavioural
disorders go opposite ways: more readmission (+0.31), less mortality (−0.73).

### Shared levels, shared risk

![correlogram](../output/hier4/correlogram.png)

Model-free correlation of two primary codes' log-odds shifts (969 codes with at
least 30 training admissions) by the deepest ICD level they share, with a
jackknife over chapters:

| shared level | readmission (95%) | model | mortality (95%) | model |
|---|---|---|---|---|
| none (different chapter) | −0.02 (−0.03 to −0.01) | 0.00 | −0.03 (−0.06 to 0.01) | 0.00 |
| chapter | 0.11 (−0.03 to 0.26) | 0.18 | 0.19 (−0.11 to 0.50) | 0.51 |
| block | 0.40 (0.23 to 0.57) | 0.47 | 0.31 (−0.01 to 0.62) | 0.72 |
| category | 0.58 (0.38 to 0.77) | 0.67 | 0.61 (0.39 to 0.82) | 0.86 |

For readmission the fitted model is close to the model-free values at every
level. For mortality it overstates similarity within a chapter: the model's
total variance of a code's shift is 2.07, against 1.07 model-free. The likely
cause is the chapter level. Its variance is estimated from 22 chapters, and
the Gaussian takes pregnancy's −3.4 as typical spread. The overstatement does
not reach the predictions: a code's leave-code-out prior conditions on its
block and category siblings, and its intervals are calibrated (above). A
heavier-tailed chapter prior would be the fix if it mattered.

## Conclusions

1. The vocabulary prior transfers to 1-year mortality unchanged. The code's
   own data with the prior is best at every size in both modes, and the prior
   alone is calibrated and removes 10% of the base-rate log-loss.
2. The truly multi-level model with the official levels works, but only with
   the category. Chapter + block + code alone is a weaker prior (−0.01 to
   −0.02 AUROC alone, about −0.002 with own data). With the category it ties
   with the character-prefix tree. Either tree is fine; the prefix tree needs
   no lookup table.
3. The per-level coefficients are interpretable: for mortality, most of the
   variation between codes is between chapters.

## Caveats

* One patient split; dev only. The ± are over 3 training subsets, and
  full-data points are single runs.
* Deaths after the 1-year follow-up of a patient's last discharge are not
  recorded, but that does not affect a 365-day label.
* In every-code mode, predictions from the same admission are correlated.

## Reproduce

```
.venv/Scripts/python -m hierprior.levels      # all fits, curves, calibration; ~13 min with 8 jobs
.venv/Scripts/python -m hierprior.analyze4    # tables and figures
```
