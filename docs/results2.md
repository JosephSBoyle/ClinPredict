# Learned group priors on the ancestor-indicator LR: results

Study 2 (`docs/idea2.md`). Task, cohort, label and patient split are the same as
in study 1 (`docs/results.md`): 30-day readmission after an adult MIMIC-IV 3.1
admission, predicted from its ICD-10-CM diagnosis codes. Code:
`hierprior/grouped.py` (models), `hierprior/experiment2.py`,
`hierprior/analyze2.py`, `hierprior/explore2.py`. Raw results:
`output/hier2/runs/`. Full tables: `output/hier2/summary.md`.

**TL;DR**

* **The enhanced model does not beat the baseline.** It uses learned per-group
  Gaussian priors, each centred on the parent group's weight. Against one fixed
  L2 penalty it is worse at small training sizes (-0.003 to -0.006 AUROC below
  about 5k admissions) and slightly better above about 17k (+0.001 to +0.003
  overall, +0.003 to +0.004 on admissions with an unseen code). In the temporal
  split it is worse or level at every size from 1.8k admissions up.
* **Learned variances centred on 0 give a small, steady gain.** This
  exploratory variant has the same per-group variances but is centred on 0,
  like the baseline. From 5k admissions up it gains +0.001 to +0.0025 AUROC in
  the random split and +0.0003 to +0.0008 in the temporal split, plus +0.0005
  to +0.001 nats of log-loss in both. The effect is small, but its sd across
  subsamples is smaller still.
* **The learned variances hardly vary between groups.** After empirical
  Bayes, the prior sd of 90% of groups lies within a factor of 1.2 of the
  others (1.3 to 1.8 after 50 EM steps). The one clear pattern is in the
  enhanced model: the top levels (letter and 2-character groups) get much
  tighter priors than the deeper ones. With the ancestor indicators
  present, the data say "shrink every group about equally", so a single
  penalty is already close to the best prior of this form.
* **Imputation adds little.** Giving unseen codes their parent group's weight
  instead of 0 is worth +0.000 to +0.002 AUROC on the affected admissions. The
  ancestor indicators already carry nearly all the information about an unseen
  code.
* **The speed fixes work.** A full-data fit of the whole hyperparameter grid
  takes 15 s to 2 min per model. With study 1's solver the same data size took
  2 to over 4 CPU-hours. Study 1's missing full-data point (protocol N) now
  runs in 7 minutes and has been added to `docs/results.md`.

![scaling](../output/hier2/scaling.png)

## Setup

**Inputs.** Every non-root node of the character-prefix tree (19,284 codes and
14,025 groups) gets a 0/1 input: "the admission has some code under this node".
E11.65 switches on E, E1, E11, E116 and E1165. A code never seen in training
still switches on its groups.

**Models** (penalised maximum likelihood, unpenalised intercept):

| name | prior on the input weights | tuned on dev |
|---|---|---|
| baseline | w_v ~ N(0, 1/lambda), the same for every node (study 1's ancestor-indicator LR) | lambda |
| enhanced | w_v ~ N(w_parent(v), sigma2_parent(v) / alpha), root weight 0 | alpha |
| var-only [exploratory] | w_v ~ N(0, sigma2_parent(v) / alpha) | alpha |
| codes-only [reference] | study 1's L2 LR on the code inputs alone | lambda |

sigma2_g is one variance per parent group g, estimated by empirical Bayes.
EM starts from sigma2 = 0.01 (the scale of the baseline's usual lambda = 100),
uses a diagonal Laplace posterior, and shrinks each group towards the pooled
value for its depth with 3 pseudo-children. It stops after 10 steps or when the
mean |log change| drops below 0.01. There is no support term. In the enhanced
model an unseen node has no data, so its weight equals its parent's weight.

**Settings.** No patients are dropped.
* *Random*: training patients subsampled at 0.3%-100% of the 70% train split
  (log-spaced, 3 seeds; 1 at 100%). Dev and test are the 10% and 20% splits.
* *Temporal*: train on patients whose `anchor_year_group` is before
  2020-2022 (their 70% + 20% splits, subsampled the same way). Dev is their
  10% split; test is every patient in the 2020-2022 bin (40,863 admissions).

"Unseen" codes are codes with no training admission. "Affected" admissions
are dev/test admissions with at least one unseen code. The hyperparameter is
chosen by dev AUROC over a 9-point log grid, and test is reported.

## Results

### Random subsets

Test AUROC, all admissions / admissions with an unseen code:

| train adm | unseen codes | affected test adm (of 49,245) | baseline | enhanced | var-only | codes-only |
|---|---|---|---|---|---|---|
| 514 | 17,630 | 39,196 | 0.594±0.007 / 0.589±0.011 | 0.592±0.003 / 0.589±0.004 | 0.594±0.008 / 0.589±0.011 | 0.565±0.007 / 0.557±0.006 |
| 1,690 | 16,135 | 27,358 | 0.633±0.014 / 0.633±0.015 | 0.626±0.015 / 0.626±0.017 | 0.633±0.015 / 0.633±0.015 | 0.610±0.008 / 0.609±0.010 |
| 5,230 | 14,034 | 16,323 | 0.660±0.006 / 0.653±0.006 | 0.657±0.007 / 0.650±0.006 | 0.661±0.006 / 0.654±0.006 | 0.643±0.005 / 0.631±0.008 |
| 17,406 | 10,784 | 7,489 | 0.675±0.001 / 0.671±0.007 | 0.678±0.003 / 0.675±0.006 | 0.676±0.001 / 0.672±0.007 | 0.663±0.002 / 0.656±0.011 |
| 52,348 | 7,018 | 3,351 | 0.689±0.001 / 0.679±0.009 | 0.691±0.001 / 0.682±0.008 | 0.692±0.001 / 0.682±0.010 | 0.682±0.001 / 0.669±0.010 |
| 173,919 | 1,765 | 1,264 | 0.699 / 0.685 | 0.700 / 0.689 | 0.701 / 0.685 | 0.694 / 0.672 |

Paired differences from the baseline on the same subsample (AUROC all /
affected):

| train adm | enhanced | var-only |
|---|---|---|
| 514 | -0.0025±0.0087 / -0.0000±0.0076 | -0.0003±0.0003 / -0.0003±0.0004 |
| 1,690 | -0.0064±0.0069 / -0.0070±0.0074 | +0.0002±0.0010 / -0.0001±0.0012 |
| 5,230 | -0.0031±0.0015 / -0.0025±0.0009 | +0.0011±0.0003 / +0.0014±0.0006 |
| 17,406 | +0.0032±0.0020 / +0.0040±0.0033 | +0.0014±0.0003 / +0.0011±0.0007 |
| 52,348 | +0.0019±0.0006 / +0.0027±0.0019 | +0.0025±0.0000 / +0.0028±0.0009 |
| 173,919 | +0.0013 / +0.0039 | +0.0018 / +0.0003 |

Log-loss differences are all under 0.0035 nats per admission, and for the
enhanced model they do not always go the same way as AUROC. The var-only model
improves log-loss by +0.0005 to +0.001 at every size from 5k admissions up, in
both settings (`output/hier2/summary.md`).

### Temporal (train before 2020, test 2020-2022)

| train adm | unseen codes | affected test adm (of 40,863) | baseline | enhanced | var-only | codes-only |
|---|---|---|---|---|---|---|
| 524 | 17,671 | 33,668 | 0.599±0.006 / 0.600±0.005 | 0.602±0.005 / 0.603±0.006 | 0.598±0.006 / 0.599±0.005 | 0.572±0.006 / 0.570±0.007 |
| 1,823 | 16,157 | 24,632 | 0.640±0.016 / 0.640±0.016 | 0.637±0.018 / 0.639±0.019 | 0.640±0.016 / 0.641±0.016 | 0.616±0.006 / 0.613±0.005 |
| 5,627 | 13,952 | 14,601 | 0.668±0.004 / 0.672±0.008 | 0.663±0.007 / 0.666±0.008 | 0.669±0.004 / 0.673±0.008 | 0.650±0.002 / 0.651±0.004 |
| 18,671 | 10,704 | 7,096 | 0.686±0.001 / 0.689±0.005 | 0.682±0.002 / 0.685±0.007 | 0.687±0.001 / 0.690±0.004 | 0.670±0.002 / 0.670±0.008 |
| 55,753 | 6,865 | 3,217 | 0.695±0.001 / 0.684±0.008 | 0.694±0.001 / 0.682±0.005 | 0.696±0.001 / 0.687±0.007 | 0.684±0.001 / 0.671±0.009 |
| 186,539 | 1,680 | 1,239 | 0.704 / 0.665 | 0.704 / 0.667 | 0.705 / 0.669 | 0.697 / 0.659 |

Paired: enhanced minus baseline is +0.003 at 524 admissions (sd 0.009), then
-0.002, -0.006, -0.004, -0.001 and 0.000. Var-only minus baseline is +0.0003
to +0.0008 overall at every size from 1.8k admissions up, and +0.002 to +0.004
on affected admissions at the two largest sizes.

The temporal split is only partly temporal. `anchor_year_group` is a
per-patient bin, and patients anchored earlier keep being admitted: an
estimated 25% of the "before 2020" training admissions actually took place in
2020 or later (E4). At full data, only 1,680 codes in the test period are
unseen, and none occurs more than 9 times in test (the COVID-19 codes, for
example, already appear in training). So this setting mostly tests the same
thing as the random one, with a 2-percentage-point drop in prevalence (17.7%
in test).

### Do the imputed weights help? (enhanced model)

On test admissions with an unseen code: AUROC with the unseen codes and groups
at their parent's weight, minus AUROC with them at 0.

| train adm | random | temporal |
|---|---|---|
| ~0.5k | +0.0024±0.0023 | +0.0024±0.0029 |
| ~1.7k | +0.0008±0.0007 | +0.0009±0.0013 |
| ~5k | +0.0008±0.0001 | +0.0006±0.0005 |
| ~17k | +0.0003±0.0008 | +0.0005±0.0008 |
| ~53k | -0.0000±0.0004 | +0.0006±0.0018 |
| full | +0.0007 | -0.0006 |

The gain is positive but at most about 0.002, smaller than study 1's
+0.002-0.005. There the model had no group inputs, so imputation was its only
route to an unseen code.

## Exploratory analyses

All of these were chosen after seeing the main results.

**E1. How a code's weight is estimated (drill-down).** Two groups are shown:
I50.4x (combined systolic and diastolic heart failure) and E11.4x (type 2
diabetes with neurological complications), each at ~5k and at 174k training
admissions. For each code, the figure plots the prior (the group's weight, sd
sqrt(sigma2/alpha)) and what the data alone say about the code's own weight
(Laplace approximation of the likelihood, other weights held at the fit).
The fitted weight combines the two, weighted by their precisions.

![drill-down](../output/hier2/explore_drilldown.png)

The prior on a code's *own* weight is narrow: sd 0.03 at 5k admissions, 0.04-0.05
at full data. The data are weak by comparison: the data sd is 0.1-0.2 for
codes with a few hundred admissions and 0.3-2 for rare codes. Every code's
own weight therefore stays near its group's weight. Example: I50.41 (acute
combined HF, 55 admissions). Alone the data say +0.85±0.34, the enhanced fit
gives +0.08 (group weight 0.065) and the baseline gives +0.06. The two models
differ by a few hundredths of a logit per code, which is why their AUROCs
differ in the third decimal. Unseen codes (I50.40 and I50.41 at 5k, E11.41,
E11.49) sit exactly on the group weight in the enhanced model and at 0 in the
baseline. In both, the group inputs (I, I5, I50, I504) carry the code's
risk.

**E2. Learned prior spread by level.** In the enhanced model, the letter and
2-character levels get far tighter priors (median sd 0.001-0.025) than the
3-7-character levels (0.02-0.06), for training sizes above 1k admissions. A group's weight barely departs from its
chapter's, but codes depart from their category. Within a level, the learned
sds of 90% of groups are within a factor of 1.2 of each other, and in the
var-only model all levels are nearly equal. The level-to-level swings with
training size mostly track the tuned alpha.

![variance by depth](../output/hier2/explore_variance_by_depth.png)

**E3. Letting the variances move further (50 EM steps, no early stop;
random setting).** The group sds spread more, with a 5th-95th percentile ratio
of 1.3 (enhanced) and 1.7-1.8 (var-only). Results are mixed: worse at 1.7k
admissions (enhanced 0.621, var-only 0.625, vs baseline 0.633), better at 17k
(var-only 0.681±0.001 vs 0.675±0.001, the largest gain in this study), and
level at full data (0.700-0.701 vs 0.699). Better-learned variances help when
there is enough data to learn them, and hurt below about 5k admissions.
(A MacKay evidence update was also tried on dev at 5k and 17k. It sends the
deep levels' variances to 0 and was no better on dev, so it was not run on
test.)

**E4. How temporal the split is.** Estimated calendar year = admit year -
anchor_year + middle of the anchor_year_group (±1 year). Training admissions
fall at 2015 / 2018 / 2021 (10th / 50th / 90th percentile), with 25% in 2020
or later. Test admissions fall at 2021 / 2021 / 2022.

## Speed

Study 1's full-data runs took 2 CPU-hours (protocol B) and more than 4
(protocol N, killed). Four changes, shared by both studies' code, fix this:

1. **Tolerance relative to data size.** Newton-CG stops when the gradient norm,
   in Jacobi-scaled coordinates, falls below 1e-3 x sqrt(n). This bounds the
   objective gap per admission. The old absolute 1e-5 (and L-BFGS at 1e-6 for
   the L2 models) was far stricter than the AUROC needs. Test AUROC matches the
   old solver to 1e-4.
2. **L2 models moved from L-BFGS to the same Newton-CG solver.**
3. **Warm-started hyperparameter grid**, walked from strong to weak
   regularisation. Empirical Bayes runs once per training set, with
   warm-started, early-stopped EM.
4. **One job per (setting, size, seed, model)**, so the few large fits run
   in parallel instead of one after another on a single core. The tree
   transform for the enhanced model is applied level by level instead of
   being built as a matrix (Z @ T would be ~50M non-zeros at full data).

| | study 1 solver | new solver |
|---|---|---|
| study 1, protocol B, 10k admissions, 4 models x 9 settings (same machine, sequential) | 155 s | 82 s |
| study 1, protocol N, 174k admissions, all 4 models | killed after >4 CPU-h | 437 s |
| study 2, 174k admissions, per model (9-point grid + EB) | - | 15-124 s |

The whole study 2 grid (128 jobs) ran in about 15 minutes on 5 workers. One
slow case remains: study 1's beta = 1 model is badly conditioned during
empirical Bayes at alpha = 1. It accounts for 34 of the 82 s above, and it
follows from study 1's specification.

## Caveats

* One train/dev/test split and one temporal cut. The ± are over training
  subsamples only, and the full-data points are single runs.
* EB variances depend on the EM start and step count (E3). The main runs use
  10 steps from sigma2 = 0.01, so weakly identified groups stay near their
  level's pooled value. Alpha then absorbs the overall scale.
* The temporal split uses the per-patient anchor bin (E4), so it is not a
  clean calendar split and very few codes are truly new.
* The drill-down's "data alone" estimates are conditional on the enhanced
  fit's other weights. They explain the enhanced fit but are not directly
  comparable to the baseline's weights.

## Reproduce

```
.venv/Scripts/python -m hierprior.experiment2 --jobs 5                                   # ~15 min
.venv/Scripts/python -m hierprior.experiment2 --settings R --fracs 0.01,0.1,1.0 --models hier_em50,hier0_em50
.venv/Scripts/python -m hierprior.analyze2                                               # scaling.png + summary.md
.venv/Scripts/python -m hierprior.explore2
.venv/Scripts/python -m hierprior.experiment --protocols N --fracs 1.0 --jobs 1 --sub runs_fast   # study 1's missing point
```
