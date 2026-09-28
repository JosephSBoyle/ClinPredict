# Hierarchical priors for unseen ICD codes: executive summary

**Question.** When predicting 30-day readmission from ICD-10 diagnosis codes,
can the code hierarchy help with codes the model never saw in training? The
idea is to give each code a prior built from its relatives.
(MIMIC-IV 3.1, 248k adult admissions, 19k codes, readmission rate 20%.
Details: `docs/results.md` for study 1 and `docs/results2.md` for study 2.)

**Answer.** Yes, but a simple model captures almost all of the benefit. Give
each code extra 0/1 inputs for its broader groups (E11.65 also switches on
E11.6, E11 and E) and fit with one ordinary L2 penalty. This "ancestor-indicator"
model beats plain logistic regression by +0.005 to +0.03 AUROC, most at small
training sizes. None of the learned hierarchical priors reliably improves on it.

| | Study 1 | Study 2 |
|---|---|---|
| Prior tested | code weight ~ N(group mean), weaker for well-supported codes | learned per-group Gaussian priors on the ancestor-indicator inputs |
| Result vs its baseline | as specified: tie with plain LR. Dropping the support term: +0.007 to +0.027 | centred on parent: -0.006 to +0.004 (worse below 5k admissions). Centred on 0 (exploratory): 0 to +0.0025 |
| Value of imputed weights for unseen codes | +0.002 to +0.005 AUROC on affected admissions | 0 to +0.002 (the group inputs already carry the signal) |
| Best model | ancestor-indicator LR | ancestor-indicator LR, or the same with learned variances centred on 0 (tiny gain) |

**Why the learned priors add little.**
1. With group inputs present, a code's own weight is only a small correction
   (prior sd around 0.04 logits). The data on rare codes are far noisier than
   that, so every sensible prior pulls them to about the same place.
2. The variances learned by empirical Bayes are nearly equal across groups
   (within a factor of 1.2 to 1.8). A single penalty is therefore already close
   to the best prior of this form. The one clear structure is that chapter-level
   groups barely differ from each other.
3. Co-coded diagnoses explain most of a code's effect, so an unseen code's
   conditional weight is small whatever the prior says. In the one-code-at-a-time
   setting of study 1, the imputed marginal effects are accurate (r = 0.6-0.8).

**Speed.** Four changes to the solver made full-data fits take minutes instead
of hours, with the same AUROC to 1e-4: a size-relative stopping tolerance,
Newton-CG for all models, warm-started hyperparameter grids, and one parallel
job per model. The study 1 run that was killed after more than 4 CPU-hours now
takes 7 minutes, and its result has been added. The whole study 2 grid takes
about 15 minutes.

**Caveats.** Single data split. The "temporal" split is only partly temporal:
MIMIC's year bins are per patient, so 25% of "pre-2020" training admissions
actually took place in 2020 or later, and almost no codes are truly new. The
learned variances depend on how long empirical Bayes runs. Running it longer
helped at 17k admissions (+0.006) but hurt below 5k.

**Recommendation.** Use the ancestor-indicator LR as the default way to use
the ICD hierarchy. Stop tuning priors on this setup. A real test of prediction
for unseen codes needs codes that are actually new: a calendar-year split built
from estimated admission dates, or transfer to another hospital or to the
ICD-9 to ICD-10 switch. Richer inputs (procedures, medications) would move
absolute AUROC (0.70) more than any prior studied here.
