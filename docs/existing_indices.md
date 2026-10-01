# Existing comorbidity indices

Two hand-built comorbidity scores dominate mortality and readmission prediction from ICD codes. Both are natural baselines for a learned ICD-hierarchy model.

## Charlson Comorbidity Index (CCI)

- **Original:** [Charlson et al. 1987, *J Chronic Dis*](https://doi.org/10.1016/0021-9681(87)90171-8). Built to predict 1-year mortality in medical inpatients.
- **Model:** 17 conditions in the ICD versions (19 in the 1987 original), each with an integer weight (1, 2, 3 or 6). The score is the sum of the weights for the conditions present, and is used as a single covariate or as categories (0, 1–2, 3–4, ≥5).
- **ICD coding:** [Quan et al. 2005, *Med Care*](https://doi.org/10.1097/01.mlr.0000182534.19832.83) gives the standard ICD-9-CM and ICD-10 mappings. [Quan et al. 2011, *Am J Epidemiol*](https://doi.org/10.1093/aje/kwq433) re-estimated the weights on modern data, which leaves 12 conditions with non-zero weight.
- **Use:** It is the most widely cited comorbidity score and the default adjustment in mortality studies. It is also the "C" in the LACE readmission index ([van Walraven et al. 2010, *CMAJ*](https://doi.org/10.1503/cmaj.091117)).

## Elixhauser comorbidity measure

- **Original:** [Elixhauser et al. 1998, *Med Care*](https://doi.org/10.1097/00005650-199801000-00004). It defines 30 binary comorbidity flags from administrative data and excludes the conditions that are the main reason for admission. AHRQ maintains the current version, which has 38 flags.
- **Single-score versions:**
  - [van Walraven et al. 2009, *Med Care*](https://doi.org/10.1097/MLR.0b013e31819432e5) gives integer weights (−7 to +12) for the flags, derived from a logistic regression on in-hospital mortality.
  - [Moore et al. 2017, *Med Care*](https://doi.org/10.1097/MLR.0000000000000735) gives the AHRQ Elixhauser Mortality and Readmission indices, which use separate weights for in-hospital mortality and 30-day readmission.
- **Performance:** It usually discriminates slightly better than Charlson. One example is [Sharabiani et al. 2012, *Med Care*](https://doi.org/10.1097/MLR.0b013e31825f64d0), a systematic review. Typical AUROC gains are about 0.01–0.05.

## Relevance here

Both indices are fixed, expert-chosen groupings of ICD codes with linear weights, so they are a coarse, hand-made version of what our hierarchy model learns. On MIMIC they can be computed straight from diagnosis codes, for example with the `comorbidipy` Python package or the AHRQ HCUP software. Using comorbidity indices alone typically gives AUROC of about 0.6–0.65 for readmission and higher for mortality.
