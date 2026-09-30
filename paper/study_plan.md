# Study plan: the ICD-10 hierarchy for small local prediction models

**Working title.** *Borrowing strength from the ICD-10 hierarchy: interpretable
prediction models for hospitals with little data*

## Question

A hospital fits its own readmission or mortality model on N local admissions
(N = 250 to 20,000). Does a prior built from the ICD-10 hierarchy beat what it
would otherwise use?

1. A hand-crafted score, used as published or refitted locally: LACE and
   HOSPITAL for 30-day readmission, Charlson for 1-year mortality.
2. Expert code groupings as features: Charlson, Elixhauser, CCSR, CMS-HCC.
3. Logistic regression on raw codes.

## Positioning

The model stays a logistic regression. Each code's coefficient is the sum of a
chapter, a block, a category and a code term,
`w_c = θ_chapter + θ_block + θ_category + θ_code`, and each term can be read
off and audited. The model is as transparent as a points score, and a hospital
can deploy it where a neural network would not be accepted.

Related work, cited rather than used as baselines:
- Ontology-aware neural networks. GRAM (Choi et al., KDD 2017) learns attention
  over a code's ancestors' embeddings inside an RNN. It shows that ontologies
  help rare codes, but it is not interpretable at the coefficient level. A
  GRAM-style model may be added as an uninterpretable reference to price the
  cost of interpretability, but it is not a comparator.
- Structured sparsity (tree-guided and hierarchical group lasso) and roll-up or
  ancestor features in claims modelling.
- Hand-crafted groupings (Elixhauser, Charlson, CCSR, CMS-HCC). These are expert
  priors on the same vocabulary, and the main comparison.

## Data and design

- MIMIC-IV 3.1, adult admissions with all diagnoses in ICD-10-CM, in-hospital
  deaths excluded. The patient split is the one used in studies 1 to 4
  (70 / 10 / 20).
- Outcomes: 30-day readmission, and death within 365 days of discharge.
- Local training sets: random subsets of the training patients at
  N ∈ {250, 500, 1k, 2k, 5k, 10k, 20k}, 10 draws each, plus the full training
  set as a reference.
- Every model gets the inputs a real local model would have: age, sex,
  admission type, length of stay, and admissions in the previous 12 months.
- Hyperparameters and all analysis choices are fixed on dev. The test split is
  scored once. The analysis plan is pre-registered (OSF) before test is
  touched. Reported to TRIPOD+AI.

## Models

| # | Model | Interpretable |
|---|---|---|
| M0 | Published score as is (LACE, HOSPITAL, Charlson) | yes |
| M1 | M0's inputs refitted locally | yes |
| M2 | Basic inputs + expert groups (Elixhauser, CCSR, CMS-HCC; one model each) | yes |
| M3 | Basic inputs + codes, one L2 penalty | yes |
| M4 | M3 + prefix / ancestor features, one L2 penalty | yes |
| M5 | M3 with the hierarchical prior over chapter, block, category and code, variances by marginal likelihood | yes |
| (R) | GRAM-style network, reference only | no |

## Endpoints

- **Primary:** difference in test AUROC between M4 and the best of the M2
  models at N = 2,000, for each outcome, with a patient-bootstrap 95% CI.
- **Secondary:**
  - Log-loss reduction as a percentage of the base-rate model's.
  - Calibration slope and intercept.
  - Data-equivalence ratio: the N at which M2 or M3 matches M4.
  - AUROC on admissions whose diagnoses were rare (fewer than 10 occurrences)
    or unseen in the local training set.

## Sensitivity analyses (coding closer to low-resource practice)

1. Truncate codes to 4 characters, which approximates WHO ICD-10. The chapter,
   block and category levels are unchanged.
2. Keep only the first k diagnoses per admission (k = 1, 3, 5), for sparse
   coding.
3. Non-random "small hospital" subsets, for example admissions through one
   entry route or in one period, to check that the results hold beyond random
   subsampling.

## Deliverables

- A paper of 3,000 to 5,000 words, with code and a preprint.
- A scikit-learn-compatible `ICDPrefixFeatures` transformer and the
  hierarchical-prior fit, as a small package (a JOSS paper is optional).

## Risks

- **Expert groupings may capture most of the gain.** That is still publishable:
  a data-driven prior matching decades of expert curation, with no mapping
  tables to maintain.
- **One US hospital system.** Frame the paper as "models trained on small
  datasets", and keep low-resource settings as motivation, not as a claim.
- **Readmission scores need inputs** (ED visits, discharge labs for HOSPITAL).
  These are available in MIMIC-IV, but the extraction needs checking early.

## Timeline (about 5 weeks)

| Week | Work |
|---|---|
| 1 | Baselines: LACE, HOSPITAL, Charlson, Elixhauser, CCSR and CMS-HCC mappings; basic inputs |
| 2 | Pre-register; run the dev grid; sensitivity analyses |
| 3 | Score test once; figures |
| 4–5 | Write; package the code; preprint and submit |

## Venues

Fast, soundness-based review:
- **TMLR.** Reviews for correctness and interest, not significance. Decisions
  in about two months; free and open access. Frame it for an ML audience.
- **PeerJ / PeerJ Computer Science.** Soundness-based review, modest fees.
- **BMJ Open.** Clinical framing; publishes many prediction-model studies.
- **BMC Medical Informatics and Decision Making.** Mid-tier, broad scope.
- **BMC Research Notes.** For a short version with one headline result.

Methods-focused, more selective:
- Diagnostic and Prognostic Research.
- JAMIA Open.

Short and conference options:
- MIE or MedInfo (short papers).
- AMIA Informatics Summit.
- ML4H.

Plan: arXiv or medRxiv preprint on submission. Submit to TMLR first, then BMJ
Open or PeerJ Computer Science.
