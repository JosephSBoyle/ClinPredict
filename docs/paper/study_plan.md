# Study plan: the ICD-10 hierarchy for small local prediction models

**Working title.**
*Intepretable prediction models for low-data regimes using multilevel bayesian priors*

## ABSTRACT
Supervised learning typically limits prediction to features seen during training.
In a medical prediction tasks the input clinical features are high-dimensional,
generally n_features >> n_patients. The data are also sparse, and codes (features)
not seen in the training data are rarely incorporated into test-time predictions,
despite often being hierarchically related to other codes with known parameters.
In this study we use the MIMIC IV database to predict 30 day readmissions using
various proportions of the training data using an enhanced logistic regression
model. The enhanced LR model encodes a prior that a codes' related codes act
as a gaussian prior on it's own weights, with strength inversely proportional
to both the variance of that gaussian and to the support of the code itself.

## Question

A hospital fits its own readmission or mortality model on N local admissions.
Can we use the ICD-10 hierarchy to produce informative priors for unseen/rare
codes in a way which produces better predictions?

## Positioning

Each code's LR coefficient is the sum of a chapter, a block, a category and
a code term: `w_c = θ_chapter + θ_block + θ_category + θ_code`.
The model is as transparent and acceptable where a neural network would not
be accepted for ethical reasons.

Related work: GRAM (Choi et al., KDD 2017) learns attention over a code's
ancestors' embeddings inside an RNN. It shows that ontologies help rare
codes, but it is not interpretable at the coefficient level.

## Data and design
Datasets:
NHS Notts NNICB via GPRCC db.
MIMIC-IV 3.1

Tasks:
death within 365 days of discharge
[30-day readmissions]

Additional input data: age, sex, admission type, length of stay, and admissions in the last
12 months Hyperparameters and all analysis choices are fixed on dev. The test split is
  scored once. Reported to TRIPOD+AI.

## Logistic Regression Models
### Baselines:
1. Standard LR. Ignore unseen codes
2. Rollup to the subcategory level (parent of leaf)
3. Elixhauser 1998 via the AHRQ mortality

### Proposed models
3. Hierarchical
w_E11.65 = θ_chapter + θ_block + θ_E11 + θ_E11.6 + θ_E11.65
All θ penalised the same with L2 norm.

4. Proposed model 2:
θ_a ~ N(0, τ²_level(a))
τ̂² = argmax_τ² ∫ p(data | θ) p(θ | τ²) dθ
θ penalised based on the empirical variance of each level.

## Endpoints

**Primary:** absolute difference in test AUROC between the baseline LR models
and the proposed variants.

**Secondary**
- Calibration slope and intercept.
- The number of samples at which standard LR w/ l2  matches the proposed hierarchical
models
- discr./calib. on admissions whose diagnoses were rare (<10) or unseen (0)
in the local training set.
- analyse cases in which the prediction under basic LR and hierprior-LR would be most
different.
