# Hierarchical priors for out-of-distribution prediction in sparse data regimes

ABSTRACT
Typically, supervised learning limits itself to features seen during training.
In a medical prediction tasks the input clinical features are high-dimensional,
generally n_features >> n_patients. The data are also sparse, and codes (features)
not seen in the training data are rarely incorporated into test-time predictions,
despite often being hierarchically related to other codes with known parameters.
In this study we use the MIMIC IV database to predict 30 day readmissions using
various proportions of the training data using an enhanced logistic regression
model. The enhanced LR model encodes a prior that a codes' related codes act
as a gaussian prior on it's own weights, with strength inversely proportional
to both the variance of that gaussian and to the support of the code itself.

METHOD
ICD10 codes as inputs.
Randomly select an input sibling for each final-level hierarchy (seed =0).
Drop all patients with these codes from the training but not dev/test splits
to test whether the prior weights are appropriate on dev.
Compare this with a baseline model which simply doesn't attempt to predict
based off of those codes-- do we do better on AUC?

Due to collinearity also consider the regime where our LR model has exactly
one input feature (i.e. in the baseline case for many codes we simply have
no data and resort to our readmissions prevalence prior).

Output a scaling chart showing the performance as a fn of training data; log-scale.

EXPLORATION
do some exploratory analysis and follow-up experiments clearly labelled as exploratory.
