# Learned hierarchical priors on grouped code features

ABSTRACT
In the first study the best model was the "ancestor-indicator" model: alongside
each ICD10 code it gets a yes/no input for every broader group the code belongs
to (E11.65 also switches on "some E11.6x", "some E11", "some E code"). Unseen
codes still switch on their groups, so it can predict from them. It uses one
fixed L2 penalty. Here we instead learn a gaussian prior per group, centred on
the parent group's weight, and ask whether that beats the fixed penalty.

METHOD
Same cohort, label and splits. Ancestor-indicator inputs.
Enhanced: per-group variances learned from data, one global strength tuned on
dev, no support term. Baseline: single L2 penalty.
Unseen codes arise naturally, no patient dropping:
- random training subsets at log-spaced sizes, 3 seeds
- temporal: train on earlier anchor_year_group bins, test on the latest
Report AUC on all test admissions and on those with an unseen code.
Output a scaling chart as a fn of training data; log-scale.

EXPLORATION
do some exploratory analysis and follow-up experiments clearly labelled as exploratory.
Drill into an example e.g. diabetes or HF and show the data's variance and how a code's
parameters have been estimated i.e. draw the gaussian and so on.