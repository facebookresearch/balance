---
id: ipw
title: Inverse Propensity Score Weighting
description: Inverse Propensity Score Weighting
sidebar_position: 1
keywords:
    - inverse propensity score weighting
    - ipw
    - ipsw
---
## Introduction
The inverse propensity score weighting is a statistical method to adjust a non-random sample to represent a population by weighting the sample units. It assumes two samples:

(1) A sample of respondents to a survey (or in a more general framework, a biased panel), will be referred to as "sample".

(2) A sample of a target population, often referred to as "reference sample" or "reference survey" [1],  will be referred to as "target". This sample includes a larger coverage of the population or a better sampling properties in a way that represents the population better. It often includes only a limited number of covariates and doesn't include the outcome variables (the survey responses). In different cases it can be the whole target population (in case it is available), a census data (based on a survey) or an existing survey.


## Mathematical model

Let $S$ represent the sample of respondents, with $n$ units, and $T$ represent the target population, with $N$ units. We may assume each unit $i$ in the sample and target have a base weight, which is referred to as a design weight, $d_i$. These are often set to be 1 for the sample (assuming unknown sampling probabilities), and are based on the sampling procedure for the target. In addition, we assume all units in sample and target have a covariates vector attached, $x_i$. Note that we assume that the same covariates are available for the sample and the target, otherwise we ignore the non-overlapping covariates.



Define the propensity score as the probability to be included in the sample (the respondents group) conditioned on the characteristics of the unit, i.e. let $p_i = Pr\{i \in S | x_i\}$, $i=1...n$. $p_i$ is then estimated using logistic regression, assuming a linear relation between the covariates and the logit of the probability: $\ln(\frac{p_i}{1-p_i})=\beta_0+\beta_1 x_i$.


Note that balance's implementation for ```ipw``` uses a regularized logistic model through using [LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics)) (by using [glmnet-python](https://glmnet-python.readthedocs.io/en/latest/glmnet_vignette.html)). This is in order to keep the inflation of the variance as minimal as possible while still addressing the meaningful differences in the covariates between the sample and the target.

### How are the regularization parameter and trimming ratio parameter chosen?
There are two options to choose the regularization parameter and trimming ratio parameter in balance:

1. Bounding the design effect by setting ```max_de = X```. In this case the regularization parameter and the trimming ratio parameter are chosen by a grid search over the 10 models with the largest design effect. This is based on the assumption that a larger design effect often implies better covariate balancing. Within these 10 models, the model with the smallest ASMD is chosen.

2. Choosing the regularization parameter by the "1se rule" (or "One Standard Error Rule") of cross validation, i.e. the largest penalty factor $\lambda$ at which the MSE is at most 1 standard error from the minimal MSE . This is applied when ```max_de``` is set to ```None```. In this case the trimming ratio parameter is set by the user, and default to 20.

### Weights estimation

The estimated propensity scores are then used to estimate the weights of the sample by setting $w_i = \frac{1-p_i}{p_i} d_i$.





## References
[1] Lee, S., & Valliant, R. (2009). Estimation for volunteer panel web surveys using propensity score adjustment and calibration adjustment. Sociological Methods & Research, 37(3), 319-343.

 - More about [Inverse Probability Weighting](https://en.wikipedia.org/wiki/Inverse_probability_weighting) in Wikipedia.
