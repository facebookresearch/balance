---
id: cbps
title: Covariate Balancing Propensity Score
description: Covariate Balancing Propensity Score
sidebar_position: 2
keywords:
    - Covariate Balancing Propensity Score
    - cbps
---
# Covariate Balancing Propensity Score (CBPS)

The Covariate Balancing Propensity Score (CBPS) algorithm estimates the propensity score in a way that maximizes the covariate balance as well as the prediction accuracy of sample inclusion - against some target population of interest. Its main advantage is in cases when the researcher wants better balance on the covariates than traditional propensity score methods - because one believes the assignment model might be misspecified and would like to avoid the need to fit followup models to improve the balance of the covariates.


## References and implementation

**Reference:** Imai, K., & Ratkovic, M. (2014). Covariate balancing propensity score. *Journal of the Royal Statistical Society: Series B: Statistical Methodology*, 243-263. ([link](https://imai.fas.harvard.edu/research/files/CBPS.pdf))

**R package:** https://cran.r-project.org/web/packages/CBPS/ ([github repo](https://github.com/kosukeimai/CBPS/) )

The implementation of CBPS in balance is based on the R package, but is enhanced so to match balance's workflow by adding: features transformations, ability to bound the design effect by running a constrained optimization, and weight trimming.

For the implementation in balance see [code](https://github.com/facebookresearch/balance/blob/main/balance/weighting_methods/cbps.py).

The CBPS implementation in balance was written by Luke Sonnet and Tal Sarig.

## Introduction

**Goal:** Estimate the propensity score that will also result in maximizing the covariate balance.

**Background:** When estimating propensity score, there is often a process of adjusting the model and choosing the covariates for better covariate balancing. The goal of CBPS is to allow the researcher to avoid this iterative process and suggest an estimator that is optimizing both the propensity score and the balance of the covariates together.

**Advantages of this method over propensity score methods:**

1. Preferable in the cases of misspecification of the propensity score model, which may lead to a bias in the estimated measure.
2. Simple to adjust and extend to other settings in causal inference.
3. Inherits theoretical properties of GMM (generalized method of moments) and EL (empirical likelihood), which offers some theoretical guarantees of the method.


## Methodology

A full description of the methodology and details are described in [Imai and Ratkovic (2014)](https://imai.fas.harvard.edu/research/files/CBPS.pdf). We provide here a short description of the methodology.

Consider a sample of respondents of size $n$ and a random sample from a target populaiton of size $N$. For each $i \in Sample \cup Target$, let $I_i$ be the indicator for inclusion in sample (0 for target and 1 for sample) and $X_i$ be a vector of observed covariates. The propensity score is defined as the conditional probability of being included in the sample conditioned on the covariates, $P(I_i=1 | X_i=x)$.

Let $Y_i$ be the potential outcome observed only for $i\in Sample$.

### Assumptions

1. The propensity is bounded away from 0 and 1 (all individuals have a theoretical probability to be in the respondents group): $0<P(I_i=1 | X_i=x)<1$  for all $x$.
2. Ignorability assumption: $({(Y_i(0), Y_i(1))}\perp I_i) | X_i$, where $Y_i(0)$ indicates the response of unit $i$ if it is from the sample, and $Y_i(1)$ indicates the hypothetical response of unit $i$ if it is from the target population. Rosenbaum and Rubin (1983) [2] showed that this assumption implies that the outcome is independent of the inclusion in the sample given the (theoretical) propensity score (this is the "dimension reduction" property of the propensity score). I.e.: $({(Y_i(0), Y_i(1))}\perp I_i) | P(I_i=1 | X_i=x)$.


### Recap - Propensity score estimation

Using a logistic regression model, the propensity score is modeled by: $\pi _\beta(X_i)=P(I_i=1|X_i=x)=\frac{\exp(X_i ^T \beta)}{1+\exp(X_i ^T \beta)}$ for all $i \in Sample$.

This is estimated by maximizing the log-likelihood, which results in:
$$
\hat{\beta}_{MLE}=\arg\max_\beta \sum_{i=1}^n I_i\log(\pi_\beta(X_i))+(1-I_i)\log(1-\pi_\beta(X_i))
$$

which implies the first order condition:
$$
\frac{1}{n}\sum_{i=1}^n \left[ \frac{I_i\pi^\prime_\beta(X_i)}{\pi_\beta(X_i)} +\frac{(1-I_i)\pi^\prime_\beta(X_i)}{1-\pi_\beta(X_i)}\right]=0
$$
where the derivative of $\pi$ is by $\beta^T$.
This condition can be viewed as a condition that balances a certain function of the covariates, in this case the derivative of the propensity score $\pi^\prime_\beta(X_i)$.

### CBPS

Generally, we can expand the above to hold for any function $f$: $\mathbb{E} \left\{ \frac{I_if(X_i)}{\pi_\beta(X_i)} +\frac{(1-I_i)f(X_i)}{1-\pi_\beta(X_i)}\right\} =0$ (given the expectation exists). CBPS chooses $f(x)=x$ as the balancing function $f$ in addition to the traditional logistic regression condition (this is what implemented in R and in balance), but generally any function the researcher may choose could be used here. The function $f(x)=x$ results in balancing the first moment of each covariate


### Estimation of CBPS
The estimation is done by using **Generalized Methods of Moments [(GMM)](https://en.wikipedia.org/wiki/Generalized_method_of_moments):**
Given moments conditions of the form $\mathbb{E}\{g(X_i,\theta)\}=0$, the optimal solution minimizes the norm of the sample analog, $\frac{1}{n}\sum_{i=1}^n g(x_i,\theta)$, with respect to $\theta$. This results in an estimator of the form:
$$
\hat{\theta}=\arg\min_\theta \frac{1}{n}\sum_{i=1}^n g^T(x_i,\theta)Wg(x_i,\theta),
$$
where $W$ is semi-definite positive matrix, often chosen to be the variance matrix $W(\theta)=\left(\frac{1}{n}\sum_{i=1}^n g(x_i,\theta)g^T(x_i,\theta)\right)^{-1}$ (which is unknown).
This can be solved by iterative algorithm, by starting with $\hat{W}=I$, computing $\hat{\theta}$ and $W(\hat{\theta})$, and so on (for two-step GMM, we stop after optimizing to $\hat{\theta}$). Alternatively, it can be solved by *Continuously updating GMM* algorithm, which estimate $\hat{\theta}$ and $\hat{W}$ on the same time.
The model is over-identified if the number of equations is larger than the number of parameters.

For CBPS, the sample analog for the covariate balancing moment condition is:
$\frac{1}{n}\sum_{i=1}^n\left[\frac{I_ix_i}{\pi_\beta(x_i)} +\frac{(1-I_i)x_i}{1-\pi_\beta(x_i)}\right]$, which can be written as $\frac{1}{n}\sum_{i=1}^n\frac{I_i-\pi_\beta(X_i)}{\pi_\beta(x_i)(1-\pi_\beta(x_i))}x_i=0$ (for $I_i\in\{0,1\}$).

Let
$$
g_i(I_i,X_i)=\left(\begin{matrix}   \frac{I_i-\pi_\beta(X_i)}{\pi_\beta(X_i)(1-\pi_\beta(X_i))}\pi^\prime_\beta(X_i)\ \\   \frac{I_i-\pi_\beta(X_i)}{\pi_\beta(X_i)(1-\pi_\beta(X_i))} X_i\ \end{matrix}\right)
$$

be the vector representing the moments we would like to solve. This contains the two conditions of maximizing the log-likelihood and balancing the covariates. Note that this is over-identified, since the number of equations is larger then the number of parameters. Another option is to consider the “just-identified” ("exact") CBPS, where we consider only the covariate balancing conditions and not the propensity score condition.

Using GMM, we have
$$
\hat{\beta}=\arg\min_\beta \bar{g}^T \Sigma^{-1}_\beta \bar{g}
$$
where $\bar{g}=\frac{1}{n}\sum_{i=1}^n g_i$ and  $\Sigma_\beta=\mathbb{E}\left[\frac{1}{n}\sum_{i=1}^n g_i g^T_i | X_i\right]$, which can be estimated by
$$
\hat{\Sigma}_\beta=\frac{1}{n}\sum_{i=1}^n \left(
    \begin{matrix}
    \pi_\beta(X_i)(1-\pi_\beta(X_i))X_iX_i^T &
    X_i X_i^T \\
    X_iX_i^T &
    \pi_\beta(X_i)(1-\pi_\beta(X_i))X_iX_i^T
    \end{matrix}
\right)
$$

To optimize this, we use the two-step GMM, using gradient-based optimization, starting with $\beta^{MLE}$ (from the original logistic regression):

1. $\beta_0=\hat{\beta}_{MLE}$
2. $\hat{W}_0=\Sigma_{\beta_0}^{-1}$
3. $\hat{\beta}=\arg\min_\beta \bar{g}^T\hat{W}_{\hat{\beta}_0}\bar{g}$ - use gradient based optimization

## References
[1] Imai, K., & Ratkovic, M. (2014). Covariate balancing propensity score. *Journal of the Royal Statistical Society: Series B: Statistical Methodology*, 243-263.

[2] PAUL R. ROSENBAUM, DONALD B. RUBIN, The central role of the propensity score in observational studies for causal effects, Biometrika, Volume 70, Issue 1, April 1983, Pages 41–55, https://doi.org/10.1093/biomet/70.1.41
