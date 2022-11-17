---
id: poststratify
title: Post-Stratification
description: Post-Stratification
sidebar_position: 3
keywords:
    - Post-Stratification
    - poststratify
---
## Introduction

Post-stratification is one of the most common weighing approaches in survey statistics. It origins from a stratified sample, where the population is divided into subpopulations (strata) and the sample is conducted independently on each of them. However, when one doesn't know in advance the subpopulations to sample from (for example, when the stratum of the units in the sample is unknown in advance), or when non-response is presented, stratification can be done after the sample has been selected.

The goal of post-stratification is to have the sample match exactly the joint-distribution of the target population. However, this is also the main limitation of this method. It is limited by the number of variables we are able to use for adjustment due to the nature of fitting the target exactly, and thus require a minimal number of respondent in each strata. Hence, usually at most 2 to 4 variables are used (with limited number of buckets). In addition, continues variables cannot be used for adjustment (unless bucketed). A more general approach is the inverse propensity score weighting ([ipw](../ipw)).

## Methodology
The idea behind post-stratification is simple. For each cell (strata) in the population, compute the percent of the total population in this cell. Then fit weights so that they adjust the sample so to have the same proportions for each strata as in the population.

We will illustrate this with an example. Assume that we have sampled people from a certain population to a survey and asked for their age and gender so to use these for weighing. Assume also that the joint distribution of age and gender in this population is known from a census, and is the following:

|        | Young adults | Adults | Total |
|--------|--------------|--------|-------|
| Female | 120          | 380    | 500   |
| Male   | 80           | 420    | 500   |
| Total  | 200          | 800    | 1000  |


In addition, assume that for the specific survey we ran young adults tend to reply more, so that the distribution of responses in the survey is the following:

|        | Young adults | Adults | Total |
|--------|--------------|--------|-------|
| Female | 30           | 10     | 40    |
| Male   | 50           | 10     | 60    |
| Total  | 80           | 20     | 100   |

The post-stratification weights are then computed as follows:

- Proportion of Female young adults in the population is $120/1000 = 0.12$
- Proportion of Female young adults in the sample is $30/100 = 0.3$

Inflation factor - this is the inverse probability factor indicating by how much we need to multiply the total sample size to get to the total population size. It is equal to population size / sample size. In our case it is: $1000/100 = 10$.

Calculate weights for each Female young adult in the sample: (population %) / (sample %) * (inflation factor). In our example this is: $0.12/0.3 * 10= 0.4 * 10= 4$.

This means that the assigned weight of each Female young adult in the sample is 4.

Similarly, we can compute the weight for people from each cell in the table:

|        | Young adults         | Adults              |
|--------|----------------------|---------------------|
| Female | $0.12/0.3 * 10 = 4$  | $0.38/0.1 * 10 = 38$|
| Male   | $0.08/0.5 * 10 = 1.6$| $0.42/0.1 *10 = 42$ |



## References
- More about post-stratification: [Introduction to post-stratification](https://docs.wfp.org/api/documents/WFP-0000121326/download/)
- Kolenikov, Stas. 2016. “Post-Stratification or Non-Response Adjustment?” Survey Practice 9 (3). https://doi.org/10.29115/SP-2016-0014.
