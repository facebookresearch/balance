---
id: adjusting_sample_to_population
title: Adjusting Sample to Population
description: How to produce weights for a sample to represent the target population of interest
sidebar_position: 2
keywords:
    - adjustment
---

To produce the balancing weights, use the ```Sample.adjust()``` method to adjust a sample to population:

```
adjusted = sample.adjust()
```
The output of this method is an adjusted `Sample` class object of the form:

```
    Adjusted balance Sample object with target set using ipw
    1000 observations x 3 variables: gender,age_group,income
    id_column: id, weight_column: weight,
    outcome_columns: happiness

        target:

            balance Sample object
            10000 observations x 3 variables: gender,age_group,income
            id_column: id, weight_column: weight,
            outcome_columns: None

        3 common variables: income,gender,age_group
```
Note that the `adjust` method in balance is performing three main steps:
1. **Pre-processing** of the data - getting data ready for adjustment using best practices in the field:
    * Handling missing values - balance handles missing values automatically by adding a column '_is_na' to any variable that contains missing values. The advantage of this is that these are then considered as a separate category for the adjustment.
    * Feature engineering -  by default, balance applies feature engineering to be able to fit the covariate distribution better, and not only the first moment. Specifically, each continues variable is bucketed into 10 quantiles buckets. Furthermore, rare categories in categorical variables are grouped together so to avoid overfitting rare events.
2. **Fitting the model** and calculating the weights: the model fitted depends on the ```method``` chosen by the user. Current options are [inverse propensity score weighting](../statistical_methods/ipw.md) using regularized logistic regression (```ipw```), [covariate balancing propensity score](../statistical_methods/cbps.md) (```cbps```), [post-stratification](../statistical_methods/poststratify.md) (```poststratify```), and [raking](../statistical_methods/rake.md) (```rake```).
3. **Post-processing** of the weights:
    * Trimming weights - balance trims the weights in order to avoid over fitting of the model and unnecessary variance inflation.
    * Normalizing weights to population size. The resulting weights of balance can be described as approximating the number of unit in the population this unit of the sample represents.

## Optional arguments

* **`method`**: `ipw`, `poststratify`, `rake`, or `cbps`.  Default is `ipw`.
    * `ipw`: stands for [Inverse Propensity Weighting](https://en.wikipedia.org/wiki/Inverse_probability_weighting). The propensity scores are calculated with [LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics)) [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression).  Details about the implementation can be found [here](../../statistical_methods/ipw/). For a quick-start tutorial, see [here](https://import-balance.org/docs/tutorials/quickstart/).
   * `cbps`: stands for [Covariate Balancing Propensity Score](https://imai.fas.harvard.edu/software/CBPS.html). The CBPS algorithm estimates the propensity score in a way that optimizes prediction of the probability of sample inclusion as well as the covariates balance. Its main advantage is in cases when the researcher wants better balance on the covariates than traditional propensity score methods - because one believes the assignment model might be misspecified and would like to avoid an iterative procedure of balancing the covariates. Details about the implementation can be found [here](../../statistical_methods/cbps/). For a quick-start tutorial, see [here](https://import-balance.org/docs/tutorials/quickstart_cbps/).
   * `poststratify`: stands for post-stratification. Details about the implementation can be found [here](../../statistical_methods/poststratify/).
   * `rake`: Details about the implementation can be found [here](../../statistical_methods/rake/). For a quick-start tutorial, see [here](https://import-balance.org/docs/tutorials/quickstart_rake/).

* **`variables`**: allows user to pass a list of the covariates that they want to adjust for; if variables argument is not specified, all joint variables in sample and target are used.

* **`transformations`**: which transformations to apply to data before fitting the model. Default is cutting numeric variables into 10 quantile buckets and lumping together infrequent levels with less than 5% prevalence into `lumped_other` category. The transformations are done on both the sample dataframe and the target dataframe together. User can also specify specific transformations in a dictionary format. For a quick-start tutorial on transformations and formulas, see [here](https://import-balance.org/docs/tutorials/balance_transformations_and_formulas/).

* **`max_de`**: (for `ipw` and `cbps` methods): The default value is 1.5. It limits the [**design effect**](https://en.wikipedia.org/wiki/Design_effect) to be within 1.5. If set to None, the optimization is performed by cross-validation of the logistic model for ipw (see the `choose_regularization` function for more details) or without constrained optimization for cbps. Setting `max_de` to `None` can sometimes significantly improve the running time of the code.

* **`weight_trimming_mean_ratio`** **or** **`weight_trimming_percentile`**: (only one of these arguments can be specified). `weight_trimming_mean_ratio` indicates the ratio from above according to which the weights are trimmed by mean(weights) * ratio. Default is 20. If `weight_trimming_percentile` is not none, [winsorization](https://en.wikipedia.org/wiki/Winsorizing) is applied. Default is None, i.e. trimming from above is applied. However, note that when `max_de` is not None (and default is 1.5), the trimming-ratio is optimized by `ipw` and these arguments are ignored.

* **`na_action`** (for `ipw` method): how to handle missing values in the data (sample and target). Default is to replace NAs with 0's and add indicator for which observations were NA (this is done after applying the transformations). Another option is `drop`, which drops all observations with NA values.

* **`formula`** (for `ipw` and `cbps` methods): The formula according to which build the model matrix for the logistic regression. Default is a linear additive formula of all covariates. For a quick-start tutorial on transformations and formulas, see [here](https://import-balance.org/docs/tutorials/balance_transformations_and_formulas/).

* **`penalty_factor`** (for `ipw` method): the penalty used in the regularized logistic regression.
