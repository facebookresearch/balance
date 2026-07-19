---
id: rake
title: Raking
description: rake
sidebar_position: 5
keywords:
    - rake
    - raking
---
## Introduction

Raking, also known as iterative proportional fitting, is a statistical technique widely used in survey sampling to adjust weights and enhance the representativeness of the collected data. When a sample is drawn from a population, there might be differences in the distribution of certain variables between the sample and the population. Raking, similar to other methods in the `balance` package, helps to account for these differences, making the sample's distribution closely resemble that of the population.

Raking is an iterative process that involves adjusting the weights of sampled units based on the marginal distributions of certain variables in the population. Typically, we have access to such marginal distributions, but not their combined joint distribution. The variables chosen for raking are usually demographic variables, such as age, gender, education, income, and other socioeconomic variables, which are known to influence survey outcomes. By adjusting the weights of the sampled units, raking helps to correct for potential biases that may arise due to nonresponse, undercoverage, or oversampling of certain groups.


## Methodology

Raking essentially applies [post-stratification](https://import-balance.org/docs/docs/statistical_methods/poststratify/) repeatedly over all the covariates. For example, we may have the marginal distribution of age\*gender and education. Raking would first adjust weights to match the age\*gender distribution and then take these weights as input to adjust for education. It would then adjust again to age\*gender and then again to education, and so forth. This process will repeat until either a max_iteration is met, or the weights have converged and no longer seem to change from one iteration to another.

Raking is a valuable technique for addressing potential biases and enhancing the representativeness of survey data. By iteratively adjusting the weights of sampled units based on the marginal distribution of key variables, raking ensures that survey estimates are more accurate and reliable.

You can see a detailed example of how to perform raking in `balance` in the tutorial: [**quickstart_rake**](https://import-balance.org/docs/tutorials/quickstart_rake/).

## Diagnostics example

`BalanceFrame.fit(method="rake")` stores fit metadata by default, so its
diagnostics table includes compact `model_glance` rows. The complete example
below uses the package's example data and removes missing adjustment values.

```python
from balance import load_data
from balance.balance_frame import BalanceFrame
from balance.sample_frame import SampleFrame

target_df, sample_df = load_data()
variables = ["gender", "age_group"]
sample_cells = sample_df.dropna(subset=variables).assign(weight=1.0)
target_cells = target_df.dropna(subset=variables).assign(weight=1.0)

sample = SampleFrame.from_frame(
    sample_cells[["id", "weight", *variables]],
    id_column="id",
    weight_column="weight",
)
target = SampleFrame.from_frame(
    target_cells[["id", "weight", *variables]],
    id_column="id",
    weight_column="weight",
)
adjusted_rake = BalanceFrame(sample=sample, target=target).fit(
    method="rake",
    variables=variables,
    transformations=None,
)
rake_glance = adjusted_rake.diagnostics().query("metric == 'model_glance'")
display(rake_glance[["var", "val"]].round(4))
```

Example output:

| var | val |
| --- | ---: |
| converged | 1 |
| iterations | 3 |
| final_conv | 0.0001 |
| n_variables | 2 |

`converged` is `1` when iterative proportional fitting converged, `iterations`
is the number of stored convergence-history rows, `final_conv` is the last
recorded convergence value, and `n_variables` is the number of raking variables
recorded in the model metadata.

## References
- https://en.wikipedia.org/wiki/Raking
- https://www.pewresearch.org/methods/2018/01/26/how-different-weighting-methods-work/
- Practical Considerations in Raking Survey Data ([url](https://www.surveypractice.org/article/2953-practical-considerations-in-raking-survey-data))
