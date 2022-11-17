[![balance_logo_horizontal](https://raw.githubusercontent.com/facebookresearch/balance/main/website/static/img/balance_logo/PNG/Horizontal/balance_Logo_Horizontal_FullColor_RGB.png)](https://import-balance.org/)


# *balance*: a python package for balancing biased data samples

*balance is currently in beta and under active development!*

## What is *balance*?

***balance* is a Python package** offering a simple workflow and methods for **dealing with biased data samples** when looking to infer from them to some population of interest.

Biased samples often occur in [survey statistics](https://en.wikipedia.org/wiki/Survey_methodology) when respondents present [non-response bias or survey suffers from sampling bias](https://en.wikipedia.org/wiki/Sampling_bias) (that are not [missing completely at random](https://en.wikipedia.org/wiki/Missing_data#Missing_completely_at_random)). A similar issue arises in [observational studies](https://en.wikipedia.org/wiki/Observational_study) when comparing the treated vs untreated groups, and in any data that suffers from selection bias.

Under the missing at random assumption ([MAR](https://en.wikipedia.org/wiki/Missing_data#Missing_at_random)), bias in samples could sometimes be (at least partially) mitigated by relying on auxiliary information (a.k.a.: “covariates” or “features”) that is present for all items in the sample, as well as present in a sample of items from the population. For example, if we want to infer from a sample of respondents to some survey, we may wish to adjust for non-response using demographic information such as age, gender, education, etc. This can be done by weighing the sample to the population using auxiliary information.

The package is intended for researchers who are interested in balancing biased samples, such as the ones coming from surveys, using a Python package. This need may arise by survey methodologists, demographers, UX researchers, market researchers, and generally data scientists, statisticiains, and machine learners.


# Installation

## Requirements
You need Python 3.8 or later to run balance. balance can be built and run
from OSX, Linux, and Windows

The required Python dependencies are:
```python
REQUIRES = [
    "numpy",
    "pandas<=1.4.3",
    "ipython",
    "scipy<=1.8.1",
    "patsy",
    "seaborn<=0.11.1",
    "plotly",
    "matplotlib",
    "statsmodels",
    "scikit-learn",
    "ipfn",
]
```

Note that glmnet_python must be installed from the [Github source](https://github.com/bbalasub1/glmnet_python.git@1.0)

See [setup.py](https://github.com/facebookresearch/balance/blob/main/setup.py) for more details. **TODO**: add details on using setup.py.

## Installing balance
As a prerequisite, you must install glmnet_python from source:
```
python -m pip install git+https://github.com/bbalasub1/glmnet_python.git@1.0
```

### Installing via PyPi
We recommend installing balance from PyPi via pip for the latest stable version:

```
python -m pip install balance
```

Installation will use Python wheels from PyPI, available for [OSX, Linux, and Windows](https://pypi.org/project/balance/#files).

### Installing from Source/Git

You can install the latest (bleeding edge) version from Git:

```
python -m pip install git+https://github.com/facebookresearch/balance.git
```

Alternatively, if you have a local clone of the repo:

```
cd balance
python -m pip install .
```


# Getting started

## balance’s workflow in high-level

The core workflow in balance deals with fitting and evaluating weights to a sample. For each unit in the sample (such as a respondent to a survey), balance fits a weight that can be (loosely) interpreted as the number of people from the target population that this respondent represents. This aims to help mitigate the coverage and non-response biases, as illustrated in the following figure.

![total_survey_error_img](https://raw.githubusercontent.com/facebookresearch/balance/main/website/docs/docs/img/total_survey_error_image.png?token=GHSAT0AAAAAAB25KSTWSBZGTWAJ7LJ3U3G6Y3VG4XA)


The weighting of survey data through balance is done in the following main steps:

1. Loading data of the respondents of the survey.
2. Loading data about the target population we would like to correct for.
3. Diagnostics of the sample covariates so to evaluate whether weighting is needed.
4. Adjusting the sample to the target.
5. Evaluation of the results.
6. Use the weights for producing population level estimations.
7. Saving the output weights.

**TODO**: add a simple chart that describes the flow

**TODO**: link to the quick start tutorial

## Code example of using balance

You may run the following code to play with balance's basic workflow:

```python
from balance import load_data, Sample

# load simulated example data
target_df, sample_df = load_data()

# Import dample and target data into a Sample object
sample = Sample.from_frame(sample_df, outcome_columns=["happiness"])
target = Sample.from_frame(target_df)

# Set the target to be the target of sample
sample_with_target = sample.set_target(target)

# Check basic diagnostics of sample vs target before adjusting:
sample_with_target.covars().mean().T
sample_with_target.covars().asmd().T
sample_with_target.covars().plot()

# Using ipw to fit survey weights
adjust = sample_with_target.adjust(max_de=None)

print(adjust.summary())
# Covar ASMD reduction: 62.3%, design effect: 2.249
# Covar ASMD (7 variables):0.335 -> 0.126
# Model performance: Model proportion deviance explained: 0.174


# A detailed diagnostics is available for after the adjustment

# For covars:
adjust.covars().covars().mean().T
adjust.covars().asmd().T
adjust.covars().plot()  # interactive plots
adjust.covars().plot(library = "seaborn", dist_type = "kde")  # static plots

# For weights:
adjust.weights().plot()
adjust.weights().design_effect()

# For the outcome:
print(adjust.outcomes().summary())
# 1 outcomes: ['happiness']
# Mean outcomes:
#             happiness
# source
# self        54.221388
# unadjusted  48.392784
#
# Response rates (relative to number of respondents in sample):
#    happiness
# n     1000.0
# %      100.0
adjust.outcomes().plot()

# Finally, the adjusted data can be downloded using:
adjust.to_download()
adjust.to_csv()
```

To see the full output of the code above, please go over to **TODO**: add link.


## Implemented methods for adjustments

balance currently implements various adjustment methods.

For weight adjustment, it uses [inverse probability/propensity weighting](https://en.wikipedia.org/wiki/Inverse_probability_weighting) (IPW) with:

**TODO**: link to the website links instead of the ones we have below.
**TODO:** Update descriptions according the adjustment page
1. Logistic regression using L1 (LASSO) penalization.
2. Covariate Balancing Propensity Score (CBPS).
3. Post-stratification.

## Implemented methods for diagnostics/evaluation

**TODO**: link to the website links instead of the ones we have below.
For diagnostics the main tools (comparing before, after applying weights, and the target population) are:

1. Plots
    1. barplots
    2. density plots (for weights and covariances)
    3. qq-plots
2. Statistical summaries
    1. Weights distributions
        1. [Kish’s design effect](https://en.wikipedia.org/wiki/Design_effect#Haphazard_weights_with_estimated_ratio-mean_(%7F'%22%60UNIQ--postMath-0000003A-QINU%60%22'%7F)_-_Kish's_design_effect)
        2. Main summaries (mean, median, variances, quantiles)
    2. Covariate distributions
        1. Absolute Standardized Mean Difference (ASMD). For continuous variables, it is [Cohen's d](https://en.wikipedia.org/wiki/Effect_size#Cohen's_d). Categorical variables are one-hot encoded, Cohen's d is calculated for each category and ASMD for a categorical variable is defined as Cohen's d, average across all categories.

# More details

## Getting help, submitting bug reports and contributing code

You are welcome to:

* Ask for help in: https://stats.stackexchange.com/questions/tagged/balance
* Submit bug-reports and features' suggestions at: https://github.com/facebookresearch/balance/issues
* Send a pull request on: https://github.com/facebookresearch/balance. See the [CONTRIBUTING](https://github.com/facebookresearch/balance/blob/main/CONTRIBUTING.md) file for how to help out. And our [CODE OF CONDUCT](https://github.com/facebookresearch/balance/blob/main/LICENSE-DOCUMENTATION) for our expectations from contributors.

## Citing *balance*

**TODO**: Update.

## License
The *balance* package is licensed under the [GPLv2 license](https://github.com/facebookresearch/balance/blob/main/LICENSE), and all the documentation on the site is under [CC-BY](https://github.com/facebookresearch/balance/blob/main/LICENSE-DOCUMENTATION).

# News

**TODO**: Link to the NEWS.md file

## Acknowledgements / People

The *balance* package is actively maintained by people from the [Core Data Science](https://research.facebook.com/teams/core-data-science/) team (in Tel Aviv and Boston), by [Tal Sarig](https://research.facebook.com/people/sarig-tal/), [Tal Galili](https://research.facebook.com/people/galili-tal/) and [Steve Mandala](https://research.facebook.com/people/mandala-steve/).

The *balance* package was (and is) developed by many people, including: Adam Obeng, Kevin Liou, Sean Taylor, [Daniel Haimovich](https://research.facebook.com/people/haimovich-daniel/), [Luke Sonnet](https://lukesonnet.com/), [Tal Sarig](https://research.facebook.com/people/sarig-tal/), [Tal Galili](https://research.facebook.com/people/galili-tal/), [Roee Eilat](https://research.facebook.com/people/eilat-roee/), [Barak Yair Reif](https://www.linkedin.com/in/barak-yair-reif-2154365/?originalSubdomain=il), [Steve Mandala](https://research.facebook.com/people/mandala-steve/). and others.

The *balance* package was open-sourced by [Tal Sarig](https://research.facebook.com/people/sarig-tal/), [Tal Galili](https://research.facebook.com/people/galili-tal/) and [Steve Mandala](https://research.facebook.com/people/mandala-steve/) in late 2022.
