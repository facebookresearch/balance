[![balance_logo_horizontal](https://raw.githubusercontent.com/facebookresearch/balance/main/website/static/img/balance_logo/PNG/Horizontal/balance_Logo_Horizontal_FullColor_RGB.png)](https://import-balance.org/)


# *balance*: a python package for balancing biased data samples

*balance* is currently **in beta** and under active development. Follow us [on github](https://github.com/facebookresearch/balance)!

## What is *balance*?

**[*balance*](https://import-balance.org/) is a Python package** offering a simple workflow and methods for **dealing with biased data samples** when looking to infer from them to some population of interest.

Biased samples often occur in [survey statistics](https://en.wikipedia.org/wiki/Survey_methodology) when respondents present [non-response bias](https://en.wikipedia.org/wiki/Participation_bias) or survey suffers from [sampling bias](https://en.wikipedia.org/wiki/Sampling_bias) (that are not [missing completely at random](https://en.wikipedia.org/wiki/Missing_data#Missing_completely_at_random)). A similar issue arises in [observational studies](https://en.wikipedia.org/wiki/Observational_study) when comparing the treated vs untreated groups, and in any data that suffers from selection bias.

Under the missing at random assumption ([MAR](https://en.wikipedia.org/wiki/Missing_data#Missing_at_random)), bias in samples could sometimes be (at least partially) mitigated by relying on auxiliary information (a.k.a.: “covariates” or “features”) that is present for all items in the sample, as well as present in a sample of items from the population. For example, if we want to infer from a sample of respondents to some survey, we may wish to adjust for non-response using demographic information such as age, gender, education, etc. This can be done by weighing the sample to the population using auxiliary information.

The package is intended for researchers who are interested in balancing biased samples, such as the ones coming from surveys, using a Python package. This need may arise by survey methodologists, demographers, UX researchers, market researchers, and generally data scientists, statisticiains, and machine learners.


# Installation

## Requirements
You need Python 3.8 or later to run *balance*. *balance* can be built and run
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

Note that glmnet_python must be installed from the [Github source](https://github.com/bbalasub1/glmnet_python)

See [setup.py](https://github.com/facebookresearch/balance/blob/main/setup.py) for more details.

## Installing *balance*
As a prerequisite, you must install glmnet_python from source:
```
python -m pip install git+https://github.com/bbalasub1/glmnet_python.git@1.0
```

### Installing via PyPi
We recommend installing *balance* from PyPi via pip for the latest stable version:

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

The core workflow in [*balance*](https://import-balance.org/) deals with fitting and evaluating weights to a sample. For each unit in the sample (such as a respondent to a survey), balance fits a weight that can be (loosely) interpreted as the number of people from the target population that this respondent represents. This aims to help mitigate the coverage and non-response biases, as illustrated in the following figure.

![total_survey_error_img](https://raw.githubusercontent.com/facebookresearch/balance/main/website/docs/docs/img/total_survey_error_flow_v02.png?token=GHSAT0AAAAAAB25KSTWSBZGTWAJ7LJ3U3G6Y3VG4XA)


The weighting of survey data through *balance* is done in the following main steps:

1. Loading data of the respondents of the survey.
2. Loading data about the target population we would like to correct for.
3. Diagnostics of the sample covariates so to evaluate whether weighting is needed.
4. Adjusting the sample to the target.
5. Evaluation of the results.
6. Use the weights for producing population level estimations.
7. Saving the output weights.

You can see a step-by-step description (with code) of the above steps in the [General Framework](https://import-balance.org/docs/docs/general_framework/) page.

## Code example of using *balance*

You may run the following code to play with *balance*'s basic workflow (these are snippets taken from the [quickstart tutorial](https://import-balance.org/docs/tutorials/quickstart/)):

We start by loading data, and adjusting it:

```python
from balance import load_data, Sample

# load simulated example data
target_df, sample_df = load_data()

# Import sample and target data into a Sample object
sample = Sample.from_frame(sample_df, outcome_columns=["happiness"])
target = Sample.from_frame(target_df)

# Set the target to be the target of sample
sample_with_target = sample.set_target(target)

# Check basic diagnostics of sample vs target before adjusting:
# sample_with_target.covars().plot()

```

*You can read more on evaluation of the pre-adjusted data in the [Pre-Adjustment Diagnostics](https://import-balance.org/docs/docs/general_framework/pre_adjustment_diagnostics/) page.*

Next, we adjust the sample to the population by fitting balancing survey weights:

```python
# Using ipw to fit survey weights
adjusted = sample_with_target.adjust(max_de=None)
```

*You can read more on adjustment process in the [Adjusting Sample to Population](https://import-balance.org/docs/docs/general_framework/adjusting_sample_to_population/) page.*

The above code gets us an `adjusted` object with weights. We can evaluate the benefit of the weights to the coveriate balance, for example by running:

```python
print(adjusted.summary())
    # Covar ASMD reduction: 62.3%, design effect: 2.249
    # Covar ASMD (7 variables):0.335 -> 0.126
    # Model performance: Model proportion deviance explained: 0.174

adjusted.covars().plot(library = "seaborn", dist_type = "kde")
```

And get:

![](https://import-balance.org/assets/images/fig_07_seaborn_after-ac7514f6b150f431b36329bb9ebd9d0a.png)

We can also check the impact of the weights on the outcome using:

```python
# For the outcome:
print(adjusted.outcomes().summary())
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
adjusted.outcomes().plot()
```
![](https://import-balance.org/assets/images/fig_09_seaborn_outcome_kde_after-26fa9668164349253b2614335961ade9.png)

*You can read more on evaluation of the post-adjusted data in the [Evaluating and using the adjustment weights](https://import-balance.org/docs/docs/general_framework/evaluation_of_results/) page.*


Finally, the adjusted data can be downloded using:
```python
adjusted.to_download()  # Or:
# adjusted.to_csv()
```

To see a more detailed step-by-step code example with code output prints and plots (both static and interactive), please go over to the [quickstart tutorial](https://import-balance.org/docs/tutorials/quickstart/).


## Implemented methods for adjustments

*balance* currently implements various adjustment methods. Click the links to learn more about each:

1. [Logistic regression using L1 (LASSO) penalization.](https://import-balance.org/docs/docs/statistical_methods/ipw/)
2. [Covariate Balancing Propensity Score (CBPS).](https://import-balance.org/docs/docs/statistical_methods/cbps/)
3. [Post-stratification.](https://import-balance.org/docs/docs/statistical_methods/poststratify/)

## Implemented methods for diagnostics/evaluation

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

*You can read more on evaluation of the post-adjusted data in the [Evaluating and using the adjustment weights](https://import-balance.org/docs/docs/general_framework/evaluation_of_results/) page.*

# More details

## Getting help, submitting bug reports and contributing code

You are welcome to:

* Learn more in the [*balance*](https://import-balance.org/) website.
* Ask for help on: https://stats.stackexchange.com/questions/tagged/balance
* Submit bug-reports and features' suggestions at: https://github.com/facebookresearch/balance/issues
* Send a pull request on: https://github.com/facebookresearch/balance. See the [CONTRIBUTING](https://github.com/facebookresearch/balance/blob/main/CONTRIBUTING.md) file for how to help out. And our [CODE OF CONDUCT](https://github.com/facebookresearch/balance/blob/main/LICENSE-DOCUMENTATION) for our expectations from contributors.

## Citing *balance*

**TODO**: TBD.

## License
The *balance* package is licensed under the [GPLv2 license](https://github.com/facebookresearch/balance/blob/main/LICENSE), and all the documentation on the site is under [CC-BY](https://github.com/facebookresearch/balance/blob/main/LICENSE-DOCUMENTATION).

# News

**TODO**: TBD.

## Acknowledgements / People

The *balance* package is actively maintained by people from the [Core Data Science](https://research.facebook.com/teams/core-data-science/) team (in Tel Aviv and Boston), by [Tal Sarig](https://research.facebook.com/people/sarig-tal/), [Tal Galili](https://research.facebook.com/people/galili-tal/) and [Steve Mandala](https://research.facebook.com/people/mandala-steve/).

The *balance* package was (and is) developed by many people, including: [Roee Eilat](https://research.facebook.com/people/eilat-roee/), [Tal Galili](https://research.facebook.com/people/galili-tal/), [Daniel Haimovich](https://research.facebook.com/people/haimovich-daniel/), [Kevin Liou](https://www.linkedin.com/in/kevinycliou), [Steve Mandala](https://research.facebook.com/people/mandala-steve/), [Adam Obeng](https://adamobeng.com/) (author of the initial internal Meta version), [Tal Sarig](https://research.facebook.com/people/sarig-tal/),  [Luke Sonnet](https://www.linkedin.com/in/luke-sonnet), [Sean Taylor](https://seanjtaylor.com), [Barak Yair Reif](https://www.linkedin.com/in/barak-yair-reif-2154365/), and others. If you worked on balance in the past, please email us to be added to this list.

The *balance* package was open-sourced by [Tal Sarig](https://research.facebook.com/people/sarig-tal/), [Tal Galili](https://research.facebook.com/people/galili-tal/) and [Steve Mandala](https://research.facebook.com/people/mandala-steve/) in late 2022.

Branding created by [Dana Beaty](https://www.danabeaty.com/), from the Meta AI Design and Marketing Team.
