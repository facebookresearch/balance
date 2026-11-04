[![balance_logo_horizontal](https://raw.githubusercontent.com/facebookresearch/balance/main/website/static/img/balance_logo/PNG/Horizontal/balance_Logo_Horizontal_FullColor_RGB.png)](https://import-balance.org/)

# _balance_: a python package for balancing biased data samples

<div align="center">

[![Current Release](https://img.shields.io/github/release/facebookresearch/balance.svg)](https://github.com/facebookresearch/balance/releases)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-fcbc2c.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Build & Test](https://github.com/facebookresearch/balance/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/facebookresearch/balance/actions/workflows/build-and-test.yml?query=branch%3Amain)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/talgalili/89d05034d314ebda47c1e16607e1ee22/raw/coverage-balance.json)](https://github.com/facebookresearch/balance/actions/workflows/coverage.yml?query=branch%3Amain)
[![CodeQL](https://github.com/facebookresearch/balance/actions/workflows/codeql.yml/badge.svg)](https://github.com/facebookresearch/balance/actions/workflows/codeql.yml?query=branch%3Amain)
[![Deploy Website](https://github.com/facebookresearch/balance/actions/workflows/deploy-website.yml/badge.svg)](https://github.com/facebookresearch/balance/actions/workflows/deploy-website.yml?query=branch%3Amain)
[![Release](https://github.com/facebookresearch/balance/actions/workflows/release.yml/badge.svg)](https://github.com/facebookresearch/balance/actions/workflows/release.yml?query=branch%3Amain)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2307.06024-blue.svg)](https://doi.org/10.48550/arXiv.2307.06024)
[![Downloads](https://pepy.tech/badge/balance)](https://pepy.tech/project/balance)

</div>

> [!NOTE]
> _balance_ is currently **in beta** and is actively supported. Follow us [on github](https://github.com/facebookresearch/balance).

## What is _balance_?

**[_balance_](https://import-balance.org/) is a Python package** offering a
simple workflow and methods for **dealing with biased data samples** when
looking to infer from them to some population of interest.

Biased samples often occur in
[survey statistics](https://en.wikipedia.org/wiki/Survey_methodology) when
respondents present
[non-response bias](https://en.wikipedia.org/wiki/Participation_bias) or survey
suffers from [sampling bias](https://en.wikipedia.org/wiki/Sampling_bias) (that
are not
[missing completely at random](https://en.wikipedia.org/wiki/Missing_data#Missing_completely_at_random)).
A similar issue arises in
[observational studies](https://en.wikipedia.org/wiki/Observational_study) when
comparing the treated vs untreated groups, and in any data that suffers from
selection bias.

Under the missing at random assumption
([MAR](https://en.wikipedia.org/wiki/Missing_data#Missing_at_random)), bias in
samples could sometimes be (at least partially) mitigated by relying on
auxiliary information (a.k.a.: "covariates" or "features") that is present for
all items in the sample, as well as present in a sample of items from the
population. For example, if we want to infer from a sample of respondents to
some survey, we may wish to adjust for non-response using demographic
information such as age, gender, education, etc. This can be done by weighing
the sample to the population using auxiliary information.

The package is intended for researchers who are interested in balancing biased
samples, such as the ones coming from surveys, using a Python package. This need
may arise by survey methodologists, demographers, UX researchers, market
researchers, and generally data scientists, statisticians, and machine learners.

More about the methodological background can be found in
[Sarig, T., Galili, T., & Eilat, R. (2023). balance – a Python package for balancing biased data samples](https://arxiv.org/abs/2307.06024).

# Installation

## Requirements

You need Python 3.9, 3.10, 3.11, 3.12, 3.13, or 3.14 to run _balance_. _balance_
can be built and run from Linux, OSX, and Windows.

The required Python dependencies are:

```python
REQUIRES = [
    # Numpy and pandas: carefully versioned for binary compatibility
    "numpy>=1.21.0,<2.0; python_version<'3.12'",
    "numpy>=1.24.0; python_version>='3.12'",
    "pandas>=1.5.0,<4.0.0; python_version<'3.12'",
    "pandas>=2.0.0,<4.0.0; python_version>='3.12'",
    # Scientific stack
    "scipy>=1.7.0,<1.14.0; python_version<'3.12'",
    "scipy>=1.11.0; python_version>='3.12'",
    "scikit-learn>=1.0.0,<1.4.0; python_version<'3.12'",
    "scikit-learn>=1.3.0; python_version>='3.12'",
    "ipython",
    "patsy",
    "seaborn",
    "plotly",
    "matplotlib",
    "statsmodels",
    "session-info",
]
```

See
[pyproject.toml](https://github.com/facebookresearch/balance/blob/main/pyproject.toml)
for more details.

## Installing _balance_

### Installing via PyPi

We recommend installing _balance_ from PyPi via pip for the latest stable
version:

```bash
python -m pip install balance
```

Installation will use Python wheels from PyPI, available for
[OSX, Linux, and Windows](https://pypi.org/project/balance/#files).

### Installing from Source/Git

You can install the latest (bleeding edge) version from Git:

```bash
python -m pip install git+https://github.com/facebookresearch/balance.git
```

Alternatively, if you have a local clone of the repo:

```bash
cd balance
python -m pip install .
```

Or using dev-dependencies:

```bash
cd balance
python -m pip install .[dev]
```

# Getting started

## balance's workflow in high-level

The core workflow in [_balance_](https://import-balance.org/) deals with fitting
and evaluating weights to a sample. For each unit in the sample (such as a
respondent to a survey), balance fits a weight that can be (loosely) interpreted
as the number of people from the target population that this respondent
represents. This aims to help mitigate the coverage and non-response biases, as
illustrated in the following figure.

![total_survey_error_img](https://raw.githubusercontent.com/facebookresearch/balance/main/website/docs/docs/img/total_survey_error_flow_v02.png)

The weighting of survey data through _balance_ is done in the following main
steps:

1. Loading data of the respondents of the survey.
2. Loading data about the target population we would like to correct for.
3. Diagnostics of the sample covariates so to evaluate whether weighting is
   needed.
4. Adjusting the sample to the target.
5. Evaluation of the results.
6. Use the weights for producing population level estimations.
7. Saving the output weights.

You can see a step-by-step description (with code) of the above steps in the
[General Framework](https://import-balance.org/docs/docs/general_framework/)
page.

## Code example of using _balance_

You may run the following code to play with _balance_'s basic workflow (these
are snippets taken from the
[quickstart tutorial](https://import-balance.org/docs/tutorials/quickstart/)):

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

_You can read more on evaluation of the pre-adjusted data in the
[Pre-Adjustment Diagnostics](https://import-balance.org/docs/docs/general_framework/pre_adjustment_diagnostics/)
page._

Next, we adjust the sample to the population by fitting balancing survey
weights:

```python
# Using ipw to fit survey weights
adjusted = sample_with_target.adjust()
```

_You can read more on adjustment process in the
[Adjusting Sample to Population](https://import-balance.org/docs/docs/general_framework/adjusting_sample_to_population/)
page._

The above code gets us an `adjusted` object with weights. We can evaluate the
benefit of the weights to the covariate balance, for example by running:

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

_You can read more on evaluation of the post-adjusted data in the
[Evaluating and using the adjustment weights](https://import-balance.org/docs/docs/general_framework/evaluation_of_results/)
page._

Finally, the adjusted data can be downloaded using:

```python
adjusted.to_download()  # Or:
# adjusted.to_csv()
```

To see a more detailed step-by-step code example with code output prints and
plots (both static and interactive), please go over to the
[tutorials section](https://import-balance.org/docs/tutorials/).

## Implemented methods for adjustments

_balance_ currently implements various adjustment methods. Click the links to
learn more about each:

1. [Logistic regression using L1 (LASSO) penalization.](https://import-balance.org/docs/docs/statistical_methods/ipw/)
2. [Covariate Balancing Propensity Score (CBPS).](https://import-balance.org/docs/docs/statistical_methods/cbps/)
3. [Post-stratification.](https://import-balance.org/docs/docs/statistical_methods/poststratify/)
4. [Raking.](https://import-balance.org/docs/docs/statistical_methods/rake/)

## Implemented methods for diagnostics/evaluation

For diagnostics the main tools (comparing before, after applying weights, and
the target population) are:

1. Plots
   1. barplots
   2. density plots (for weights and covariances)
   3. qq-plots
   4. love plot — per-covariate ASMD before-vs-after on a sorted scatter
      with a +0.1 reference cutoff (ASMD is non-negative, so only the
      positive threshold line is drawn), in the spirit of R's
      [`cobalt::love.plot`](https://cran.r-project.org/web/packages/cobalt/vignettes/cobalt.html#love.plot)
      (added in v0.21 via `BalanceDFCovars.love_plot()`)
2. Statistical summaries
   1. Weights distributions
      1. [Kish's design effect](<https://en.wikipedia.org/wiki/Design_effect#Haphazard_weights_with_estimated_ratio-mean_(%7F'%22%60UNIQ--postMath-0000003A-QINU%60%22'%7F)_-_Kish's_design_effect>)
      2. Main summaries (mean, median, variances, quantiles)
   2. Covariate distributions
      1. Absolute Standardized Mean Difference (ASMD). For continuous variables,
         it is [Cohen's d](https://en.wikipedia.org/wiki/Effect_size#Cohen's_d).
         Categorical variables are one-hot encoded, Cohen's d is calculated for
         each category and ASMD for a categorical variable is defined as Cohen's
         d, average across all categories.

_You can read more on evaluation of the post-adjusted data in the
[Evaluating and using the adjustment weights](https://import-balance.org/docs/docs/general_framework/evaluation_of_results/)
page._

## Estimating a population outcome: IPW vs. outcome-model estimators

Once a sample is reweighted to a target, _balance_ offers more than one way to
estimate the target-population mean of an outcome. They use the same covariates
differently, so comparing them is a useful robustness check:

- **μ̂_IPW — inverse-propensity (Hájek) weighted mean:** `outcomes().mean()`
  (the adjusted `self` row). The weighted average of the responders' *observed*
  outcome using the adjusted weights. Consistent when the **weights** (propensity
  model) are correct.
- **μ̂_OM — outcome-model / g-computation estimate:** `outcomes_hat().mean()`
  (the target-population row). Fit a learner `ĝ(X) ≈ E[Y|X]` on the responders,
  apply it to the **target** covariates, and average the predicted outcomes `ŷ_T`
  with the target weights. Consistent when the **outcome model** is correct.
  Added in v0.23.

The two estimators are complementary — agreement between μ̂_IPW and μ̂_OM is
reassuring; disagreement points at a misspecified weighting or outcome model.
A doubly-robust / AIPW estimator that combines both is available as `bf.aipw()`
(point estimate only; optional K-fold cross-fitting is available and variance/CI
remains deferred).

_For a runnable end-to-end walkthrough, see the [outcome-model tutorial](https://import-balance.org/docs/tutorials/outcome_model/)._

```python
from balance import Sample

bf = (
    Sample.from_frame(sample_df, outcome_columns=["happiness"])
    .set_target(Sample.from_frame(target_df))
    .adjust(method="ipw")
)

# μ̂_IPW — weighted mean of the observed outcome (`self` row):
bf.outcomes().mean()

# μ̂_OM — fit ĝ(X) on responders, apply to the target, average ŷ_T (target row):
bf.fit_outcome_model()
bf.predict_outcomes(on="target")
bf.outcomes_hat().mean()

# An honest confidence interval for μ̂_OM via a nonparametric bootstrap
# (resample responders, refit ĝ*, predict on the fixed target, re-average):
bf.outcomes_hat().mean_with_ci(ci_method="bootstrap", n_bootstrap=200, random_seed=2020)

# outcomes_hat().summary() reports the estimator and *scopes* any doubly-robust
# claim to the fit weights — it never prints a blanket "doubly robust" (a plain
# g-computation for the non-linear default; DR only for a weighted linear+intercept fit):
bf.outcomes_hat().summary()
```

**Train/holdout transfer.** You can fit the outcome model on one frame (a *train*
split) and apply it to a *different* frame (a *holdout* / scoring split) with the
same covariate schema, mirroring how `set_fitted_model` transfers a fitted IPW
model. `set_fitted_outcome_model` copies the already-fitted model onto the holdout
**without re-fitting** (the fitted learner is shared by identity), so predicting on
the holdout target gives `μ̂_OM` computed with the train model:

```python
# train_bf carries a fitted outcome model (train_bf.fit_outcome_model(...));
# holdout_bf has the same covariates plus its own target to score:
scored = holdout_bf.set_fitted_outcome_model(train_bf, inplace=False)
scored.predict_outcomes(on="target")
scored.outcomes_hat().mean()      # μ̂_OM on the holdout target via train_bf's model
```

Separately, `bf.outcomes().weights_impact_on_outcome_ss()` is a *diagnostic*
(not a third estimator): it reports how much applying the weights moves the
outcome mean (weighted vs. unweighted), with a significance test — useful for
judging whether the reweighting materially changed the outcome estimate. See the
[outcome-model design doc](https://github.com/facebookresearch/balance/blob/main/docs/architecture/architecture_0_23_0.md)
for the estimator theory and the AIPW roadmap.

## Design-based inference

_balance_ complements an adjacent library that handles the design-based
inference step _balance_ does not — covering the non-probability → DiD
half of a survey-based campaign or policy evaluation in Python.

### Survey-weighted Difference-in-Differences (diff-diff)

_balance_ pairs naturally with
[`diff-diff`](https://github.com/igerber/diff-diff), the open-source Python
package for modern Difference-in-Differences (Callaway & Sant'Anna 2021,
Sun & Abraham 2021, Borusyak-Jaravel-Spiess 2024, Synthetic DiD, Continuous
DiD, Triple Difference) with built-in survey-design variance. The two
libraries solve adjacent halves of a survey-based campaign or policy
evaluation:

- _balance_ produces non-probability weights for a sample against a target
  population frame (IPW, CBPS, rake, post-stratification).
- _diff-diff_ consumes those weights via its `SurveyDesign` and returns
  design-consistent ATT(g, t) with HonestDiD sensitivity.

The thin adapter `balance.interop.diff_diff` (added in the upcoming v0.21
release; see CHANGELOG.md) turns the handoff into a single import. Install
the optional extra:

```bash
pip install "balance[did]"
```

Then the canonical workflow is:

```python
from balance import Sample
from balance.interop.diff_diff import fit_did

# `set_target(...)` requires a Sample/SampleFrame, not a raw DataFrame --
# wrap your target DataFrame with Sample.from_frame(...) first.
s = (
    Sample.from_frame(df, ...)
    .set_target(Sample.from_frame(target_df))
    .adjust(method="ipw")
)
results = fit_did(s, estimator="CallawaySantAnna", outcome="y", time="t",
                  unit="state", treatment_first="first_treat",
                  estimation_method="dr")
```

`fit_did` builds the `SurveyDesign` from the active _balance_ weight
column, strips _balance_'s history columns, and routes kwargs into the
chosen diff-diff estimator. For a complete walk-through on a BRFSS-shaped
public-health panel (including HonestDiD sensitivity), see the tutorial:
[`tutorials/balance_diff_diff_brfss.ipynb`](https://github.com/facebookresearch/balance/blob/main/tutorials/balance_diff_diff_brfss.ipynb).
Upstream project: [`github.com/igerber/diff-diff`](https://github.com/igerber/diff-diff).

## Developer and AI assistant resources

- [`ARCHITECTURE.md`](https://github.com/facebookresearch/balance/blob/main/ARCHITECTURE.md) — internal architecture documentation (class hierarchy, data flow, design decisions)
- [`.github/copilot-instructions.md`](https://github.com/facebookresearch/balance/blob/main/.github/copilot-instructions.md) — instructions for LLM/AI coding assistants working on balance

## Other resources

- Presentation:
  ["Balancing biased data samples with the 'balance' Python package"](https://github.com/facebookresearch/balance/blob/main/website/static/docs/Balancing_biased_data_samples_with_the_balance_Python_package_-_ISA_conference_2023-06-01.pdf) -
  presented in the Israeli Statistical Association (ISA) conference on June
  1st 2023.

# More details

## Getting help, submitting bug reports and contributing code

You are welcome to:

- Learn more in the [_balance_](https://import-balance.org/) website.
- Ask for help on:
  https://github.com/facebookresearch/balance/issues/new?template=support_question.md
- Submit bug-reports and features' suggestions at:
  https://github.com/facebookresearch/balance/issues
- Send a pull request on: https://github.com/facebookresearch/balance. See the
  [CONTRIBUTING](https://github.com/facebookresearch/balance/blob/main/CONTRIBUTING.md)
  file for how to help out. And our
  [CODE OF CONDUCT](https://github.com/facebookresearch/balance/blob/main/LICENSE-DOCUMENTATION)
  for our expectations from contributors.

## Citing _balance_

Sarig, T., Galili, T., & Eilat, R. (2023). balance – a Python package for
balancing biased data samples.
[https://arxiv.org/abs/2307.06024](https://arxiv.org/abs/2307.06024)

```bibtex
@misc{sarig2023balance,
      title={balance - a Python package for balancing biased data samples},
      author={Tal Sarig and Tal Galili and Roee Eilat},
      year={2023},
      eprint={2307.06024},
      archivePrefix={arXiv},
      primaryClass={stat.CO}
}
```

## License

The _balance_ package is licensed under the
[MIT license](https://github.com/facebookresearch/balance/blob/main/LICENSE),
and all the documentation on the site (including text and images) is under
[CC-BY](https://github.com/facebookresearch/balance/blob/main/LICENSE-DOCUMENTATION).

# News

You can follow updates on our:

- [Blog](https://import-balance.org/blog/)
- [Changelog](https://github.com/facebookresearch/balance/blob/main/CHANGELOG.md)

## Acknowledgements / People

The _balance_ package is actively maintained by people from the
[Central Applied Science](https://research.facebook.com/teams/central-applied-science/)
team (in Menlo Park and Tel Aviv), by
[Wesley Lee](https://www.linkedin.com/in/wesley-lee),
[Tal Sarig](https://research.facebook.com/people/sarig-tal/), and
[Tal Galili](https://research.facebook.com/people/galili-tal/).

The _balance_ package was (and is) developed by many people, including:
[Roee Eilat](https://research.facebook.com/people/eilat-roee/),
[Tal Galili](https://research.facebook.com/people/galili-tal/),
[Daniel Haimovich](https://research.facebook.com/people/haimovich-daniel/),
[Kevin Liou](https://www.linkedin.com/in/kevinycliou),
[Steve Mandala](https://research.facebook.com/people/mandala-steve/),
[Adam Obeng](https://adamobeng.com/) (author of the initial internal Meta
version), [Tal Sarig](https://research.facebook.com/people/sarig-tal/),
[Luke Sonnet](https://www.linkedin.com/in/luke-sonnet),
[Sean Taylor](https://seanjtaylor.com),
[Barak Yair Reif](https://www.linkedin.com/in/barak-yair-reif-2154365/),
[Soumyadip Sarkar](https://github.com/neuralsorcerer), and others. If you worked
on balance in the past, please email us to be added to this list.

The _balance_ package was open-sourced by
[Tal Sarig](https://research.facebook.com/people/sarig-tal/),
[Tal Galili](https://research.facebook.com/people/galili-tal/) and
[Steve Mandala](https://research.facebook.com/people/mandala-steve/) in
late 2022.

Branding created by [Dana Beaty](https://www.danabeaty.com/), from the Meta AI
Design and Marketing Team. For logo files, see
[here](https://github.com/facebookresearch/balance/tree/main/website/static/img/).
