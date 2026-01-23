---
title: Balance v0.15.0 - more sklearn models, diagnostics, and stability (100% test-coverage!)

authors:
  - name: Soumyadip Sarkar
    title: Independent Researcher
    url: https://www.linkedin.com/in/neuralsorcerer
  - name: Tal Galili
    title: Machine Learning Engineer
    url: https://www.linkedin.com/in/tal-galili-5993085/

tags: [python, open-source, survey-statistics, package-update]
hide_table_of_contents: true
---

🎉 **balance v0.15.0 is out!**

### What is balance?

[**balance**](https://pypi.org/project/balance/) is a Python package (from Meta) offering a simple workflow and methods for dealing with biased data samples when looking to infer from them to some population of interest. Biased samples often occur in survey statistics when respondents present non-response bias or surveys suffer from sampling bias (that are not missing completely at random). A similar issue arises in observational studies when comparing treated vs untreated groups, and in any data that suffers from selection bias.

### Highlights from **[v0.15.0](https://pypi.org/project/balance/)** (since [v0.12.0](https://github.com/facebookresearch/balance/releases/tag/0.12.0)):

✅ **More control over modeling:** The ability to run any sklearn model (instead of just LogisticRegression) to fit inverse-propensity-score weights. Plus formula-driven summaries and explicit missing-data handling.

✅ **Stronger diagnostics:** The way weights influence covariate imbalance can now be evaluated not just with ASMD (as before), but also with various distribution distance metrics (KLD, EMD, CVMD, KS).

✅ **Reliable code:** Test coverage was increased to 100%, with full type-checking across the whole codebase. Plus CLI enhancements and improved docs/tutorials.

[![balance_logo_horizontal](https://raw.githubusercontent.com/facebookresearch/balance/main/website/static/img/balance_logo/PNG/Horizontal/balance_Logo_Horizontal_FullColor_RGB.png)](https://import-balance.org/)

<!--truncate-->

## Updated Tutorials

- Post-stratification tutorial notebook (and expanded documentation): https://import-balance.org/docs/tutorials/quickstart_poststratify/
- CLI tutorial: https://import-balance.org/docs/tutorials/balance_cli_tutorial/
- Customizing IPW models: https://import-balance.org/docs/tutorials/quickstart/


## More Flexible IPW Modeling and Summary Tools

### Bring Your Own IPW sklearn Estimator

`.adjust(method="ipw")` now accepts **any scikit-learn classifier** via the `model` argument, so you can use estimators like random forests or gradient boosting. You can also pass a configured `LogisticRegression` instance or provide JSON-configured parameters through the CLI.

> Examples below assume `sample` and `target` are `Sample` objects unless otherwise noted.

Example setup used by multiple snippets:

```python
import pandas as pd
from balance.sample_class import Sample

sample_df = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "age": [20, 30, 40, 50],
        "group": ["A", "B", "A", "B"],
        "outcome": [1.0, 2.0, 3.0, 4.0],
        "weight": [1.0, 1.0, 1.0, 1.0],
    }
)
target_df = pd.DataFrame(
    {
        "id": [10, 11, 12, 13, 14, 15],
        "age": [25, 35, 45, 55, 65, 75],
        "group": ["A", "B", "A", "B", "A", "B"],
        "outcome": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        "weight": [1.0] * 6,
    }
)

sample = Sample.from_frame(
    sample_df,
    id_column="id",
    weight_column="weight",
    outcome_columns=["outcome"],
    standardize_types=False,
)
target = Sample.from_frame(
    target_df,
    id_column="id",
    weight_column="weight",
    outcome_columns=["outcome"],
    standardize_types=False,
)

sample.df
```

Output:

```
  id  age group  outcome  weight
0  1   20     A      1.0     1.0
1  2   30     B      2.0     1.0
2  3   40     A      3.0     1.0
3  4   50     B      4.0     1.0
```

Example:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0)
adjusted = sample.adjust(target, method="ipw", model=rf)
```

A detailed example is given here: https://import-balance.org/docs/tutorials/quickstart/


## Diagnostics That Go Further

### Beyond ASMD: New Distance Metrics


Balance now exposes **[KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)**, **[Earth Mover's Distance (EMD)](https://en.wikipedia.org/wiki/Earth_mover%27s_distance)**, **[Cramér-von Mises distance (CVMD)](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93von_Mises_criterion)**, and **[Kolmogorov–Smirnov (KS)](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)** statistics through `BalanceDF` diagnostics. These diagnostics work with weighted or unweighted comparisons, include discrete/continuous variants, and respect one-hot categorical aggregation when enabled.

Example:

```python
# Compare covariate distributions between a sample and target.
sample.set_target(target).covars().kld()
sample.set_target(target).covars().emd()
sample.set_target(target).covars().cvmd()
sample.set_target(target).covars().ks()
```

Output:

```
             age  group[A]  group[B]  mean(kld)
index
covars  0.232348       0.0       0.0   0.116174

         age  group[A]  group[B]  mean(emd)
index
covars  15.0       0.0       0.0        7.5

             age  group[A]  group[B]  mean(cvmd)
index
covars  0.083333       0.0       0.0    0.041667

        age  group[A]  group[B]  mean(ks)
index
covars  0.5       0.0       0.0      0.25
```

### Richer Adjusted Sample Summaries

Adjusted samples now surface more information at a glance:

- `Sample.__str__()` shows adjustment method, trimming parameters, design effect, and effective sample size.
- `Sample.summary()` groups covariate diagnostics, reports design effect alongside ESSP/ESS, and surfaces weighted outcome means when available.

Example:

```python
adjusted = sample.adjust(target, method="ipw")
adjusted.summary()
```

Output:

```
Adjustment details:
    method: ipw
    weight trimming mean ratio: 20
Covariate diagnostics:
    Covar ASMD reduction: -3.0%
    Covar ASMD (3 variables): 0.401 -> 0.413
    Covar mean KLD reduction: 2.2%
    Covar mean KLD (2 variables): 0.116 -> 0.114
Weight diagnostics:
    design effect (Deff): 1.001
    effective sample size proportion (ESSP): 0.999
    effective sample size (ESS): 4.0
Outcome weighted means:
            outcome
source
self          2.479
unadjusted    2.500
target        4.000
Model performance: Model proportion deviance explained: 0.034
```



## Smarter Weighting Workflows

### CLI Now Supports Outcome Columns

The CLI now supports `--outcome_columns`, letting you explicitly declare which columns are outcomes. Remaining columns are moved to `ignored_columns` instead of being treated implicitly.

Example:

```python
from argparse import Namespace
from balance.cli import BalanceCLI

BalanceCLI(Namespace(outcome_columns="y,z")).outcome_columns()
```

Output:

```
['y', 'z']
```

### High-Cardinality Covariate Warnings

Balance warns when categorical covariates have >=80% unique values (e.g., user IDs), helping identify problematic columns before fitting.

Example:

```python
# If user_id is high-cardinality, adjust will emit a warning before fitting.
import pandas as pd
from balance.utils.pandas_utils import _detect_high_cardinality_features

_detect_high_cardinality_features(
    pd.DataFrame({"id": ["a", "b", "c"], "group": ["a", "a", "b"]}),
    threshold=0.8,
)
```

Output:

```
[HighCardinalityFeature(column='id', unique_count=3, unique_ratio=1.0, has_missing=np.False_)]
```

## Developer Improvements, Bug Fixes & Breaking Changes

README badges for build status, version support, release tracking, and unittest coverage: https://import-balance.org/docs/docs/overview/

`poststratify()` now supports `na_action` to either drop missing rows or treat missing values as their own category; **breaking change:** missing values default to a `"__NaN__"` category, so legacy drop behavior requires `na_action="drop"`.

```python
import pandas as pd
from balance.weighting_methods.poststratify import poststratify

sample_df = pd.DataFrame({"gender": ["Female", None, "Male", "Female"]})
target_df = pd.DataFrame({"gender": ["Female", None, None, "Male"]})

poststratify(
    sample_df=sample_df,
    sample_weights=pd.Series([1, 1, 1, 1]),
    target_df=target_df,
    target_weights=pd.Series([1, 1, 1, 1]),
    variables=["gender"],
    na_action="add_indicator",
)["weight"].tolist()
```

```text
[0.5, 2.0, 1.0, 0.5]
```

`model_matrix(add_na=False)` now actually drops rows with missing values while preserving categorical levels (matching the documented behavior, not just warning):

```python
import pandas as pd
from balance.utils.model_matrix import model_matrix

df = pd.DataFrame({"age": [20, None, 30], "group": ["A", "B", "A"]})
model_matrix(df, add_na=False)["sample"]
```

```text
    age  group[A]  group[B]
0  20.0       1.0       0.0
2  30.0       1.0       0.0
```

Under the hood, developers get: trimming parity for `rake()`/`poststratify()` via `trim_weights(..., target_sum_weights=...)`, warnings for very large targets (>=100k and >=10× sample), more consistent percentile trimming via explicit clipping, formula-driven summaries in `descriptive_stats(formula=...)`, consolidated diagnostics helpers (and **breaking change**: IPW `Sample.diagnostics()` output shape now always includes iteration/intercept summaries plus hyperparameters), and a split of the old `balance.util` into focused `balance.utils` submodules. Testing/typing updates include **100% coverage(!)**, migrating 32 files to `# pyre-strict`, modernized PEP 604 type hints, `TypedDict` definitions for plotting, renaming `Balance_*` classes to **BalanceDF** variants, adding Pyre checking in GitHub Actions via `.pyre_configuration.external`, and aligning formatting/CI tooling with Black 25.1.0. The raking algorithm was refactored to remove the `ipfn` dependency in favor of a vectorized NumPy implementation.

Bug fixes (v0.13–v0.15) include: stable CBPS probability computation to avoid overflow, honoring `weighted=False` for target data in categorical QQ plots, earlier validation errors for null weights in `Sample.from_frame`, and `trim_weights()` now returning a consistent `float64` Series while preserving index ordering.

**Breaking changes to watch when upgrading:** `poststratify()` defaults to `"__NaN__"` missing-category handling (use `na_action="drop"` to drop), `model_matrix(add_na=False)` drops missing-data rows, percentile trimming uses explicit clipping bounds (thresholds may shift by ~1 observation), and IPW `Sample.diagnostics()` output shape changed to always include iteration/intercept summaries and hyperparameter settings.

Details are in: https://import-balance.org/docs/docs/changelog/

## Community & Contributors

A huge thank you to everyone who contributed to versions 0.13–0.15, including **@neuralsorcerer**, **@talgalili**, **@wesleytlee**, the BPG team in Tel-Aviv, and the broader community!

Want to contribute? Check out our [contributing guide](https://github.com/facebookresearch/balance/blob/main/CONTRIBUTING.md).


## Get Started with v0.15.0

Upgrade today:

    python -m pip install -U balance

Resources:
- **Website:** https://import-balance.org/
- **GitHub:** https://github.com/facebookresearch/balance
- **Documentation:** https://import-balance.org/docs/docs/general_framework/
- **Tutorials:** https://import-balance.org/docs/tutorials/
- **Blog:** https://import-balance.org/blog/
- **Paper:** [balance – a Python package for balancing biased data samples](https://arxiv.org/abs/2307.06024)

Need help?
- **Ask questions:** https://github.com/facebookresearch/balance/issues/new?template=support_question.md
- **Report bugs:** https://github.com/facebookresearch/balance/issues/new?template=bug_report.md
- **Request features:** https://github.com/facebookresearch/balance/issues/new?template=feature_request.md
