---
title: Balance keeps leveling up - v0.15.0 deepens diagnostics, missing-data handling, and workflow clarity

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

**tl;dr – balance v0.15.0**

We’re excited to announce [**balance v0.15.0**](https://pypi.org/project/balance/)! Since [v0.12.0](https://github.com/facebookresearch/balance/releases/tag/0.12.0), balance has focused on deeper diagnostics, clearer missing-data behavior, and more flexible workflows across the core API and CLI. This post highlights the key changes from **v0.12.0 -> v0.15.0**:

- **Stronger diagnostics:** New distribution distance metrics (EMD, CVMD, KS), plus richer adjusted sample summaries and more consistent display formatting.
- **More control over modeling:** Customizable IPW estimators, formula-driven summaries, and explicit missing-data handling.
- **Better workflows:** CLI enhancements, improved docs/tutorials, and developer tooling upgrades like type checking and coverage.

[![balance_logo_horizontal](https://raw.githubusercontent.com/facebookresearch/balance/main/website/static/img/balance_logo/PNG/Horizontal/balance_Logo_Horizontal_FullColor_RGB.png)](https://import-balance.org/)

<!--truncate-->


## Diagnostics That Go Further

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

### New distribution distance metrics

Balance now exposes **[Earth Mover’s Distance (EMD)](https://en.wikipedia.org/wiki/Earth_mover%27s_distance)**, **[Cramér-von Mises distance (CVMD)](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93von_Mises_criterion)**, and **[Kolmogorov–Smirnov (KS)](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)** statistics through `BalanceDF` diagnostics. These diagnostics work with weighted or unweighted comparisons, include discrete/continuous variants, and respect one-hot categorical aggregation when enabled.

Example:

```python
# Compare covariate distributions between a sample and target.
sample.set_target(target).covars().emd(on_linked_samples=False)
sample.set_target(target).covars().cvmd(on_linked_samples=False)
sample.set_target(target).covars().ks(on_linked_samples=False)
```

Output:

```
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

### Richer adjusted sample summaries

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

### More consistent output formatting

Weight diagnostics now follow the formatting of `Sample.summary()`, showing design effect and effective sample size metrics on separate lines and replacing abbreviations like “eff.” with the full “effective” wording.

Example:

```python
print(sample.adjust(target, method="ipw"))
```

Output:

```
        Adjusted balance Sample object with target set using ipw
        4 observations x 2 variables: age,group
        id_column: id, weight_column: weight,
        outcome_columns: outcome
        
        adjustment details:
            method: ipw
            weight trimming mean ratio: 20
            design effect (Deff): 1.001
            effective sample size proportion (ESSP): 0.999
            effective sample size (ESS): 4.0
                
            target:
                 
	        balance Sample object
	        6 observations x 2 variables: age,group
	        id_column: id, weight_column: weight,
	        outcome_columns: outcome
	        
            2 common variables: age,group
```


## More Flexible Modeling and Summary Tools

### Bring your own IPW estimator

`.adjust(method="ipw")` now accepts **any scikit-learn classifier** via the `model` argument, so you can use estimators like random forests or gradient boosting. You can also pass a configured `LogisticRegression` instance or provide JSON-configured parameters through the CLI.

Example:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0)
adjusted = sample.adjust(target, method="ipw", model=rf)
adjusted.is_adjusted()
```

Output:

```
True
```

### New covariate diagnostics

Balance now calculates KL divergence for covariates (numeric and one-hot categorical), available through `sample.covars().kld()` and compatible with linked-sample aggregation.

Example:

```python
sample.set_target(target).covars().kld(on_linked_samples=False)
```

Output:

```
             age  group[A]  group[B]  mean(kld)
index                                          
covars  0.232348       0.0       0.0   0.116174
```

### Formula-driven descriptive statistics

`descriptive_stats()` now accepts a **formula argument** that is always applied (even for numeric-only frames), letting you control the terms and dummy variables used in summary statistics.

Example:

```python
import pandas as pd
from balance.stats_and_plots.weighted_stats import descriptive_stats

df = pd.DataFrame({"age": [20, 30, 40], "group": ["A", "B", "A"]})
descriptive_stats(df, stat="mean", formula="age + group")
```

Output:

```
    age  group[A]  group[B]
0  30.0  0.666667  0.333333
```

### Missing data handling in poststratification

`poststratify()` now includes `na_action` to either drop missing rows or treat missing values as their own category. **Breaking change:** missing values default to a `"__NaN__"` category, so if you want legacy “drop” behavior you must pass `na_action="drop"` explicitly.

Example:

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

Output:

```
[0.5, 2.0, 1.0, 0.5]
```

### Model matrix NAs now behave as documented

`model_matrix(add_na=False)` now actually drops rows with missing values while preserving categorical levels—matching the documented behavior rather than only logging a warning.

Example:

```python
import pandas as pd
from balance.utils.model_matrix import model_matrix

df = pd.DataFrame({"age": [20, None, 30], "group": ["A", "B", "A"]})
model_matrix(df, add_na=False)["sample"]
```

Output:

```
    age  group[A]  group[B]
0  20.0       1.0       0.0
2  30.0       1.0       0.0
```


## Smarter Weighting Workflows

### Rake and poststratify trimming parity

`rake()` and `poststratify()` now respect `weight_trimming_mean_ratio` and `weight_trimming_percentile`, trimming and renormalizing weights through the enhanced `trim_weights(..., target_sum_weights=...)` API.

Example:

```python
import pandas as pd
from balance.sample_class import Sample

sample_df = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "gender": ["F", "M", "F", "M"],
        "region": ["North", "North", "South", "South"],
        "weight": [1.0, 1.0, 1.0, 1.0],
    }
)
target_df = pd.DataFrame(
    {
        "id": [10, 11, 12, 13, 14, 15],
        "gender": ["F", "F", "M", "M", "F", "M"],
        "region": ["North", "South", "North", "South", "North", "South"],
        "weight": [1.0] * 6,
    }
)

sample = Sample.from_frame(
    sample_df,
    id_column="id",
    weight_column="weight",
    outcome_columns=[],
    standardize_types=False,
)
target = Sample.from_frame(
    target_df,
    id_column="id",
    weight_column="weight",
    outcome_columns=[],
    standardize_types=False,
)

adjusted = sample.adjust(
    target,
    method="rake",
    variables=["gender", "region"],
    weight_trimming_mean_ratio=20.0,
)
adjusted.weights().summary()
```

Output:

```
                                var  val
0                     design_effect  1.0
1       effective_sample_proportion  1.0
2             effective_sample_size  4.0
3                               sum  6.0
4                    describe_count  4.0
5                     describe_mean  1.0
6                      describe_std  0.0
7                      describe_min  1.0
8                      describe_25%  1.0
9                      describe_50%  1.0
10                     describe_75%  1.0
11                     describe_max  1.0
12                    prop(w < 0.1)  0.0
13                    prop(w < 0.2)  0.0
14                  prop(w < 0.333)  0.0
15                    prop(w < 0.5)  0.0
16                      prop(w < 1)  0.0
17                     prop(w >= 1)  1.0
18                     prop(w >= 2)  0.0
19                     prop(w >= 3)  0.0
20                     prop(w >= 5)  0.0
21                    prop(w >= 10)  0.0
22               nonparametric_skew  0.0
23  weighted_median_breakdown_point  0.5
```

### High-cardinality covariate warnings 

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

### Large target dataset warning

`Sample.adjust()` now warns when the target is **very large** (>=100k rows and >=10× the sample), clarifying that uncertainty is primarily driven by the sample rather than the target—similar to a one-sample comparison.

Example:

```python
# When target is much larger than the sample, adjust emits a warning.
large_target_size = 100_000
large_sample_size = 1
large_target_size >= 100_000 and large_target_size >= 10 * large_sample_size
```

Output:

```
True
```

### Weight trimming consistency improvements

`trim_weights()` now computes percentile thresholds with explicit clipping bounds for consistent behavior across Python/NumPy versions. This can shift percentile-based trimming by roughly one observation at typical limits.

Example:

```python
from balance.adjustment import trim_weights

import pandas as pd

weights = pd.Series([1.0, 2.0, 100.0])
trim_weights(weights, weight_trimming_percentile=0.99)
```

Output:

```
[1.0, 2.0, 100.0]
```


## CLI and Documentation Improvements

### CLI outcome control

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

### Expanded CLI docs and tutorial

`balance.cli` is now fully documented, and the CLI tutorial notebook includes richer diagnostics exploration, command echoing, and a `load_data()` example.

### New tutorials and docs

- Post-stratification tutorial notebook and expanded documentation.
- Additional examples showing custom IPW models.
- README badges for build status, version support, and release tracking.


## Under-the-Hood Improvements for Developers

### Diagnostics and utility refactors

- Diagnostics helpers were consolidated for more stable outputs; for IPW, `Sample.diagnostics()` now always emits iteration/intercept summaries plus hyperparameter settings (breaking change in output shape).
- The old `balance.util` module was split into focused `balance.utils` submodules.
- `test_util.py` was split into five targeted modules mirroring the new utils layout.

### Raking algorithm refactor

The raking implementation removed the `ipfn` dependency in favor of a vectorized NumPy implementation, improving performance and reducing external dependencies.

### Type checking and coverage tooling

- Added **Pyre type checking** to GitHub Actions for library code via `.pyre_configuration.external`.
- Added a **test coverage workflow and README badge**, with HTML/XML coverage reports.
- Aligned formatting across internal and GitHub tooling using Black 25.1.0 and updated CI/pre-commit configuration accordingly.

### Structural and typing improvements

- Migrated 32 Python files to `# pyre-strict`, modernized type hints to PEP 604 syntax where compatible, and introduced `TypedDict` definitions for plotting.
- Renamed `Balance_*` classes to **BalanceDF** variants (e.g., `BalanceCovarsDF -> BalanceDFCovars`).


## Bug Fix Highlights

A few notable fixes from v0.13–v0.15 include:

- Stable CBPS probability computation to avoid overflow warnings.
- Respecting `weighted=False` for target data in categorical QQ plots.
- Early validation errors for null weights in `Sample.from_frame`.
- `trim_weights()` now returns a consistent `float64` Series and preserves index ordering.


## Breaking Changes to Watch

If you’re upgrading from v0.12.x or v0.13.x, these behavior changes are important:

- `poststratify()` **now defaults to treating missing values as their own category** (`"__NaN__"`). Use `na_action="drop"` for legacy behavior.
- `model_matrix(add_na=False)` now **drops rows with missing data** instead of just warning.
- Percentile trimming uses explicit clipping bounds for cross-platform consistency, which may shift thresholds by roughly one observation.
- `Sample.diagnostics()` for IPW now always emits iteration/intercept summaries and hyperparameter settings, changing its output shape.


## Community & Contributors

A huge thank you to everyone who contributed to versions 0.13–0.15, including **@neuralsorcerer**, **@talgalili**, **@wesleytlee**, and the broader community!

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