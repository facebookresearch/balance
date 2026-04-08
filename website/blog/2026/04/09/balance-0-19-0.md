---
title: Balance v0.19.0 - a new architecture for weighting workflows, with a clear upgrade path from v0.16

authors:
  - name: Tal Galili
    title: Machine Learning Engineer @ Meta
    url: https://www.linkedin.com/in/tal-galili-5993085/
  - name: Soumyadip Sarkar
    title: Independent Researcher
    url: https://www.linkedin.com/in/neuralsorcerer

tags: [python, open-source, survey-statistics, package-update]
hide_table_of_contents: true
---

🎉 **balance v0.19.0 is out!**

If you are upgrading from **v0.16.0 → v0.19.0**, this release cycle delivers three big things:

1. A **new core architecture** (`SampleFrame` + `BalanceFrame`) in v0.19.0.
2. A **modernized diagnostics surface** (including `r_indicator`, formula-aware diagnostics, and safer edge-case behavior).
3. A **clean migration path** from deprecated `Sample` convenience methods to the `weights()`, `covars()`, and `outcomes()` APIs.

This post summarizes the changes carefully and accurately across **v0.16.0, v0.17.0, v0.18.0, and v0.19.0**, then shows runnable examples with outputs.

[![balance_logo_horizontal](https://raw.githubusercontent.com/facebookresearch/balance/main/website/static/img/balance_logo/PNG/Horizontal/balance_Logo_Horizontal_FullColor_RGB.png)](https://import-balance.org/)

<!--truncate-->

## What is balance?

[**balance**](https://pypi.org/project/balance/) is a Python package (from Meta) for adjusting biased samples when inferring from a sample to a target population.

It supports practical workflows in survey statistics, observational analyses, and any setup where sample composition differs from the population you want to describe.

## Why v0.19.0 matters

v0.19.0 is a **major architecture release**:

- New **`SampleFrame`** class: explicit data container with column-role metadata.
- New **`BalanceFrame`** class: adjustment/diagnostics orchestrator for sample-target workflows.
- Existing **`Sample` API remains backward compatible**, now implemented via inheritance:
  - `Sample → BalanceFrame → SampleFrame`

This separation makes workflows more composable while preserving existing user code.

## Release-by-release changes (v0.16 → v0.19)

## v0.16.0 (2026-02-09)

### New capabilities

- Added outcome-weight impact diagnostics (`y*w0` vs `y*w1`) with confidence intervals, surfaced in:
  - `BalanceDFOutcomes`
  - `Sample.diagnostics()`
  - CLI via `--weights_impact_on_outcome_method`
- Added pandas 3 support.
- Updated categorical distribution metrics (`kld`, `emd`, `cvmd`, `ks`) on `BalanceDF.covars()` to operate on raw categorical variables (with NA indicators) instead of one-hot columns.
- Added `use_model_matrix=False` for raw-covariate IPW with custom models.
- Added configurable ID-candidate names for ID auto-detection.
- Added formula support to `BalanceDF.model_matrix()`.

### Breaking behavior

- Diagnostics that normalize/aggregate weights now raise `ValueError` when **all weights are zero**:
  - `design_effect`
  - `nonparametric_skew`
  - `prop_above_and_below`
  - `weighted_median_breakdown_point`

## v0.17.0 (2026-03-17)

### Breaking behavior

- CLI default classification changed:
  - when `--outcome_columns` is not provided, unmentioned columns now go to `ignore_columns` (not `outcome_columns`).

### New functionality

- Added/improved ASCII plotting:
  - `ascii_comparative_hist`
  - population → adjusted → sample ordering in comparative output
  - `ascii_plot_dist(comparative=...)`

### Reliability and refactoring

- Dataset loading implementations moved out of `balance.datasets.__init__` to a focused loading module (public API preserved via re-export).
- Multiple robustness fixes across diagnostics/model-matrix/CLI parsing.

## v0.18.0 (2026-03-24)

### New functionality

- Added validated public `r_indicator(sample_p, target_p)` implementation.
- Added convenience wrapper: `sample.weights().r_indicator()`.

### Deprecations announced in v0.18.0 (removed in v0.19.0)

- `Sample.design_effect()`
- `Sample.design_effect_prop()`
- `Sample.plot_weight_density()`
- `Sample.covar_means()`
- `Sample.outcome_sd_prop()`
- `Sample.outcome_variance_ratio()`

### Important bug fix

- Fixed potential memory explosion in raking helper expansion by introducing capped length (`max_length`, default `10000`) with Hare-Niemeyer allocation when needed.

## v0.19.0 (2026-04-06)

### Architecture and API evolution

- Added `SampleFrame` for explicit role-aware DataFrame management.
- Added `BalanceFrame` for target-linking, adjustment, diagnostics, and export/filter workflows.
- Added conversion helpers across `Sample`, `SampleFrame`, and `BalanceFrame`.
- Added sequential/compound adjustment support through repeated `.adjust()` with preserved weight history.
- Added formula propagation in `Sample.covars(formula=...)` for downstream diagnostics (including `kld`).

### Breaking changes in v0.19.0

The v0.18 deprecations are now removed. Use:

- `sample.weights().design_effect()`
- `sample.weights().design_effect_prop()`
- `sample.weights().plot()`
- `sample.covars().mean()`
- `sample.outcomes().outcome_sd_prop()`
- `sample.outcomes().outcome_variance_ratio()`

### Additional technical improvements

- `Sample.is_adjusted` now behaves as both property and callable via `_CallableBool`.
- `BalanceDF` decoupled from hard `Sample` dependency through `BalanceDFSource` protocol.
- Summary/diagnostics builders extracted into reusable helpers.
- Comprehensive architecture-focused tests added for new classes and behavior parity.

## Practical examples (run on balance 0.19.0)

### 1) Build with `SampleFrame`, orchestrate with `BalanceFrame`

```python
import pandas as pd
from balance.sample_frame import SampleFrame
from balance.balance_frame import BalanceFrame

sample_df = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "age": [21, 35, 42, 57],
        "gender": ["F", "M", "F", "M"],
        "weight": [1.0, 1.0, 1.0, 1.0],
        "outcome": [0.2, 0.4, 0.6, 0.8],
    }
)

target_df = pd.DataFrame(
    {
        "id": [10, 11, 12, 13, 14, 15],
        "age": [20, 30, 40, 50, 60, 70],
        "gender": ["F", "M", "F", "M", "F", "M"],
        "weight": [1.0] * 6,
        "outcome": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
    }
)

sf = SampleFrame.from_frame(
    sample_df,
    id_column="id",
    weight_column="weight",
    outcome_columns=["outcome"],
    standardize_types=False,
)
tf = SampleFrame.from_frame(
    target_df,
    id_column="id",
    weight_column="weight",
    outcome_columns=["outcome"],
    standardize_types=False,
)

bf = BalanceFrame(sample=sf, target=tf)
print("covar_columns", sf.covar_columns)
print("outcome_columns", sf.outcome_columns)
print("has_target", bf.has_target)
print("is_adjusted", bf.is_adjusted)
print("adjust_null_is_adjusted", bf.adjust(method="null").is_adjusted)
```

```text
covar_columns ['age', 'gender']
outcome_columns ['outcome']
has_target True
is_adjusted False
adjust_null_is_adjusted True
```

### 2) Sequential adjustments and `is_adjusted` compatibility

```python
import pandas as pd
from balance.sample_class import Sample

sample_df = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "age": [21, 35, 42, 57],
        "gender": ["F", "M", "F", "M"],
        "weight": [1.0, 1.0, 1.0, 1.0],
    }
)

target_df = pd.DataFrame(
    {
        "id": [10, 11, 12, 13, 14, 15],
        "age": [20, 30, 40, 50, 60, 70],
        "gender": ["F", "M", "F", "M", "F", "M"],
        "weight": [1.0] * 6,
    }
)

s = Sample.from_frame(sample_df, id_column="id", weight_column="weight", standardize_types=False)
t = Sample.from_frame(target_df, id_column="id", weight_column="weight", standardize_types=False)

s2 = s.set_target(t).adjust(method="null").adjust(method="null")

print("is_adjusted_property", s2.is_adjusted)
print("is_adjusted_call", s2.is_adjusted())
print("weight_columns_all", s2.weight_columns_all)
print("active_weight", s2.weight_column)
```

```text
is_adjusted_property True
is_adjusted_call True
weight_columns_all ['weight', 'weight_pre_adjust', 'weight_adjusted_1', 'weight_adjusted_2']
active_weight weight
```

### 3) `r_indicator()` through the weights API

```python
import pandas as pd
from balance.sample_class import Sample

sample_df = pd.DataFrame({"id": [1, 2, 3, 4], "x": [0, 1, 0, 1], "weight": [1, 1, 1, 1]})
target_df = pd.DataFrame({"id": [10, 11, 12, 13], "x": [0, 0, 1, 1], "weight": [1, 1, 1, 1]})

s = Sample.from_frame(sample_df, id_column="id", weight_column="weight", standardize_types=False)
t = Sample.from_frame(target_df, id_column="id", weight_column="weight", standardize_types=False)
a = s.set_target(t).adjust(method="null")

print(round(float(a.weights().r_indicator()), 6))
```

```text
1.0
```

### 4) Formula-aware diagnostics propagation in `covars()`

```python
import pandas as pd
from balance.sample_class import Sample

sample_df = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "age": [20, 30, 40, 50],
        "gender": ["F", "M", "F", "M"],
        "weight": [1, 1, 1, 1],
    }
)

target_df = pd.DataFrame(
    {
        "id": [10, 11, 12, 13, 14, 15],
        "age": [25, 35, 45, 55, 65, 75],
        "gender": ["F", "M", "F", "M", "F", "M"],
        "weight": [1, 1, 1, 1, 1, 1],
    }
)

s = Sample.from_frame(sample_df, id_column="id", weight_column="weight", standardize_types=False)
t = Sample.from_frame(target_df, id_column="id", weight_column="weight", standardize_types=False)
a = s.set_target(t).adjust(method="null")

print(a.covars(formula="age + gender + age:gender").kld().round(6))
```

```text
                        age  age:gender[T.M]  gender[F]  gender[M]  mean(kld)
source
self               0.232348         0.120078        0.0        0.0   0.117476
unadjusted         0.232348         0.120078        0.0        0.0   0.117476
unadjusted - self  0.000000         0.000000        0.0        0.0   0.000000
```

## Upgrade checklist (v0.16 → v0.19)

Before upgrading production workflows, verify these items:

- Replace removed `Sample.*` convenience methods with `weights()/covars()/outcomes()` accessors.
- If all-zero weights are possible, handle `ValueError` in affected diagnostics.
- If CLI pipelines relied on implicit outcomes, validate the v0.17 `ignore_columns` behavior.
- If you build large/high-precision raking marginals, review capped-expansion behavior from v0.18.

## Community and next steps

Thanks to all contributors and users who tested edge cases and helped shape this release cycle.

To dive deeper into internals and architecture:

- [ARCHITECTURE.md](https://github.com/facebookresearch/balance/blob/main/ARCHITECTURE.md)
- [docs/architecture/architecture_0_19_0.md](https://github.com/facebookresearch/balance/blob/main/docs/architecture/architecture_0_19_0.md)

## Get started with v0.19.0

Upgrade:

    python -m pip install -U balance

Resources:

- **Website:** https://import-balance.org/
- **GitHub:** https://github.com/facebookresearch/balance
- **Architecture docs:** https://github.com/facebookresearch/balance/blob/main/ARCHITECTURE.md
- **Detailed 0.19 architecture notes:** https://github.com/facebookresearch/balance/blob/main/docs/architecture/architecture_0_19_0.md
- **Changelog:** https://import-balance.org/docs/docs/changelog/
- **Tutorials:** https://import-balance.org/docs/tutorials/
- **Blog:** https://import-balance.org/blog/
- **Paper:** [balance – a Python package for balancing biased data samples](https://arxiv.org/abs/2307.06024)

Need help?

- **Ask questions:** https://github.com/facebookresearch/balance/issues/new?template=support_question.md
- **Report bugs:** https://github.com/facebookresearch/balance/issues/new?template=bug_report.md
- **Request features:** https://github.com/facebookresearch/balance/issues/new?template=feature_request.md

We welcome your feedback, questions, and contributions as we continue making **balance** the go-to tool for survey statistics and bias adjustment in Python!
