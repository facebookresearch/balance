---
title: Balance v0.20.0 - reusable fitted weighting workflows, poststratify formulas, and a major API evolution

authors:
  - name: Soumyadip Sarkar
    title: Independent Researcher
    url: https://www.linkedin.com/in/neuralsorcerer
  - name: Tal Galili
    title: Machine Learning Engineer @ Meta
    url: https://www.linkedin.com/in/tal-galili-5993085/

tags: [python, open-source, survey-statistics, package-update]
hide_table_of_contents: true
---

🎉 **balance v0.20.0 is out!**

### What is balance?

[**balance**](https://pypi.org/project/balance/) is a Python package (from Meta) for dealing with biased samples when estimating population-level quantities. It supports common survey-weighting workflows (e.g., IPW, CBPS, rake, poststratify), diagnostics, and CLI-based pipelines.

### In one sentence: what changed since 0.15.0?

From **0.16.0 → 0.20.0**, balance added stronger diagnostics and pandas 3 support, improved CLI/data-handling behavior, introduced the `r_indicator`, shipped a large architecture refactor (`SampleFrame` + `BalanceFrame`) with backward compatibility, and then expanded into reusable fit/predict weighting workflows with clearer, stricter APIs.

[![balance_logo_horizontal](https://raw.githubusercontent.com/facebookresearch/balance/main/website/static/img/balance_logo/PNG/Horizontal/balance_Logo_Horizontal_FullColor_RGB.png)](https://import-balance.org/)

<!--truncate-->

## The big picture for this release line (0.16 → 0.20)

If you're upgrading multiple versions at once, this is the key storyline:

1. **0.16.0** focused on diagnostics and compatibility (outcome-weight impact diagnostics, categorical metric handling improvements, pandas 3 support).
2. **0.17.0** tightened CLI behavior and improved ASCII plotting/reporting.
3. **0.18.0** added representativeness tooling (`r_indicator`) and introduced deprecations that prepared for API cleanup.
4. **0.19.0** delivered the largest internal/API evolution: `SampleFrame` + `BalanceFrame`, while keeping `Sample` backward compatible.
5. **0.20.0** built on that foundation: reusable model fitting/prediction workflows, poststratify formula support, accessor consistency, and stricter validation.

## Highlights of v0.20.0

### 1) Reusable fitted workflows on `BalanceFrame`

`BalanceFrame` now supports a sklearn-style pattern with:

- `fit(...)`
- `design_matrix(...)`
- `predict_proba(...)`
- `predict_weights(...)`

This introduces a cleaner train/score workflow for weighting models; the runnable holdout example below uses IPW.

### 2) Formula support for poststratification

`poststratify(...)` and `adjust(method="poststratify", ...)` now accept `formula=` (as an alternative to `variables=`), with strict validation.

Supported syntax is intentionally constrained for poststratification cell definitions:

- Supported: `:`, `.`, `-`, optional leading `~`
- Rejected: `+`, `*`

That restriction is deliberate: poststratification defines **joint cells**, and rejecting additive/main-effects style operators helps prevent model-style formula misunderstandings.

### 3) Clearer accessor semantics

Accessor naming is now consistent:

- `*_column` → column name (`str`)
- `*_series` → data (`pd.Series`)
- `df_*` → DataFrame view

Notably, `id_column` now returns the **name** (string), while `id_series` returns the ID data.

### 4) Safer behavior and stricter validation

Two notable correctness/safety improvements in 0.20:

- Unknown kwargs passed to `poststratify(...)` now raise `TypeError` instead of being silently ignored.
- Replacing target data on already-adjusted objects now warns in-place because it resets adjustment state.

## Version-by-version details (0.16.0 → 0.20.0)

## 0.16.0 (2026-02-09)

**Major additions**

- Outcome-weight impact diagnostics (`y*w0` vs `y*w1`) with confidence intervals.
- Pandas 3.x compatibility updates.
- KLD/EMD/CVMD/KS comparisons on raw categorical covariates (with NA indicators).
- Raw-covariate IPW fitting (`use_model_matrix=False`) for custom estimators.
- Configurable ID-column candidate support in ID inference.
- `formula=` support in `BalanceDF.model_matrix()`.

**Breaking behavior**

- Several weight diagnostics now raise `ValueError` when all weights are zero.

## 0.17.0 (2026-03-17)

**Major additions**

- `ascii_comparative_hist` and improved comparative ASCII plotting.
- `ascii_plot_dist(..., comparative=...)` switch.

**Breaking behavior**

- CLI default behavior changed: unmentioned columns now go to `ignore_columns` (not `outcome_columns`).

**Robustness improvements**

- Better model-matrix handling, diagnostics input normalization, and stricter parsing of comma-separated CLI column lists.

## 0.18.0 (2026-03-24)

**Major additions**

- Public `r_indicator(sample_p, target_p)` + `sample.weights().r_indicator()`.

**Deprecations (removed in 0.19.0)**

- Legacy `Sample` convenience methods were deprecated in favor of `weights()/covars()/outcomes()` accessor methods.

**Important bug fix**

- In raking proportion helpers, expansion now uses capped allocation behavior to avoid very large LCM-driven expansions that could cause memory blow-ups.

## 0.19.0 (2026-04-06)

**Major architecture release**

- Added `SampleFrame` (data + explicit column-role metadata).
- Added `BalanceFrame` (adjustment orchestration over sample + target).
- Refactored `Sample` to inherit from both new classes while preserving backward compatibility.

**Additional capabilities**

- Compound/sequential adjustments.
- Conversions among `Sample`, `SampleFrame`, and `BalanceFrame`.
- Formula support in covariate diagnostics flows.

**Breaking behavior**

- Removal of methods that had been deprecated in 0.18.0.

## 0.20.0 (2026-04-26)

**Major additions**

- Added fitted-model workflow methods on `BalanceFrame` (`fit`, `design_matrix`, `predict_proba`, `predict_weights`).
- Poststratify `formula=` support and strict formula validation.
- `set_as_pre_adjust()` to establish a new pre-adjust baseline.
- `set_fitted_model(...)` for fitted-model transfer workflows.

**Breaking behavior**

- `id_column` now returns a column name (`str`) consistently.
- Unknown `poststratify(...)` kwargs now raise `TypeError`.

## Working examples

The snippets below were executed against **balance 0.20.0** in this repo. The output blocks show the values printed by the snippets.

### Example 1: Fit once, then predict weights on holdout data (IPW)

In this example, we:

1. Build three `SampleFrame` objects: training sample, target population, and holdout sample.
2. Fit an IPW model once on the training `BalanceFrame` (sample + target).
3. Print the fitted/training adjusted weights.
4. Reuse the fitted model to predict weights on a separate holdout `BalanceFrame` with the same covariate schema.

This demonstrates the core train/score workflow introduced in the 0.20 line for fitted models.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from balance.sample_frame import SampleFrame
from balance.balance_frame import BalanceFrame

sample_df = pd.DataFrame(
    {
        "id": ["1", "2", "3", "4", "5", "6"],
        "age": [20, 25, 30, 35, 40, 45],
        "gender": ["F", "F", "M", "M", "F", "M"],
        "weight": [1.0] * 6,
    }
)
target_df = pd.DataFrame(
    {
        "id": ["11", "12", "13", "14", "15", "16", "17", "18"],
        "age": [22, 27, 32, 37, 42, 47, 52, 57],
        "gender": ["F", "M", "F", "M", "F", "M", "F", "M"],
        "weight": [1.0] * 8,
    }
)
holdout_df = pd.DataFrame(
    {
        "id": ["101", "102", "103"],
        "age": [29, 41, 53],
        "gender": ["F", "M", "F"],
        "weight": [1.0, 1.0, 1.0],
    }
)

sf = SampleFrame.from_frame(
    sample_df,
    id_column="id",
    weight_column="weight",
    covar_columns=["age", "gender"],
    standardize_types=False,
)
tf = SampleFrame.from_frame(
    target_df,
    id_column="id",
    weight_column="weight",
    covar_columns=["age", "gender"],
    standardize_types=False,
)
hf = SampleFrame.from_frame(
    holdout_df,
    id_column="id",
    weight_column="weight",
    covar_columns=["age", "gender"],
    standardize_types=False,
)

bf = BalanceFrame(sample=sf, target=tf)
fitted = bf.fit(method="ipw", model=LogisticRegression(random_state=0, max_iter=200), inplace=False)

print(fitted.weights().df["weight"].round(3).tolist())
print(fitted.predict_weights(data=BalanceFrame(sample=hf, target=tf)).round(3).tolist())
```

Output (from the two `print(...)` lines):

```text
[1.689, 1.127, 1.479, 1.083, 1.539, 1.083]
[2.709, 2.582, 2.709]
```

### Example 2: Poststratification with `formula=`

Here we create a tiny sample/target pair with two categorical variables (`gender` and `age_group`) and run:

- `method="poststratify"`
- `formula="gender:age_group"`

The interaction formula defines poststratification cells by the joint distribution of those two variables. The printed output is the adjusted weight vector for the four sample rows.

```python
import pandas as pd
from balance.sample_class import Sample

sample_df = pd.DataFrame(
    {
        "id": ["1", "2", "3", "4"],
        "gender": ["F", "F", "M", "M"],
        "age_group": ["18-34", "35+", "18-34", "35+"],
        "weight": [1.0, 1.0, 1.0, 1.0],
    }
)
target_df = pd.DataFrame(
    {
        "id": ["11", "12", "13", "14", "15", "16"],
        "gender": ["F", "F", "F", "M", "M", "M"],
        "age_group": ["18-34", "18-34", "35+", "18-34", "35+", "35+"],
        "weight": [1.0] * 6,
    }
)

sample = Sample.from_frame(sample_df, id_column="id", weight_column="weight", standardize_types=False)
target = Sample.from_frame(target_df, id_column="id", weight_column="weight", standardize_types=False)
adj = sample.adjust(target, method="poststratify", formula="gender:age_group")
print(adj.weights().df["weight"].round(3).tolist())
```

Output (from `print(...)`):

```text
[2.0, 1.0, 1.0, 2.0]
```

### Example 3: All-zero weight diagnostics now raise

This snippet intentionally passes an all-zero weight vector to `design_effect(...)`.

The raised `ValueError` is expected behavior from 0.16.0 onward for diagnostics that require at least one positive weight.

```python
import pandas as pd
from balance.stats_and_plots.weights_stats import design_effect

design_effect(pd.Series([0.0, 0.0, 0.0]))
```

Output:

```text
ValueError: weights (w) must include at least one positive value.
```

### Example 4: `r_indicator(...)`

This computes the `r_indicator` from two small propensity vectors (`sample_p` and `target_p`) and prints the scalar result rounded to 6 decimals.

Use this as a minimal, direct check that the function is available and producing numeric output in your environment.

```python
import numpy as np
from balance.stats_and_plots.weighted_comparisons_stats import r_indicator

sample_p = np.array([0.2, 0.4, 0.6, 0.8])
target_p = np.array([0.3, 0.5, 0.7, 0.9])
print(round(float(r_indicator(sample_p, target_p)), 6))
```

Output (from `print(...)`):

```text
0.510102
```

## Migration checklist (0.15.x → 0.20.0)

- Replace deprecated/removed `Sample` convenience methods with `weights()/covars()/outcomes()` flows.
- Review CLI assumptions around implicit outcome columns (`ignore_columns` default behavior changed in 0.17.0).
- Audit ID accessor usage (`id_column` now name; `id_series` for values).
- Validate poststratify calls for unknown kwargs and formula syntax.
- Handle possible all-zero weight errors in diagnostics workflows.

## Documentation to keep handy

- New API tutorial (`SampleFrame` / `BalanceFrame`): https://import-balance.org/docs/tutorials/quickstart_new_api/
- Quickstart tutorial: https://import-balance.org/docs/tutorials/quickstart/
- CLI tutorial: https://import-balance.org/docs/tutorials/balance_cli_tutorial/
- Architecture deep dive for 0.19.0: https://github.com/facebookresearch/balance/blob/main/docs/architecture/architecture_0_19_0.md


## Get Started with v0.20.0

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


Thanks to everyone using and improving balance. Contributions and feedback are always welcome.
