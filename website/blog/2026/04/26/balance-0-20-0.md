---
title: Balance v0.20.0 - reusable fit/predict weighting workflows, richer diagnostics, and a refactored core

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

### What's new since 0.15.0 -> 0.20.0?

Highlights: **balance now offers a reusable, sklearn-style fit/predict workflow for survey weighting**: fit a weighting model once on one sample-and-target pair, then apply it to a different (e.g., larger or later-arriving) cohort with a single call. Under the hood, this is supported by a substantial architectural refactor — `Sample` is now a thin facade over two new classes (`SampleFrame` for data + column-role metadata, `BalanceFrame` for adjustment orchestration), all while keeping the existing `Sample` API fully backward compatible.

Around that core change, this release line also brings stronger diagnostics (confidence intervals for the bias the weights manage to remove from an outcome, the `r_indicator` representativeness statistic, comparative ASCII plots for LLM/CLI workflows), formula support in places that previously only accepted variable lists, pandas 3.x compatibility, and a number of validation/robustness improvements.

[![balance_logo_horizontal](https://raw.githubusercontent.com/facebookresearch/balance/main/website/static/img/balance_logo/PNG/Horizontal/balance_Logo_Horizontal_FullColor_RGB.png)](https://import-balance.org/)

<!--truncate-->

## Reusable fit/predict workflows

Before 0.20.0, fitting weights and using them was a single act: you pointed `Sample.adjust(...)` at a sample/target pair and got back an adjusted object. If new responders arrived next week, or you wanted to score a holdout (often, much larger) cohort with the same model, you had to refit from scratch.

The balance library now supports an explicit, reusable workflow - modeled after sklearn:

- `fit(...)` — fit a weighting model (IPW, CBPS, or poststratify)
- `predict_weights(data=...)` — generate weights on a separate `BalanceFrame` with the same covariate schema
- Also:
    - `set_fitted_model(fitted)` - for transferring a fitted model from one frame to another.
    - `predict_proba(...)` — predicted propensities on training or new data
    - `design_matrix(...)` — inspect the design matrix used for fitting/scoring

Each fit stores everything needed to reconstruct weights on new data: training design weights, trimming options, CBPS coefficients, poststratify cell-ratio tables, NA handling, and so on.
This is the workflow that was hardest to express before, and it is now a one-liner:

For the rest of this post we'll use balance's bundled simulated data — `sim_data_01` — which contains a 1000-row biased sample and a 10000-row target with covariates (`age_group`, `gender`, `income`) and an outcome (`happiness`). We split the sample into a 100-row training slice and a 900-row holdout slice, fit the IPW model on training, and transfer it to the holdout. We'll keep `bf_holdout` around as our running example for every diagnostic in the rest of this post (you could equally use `Sample` here — the `BalanceFrame` API mirrors it):

```python
from sklearn.linear_model import LogisticRegression
from balance import BalanceFrame, SampleFrame, load_data

target_df, sample_df = load_data()
train_df = sample_df.iloc[:100].copy()
holdout_df = sample_df.iloc[100:].copy()

covar_cols = ["age_group", "gender", "income"]
train_sf   = SampleFrame.from_frame(train_df,   id_column="id", covar_columns=covar_cols, outcome_columns=["happiness"], standardize_types=False)
target_sf  = SampleFrame.from_frame(target_df,  id_column="id", covar_columns=covar_cols, outcome_columns=["happiness"], standardize_types=False)
holdout_sf = SampleFrame.from_frame(holdout_df, id_column="id", covar_columns=covar_cols, outcome_columns=["happiness"], standardize_types=False)

bf_train = BalanceFrame(sample=train_sf, target=target_sf)
fitted = bf_train.fit(method="ipw", model=LogisticRegression(random_state=0, max_iter=200), inplace=False)

bf_holdout = BalanceFrame(sample=holdout_sf, target=target_sf)
bf_holdout.set_fitted_model(fitted)

print(bf_holdout.summary())
# Can also use fitted.predict_weights(data=bf_holdout) to only get the weights for the holdout
```

Output:

```text
Adjustment details:
    method: ipw
    weight trimming mean ratio: 20
Covariate diagnostics:
    Covar ASMD reduction: 5.4%
    Covar ASMD (7 variables): 0.336 -> 0.318
    Covar mean KLD reduction: 37.2%
    Covar mean KLD (3 variables): 0.188 -> 0.118
Weight diagnostics:
    design effect (Deff): 1.292
    effective sample size proportion (ESSP): 0.774
    effective sample size (ESS): 696.5
Outcome weighted means:
            happiness
source
self           50.330
target         56.278
unadjusted     48.487
Model performance: Model proportion deviance explained: 0.572
```

## A refactored core (without breaking the old API)

The fit/predict workflow above is possible because `Sample` was split internally into two foundational classes:

- **`SampleFrame`** — a DataFrame container that explicitly tracks which columns are covariates, weights, outcomes, predicted outcomes, and ignored. Created via `SampleFrame.from_frame()`. Adds first-class weight-history management (`add_weight_column()`, `set_active_weight()`, `rename_weight_column()`, `set_weight_metadata()`).
- **`BalanceFrame`** — an adjustment orchestrator that pairs a responder `SampleFrame` with a target `SampleFrame`. It owns `set_target()`, `adjust()`, `summary()`, `diagnostics()`, `covars()/weights()/outcomes()`, plus the new fit/predict surface.

`Sample` now inherits from both (`Sample → BalanceFrame → SampleFrame → object`). **No public `Sample` API changes**: every existing call site continues to work. There are also bidirectional conversions (`Sample.to_sample_frame()`, `BalanceFrame.from_sample(...)`, `BalanceFrame.to_sample()`) so you can move between the convenience facade and the more composable classes as needed.

A practical bonus of the refactor: `adjust()` can now be called multiple times on the same object, **compounding adjustments** (e.g., IPW first to fix broad imbalances, then rake on a specific marginal). The original unadjusted baseline is preserved for diagnostics like `asmd_improvement()`. There is also a new `set_as_pre_adjust()` to lock in the current state as a fresh pre-adjust baseline.

For the deep dive into the architectural change (class hierarchy, `_links` graph, weight-history tracking), see the [0.19.0 architecture document](https://github.com/facebookresearch/balance/blob/main/docs/architecture/architecture_0_19_0.md).

## Stronger diagnostics

### How much bias do the weights actually remove from your outcome?

ASMD tells you whether covariates are better balanced after weighting. It does *not* directly tell you how much the weights changed your outcome estimate, or whether that change is statistically meaningful. balance now exposes paired outcome-weight impact diagnostics — comparing `y * w_unadjusted` to `y * w_adjusted` — with p-value and confidence intervals, available through `BalanceDFOutcomes`, `Sample.diagnostics()`, and the CLI (`--weights_impact_on_outcome_method`):

```python
print(bf_holdout.outcomes().summary())
```

Output:

```text
1 outcomes: ['happiness']
Mean outcomes (with 95% confidence intervals):
source      self  target  unadjusted           self_ci         target_ci     unadjusted_ci
happiness  50.33  56.278      48.487  (49.275, 51.384)  (55.961, 56.595)  (47.551, 49.424)

Weights impact on outcomes (t_test):
           mean_yw0  mean_yw1  mean_diff  diff_ci_lower  diff_ci_upper  t_stat  p_value      n
outcome
happiness    48.487     50.33      1.842         -0.214          3.898   1.759    0.079  900.0

Response rates (relative to number of respondents in sample):
   happiness
n      900.0
%      100.0
Response rates (relative to notnull rows in the target):
    happiness
n      900.0
%        9.0
Response rates (in the target):
    happiness
n    10000.0
%      100.0
```

The "Weights impact on outcomes" block is the new piece: `mean_yw0` and `mean_yw1` are the unweighted vs. weighted outcome means, `mean_diff` is their difference, with a CI and a paired-sample test on the per-row impact `y * (w1 − w0)`. Here the weights nudge `happiness` from 48.5 → 50.3 (closer to the target's 56.3), but the CI still includes zero and `p ≈ 0.08` — moderate but not statistically significant evidence that this propensity model meaningfully shifted the outcome estimate.

### `r_indicator` for representativeness

The R-indicator (Schouten et al.) is a single scalar summarizing how representative a sample is of a target, derived from the spread of estimated response propensities. balance now ships a validated `r_indicator(sample_p, target_p)` (Eq. 2.2.2 over concatenated propensity vectors, with input validation), plus a convenience `sample.weights().r_indicator()`:

```python
print(bf_holdout.weights().r_indicator())
```

```text
0.5094452588289895
```

R takes values in `[0, 1]`: **1** means the sample is fully representative of the target (response propensities are identical for everyone — i.e., constant), and **0** is the worst case (maximum spread in propensities, indicating strong selection). The ~0.51 value here flags substantial non-response bias in this sample/target pair, consistent with the heavily skewed gender and age_group marginals shown elsewhere in this post.

### Comparative ASCII plots (LLM- and terminal-friendly)

Distribution diagnostics now have a proper text-based mode, available straight off `BalanceFrame.covars().plot(library="balance", dist_type="hist_ascii")`. Categorical variables render as grouped horizontal bars, numeric variables as a comparative histogram (population → adjusted → sample, with `█` shared with population, `▒` excess, `]` deficit). This makes it easy to inspect adjustments in plain text — particularly useful for CLI workflows, log captures, and LLM agents reading reports.

```python
print(bf_holdout.covars().plot(library="balance", dist_type="hist_ascii"))
```

Output (one section per covariate — categoricals as grouped bars, numeric `income` as a comparative histogram with auto-binning via Sturges' rule):

```text
=== age_group (categorical) ===

Category | population  adjusted  sample
         |
18-24    | ████████████████████████ (19.7%)
         | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (28.6%)
         | ▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐ (48.6%)

25-34    | █████████████████████████████████████ (29.7%)
         | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (36.9%)
         | ▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐ (30.2%)

35-44    | █████████████████████████████████████ (29.9%)
         | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (18.6%)
         | ▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐ (15.7%)

45+      | █████████████████████████ (20.6%)
         | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (15.8%)
         | ▐▐▐▐▐▐▐ (5.6%)

Legend: █ population  ▒ adjusted  ▐ sample
Bar lengths are proportional to weighted frequency within each dataset.

=== gender (categorical) ===

Category | population  adjusted  sample
         |
Female   | ██████████████████████████████████████████ (50.0%)
         | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (28.7%)
         | ▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐ (29.2%)

Male     | ██████████████████████████████████████████ (50.0%)
         | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (71.3%)
         | ▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐ (70.8%)

Legend: █ population  ▒ adjusted  ▐ sample
Bar lengths are proportional to weighted frequency within each dataset.

=== income (numeric, comparative) ===

Range            | population (%) | adjusted (%)      | sample (%)
-------------------------------------------------------------------------
[0.00, 8.57)     | ████████ 49.0  | ████████▒▒▒▒ 72.6 | ████████▒▒▒▒ 73.4
[8.57, 17.14)    | ████ 23.1      | ███] 20.3         | ███] 19.1
[17.14, 25.71)   | ██ 13.2        | █] 5.0            | █] 5.1
[25.71, 34.28)   | █ 7.3          | ] 1.3             | ] 1.6
[34.28, 42.85)   | █ 3.9          | ] 0.4             | ] 0.4
[42.85, 51.41)   | 1.8            | 0.1               | 0.1
[51.41, 59.98)   | 0.9            | 0.3               | 0.2
[59.98, 68.55)   | 0.4            | 0.0               | 0.0
[68.55, 77.12)   | 0.2            | 0.0               | 0.0
[77.12, 85.69)   | 0.1            | 0.0               | 0.0
[85.69, 94.26)   | 0.0            | 0.0               | 0.0
[94.26, 102.83)  | 0.0            | 0.0               | 0.0
[102.83, 111.40) | 0.0            | 0.0               | 0.0
[111.40, 119.97) | 0.0            | 0.0               | 0.0
[119.97, 128.54] | 0.0            | 0.0               | 0.0
-------------------------------------------------------------------------
Total            | 100.0          | 100.0             | 100.0

Key: █ = shared with population, ▒ = excess,    ] = deficit
```

`age_group` gets a clear correction — the sample's `18-24` over-representation (48.6% vs 19.7% in the population) is pulled down to 28.6% post-adjustment. `gender` barely moves (sample 29.2% Female → adjusted 28.7%, vs target 50%). `income` is similar between sample and adjusted because the IPW model didn't lean on it. This is exactly the kind of "where did the weights actually help and where didn't they?" diagnosis that's hard to read off a single ASMD number.

### Distribution distances on raw categoricals

`KLD`/`EMD`/`CVMD`/`KS` on `BalanceDF.covars()` now operate directly on raw categorical variables (with NA indicators), rather than requiring one-hot encoding. This produces more faithful comparisons for categorical covariates, especially with missing values:

```python
print(bf_holdout.covars().kld())
```

```text
                   age_group    gender    income  mean(kld)
source
self                0.056796  0.188210  0.109138   0.118048
unadjusted          0.268381  0.183109  0.112641   0.188044
unadjusted - self   0.211585 -0.005101  0.003503   0.069996
```

The `unadjusted - self` row is the per-variable improvement. Here, the IPW weights cut KLD on `age_group` from 0.27 → 0.06 (a substantial fix) but barely move `gender` or `income` — matching what we saw in the ASCII plot above. The same shape holds for `bf_holdout.covars().emd()`, `cvmd()`, and `ks()`.

## More flexibility in the workflow

### Sequential adjustments + formula support in poststratification

Two related changes that pair naturally. First, `adjust()` is now compounding: calling it on an already-adjusted `BalanceFrame` uses the current weights as design weights for the next adjustment, so you can layer methods (e.g. IPW first to soak up broad imbalances, then poststratify on a specific marginal for fine-tuning). Second, `poststratify(...)` and `BalanceFrame.adjust(method="poststratify", ...)` now accept `formula=` as an alternative to `variables=`. Only interaction-style operators are supported — `:`, `.`, `-`, optional leading `~`. Additive `+` and `*` are explicitly rejected, because poststratification defines cells by the *joint* distribution: `a + b`, `a * b`, and `a:b` would all yield identical cells, and rejecting `+`/`*` prevents users from silently writing what looks like a main-effects model.

Continuing from `bf_holdout` above (already IPW-adjusted), let's add a poststratify layer on the joint `age_group:gender` cells — the two covariates the IPW step left most imbalanced:

```python
adj = bf_holdout.adjust(method="poststratify", formula="age_group:gender")
print(adj.summary())
```

Output:

```text
Adjustment details:
    method: poststratify
Covariate diagnostics:
    Covar ASMD reduction: 47.9%
    Covar ASMD (7 variables): 0.336 -> 0.175
    Covar mean KLD reduction: 67.2%
    Covar mean KLD (3 variables): 0.188 -> 0.062
Weight diagnostics:
    design effect (Deff): 2.395
    effective sample size proportion (ESSP): 0.417
    effective sample size (ESS): 375.7
Outcome weighted means:
            happiness
source
self           55.453
target         56.278
unadjusted     48.487
```

The compounded adjustment cuts ASMD from 0.336 → 0.175 (vs. IPW alone's 0.336 → 0.318) and brings the weighted `happiness` mean from 50.3 (IPW only) up to 55.5 — within reach of the target's 56.3. Note `asmd_improvement()` is computed against the original *unadjusted* baseline, so it reflects total improvement across both adjustment steps, not just the last one. The cost is a higher design effect (2.4 vs. 1.3) and a smaller ESS, the usual trade-off for tighter cell-level matching.

Drilling into per-covariate balance:

```python
print(adj.covars().summary())
```

```text
source                  self  target  unadjusted         self_ci         target_ci   unadjusted_ci
age_group[T.25-34]     0.298   0.297       0.302   (0.26, 0.337)    (0.288, 0.306)  (0.272, 0.332)
age_group[T.35-44]     0.299   0.299       0.157  (0.254, 0.343)     (0.29, 0.308)   (0.133, 0.18)
age_group[T.45+]       0.208   0.206       0.056  (0.147, 0.269)    (0.198, 0.214)  (0.041, 0.071)
gender[Female]         0.500   0.455       0.292  (0.449, 0.551)    (0.445, 0.465)  (0.263, 0.322)
gender[Male]           0.500   0.455       0.708  (0.449, 0.551)    (0.445, 0.465)  (0.678, 0.737)
income                 6.710  12.738       6.257  (5.981, 7.438)  (12.482, 12.993)  (5.791, 6.724)
```

`age_group` is now nearly perfectly aligned with the target (the joint poststratify cells did their job), while `income` — which was *not* in the formula — remains skewed toward the lower end. That's the intended behavior: poststratification only reweights the cells you ask it to, so combining it with IPW lets you choose where to be exact and where to settle for a propensity-score approximation.

What about the `gender[Female] = 0.500` for `self` vs. `0.455` for `target`? It's a clean illustration of how poststratification handles missing cells. The simulated target has missing values in `gender` (`target_df["gender"].value_counts(dropna=False, normalize=True)` shows ~9% NaN: `Female 45.5% / Male 45.5% / NaN 9%`), but our holdout — `sample_df.iloc[100:]` — happens to contain none (the simulator only injects NaNs into rows 3–90 of `sample_df`, which sit in the *training* slice). Poststratify can only reweight cells that exist in the sample; with no NaN cell in the holdout, it allocates the full weight across just `Female` and `Male`, which produces `45.5% / 91% = 50%` for each — exactly matching the conditional Female/Male split *within* the non-NaN portion of the target. The 9% NaN mass in the target is unreachable from this sample, so the marginal stays at 45.5%/45.5% on the target side. (If your sample *does* contain NaNs, balance treats them as their own cell — see the `na_action` argument to `poststratify` for the modes.)

### Raw-covariate IPW for sklearn estimators with native categorical support

`Sample.adjust(method="ipw", use_model_matrix=False)` now fits propensity models directly on raw covariates (without building a one-hot model matrix). String/object/bool columns are converted to pandas `Categorical` so estimators like `HistGradientBoostingClassifier` (with `categorical_features="from_dtype"`) handle them natively. The categorical workflow requires scikit-learn ≥ 1.4 — under balance's current dependency pins (see [`pyproject.toml`](https://github.com/facebookresearch/balance/blob/main/pyproject.toml)), that's available by default on Python 3.12+; on Python 3.9–3.11 it requires opting into a newer sklearn. Keep `use_model_matrix=True` (the default) if you don't need raw-categorical fitting.

## Robustness, pandas 3, and safer defaults

- **Pandas 3.x compatibility** across the package — string/NA handling, dtype checks, and weight-dtype coercion were updated; tests now run cleanly on pandas 3.
- **Hare–Niemeyer allocation in raking** — the old proportion-expansion path could hit memory blow-ups when the LCM of marginal lengths grew very large. Expansion is now capped (default 10,000 rows) using largest-remainder allocation, which preserves totals exactly with minimal rounding error.
- **Consistent accessor naming** — `*_column` returns the column **name** (`str`), `*_series` returns the data (`pd.Series`), `df_*` returns a DataFrame view. In particular, `id_column` now returns the name (matching `weight_column`); use `id_series` for the data. The deprecated old behavior emits a `FutureWarning` until 2026-06-01.
- **Stricter validation**:
  - Unknown kwargs to `poststratify(...)` now raise `TypeError` instead of being silently ignored.
  - Replacing the target on an already-adjusted `BalanceFrame` warns in-place, since it resets the adjustment.
  - Weight diagnostics (`design_effect`, `nonparametric_skew`, `prop_above_and_below`, `weighted_median_breakdown_point`) raise `ValueError` if all weights are zero.
  - CLI column-list arguments trim whitespace and reject empty entries (e.g., `"id,,weight"`).
- **CLI default change** — columns not explicitly mentioned now flow to `ignore_columns` rather than being implicitly treated as outcomes.

## Migration checklist (0.15.x → 0.20.0)

- Replace removed `Sample` convenience methods (`design_effect`, `design_effect_prop`, `plot_weight_density`, `covar_means`, `outcome_sd_prop`, `outcome_variance_ratio`) with the `weights()/covars()/outcomes()` accessor flows.
- Audit `id_column` usage — it now returns a name; use `id_series` for the data.
- Review CLI assumptions: unmentioned columns now go to `ignore_columns`, not `outcome_columns`.
- Validate `poststratify(...)` calls for typo'd kwargs and formula syntax.
- Handle possible `ValueError` from weight diagnostics if all-zero weights are reachable in your pipeline.

For the full version-by-version breakdown of every change, deprecation, and bug fix, see the [CHANGELOG](https://import-balance.org/docs/docs/changelog/).

## Documentation to keep handy

- New API tutorial (`SampleFrame` / `BalanceFrame`): https://import-balance.org/docs/tutorials/quickstart_new_api/
- Quickstart tutorial: https://import-balance.org/docs/tutorials/quickstart/
- CLI tutorial: https://import-balance.org/docs/tutorials/balance_cli_tutorial/
- Architecture deep dive: https://github.com/facebookresearch/balance/blob/main/docs/architecture/architecture_0_19_0.md

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
