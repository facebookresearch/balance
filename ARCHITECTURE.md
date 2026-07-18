# balance — Architecture

This document describes the internal architecture of the **balance** Python package.
For usage and API documentation, see the [README](README.md) and [import-balance.org](https://import-balance.org/).

For LLM/AI coding assistant instructions, see [`.github/copilot-instructions.md`](.github/copilot-instructions.md).

## Class Hierarchy

The core object model uses a three-class inheritance hierarchy:

```
                                    object
                                   /      \
                                  /        \
                ┌─────────────────┐          ┌─────────────────┐
                │   SampleFrame   │          │  BalanceFrame   │
                │  (1373 lines)   │          │  (1918 lines)   │
                │                 │          │                 │
                │  DataFrame +    │◄─ ─ ─ ─ ┤  Adjustment     │
                │  column-role    │ composes │  orchestrator   │
                │  metadata       │ (via     │  (sample+target)│
                │                 │_sf_sample)│                │
                └────────┬────────┘          └────────┬────────┘
                          \                          /
                           \    MULTIPLE            /
                            \   INHERITANCE        /
                             \                    /
                          ┌───┴──────────────────┴───┐
                          │          Sample          │
                          │        (240 lines)       │
                          │                          │
                          │  class Sample(           │
                          │    BalanceFrame,         │
                          │    SampleFrame):         │
                          │                          │
                          │  Thin backward-          │
                          │  compatible facade       │
                          └──────────────────────────┘

MRO: Sample → BalanceFrame → SampleFrame → object

Key: BalanceFrame does NOT inherit from SampleFrame.
     It COMPOSES a SampleFrame instance via _sf_sample.
     Sample inherits from BOTH via multiple inheritance.
```

### Key classes

- **`Sample`** (`sample_class.py`) — main user-facing object, constructed via `from_frame()` factory.
  Internally a thin inheritance wrapper: `class Sample(BalanceFrame, SampleFrame)`.
  All public API unchanged — `from_frame()`, `set_target()`, `adjust()`, `summary()`,
  `covars()`, `weights()`, `outcomes()`, etc. all work identically.
- **`SampleFrame`** (`sample_frame.py`) — DataFrame container with explicit column-role metadata
  (covars, weights, outcomes, outcomes_hat, ignored). Created via `SampleFrame.from_frame()`.
  Provides weight management methods (`add_weight_column()`, `set_active_weight()`,
  `rename_weight_column()`, `set_weight_metadata()`), `set_weights()`, and `trim()`.
- **`BalanceFrame`** (`balance_frame.py`) — adjustment orchestrator pairing a responder
  `SampleFrame` with a target `SampleFrame`. Handles `adjust()`, `summary()`, `diagnostics()`,
  `covars()`, `weights()`, `outcomes()`, `set_weights()` (delegates to `_sf_sample`),
  `trim()` (delegates to `_sf_sample`), and all linked-source comparisons.
  Also exposes sklearn-style convenience methods for IPW workflows:
  `fit()`, `design_matrix()`, `predict_proba()`, and `predict_weights()`.
  Supports compound/sequential adjustments with unified weight history tracking.
- **`BalanceDF`** hierarchy (`balancedf_class.py`) — role-specific views:
  - `BalanceDFCovars` — covariate access and statistics
  - `BalanceDFWeights` — weight diagnostics (design effect, density plots)
  - `BalanceDFOutcomes` — outcome analysis
  - `BalanceDFOutcomesHat` — predicted-outcome (Ŷ) analysis (weighted mean / CI over the
    `outcomes_hat` columns); created via the `outcomes_hat()` factory on `SampleFrame` /
    `BalanceFrame` (returns `None` when there are no Ŷ columns) and exported from `balance`
  - `BalanceDFSource` (protocol) — 8 required members: `weight_series`, `id_series`,
    `_links`, `_covar_columns()`, `_outcome_columns`, `_outcomes_hat_columns`,
    `set_weights()`, `trim()`

### Accessor naming convention

All data-access properties follow a consistent naming pattern:

| Suffix | Returns | Examples |
|--------|---------|----------|
| `*_column` | Column name (`str`) | `id_column`, `weight_column` |
| `*_series` | Column data (`pd.Series`) | `id_series`, `weight_series` |
| `*_columns` | List of names (`list[str]`) | `covar_columns`, `outcome_columns`, `outcomes_hat_columns`, `weight_columns_all` |
| `df_*` | DataFrame | `df_covars`, `df_weights`, `df_outcomes`, `df_outcomes_hat`, `df_ignored` |

Note: the `_*` protocol accessors `_outcome_columns` and `_outcomes_hat_columns` return the column *data* (a `DataFrame | None`), not names — the names live on `outcome_columns` / `outcomes_hat_columns`.

**Migration warnings** (`FutureWarning`, will be removed after 2026-06-01):
- `id_column` — changed in 0.20.0 from returning data to returning the name. Use `id_series` for data.
- `weight_column` — changed in 0.19.0 from returning data to returning the name. Use `weight_series` for data.

### Where each responsibility lives

```
┌──────────────────────────┬──────────────────────────────┐
│      Responsibility      │          Class               │
├──────────────────────────┼──────────────────────────────┤
│ DataFrame storage        │ SampleFrame._df              │
│ Column-role metadata     │ SampleFrame._column_roles    │
│ outcomes_hat (Ŷ) data    │ SampleFrame (canonical)      │
│  (df_outcomes_hat /      │  add_outcomes_hat_column();  │
│   _outcomes_hat_columns) │  BalanceFrame delegates      │
│ Fit outcome model ĝ(X)   │ SampleFrame.fit_outcome_model│
│  (store on frame)        │  → _outcome_model /          │
│                          │    outcome_model (property)  │
│ Predict/persist Ŷ        │ SampleFrame.predict_outcomes │
│  from stored model       │  / fit_predict_outcomes      │
│ Fit outcome model via BF │ BalanceFrame.fit_outcome_model│
│  (delegate to responder) │  / fit_predict_outcomes      │
│                          │  → _sf_sample._outcome_model │
│ outcome_model (read)     │ BalanceFrame (property,      │
│                          │  delegates to _sf_sample)    │
│ Transfer ĝ to the target │ BalanceFrame.predict_outcomes│
│  produce Ŷ_T             │  (on="target"; deep-copies   │
│                          │  _sf_target before writing)  │
│ Apply a foreign fitted   │ BalanceFrame.                │
│  model (train/holdout)   │  set_fitted_outcome_model    │
│                          │  (shares fit by identity)    │
│ Outcome-model estimate   │ outcomes_hat().mean()        │
│  μ̂_OM (target row)       │  (raises if target Ŷ unpop.) │
│ IPW/Hájek estimate μ̂_IPW │ outcomes().mean() (self)     │
│ AIPW estimate μ̂_DR       │ BalanceFrame.aipw()          │
│  (doubly robust)         │  (μ̂_OM + IPW-wtd residuals)  │
│ ID/weight columns        │ SampleFrame                  │
│ Type standardization     │ SampleFrame.from_frame()     │
│ Weight management        │ SampleFrame (canonical)      │
│  (add/set/rename/trim)   │  BalanceFrame delegates      │
│ set_weights()            │ SampleFrame (canonical)      │
│                          │  BalanceFrame delegates to   │
│                          │  _sf_sample.set_weights()    │
│ trim()                   │ SampleFrame (canonical)      │
│                          │  BalanceFrame delegates      │
│ covars()/weights()/etc.  │ BalanceFrame                 │
│ set_target()             │ BalanceFrame                 │
│ adjust()                 │ BalanceFrame                 │
│ _build_adjusted_frame()  │ BalanceFrame                 │
│ _next_weight_action_no() │ BalanceFrame (shared counter │
│                          │  for adjusted_N/trimmed_N)   │
│ summary()/diagnostics()  │ BalanceFrame (→summary_utils)│
│ has_target/is_adjusted   │ BalanceFrame (_CallableBool) │
│ _links dict              │ BalanceFrame                 │
│                          │  (defaultdict(list))         │
│ model                    │ BalanceFrame (property)      │
│ to_csv()/to_download()   │ BalanceFrame                 │
│ model_matrix()           │ BalanceFrame                 │
│ Construction guard       │ Sample.__new__               │
│ Factory method           │ Sample.from_frame()          │
│                          │  → SampleFrame.from_frame()  │
│                          │  → cls._create()             │
└──────────────────────────┴──────────────────────────────┘
```

## The 5-step workflow

```python
from balance import Sample

# 1. Create Sample objects
sample = Sample.from_frame(sample_df, id_column="id", outcome_columns="outcome")
target = Sample.from_frame(target_df, id_column="id", weight_column="count")

# 2. Link sample to target population
sample = sample.set_target(target)

# 3. Pre-adjustment diagnostics
sample.covars().plot()           # Visual covariate balance check

# 4. Adjust (weight)
adjusted = sample.adjust(
    variables=["age", "gender", "os"],
    method="ipw",                # or "cbps", "poststratify", "rake"
    max_de=2,                    # cap design effect (ipw/cbps only)
)

# 5. Post-adjustment evaluation
adjusted.summary()               # Summary table
adjusted.covars().plot()          # Post-adjustment balance
adjusted.covars().asmd()          # ASMD per covariate
adjusted.weights().design_effect()          # Variance inflation factor
```

## Compound/sequential adjustments

`adjust()` can be called multiple times. Each call uses the previous step's weights as design weights, enabling multi-stage reweighting pipelines. Internally, `_build_adjusted_frame()` manages a unified weight history:

| After        | Weight columns in `_df`                           | Active   |
|--------------|---------------------------------------------------|----------|
| Before adj.  | weight                                            | weight   |
| 1st adjust   | weight, weight_pre_adjust, weight_adjusted_1      | weight   |
| 2nd adjust   | weight, weight_pre_adjust, weight_adjusted_1, _2  | weight   |
| After trim   | ... weight_adjusted_1, _2, weight_trimmed_3       | weight   |

- `weight_pre_adjust` — frozen copy of original design weights (1st adjustment only)
- `weight_adjusted_N` — output of the Nth adjustment step
- `weight_trimmed_N` — output of the Nth trim step
- `weight` — always overwritten with the latest values (keeps its original name)
- `_next_weight_action_number()` — shared counter across `weight_adjusted_N` and `weight_trimmed_N`

For compound adjustments, `_sf_sample_pre_adjust` always points to the very first baseline, and `_links["unadjusted"]` chains back through the full adjustment history.

## Fit-artifact workflow

`BalanceFrame.fit(method="ipw")` is an alias for `adjust(...)` that enables
`store_fit_matrices=True` and `store_fit_metadata=True` by default for the built-in
IPW method. By default `fit()` mutates `self` and returns `self` (sklearn-style
`inplace=True`); pass `inplace=False` for functional-style usage that returns
a new object. This stores fit-time artifacts in `model` so downstream calls can
reuse the exact training transformation/predictions without recomputing preprocessing:

- `design_matrix(on=...)` → stored model matrices (IPW only)
- `predict_proba(on=..., output=...)` → stored probabilities or link values (IPW only)
- `predict_weights()` → dispatches by method; IPW uses stored links + design weights

`set_fitted_model(fitted)` applies a fitted model from one BalanceFrame to another,
producing a fully adjusted holdout BalanceFrame for train/holdout-split workflows.
`predict_weights()` dispatches by the
model's `method` key, currently supporting `"ipw"` with extensibility for future
methods (CBPS, rake, poststratify).

When these artifacts are not stored (e.g. plain `adjust(method="ipw")`), the API
raises actionable errors that direct users to `fit(method="ipw")` or the explicit
`ipw(..., store_fit_matrices=True/store_fit_metadata=True)` flags.

## Outcome-model workflow (`outcome_models/`)

The `outcome_models/` package is the outcome-modelling counterpart to the IPW
fit-artifact workflow, on a separate axis: instead of a propensity model over a
sample-vs-target indicator, it fits a learner `ĝ(X) ≈ E[Y|X]` of an *observed
outcome* on covariates. The package ships **pure DataFrame functions**, and
`SampleFrame` wires them onto a frame as an sklearn-style trio (the standalone,
no-target fit/store step):

- `fit_outcome_model(covars_df, outcomes_df, *, sample_weight=None, model="auto", ...)`
  builds a design matrix via `build_design_matrix` (train mode), fits a regressor
  (continuous outcome) or classifier (binary outcome) per outcome column, and
  returns a stored model dict (`method="outcome_model"`, `fit`, `X_matrix_columns`,
  `fit_scaler`, `categorical_levels`, `fit_matrix_type`, `weighted`,
  `prediction_kind`, `perf`, …) mirroring the IPW model dict so the same replay works.
- `predict_outcome(model, new_covars_df)` rebuilds the design matrix in replay mode
  (`project_to_columns` + stored scaler + re-applied `categorical_levels`) and returns
  `ŷ` per outcome (`.predict` for a regressor, `P̂(Y=1)` for a classifier).

Preprocessing is learner-dependent (`use_model_matrix="auto"`): tree/boosting learners
use the native-categorical path on scikit-learn >= 1.4 (one-hot fallback on < 1.4),
linear learners use one-hot + `StandardScaler`; the matrix is densified for
`HistGradientBoosting*`.

On top of these primitives, `SampleFrame` (and, via the MRO, `Sample`) exposes the
sklearn-style trio, which stores the fitted model on the frame:

- `fit_outcome_model(*, model="auto", outcome_columns=None, variables=None, weighted=False, ..., inplace=True)`
  resolves the outcome column(s) (default: all `outcome_columns`), extracts `df_covars`
  (optionally restricted to the `variables=` subset) and the observed outcome(s),
  **drops rows with a missing outcome `Y`** (covariates/weights realign), and — when
  `weighted=True` (the default is unweighted) — aligns the active weight to the covariate index,
  fits via `fit_outcome_model`, and stores the model dict on `_outcome_model` (exposed via
  the read-only `outcome_model` property, mirroring `BalanceFrame.model`). Like sklearn's
  `fit`, it does **not** persist `outcomes_hat`; a re-fit drops any `<outcome>_hat` columns
  a prior `predict_outcomes` left behind so a stale Ŷ can't linger against a new model.
- `predict_outcomes(*, data=None, populate=True)` replays the stored model on this frame's
  covariates (or on `data`'s covariates when a `SampleFrame` is passed) and returns a
  `{"<outcome>_hat": ŷ}` DataFrame, persisting the `<outcome>_hat` columns via
  `add_outcomes_hat_column` when `populate=True`.
- `fit_predict_outcomes(*, populate=True, **fit_kwargs)` fits then predicts-on-self in one call.

The new `_outcome_model` frame state is initialised in `SampleFrame._create`, reference-shares
its fitted estimators on `SampleFrame.__deepcopy__` (immutable post-fit — the dict is shallow
copied, the estimators are kept by reference), and is synced onto `Sample` via
`BalanceFrame._sync_sampleframe_state_from_responder` (mirroring `_prediction_metadata`).

`BalanceFrame` orchestrates the *transfer to the target* and the *estimate* (the counterpart to
how it orchestrates `adjust()`/`set_fitted_model()` for weights). It **delegates** the fit to the
responder so the model has a single home that rides the lifecycle:

- `BalanceFrame.fit_outcome_model(*, target=None, inplace=True, **kw)` /
  `fit_predict_outcomes(...)` call `_sf_sample.fit_outcome_model(...)`, so the model lands on
  `_sf_sample._outcome_model`. Because a `Sample` is *both* a `BalanceFrame` and a `SampleFrame`,
  this `BalanceFrame` method takes MRO precedence over `SampleFrame.fit_outcome_model`, so a model
  fit on a `Sample` lands on `_sf_sample` (not the Sample's own inherited attribute) — this is what
  lets it **survive `adjust()`** (which deep-copies `_sf_sample`).
- `BalanceFrame.outcome_model` is a read-only property that **delegates to `_sf_sample`** (single
  source of truth; mirrors `df_outcomes_hat` / `_outcomes_hat_columns`, and parallels
  `BalanceFrame.model` for the weighting axis).
- `BalanceFrame.predict_outcomes(*, on="sample"|"target"|"both", populate=True)` replays the
  responder's stored model. `on="target"` (the default when a target is set) scores the target's
  covariates and populates its `<outcome>_hat` columns, **deep-copying `_sf_target` before writing**
  so the caller's target object is not mutated in place; `on="both"` returns a `(sample, target)`
  tuple (mirroring `predict_weights`/`design_matrix`).
- `BalanceFrame.set_fitted_outcome_model(fitted, *, inplace=True)` is the train/holdout transfer —
  the outcome-axis counterpart to `set_fitted_model` for weights. It copies an already-fitted
  outcome model from another frame (`fitted`, a `BalanceFrame`/`SampleFrame`/`Sample`) onto `self`'s
  responder **sharing the fitted estimators by identity** (a shallow copy of the model dict, not a
  refit or deep clone — `scored.outcome_model["fit"][c] is train.outcome_model["fit"][c]`), so a
  subsequent `predict_outcomes(on="target")` replays the transferred model on `self`'s own target for
  `μ̂_OM` on the holdout. It reuses the same `"matching sample covariate column names"` check as
  `set_fitted_model`, and **rejects a non-deterministic transfer** (a stored `transformations` of
  `quantize`/`fct_lump`, or `na_action="drop"`) that can't be replayed deterministically on a foreign
  frame; `inplace` matches `set_fitted_model`.
- The estimate is `μ̂_OM = outcomes_hat().mean()` — the **target** row = `Σ w_T ŷ_T / Σ w_T`.
  `BalanceFrame.outcomes_hat()` builds the view when *either* the responder or a linked source
  (target/unadjusted) carries `outcomes_hat`, and it **raises an actionable error** (pointing at
  `predict_outcomes(on="target")`) when a model is fit but the target's `outcomes_hat` is not
  populated, so the population estimate is never silently replaced by the responder's in-sample mean.
- Honest inference: `outcomes_hat().mean_with_ci(ci_method="bootstrap", n_bootstrap=200,
  random_seed=2020)` (the **default** `ci_method`) computes a percentile CI for `μ̂_OM` via a
  nonparametric bootstrap — resample the responders, refit `ĝ*` with the stored fit-configuration
  (and fit-weighting), predict on the **fixed** target, re-average — capturing the outcome-model
  estimation uncertainty the analytic `ci_of_weighted_mean` ignores. The reusable engine
  (`bootstrap_outcome_estimate`) lives in `outcome_models/` and keeps only the `B` scalar outputs;
  the override is a bespoke path on the BalanceFrame-backed view (it bypasses the linked-view
  machinery, which can't reach the learner) and **raises on a lone/target-less view**. Deterministic
  given `random_seed` (`numpy.random.default_rng`).
- `outcomes_hat().summary()` reports the estimator type and **scopes any doubly-robust statement to
  the fit weights** — never a blanket "doubly robust": a linear + intercept learner fit with
  non-uniform weights is reported as `"doubly robust w.r.t. weights <col>"` (the WLS special case),
  everything else (the non-linear default, a uniform-weight fit, or a no-intercept linear fit) as
  plain `"g-computation (not doubly robust)"`.

Lifecycle: `set_target()` **preserves** `_outcome_model` across its responder reset (a model fit
before *or* after `adjust()` is not lost when the target is replaced); `keep_only_some_rows_columns`
that **drops responder rows invalidates** the model (its `training_sample_index` no longer matches
the retained rows), while a column-only filter keeps it. A doubly-robust / AIPW estimator combining
`outcomes_hat` with the IPW weights is a later phase. See the design doc
[architecture_0_23_0.md](docs/architecture/architecture_0_23_0.md).

## Weighting methods (`weighting_methods/`)

| Method | File | When to use |
|--------|------|-------------|
| IPW | `ipw.py` | Default. Lasso-regularized logistic regression propensity scoring |
| CBPS | `cbps.py` | Recommended for production. Directly optimizes covariate balance |
| Rake | `rake.py` | When you only have marginal distributions (not joint) |
| Poststratify | `poststratify.py` | When you have population cell counts (joint distribution). Categorical variables only |
| Null | `adjust_null.py` | Passthrough (no adjustment) |

Key parameters across methods: `max_de` (design effect cap, default 1.5), `transformations` (override auto-transformations), `weight_trimming_mean_ratio` (trim extreme weights), `na_action` (handle NAs).

## Supporting modules

- `stats_and_plots/` — statistical summaries (weighted mean/var/sd/quantile, weighted R² via `weighted_r2`), weighted comparisons (ASMD), plots (seaborn/plotly/ASCII)
- `utils/` — data transformations, input validation, model matrix (patsy), pandas helpers, file/logging utils
- `datasets/` — simulated data generators and sample CSVs
- `adjustment.py` — weight trimming (mean ratio, percentile winsorization)
- `cli.py` — command-line interface (`BalanceCLI`)
- `summary_utils.py` — diagnostics and summary builders (`_build_summary()`, `_build_diagnostics()`), extracted from `sample_class.py`
- `testutil.py` — test fixtures and helpers

## File layout

In the open-source repo, the top-level structure is: `balance/` (package source), `tests/`, `tutorials/`, `website/`, `pyproject.toml`, `CHANGELOG.md`.

Within `balance/`, the core files are: `sample_class.py` (Sample), `sample_frame.py` (SampleFrame), `balance_frame.py` (BalanceFrame), `balancedf_class.py` (BalanceDF views), `adjustment.py` (weight trimming), `cli.py`, `summary_utils.py`, `util.py`, `typing.py`.
Subdirs: `weighting_methods/`, `outcome_models/` (outcome-model learner + fit/predict), `stats_and_plots/`, `utils/`, `datasets/`.

## Detailed architecture documentation

- **[Three-class architecture deep dive](docs/architecture/architecture_0_19_0.md)**: Detailed diagrams of the class hierarchy, column classification, object lifecycle, BalanceDF expansion, and data flow.
