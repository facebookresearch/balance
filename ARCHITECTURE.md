# balance — Architecture

This document describes the internal architecture of the **balance** Python package.
For usage and API documentation, see the [README](README.md) and [import-balance.org](https://import-balance.org/).

For LLM/AI coding assistant instructions, see [CLAUDE.md](../CLAUDE.md).

## Class Hierarchy

The core object model uses a three-class inheritance hierarchy:

```
                                    object
                                   /      \
                                  /        \
                ┌─────────────────┐          ┌─────────────────┐
                │   SampleFrame   │          │  BalanceFrame   │
                │  (1208 lines)   │          │  (1882 lines)   │
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
                          │        (237 lines)       │
                          │                          │
                          │  class Sample(           │
                          │    BalanceFrame,         │
                          │    SampleFrame):         │
                          │                          │
                          │  Thin backward-          │
                          │  compatible facade       │
                          └──────────────────────────┘

MRO: Sample → BalanceFrame → SampleFrame → object
```

### Key classes

- **`Sample`** (`sample_class.py`) — main user-facing object, constructed via `from_frame()` factory.
  Internally a thin inheritance wrapper: `class Sample(BalanceFrame, SampleFrame)`.
  All public API unchanged — `from_frame()`, `set_target()`, `adjust()`, `summary()`,
  `covars()`, `weights()`, `outcomes()`, etc. all work identically.
- **`SampleFrame`** (`sample_frame.py`) — DataFrame container with explicit column-role metadata
  (covars, weights, outcomes, predicted outcomes, ignored). Created via `SampleFrame.from_frame()`.
- **`BalanceFrame`** (`balance_frame.py`) — adjustment orchestrator pairing a responder
  `SampleFrame` with a target `SampleFrame`. Handles `adjust()`, `summary()`, `diagnostics()`,
  `covars()`, `weights()`, `outcomes()`, and all linked-source comparisons.
- **`BalanceDF`** hierarchy (`balancedf_class.py`) — role-specific views:
  - `BalanceDFCovars` — covariate access and statistics
  - `BalanceDFWeights` — weight diagnostics (design effect, density plots)
  - `BalanceDFOutcomes` — outcome analysis

### Where each responsibility lives

```
┌──────────────────────────┬──────────────────────────────┐
│      Responsibility      │      Class (0.19.0)          │
├──────────────────────────┼──────────────────────────────┤
│ DataFrame storage        │ SampleFrame._df              │
│ Column-role metadata     │ SampleFrame._column_roles    │
│ ID/weight columns        │ SampleFrame                  │
│ Type standardization     │ SampleFrame.from_frame()     │
│ covars()/weights()/etc.  │ BalanceFrame                 │
│ set_target()             │ BalanceFrame                 │
│ adjust()                 │ BalanceFrame                 │
│ summary()/diagnostics()  │ BalanceFrame                 │
│ has_target/is_adjusted   │ BalanceFrame (_CallableBool) │
│ _links dict              │ BalanceFrame                 │
│ Construction guard       │ Sample.__new__               │
│ Factory method           │ Sample.from_frame()          │
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

- `stats_and_plots/` — statistical summaries, weighted comparisons (ASMD), plots (seaborn/plotly/ASCII)
- `utils/` — data transformations, input validation, model matrix (patsy), pandas helpers, file/logging utils
- `datasets/` — simulated data generators and sample CSVs
- `adjustment.py` — weight trimming (mean ratio, percentile winsorization)
- `cli.py` — command-line interface (`BalanceCLI`)
- `summary_utils.py` — diagnostics and summary builders
- `testutil.py` — test fixtures and helpers

## File layout

Core: `sample_class.py` (Sample), `sample_frame.py` (SampleFrame), `balance_frame.py` (BalanceFrame), `balancedf_class.py` (BalanceDF views), `adjustment.py` (weight trimming), `cli.py`, `summary_utils.py`, `util.py`, `typing.py`.
Subdirs: `weighting_methods/`, `stats_and_plots/`, `utils/`, `datasets/`.
OSS mirror: `parent_balance/` (tests, pyproject.toml, CHANGELOG.md, website).

## Architectural change history

- **[0.19.0 — Three-class refactor](docs/architecture/architecture_0_19_0.md)**: Monolithic `Sample` class (~2383 lines) refactored into `SampleFrame` + `BalanceFrame` + thin `Sample` facade. Detailed diagrams of class hierarchy, column classification, object lifecycle, BalanceDF expansion, and data flow.
