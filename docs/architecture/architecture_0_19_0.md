All existing tests pass, plus 4500+ lines of new test code added (3 entirely new test files totaling 4505 lines, plus significant expansions of existing test files).

## Architectural Summary — balance 0.19.0

This release refactors the entire balance object model from a monolithic `Sample` class (~2383 lines) into a clean three-class inheritance hierarchy while maintaining 100% backward compatibility.

### Files Changed vs D97942143

**42 files changed, 15576 insertions(+), 2935 deletions(-)** across the full stack.

**Core source files (before → after line counts):**

```
┌──────────────────────────────┬─────────┬─────────┬───────────────────────────────────────┐
│ File                         │ Before  │ After   │ What changed                          │
├──────────────────────────────┼─────────┼─────────┼───────────────────────────────────────┤
│ sample_class.py              │ 2383    │  240    │ Gutted: all logic moved to base       │
│                              │         │         │ classes. Now thin facade with          │
│                              │         │         │ __new__, from_frame, deepcopy,         │
│                              │         │         │ backward-compat aliases only.          │
├──────────────────────────────┼─────────┼─────────┼───────────────────────────────────────┤
│ balance_frame.py             │    0    │ 1921    │ NEW file. Adjustment orchestrator:     │
│                              │ (new)   │         │ adjust, set_target, set_unadjusted,    │
│                              │         │         │ covars/weights/outcomes factories,      │
│                              │         │         │ summary, diagnostics, has_target,       │
│                              │         │         │ is_adjusted, model, guard methods.      │
├──────────────────────────────┼─────────┼─────────┼───────────────────────────────────────┤
│ sample_frame.py              │    0    │ 1379    │ NEW file. DataFrame container with     │
│                              │ (new)   │         │ _column_roles dict, explicit           │
│                              │         │         │ covar_columns param, column-role        │
│                              │         │         │ metadata and BalanceDFSource protocol.  │
├──────────────────────────────┼─────────┼─────────┼───────────────────────────────────────┤
│ balancedf_class.py           │ 2987    │ 3324    │ 497 lines changed. Protocol member     │
│                              │         │         │ rename (weight_column→weight_series),   │
│                              │         │         │ BalanceDFSource protocol added,         │
│                              │         │         │ links_override support, snake_case.     │
├──────────────────────────────┼─────────┼─────────┼───────────────────────────────────────┤
│ summary_utils.py             │    0    │  554    │ NEW file. Extracted summary &          │
│                              │ (new)   │         │ diagnostics builders (_build_summary,   │
│                              │         │         │ _build_diagnostics) from sample_class.  │
└──────────────────────────────┴─────────┴─────────┴───────────────────────────────────────┘
```

**Other source files changed:**

| File | Changes |
|------|---------|
| `__init__.py` | New exports (BalanceFrame, SampleFrame, BalanceDFSource), print→logger.info (12 lines) |
| `cli.py` | Param renames (`ignore_columns` → `ignored_columns`) (29 lines) |
| `stats_and_plots/impact_of_weights_on_outcome.py` | Guard + weight accessor renames (19 lines) |
| `stats_and_plots/weighted_comparisons_stats.py` | 98 lines changed — refactored summary/diagnostics helpers |
| `utils/input_validation.py` | Minor type narrowing updates (9 lines) |
| `stats_and_plots/weighted_comparisons_plots.py` | Minor plot-related updates (4 lines) |
| `weighting_methods/rake.py` | Minor updates (3 lines) |
| `BUCK` | 3 lines — build target updates |

**Test files changed (15):**
(Line counts below are total lines changed per `sl diff --stat`, not net additions.)

| File | Changes |
|------|---------|
| `test_balance_frame.py` | **NEW** — 2252-line BalanceFrame-level tests |
| `test_sample_frame.py` | **NEW** — 1192-line SampleFrame tests |
| `test_e2e_from_tutorials.py` | **NEW** — 1061-line end-to-end regression tests |
| `test_balancedf.py` | 949 lines changed — BalanceDF protocol + links_override tests |
| `test_sample.py` | 426 lines changed — updated for new architecture |
| `test_sample_internal.py` | 334 lines changed — internal API tests |
| `test_sample_diagnostics_helper.py` | 138 lines changed — summary/diagnostics refactor |
| `test_stats_and_plots.py` | 112 lines changed — weight accessor renames |
| `test_rake.py` | 49 lines changed — weight accessor renames |
| `conftest.py` | +46 lines — expanded shared pytest fixtures (existed on D97942143 with 18 lines, now 64 lines) |
| `test_ipw.py` | 24 lines changed — weight accessor renames |
| `test_cli.py` | 8 lines changed — param renames |
| `test_adjust_null.py` | 8 lines changed — minor renames |
| `test_util_pandas_utils.py` | 6 lines changed — minor renames |
| `test_util_data_transformation.py` | 4 lines changed — minor renames |

**Other files:**

| File | Changes |
|------|---------|
| `CLAUDE.md` | Updated with new architecture documentation (202 lines) |
| `CHANGELOG.md` | Rewritten for 0.19.0 release (405 lines) |
| `pyproject.toml` | Build config updates |
| `scripts/make_docs.sh` | Doc generation script updates |
| `run_balance_tests.sh` | Test infrastructure improvements |
| `tutorials/balance_quickstart_new_api.ipynb` | New API tutorial |
| `website/docs/tutorials/index.mdx` | Tutorial index update |
| `website/docs/tutorials/quickstart_new_api.mdx` | New API tutorial docs |
| `tutorials/balance_quickstart.ipynb` | Expanded quickstart tutorial (1087 lines) |
| `tutorials/balance_ascii_plots.ipynb` | Minor update (2 lines) |
| `website/yarn.lock` | Dependency lock file updates (18 lines) |
| `parent_balance/ARCHITECTURE.md` | **NEW** — architecture documentation (186 lines) |
| `parent_balance/docs/architecture/architecture_0_19_0.md` | **NEW** — detailed 0.19.0 architecture (1082 lines) |
| `parent_balance/README.md` | Minor update (5 lines) |

---

### Diagram 1: Class Hierarchy — Before vs After

**BEFORE (0.18.x): Monolithic Sample**

```
                         ┌─────────────────────────────────────────────────────┐
                         │                     Sample                          │
                         │  (standalone class, ~2383 lines, ALL logic here)    │
                         │                                                     │
                         │  Class-level attrs (initialized to None):           │
                         │    _df = None                                       │
                         │    id_column = None                                 │
                         │    weight_column = None     ← pd.Series             │
                         │    _outcome_columns = None  ← pd.DataFrame          │
                         │    _ignored_column_names = None ← list[str]         │
                         │    _adjustment_model = None ← dict                  │
                         │    _df_dtypes = None                                │
                         │    _links = None  ← defaultdict(list)               │
                         │                                                     │
                         │  Construction:                                      │
                         │    __init__() ← stack inspection guard              │
                         │    from_frame() ← 200+ lines, all validation        │
                         │                                                     │
                         │  DataFrame access:                                  │
                         │    df (property) ← reconstructs via concat          │
                         │    _covar_columns() / _covar_columns_names()        │
                         │    _special_columns() / _special_columns_names()    │
                         │    ignored_columns() → pd.DataFrame | None          │
                         │    model_matrix()                                   │
                         │                                                     │
                         │  BalanceDF view factories:                           │
                         │    covars(formula) → BalanceDFCovars(self)           │
                         │    weights() → BalanceDFWeights(self)               │
                         │    outcomes() → BalanceDFOutcomes(self) | None      │
                         │                                                     │
                         │  Linking & adjustment:                              │
                         │    set_target(target) → deepcopy + _links["target"] │
                         │    set_unadjusted(s) → deepcopy + _links["unadj"]  │
                         │    has_target() → bool                              │
                         │    is_adjusted() → bool                             │
                         │    adjust(target, method) → deepcopy + weighting    │
                         │    set_weights(weights)                             │
                         │                                                     │
                         │  Diagnostics:                                       │
                         │    summary() → str                                  │
                         │    diagnostics() → pd.DataFrame                     │
                         │    model() → dict | None                            │
                         │    _check_if_adjusted()                             │
                         │    _no_target_error()                               │
                         │    _check_outcomes_exists()                         │
                         │                                                     │
                         │  I/O:                                               │
                         │    to_csv(), to_download()                          │
                         │    keep_only_some_rows_columns()                    │
                         │    __str__(), __repr__()                            │
                         └─────────────────────────────────────────────────────┘
```

**AFTER (0.19.0): Inheritance Hierarchy**

```
                                        object
                                       /      \
                                      /        \
                    ┌─────────────────┐          ┌─────────────────┐
                    │   SampleFrame   │          │  BalanceFrame   │
                    │  (1379 lines)   │          │  (1921 lines)   │
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

**Side-by-side: Where each responsibility lives**

```
┌──────────────────────────┬─────────────────────┬──────────────────────────────┐
│      Responsibility      │   OLD (0.18.x)      │      NEW (0.19.0)            │
├──────────────────────────┼─────────────────────┼──────────────────────────────┤
│ DataFrame storage        │ Sample._df          │ SampleFrame._df              │
│ Column-role metadata     │ Sample (ad-hoc)     │ SampleFrame._column_roles    │
│ ID column                │ Sample.id_column    │ SampleFrame._id_column_name  │
│ Weight column            │ Sample.weight_column│ SampleFrame._weight_column_name │
│ Outcome columns          │ Sample._outcome_cols│ SampleFrame._column_roles    │
│ Ignored columns          │ Sample._ignored_... │ SampleFrame._column_roles    │
│ Type standardization     │ Sample.from_frame() │ SampleFrame.from_frame()     │
│ Weight guessing          │ Sample.from_frame() │ SampleFrame.from_frame()     │
│ ID validation            │ Sample.from_frame() │ SampleFrame.from_frame()     │
│ covars()/weights()/etc.  │ Sample              │ BalanceFrame                 │
│ set_target()             │ Sample              │ BalanceFrame                 │
│ set_unadjusted()         │ Sample              │ BalanceFrame                 │
│ adjust()                 │ Sample              │ BalanceFrame                 │
│ summary()/diagnostics()  │ Sample              │ BalanceFrame                 │
│ has_target/is_adjusted   │ Sample (methods)    │ BalanceFrame (_CallableBool) │
│ model                    │ Sample.model()      │ BalanceFrame.model (property)│
│ _links dict              │ Sample._links       │ BalanceFrame._links          │
│                          │                     │  (defaultdict(list))         │
│ set_weights()            │ Sample              │ BalanceFrame                 │
│                          │                     │  (delegates to _sf_sample)   │
│ to_csv()/to_download()   │ Sample              │ BalanceFrame                 │
│ keep_only_some_rows_cols │ Sample              │ BalanceFrame                 │
│ model_matrix()           │ Sample              │ BalanceFrame                 │
│ Construction guard       │ Sample.__init__     │ Sample.__new__               │
│ Factory method           │ Sample.from_frame() │ Sample.from_frame()          │
│                          │                     │  → SampleFrame.from_frame()  │
│                          │                     │  → cls._create()             │
└──────────────────────────┴─────────────────────┴──────────────────────────────┘
```

---

### Diagram 2: SampleFrame Internals — Column Classification

**Private Attributes**

```
SampleFrame instance
│
├── _df : pd.DataFrame
│   └── The actual data (all columns present)
│
├── _id_column_name : str
│   └── Name of the ID column (e.g. "id")
│
├── _weight_column_name : str | None
│   └── Name of the active weight column (e.g. "weight")
│
├── _column_roles : dict[str, list[str]]
│   │   Explicit metadata tracking which columns serve which purpose.
│   │   Keys:
│   ├── "covars"   → ["age", "gender", "os"]
│   ├── "weights"  → ["weight"]
│   ├── "outcomes" → ["happiness"]
│   ├── "predicted"→ ["pred_happiness"]
│   ├── "ignored"  → ["notes", "timestamp"]
│   └── (empty lists for unused roles)
│
├── _weight_metadata : dict[str, Any]
│   └── Per-column provenance metadata (e.g. {"method": "ipw"})
│
├── _links : dict
│   └── {} (plain dict, initially empty; BalanceFrame has its own _links)
│
└── _df_dtypes : pd.Series | None
    └── Original dtypes before standardization (for change warnings)
```

**Column Classification in `from_frame()`**

```
Input DataFrame: [id, age, gender, os, happiness, notes, weight]

Each column is assigned to exactly ONE role:

  ┌───────────┬──────────────────────────────────────────────────────────┐
  │  Column   │  Role Assignment                                        │
  ├───────────┼──────────────────────────────────────────────────────────┤
  │  id       │  → id_column          (explicit or auto-detected)       │
  │  weight   │  → weight_column      (explicit or auto-detected)       │
  │  happiness│  → outcome_columns    (user must specify explicitly)    │
  │  notes    │  → ignored_columns    (user must specify explicitly)    │
  │  age      │  → covar_columns      ┐                                │
  │  gender   │  → covar_columns      ├─ INFERRED: everything left     │
  │  os       │  → covar_columns      ┘  after removing the above      │
  └───────────┴──────────────────────────────────────────────────────────┘

  covar_columns = all columns − {id, weight, outcomes, ignored, predicted_outcomes}

  Key: covariates are the RESIDUAL — every column that isn't explicitly
  assigned to another role becomes a covariate. Ignored columns are
  EXCLUDED from covariates (that's the whole point of ignoring them).
```

**Auto-Detection Logic**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    from_frame() Auto-Detection                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ID Column:                                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 1. If id_column provided → use it                           │   │
│  │ 2. Else try id_column_candidates (default: ["id"])          │   │
│  │ 3. Validate: not null, unique (if check_id_uniqueness)      │   │
│  │ 4. Cast to string if needed                                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Weight Column:                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 1. If weight_column provided → use it                       │   │
│  │ 2. Else if "weight" in columns → use "weight"               │   │
│  │ 3. Else if "weights" in columns → use "weights"             │   │
│  │ 4. Else create new column "weight" = 1.0                    │   │
│  │ 5. Validate: not null, numeric, non-negative                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Type Standardization (if standardize_types=True):                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Int64 → float64,  Int32 → float32                           │   │
│  │ int64 → float64,  int32 → float32                           │   │
│  │ int16 → float16,  int8 → float16                            │   │
│  │ string → object (pandas < 3.0 only)                         │   │
│  │ pandas.NA → numpy.nan                                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Covariate Inference:                                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ covars = all columns − id − weight − outcomes − ignored     │   │
│  │                      − predicted_outcomes                    │   │
│  │                                                              │   │
│  │ If covar_columns explicitly provided:                        │   │
│  │   validate they exist, no overlap with reserved columns      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

**DataFrame View Properties**

```
SampleFrame properties that expose filtered views of _df:

  sf._df  (full DataFrame, all columns)
    │
    ├── sf.df → self._df.copy()
    │           (simple copy of full DataFrame, NOT pd.concat)
    │
    ├── sf.df_covars → _df[covar_columns]
    │
    ├── sf.df_outcomes → _df[outcome_columns]  (or None)
    │
    ├── sf.df_weights → _df[[weight_column_name]]  (DataFrame)
    │
    ├── sf.df_ignored → _df[ignored_columns]  (or None)
    │
    ├── sf.id_column → _df[_id_column_name]  (Series)
    │
    └── sf.weight_series → _df[_weight_column_name]  (Series)
```

---

### Diagram 3: BalanceFrame Internal Structure & Property Delegation

```
┌──────────────────────────────────────────────────────────────────────┐
│                         BalanceFrame                                  │
│                                                                      │
│  Private attributes (set in _create()):                              │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ _sf_sample : SampleFrame                                      │  │
│  │   └── The responder's data (DataFrame + column roles)         │  │
│  │                                                                │  │
│  │ _sf_sample_pre_adjust : SampleFrame                           │  │
│  │   └── Snapshot of _sf_sample before adjust() modifies weights │  │
│  │                                                                │  │
│  │ _sf_target : SampleFrame | None                               │  │
│  │   └── The target population data (set by set_target)          │  │
│  │                                                                │  │
│  │ _adjustment_model : dict | None                               │  │
│  │   └── Model info from weighting method (set by adjust)        │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Delegated properties (from _sf_sample via SampleFrame):             │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ .id_column   → self._sf_sample.id_column   (pd.Series)       │  │
│  │ .weight_series → self._sf_sample.weight_series (pd.Series)   │  │
│  │ ._df         → self._sf_sample._df                           │  │
│  │ ._covar_columns() → self._sf_sample._covar_columns()         │  │
│  │ ._outcome_columns → reimplements via _sf_sample._column_roles │  │
│  │                    and _sf_sample._df (NOT a simple delegate  │  │
│  │                    to _sf_sample._outcome_columns property)   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  OWN attributes (NOT delegated):                                     │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ ._links      → collections.defaultdict(list)  (set in _create)│  │
│  │ .df          → pd.concat(id, covars, outcomes, weights,       │  │
│  │                  ignored) — reconstructed, NOT from _sf_sample │  │
│  │ .set_weights()→ delegates to _sf_sample.set_weights();          │  │
│  │                  shared _sf_sample/_sf_sample_pre_adjust ref    │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Own properties:                                                     │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ .has_target  → _CallableBool  (bf.has_target / bf.has_target())│  │
│  │ .is_adjusted → _CallableBool                                  │  │
│  │     (impl: _sf_sample is not _sf_sample_pre_adjust)           │  │
│  │ .model       → dict | None  (@property)                       │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  BalanceDF view factories:                                           │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ .covars(formula) → BalanceDFCovars(self)                      │  │
│  │ .weights()       → BalanceDFWeights(self)                     │  │
│  │ .outcomes()      → BalanceDFOutcomes(self) | None             │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Linking methods:                                                    │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ .set_target(target)     → two paths:                          │  │
│  │   BF/Sample target: deepcopy(self), set _links["target"]     │  │
│  │   SF target: inplace (default), mutates self                 │  │
│  │ .set_unadjusted(other)  → new BF with _links["unadjusted"]   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Adjustment:                                                         │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ .adjust(method, **kwargs)  [weighting = black box]            │  │
│  │   → call weighting method (ipw/cbps/rake/poststratify/null)   │  │
│  │   → _build_adjusted_frame(result, method):                    │  │
│  │       deepcopy _sf_sample                                     │  │
│  │       freeze "weight_pre_adjust" (1st adj only)               │  │
│  │       N = _next_weight_action_number()                        │  │
│  │       add_weight_column("weight_adjusted_N",                  │  │
│  │         result["weight"], metadata={method, ...})             │  │
│  │       set_weights(result["weight"]) → overwrites              │  │
│  │         original weight col (keeps its name)                  │  │
│  │       _create(sample=new_sf, sf_target=target)                │  │
│  │       _sf_sample_pre_adjust = original pre-adjust             │  │
│  │       chain _links["unadjusted"] through history              │  │
│  │       store _adjustment_model                                 │  │
│  │       return new adjusted instance                             │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Guard methods:                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ _require_adjusted()  ← raises if not adjusted                 │  │
│  │ _require_target()    ← raises if no target                    │  │
│  │ _require_outcomes()  ← raises if no outcomes                  │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

### Diagram 4: Object Lifecycle — State Transitions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Object Lifecycle (0.19.0)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. CONSTRUCTION                                                            │
│  ┌──────────────────────────────────────────────────────┐                   │
│  │ Sample.from_frame(df, id_column=..., ...)           │                   │
│  │   │                                                  │                   │
│  │   ├─→ SampleFrame.from_frame(df, ...)               │                   │
│  │   │     Creates SampleFrame with:                    │                   │
│  │   │       _df, _id_column_name, _weight_column_name, │                   │
│  │   │       _column_roles, _links={}, _df_dtypes       │                   │
│  │   │                                                  │                   │
│  │   └─→ cls._create(sample=sf, sf_target=None)        │                   │
│  │         Creates BalanceFrame wrapping SampleFrame:    │                   │
│  │           _sf_sample = sf                             │                   │
│  │           _sf_sample_pre_adjust = sf                  │                   │
│  │           _sf_target = None                           │                   │
│  │           _adjustment_model = None                    │                   │
│  │           _links = defaultdict(list)                   │                   │
│  │         + if isinstance(instance, SampleFrame):        │                   │
│  │             copies 6 SF attrs onto instance:           │                   │
│  │             _df, _id_column_name, _column_roles,       │                   │
│  │             _weight_column_name, _weight_metadata,     │                   │
│  │             _df_dtypes  (needed for Sample MRO)        │                   │
│  └──────────────────────────────────────────────────────┘                   │
│          │                                                                  │
│          ▼                                                                  │
│  ┌────────────────────┐                                                    │
│  │  BARE SAMPLE       │  has_target=False, is_adjusted=False               │
│  │  _links = {}       │  Can: covars(), weights(), outcomes(), summary()   │
│  └────────┬───────────┘                                                    │
│           │                                                                 │
│           │ sample.set_target(target_sample)                                │
│           │   TWO PATHS:                                                    │
│           │   • BalanceFrame/Sample target → deepcopy(self), immutable      │
│           │   • SampleFrame target → inplace=True (default), mutates self  │
│           │     Also RESETS adjustment state if previously adjusted:        │
│           │       _sf_sample = _sf_sample_pre_adjust (reverts weights)     │
│           │       _adjustment_model = None                                 │
│           │   → sets _links["target"] = target                             │
│           ▼                                                                 │
│  ┌────────────────────┐                                                    │
│  │ SAMPLE WITH TARGET │  has_target=True, is_adjusted=False                │
│  │ _links = {         │  Can: all above + adjust(), diagnostics(),         │
│  │   "target": target │       comparative asmd/kld/emd/plot                │
│  │ }                  │                                                    │
│  └────────┬───────────┘                                                    │
│           │                                                                 │
│           │ sample.adjust(method="ipw", max_de=2)                          │
│           │   → [weighting method — black box]                              │
│           │   → _build_adjusted_frame(result, method):                     │
│           │       deepcopy _sf_sample                                      │
│           │       freeze "weight_pre_adjust" (1st adj only)                │
│           │       N = _next_weight_action_number()                         │
│           │       add_weight_column("weight_adjusted_N",                   │
│           │         result["weight"], metadata={method, ...})              │
│           │       set_weights(result["weight"]) → overwrites               │
│           │         original weight col (keeps its name)                   │
│           │       _create(sample=new_sf, sf_target=target)                │
│           │       new._sf_sample_pre_adjust = original pre-adjust         │
│           │       chain _links["unadjusted"] through history              │
│           │       new._links["target"] = target (preserved)               │
│           │       new._adjustment_model = {...}                            │
│           ▼                                                                 │
│  ┌────────────────────┐                                                    │
│  │ ADJUSTED SAMPLE    │  has_target=True, is_adjusted=True                 │
│  │ _links = {         │  Can: all above + summary() with full metrics,    │
│  │   "target": target,│       3-way comparisons (self vs unadj vs target) │
│  │   "unadjusted": pre│                                                    │
│  │ }                  │  Weights have been modified by the weighting method│
│  │ _adjustment_model  │  _sf_sample_pre_adjust still holds original state │
│  │   = {method, ...}  │                                                    │
│  └────────────────────┘                                                    │
│                                                                             │
│  Note: adjust() always returns a NEW object (via _create(), not deepcopy).  │
│  set_target() has TWO paths:                                                │
│    • BalanceFrame/Sample target → returns deepcopy (immutable)              │
│    • SampleFrame target → default inplace=True (mutates self)              │
│  __new__ uses stack inspection to block direct Sample() construction.       │
│  __deepcopy__ is overridden to ensure complete independent copies.          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Diagram 5: BalanceDF Linked-Samples Expansion

**How `.covars().asmd()` fans out across linked objects**

```
Given: adjusted_sample (has_target=True, is_adjusted=True)
       adjusted_sample._links = {"target": target, "unadjusted": pre_adjust}

Step 1: Create BalanceDFCovars view
──────────────────────────────────

  adjusted_sample.covars()
    │
    └─→ BalanceDFCovars(
           sample = adjusted_sample,    ← BalanceDFSource (the backing object)
           links = _build_links_dict(), ← {"target": ..., "unadjusted": ...}
           formula = None               ← optional model matrix formula
         )
         (internally calls super().__init__(sample._covar_columns(),
          sample, name="covars", links=links))

Step 2: Call .asmd() → triggers _call_on_linked()
──────────────────────────────────────────────────

  covars_view.asmd()
    │
    └─→ _apply_comparison_stat_to_BalanceDF(method="_asmd_BalanceDF")
          │
          └─→ _call_on_linked(method_name="_asmd_BalanceDF", ...)
                │
                │  First: _balancedf_child_from_linked_samples()
                │  Expands self into a dict of same-type views for ALL linked objects:
                │
                │  ┌─────────────────────────────────────────────────────────┐
                │  │                                                         │
                │  │  result = {                                             │
                │  │    "self": BalanceDFCovars(adjusted_sample),            │
                │  │                                                         │
                │  │    "target": BalanceDFCovars(target_sample),            │
                │  │      ↑ created by: target_sample.covars()               │
                │  │                                                         │
                │  │    "unadjusted": BalanceDFCovars(pre_adjust_sample),   │
                │  │      ↑ created by: pre_adjust_sample.covars()           │
                │  │  }                                                      │
                │  │                                                         │
                │  │  Key insight: uses self.__name ("covars") to call       │
                │  │  the same factory method on each linked object          │
                │  │                                                         │
                │  └─────────────────────────────────────────────────────────┘
                │
                │  Then: calls _asmd_BalanceDF(sample_bdf, target_bdf)
                │  for each pair (self vs target, self vs unadjusted):
                │
                │  ┌─────────────────────────────────────────────────────────┐
                │  │                                                         │
                │  │  _asmd_BalanceDF(                                       │
                │  │    sample_BalanceDF = covars["self"],                   │
                │  │    target_BalanceDF = covars["target"]                  │
                │  │  ) → pd.DataFrame of ASMD values                       │
                │  │                                                         │
                │  │  _asmd_BalanceDF(                                       │
                │  │    sample_BalanceDF = covars["self"],                   │
                │  │    target_BalanceDF = covars["unadjusted"]              │
                │  │  ) → pd.DataFrame of ASMD values                       │
                │  │                                                         │
                │  └─────────────────────────────────────────────────────────┘
                │
                └─→ pd.concat(all results, with "source" column)
                    │
                    └─→ Final output: DataFrame with columns like:
                        ┌──────────────────────────────────────────┐
                        │ source      │ variable     │ asmd       │
                        ├─────────────┼──────────────┼────────────┤
                        │ self        │ age[25-34]   │ 0.021      │
                        │ self        │ gender[F]    │ 0.097      │
                        │ target      │ age[25-34]   │ 0.000      │
                        │ target      │ gender[F]    │ 0.000      │
                        │ unadjusted  │ age[25-34]   │ 0.005      │
                        │ unadjusted  │ gender[F]    │ 0.375      │
                        └──────────────────────────────────────────┘
```

**Full expansion for different link states**

```
┌────────────────────┬────────────────────────────────────────────────────┐
│ Object State       │  _balancedf_child_from_linked_samples() returns    │
├────────────────────┼────────────────────────────────────────────────────┤
│                    │                                                    │
│ Bare sample        │  {"self": BalanceDFCovars(sample)}                │
│ (no links)         │  → 1 entry only                                   │
│                    │                                                    │
│ With target        │  {"self":   BalanceDFCovars(sample),              │
│                    │   "target": BalanceDFCovars(target)}              │
│                    │  → 2 entries                                       │
│                    │                                                    │
│ Adjusted           │  {"self":       BalanceDFCovars(adjusted),        │
│                    │   "target":     BalanceDFCovars(target),          │
│                    │   "unadjusted": BalanceDFCovars(pre_adjust)}     │
│                    │  → 3 entries — enables 3-way comparison           │
│                    │                                                    │
│ Works the same     │  Same pattern for:                                │
│ for all BalanceDF  │   .weights()._balancedf_child_...()               │
│ subtypes           │   .outcomes()._balancedf_child_...()              │
└────────────────────┴────────────────────────────────────────────────────┘
```

---

### Diagram 6: BalanceDFSource Protocol Satisfaction

```
┌──────────────────────────────────────────────────────────────────────┐
│              BalanceDFSource Protocol (runtime_checkable)             │
│                                                                      │
│  Required members (7):                                               │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  weight_series  → pd.Series                                   │  │
│  │  id_column      → pd.Series                                   │  │
│  │  _links         → dict[str, BalanceDFSource]                  │  │
│  │  _covar_columns() → pd.DataFrame                              │  │
│  │  _outcome_columns → pd.DataFrame | None                       │  │
│  │  set_weights(weights) → None                                  │  │
│  │  trim(ratio, percentile, ...) → BalanceDFSource               │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────┬───────────────────────────┘
                                           │
               ┌───────────────────────────┼───────────────────────────┐
               │                           │                           │
               ▼                           ▼                           ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│     SampleFrame      │  │    BalanceFrame       │  │       Sample         │
│                      │  │                       │  │                      │
│ weight_series:       │  │ weight_series:        │  │ Inherits from both   │
│  _df[_weight_col]    │  │  delegates to         │  │ → satisfies via MRO  │
│                      │  │  _sf_sample           │  │                      │
│ id_column:           │  │                       │  │                      │
│  _df[_id_col]        │  │ id_column:            │  │                      │
│                      │  │  delegates to         │  │                      │
│ _links:              │  │  _sf_sample           │  │                      │
│  plain dict {}       │  │                       │  │                      │
│                      │  │ _links:               │  │                      │
│ _covar_columns():    │  │  own defaultdict(list) │  │                      │
│  _df[covar_cols]     │  │  (NOT delegated)      │  │                      │
│                      │  │                       │  │                      │
│ _outcome_columns:    │  │ _covar_columns():     │  │                      │
│  _df[outcome_cols]   │  │  delegates to         │  │                      │
│                      │  │  _sf_sample           │  │                      │
│ set_weights():       │  │                       │  │                      │
│  updates _df         │  │ set_weights():        │  │                      │
│                      │  │  delegates to         │  │                      │
│                      │  │  _sf_sample           │  │                      │
└──────────────────────┘  └───────────────────────┘  └──────────────────────┘
       ✅ satisfies             ✅ satisfies              ✅ satisfies
       (directly)               (delegation)               (via inheritance)
```

---

### Diagram 7: BalanceDF Class Hierarchy and View Creation

```
┌──────────────────────────────────────────────────────────────────────┐
│                          BalanceDF (base)                             │
│                                                                      │
│  Stores:                                                             │
│    __sample : BalanceDFSource  (the backing object)                  │
│    __df : pd.DataFrame         (the role-specific data)              │
│    __name : str                 ("covars"|"weights"|"outcomes")      │
│    __links_override : dict | None (explicit links, if provided)      │
│                                                                      │
│  Key methods:                                                        │
│    _balancedf_child_from_linked_samples() → dict of same-type views  │
│    _call_on_linked(method, ...) → concatenated results               │
│    _apply_comparison_stat_to_BalanceDF() → dispatches comparisons    │
│    mean() / std() / var_of_mean() / ci_of_mean()                    │
│    asmd() / kld() / emd() / cvmd() / ks()                          │
│    asmd_improvement()                                                │
│    model_matrix()                                                    │
│    plot()                                                            │
│    summary()                                                         │
│    to_csv() / to_download()                                          │
└──────────────┬──────────────────┬──────────────────┬─────────────────┘
               │                  │                  │
    ┌──────────┴──────┐ ┌────────┴────────┐ ┌───────┴────────┐
    │BalanceDFCovars  │ │BalanceDFWeights │ │BalanceDFOutcomes│
    │                 │ │                 │ │                 │
    │ _formula        │ │ _weights → None │ │ Outcomes-only   │
    │ model_matrix()  │ │ design_effect() │ │ methods:        │
    │   (override)    │ │ design_effect   │ │  relative_      │
    │ _kld_formula()  │ │   _prop()       │ │   response_     │
    │ from_frame()    │ │ r_indicator()   │ │   rates()       │
    │                 │ │ trim()          │ │  target_response│
    │                 │ │ summary()       │ │   _rates()      │
    │                 │ │   (override)    │ │  weights_impact │
    │                 │ │ plot() defaults │ │   _on_outcome_ss│
    │                 │ │   to KDE        │ │  outcome_sd_    │
    │                 │ │                 │ │   prop()         │
    │                 │ │                 │ │  outcome_        │
    │                 │ │                 │ │   variance_ratio│
    │                 │ │                 │ │  summary()      │
    │                 │ │                 │ │   (override)    │
    └─────────────────┘ └─────────────────┘ └─────────────────┘

    Created by:          Created by:          Created by:
    bf.covars()          bf.weights()         bf.outcomes()
    sf.covars()          sf.weights()         sf.outcomes()
```

---

### Diagram 8: Data Flow — from_frame() to adjust() to summary()

```
                          User Code
                             │
    ┌────────────────────────┴────────────────────────┐
    │                                                  │
    ▼                                                  ▼
Sample.from_frame(df)                    Sample.from_frame(target_df)
    │                                                  │
    ├─→ SampleFrame.from_frame()                      ├─→ SampleFrame.from_frame()
    │     validate, classify columns                   │     validate, classify columns
    │     standardize types                            │     standardize types
    │                                                  │
    ├─→ _create(sample=sf)                            ├─→ _create(sample=sf)
    │     wrap in BalanceFrame                         │     wrap in BalanceFrame
    │                                                  │
    ▼                                                  ▼
  sample                                            target
    │                                                  │
    └───────────┬──────────────────────────────────────┘
                │
                ▼
    sample.set_target(target)
    ┌──────────────────────────────────────┐
    │ new = deepcopy(sample)               │
    │ new._links["target"] = target        │
    │ return new                           │
    └──────────────┬───────────────────────┘
                   │
                   ▼
    sample_with_target.adjust(method="ipw")
    ┌──────────────────────────────────────────────────┐
    │ (does NOT deepcopy self — creates new instance    │
    │  from deepcopy of _sf_sample via _create())       │
    │                                                   │
    │ ┌───────────────────────────────────────────────┐ │
    │ │        Weighting Method (BLACK BOX)            │ │
    │ │  Input: sample covars, sample weights,         │ │
    │ │         target covars, target weights           │ │
    │ │  Output: {"weight": new_weights,               │ │
    │ │           "model": model_details}               │ │
    │ └───────────────────────────────────────────────┘ │
    │                                                   │
    │ → _build_adjusted_frame(result, method):          │
    │   1. deepcopy _sf_sample                          │
    │   2. freeze "weight_pre_adjust" (1st adj only)    │
    │   3. N = _next_weight_action_number()             │
    │   4. add_weight_column("weight_adjusted_N",       │
    │      result["weight"], metadata={method, ...})    │
    │   5. set_weights(result["weight"]) → overwrites   │
    │      original weight col (keeps its name)         │
    │   6. _create(sample=new_sf, sf_target=target)    │
    │   7. _sf_sample_pre_adjust = original pre_adjust │
    │   8. chain _links["unadjusted"] through history   │
    │   9. _links["target"] = target (preserved)        │
    │  10. store _adjustment_model                      │
    │  11. return new adjusted BalanceFrame              │
    └──────────────────┬───────────────────────────────┘
                       │
                       ▼
    adjusted.summary()
    ┌──────────────────────────────────────────────────┐
    │ covars_asmd = adjusted.covars().asmd()            │
    │   └─→ 3-way expansion (self, target, unadjusted)│
    │                                                   │
    │ covars_kld = adjusted.covars().kld(...)           │
    │   └─→ 3-way expansion                            │
    │                                                   │
    │ design_effect = adjusted.weights().design_effect()│
    │                                                   │
    │ outcome_means = adjusted.outcomes().mean()        │
    │   └─→ 3-way expansion                            │
    │                                                   │
    │ → formatted string with all diagnostics           │
    └──────────────────────────────────────────────────┘
```

---

### Diagram 9: _links Graph — Before vs After

**BEFORE (0.18.x)**

```
┌─────────────┐
│   Sample     │
│  (adjusted)  │
│              │
│ _links = {   │─────"target"──────→ ┌─────────────┐
│   "target":  │                      │   Sample     │
│   "unadj":   │                      │  (target)    │
│ }            │─────"unadjusted"──→ ┌┤─────────────┐│
└─────────────┘                      ││   Sample     ││
                                     ││ (pre-adjust) ││
                                     ││              ││
                                     ││ _links = {   ││──"target"──→ same target
                                     ││  "target":   ││
                                     ││ }            ││
                                     │└─────────────┘│
                                     └───────────────┘

  All nodes are Sample objects.
  _links is collections.defaultdict(list) but values are Samples, not lists.
```

**AFTER (0.19.0)**

```
┌──────────────────┐
│  Sample/         │
│  BalanceFrame     │
│  (adjusted)      │
│                  │
│ _links = {       │────"target"─────→ ┌──────────────────┐
│   "target":      │                    │  Sample/          │
│   "unadjusted":  │                    │  BalanceFrame     │
│ }                │                    │  (target)         │
│ (defaultdict on  │                    └──────────────────┘
│  BalanceFrame    │
│  itself)         │────"unadjusted"──→ ┌──────────────────┐
│                  │                    │  Sample/          │
│                  │                    │  BalanceFrame     │
│                  │                    │  (pre-adjust)     │
│                  │                    │                   │
│                  │                    │ _links = {        │──"target"──→ same target
│                  │                    │  "target": ...    │
│                  │                    │ }                 │
│                  │                    └──────────────────┘
└──────────────────┘

  _links lives directly on BalanceFrame (collections.defaultdict(list)),
  set in BalanceFrame._create(). NOT delegated to _sf_sample.
  SampleFrame has its own separate _links = {} (plain dict).
  Structurally identical graph to before, but _links now on BalanceFrame.
```

---

### Diagram 10: Sample.__new__ Construction Guard (brief)

```
Sample() called
    │
    ▼
__new__(cls, responders=None, target=None)
    │
    ├── If both None → check call stack
    │     │
    │     ├── caller in _ALLOWED_CALLERS (module-level frozenset):
    │     │               {"__deepcopy__", "__newobj__", "__newobj_ex__",
    │     │                "__reduce__", "__reduce_ex__", "deepcopy",
    │     │                "_reconstruct", "from_frame", "_create"}
    │     │     → ALLOW: return object.__new__(cls)
    │     │
    │     └── caller is anything else
    │           → BLOCK: raise NotImplementedError
    │                    "Use Sample.from_frame() instead"
    │
    └── If args provided → ALLOW (legacy path)

Purpose: Forces users to use Sample.from_frame() factory method.
         Internal paths (deepcopy, pickle via __newobj__/__newobj_ex__/
         __reduce__/__reduce_ex__, _reconstruct, from_frame, _create)
         are whitelisted via _ALLOWED_CALLERS frozenset (module-level).
```

---

### Additional Implementation Details

**BalanceFrame.__new__ constructor** (lines 248-280 of balance_frame.py):
(Note: BalanceFrame also has a no-op __init__ at lines 282-289.)

BalanceFrame also has a `__new__` that supports both:
- Public construction: `BalanceFrame(sample=sf)` → calls `_create(sample=sf)`
- Internal paths (deepcopy): when `sample` is None, returns bare `object.__new__(cls)`

Unlike Sample's guard (which uses stack inspection to whitelist callers),
BalanceFrame.__new__ uses simple argument checks: if `sample is None` and
`sf_target is None`, it returns a bare object for deepcopy support. No stack
inspection is involved.

**_build_links_dict()** (balance_frame.py):

Bridge method that constructs a links dictionary suitable for BalanceDF views.
Builds the dict directly from `_sf_target` and `_sf_sample_pre_adjust` (NOT
from `self._links`). Passed as `links=` override to covars()/weights()/outcomes(),
enabling the linked-sample expansion pattern shown in Diagram 5.

**conftest.py** (expanded, tests/):

Shared pytest fixtures (64 lines, expanded from 18 lines on master) providing
reusable SampleFrame and BalanceFrame instances for test modules. The original
file only set matplotlib/plotly non-interactive backends. Now also includes
pytest fixtures, reducing boilerplate across test_sample_frame.py,
test_balance_frame.py, and test_sample_internal.py.

**BalanceFrame convenience properties** (balance_frame.py):

`df_responders`, `df_target`, `df_responders_unadjusted` properties expose
the responder/target/pre-adjust DataFrames directly. `df_all` concatenates
all three with a `"source"` column for easy comparison.

`_validate_covariate_overlap()` is a static method called during construction
and `set_target()` to ensure responders and target share at least one covariate.

**BalanceFrame ↔ Sample conversion methods** (balance_frame.py):

`BalanceFrame.from_sample(sample)` converts a Sample (with target) to a
BalanceFrame. `BalanceFrame.to_sample()` converts back to a Sample. These
go through Sample, not directly to/from SampleFrame. For SampleFrame
extraction, use `SampleFrame.from_sample(sample)` or `Sample.to_sample_frame()`.


**summary_utils.py** (NEW, 554 lines):

Extracted from sample_class.py. Contains `_build_summary()` and
`_build_diagnostics()` — the two main functions that assemble human-readable
summary strings and diagnostics DataFrames. Also contains
`_concat_metric_val_var()` which is re-exported from sample_class.py for
backward compatibility. This separation keeps BalanceFrame focused on
orchestration and avoids circular imports between balance_frame.py and
sample_class.py.

**weighted_comparisons_stats.py** (98 lines changed):

Refactored to support summary_utils extraction. Functions that were
previously called inline from Sample are now importable standalone.

**Sample backward-compat aliases** (sample_class.py, lines 163-181):

Sample provides two backward-compat property aliases for internal code
that previously accessed `._sample_frame` and `._balance_frame`:
- `_sample_frame` (property): alias for `_sf_sample`. The setter also
  updates `_sf_sample_pre_adjust` (keeping them in sync for new Samples).
- `_balance_frame` (property): returns `self` if `has_target()`, else None.
  The setter is a no-op since Sample IS a BalanceFrame.

**BalanceFrame backward-compat aliases** (balance_frame.py, lines 404-421):

Three read-only properties for old code that accessed `.responders`,
`.target`, `.unadjusted`:
- `responders` → `_sf_sample` (the responder SampleFrame)
- `target` → `_sf_target` (the target SampleFrame or None)
- `unadjusted` → `_sf_sample_pre_adjust` if adjusted, else None

These are marked for removal in a future diff.

**Large-target diagnostic warning** (balance_frame.py, lines 788-805):

`adjust()` detects when the target population is much larger than the sample
(>10x and >100k rows) and emits a `logger.warning()` noting that in this regime
the target's contribution to variance becomes negligible, and standard errors
will be driven almost entirely by the sample (similar to one-sample inference).
This is a pre-adjustment diagnostic — it does NOT block the adjustment.

**High-cardinality feature detection** (balance_frame.py, lines 807-847):

`adjust()` detects high-cardinality features in both sample and target
covariates before calling the weighting method. Uses
`_detect_high_cardinality_features()` from `balance.util` on both
`resp_covars` and `target_covars`, merges results (taking max
`unique_count` per column), sorts by count descending, and emits a
`logger.warning()` with formatted details (column name, unique count,
unique ratio, missing-values flag). This is a pre-adjustment diagnostic —
it does NOT block the adjustment.

**SampleFrame weight management methods** (sample_frame.py):

SampleFrame exposes a suite of weight-management methods that
`_build_adjusted_frame()` relies on internally:
- `add_weight_column(name, values, metadata)` — adds a new weight column
  to `_df` and registers it in `_column_roles["weights"]` (lines 1207+)
- `set_active_weight(column_name)` — switches which weight column is
  considered "active" (returned by `df_weights`) (lines 1142-1169)
- `rename_weight_column(old_name, new_name)` — renames a weight column in
  `_df`, `_column_roles`, `_weight_column_name`, and `_weight_metadata`
  (lines 1171-1205)
- `set_weight_metadata(column, metadata)` — stores arbitrary provenance
  metadata for a weight column (lines 1087-1117)
- `weight_metadata(column)` — retrieves metadata for a weight column
  (lines 1119-1140)

**Unified weight history tracking** (`_build_adjusted_frame`, balance_frame.py lines 577-682):

`_build_adjusted_frame()` uses a unified weight history tracking approach.
After each adjustment the SampleFrame accumulates weight columns:

| After        | Weight columns                                    | Active   |
|--------------|---------------------------------------------------|----------|
| Before adj.  | weight                                            | weight   |
| 1st adjust   | weight, weight_pre_adjust, weight_adjusted_1      | weight   |
| 2nd adjust   | weight, weight_pre_adjust, weight_adjusted_1, _2  | weight   |
| 3rd adjust   | ... weight_adjusted_1, _2, _3                     | weight   |

- `weight_pre_adjust` — frozen copy of original design weights (1st adj only)
- `weight_adjusted_N` — output of the Nth adjustment step
- `weight` — always overwritten with the latest adjusted values

The active weight column always keeps its original name (via
`set_weights(result["weight"], use_index=True)` which overwrites the values).
This preserves the public API contract that `weight_series` always has
the same name, for both BalanceFrame and Sample subclasses.

For compound adjustments, `_sf_sample_pre_adjust` always points to the
very first baseline (not the last step), and `_links["unadjusted"]`
chains back through the full adjustment history.

**_CallableBool.__mul__ / __rmul__** (balance_frame.py, lines 100-104):

`_CallableBool` supports multiplication via `__mul__` and `__rmul__`,
delegating to `self._value * other` and `other * self._value`. This
allows `_CallableBool` instances to be used in numeric expressions
without calling them first (e.g., `self.is_adjusted * "Adjusted "` would
work). Note: the actual `__str__` method uses the callable style
`self.is_adjusted() * "Adjusted "` (where `__call__` returns a plain
bool), so `__mul__` is not exercised there — it is a convenience for
callers who use property-style access in arithmetic contexts.

  ---
  D97942143 was compared vs D98780772 by:
  1. Using sl cat -r D97942143 <file> to read each file's contents at D97942143
  2. Using sl cat -r D98780772 <file> to read each file's contents at D98780772
  3. Using sl diff -r D97942143 -r D98780772 --stat to get the diff stats
