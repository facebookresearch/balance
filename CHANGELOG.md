# 0.19.0 (Unreleased - TBD)

## Breaking Changes

- **Removed `Sample.design_effect()`** — use `sample.weights().design_effect()` instead.
  Deprecated since 0.18.0.
- **Removed `Sample.design_effect_prop()`** — use `sample.weights().design_effect_prop()` instead.
  Deprecated since 0.18.0.
- **Removed `Sample.plot_weight_density()`** — use `sample.weights().plot()` instead.
  Deprecated since 0.18.0.
- **Removed `Sample.covar_means()`** — use `sample.covars().mean()` instead
  (with `.rename(index={'self': 'adjusted'}).reindex(['unadjusted', 'adjusted', 'target']).T` for the same format).
  Deprecated since 0.18.0.
- **Removed `Sample.outcome_sd_prop()`** — use `sample.outcomes().outcome_sd_prop()` instead.
  Deprecated since 0.18.0.
- **Removed `Sample.outcome_variance_ratio()`** — use `sample.outcomes().outcome_variance_ratio()` instead.
  Deprecated since 0.18.0.

## New Features

- **Compound/sequential adjustments** — `adjust()` can now be called multiple
  times on the same object. Each call uses the current (previously adjusted)
  weights as design weights, compounding adjustments. For example, run IPW first
  to correct broad imbalances, then rake on a specific variable for fine-tuning.
  The active weight column always keeps its original name (e.g., `"weight"`);
  the full weight history is tracked via `weight_pre_adjust` (frozen original
  design weights) and `weight_adjusted_1`, `weight_adjusted_2`, etc. The
  original unadjusted baseline is always preserved for diagnostics
  (`asmd_improvement()` shows total improvement across all steps).

- **Added formula support to `Sample.covars()` for downstream diagnostics**
  - `Sample.covars()` now accepts a `formula` argument and stores it on the
    returned `BalanceDFCovars` object.
  - `BalanceDFCovars.kld()` now honors formula-driven model matrices (including
    interactions such as `"age_group * gender"`) when a formula is provided via
    `covars(formula=...)`.
  - Formula settings are now propagated to linked covariate views (`target`,
    `unadjusted`) so comparative diagnostics run on consistent design matrices.

## Internal Changes

- **Naming consistency: weight-related property renames**
  - `SampleFrame.active_weight_column` (str) → `weight_column` — simpler name for
    the active weight column name.
  - `SampleFrame.weight_column` (Series) → `weight_series` — clarifies this returns
    weight values, not a column name.
  - `SampleFrame.weight_columns` (list) → `weight_columns_all` — avoids confusion
    with the singular `weight_column`.
  - `SampleFrame._active_weight_column` → `_weight_column_name` — clearer: stores
    a column name string.
  - `BalanceFrame.weight_column` (Series) → `weight_series` — matches SampleFrame.
  - `BalanceDFSource.weight_column` → `weight_series` — protocol accessor renamed.
  - All internal references updated across balance, graviton, and test files.

- **Naming consistency: BalanceFrame internal attributes**
  - `BalanceFrame._sf_with_outcomes` → `_sf_sample` — shorter, clearer.
  - `BalanceFrame._sf_with_outcomes_pre_adjust` → `_sf_sample_pre_adjust`.
  - `BalanceFrame(sf_with_outcomes=...)` constructor param → `sample=`.
  - `BalanceFrame.id_column` and `weight_series` now delegate to `_sf_sample`
    instead of caching redundant copies, removing stale-state risk.

- **`has_target` and `model` are now properties**
  - `BalanceFrame.has_target` is a `_CallableBool` property — both `bf.has_target`
    and `bf.has_target()` work (the latter for backward compatibility).
  - `BalanceFrame.model` is a plain `@property` — all `model()` call sites updated
    to `model`.

- **Guard method renames for consistency**
  - `BalanceFrame._check_if_adjusted` → `_require_adjusted` — imperative verb pattern.
  - `BalanceFrame._no_target_error` → `_require_target` — consistent with above.
  - `BalanceFrame._check_outcomes_exists` → `_require_outcomes` — consistent with above.
  - Error messages now use `type(self).__name__` instead of hardcoded "Sample".

- **Typing modernization and style cleanup**
  - `Dict` → `dict`, `Tuple` → `tuple` in annotations (`balancedf_class.py`).
  - `List[str]` → `list[str]` in annotations (`sample_class.py`).
  - `cast(pd.DataFrame, ...)` → `_assert_type(...)` (`sample_frame.py`).
  - `_check_if_not_BalanceDF` → `_check_if_not_balancedf` — snake_case convention.
  - `_BalanceDF_child_from_linked_samples` → `_balancedf_child_from_linked_samples`.

- **Refactored `Sample` to delegate to `SampleFrame` and `BalanceFrame` internally**
  - `Sample` is now a thin facade: `set_target()` creates a backing `BalanceFrame`,
    and `adjust()`, `summary()`, `diagnostics()`, `model()`, `is_adjusted`, and
    `keep_only_some_rows_columns()` delegate to it.
  - High-cardinality feature detection and large-target warnings moved from
    `Sample.adjust()` to `BalanceFrame.adjust()` so both APIs share the same logic.
  - No public API changes — all existing `Sample` methods continue to work identically.

## New Features

- **Added `SampleFrame` — a DataFrame container with explicit column-role metadata**
  - New class in `sample_frame.py` that holds a single DataFrame and tracks which
    columns are covariates, weights, outcomes, predicted outcomes, and ignored.
  - Factory methods: `SampleFrame.from_frame()` (with auto-detection of id/weight
    columns) and `SampleFrame.from_csv()`.
  - DataFrame-access properties use `df_*` prefix convention: `df_covars`,
    `df_weights`, `df_outcomes`, `df_ignored`. All return copies for mutation safety.
  - Column-role list properties: `covar_columns`, `weight_columns`,
    `outcome_columns`, `predicted_outcome_columns`, `ignored_columns` (all return
    copies). `misc_columns` is accepted as a deprecated alias.
  - Internal `_create()` factory with `_skip_copy` optimization for callers that
    have already deep-copied.
  - Comprehensive validation: null/negative/non-numeric weights, null IDs,
    duplicate IDs, overlapping column roles.

- **Added weight provenance tracking to `SampleFrame`**
  - `set_weight_metadata()` / `weight_metadata()`: store/retrieve arbitrary
    provenance dicts (method, hyperparameters, timestamps) for weight columns.
  - `set_active_weight()`: switch the active weight column returned by `df_weights`.
  - `add_weight_column()`: append a new weight column with length validation,
    duplicate-name guard (including non-weight columns), and optional metadata.

- **Added `BalanceFrame` — immutable adjustment orchestrator for survey weighting**
  - New class in `balance_frame.py` that pairs a responder `SampleFrame` with a
    target `SampleFrame` for survey/observational data reweighting.
  - `__new__`-based constructor: `BalanceFrame(sample=..., sf_target=...)` with
    covariate overlap validation.
  - `__new__`-based constructor now supports target-less construction:
    `BalanceFrame(sample=sf)` creates a BalanceFrame without a target.
  - `set_target(target, in_place=True)`: set or replace the target population.
    When `in_place=True` (default), modifies and returns self; when `False`,
    returns a new BalanceFrame. Resets adjustment state when target changes.
  - `has_target()`: check if a target population is set.
  - `adjust(method="ipw")`: returns a NEW BalanceFrame (immutable pattern) with
    adjusted weights. Supports string methods (`"ipw"`, `"cbps"`, `"rake"`,
    `"poststratify"`, `"null"`) and custom callables. Raises `ValueError` if
    no target is set.
  - Properties: `responders`, `target`, `unadjusted`, `is_adjusted`.
  - `model()`: returns the adjustment model dictionary.
  - `id_column` property: returns the ID column of the responder SampleFrame.
  - Records weight provenance metadata on the adjusted weight column.
  - Default transformations applied when neither SampleFrame has custom transforms.
  - Calls weighting functions directly with DataFrames (no Sample dependency).
  - `covars()` → `BalanceDFCovars`, `weights()` → `BalanceDFWeights`,
    `outcomes()` → `BalanceDFOutcomes`: wire responder SampleFrame directly to
    BalanceDF constructors via the BalanceDFSource protocol (no adapter needed).
  - `_build_links_dict()`: creates linked sources dict for target and unadjusted
    so that `.mean()`, `.asmd()`, `.summary()` etc. include comparisons across
    sources.
  - Added `covars()`, `weights()`, `outcomes()` methods to `SampleFrame` so that
    linked SampleFrames can produce BalanceDF views (required by the links
    machinery in `_balancedf_child_from_linked_samples`).
  - `summary()`: consolidated human-readable summary of covariate ASMD/KLD,
    weight design effect/ESS/ESSP, and outcome means. Delegates to shared
    `_build_summary()` in `summary_utils.py`.
  - `diagnostics()`: DataFrame-based diagnostics table with size, weight, model,
    and covariate ASMD metrics. Delegates to shared `_build_diagnostics()`.
  - `design_effect()`: returns Kish's design effect (Deff) of responder weights.
  - `design_effect_prop()`: returns effective sample size proportion (ESSP).
  - `covar_means()`: compares covariate means across unadjusted/adjusted/target.
  - `outcome_sd_prop()`: relative change in outcome SD after adjustment, with
    zero-division guard for constant-valued outcomes (returns NaN).
  - `outcome_variance_ratio()`: ratio of outcome variance (adjusted/unadjusted).
  - Private helpers: `_design_effect_diagnostics()`, `_quick_adjustment_details()`.
  - `df` property: combined DataFrame (responder + target + unadjusted) with a
    ``"source"`` column. Mirrors `Sample.df`.
  - `keep_only_some_rows_columns()`: immutable row/column filtering via
    `pd.DataFrame.eval` expressions and column name lists. Uses `_filter_sf()`
    static method with None-weight guard.
  - `to_csv()`: write combined DataFrame to CSV via `to_csv_with_defaults()`.
  - `to_download()`: create IPython `FileLink` for interactive download.

- **Sample internally backed by SampleFrame**
  - `Sample._df`, `_outcome_columns`, and `_ignored_column_names` are now `@property`
    descriptors that delegate to a backing `_sample_frame: SampleFrame` instance.
  - `from_frame()` refactored: all DataFrame mutations during construction use a local
    `working_df` variable; at the end, `SampleFrame._create()` is called with explicit
    column roles. The public API is fully backward-compatible.
  - `adjust()` now records weight provenance metadata on the backing SampleFrame via
    `set_weight_metadata()`, enabling downstream code to inspect how weights were produced.
  - `keep_only_some_rows_columns()`: column filtering now always preserves outcome
    columns in the keep set (per-link filtering uses each linked object's own outcomes).

- **`Sample.is_adjusted` is now a `@property` returning `_CallableBool`** — works both
  as `sample.is_adjusted` (property, consistent with BalanceFrame) and
  `sample.is_adjusted()` (legacy method call, backward compatible).

- **Added bidirectional conversion between Sample, SampleFrame, and BalanceFrame**
  - `SampleFrame.from_sample(sample)`: converts a Sample to a SampleFrame with
    proper column-role mapping (id, weight, outcomes, ignored).
  - `Sample.to_sample_frame()`: convenience method delegating to
    `SampleFrame.from_sample()`.
  - `BalanceFrame.from_sample(sample)`: converts a Sample (with target) to a
    BalanceFrame, preserving adjustment state (unadjusted responders, model).
  - `Sample.to_balance_frame()`: convenience method delegating to
    `BalanceFrame.from_sample()`.
  - `BalanceFrame.to_sample()`: converts a BalanceFrame back to a Sample
    (reconstructs responder, target, and optionally unadjusted links).
  - All conversion methods use lazy imports to avoid circular dependencies.

## Infrastructure

- **`BalanceDF.__init__()`: added optional `links` parameter for explicit link injection**
  - Allows BalanceDF to work with sources that do not carry mutable `_links`
    (e.g. the upcoming SampleFrame class).
  - When `links` is provided, `_balancedf_child_from_linked_samples()` uses the
    explicit dict; otherwise falls back to `sample._links` (backward compatible).

## Code Quality & Refactoring

- **Defined `BalanceDFSource` protocol and decoupled `BalanceDF` from `Sample`**
  - Added `BalanceDFSource` — a `typing.Protocol` (runtime-checkable) that captures
    the 6 attributes/methods `BalanceDF` accesses on its backing object:
    `weight_series`, `id_column`, `_links`, `_covar_columns()`,
    `_outcome_columns`, and `set_weights()`.
  - Updated `BalanceDF.__init__`, `BalanceDFCovars.__init__`,
    `BalanceDFWeights.__init__`, and `BalanceDFOutcomes.__init__` to accept
    `BalanceDFSource` instead of `Sample`.
  - Removed the hard top-level `from balance.sample_class import Sample` import
    from `balancedf_class.py`. The only remaining `Sample` usage
    (`BalanceDFCovars.from_frame()`) uses a lazy import.
  - Both `Sample` and the upcoming `SampleFrame` satisfy this protocol, enabling
    BalanceDF to work with either without an adapter class.

- **Extracted `_build_summary()` and `_build_diagnostics()` into `summary_utils.py`**
  - Moved the summary and diagnostics logic from `Sample.summary()` and
    `Sample.diagnostics()` into standalone functions that accept plain
    DataFrames/Series. `Sample` methods now delegate to these shared functions.
  - This is a pure refactor — no behavior changes. Enables code reuse by the
    upcoming `BalanceFrame` class without duplicating summary/diagnostics logic.
  - Also moves `_concat_metric_val_var` to `summary_utils.py` (re-exported from
    `sample_class.py` for backward compatibility).

## Tutorials

- Added `balance_quickstart_new_api.ipynb` — end-to-end tutorial demonstrating the
  new SampleFrame/BalanceFrame API. Mirrors the original `balance_quickstart.ipynb`
  step-by-step but uses only the new classes (no `Sample`). Covers: loading data,
  creating SampleFrames, building a BalanceFrame, adjusting (IPW + CBPS), inspecting
  diagnostics (summary, ASMD, covariate means, design effect), visualization
  (plotly, seaborn KDE, ASCII plots), outcome analysis, transformations, filtering
  rows/columns, and exporting to CSV.

## Documentation

- **Added `ARCHITECTURE.md`** — new top-level architecture document covering the class hierarchy,
  5-step workflow, key classes, weighting methods, supporting modules, and file layout.
  `CLAUDE.md` now links to this file instead of duplicating architecture content.
- **Added `docs/architecture/architecture_0_19_0.md`** — detailed historical record of the 0.19.0
  refactor with ASCII diagrams covering: class hierarchy before/after, SampleFrame internals,
  BalanceFrame internal structure, object lifecycle state transitions, BalanceDF linked-samples
  expansion, BalanceDFSource protocol, BalanceDF class hierarchy, data flow, _links graph, and
  Sample.__new__ guard.
- **Updated `README.md`** — added "Developer and AI assistant resources" section linking to
  `ARCHITECTURE.md` and `CLAUDE.md`.
## LLM/GenAI

- **Updated `CLAUDE.md` project context files** for Claude Code users, covering architecture,
  build/test instructions (Meta and open-source), code conventions, and pre-submit checklist.
  Architecture content moved to `ARCHITECTURE.md`; `CLAUDE.md` now contains a brief summary
  with a link.
- **Updated `.github/copilot-instructions.md`** review checklist to reduce duplication with
  `CLAUDE.md` and add missing conventions (MIT license header, `from __future__ import annotations`,
  factory pattern, seed fixing, deprecation style).

## Tests

- Added `TestBalanceFrameEndToEnd` class in `test_balance_frame.py` (12 tests):
  - `test_ipw_end_to_end_equivalence` — full workflow equivalence for IPW
  - `test_cbps_end_to_end_equivalence` — full workflow equivalence for CBPS
  - `test_rake_end_to_end_equivalence` — full workflow equivalence for raking
  - `test_poststratify_end_to_end_equivalence` — full workflow equivalence for
    post-stratification
  - `test_unadjusted_covars_mean_sources` — verifies unadjusted has self+target only
  - `test_adjusted_covars_mean_sources` — verifies adjusted has self+target+unadjusted
  - `test_immutability_across_methods` — verifies adjust() does not mutate original
  - `test_diagnostics_equivalence` — diagnostics() shape/metrics match between APIs
  - `test_covar_means_equivalence` — covar_means() matches between old and new APIs
  - `test_full_lifecycle_with_transformations` — adjust with custom transformations
  Each per-method test exercises: `covars().mean()`, `covars().asmd()`,
  `weights().summary()`, `design_effect()`, `outcomes().mean()`, `summary()`,
  `to_csv()` — verifying numerical equivalence with the old Sample API.

- Added `TestSampleFrameBalanceDFSourceProtocol` class in `test_sample_frame.py`
  (21 tests):
  - `test_isinstance_balancedf_source` — verifies `isinstance(sf, BalanceDFSource)`
  - `test_weight_column_returns_series`, `test_weight_column_returns_copy`,
    `test_weight_series_no_active_raises` — weight_series property
  - `test_id_column_returns_series` — id_column property
  - `test_links_default_empty`, `test_links_preserved_in_deepcopy` — _links attribute
  - `test_covar_columns_method`, `test_covar_columns_method_returns_copy` —
    _covar_columns() method
  - `test_outcome_columns_property`, `test_outcome_columns_none_when_no_outcomes` —
    _outcome_columns property
  - `test_set_weights_series`, `test_set_weights_float`,
    `test_set_weights_none_resets_to_one`, `test_set_weights_length_mismatch_raises`,
    `test_set_weights_no_active_raises` — set_weights() method
  - `test_balancedf_covars_with_sample_frame`,
    `test_balancedf_weights_with_sample_frame`,
    `test_balancedf_outcomes_with_sample_frame` — end-to-end BalanceDF construction

- Added comprehensive tests in `test_balance_frame.py` (7 test classes, ~25 tests):
  - `TestBalanceFrameConstruction` — basic construction, type errors, bare instance
  - `TestBalanceFrameCovarOverlap` — zero overlap, partial overlap, full overlap
  - `TestBalanceFrameDeepCopy` — deepcopy of unadjusted BalanceFrame
  - `TestBalanceFrameRepr` — repr/str output
  - `TestBalanceFrameCreateDirect` — _create() factory, property accessibility
  - `TestBalanceFrameAdjust` — IPW adjustment, immutability, custom callable,
    weight metadata, already-adjusted guard, method name storage, custom transforms,
    invalid method, deepcopy of adjusted state
  - `TestBalanceFrameCovarsWeightsOutcomes` — covars/weights/outcomes integration
    (34 tests): type checks, linked sources, `.df`, `.names()`, `.mean()`,
    `.std()`, `.var_of_mean()`, `.ci_of_mean()`, `.mean_with_ci()`, `.summary()`,
    `.model_matrix()`, `.asmd()`, `.asmd_improvement()`, `.to_csv()`,
    `.design_effect()`, `.trim()`, `.relative_response_rates()`,
    `.target_response_rates()`, numerical equivalence with Sample API for both
    covars and weights
  - `TestBalanceFrameSummaryDiagnostics` — summary(), diagnostics(),
    design_effect_prop(), _design_effect_diagnostics(), _quick_adjustment_details()
    (16 tests): section presence, unadjusted behavior, outcome output,
    cross-validation with Sample API, load_data equivalence, known/uniform/zero
    weights design effect prop, pre-computed de/ess/essp
  - `TestBalanceFrameAnalytics` — design_effect(), covar_means(), outcome_sd_prop(),
    outcome_variance_ratio() (16 tests): known/uniform weights, shape/columns,
    unadjusted raises, no-outcomes raises, constant outcome NaN guard,
    cross-validation with Sample API, null method expected values
  - `TestBalanceFrameDfExportFilter` — df property, keep_only_some_rows_columns,
    to_csv, to_download (16 tests): source column presence, row counts, adjusted
    vs unadjusted, immutability, column filtering, undefined variable handling,
    no-active-weight guard, CSV roundtrip, file export, FileLink type check
  - `TestBalanceFrameMissingIntegration` — full pipeline (adjust → summary →
    diagnostics → to_csv), null method weights, unadjusted CSV sources,
    no-match filter (4 tests)

- Added `TestSampleFrameFromSample` class in `test_sample_frame.py` (8 tests):
  - basic, with_outcomes, with_ignored_columns, preserves_data, independence,
    type_error, roundtrip_covars_match, no_outcomes

- Added `TestBalanceFrameFromSample` class in `test_balance_frame.py` (7 tests):
  - unadjusted, adjusted, covars_preserved, no_target_raises, type_error,
    with_outcomes, roundtrip_equivalence (with load_data + IPW)

- Added `TestBalanceFrameToSample` class in `test_balance_frame.py` (12 tests):
  - has_target, not_adjusted, covars_preserved, weight_values, id_values,
    target_data, adjusted, adjusted_weight_column, with_outcomes,
    roundtrip_sample_bf_sample, roundtrip_adjusted, roundtrip_load_data

- Added `TestSampleInternalSampleFrame` class in `test_sample.py` (19 tests):
  - `test_sample_has_sample_frame` — Sample has a SampleFrame backing
  - `test_df_property_returns_sample_frame_df` — `_df` delegates to SampleFrame
  - `test_df_setter_updates_sample_frame` — setting `_df` updates SampleFrame
  - `test_outcome_columns_property/none/setter` — outcome columns delegate
  - `test_ignored_column_names_property/empty/setter` — ignored columns delegate
  - `test_covar_columns_inferred_correctly` — covars inferred by exclusion
  - `test_from_frame_builds_sample_frame_with_correct_roles` — all roles correct
  - `test_set_target_preserves_sample_frame` — set_target keeps SampleFrame
  - `test_unadjusted_has_no_weight_metadata` — no metadata before adjust
  - `test_adjust_records_weight_metadata` — adjust() records method + adjusted
  - `test_adjust_ipw_records_weight_metadata` — IPW method recorded
  - `test_adjust_callable_records_weight_metadata` — callable __name__ recorded
  - `test_set_weights_syncs_sample_frame` — set_weights updates SampleFrame
  - `test_keep_only_some_rows_columns_preserves_outcomes` — outcomes kept in column filter
  - `test_deepcopy_preserves_sample_frame` — deepcopy creates independent SampleFrame

- Added `TestCallableBool` class in `test_sample.py` (11 tests):
  - `test_bool_true/false`, `test_call_true/false`, `test_repr`,
    `test_eq_with_bool`, `test_eq_with_callable_bool`,
    `test_eq_not_implemented_for_other_types`, `test_hash`,
    `test_mul`, `test_rmul`

- Added `test_Sample_is_adjusted_property_and_callable` in `test_sample.py`:
  verifies `is_adjusted` works both as property and method call

- Added `TestSampleConversion` class in `test_sample.py` (6 tests):
  - to_sample_frame_basic, to_sample_frame_with_outcomes,
    to_sample_frame_with_ignored, to_balance_frame_unadjusted,
    to_balance_frame_adjusted, to_balance_frame_no_target_raises

- Added `TestBalanceDFSourceProtocol` class in `test_balancedf.py` (8 tests):
  - `test_sample_satisfies_protocol` — verifies `Sample` passes `isinstance` check
  - `test_protocol_is_runtime_checkable` — verifies protocol is runtime-checkable
  - `test_non_conforming_object_fails_isinstance` — verifies non-conforming objects
    fail the isinstance check
  - `test_balancedf_with_mock_source` — constructs BalanceDF with a minimal mock
  - `test_balancedf_covars_with_mock_source` — BalanceDFCovars with mock,
    verifies mean() and _df_with_ids() work
  - `test_balancedf_weights_with_mock_source` — BalanceDFWeights with mock,
    verifies design_effect()
  - `test_balancedf_outcomes_with_mock_source` — BalanceDFOutcomes with mock
  - `test_existing_sample_api_unchanged` — regression test for existing Sample API

- Added 3 new tests in `test_sample_diagnostics_helper.py`:
  - `test_build_summary_matches_sample_summary` — verifies `_build_summary()`
    produces identical output to `Sample.summary()` for an IPW-adjusted sample.
  - `test_build_diagnostics_matches_sample_diagnostics` — verifies
    `_build_diagnostics()` matches `Sample.diagnostics()` for null adjustment.
  - `test_build_diagnostics_with_ipw_matches_sample_diagnostics` — same check
    for IPW adjustment.

# 0.18.0 (2026-03-24)

## New Features

- **Implemented `r_indicator()` with validated sample-variance formula**
  - Added a public `r_indicator(sample_p, target_p)` implementation in
    `weighted_comparisons_stats` using the documented Eq. 2.2.2 formulation
    over concatenated propensity vectors and explicit input-size validation.
  - Added validation for non-finite and out-of-range propensity values,
    and expanded unit coverage for formula correctness and edge cases.
  - Added `BalanceDFWeights.r_indicator()` as a convenience wrapper, so
    `sample.weights().r_indicator()` computes the r-indicator directly.

## Deprecations

- **`Sample.design_effect()` is deprecated** — use `sample.weights().design_effect()` instead.
  The method already exists on `BalanceDFWeights`; the `Sample` method now emits a
  `DeprecationWarning` and delegates. Will be removed in balance 0.19.0.
- **`Sample.design_effect_prop()` is deprecated** — use `sample.weights().design_effect_prop()` instead.
  New method added to `BalanceDFWeights`. Will be removed in balance 0.19.0.
- **`Sample.plot_weight_density()` is deprecated** — use `sample.weights().plot()` instead.
  Will be removed in balance 0.19.0.
- **`Sample.covar_means()` is deprecated** — use `sample.covars().mean()` instead
  (with `.rename(index={'self': 'adjusted'}).reindex([...]).T` for the same format).
  Will be removed in balance 0.19.0.
- **`Sample.outcome_sd_prop()` is deprecated** — use `sample.outcomes().outcome_sd_prop()` instead.
  New method added to `BalanceDFOutcomes`. Will be removed in balance 0.19.0.
- **`Sample.outcome_variance_ratio()` is deprecated** — use `sample.outcomes().outcome_variance_ratio()` instead.
  New method added to `BalanceDFOutcomes`. Will be removed in balance 0.19.0.

## LLM/GenAI

- **Added `CLAUDE.md` project context files** for Claude Code users, covering architecture,
  build/test instructions (Meta and open-source), code conventions, and pre-submit checklist.
- **Added `.github/copilot-instructions.md`** review checklist.

## Bug Fixes

- **`prepare_marginal_dist_for_raking` / `_realize_dicts_of_proportions`: fixed memory explosion from LCM expansion**
  - When proportions had high decimal precision or many covariates were passed,
    the LCM of the individual per-variable array lengths could reach tens of
    millions (or more), causing OOM crashes.
  - Both functions now accept a `max_length` parameter (default `10000`). When
    the natural LCM exceeds `max_length`, the output is capped at `max_length`
    rows and counts are allocated via the **Hare-Niemeyer (largest remainder)**
    method, which guarantees the total stays exactly `max_length` with minimal
    rounding error per category.
  - A warning is logged whenever the cap is applied.
  - A new internal helper `_hare_niemeyer_allocation` implements the allocation logic.

# 0.17.0 (2026-03-17)

## Breaking Changes

- **CLI: unmentioned columns now go to `ignore_columns` instead of `outcome_columns`**
  - Previously, when `--outcome_columns` was not explicitly set, all columns that
    were not the id, weight, or a covariate were automatically classified as
    outcome columns. Now those columns are placed into `ignore_columns` instead.
  - Columns that are explicitly mentioned — the id column, weight column,
    covariate columns, and outcome columns — are **not** ignored.

## New Features

- **ASCII comparative histogram and plot improvements**
  - Added `ascii_comparative_hist` for comparing multiple distributions against a
    baseline using inline visual indicators (`█`, `▒`, `▐`, `░`).
  - Comparative ASCII plots now order datasets as population → adjusted → sample.
  - `ascii_plot_dist` accepts a new `comparative` keyword (default `True`) to
    toggle between comparative and grouped-bar histograms for numeric variables.

## Code Quality & Refactoring

- **Moved dataset loading implementations out of `balance.datasets.__init__`**
  - Refactored `load_sim_data`, `load_cbps_data`, and `load_data` into
    `balance.datasets.loading_data` and re-exported them from
    `balance.datasets` to preserve the public API while keeping module
    responsibilities focused.

## Documentation

- **ASCII plot documentation and tutorial examples**
  - Added rendered text-plot examples to ASCII plot docstrings and documented
    `library="balance"` support. Updated `balance_quickstart.ipynb` with
    adjusted vs unadjusted ASCII plot examples.
- **Improved `keep_columns` documentation**
  - Updated docstrings for `has_keep_columns()`, `keep_columns()`, and the
    `--keep_columns` argument to clarify that keep columns control which columns
    appear in the final output CSV. Keep columns that are not id, weight,
    covariate, or outcome columns will be placed into ``ignore_columns`` during
    processing but are still retained and available in the output.
- **Clarified `_prepare_input_model_matrix` argument docs**
  - Updated docstrings in `balance.utils.model_matrix` with
    explicit descriptions for `sample`, `target`, `variables`, and `add_na`
    behavior when preparing model-matrix inputs.

## Bug Fixes

- **Weight diagnostics now consistently accept DataFrame inputs**
  - `design_effect`, `nonparametric_skew`, `prop_above_and_below`, and
    `weighted_median_breakdown_point` now explicitly normalize DataFrame inputs
    to their first column before computation, matching validation behavior and
    returning scalar/Series outputs consistently.
- **Model-matrix robustness improvements**
  - `_make_df_column_names_unique()` now avoids suffix collisions when columns
    like `a`, `a_1`, and repeated `a` names appear together, renaming
    duplicates deterministically to prevent downstream clashes.
  - `_prepare_input_model_matrix()` now raises a deterministic `ValueError`
    when the input sample has zero rows, instead of relying on an assertion.
- **Stabilized `prop_above_and_below()` return paths**
  - `prop_above_and_below()` now builds concatenated outputs only from present
    Series objects and returns `None` when both `below` and `above` are `None`,
    avoiding ambiguous concat inputs while preserving existing behavior for valid
    threshold sets.
- **Validated and normalized comma-separated CLI column arguments**
  - CLI column-list arguments now trim surrounding whitespace and reject empty
    entries (for example, `"id,,weight"`) with clear `ValueError` messages,
    preventing malformed column specifications from silently propagating.
  - Applied to `--covariate_columns`, `--covariate_columns_for_diagnostics`,
    `--batch_columns`, `--keep_columns`, and `--outcome_columns` parsing.

## Tests

- **Added end-to-end adjustment test with ASCII plot output and expanded ASCII plot edge-case coverage**
  - `TestAsciiPlotsAdjustmentEndToEnd` runs the full adjustment pipeline and
    asserts exact expected ASCII output. Added tests for `ascii_plot_dist` with
    `comparative=False` and mixed categorical+numeric routing.
- **Expanded warning coverage for `Sample.from_frame()` ID inference**
  - Added assertions that validate all three expected warnings are emitted when inferring an `id` column and default weights, including ID guessing, ID string casting, and automatic weight creation.
- **Expanded IPW helper and diagnostics test coverage**
  - Added tests for `link_transform()` and `calc_dev()` to validate behavior
    for extreme probabilities and finite 10-fold deviance summaries.
  - Refactored diagnostics tests to use a shared IPW setup helper, added
    edge-case assertions for solver/penalty values, NaN coercion of non-scalar
    inputs, and now assert labels match fitted model parameters.
- **Expanded `prop_above_and_below()` edge-case coverage**
  - Added focused tests for empty threshold iterables, mixed `None` threshold groups in dict mode, and explicit all-`None` threshold handling across return formats.
- **Added unit coverage for CLI I/O and empty-batch handling**
  - Added focused tests for `BalanceCLI.process_batch()` empty-sample failure payloads, `load_and_check_input()` CSV loading paths, and `write_outputs()` delimiter-aware output writing for both adjusted and diagnostics files.

# 0.16.0 (2026-02-09)

## New Features

- **Outcome weight impact diagnostics**
  - Added paired outcome-weight impact tests (`y*w0` vs `y*w1`) with confidence intervals.
  - Exposed in `BalanceDFOutcomes`, `Sample.diagnostics()`, and the CLI via
    `--weights_impact_on_outcome_method`.
- **Pandas 3 support**
  - Updated compatibility and tests for pandas 3.x
- **Categorical distribution metrics without one-hot encoding**
  - KLD/EMD/CVMD/KS on `BalanceDF.covars()` now operate on raw categorical variables
    (with NA indicators) instead of one-hot encoded columns.
- **Misc**
  - **Raw-covariate adjustment for custom models**
    - `Sample.adjust()` now supports fitting models on raw covariates (without a model matrix)
      for IPW via `use_model_matrix=False`. String, object, and boolean columns are converted
      to pandas `Categorical` dtype, allowing sklearn estimators with native categorical
      support (e.g., `HistGradientBoostingClassifier` with `categorical_features="from_dtype"`)
      to handle them correctly. Requires scikit-learn >= 1.4 when categorical columns are
      present.
  - **Validate weights include positive values**
    - Added a guard in weight diagnostics to error when all weights are zero.
  - **Support configurable ID column candidates**
    - `Sample.from_frame()` and `guess_id_column()` now accept candidate ID column names
      when auto-detecting the ID column.
  - **Formula support for BalanceDF model matrices**
    - `BalanceDF.model_matrix()` now accepts a `formula` argument to build
      custom model matrices without precomputing them manually.


## Bug Fixes

- **Removed deprecated setup build**
  - Replaced deprecated `setup.py` with `pyproject.toml` build in CI to avoid build failure.
- **Hardened ID column candidate validation**
  - `guess_id_column()` now ignores duplicate candidate names and validates that candidates are non-empty strings.
- **Hardened pandas 3 compatibility paths**
  - Updated string/NA handling and discrete checks for pandas 3 dtypes, and refreshed tests to accept string-backed dtypes.

## Packaging & Tests

- **Pandas 3.x compatibility**
  - Expanded the pandas dependency range to allow pandas 3.x releases.
- **Direct util imports in tests**
  - Refactored util test modules to import helpers directly from their modules instead of via `balance_util`.

## Breaking Changes

- **Require positive weights for weight diagnostics that normalize or aggregate**
  - `design_effect`, `nonparametric_skew`, `prop_above_and_below`, and
    `weighted_median_breakdown_point` now raise a `ValueError` when all weights
    are zero.
  - **Migration:** ensure your weights include at least one positive value
    before calling these diagnostics, or catch the `ValueError` if all-zero
    weights are possible in your workflow.

# 0.15.0 (2026-01-20)

## New Features

- **Added EMD/CVMD/KS distribution diagnostics**
  - `BalanceDF` now exposes Earth Mover's Distance (EMD), Cramér-von Mises distance (CVMD), and Kolmogorov-Smirnov (KS) statistics for comparing adjusted samples to targets.
  - These diagnostics support weighted or unweighted comparisons, apply discrete/continuous formulations, and respect `aggregate_by_main_covar` for one-hot categorical aggregation.
- **Exposed outcome columns selection in the CLI**
  - Added `--outcome_columns` to choose which columns are treated as outcomes
    instead of defaulting to all non-id/weight/covariate columns. Remaining columns are moved to `ignored_columns`.
- **Improved missing data handling in `poststratify()`**
  - `poststratify()` now accepts `na_action` to either drop rows with missing
    values or treat missing values as their own category during weighting.
  - **Breaking change:** the default behavior now fills missing values in
    poststratification variables with `"__NaN__"` and treats this as a distinct
    category during weighting. Previously, missing values were not handled
    explicitly, and their treatment depended on pandas `groupby` and `merge`
    defaults. To approximate the legacy behavior where missing values do not
    form their own category, pass `na_action="drop"` explicitly.
- **Added formula support for `descriptive_stats` model matrices**
  - `descriptive_stats()` now accepts a `formula` argument that is always
    applied to the data (including numeric-only frames), letting callers
    control which terms and dummy variables are included in summary statistics.

## Documentation

- **Documented the balance CLI**
  - Added full API docstrings for `balance.cli` and a new CLI tutorial notebook.
- **Created Balance CLI tutorial**
  - Added CLI command echoing, a `load_data()` example, and richer diagnostics exploration with metric/variable listings and a browsable diagnostics table. https://import-balance.org/docs/tutorials/balance_cli_tutorial/
- **Synchronized docstring examples with test cases**
  - Updated user-facing docstrings so the documented examples mirror tested inputs
    and outputs.

## Code Quality & Refactoring

- **Added warning when the sample size of 'target' is much larger than 'sample' sample size**
  - `Sample.adjust()` now warns when the target exceeds 100k rows and is at
    least 10x larger than the sample, highlighting that uncertainty is
    dominated by the sample (akin to a one-sample comparison).
- **Split util helpers into focused modules**
  - Broke `balance.util` into `balance.utils` submodules for easier navigation.

## Bug Fixes

- **Updated `Sample.__str__()` to format weight diagnostics like `Sample.summary()`**
  - Weight diagnostics (design effect, effective sample size proportion, effective sample size)
    are now displayed on separate lines instead of comma-separated on one line.
  - Replaced "eff." abbreviations with full "effective" word for better readability.
  - Improves consistency with `Sample.summary()` output format.
- **Numerically stable CBPS probabilities**
  - The CBPS helper now uses a stable logistic transform to avoid exponential
    overflow warnings during probability computation in constraint checks.
- **Silenced pandas observed default warning**
  - Explicitly sets `observed=False` in weighted categorical KLD calculations
    to retain current behavior and avoid future pandas default changes.
- **Fixed `plot_qq_categorical` to respect the `weighted` parameter for target data**
  - Previously, the target weights were always applied regardless of the
    `weighted=False` setting, causing inconsistent behavior between sample
    and target proportions in categorical QQ plots.
- **Restored CBPS tutorial plots**
  - Re-enabled scatter plots in the CBPS comparison tutorial notebook while
    avoiding GitHub Pages rendering errors and pandas colormap warnings. https://import-balance.org/docs/tutorials/comparing_cbps_in_r_vs_python_using_sim_data/
- **Clearer validation errors in adjustment helpers**
  - `trim_weights()` now accepts list/tuple inputs and reports invalid types explicitly.
  - `apply_transformations()` raises clearer errors for invalid inputs and empty transformations.
- **Fixed `model_matrix` to drop NA rows when requested**
  - `model_matrix(add_na=False)` now actually drops rows containing NA values while preserving categorical levels, matching the documented behavior.
  - Previously, `add_na=False` only logged a warning without dropping rows; code relying on the old behavior may now see fewer rows and should either handle missingness explicitly or use `add_na=True`.

## Tests

- **Aligned formatting toolchain between Meta internal and GitHub CI**
  - Added `["fbcode/core_stats/balance"]` override to Meta's internal `tools/lint/pyfmt/config.toml` to use `formatter = "black"` and `sorter = "usort"`.
  - This ensures both internal (`pyfmt`/`arc lint`) and external (GitHub Actions) environments use the same Black 25.1.0 formatter, eliminating formatting drift.
  - Updated CI workflow, pre-commit config, and `requirements-fmt.txt` to use `black==25.1.0`.
- **Added Pyre type checking to GitHub Actions** via `.pyre_configuration.external` and a new `pyre` job in the workflow. Tests are excluded due to external typeshed stub differences; library code is fully type-checked.
- **Added test coverage workflow and badge to README** via `.github/workflows/coverage.yml`. The workflow collects coverage using pytest-cov, generates HTML and XML reports, uploads them as artifacts, and displays coverage metrics. A coverage badge is now shown in README.md alongside other workflow badges.
- **Improved test coverage for edge cases and error handling paths**
  - Added targeted tests for previously uncovered code paths across the library, addressing edge cases including empty inputs, verbose logging, error handling for invalid parameters, and boundary conditions in weighting methods (IPW, CBPS, rake).
  - Tests exercise defensive code paths that handle empty DataFrames, NaN convergence values, invalid model types, and non-convergence warnings.
- **Split test_util.py into focused test modules**
  - Split the large `test_util.py` file (2325 lines) into 5 modular test files that mirror the `balance/utils/` structure:
    - `test_util_data_transformation.py` - Tests for data transformation utilities
    - `test_util_input_validation.py` - Tests for input validation utilities
    - `test_util_model_matrix.py` - Tests for model matrix utilities
    - `test_util_pandas_utils.py` - Tests for pandas utilities (including high cardinality warnings)
    - `test_util_logging_utils.py` - Tests for logging utilities
  - This improves test organization and makes it easier to locate tests for specific utilities.

## Contributors

@neuralsorcerer, @talgalili

# 0.14.0 (2025-12-14)

## New Features

- **Enhanced adjusted sample summary output**
  - `Sample.__str__()` now displays adjustment details (method, trimming
    parameters, design effect, effective sample size) when printing adjusted
    samples ([#194](https://github.com/facebookresearch/balance/pull/194),
    [#57](https://github.com/facebookresearch/balance/issues/57)).
- **Richer `Sample.summary()` diagnostics**
  - Adjusted sample summary now groups covariate diagnostics, reports design
    effect alongside ESSP/ESS, and surfaces weighted outcome means when
    available.
- **Warning of high-cardinality categorical features in `.adjust()`**
  - Categorical features where ≥80% of values are unique are flagged before
    weight fitting to help identify problematic columns like user IDs
    ([#195](https://github.com/facebookresearch/balance/pull/195),
    [#65](https://github.com/facebookresearch/balance/issues/65)).
- **Ignored column handling for Sample inputs**
  - `Sample.from_frame` accepts `ignore_columns` for columns that should remain
    on the dataframe but be excluded from covariates and outcome statistics.
    Ignored columns appear in `Sample.df` and can be retrieved via
    `Sample.ignored_columns()`.

## Code Quality & Refactoring

- **Consolidated diagnostics helpers**
  - Added `_concat_metric_val_var()` helper and `balance.util._coerce_scalar`
    for robust diagnostics row construction and scalar-to-float conversion.
  - **Breaking change:** `Sample.diagnostics()` for IPW now always emits
    iteration/intercept summaries plus hyperparameter settings.

## Bug Fixes

- **Early validation of null weight inputs**
  - `Sample.from_frame` now raises `ValueError` when weights contain `None`,
    `NaN`, or `pd.NA` values with count and preview of affected rows.
- **Percentile weight trimming across platforms**
  - `trim_weights()` now computes thresholds via percentile quantiles with
    explicit clipping bounds for consistent behavior across Python/NumPy
    versions.
  - **Breaking change:** percentile-based clipping may shift by roughly one
    observation at typical limits.
- **IPW diagnostics improvements**
  - Fixed `multi_class` reporting, normalized scalar hyperparameters to floats,
    removed deprecated `penalty` argument warnings, and deduplicated metric
    entries for stable counts across sklearn versions.

## Tests

- **Added Windows and macOS CI testing support**
  - Expanded GitHub Actions to run on `ubuntu-latest`, `macos-latest`, and
    `windows-latest` for Python 3.9-3.14.
  - Added `tempfile_path()` context manager for cross-platform temp file
    handling and configured matplotlib Agg backend via `conftest.py`.

## Contributors

@neuralsorcerer, @talgalili, @wesleytlee

# 0.13.0 (2025-12-02)

## New Features

- **Propensity modeling beyond static logistic regression**
  - `.adjust(method='ipw')` now accepts any sklearn classifier via the `model`
    argument, enabling the use of models like random forests and gradient
    boosting while preserving all existing trimming and diagnostic features.
    Dense-only estimators and models without linear coefficients are fully
    supported. Propensity probabilities are stabilized to avoid numerical
    issues.
  - Allow customization of logistic regression by passing a configured
    :class:`~sklearn.linear_model.LogisticRegression` instance through the
    `model` argument. Also, the CLI now accepts
    `--ipw_logistic_regression_kwargs` JSON to build that estimator directly for
    command-line workflows.
- **Covariate diagnostics**
  - Added KL divergence calculations for covariate comparisons (numeric and
    one-hot categorical), exposed via `sample.covars().kld()` alongside
    linked-sample aggregation support.
- **Weighting Methods**
  - `rake()` and `poststratify()` now honour `weight_trimming_mean_ratio` and
    `weight_trimming_percentile`, trimming and renormalising weights through the
    enhanced `trim_weights(..., target_sum_weights=...)` API so the documented
    parameters work as expected
    ([#147](https://github.com/facebookresearch/balance/pull/147)).

## Documentation

- Added comprehensive post-stratification tutorial notebook
  (`balance_quickstart_poststratify.ipynb`)
  ([#141](https://github.com/facebookresearch/balance/pull/141),
  [#142](https://github.com/facebookresearch/balance/pull/142),
  [#143](https://github.com/facebookresearch/balance/pull/143)).
- Expanded poststratify docstring with clear examples and improved statistical
  methods documentation
  ([#141](https://github.com/facebookresearch/balance/pull/141)).
- Added project badges to README for build status, Python version support, and
  release tracking
  ([#145](https://github.com/facebookresearch/balance/pull/145)).
- Added example of using custom logistic regression and custom sklearn
  classifier usage in (`balance_quickstart.ipynb`).
- Shorten the welcome message (for when importing the package).

## Code Quality & Refactoring

- **Raking algorithm refactor**
  - Removed `ipfn` dependency and replaced with a vectorized NumPy
    implementation (`_run_ipf_numpy`) for iterative proportional fitting,
    resulting in significant performance improvements and eliminating external
    dependency ([#135](https://github.com/facebookresearch/balance/pull/135)).

- **IPW method refactoring**
  - Reduced Cyclomatic Complexity Number (CCN) by extracting repeated code
    patterns into reusable helper functions: `_compute_deviance()`,
    `_compute_proportion_deviance()`, `_convert_to_dense_array()`.
  - Removed manual ASMD improvement calculation and now uses existing
    `compute_asmd_improvement()` from `weighted_comparisons_stats.py`

- **Type safety improvements**
  - Migrated 32 Python files from `# pyre-unsafe` to `# pyre-strict` mode,
    covering core modules, statistics, weighting methods, datasets, and test
    files
  - Modernized type hints to PEP 604 syntax (`X | Y` instead of `Union[X, Y]`)
    across 11 files for improved readability and Python 3.10+ alignment
  - Type alias definitions in `typing.py` retain `Union` syntax for Python 3.9
    compatibility
  - Enhanced plotting function type safety with `TypedDict` definitions and
    proper type narrowing
  - Replaced assert-based type narrowing with `_assert_type()` helper for
    better error messages and pyre-strict compliance

- **Renamed Balance**_DF to BalanceDF_\*\*\*\*
  - BalanceCovarsDF to BalanceDFCovars
  - BalanceOutcomesDF to BalanceDFOutcomes
  - BalanceWeightsDF to BalanceDFWeights

## Bug Fixes

- **Utility Functions**
  - Fixed `quantize()` to preserve column ordering and use proper TypeError
    exceptions ([#133](https://github.com/facebookresearch/balance/pull/133))
- **Statistical Functions**
  - Fixed division by zero in `asmd_improvement()` when `asmd_mean_before` is
    zero, now returns `0.0` for 0% improvement
- **CLI & Infrastructure**
  - Replaced deprecated argparse FileType with pathlib.Path
    ([#134](https://github.com/facebookresearch/balance/pull/134))
- **Weight Trimming**
  - Fixed `trim_weights()` to consistently return `pd.Series` with
    `dtype=np.float64` and preserve original index across both trimming methods
  - Fixed percentile-based winsorization edge case: `_validate_limit()` now
    automatically adjusts limits to prevent floating-point precision issues
    ([#144](https://github.com/facebookresearch/balance/issues/144))
  - Enhanced documentation for `trim_weights()` and `_validate_limit()` with
    clearer examples and explanations

## Tests

- Enhanced test coverage for weight trimming with
  `test_trim_weights_return_type_consistency` and 11 comprehensive tests for
  `_validate_limit()` covering edge cases, error conditions, and boundary
  conditions

## Contributors

@neuralsorcerer, @talgalili, @wesleytlee

# 0.12.1 (2025-11-03)

## New Features

- Added a welcome message when importing the package.

## Documentation

- Added 'CHANGELOG' to the docs website.
  https://import-balance.org/docs/docs/CHANGELOG/

## Bug Fixes

- Fixed plotly figures in all the tutorials.
  https://import-balance.org/docs/tutorials/

# 0.12.0 (2025-10-14)

## New Features

- **Support for Python 3.13 + 3.14**
  - Update setup.py and CI/CD integration to include Python 3.13 and 3.14.
  - Remove upper version constraints from numpy, pandas, scipy, and scikit-learn
    dependencies for Python 3.12+.

## Contributors

@talgalili, @wesleytlee

# 0.11.0 (2025-09-24)

## New Features

- **Python 3.12 support** - Complete support for Python 3.12 alongside existing
  Python 3.9, 3.10, and 3.11 support (with CI/CD integration).
  - **Implemented Python version-specific dependency constraints** - Added
    conditional version ranges for numpy, pandas, scipy, and scikit-learn that
    vary based on Python version (e.g., numpy>=1.21.0,<2.0 for Python <3.12,
    numpy>=1.24.0,<2.1 for Python >=3.12)
  - **Pandas compatibility improvements** - Replaced
    `value_counts(dropna=False)` with `groupby().size()` in frequency table
    creation to avoid FutureWarning
  - Fixed various pandas deprecation warnings and improved DataFrame handling
- **Improved raking algorithm** - Completely refactored rake weighting from
  DataFrame-based to array-based ipfn algorithm using multi-dimensional arrays
  and itertools for better performance and compatibility with latest Python
  versions. **Variables are now automatically alphabetized** to ensure
  consistent results regardless of input order.
- **poststratify method enhancement** - New `strict_matching` parameter (default
  True) handles cases where sample cells are not present in target data. When
  False, issues warning and assigns weight 0 to uncovered samples

## Bug Fixes

- **Type annotations** - Enhanced Pyre type hints throughout the codebase,
  particularly in utility functions
- **Sample class improvements** - Fixed weight type assignment (ensuring float64
  type), improved DataFrame manipulation with `.infer_objects(copy=False)` for
  pandas compatibility, and enhanced weight setting logic
- **Website dependencies** - Updated various website dependencies including
  Docusaurus and related packages

## Tests

**Comprehensive test refactoring**, including:

- **Enhanced test validation** - Added detailed explanations of test
  methodologies and expected behaviors in docstrings
- **Improved test coverage** - Tests now include edge cases like NaN handling,
  different data types, and error conditions
- **Improved test organization** (more granular) across all test modules
  (test_stats_and_plots.py, test_balancedf.py, test_ipw.py, test_rake.py,
  test_cli.py, test_weighted_comparisons_plots.py, test_cbps.py,
  test_testutil.py, test_adjustment.py, test_util.py, test_sample.py)
- **Updated GitHub workflows** to include Python 3.12 in build and test matrix
- **Fix 261 "pandas deprecation" warnings!**
- **Added type annotations** - Converted test_balancedf.py to pyre-strict with.

## Documentation

- **GitHub issue template for support questions** - Added structured template to
  help users ask questions about using the balance package

## Contributors

@talgalili, @wesleytlee, @dependabot

# 0.10.0 (2025-01-06)

## New Features

- Dependency on glmnet has been removed, and the `ipw` method now uses sklearn.
- The transition to sklearn should enable support for newer python versions
  (3.11) as well as the Windows OS!
- `ipw` method uses logistic regression with L2-penalties instead of
  L1-penalties for computational reasons. The transition from glmnet to sklearn
  and use of L2-penalties will lead to slightly different generated weights
  compared to previous versions of Balance.
- Unfortunately, the sklearn-based `ipw` method is generally slower than the
  previous version by 2-5x. Consider using the new arguments `lambda_min`,
  `lambda_max`, and `num_lambdas` for a more efficient search over the `ipw`
  penalization space.

## Misc

- Update license from GPL v2 to
  [MIT license](https://github.com/facebookresearch/balance/blob/main/LICENSE).
- Updated Python and package compatibility. Balance is now compatible with
  Python 3.11, but no longer compatible with Python 3.8 due to typing errors.
  Balance is currently incompatible with Python 3.12 due to the removal of
  distutils.

## Contributors

@wesleytlee, @talgalili, @SarigT

# 0.9.1 (2023-07-30)

## Bug Fixes

- Fix E721 flake8 issue (see:
  https://github.com/facebookresearch/balance/actions/runs/5704381365/job/15457952704)
- Remove support for python 3.11 from release.yml

## Documentation

- Added links to presentation given at ISA 2023.
- Fixed misc typos.

  # 0.9.0 (2023-05-22)

## News

- Remove support for python 3.11 due to new test failures. This will be the case
  until glmnet will be replaced by sklearn. hopefully before end of year.

## New Features

- All plotly functions: add kwargs to pass arguments to update_layout in all
  plotly figures. This is useful to control width and height of the plot. For
  example, when wanting to save a high resolution of the image.
- Add a `summary` methods to `BalanceWeightsDF` (i.e.:
  `Sample.weights().summary()`) to easily get access to summary statistics of
  the survey weights. Also, it means that `Sample.diagnostics()` now uses this
  new summary method in its internal implementation.
- `BalanceWeightsDF.plot` method now relies on the default `BalanceDF.plot`
  method. This means that instead of a static seaborn kde plot we'll get an
  interactive plotly version.

## Bug Fixes

- datasets
  - Remove a no-op in `load_data` and accommodate deprecation of pandas syntax
    by using a list rather than a set when selecting df columns (thanks @ahakso
    for the PR).
  - Make the outcome variable (`happiness`) be properly displayed in the
    tutorials (so we can see the benefit of the weighting process). This
    included fixing the simulation code in the target.
- Fix `Sample.outcomes().summary()` so it will output the ci columns without
  truncating them.

## Documentation

- Fix text based on updated from version 0.7.0 and 0.8.0.
  - https://import-balance.org/docs/docs/general_framework/adjusting_sample_to_population/
- Fix tutorials to include the outcome in the target.

## Contributors

@talgalili, @SarigT, @ahakso

# 0.8.0 (2023-04-26)

## New Features

- Add `rake` method to .adjust (currently in beta, given that it doesn't handles
  marginal target as input).
- Add a new function `prepare_marginal_dist_for_raking` - to take in a dict of
  marginal proportions and turn them into a pandas DataFrame. This can serve as
  an input target population for raking.

## Misc

- The `ipw` function now gets max_de=None as default (instead of 1.5). This
  version is faster, and the user can still choose a threshold as desired.
- Adding hex stickers graphics files

## Documentation

- New section on
  [raking.](https://import-balance.org/docs/docs/statistical_methods/rake/)
- New notebook (in the tutorial section):
  - [**quickstart_rake**](https://import-balance.org/docs/tutorials/quickstart_rake/) -
    like the
    [**quickstart**](https://import-balance.org/docs/tutorials/quickstart/)
    tutorial, but shows how to use the rake (raking) algorithm and compares the
    results to IPW (logistic regression with LASSO).

## Contributors

@talgalili, @SarigT

# 0.7.0 (2023-04-10)

## New Features

- Add `plotly_plot_density` function: Plots interactive density plots of the
  given variables using kernel density estimation.
- Modified `plotly_plot_dist` and `plot_dist` to also support 'kde' plots. Also,
  these are now the default options. This automatically percolates to
  `BalanceDF.plot()` methods.
- `Sample.from_frame` can now guess that a column called "weights" is a weight
  column (instead of only guessing so if the column is called "weight").

## Bug Fixes

- Fix `rm_mutual_nas`: it now remembers the index of pandas.Series that were
  used as input. This fixed erroneous plots produced by seaborn functions which
  uses rm_mutual_nas.
- Fix `plot_hist_kde` to work when dist_type = "ecdf"
- Fix `plot_hist_kde` and `plot_bar` when having an input only with "self" and
  "target", by fixing `_return_sample_palette`.

## Misc

- All plotting functions moved internally to expect weight column to be called
  `weight`, instead of `weights`.
- All adjust (ipw, cbps, poststratify, null) functions now export a dict with a
  key called `weight` instead of `weights`.

## Contributors

@talgalili, @SarigT

# 0.6.0 (2023-04-05)

## New Features

- Variance of the weighted mean
  - Add the `var_of_weighted_mean` function (from
    balance.stats_and_plots.weighted_stats import var_of_weighted_mean):
    Computes the variance of the weighted average (pi estimator for ratio-mean)
    of a list of values and their corresponding weights.
    - Added the `var_of_mean` option to stat in the `descriptive_stats` function
      (based on `var_of_weighted_mean`)
    - Added the `.var_of_mean()` method to BalanceDF.
  - Add the `ci_of_weighted_mean` function (from
    balance.stats_and_plots.weighted_stats import ci_of_weighted_mean): Computes
    the confidence intervals of the weighted mean using the (just added)
    variance of the weighted mean.
    - Added the `ci_of_mean` option to stat in the `descriptive_stats` function
      (based on `ci_of_weighted_mean`). Also added kwargs support.
    - Added the `.ci_of_mean()` method to BalanceDF.
    - Added the `.mean_with_ci()` method to BalanceDF.
    - Updated `.summary()` methods to include the output of `ci_of_mean`.
- All bar plots now have an added ylim argument to control the limits of the y
  axis. For example use:
  `plot_dist(dfs1, names=["self", "unadjusted", "target"], ylim = (0,1))` Or
  this: `s3_null.covars().plot(ylim = (0,1))`
- Improve 'choose_variables' function to control the order of the returned
  variables
  - The return type is now a list (and not a Tuple)
  - The order of the returned list is based on the variables argument. If it is
    not supplied, it is based on the order of the column names in the
    DataFrames. The df_for_var_order arg controls which df to use.
- Misc
  - The `_prepare_input_model_matrix` and downstream functions (e.g.:
    `model_matrix`, `sample.outcomes().mean()`, etc) can now handle DataFrame
    with special characters in the column names, by replacing special characters
    with '\_' (or '\_i', if we end up with columns with duplicate names). It
    also handles cases in which the column names have duplicates (using the new
    `_make_df_column_names_unique` function).
  - Improve choose_variables to control the order of the returned variables
    - The return type is now a list (and not a Tuple)
    - The order of the returned list is based on the variables argument. If it
      is not supplied, it is based on column names in the DataFrames. The
      df_for_var_order arg controls which df to use.

## Contributors

@talgalili, @SarigT

# 0.5.0 (2023-03-06)

## New Features

- The `datasets.load_data` function now also supports the input "sim_data_cbps",
  which loads the simulated data used in the CBPS R vs Python tutorial. It is
  also used in unit-testing to compare the CBPS weights produced from Python
  (i.e.: balance) with R (i.e.: the CBPS package). The testing shows how the
  correlation of the weights from the two implementations (both Pearson and
  Spearman) produce a correlation of >0.98.
- cli improvements:
  - Add an option to set formula (as string) in the cli.

## Documentation

- New notebook (in the tutorial section):
  - Comparing results of fitting CBPS between R's `CBPS` package and Python's
    `balance` package (using simulated data).
    [link](https://import-balance.org/docs/tutorials/comparing_cbps_in_r_vs_python_using_sim_data/)

## Contributors

@stevemandala, @talgalili, @SarigT

# 0.4.0 (2023-02-08)

## New Features

- Added two new flags to the cli:
  - `--standardize_types`: This gives cli users the ability to set the
    `standardize_types` parameter in Sample.from_frame to True or False. To
    learn more about this parameter, see:
    https://import-balance.org/api_reference/html/balance.sample_class.html#balance.sample_class.Sample.from_frame
  - `--return_df_with_original_dtypes`: the Sample object now stores the dtypes
    of the original df that was read using Sample.from_frame. This can be used
    to restore the original dtypes of the file output from the cli. This is
    relevant in cases in which we want to convert back the dtypes of columns
    from how they are stored in Sample, to their original types (e.g.: if
    something was Int32 it would be turned in float32 in balance.Sample, and
    using the new flag will return that column, when using the cli, to be back
    in the Int32 type). This feature may not be robust to various edge cases. So
    use with caution.
- In the logging:
  - Added warnings about dtypes changes. E.g.: if using Sample.from_frame with a
    column that has Int32, it will be turned into float32 in the internal
    storage of sample. Now there will be a warning message indicating of this
    change.
  - Increase the default length of logger printing (from 500 to 2000)

## Bug Fixes

- Fix pandas warning: SettingWithCopyWarning in from_frame (and other places in
  sample_class.py)
- sample.from_frame has a new argument `use_deepcopy` to decide if changes made
  to the df inside the sample object would also change the original df that was
  provided to the sample object. The default is now set to `True` since it's
  more likely that we'd like to keep the changes inside the sample object to the
  df contained in it, and not have them spill into the original df.

## Contributors

@SarigT, @talgalili

# 0.3.1 (2023-02-01)

## Bug Fixes

- Sample.from_frame now also converts int16 and in8 to float16 and float16. Thus
  helping to avoid `TypeError: Cannot interpret 'Int16Dtype()' as a data type`
  style errors.

## Documentation

- Added ISSUE_TEMPLATE

## Contributors

@talgalili, @stevemandala, @SarigT

# 0.3.0 (2023-01-30)

## New Features

- Added compatibility for Python 3.11 (by supporting SciPy 1.9.2) (props to
  @tomwagstaff-opml for flagging this issue).
- Added the `session-info` package as a dependency.

## Bug Fixes

- Fixed pip install from source on Windows machines (props to @tomwagstaff-opml
  for the bug report).

## Documentation

- Added `session_info.show()` outputs to the end of the three tutorials (at:
  https://import-balance.org/docs/tutorials/)
- Misc updates to the README.

## Contributors

@stevemandala, @SarigT, @talgalili

# 0.2.0 (2023-01-19)

## New Features

- cli improvements:
  - Add an option to set weight_trimming_mean_ratio = None for no trimming.
  - Add an option to set transformations to be None (i.e. no transformations).
- Add an option to adapt the title in:
  - stats_and_plots.weighted_comparison_plots.plot_bar
  - stats_and_plots.weighted_comparison_plots.plot_hist_kde

## Bug Fixes

- Fix (and simplify) balanceDF.plot to organize the order of groups (now
  unadjusted/self is left, adjusted/self center, and target is on the right)
- Fix plotly functions to use the red color for self when only compared to
  target (since in that case it is likely unadjusted):
  balance.stats_and_plots.weighted_comparisons_plots.plotly_plot_qq and
  balance.stats_and_plots.weighted_comparisons_plots.plotly_plot_bar
- Fix seaborn_plot_dist: output None by default (instead of axis object). Added
  a return_Axes argument to control this behavior.
- Fix some test_cbps tests that were failing due to non-exact matches (we made
  the test less sensitive)

## Documentation

- New blog section, with the post:
  [Bringing "balance" to your data ](https://import-balance.org/blog/2023/01/09/bringing-balance-to-your-data/)
- New tutorial:
  - [**quickstart_cbps**](https://import-balance.org/docs/tutorials/quickstart_cbps/) -
    like the
    [**quickstart**](https://import-balance.org/docs/tutorials/quickstart/)
    tutorial, but shows how to use the CBPS algorithm and compares the results
    to IPW (logistic regression with LASSO).
  - [**balance_transformations_and_formulas**](https://import-balance.org/docs/tutorials/balance_transformations_and_formulas/) -
    This tutorial showcases ways in which transformations, formulas and penalty
    can be included in your pre-processing of the covariates before adjusting
    for them.
- API docs:
  - New: highlighting on codeblocks
  - a bunch of text fixes.
- Update README.md
  - logo
  - with contributors
  - typo fixes (props to @zbraiterman and @luca-martial).
- Added section about "Releasing a new version" to CONTRIBUTING.md
  - Available under
    ["Docs/Contributing"](https://import-balance.org/docs/docs/contributing/#releasing-a-new-version)
    section of website

## Misc

- Added automated Github Action package builds & deployment to PyPi on release.
  - See
    [release.yml](https://github.com/facebookresearch/balance/blob/main/.github/workflows/release.yml)

## Contributors

@stevemandala, @SarigT, @talgalili

# 0.1.0 (2022-11-20)

## Summary

- balance released to the world!

## Contributors

@SarigT, @talgalili, @stevemandala
