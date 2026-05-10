# Code Review Instructions: `balance` (Python)

Review instructions for the **balance** Python package (weighting and balancing utilities for correcting bias in tabular datasets). Prioritize correctness, statistical validity, reproducibility, and backward compatibility.

For project architecture, build/test commands, and file layout, see `CLAUDE.md` in the repository root (at Meta: `fbcode/core_stats/balance/CLAUDE.md`).

## Review checklist

### 0) PR scope and focus — one idea per PR
- Each PR should represent **one self-contained idea** that is easy to test, review, and accept.
- Flag PRs that combine multiple unrelated changes:
  - mixing feature additions with refactoring
  - addressing multiple independent bugs in one PR
  - combining documentation updates with unrelated code changes
  - bundling multiple distinct features together
- If a PR touches multiple subsystems or modules for unrelated reasons, suggest splitting into focused PRs.
- Large PRs are acceptable if all changes serve a single, cohesive goal (e.g., implementing one feature that naturally spans multiple files).
- When flagging scope issues:
  - list the distinct concerns or ideas present in the PR
  - suggest a logical split (e.g., "Consider separating the refactoring into its own PR")
  - explain how splitting will improve reviewability and testability
- **For externally-authored PRs imported from `facebookresearch/balance`, do not assume open-source review covered API consistency** — run §5.5 explicitly even if the GitHub PR is already approved. CI does not flag taste/consistency drift.

### 1) Correctness and statistical soundness
- Verify the implementation matches the intended method (IPW / CBPS / rake / poststratification).
- Confirm assumptions and constraints are handled explicitly (e.g., positivity, normalization, convergence criteria).
- Check output semantics: shapes, index alignment, column names, and dtype stability.
- Ensure missingness and invalid inputs have well-defined behavior (error vs. warning vs. coercion).
- When the diff touches `balance/interop/diff_diff.py`, verify the
  `weight_type="pweight"` contract is still upheld — diff-diff's
  staggered estimators (CallawaySantAnna, StackedDiD, ImputationDiD,
  HeterogeneousAdoptionDiD, TwoStageDiD, WooldridgeDiD, TROP,
  StaggeredTripleDifference, ChaisemartinDHaultfoeuille,
  TripleDifference, SyntheticDiD) each inline a `weight_type !=
  "pweight"` rejection in their `fit()` method (the canonical example
  is `CallawaySantAnna.fit` in diff-diff's `staggered.py`). The
  history-column drop in `balance.interop._common.drop_history_columns`
  must run before any `dd.aggregate_survey()` handoff to avoid leaking
  `weight_pre_adjust` / `weight_adjusted_*` as covariates. (Symbol-based
  anchors instead of line numbers — line numbers in cross-repo files
  rot silently the moment either repo is touched.)

### 2) Input validation and actionable errors
- Validate early: required columns exist, dtypes are supported, and parameter ranges are enforced.
- Weights: explicitly handle/forbid zero, negative, infinite, or NaN weights (as appropriate for the API).
- Make error messages actionable:
  - name the missing/invalid column
  - list available columns when relevant
  - state how to fix it
- Prefer existing helpers in `utils.py` instead of duplicating validation logic.

### 3) Tests (pytest) — required for behavior changes
- New or changed behavior MUST be covered by deterministic `pytest` tests under `tests/`.
- Exercise edge cases when applicable:
  - missing columns / schema mismatch
  - unexpected dtypes (object/category/int/float/bool)
  - NaN/inf handling in inputs and outputs
  - extreme/boundary weights, clipping, normalization
  - empty dataframes / single-row inputs
- Keep tests stable:
  - avoid order/time dependence and uncontrolled randomness
  - if randomness is necessary, fix seeds and assert with tolerances
- Coverage expectation: aim for >90% coverage on new code (`pytest --cov`).
- Prefer using `from balance import load_data` in tests when appropriate.

### 4) Types and docs (Pyre strict)
- The codebase is Pyre-typed (`# pyre-strict`) with `from __future__ import annotations` on every file.
- New/modified public APIs must have complete type hints. Avoid returning `Any` or widening types unless justified.
- New/modified public functions/classes must include a docstring with at least one concrete usage example.
- MIT license header required on every source file.

### 5) Backward compatibility and deprecations
- Do not silently change defaults, return shapes, column names, or CLI flags.
- If a breaking change is intentional:
  - call it out clearly in the PR summary
  - add migration guidance and “before → after” examples
- For deprecations:
  - use proper warnings
  - document timeline and replacement usage
  - update changelog accordingly

### 5.5) API surface consistency and parameter naming

For **every new public default, parameter name, or string-literal option** the PR introduces, do the following — and quote evidence in your review comment, not just a verdict:

**A. Default-value alignment.** For each new keyword default (`library=`, `threshold=`, `order_by=`, `line=`, `show=`, `bar_width=`, etc.):
1. Grep the package for the same parameter name on neighbouring methods (start with `BalanceDF.plot`, `BalanceDFCovars.*`, `BalanceDFOutcomes.*`, `BalanceDFWeights.*`, and `stats_and_plots/`).
2. Quote both defaults: "`BalanceDF.plot` defaults to `library='plotly'`, this PR introduces `library='seaborn'` on `BalanceDFCovars.love_plot`."
3. If they differ, require either (a) the new default is changed to match, or (b) the PR summary documents why divergence is correct.

**B. Parallel-parameter detection.** When a PR adds a new parameter to a method that already has a similar dispatch parameter:
1. Identify the existing dispatch parameter (e.g. `dist_type` on `BalanceDF.plot`).
2. Check whether the new parameter's accept-list could be expressed as additional values on the existing parameter's `Literal`.
3. If yes, **request removal of the new parameter and extension of the existing `Literal` instead.** Example: `plot_type="love_plot"` next to `dist_type="love_plot"` — pick `dist_type`, drop `plot_type`. One spelling per concept.

**C. Option-name truthfulness.** For each new string-literal accept value (e.g. `order_by="max"`, `library="balance"`, `metric="kld"`):
1. Read the name literally and write down what you'd expect the implementation to do.
2. Read the implementation.
3. If the name's natural reading does not match the implementation, flag it — propose either a rename or a different implementation. Example: `order_by="max"` implemented as `data.abs().max(axis=1)` is "max of `|before|` and `|after|`" — the name doesn't disambiguate signed/absolute or pre/post; rename to something explicit (`"max_abs"`, `"diff"`, etc.) or change the semantics.

**D. Type-system reach.** For each new `Literal[...]` introduced (e.g. `LovePlotLibrary`, `LovePlotOrderBy`):
1. Verify it's exported from a stable location and reused in any sibling method that takes the same parameter.
2. If a sibling method's signature uses `str` or a different `Literal`, flag the divergence.

### 6) Changelog discipline
- User-visible fixes/features MUST include an entry in `CHANGELOG.md`.
- Breaking changes MUST be explicitly labeled and include migration notes.
- **Tutorial framing for interpretive defaults.** When the PR changes (or adds) a default that controls how a tutorial output is *read* — sort order, axis direction, scale, sign convention, baseline — verify the **first** tutorial cell that exercises the default has a markdown sentence motivating it, not just the option-list cell that comes later. Concrete check: open the relevant `.ipynb` cell, find the markdown immediately above (or referencing) the new-default call, and confirm a reader running only that one cell can answer "what does the visual order/scale/sign mean here?". If they can't, request a one-sentence framing in that markdown.

### 7) Dependencies and packaging
- New dependencies should be rare, lightweight, and justified.
- If touching requirements/setup metadata:
  - verify supported Python versions
  - avoid unnecessary pin churn
  - consider transitive impact

### 8) Performance and memory
- Flag performance regressions using representative dataset sizes.
- Avoid unnecessary dataframe copies and large intermediate objects.
- Prefer vectorized operations (NumPy/pandas) over Python loops.
- For iterative/optimization routines: ensure stopping criteria, max iterations, and tolerances are documented and tested.

### 9) Style, logging, and UX consistency
- Favor clear, pandas-friendly code.
- Avoid mutating user-provided inputs in-place unless explicitly documented.
- Preserve existing logging patterns: one logger per module via `logger = logging.getLogger(__package__)`.
- Use `DeprecationWarning` for deprecations with clear replacement guidance and removal timeline.
- Keep tests fast; prefer small fixtures and shared factories/helpers in `tests/`.
- Fix random seeds in tests for reproducibility; assert with tolerances.
- `Sample` is constructed via `Sample.from_frame()` factory — not `__init__` directly.

## Review comment style
- Keep feedback concise and actionable.
- Point to exact lines/files and propose concrete fixes and/or specific missing tests.
- If uncertain, ask for a small reproducible example or an additional test to clarify behavior.

## Self-maintenance
- When review guidelines change, update both this file and `CLAUDE.md` to keep them in sync.
- This file owns the detailed review checklist. `CLAUDE.md` owns architecture, build/test, and project structure.
