# Copilot Code Review Instructions: `balance` (Python)

When performing a Copilot code review for this repository, follow the instructions below and prioritize correctness, statistical validity, reproducibility, and backward compatibility.

## Project context
This repo contains the **balance** Python package: weighting and balancing utilities for correcting bias in tabular datasets (e.g., IPW, CBPS, raking, poststratification), built on pandas and NumPy.

## Review checklist

### 1) Correctness and statistical soundness
- Verify the implementation matches the intended method (IPW / CBPS / rake / poststratification).
- Confirm assumptions and constraints are handled explicitly (e.g., positivity, normalization, convergence criteria).
- Check output semantics: shapes, index alignment, column names, and dtype stability.
- Ensure missingness and invalid inputs have well-defined behavior (error vs. warning vs. coercion).

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
- The codebase is Pyre-typed (`# pyre-strict`): new/modified public APIs must have complete type hints.
- New/modified public functions/classes must include a docstring with at least one concrete usage example.
```there are too many changes to easily follow.
Also, this new version ignores the need to check tests. It also doesn't mention helper functions should have docstrings. It should also mention the need to describe arguments.
- Avoid returning `Any` or widening types unless justified.

### 5) Backward compatibility and deprecations
- Do not silently change defaults, return shapes, column names, or CLI flags.
- If a breaking change is intentional:
  - call it out clearly in the PR summary
  - add migration guidance and “before → after” examples
- For deprecations:
  - use proper warnings
  - document timeline and replacement usage
  - update changelog accordingly

### 6) Changelog discipline
- User-visible fixes/features MUST include an entry in `CHANGELOG.md`.
- Breaking changes MUST be explicitly labeled and include migration notes.

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
- Preserve existing logging patterns and verbosity conventions.
- Keep tests fast; prefer small fixtures and shared factories/helpers in `tests/`.

## Review comment style
- Keep feedback concise and actionable.
- Point to exact lines/files and propose concrete fixes and/or specific missing tests.
- If uncertain, ask for a small reproducible example or an additional test to clarify behavior.
