# Copilot Code Review Instructions

Use this checklist when reviewing pull requests for the **balance** Python package, which provides weighting and balancing utilities for correcting bias in tabular datasets (for example, via IPW, CBPS, rake, and poststratification routines built on pandas and NumPy).

## Core expectations
1. **Tests are present and meaningful.** Confirm new or changed behavior is covered by `pytest` tests under `tests/`. Edge cases (e.g., missing columns, unexpected dtypes, boundary weights) should be exercised, and tests should be deterministic.
2. **Changelog updates accompany user-visible changes.** Require `CHANGELOG.md` entries for fixes or features. If backward-incompatible behavior is introduced, the entry must clearly flag the breaking change and migration notes.
3. **Typed, example-rich docs for new APIs.** The codebase is Pyre-typed (`# pyre-strict`); ensure new or modified public functions/classes include complete type hints and a docstring with at least one concrete usage example, ideally mirroring the test plan.
4. **Backward compatibility.** Watch for changes to defaults, return shapes, or CLI flags that could break existing users. If intentional, the pull request summary and changelog should call this out explicitly.
5. **Dependency discipline.** New dependencies should be rare, lightweight, and justified. When touching `requirements` files or setup metadata, verify compatibility with supported Python versions and existing dependency pins.
6. **Documentation and examples.** When behavior changes, ensure tutorials, README snippets, and inline examples stay accurate (especially around dataset loading, weighting methods, and plotting utilities).

## Style and implementation notes
- Favor clear, pandas-friendly code that avoids mutating input data structures in-place unless documented.
- Maintain existing logging patterns and error messages for consistent user experience.
- Keep tests fast and avoid large fixtures; prefer factory helpers in `tests/` when possible.

Keep comments concise and actionable, pointing authors to exact lines that need updates.
