# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Shared helpers for ``balance.interop.*`` adapters.

This module is the single canonical home for adapter helpers that ALL
:mod:`balance.interop` sub-modules need. Today the only consumer is
:mod:`balance.interop.diff_diff`; when the future
:mod:`balance.interop.svy` lands it will reuse the SAME helpers, so the
two adapters cannot drift in their handling of balance-side state.

Canonical balance-side hidden state (the "5-state" spec)
========================================================

After ``Sample.adjust(method="ipw" | "cbps" | "rake" | ...)`` runs, a
:class:`balance.Sample` carries five distinct pieces of state that any
downstream library must handle correctly. Adapters that fail to drop or
preserve any of these introduce silent bugs in surveys that span the
balance / diff-diff / svy boundary:

1. **``weight_pre_adjust``** — column on ``Sample.df`` holding the
   pre-adjustment design weight (or ``1.0`` if ``Sample.from_frame`` was
   given no ``weight_column``). Written by ``balance_frame.py:701-744``.
   Must be DROPPED before handing the DataFrame to a survey-aware
   estimator: diff-diff and svy will silently treat it as a covariate.

2. **``weight_adjusted_N``** (``N = 1, 2, …``) — column(s) on
   ``Sample.df`` recording the chain of compounded ``adjust()`` calls.
   The active weight column NAME stays the same across compounded calls
   (per ``balance_frame.py:707-712``); ``adjust()`` overwrites the
   values in that active column with the latest adjustment and stores
   each prior adjustment as a new ``weight_adjusted_N`` history column.
   Same drop rule as ``weight_pre_adjust`` for the history columns.

3. **``weight_trimmed_*``** — column(s) recording the pre-trim weight
   when the user passes ``transformations="trim"`` to ``adjust()``. Same
   drop rule.

4. **``Sample._links["unadjusted"]`` / ``Sample._links["target"]``** —
   internal back-references to the original responder and target
   ``Sample``. NOT a column — adapters do not need to drop, but
   diagnostic helpers (e.g. ``as_balance_diagnostic``) walk these to
   compute ASMD pre / post.

5. **``Sample._adjustment_model``** — the fitted IPW / CBPS / rake model
   object. Out of scope for diff-diff and svy directly, but
   :func:`attach_balance_provenance` lets downstream tooling reach it
   via the ``_balance_adjustment`` side-channel.

The :func:`drop_history_columns` helper handles items 1-3 with one call
so adapters do not have to enumerate the bookkeeping prefixes
themselves.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import pandas as pd
from balance.sample_class import Sample

# Route through the package logger ("balance"), matching the convention used
# by ``balance/sample_class.py`` and ``balance/balance_frame.py`` so adapter
# log output inherits the configuration set up in ``balance/__init__.py``.
logger: logging.Logger = logging.getLogger(__package__)


def active_weight_column(s: Sample) -> str:
    """Return the active weight column NAME (``str``) on a balance ``Sample``.

    balance's ``weight_column`` property at ``sample_frame.py:570-600``
    returns ``str | None`` (a column name, not a Series — that's
    ``weight_series`` at ``sample_frame.py:761-786``). It also emits a
    ``FutureWarning`` during the v0.20 migration to clarify the
    str-vs-Series semantics; this helper suppresses that warning so
    adapter callers don't see it.

    Parameters
    ----------
    s :
        A balance ``Sample`` (typically post-``adjust()``).

    Returns
    -------
    str
        The name of the column on ``s.df`` that carries the active weights.

    Raises
    ------
    ValueError
        When ``s.weight_column`` is ``None``. ``adjust()`` ensures the
        ``Sample`` has an active weight column, but it preserves the
        existing active-column name rather than canonicalizing it to
        ``"weight"`` -- a ``Sample`` constructed with a custom
        ``weight_column="w"`` keeps that name post-adjustment. A ``None``
        value here therefore means the ``Sample`` was constructed without
        an active weight column and ``adjust()`` was never run.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        col: str | None = s.weight_column
    if col is None:
        raise ValueError(
            "balance.Sample.weight_column is None — cannot hand off to a "
            "survey-aware library without a weight column. Call "
            "`s.adjust(...)` first or pass `weight_column=...` to "
            "`Sample.from_frame(...)`."
        )
    return col


def _is_balance_history_column(name: object) -> bool:
    """Return True if ``name`` matches a column that ``BalanceFrame.adjust()``
    is documented to emit (and only those names).

    balance writes:

    * ``"weight_pre_adjust"`` (a single literal column name; see
      ``balance_frame.py:719-722``)
    * ``"weight_adjusted_<N>"`` where ``N`` is an integer step counter
      (see ``balance_frame.py:727-729``)
    * ``"weight_trimmed_<N>"`` where ``N`` is an integer trim counter
      (see ``sample_frame.py:1014``)

    We anchor the integer suffix here so unrelated user covariates whose
    names happen to start with one of these prefixes (e.g. a column literally
    called ``"weight_adjusted_q1_response"``) are NOT silently dropped from
    the interop handoff.
    """
    s: str = str(name)
    if s == "weight_pre_adjust":
        return True
    for prefix in ("weight_adjusted_", "weight_trimmed_"):
        if s.startswith(prefix):
            suffix: str = s[len(prefix) :]
            if suffix.isdigit():
                return True
    return False


def drop_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip balance's adjustment-history columns from ``df``.

    After ``BalanceFrame.adjust(method=...)`` the responder frame carries
    ``weight_pre_adjust``, ``weight_adjusted_N``, and (if trimming was
    requested) ``weight_trimmed_N`` columns — see hidden-state items 1-3
    in this module's docstring. Diff-diff has no contract about these
    columns and will silently treat them as covariates if forwarded.
    Svy's ``RepWeights.columns_from_data()`` auto-detect would pick them
    up as replicate weights. Both behaviours are silent failures; this
    helper is the single defensive scrub.

    Matching is intentionally narrow: only the literal
    ``"weight_pre_adjust"`` column and the digit-suffixed
    ``"weight_adjusted_<N>"`` / ``"weight_trimmed_<N>"`` patterns that
    balance actually emits are dropped. Unrelated user covariates that
    happen to share one of these prefixes (e.g.
    ``"weight_adjusted_q1_response"``) are preserved.

    Returns a new DataFrame (``df.drop`` returns a copy by default), so
    callers can safely mutate the result without affecting the source
    Sample.
    """
    history: list[str] = [c for c in df.columns if _is_balance_history_column(c)]
    if history:
        return df.drop(columns=history)
    # Always return a fresh copy: the docstring promises callers can safely
    # mutate the result without affecting ``Sample.df``. ``df.drop(columns=)``
    # already returns a new frame in the history-found branch, so we mirror
    # that contract here for the no-history path.
    return df.copy()


def drop_nan_weight_rows(
    df: pd.DataFrame, weight_col: str, *, ctx: str
) -> pd.DataFrame:
    """Drop rows whose active weight is NaN, warning the caller if any are
    found.

    Why this is necessary
    ---------------------
    ``BalanceFrame.adjust(na_action='drop')`` (the IPW default; see
    ``weighting_methods/ipw.py:738-745``) does NOT actually remove rows
    that had missing covariates -- it keeps them in ``Sample.df`` and
    sets their active weight to ``NaN``. Diff-diff's ``SurveyDesign`` and
    ``aggregate_survey`` propagate those ``NaN`` weights into variance
    estimation: best case the call raises far away from the
    ``Sample`` -> adapter seam; worst case the fitted ATT is silently
    contaminated by rows that have no defined contribution.

    Adapter callers should NOT see this concern -- the contract is "I
    handed balance my data, please weight it for diff-diff". So we drop
    NaN-weight rows defensively at the seam, with a ``UserWarning`` so
    the row-count loss is visible in notebooks and CI logs.

    Parameters
    ----------
    df :
        DataFrame to filter (typically ``Sample.df``).
    weight_col :
        Name of the active weight column on ``df``.
    ctx :
        Short label naming the calling adapter function -- included in
        the warning so the user can trace which seam dropped rows.

    Returns
    -------
    pd.DataFrame
        Either ``df`` itself (when no NaN-weight rows exist) or a new
        DataFrame with NaN-weight rows removed.
    """
    if weight_col not in df.columns:
        # Inconsistent ``Sample`` state: ``Sample.weight_column`` claims an
        # active column name that no longer exists on ``Sample.df`` (e.g.
        # the user mutated ``sample.df`` and dropped the weight column
        # after ``adjust()``). Failing fast here gives a clear actionable
        # message instead of letting the downstream adapter raise a
        # cryptic ``KeyError`` deep in ``SurveyDesign``/``aggregate_survey``.
        raise ValueError(
            f"{ctx}: weight column {weight_col!r} is not present in df.columns "
            f"({sorted(map(str, df.columns))}). This usually means the "
            "Sample's active weight column was dropped or renamed after "
            "adjust(). Re-fit the Sample or pass a DataFrame whose columns "
            "match Sample.weight_column."
        )
    weights: pd.Series = df[weight_col]
    nan_mask: pd.Series = weights.isna()
    n_nan: int = int(nan_mask.sum())
    if n_nan == 0:
        return df
    warnings.warn(
        f"{ctx}: dropping {n_nan} row(s) of {len(df)} with NaN weights "
        f"in column {weight_col!r} before forwarding to the downstream "
        "adapter. These rows typically come from "
        "`Sample.adjust(na_action='drop')` in IPW, which retains the row "
        "but sets the active weight to NaN. Pass "
        "`na_action='add_indicator'` (or pre-impute missing values) if "
        "you want every row to carry weight downstream.",
        UserWarning,
        stacklevel=3,
    )
    return df.loc[~nan_mask].copy()


def validate_nonzero_weights(df: pd.DataFrame, weight_col: str, *, ctx: str) -> None:
    """Raise if the active weight column has no positive entries.

    Zero-individual weights are fine (a zero-weight row is just a
    non-contributing observation), but if EVERY weight is zero or
    negative, the downstream ``SurveyDesign`` will produce undefined
    variance: ``sum(w) == 0`` propagates ``inf`` / ``NaN`` through the
    Kish ESS / design-effect / Binder-1983 sandwich at every variance
    seam. We catch this at the adapter boundary so the user sees a
    clear, actionable error instead of a downstream ``ZeroDivisionError``
    or ``inf`` value buried in an estimator's diagnostics dict.

    Negative weights are also rejected -- balance-style design weights
    are non-negative by construction, and a negative entry is almost
    always a sign that the user has confused a residual / centred
    column with the active weight column.

    Parameters
    ----------
    df :
        DataFrame whose active weight column to validate.
    weight_col :
        Name of the active weight column on ``df``.
    ctx :
        Short label naming the calling adapter function -- included in
        the error message so the user can trace which seam tripped.

    Raises
    ------
    ValueError
        When ``df[weight_col]`` contains no entries strictly greater
        than zero (after NaN filtering, which is handled separately by
        :func:`drop_nan_weight_rows`).
    """
    if weight_col not in df.columns:
        # Mirror the early-fail message from ``drop_nan_weight_rows`` so
        # both helpers fail with the same actionable wording when the
        # active weight column has been dropped from ``Sample.df``.
        raise ValueError(
            f"{ctx}: weight column {weight_col!r} is not present in df.columns "
            f"({sorted(map(str, df.columns))}). This usually means the "
            "Sample's active weight column was dropped or renamed after "
            "adjust(). Re-fit the Sample or pass a DataFrame whose columns "
            "match Sample.weight_column."
        )
    weights: pd.Series = df[weight_col]
    # We compare to 0 with ``> 0`` (rather than ``!= 0``) so a vector of
    # all-negative weights also fails fast -- ``sum_negative_weights ** 2``
    # is positive but the SurveyDesign still produces a meaningless fit.
    if not (weights > 0).any():
        raise ValueError(
            f"{ctx}: active weight column {weight_col!r} has no positive "
            f"entries ({len(weights)} rows, all <= 0). The downstream "
            "SurveyDesign would produce undefined variance "
            "(sum of positive weights = 0), propagating inf / NaN through "
            "Kish ESS / design-effect / Binder-1983 sandwich computations. "
            "Re-fit `Sample.adjust(...)` with `transformations='trim'`, "
            "or pre-filter the input to rows with positive design weights."
        )


def validate_row_count(s: Sample, n_target: int, *, ctx: str) -> None:
    """Raise if a balance ``Sample`` lost rows relative to ``n_target``.

    "Lost rows" includes BOTH the row-count drop case AND the case where
    ``BalanceFrame.adjust(na_action="drop")`` retains the row but writes
    its weight back as ``NaN`` (see ``balance/balance_frame.py:740-744`` --
    the shorter weight vector is reindexed onto the original frame, so
    ``len(s.df)`` matches ``n_target`` and a naive length check yields a
    false pass). Counting NaN weights is the only reliable adapter-side
    signal that ``na_action='drop'`` removed records from the active
    weight column.

    Both the diff-diff adapter and the future svy adapter need this
    check -- diff-diff's panel alignment and svy's ``rep_wgts`` row
    alignment are both fragile in different ways when rows are silently
    missing from the active weighting.

    Parameters
    ----------
    s :
        The balance ``Sample`` whose row count to check.
    n_target :
        The expected number of rows (typically the row count of the
        DataFrame the user originally handed to ``Sample.from_frame``).
    ctx :
        A short label naming the calling adapter function -- included in
        the error message so the user can trace which seam tripped.

    Raises
    ------
    ValueError
        When ``len(s.df) != n_target`` OR when the active weight column
        contains NaN entries (the ``na_action='drop'`` reindex case).
    """
    n_actual: int = len(s.df)
    if n_actual != n_target:
        raise ValueError(
            f"Row-count mismatch in {ctx}: balance.Sample has {n_actual} "
            f"rows but the downstream adapter expects {n_target}. Did "
            "`na_action='drop'` silently drop rows? Use "
            "`na_action='add_indicator'` instead, or pre-impute the "
            "missing values before adjust()."
        )
    # Detect the more subtle ``na_action='drop'`` case where row count is
    # preserved but weights are NaN.
    weight_col: str | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        weight_col = s.weight_column
    if weight_col is not None and weight_col in s.df.columns:
        n_nan: int = int(s.df[weight_col].isna().sum())
        if n_nan > 0:
            raise ValueError(
                f"Row-count appears unchanged in {ctx} (n={n_actual}), but "
                f"the active weight column {weight_col!r} has {n_nan} "
                "NaN entries. This is the `adjust(na_action='drop')` "
                "pattern: balance reindexes the shorter weight vector "
                "onto the original frame and marks dropped records with "
                "NaN. Use `na_action='add_indicator'` (or pre-impute "
                "missing covariates) so every row carries a defined "
                "weight downstream."
            )


def attach_balance_provenance(target: object, sample: Sample) -> None:
    """Attach ``sample`` to ``target`` as ``_balance_adjustment`` (provenance).

    Side-channel pattern. Lets downstream diagnostic tooling recover all 5
    hidden-state pieces (see this module's docstring) from a diff-diff
    result or svy ``Sample`` via one stable attribute name.

    Idempotent — silently no-ops when ``target`` already has the
    attribute (so re-fitting does not clobber the original lineage).
    Falls back to ``target.__dict__`` if ``setattr`` is rejected (frozen
    msgspec / dataclass struct); emits a ``UserWarning`` if even that
    path is blocked, since callers passing ``preserve_adjustment=True``
    deserve a clear signal that provenance was NOT attached -- silently
    no-oping makes ``_balance_adjustment`` look unreliable in tooling.

    Parameters
    ----------
    target :
        Any object (typically a diff-diff results dataclass).
    sample :
        The ``balance.Sample`` whose provenance to record.
    """
    if hasattr(target, "_balance_adjustment"):
        return
    try:
        # pyre-ignore[16]: dynamic attribute set on object — intentional.
        target._balance_adjustment = sample
        return
    except (AttributeError, TypeError):
        pass
    # Frozen dataclass / msgspec struct fallback.
    target_dict: dict[str, Any] | None = getattr(target, "__dict__", None)
    if target_dict is None:
        warnings.warn(
            f"Could not attach `_balance_adjustment` to a "
            f"{type(target).__name__} instance: object rejected both "
            "`setattr` and lacks a writable `__dict__`. Provenance will "
            "NOT be available on this result; if `preserve_adjustment=True` "
            "matters for your pipeline, store the adjustment Sample "
            "alongside the result (e.g. wrap them in a tuple, dict, or a "
            "small custom container class) -- ``dataclasses.replace`` will "
            "NOT help here because it preserves ``frozen=True``.",
            UserWarning,
            stacklevel=2,
        )
        return
    try:
        target_dict["_balance_adjustment"] = sample
    except TypeError:
        # ``__dict__`` exists but is read-only (e.g. mappingproxy on a
        # class; some msgspec structs). Same UX rationale as above:
        # warn loudly rather than silently dropping provenance.
        warnings.warn(
            f"Could not attach `_balance_adjustment` to a "
            f"{type(target).__name__} instance: `__dict__` is read-only. "
            "Provenance will NOT be available on this result.",
            UserWarning,
            stacklevel=2,
        )
