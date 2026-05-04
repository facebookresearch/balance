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

Canonical balance-side hidden state (the "5-state" spec; see
``~/balance_diff_diff/SVY_FUTURE_PROOFING.md`` §5)
=================================================

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
   ``adjust()`` rotates the active weight name forward each call. Same
   drop rule as ``weight_pre_adjust``.

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

logger: logging.Logger = logging.getLogger(__name__)


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
        When ``s.weight_column`` is ``None``. Adjusters always set the
        active column to ``"weight"`` (preserved by
        ``balance_frame.py:820-822``), so a ``None`` weight here means the
        ``Sample`` was constructed without one and ``adjust()`` was never
        run.
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


def drop_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip balance's adjustment-history columns from ``df``.

    After ``BalanceFrame.adjust(method=...)`` the responder frame carries
    ``weight_pre_adjust``, ``weight_adjusted_N``, and (if trimming was
    requested) ``weight_trimmed_*`` columns — see hidden-state items 1-3
    in this module's docstring. Diff-diff has no contract about these
    columns and will silently treat them as covariates if forwarded.
    Svy's ``RepWeights.columns_from_data()`` auto-detect would pick them
    up as replicate weights. Both behaviours are silent failures; this
    helper is the single defensive scrub.

    Returns a new DataFrame (``df.drop`` returns a copy by default), so
    callers can safely mutate the result without affecting the source
    Sample.
    """
    history: list[str] = [
        c
        for c in df.columns
        if c == "weight_pre_adjust"
        or str(c).startswith("weight_adjusted_")
        or str(c).startswith("weight_trimmed_")
    ]
    if history:
        return df.drop(columns=history)
    # Always return a fresh copy: the docstring promises callers can safely
    # mutate the result without affecting ``Sample.df``. ``df.drop(columns=)``
    # already returns a new frame in the history-found branch, so we mirror
    # that contract here for the no-history path.
    return df.copy()


def validate_row_count(s: Sample, n_target: int, *, ctx: str) -> None:
    """Raise if a balance ``Sample`` has dropped rows relative to ``n_target``.

    The most common balance-side trigger is ``na_action="drop"`` in IPW
    (``weighting_methods/ipw.py:738-745``) which silently drops rows.
    Both the diff-diff adapter and the future svy adapter need to guard
    against this — diff-diff's panel alignment and svy's ``rep_wgts``
    row alignment are both fragile in different ways.

    Parameters
    ----------
    s :
        The balance ``Sample`` whose row count to check.
    n_target :
        The expected number of rows (typically the row count of the
        DataFrame the user originally handed to ``Sample.from_frame``).
    ctx :
        A short label naming the calling adapter function — included in
        the error message so the user can trace which seam tripped.

    Raises
    ------
    ValueError
        When ``len(s.df) != n_target``.
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


def attach_balance_provenance(target: object, sample: Sample) -> None:
    """Attach ``sample`` to ``target`` as ``_balance_adjustment`` (provenance).

    Side-channel pattern recommended in
    ``~/balance_diff_diff/SVY_FUTURE_PROOFING.md`` §4. Lets downstream
    diagnostic tooling recover all 5 hidden-state pieces (see this
    module's docstring) from a diff-diff result or svy ``Sample`` via
    one stable attribute name.

    Idempotent — silently no-ops when ``target`` already has the
    attribute (so re-fitting does not clobber the original lineage).
    Falls back to ``target.__dict__`` if ``setattr`` is rejected (frozen
    msgspec / dataclass struct); silently no-ops if even that path is
    blocked.

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
        logger.debug(
            "Could not attach _balance_adjustment to %s (no __dict__)",
            type(target).__name__,
        )
        return
    target_dict["_balance_adjustment"] = sample
