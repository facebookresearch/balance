# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Love plot — visual covariate-imbalance diagnostic.

A "Love plot" (after Thomas Love) is the canonical visual for showing how
much each covariate's imbalance shrinks after applying weights. ``balance``
did not previously expose one; this is the first such helper. Reference:
R's ``cobalt::love.plot``.

The function is split into a primitive (``love_plot``) operating on raw
``pd.Series`` inputs of any covariate-keyed imbalance metric and a method
shortcut (``BalanceDFCovars.love_plot``) that pulls the chosen metric
(``"asmd"`` / ``"kld"`` / ``"emd"`` / ``"cvmd"`` / ``"ks"``) off a fitted
``BalanceFrame``'s lineage.

The primitive accepts a single ``before`` series for the pre-adjust
diagnostic case (when no "after" yet exists) and falls back to a single-
series scatter then. With both ``before`` and ``after`` it draws the
canonical before-vs-after view.
"""

from __future__ import annotations

import logging
import re

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__package__)

_BEFORE_COLOR: str = "#888888"
_AFTER_COLOR: str = "#0072B2"
_THRESHOLD_COLOR: str = "red"

# Regex for the conventional ``mean(<metric>)`` summary row that
# ``BalanceDF.{asmd,kld,emd,cvmd,ks}()`` append to their per-covariate
# series. Anchored to the exact metric names ``balance`` actually emits so
# we can't accidentally drop a real covariate that happens to be named
# like ``mean(age)``. Drop these before plotting because they are summary
# aggregates, not per-covariate values, and would distort scatter ordering.
_SUMMARY_ROW_PATTERN: re.Pattern[str] = re.compile(
    r"^mean\((?:asmd|kld|emd|cvmd|ks)\)$"
)


def _drop_summary_rows(s: pd.Series) -> pd.Series:
    """Drop any ``mean(<asmd|kld|emd|cvmd|ks>)`` summary row from a metric series."""
    keep_mask: pd.Series = (
        ~s.index.astype(str).to_series(index=s.index).str.match(_SUMMARY_ROW_PATTERN)
    )
    return s[keep_mask]


def love_plot(
    before: pd.Series,
    after: pd.Series | None = None,
    *,
    xlabel: str = "ASMD",
    threshold: float | None = 0.1,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Side-by-side scatter of per-covariate imbalance metric, before vs. after.

    A "Love plot" (after Thomas Love) is the canonical visual for showing
    how much each covariate's imbalance shrinks after applying weights.
    Reference: R's ``cobalt::love.plot``.

    The primitive is metric-agnostic: callers pass ASMD / KLD / EMD / CVMD
    / KS values (or any other covariate-keyed non-negative imbalance
    metric) as ``before`` / ``after`` series. The default ``xlabel="ASMD"``
    matches the historical use case; pass an explicit ``xlabel`` for
    other metrics. Reference lines are drawn at ``+threshold`` only, since
    these metrics are all non-negative by construction.

    Args:
        before: Per-covariate metric values. In two-series mode (when
            ``after`` is supplied) this is the BEFORE-adjustment metric
            (e.g. the output of the unadjusted view's
            ``BalanceDFCovars.asmd() / .kld() / .emd() / .cvmd() / .ks()``).
            In single-series mode (when ``after`` is ``None``) this is the
            only series plotted — typically the current / weighted metric
            when no "before" yet exists (the pre-adjust diagnostic case).
        after: Per-covariate metric values **with** weights. When ``None``
            the plot shows ``before`` only as a single-series scatter --
            this is the pre-adjust diagnostic case (no "after" yet
            exists), conceptually closer to ``asmd()`` than to
            ``asmd_improvement()``.
        xlabel: Label for the x-axis (also used as the legend label in
            single-series mode). Defaults to ``"ASMD"``; pass the metric
            name explicitly for non-ASMD callers (e.g. ``xlabel="KLD"``).
        threshold: Vertical reference line at ``+threshold``, default 0.1
            — the cobalt-convention "balance achieved" cutoff for ASMD.
            Pass ``None`` to skip the reference line entirely (recommended
            for metrics like KLD / EMD where no canonical cutoff exists).
            Must be non-negative if supplied (the metric is non-negative
            by construction).
        ax: Optional matplotlib ``Axes`` to draw into. If ``None``, a new
            figure is created sized to the number of covariates.

    Returns:
        matplotlib.axes.Axes: The Axes used, returned for further
        customization (titles, legend, save, etc.).

    Raises:
        ValueError: If ``threshold`` is negative; or if ``before`` and
            ``after`` (when both supplied) share no covariates; or if all
            entries are NaN after summary-row drop / index alignment.

    Examples:
        ::

            >>> import pandas as pd
            >>> from balance.stats_and_plots.love_plot import love_plot
            >>> before = pd.Series({"age": 0.42, "income": 0.31})
            >>> after = pd.Series({"age": 0.05, "income": 0.08})
            >>> ax = love_plot(before, after)  # doctest: +SKIP
            >>> # KLD with no reference line:
            >>> ax = love_plot(  # doctest: +SKIP
            ...     before, after, xlabel="KLD", threshold=None
            ... )
    """
    if threshold is not None and threshold < 0:
        # The metrics this primitive renders (ASMD, KLD, EMD, CVMD, KS) are
        # non-negative by construction; the canonical 0.1 cutoff (cobalt
        # convention for ASMD) is an upper bound on tolerable imbalance. A
        # negative threshold would draw the reference line on the negative
        # side of the x-axis (the line is drawn with ``axvline``), where no
        # value can ever fall, so the line would never visually align with
        # any plotted point and would silently mislead readers.
        raise ValueError(
            f"threshold must be non-negative or None; got {threshold!r}. "
            "Imbalance metrics are non-negative by construction; pass a "
            "non-negative cutoff or ``threshold=None`` to skip the line."
        )

    # Drop any ``mean(<metric>)`` summary row if present -- it is the
    # per-row mean, not a per-covariate value, and its inclusion would
    # distort the scatter ordering.
    before_clean: pd.Series = _drop_summary_rows(before)

    if after is None:
        # Pre-adjust mode: single-series scatter only.
        # Drop NaN entries so we don't end up with y-axis tick labels that
        # have no corresponding scatter point (NaN is a documented outcome
        # for unrepresented categorical levels in some metrics).
        before_clean = before_clean.dropna()
        if before_clean.empty:
            raise ValueError(
                "love_plot: no covariates to plot — all entries are NaN "
                "(this usually means the categorical levels in the sample / "
                "target frames are completely disjoint)."
            )
        # Sort by absolute value so the largest-imbalance covariate ends up
        # at the TOP of the plot (matplotlib y-axis increases upward, so
        # ``ascending=True`` puts smallest at y=0 / bottom).
        order = before_clean.abs().sort_values(ascending=True).index
        before_clean = before_clean.loc[order]
        if ax is None:
            _, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(before_clean))))
        y = np.arange(len(before_clean))
        ax.scatter(before_clean.values, y, marker="o", color=_AFTER_COLOR, label=xlabel)
        ax.set_yticks(y)
        ax.set_yticklabels(before_clean.index)
        if threshold is not None:
            # Only draw the +threshold line; the metric is non-negative.
            ax.axvline(threshold, linestyle="--", color=_THRESHOLD_COLOR, alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Covariate")
        ax.legend(loc="best")
        ax.grid(axis="x", alpha=0.3)
        return ax

    after_clean: pd.Series = _drop_summary_rows(after)

    # Align before/after indices. If they don't match exactly, fall back
    # to the intersection and warn (rather than silently dropping rows).
    if not before_clean.index.equals(after_clean.index):
        common = before_clean.index.intersection(after_clean.index)
        if len(common) == 0:
            raise ValueError("before and after share no covariates.")
        logger.warning(
            "love_plot: aligning to %d common covariates (before=%d, after=%d).",
            len(common),
            len(before_clean),
            len(after_clean),
        )
        before_clean = before_clean.loc[common]
        after_clean = after_clean.loc[common]

    # Drop covariates that are NaN in either series. Otherwise the
    # surviving series would carry y-axis tick labels with no
    # corresponding scatter point, and the abs/sort below would produce a
    # NaN-poisoned ordering.
    keep = (~before_clean.isna()) & (~after_clean.isna())
    if not keep.any():
        raise ValueError(
            "love_plot: no covariates to plot after dropping NaN entries "
            "(this usually means the categorical levels in the sample / "
            "target frames are completely disjoint)."
        )
    before_clean = before_clean[keep]
    after_clean = after_clean[keep]

    # Sort by absolute pre-weighting metric so the largest-imbalance
    # covariate is at the top of the plot.
    order = before_clean.abs().sort_values(ascending=True).index
    before_clean = before_clean.loc[order]
    after_clean = after_clean.loc[order]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(before_clean))))

    y = np.arange(len(before_clean))
    ax.scatter(
        before_clean.values, y, marker="o", label="Unweighted", color=_BEFORE_COLOR
    )
    ax.scatter(after_clean.values, y, marker="s", label="Weighted", color=_AFTER_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels(before_clean.index)
    if threshold is not None:
        # Only draw the +threshold line; the metric is non-negative.
        ax.axvline(threshold, linestyle="--", color=_THRESHOLD_COLOR, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Covariate")
    ax.legend(loc="best")
    ax.grid(axis="x", alpha=0.3)
    return ax
