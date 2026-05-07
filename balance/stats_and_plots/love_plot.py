# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Love plot — visual ASMD diagnostic.

A "Love plot" (after Thomas Love) is the canonical visual for showing how
much each covariate's ASMD shrinks after applying weights. ``balance`` did
not previously expose one; this is the first such helper. Reference: R's
``cobalt::love.plot``.

The function is split into a primitive (``love_plot``) operating on raw
``pd.Series`` inputs and a method shortcut (``BalanceDFCovars.love_plot``)
that pulls before/after ASMD off a fitted ``BalanceFrame``'s lineage.

The primitive accepts a single ``asmd_before`` series for the pre-adjust
diagnostic case (when no "after" yet exists) and falls back to a single-
series scatter then. With both ``asmd_before`` and ``asmd_after`` it
draws the canonical before-vs-after view.
"""

from __future__ import annotations

import logging

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__package__)

_BEFORE_COLOR: str = "#888888"
_AFTER_COLOR: str = "#0072B2"
_THRESHOLD_COLOR: str = "red"


def love_plot(
    asmd_before: pd.Series,
    asmd_after: pd.Series | None = None,
    *,
    threshold: float = 0.1,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Side-by-side scatter of per-covariate ASMD before vs. after weighting.

    A "Love plot" (after Thomas Love) is the canonical visual for showing
    how much each covariate's ASMD shrinks after applying weights.
    Reference: R's ``cobalt::love.plot``.

    Args:
        asmd_before: ASMD per covariate **without** weights. Typically the
            output of ``balance``'s ``BalanceDFCovars.asmd()`` on the
            unadjusted view.
        asmd_after: ASMD per covariate **with** weights. When ``None`` the
            plot shows ``asmd_before`` only as a single-series scatter --
            this is the pre-adjust diagnostic case (no "after" yet exists),
            conceptually closer to ``asmd()`` than to ``asmd_improvement()``.
        threshold: Vertical reference line, default 0.1 — the conventional
            "balance achieved" cutoff. Must be non-negative (ASMD itself
            is non-negative by construction).
        ax: Optional matplotlib ``Axes`` to draw into. If ``None``, a new
            figure is created sized to the number of covariates.

    Returns:
        matplotlib.axes.Axes: The Axes used, returned for further
        customization (titles, legend, save, etc.).

    Raises:
        ValueError: If ``threshold`` is negative, or if ``asmd_before`` and
            ``asmd_after`` (when both supplied) share no covariates.

    Examples:
        ::

            >>> import pandas as pd
            >>> from balance.stats_and_plots.love_plot import love_plot
            >>> before = pd.Series({"age": 0.42, "income": 0.31})
            >>> after = pd.Series({"age": 0.05, "income": 0.08})
            >>> ax = love_plot(before, after)  # doctest: +SKIP
    """
    if threshold < 0:
        # ASMD is non-negative by construction; the canonical 0.1 cutoff
        # (cobalt convention) is an upper bound on tolerable imbalance.
        # A negative threshold would draw the +/-threshold reference lines
        # on the wrong sides of the y-axis and silently mislead readers.
        raise ValueError(
            f"threshold must be non-negative; got {threshold!r}. "
            "ASMD is non-negative by construction; use the cobalt-convention "
            "default 0.1, or any other non-negative ASMD upper bound."
        )

    # Drop the conventional 'mean(asmd)' summary row if present -- it is
    # the per-row mean, not a per-covariate ASMD, and its inclusion would
    # distort the scatter ordering.
    before = asmd_before.drop("mean(asmd)", errors="ignore")

    if asmd_after is None:
        # Pre-adjust mode: single-series scatter of weighted ASMD only.
        # Sort by absolute ASMD so the largest-imbalance covariate ends up
        # at the TOP of the plot (matplotlib y-axis increases upward, so
        # ``ascending=True`` puts smallest at y=0 / bottom).
        order = before.abs().sort_values(ascending=True).index
        before = before.loc[order]
        if ax is None:
            _, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(before))))
        y = np.arange(len(before))
        ax.scatter(before.values, y, marker="o", color=_AFTER_COLOR, label="Weighted")
        ax.set_yticks(y)
        ax.set_yticklabels(before.index)
        ax.axvline(threshold, linestyle="--", color=_THRESHOLD_COLOR, alpha=0.5)
        ax.axvline(-threshold, linestyle="--", color=_THRESHOLD_COLOR, alpha=0.5)
        ax.set_xlabel("ASMD")
        ax.set_ylabel("Covariate")
        ax.legend(loc="best")
        ax.grid(axis="x", alpha=0.3)
        return ax

    after = asmd_after.drop("mean(asmd)", errors="ignore")

    # Align before/after indices. If they don't match exactly, fall back to
    # the intersection and warn (rather than silently dropping rows).
    if not before.index.equals(after.index):
        common = before.index.intersection(after.index)
        if len(common) == 0:
            raise ValueError("asmd_before and asmd_after share no covariates.")
        logger.warning(
            "love_plot: aligning to %d common covariates (before=%d, after=%d).",
            len(common),
            len(before),
            len(after),
        )
        before = before.loc[common]
        after = after.loc[common]

    # Sort by absolute pre-weighting ASMD so the largest-imbalance covariate
    # is at the top of the plot.
    order = before.abs().sort_values(ascending=True).index
    before = before.loc[order]
    after = after.loc[order]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(before))))

    y = np.arange(len(before))
    ax.scatter(before.values, y, marker="o", label="Unweighted", color=_BEFORE_COLOR)
    ax.scatter(after.values, y, marker="s", label="Weighted", color=_AFTER_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels(before.index)
    ax.axvline(threshold, linestyle="--", color=_THRESHOLD_COLOR, alpha=0.5)
    ax.axvline(-threshold, linestyle="--", color=_THRESHOLD_COLOR, alpha=0.5)
    ax.set_xlabel("ASMD")
    ax.set_ylabel("Covariate")
    ax.legend(loc="best")
    ax.grid(axis="x", alpha=0.3)
    return ax
