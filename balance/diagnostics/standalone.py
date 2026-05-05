# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Standalone diagnostics for already-weighted data.

This module is the diagnostics surface for already-weighted data — when weights
came from outside balance (e.g., ``dd.aggregate_survey()``'s second-stage
design or a pre-weighted Hive table). It serves both the balance↔diff-diff and
the future balance↔svy integrations (per
``~/balance_diff_diff/SVY_FUTURE_PROOFING.md`` §6).

The ordinary ``BalanceFrame.diagnostics()`` route is not available when weights
come from outside because ``BalanceFrame`` requires a five-state lineage
(``weight_pre_adjust``, ``weight_adjusted_N``, ``_links["unadjusted"]``,
``_links["target"]``, ``_adjustment_model``) that doesn't exist in that case.

This module exposes the same diagnostic surface — covariate ASMD, Kish design
effect, ESS / ESSP, R-indicator, plus a brand-new ``love_plot`` — as a tiny
``(df, weights, target=None) -> ...`` API. Every function is a thin wrapper
around an existing private balance primitive; no math is duplicated. If
``BalanceFrame.diagnostics()`` ever changes its formula, the standalone version
follows automatically because both call the same primitive.

When to use the BalanceFrame route instead
------------------------------------------
If you are *fitting* the weights yourself, prefer the ordinary
``Sample.from_frame(...).set_target(...).adjust(method=...)`` path — you get a
richer lineage object with adjustment-improvement metrics, a fitted model, and
trim controls.

When to use this module
-----------------------
- DiD pipelines that consumed ``dd.aggregate_survey()`` second-stage weights.
- Pre-weighted Hive / Presto outputs.
- External IPW / rake / poststrat weights (e.g., from R's ``survey`` package).
- Quick sanity-checks before deciding whether to refit via balance.

Example
-------
>>> import pandas as pd
>>> import balance
>>> df = pd.DataFrame(
...     {"age": [22, 35, 47, 51, 28], "income_log": [9.1, 10.4, 11.2, 11.7, 9.8]}
... )
>>> weights = pd.Series([1.2, 0.8, 1.5, 0.7, 1.1])
>>> balance.diagnostics.compute_kish_design_effect(weights)  # doctest: +SKIP
1.0698...
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from balance.stats_and_plots import weighted_comparisons_stats, weights_stats

# matplotlib is already a hard dependency of balance (BUCK:54), but we wrap
# the import in a try/except so OSS installs without optional extras still
# expose the rest of the module surface and only fail when love_plot is
# actually called.
try:
    import matplotlib.axes  # noqa: F401  # used in type hints (deferred via from __future__ import annotations)
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False


logger: logging.Logger = logging.getLogger(__package__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_series(weights: list[float] | pd.Series | npt.NDArray) -> pd.Series:
    """Coerce input weights into a ``pd.Series`` and validate.

    Lightweight wrapper that mirrors the input-validation policy of
    ``balance.stats_and_plots.weights_stats._check_weights_series_are_valid``
    (see ``stats_and_plots/weights_stats.py``) without duplicating the
    primitive itself — the underlying ``design_effect`` /
    ``weighted_comparisons_stats.asmd`` calls re-run their own validation.

    Raises:
        ValueError: If ``weights`` is empty, non-finite, or has any negative
            entry.
    """
    if isinstance(weights, pd.Series):
        s = weights
    else:
        s = pd.Series(np.asarray(weights).reshape(-1))
    if s.empty:
        raise ValueError("weights must be non-empty.")
    arr = s.to_numpy()
    if not np.all(np.isfinite(arr)):
        raise ValueError("weights must be finite.")
    if (arr < 0).any():
        raise ValueError("weights must be non-negative.")
    return s


def _check_aligned(df: pd.DataFrame, weights: pd.Series) -> None:
    """Validate that ``df`` and ``weights`` have matching length.

    Raises:
        ValueError: If ``len(df) != len(weights)``.
    """
    if len(df) != len(weights):
        raise ValueError(
            f"df has {len(df)} rows but weights has {len(weights)} entries."
        )


# ---------------------------------------------------------------------------
# ASMD
# ---------------------------------------------------------------------------


def compute_asmd(
    df: pd.DataFrame,
    weights: list[float] | pd.Series | npt.NDArray,
    target_df: pd.DataFrame | None = None,
    target_weights: list[float] | pd.Series | npt.NDArray | None = None,
    *,
    std_type: str = "target",
) -> pd.Series:
    """Compute weighted ASMD for each covariate against a target.

    Wraps ``balance.stats_and_plots.weighted_comparisons_stats.asmd``
    (see ``stats_and_plots/weighted_comparisons_stats.py:459``) — the same
    primitive that ``BalanceFrame.covars().asmd()`` calls via
    ``balancedf_class.BalanceDF._apply_comparison_stat_to_BalanceDF``
    (see ``balancedf_class.py:1551``).

    Args:
        df: DataFrame of covariates for the (already-weighted) sample.
        weights: Weights aligned to ``df``.
        target_df: Optional target population to compare against. If ``None``,
            ``compute_asmd`` returns the ASMD of the weighted sample against
            the unweighted version of itself ("self-ASMD" — useful as a coarse
            "how strong are these weights" indicator).
        target_weights: Weights for ``target_df``. If ``None`` and ``target_df``
            is provided, the target is treated as unweighted.
        std_type: ``"target"`` (default), ``"sample"``, or ``"pooled"`` —
            controls which dataset's std-dev is used as the denominator. See
            the underlying primitive for the exact formulas.

    Returns:
        pd.Series indexed by covariate (with a final ``mean(asmd)`` row
        matching the BalanceFrame convention).

    Raises:
        ValueError: If ``df`` and ``weights`` are misaligned, weights are
            negative or non-finite, or ``target_df`` columns don't match
            ``df``.

    Examples:
        ::

            >>> import pandas as pd
            >>> from balance.diagnostics.standalone import compute_asmd
            >>> sample = pd.DataFrame({"x": [1.0, 2.0]})
            >>> target = pd.DataFrame({"x": [3.0, 4.0]})
            >>> compute_asmd(sample, [1, 2], target, [1, 2]).round(3)
            x             2.828
            mean(asmd)    2.828
            dtype: float64
    """
    weights_s = _as_series(weights)
    _check_aligned(df, weights_s)

    if target_df is None:
        target_df_use = df
        target_weights_s = pd.Series(np.ones(len(df)))
    else:
        if target_weights is None:
            target_weights_s = pd.Series(np.ones(len(target_df)))
        else:
            target_weights_s = _as_series(target_weights)
            _check_aligned(target_df, target_weights_s)
        missing = set(df.columns) - set(target_df.columns)
        if missing:
            raise ValueError(f"target_df is missing columns: {sorted(missing)}")
        target_df_use = target_df.loc[:, df.columns]

    return weighted_comparisons_stats.asmd(
        sample_df=df,
        target_df=target_df_use,
        sample_weights=weights_s,
        target_weights=target_weights_s,
        std_type=std_type,  # pyre-ignore[6]
    )


# ---------------------------------------------------------------------------
# Kish design effect / ESS / ESSP
# ---------------------------------------------------------------------------


def compute_kish_design_effect(
    weights: list[float] | pd.Series | npt.NDArray,
) -> float:
    """Kish's design effect: ``E[w^2] / E[w]^2``.

    Wraps ``balance.stats_and_plots.weights_stats.design_effect``
    (see ``stats_and_plots/weights_stats.py:116-118``) — the same primitive
    called by ``BalanceFrame._design_effect_diagnostics``.

    Args:
        weights: Sample weights (non-negative, finite).

    Returns:
        float: Always >= 1.0 for non-degenerate weights; equals 1.0 exactly
        when all weights are equal.

    Raises:
        ValueError: Weights empty, negative, or non-finite.

    Examples:
        ::

            >>> from balance.diagnostics.standalone import compute_kish_design_effect
            >>> compute_kish_design_effect([1.0, 1.0, 1.0])
            1.0
    """
    return float(weights_stats.design_effect(_as_series(weights)))


def compute_kish_ess(
    weights: list[float] | pd.Series | npt.NDArray,
) -> float:
    """Kish effective sample size: ``n / Deff = (sum(w))^2 / sum(w^2)``.

    Mirrors the second tuple element of
    ``BalanceFrame._design_effect_diagnostics`` (see ``balance_frame.py``).

    Args:
        weights: Sample weights (non-negative, finite).

    Returns:
        float: ESS in the same units as ``len(weights)``. Equals ``len(weights)``
        when all weights are equal.

    Raises:
        ValueError: Weights empty, negative, or non-finite.

    Examples:
        ::

            >>> from balance.diagnostics.standalone import compute_kish_ess
            >>> compute_kish_ess([1.0, 1.0, 1.0, 1.0])
            4.0
    """
    w = _as_series(weights)
    deff = compute_kish_design_effect(w)
    return float(len(w) / deff)


def compute_essp(
    weights: list[float] | pd.Series | npt.NDArray,
) -> float:
    """Proportional ESS: ``ESS / n = 1 / Deff``. Always in ``[0, 1]``.

    Same definition as the third tuple element of
    ``BalanceFrame._design_effect_diagnostics`` (see ``balance_frame.py``).

    Args:
        weights: Sample weights (non-negative, finite).

    Returns:
        float: A value in ``[0, 1]``; equals 1.0 when all weights are equal.

    Raises:
        ValueError: Weights empty, negative, or non-finite.

    Examples:
        ::

            >>> from balance.diagnostics.standalone import compute_essp
            >>> compute_essp([1.0, 1.0, 1.0])
            1.0
    """
    return 1.0 / compute_kish_design_effect(weights)


# ---------------------------------------------------------------------------
# R-indicator
# ---------------------------------------------------------------------------


def compute_r_indicator(
    df: pd.DataFrame,
    weights: list[float] | pd.Series | npt.NDArray,
    target_df: pd.DataFrame | None = None,
    n_target: int | None = None,
    formula: str | None = None,
) -> dict[str, float]:
    """R-indicator (Schouten et al.) approximated from inverse weights.

    Wraps ``balance.stats_and_plots.weighted_comparisons_stats.r_indicator``
    (see ``stats_and_plots/weighted_comparisons_stats.py:67-72``) — the same
    primitive called from ``BalanceDFWeights.r_indicator()`` in
    ``balancedf_class.py``.

    Approximates response propensities as the inverse weights ``p_i = 1/w_i``,
    rescaled by the maximum propensity when any exceeds 1 — i.e. when
    ``max_i p_i > 1``, all propensities are divided by ``max_i p_i`` so the
    largest is exactly 1 and relative ratios are preserved. This matches the
    rescaling behavior of ``BalanceDFWeights.r_indicator()`` exactly (see
    ``balancedf_class.py:3093-3096``); per-element clipping ``min(1/w_i, 1)``
    is mathematically different — e.g. weights ``[0.5, 1.0, 2.0]`` rescale to
    ``[1.0, 0.5, 0.25]`` (preserving ratios) instead of clipping to
    ``[1.0, 1.0, 0.5]``.

    The target propensity vector is a unit vector of length ``len(target_df)``
    (or ``n_target`` if explicit). This MATCHES ``BalanceDFWeights.r_indicator()``
    at ``balancedf_class.py:3108`` which uses ``len(target_sample.weight_series)``.
    When neither is given, the function falls back to ``len(df)`` (the sample
    size) — this is correct when no separate target population exists (e.g.
    diagnosing a single panel without an external reference frame), but will
    differ from the BalanceFrame route whenever a target with a different size
    is in play.

    Args:
        df: Covariate frame (used only for length validation; the R-indicator
            formula uses inverse weights as propensities).
        weights: Sample weights.
        target_df: Optional target population frame. Only ``len(target_df)``
            is used (target rows are treated as having uniform propensity 1.0).
        n_target: Explicit target size, overrides ``len(target_df)`` when given.
            Useful when a target frame is unavailable but its row count is
            known (e.g. published ACS marginals with a known N).
        formula: Reserved for future use. Currently ignored.

    Returns:
        dict[str, float]: ``{"r_indicator": <value in [0, 1]>}`` — 1.0 means
        perfectly representative.

    Raises:
        ValueError: Weights empty, non-finite, or misaligned with ``df``.

    Examples:
        ::

            >>> import pandas as pd
            >>> from balance.diagnostics.standalone import compute_r_indicator
            >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
            >>> result = compute_r_indicator(df, [1.0, 1.0, 1.0])
            >>> "r_indicator" in result
            True
    """
    if formula is not None:
        logger.warning(
            "compute_r_indicator: 'formula' argument is reserved for future "
            "use and is currently ignored."
        )
    weights_s = _as_series(weights)
    _check_aligned(df, weights_s)
    sample_weights: npt.NDArray = weights_s.to_numpy(dtype=float)
    if not np.isfinite(sample_weights).all():
        raise ValueError("compute_r_indicator requires finite weights")
    if (sample_weights <= 0).any():
        raise ValueError("compute_r_indicator requires strictly positive weights")
    # Mirror ``BalanceDFWeights.r_indicator`` (balancedf_class.py:3093-3096):
    # take inverse-weights as propensities and, if any exceeds 1, rescale ALL
    # of them by the maximum (preserves relative ratios).
    sample_propensity: npt.ArrayLike = np.reciprocal(sample_weights)
    max_propensity: float = float(np.asarray(sample_propensity).max())
    if max_propensity > 1.0:
        sample_propensity = np.asarray(sample_propensity) / max_propensity
    # Resolve target size: explicit n_target wins, then target_df, else fall
    # back to sample size (no-target diagnostic mode). The underlying
    # ``r_indicator`` primitive concatenates sample + target propensities and
    # computes variance over the combined vector (n_obs = len(sample) +
    # len(target)), so getting the target length right is required for
    # numerical parity with the BalanceFrame route.
    if n_target is not None:
        if n_target < 1:
            raise ValueError(
                f"n_target must be >= 1 when provided; got {n_target}. "
                "The R-indicator requires at least 2 propensity values across "
                "sample + target combined; an empty or negative target makes "
                "the underlying variance computation undefined."
            )
        target_size: int = n_target
    elif target_df is not None:
        target_size = len(target_df)
    else:
        target_size = len(df)
    target_p: npt.ArrayLike = np.ones(target_size)
    r = weighted_comparisons_stats.r_indicator(sample_propensity, target_p)
    return {"r_indicator": float(r)}


# ---------------------------------------------------------------------------
# Love plot — NEW (balance does not have one; cf. R's cobalt::love.plot).
# ---------------------------------------------------------------------------


def love_plot(
    asmd_before: pd.Series,
    asmd_after: pd.Series,
    *,
    threshold: float = 0.1,
    ax: Any | None = None,
) -> Any:
    """Love plot: side-by-side ASMD before vs. after weighting.

    A "Love plot" (after Thomas Love) is the canonical visual for showing how
    much each covariate's ASMD shrunk after applying weights. balance does not
    currently expose one; this is the first implementation in the package.
    Reference: R's ``cobalt::love.plot``.

    Args:
        asmd_before: ASMD per covariate **without** weights (e.g., output of
            ``compute_asmd(df, weights=np.ones(len(df)), target_df=...)``).
        asmd_after: ASMD per covariate **with** weights.
        threshold: Vertical reference line, default 0.1 — the conventional
            "balance achieved" cutoff.
        ax: Optional matplotlib ``Axes`` to draw into. If ``None``, a new
            figure is created.

    Returns:
        matplotlib.axes.Axes: Returned for further customization (titles,
        legends, save).

    Raises:
        ImportError: If matplotlib is not available — install via
            ``pip install matplotlib`` or rely on ``compute_asmd`` directly
            for headless pipelines.
        ValueError: If ``asmd_before`` and ``asmd_after`` share no covariates.

    Examples:
        ::

            >>> import pandas as pd
            >>> from balance.diagnostics.standalone import love_plot
            >>> before = pd.Series({"age": 0.42, "income": 0.31})
            >>> after = pd.Series({"age": 0.05, "income": 0.08})
            >>> ax = love_plot(before, after)  # doctest: +SKIP
    """
    if not _MATPLOTLIB_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "love_plot requires matplotlib; "
            "install via `pip install matplotlib` or use compute_asmd directly."
        )

    # Align — drop the conventional 'mean(asmd)' summary row if present.
    before = asmd_before.drop("mean(asmd)", errors="ignore")
    after = asmd_after.drop("mean(asmd)", errors="ignore")
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

    # Sort by absolute pre-weighting ASMD (largest at top — cobalt convention).
    order = before.abs().sort_values(ascending=True).index
    before = before.loc[order]
    after = after.loc[order]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(before))))

    y = np.arange(len(before))
    ax.scatter(before.values, y, marker="o", label="Unweighted", color="#888888")
    ax.scatter(after.values, y, marker="s", label="Weighted", color="#0072B2")
    ax.set_yticks(y)
    ax.set_yticklabels(before.index)
    ax.axvline(threshold, linestyle="--", color="red", alpha=0.5)
    ax.axvline(-threshold, linestyle="--", color="red", alpha=0.5)
    ax.set_xlabel("ASMD")
    ax.set_ylabel("Covariate")
    ax.legend(loc="best")
    ax.grid(axis="x", alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


def diagnostics_table(
    df: pd.DataFrame,
    weights: list[float] | pd.Series | npt.NDArray,
    target_df: pd.DataFrame | None = None,
    target_weights: list[float] | pd.Series | npt.NDArray | None = None,
    *,
    kish: bool = True,
    essp: bool = True,
    asmd: bool = True,
    r_indicator: bool = True,
) -> pd.DataFrame:
    """One-call diagnostic summary, returned as a long-format DataFrame.

    Composition of the other functions in this module. Convenient for logging
    and notebook display; not a new primitive.

    Args:
        df: Covariates of the (already-weighted) sample.
        weights: Sample weights aligned to ``df``.
        target_df: Optional target population to compare against (drives the
            ASMD denominator and enables R-indicator).
        target_weights: Optional weights for ``target_df``.
        kish: If True, include ``deff`` and ``ess`` rows.
        essp: If True, include ``essp`` row.
        asmd: If True, include one ``asmd_<col>`` row per covariate plus the
            summary ``asmd_mean(asmd)`` row.
        r_indicator: If True, include the ``r_indicator`` row. The row is
            emitted regardless of whether ``target_df`` is provided -- when
            it is None, ``compute_r_indicator`` falls back to the sample
            size (``len(df)``) for the target propensity vector.

    Returns:
        pd.DataFrame: Single ``"value"`` column, indexed by metric name (e.g.,
        ``"deff"``, ``"ess"``, ``"essp"``, ``"asmd_<col>"``, ``"r_indicator"``).

    Raises:
        ValueError: Same conditions as the underlying primitives.

    Examples:
        ::

            >>> import pandas as pd
            >>> from balance.diagnostics.standalone import diagnostics_table
            >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
            >>> w = pd.Series([1.0, 1.0, 1.0])
            >>> table = diagnostics_table(df, w, asmd=False, r_indicator=False)
            >>> "deff" in table.index
            True
    """
    rows: dict[str, float] = {}
    # Normalise weights once so deff / ess / essp share a single
    # canonical view -- and so per-public-helper recomputation of deff
    # (and its internal ``_as_series`` validation) is avoided. We must
    # use ``_as_series`` here rather than ``len(weights)`` directly:
    # ``_as_series`` flattens multi-dim numpy arrays via ``.reshape(-1)``
    # before counting, so for a 2D array of shape (n, k) the canonical
    # length is n*k (matching ``compute_kish_ess``), whereas
    # ``len(weights)`` would return n and produce an ESS that disagrees
    # with the per-helper API.
    weights_s: pd.Series | None = None
    n_canonical: int = 0
    deff_cached: float | None = None
    if kish or essp:
        weights_s = _as_series(weights)
        n_canonical = len(weights_s)
        deff_cached = compute_kish_design_effect(weights_s)
    if kish:
        deff_value: float = deff_cached  # pyre-ignore[9]: narrowed above
        rows["deff"] = deff_value
        rows["ess"] = float(n_canonical / deff_value)
    if essp:
        deff_value_for_essp: float = deff_cached  # pyre-ignore[9]
        rows["essp"] = float(1.0 / deff_value_for_essp)
    if asmd:
        a = compute_asmd(df, weights, target_df, target_weights)
        for k, v in a.items():
            rows[f"asmd_{k}"] = float(v)
    if r_indicator:
        # Always emit the R-indicator row when requested. ``compute_r_indicator``
        # supports the no-target case (it falls back to ``len(df)`` for the
        # target propensity vector size); suppressing the row here would
        # silently drop one of the advertised diagnostics for already-weighted
        # datasets that have no separate target frame.
        #
        # When ``target_df`` IS provided, forwarding it makes the R-indicator's
        # target propensity vector have length ``len(target_df)`` -- matches
        # ``BalanceDFWeights.r_indicator()`` at ``balancedf_class.py:3108``
        # and is required for numerical parity when sample and target sizes
        # differ.
        rows.update(compute_r_indicator(df, weights, target_df=target_df))
    return pd.DataFrame.from_dict(rows, orient="index", columns=["value"])
