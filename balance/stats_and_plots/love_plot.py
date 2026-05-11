# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Love plot — visual covariate-imbalance diagnostic.

A "Love plot" (after Thomas Love) is the canonical visual for showing how
much each covariate's imbalance shrinks after applying weights. ``balance``
exposes a primitive (``love_plot``) operating on raw ``pd.Series`` inputs
and a method shortcut (``BalanceDFCovars.love_plot``) that pulls the chosen
metric off a fitted ``BalanceFrame``'s lineage.

The primitive supports static seaborn output, plotly output, and ASCII text
output. With both ``before`` and ``after`` it draws the canonical
before-vs-after view; with only ``before`` it draws a single-series
pre-adjust diagnostic.
"""

from __future__ import annotations

import logging
import math
import numbers
import re
from typing import Any, cast, get_args, Literal

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger: logging.Logger = logging.getLogger(__package__)

_BEFORE_COLOR: str = "#888888"
_AFTER_COLOR: str = "#0072B2"
_THRESHOLD_COLOR: str = "red"

_SUMMARY_ROW_PATTERN: re.Pattern[str] = re.compile(
    r"^mean\((?:asmd|kld|emd|cvmd|ks)\)$"
)
LovePlotLibrary = Literal["plotly", "seaborn", "balance"]
LovePlotOrderBy = Literal["before", "after", "diff", "alphabetical", "none"]
_LOVE_PLOT_LIBRARIES: tuple[str, ...] = cast(tuple[str, ...], get_args(LovePlotLibrary))
_LOVE_PLOT_ORDER_BY: tuple[str, ...] = cast(tuple[str, ...], get_args(LovePlotOrderBy))


def _drop_summary_rows(s: pd.Series) -> pd.Series:
    """Drop any ``mean(<asmd|kld|emd|cvmd|ks>)`` summary row from a metric series."""
    keep_mask: pd.Series = (
        ~s.index.astype(str).to_series(index=s.index).str.match(_SUMMARY_ROW_PATTERN)
    )
    return s[keep_mask]


def _validate_metric_series(series: pd.Series, *, name: str) -> pd.Series:
    """Validate and coerce a love-plot metric series to finite numeric values."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series; got {type(series)!r}.")

    clean = _drop_summary_rows(series)
    if clean.index.has_duplicates:
        duplicated = clean.index[clean.index.duplicated()].astype(str).unique().tolist()
        raise ValueError(
            f"{name} contains duplicate covariate labels after summary-row drop: "
            f"{duplicated}. Love plots require one metric value per covariate."
        )

    try:
        numeric = pd.to_numeric(clean, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain numeric imbalance metric values."
        ) from exc

    finite_or_na = numeric.isna() | np.isfinite(numeric.astype(float))
    if not finite_or_na.all():
        bad_labels = numeric.index[~finite_or_na].astype(str).tolist()
        raise ValueError(
            f"{name} contains non-finite imbalance values for covariates: "
            f"{bad_labels}."
        )
    return numeric


def _prepare_love_plot_data(
    before: pd.Series,
    after: pd.Series | None,
    *,
    order_by: LovePlotOrderBy,
) -> pd.DataFrame:
    """Clean, align, and sort love-plot input series."""
    if order_by not in _LOVE_PLOT_ORDER_BY:
        raise ValueError(
            f"order_by must be one of {_LOVE_PLOT_ORDER_BY}; got {order_by!r}."
        )

    before_clean: pd.Series = _validate_metric_series(before, name="before")
    if after is None:
        before_clean = before_clean.dropna()
        if before_clean.empty:
            raise ValueError(
                "love_plot: no covariates to plot — all entries are NaN "
                "(this usually means the categorical levels in the sample / "
                "target frames are completely disjoint)."
            )
        data = pd.DataFrame({"value": before_clean})
    else:
        after_clean: pd.Series = _validate_metric_series(after, name="after")
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

        keep = (~before_clean.isna()) & (~after_clean.isna())
        if not keep.any():
            raise ValueError(
                "love_plot: no covariates to plot after dropping NaN entries "
                "(this usually means the categorical levels in the sample / "
                "target frames are completely disjoint)."
            )
        data = pd.DataFrame(
            {"Unweighted": before_clean[keep], "Weighted": after_clean[keep]}
        )

    if order_by == "alphabetical":
        # Keep the internal bottom-to-top order used by seaborn/plotly,
        # but sort labels by their string representation so mixed-type
        # covariate labels (e.g. ints and strings) do not raise in pandas.
        order = sorted(data.index, key=lambda x: str(x), reverse=True)
        return data.loc[order]
    if order_by == "none":
        return data
    if order_by == "after" and "Weighted" in data.columns:
        order_values = data["Weighted"].abs()
    elif order_by == "diff" and "Weighted" in data.columns:
        # Signed difference (after - before): negative = improvement,
        # positive = worsening. ascending=True puts the smallest (most-
        # improved) at y=0 (bottom) and the most-worsened at the top, so
        # regressions float to the top of the plot.
        order_values = data["Weighted"] - data["Unweighted"]
    else:
        order_values = data.iloc[:, 0].abs()
    order = order_values.sort_values(ascending=True).index
    return data.loc[order]


def _seaborn_love_plot(
    data: pd.DataFrame,
    *,
    xlabel: str,
    threshold: float | None,
    ax: matplotlib.axes.Axes | None,
    line: bool,
) -> matplotlib.axes.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(data) + 1.0)))

    y = np.arange(len(data))
    if list(data.columns) == ["value"]:
        sns.scatterplot(
            x=data["value"].values,
            y=y,
            marker="o",
            color=_AFTER_COLOR,
            label=xlabel,
            ax=ax,
        )
    else:
        if line:
            ax.hlines(
                y=y,
                xmin=data["Unweighted"].values,
                xmax=data["Weighted"].values,
                colors="#BBBBBB",
                linewidth=1,
                zorder=1,
            )
        sns.scatterplot(
            x=data["Unweighted"].values,
            y=y,
            marker="o",
            label="Unweighted",
            color=_BEFORE_COLOR,
            ax=ax,
            zorder=2,
        )
        sns.scatterplot(
            x=data["Weighted"].values,
            y=y,
            marker="s",
            label="Weighted",
            color=_AFTER_COLOR,
            ax=ax,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(data.index)
    if threshold is not None:
        ax.axvline(threshold, linestyle="--", color=_THRESHOLD_COLOR, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Covariate")
    ax.legend(loc="best")
    ax.grid(axis="x", alpha=0.3)
    return ax


def _safe_plotly_show(fig: Any) -> None:
    """Display a Plotly figure, tolerating missing notebook mime renderers.

    ``fig.show()`` raises ``ValueError`` in CLI/test environments where Plotly's
    mime renderer cannot import ``nbformat``. Mirrors the guard used by
    :func:`weighted_comparisons_plots._safe_plotly_iplot`: log a warning and
    return the figure rather than crashing.
    """
    from balance.stats_and_plots.weighted_comparisons_plots import (
        _is_nbformat_mime_error,
    )

    try:
        fig.show()
    except ValueError as error:
        if not _is_nbformat_mime_error(error):
            raise
        logger.warning(
            "Plotly notebook mime rendering unavailable; returning figure "
            "without displaying it. Original error: %s",
            error,
        )


def _plotly_love_plot(
    data: pd.DataFrame,
    *,
    xlabel: str,
    threshold: float | None,
    line: bool,
    show: bool,
    **layout_kwargs: Any,
) -> Any:
    import plotly.graph_objects as go

    fig = go.Figure()
    # Use numeric y positions with explicit tick labels rather than a
    # categorical y-axis. Distinct index values that stringify to the same
    # text (e.g. ``1`` and ``"1"``) would otherwise collapse onto the same
    # categorical row in Plotly, overlapping markers and connector lines.
    y_positions = list(range(len(data)))
    y_tick_labels = [str(i) for i in data.index]
    if list(data.columns) == ["value"]:
        fig.add_trace(
            go.Scatter(
                x=data["value"].tolist(),
                y=y_positions,
                mode="markers",
                marker={"symbol": "circle", "color": _AFTER_COLOR},
                name=xlabel,
            )
        )
    else:
        if line:
            x_values: list[float | None] = []
            y_values: list[float | None] = []
            for y_pos, (_covar, row) in zip(y_positions, data.iterrows()):
                x_values.extend(
                    [float(row["Unweighted"]), float(row["Weighted"]), None]
                )
                y_values.extend([y_pos, y_pos, None])
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    line={"color": "#BBBBBB", "width": 1},
                    hoverinfo="skip",
                    showlegend=False,
                    name="Change",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=data["Unweighted"].tolist(),
                y=y_positions,
                mode="markers",
                marker={"symbol": "circle", "color": _BEFORE_COLOR},
                name="Unweighted",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data["Weighted"].tolist(),
                y=y_positions,
                mode="markers",
                marker={"symbol": "square", "color": _AFTER_COLOR},
                name="Weighted",
            )
        )
    if threshold is not None:
        fig.add_shape(
            type="line",
            x0=threshold,
            x1=threshold,
            xref="x",
            y0=0,
            y1=1,
            yref="paper",
            line={"color": _THRESHOLD_COLOR, "dash": "dash"},
            opacity=0.5,
        )
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title="Covariate",
        yaxis={"tickmode": "array", "tickvals": y_positions, "ticktext": y_tick_labels},
        width=900,
        height=max(360, 45 * len(data) + 120),
        margin={"l": 120, "r": 40, "t": 60, "b": 60},
        template="plotly_white",
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    if show:
        _safe_plotly_show(fig)
    return fig


def _ascii_scale_max(max_value: float, threshold: float | None) -> float:
    """Return a readable positive x-axis maximum for ASCII love plots."""
    candidates = [max_value]
    if threshold is not None:
        candidates.append(float(threshold))
    raw_max = max(candidates)
    if raw_max <= 0:
        return 1.0
    # Love plots are easiest to scan with tenths when values are small; keep
    # enough headroom so markers at the maximum do not run into annotations.
    step = 0.1 if raw_max <= 1 else 0.5
    return math.ceil((raw_max * 1.05) / step) * step


def _ascii_position(value: float, axis_max: float, *, width: int) -> int:
    if axis_max <= 0:
        return 0
    clipped = min(max(abs(value), 0.0), axis_max)
    return int(round((clipped / axis_max) * width))


def _ascii_axis(
    width: int, axis_max: float, threshold: float | None
) -> tuple[str, str]:
    """Build compact tick-label and threshold-guide rows for ASCII output."""
    if axis_max <= 1:
        tick_values = np.arange(0, axis_max + 0.0001, 0.1)
    else:
        tick_values = np.linspace(0, axis_max, 6)
    label_chars = [" "] * (width + 1)
    guide_chars = [" "] * (width + 1)
    last_label_end = -1
    for tick_value in tick_values:
        pos = _ascii_position(float(tick_value), axis_max, width=width)
        label = f"{tick_value:.1f}" if axis_max <= 1 else f"{tick_value:.2g}"
        if tick_value == 0:
            label = "0"
        if len(label) > len(label_chars):
            continue
        start = min(pos, len(label_chars) - len(label))
        end = start + len(label) - 1
        if start <= last_label_end:
            continue
        for offset, char in enumerate(label):
            label_chars[start + offset] = char
        last_label_end = end
    if threshold is not None:
        pos = _ascii_position(float(threshold), axis_max, width=width)
        guide_chars[pos] = "|"
    return "".join(label_chars).rstrip(), "".join(guide_chars).rstrip()


def _ascii_overlay_threshold(chars: list[str], position: int) -> None:
    """Overlay a threshold marker without hiding occupied marker/connector cells."""
    chars[position] = "|" if chars[position] == " " else "!"


def _ascii_change_plot(
    before_value: float,
    after_value: float,
    *,
    axis_max: float,
    width: int,
    threshold: float | None,
    line: bool,
) -> str:
    """Render one before/after row on a shared ASCII x-axis."""
    before_pos = _ascii_position(before_value, axis_max, width=width)
    after_pos = _ascii_position(after_value, axis_max, width=width)
    chars = [" "] * (width + 1)
    if line and after_pos < before_pos:
        for pos in range(after_pos + 1, before_pos):
            chars[pos] = "-"
        if after_pos + 1 < before_pos:
            chars[after_pos + 1] = "<"
    elif line and after_pos > before_pos:
        for pos in range(before_pos + 1, after_pos):
            chars[pos] = "-"
        if before_pos + 1 < after_pos:
            chars[after_pos - 1] = ">"
    chars[before_pos] = "o"
    chars[after_pos] = "*" if chars[after_pos] != "o" else "@"
    if threshold is not None:
        _ascii_overlay_threshold(
            chars, _ascii_position(float(threshold), axis_max, width=width)
        )
    return "".join(chars).rstrip()


def _ascii_love_plot(
    data: pd.DataFrame,
    *,
    xlabel: str,
    threshold: float | None,
    line: bool,
    bar_width: int,
) -> str:
    max_label_width = max(len(str(i)) for i in data.index)
    covar_width = min(max(max_label_width, len("Covariate")), 40)
    max_value = float(data.abs().max().max())
    axis_max = _ascii_scale_max(max_value, threshold)
    threshold_text = "none" if threshold is None else f"{threshold:.3g}"
    display_data = data.iloc[::-1]
    lines = [f"Love plot ({xlabel}) - Threshold = {threshold_text}"]

    if list(data.columns) == ["value"]:
        axis_labels, threshold_guide = _ascii_axis(bar_width, axis_max, threshold)
        lines.append(f"{'Covariate':<{covar_width}} | {xlabel:>10} | Plot")
        lines.append("-" * (covar_width + bar_width + 17))
        lines.append(f"{'':<{covar_width}} | {'':>10} | {axis_labels}")
        if threshold is not None:
            lines.append(f"{'':<{covar_width}} | {'':>10} | {threshold_guide}")
        for covar, row in display_data.iterrows():
            value = float(row["value"])
            marker_pos = _ascii_position(value, axis_max, width=bar_width)
            chars = [" "] * (bar_width + 1)
            chars[marker_pos] = "*"
            if threshold is not None:
                _ascii_overlay_threshold(
                    chars,
                    _ascii_position(float(threshold), axis_max, width=bar_width),
                )
            lines.append(
                f"{str(covar):<{covar_width}.{covar_width}} | "
                f"{value:>10.4g} | {''.join(chars).rstrip()}"
            )
        lines.extend(["", "Legend: * = value"])
        if threshold is not None:
            lines.append("        | = threshold, ! = threshold overlap")
    else:
        axis_labels, threshold_guide = _ascii_axis(bar_width, axis_max, threshold)
        lines.append(
            f"{'Covariate':<{covar_width}} | {'Unweighted':>10} | "
            f"{'Weighted':>10} | Change"
        )
        lines.append("-" * (covar_width + bar_width + 40))
        lines.append(f"{'':<{covar_width}} | {'':>10} | {'':>10} | {axis_labels}")
        if threshold is not None:
            lines.append(
                f"{'':<{covar_width}} | {'':>10} | {'':>10} | {threshold_guide}"
            )
        for covar, row in display_data.iterrows():
            before_value = float(row["Unweighted"])
            after_value = float(row["Weighted"])
            direction = "improved" if abs(after_value) <= abs(before_value) else "worse"
            change_plot = _ascii_change_plot(
                before_value,
                after_value,
                axis_max=axis_max,
                width=bar_width,
                threshold=threshold,
                line=line,
            )
            lines.append(
                f"{str(covar):<{covar_width}.{covar_width}} | "
                f"{before_value:>10.4g} | {after_value:>10.4g} | "
                f"{change_plot:<{bar_width + 1}} ({direction})"
            )
        lines.extend(["", "Legend: o = unweighted, * = weighted, @ = overlap"])
        if line:
            lines.extend(
                [
                    "        <----- improved (moving toward 0)",
                    "        -----> worse (moving away from 0)",
                ]
            )
        if threshold is not None:
            lines.append("        | = threshold, ! = threshold overlap")
    return "\n".join(lines)


def love_plot(
    before: pd.Series,
    after: pd.Series | None = None,
    *,
    xlabel: str = "ASMD",
    threshold: float | None = 0.1,
    ax: matplotlib.axes.Axes | None = None,
    library: LovePlotLibrary = "plotly",
    line: bool = True,
    order_by: LovePlotOrderBy = "diff",
    show: bool = False,
    bar_width: int = 50,
    **layout_kwargs: Any,
) -> Any:
    """Plot per-covariate imbalance, before vs. after weighting.

    Args:
        before: Per-covariate metric values before adjustment, or the only
            series to plot when ``after`` is ``None``.
        after: Per-covariate metric values after adjustment.
        xlabel: Metric label for the x-axis and ASCII header.
        threshold: Optional non-negative vertical/reference threshold. Pass
            ``None`` to skip it.
        ax: Optional matplotlib ``Axes`` for ``library="seaborn"``.
        library: One of ``"plotly"`` (default; interactive
            ``plotly.graph_objects.Figure``), ``"seaborn"`` (static
            seaborn/matplotlib axes), or ``"balance"`` (ASCII string).
        line: If ``True`` (the default) and both series are supplied,
            connect each before/after pair with a horizontal line.
        order_by: Covariate sorting. ``"diff"`` (default) orders by signed
            ``after - before`` so the most-improved covariates are at the
            bottom and the most-worsened are at the top; ``"before"`` /
            ``"after"`` order by absolute pre / post values; ``"alphabetical"``
            sorts by covariate name; ``"none"`` keeps input order.
        show: For ``library="plotly"``, whether to call ``fig.show()``.
        bar_width: Width of ASCII bars for ``library="balance"``.
        **layout_kwargs: Additional Plotly layout options when
            ``library="plotly"``.

    Returns:
        ``matplotlib.axes.Axes`` for static output, ``plotly`` ``Figure`` for
        plotly output, or ``str`` for ASCII output.

    Examples:
        ::

            >>> import pandas as pd
            >>> from balance.stats_and_plots.love_plot import love_plot
            >>> before = pd.Series({"age": 0.42, "income": 0.31})
            >>> after = pd.Series({"age": 0.05, "income": 0.08})
            >>> fig = love_plot(before, after)  # doctest: +SKIP
            >>> ax = love_plot(  # doctest: +SKIP
            ...     before, after, library="seaborn", line=True
            ... )
            >>> text = love_plot(  # doctest: +SKIP
            ...     before, after, library="balance", line=True
            ... )
    """
    if threshold is not None:
        if not isinstance(threshold, numbers.Real) or isinstance(threshold, bool):
            raise TypeError(
                f"threshold must be a non-negative finite number or None; got {type(threshold)!r}."
            )
        if not math.isfinite(float(threshold)) or threshold < 0:
            raise ValueError("threshold must be non-negative and finite, or None.")
    if library not in _LOVE_PLOT_LIBRARIES:
        raise ValueError(
            f"library must be one of {_LOVE_PLOT_LIBRARIES}; got {library!r}."
        )
    if not isinstance(line, bool):
        raise TypeError(f"line must be a bool; got {type(line)!r}.")
    if not isinstance(show, bool):
        raise TypeError(f"show must be a bool; got {type(show)!r}.")
    if not isinstance(bar_width, int) or isinstance(bar_width, bool) or bar_width <= 0:
        raise ValueError("bar_width must be a positive integer.")
    if ax is not None and library != "seaborn":
        raise ValueError("ax can only be used with library='seaborn'.")

    data = _prepare_love_plot_data(before, after, order_by=order_by)
    if library == "seaborn":
        if layout_kwargs:
            logger.warning(
                "Ignoring plotly layout kwargs for library=%r: %s",
                library,
                sorted(layout_kwargs.keys()),
            )
        return _seaborn_love_plot(
            data, xlabel=xlabel, threshold=threshold, ax=ax, line=line
        )
    if library == "plotly":
        return _plotly_love_plot(
            data,
            xlabel=xlabel,
            threshold=threshold,
            line=line,
            show=show,
            **layout_kwargs,
        )
    return _ascii_love_plot(
        data, xlabel=xlabel, threshold=threshold, line=line, bar_width=bar_width
    )
