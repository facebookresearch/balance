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

The primitive supports matplotlib/seaborn-style static output, plotly output,
and ASCII text output. With both ``before`` and ``after`` it draws the
canonical before-vs-after view; with only ``before`` it draws a single-series
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

logger: logging.Logger = logging.getLogger(__package__)

_BEFORE_COLOR: str = "#888888"
_AFTER_COLOR: str = "#0072B2"
_THRESHOLD_COLOR: str = "red"

_SUMMARY_ROW_PATTERN: re.Pattern[str] = re.compile(
    r"^mean\((?:asmd|kld|emd|cvmd|ks)\)$"
)
LovePlotLibrary = Literal["seaborn", "matplotlib", "plotly", "balance"]
LovePlotOrderBy = Literal["before", "after", "max", "alphabetical", "none"]
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
        # Keep the internal bottom-to-top order used by matplotlib/plotly,
        # but sort labels by their string representation so mixed-type
        # covariate labels (e.g. ints and strings) do not raise in pandas.
        order = sorted(data.index, key=lambda x: str(x), reverse=True)
        return data.loc[order]
    if order_by == "none":
        return data
    if order_by == "after" and "Weighted" in data.columns:
        order_values = data["Weighted"].abs()
    elif order_by == "max" and "Weighted" in data.columns:
        order_values = data.abs().max(axis=1)
    else:
        order_values = data.iloc[:, 0].abs()
    order = order_values.sort_values(ascending=True).index
    return data.loc[order]


def _matplotlib_love_plot(
    data: pd.DataFrame,
    *,
    xlabel: str,
    threshold: float | None,
    ax: matplotlib.axes.Axes | None,
    line: bool,
) -> matplotlib.axes.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(data))))

    y = np.arange(len(data))
    if list(data.columns) == ["value"]:
        ax.scatter(
            data["value"].values, y, marker="o", color=_AFTER_COLOR, label=xlabel
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
        ax.scatter(
            data["Unweighted"].values,
            y,
            marker="o",
            label="Unweighted",
            color=_BEFORE_COLOR,
            zorder=2,
        )
        ax.scatter(
            data["Weighted"].values,
            y,
            marker="s",
            label="Weighted",
            color=_AFTER_COLOR,
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
    y_labels = data.index.astype(str).tolist()
    if list(data.columns) == ["value"]:
        fig.add_trace(
            go.Scatter(
                x=data["value"].tolist(),
                y=y_labels,
                mode="markers",
                marker={"symbol": "circle", "color": _AFTER_COLOR},
                name=xlabel,
            )
        )
    else:
        if line:
            x_values: list[float | None] = []
            y_values: list[str | None] = []
            for covar, row in data.iterrows():
                covar_label = str(covar)
                x_values.extend(
                    [float(row["Unweighted"]), float(row["Weighted"]), None]
                )
                y_values.extend([covar_label, covar_label, None])
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
                y=y_labels,
                mode="markers",
                marker={"symbol": "circle", "color": _BEFORE_COLOR},
                name="Unweighted",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data["Weighted"].tolist(),
                y=y_labels,
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
        height=max(300, 30 * len(data)),
        template="plotly_white",
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    if show:
        fig.show()
    return fig


def _ascii_bar(value: float, max_value: float, *, width: int, char: str) -> str:
    if max_value <= 0:
        return ""
    n_chars = int(round((abs(value) / max_value) * width))
    if value != 0 and n_chars == 0:
        n_chars = 1
    return char * n_chars


def _ascii_love_plot(
    data: pd.DataFrame,
    *,
    xlabel: str,
    threshold: float | None,
    bar_width: int,
) -> str:
    max_label_width = max(len(str(i)) for i in data.index)
    covar_width = min(max(max_label_width, len("Covariate")), 40)
    max_value = float(data.abs().max().max())
    threshold_text = "none" if threshold is None else f"{threshold:.3g}"
    lines = [
        f"Love plot ({xlabel})",
        f"Threshold: {threshold_text}",
    ]
    display_data = data.iloc[::-1]
    if list(data.columns) == ["value"]:
        lines.append(f"{'Covariate':<{covar_width}} | {xlabel:>10} | Plot")
        lines.append("-" * (covar_width + bar_width + 18))
        for covar, row in display_data.iterrows():
            value = float(row["value"])
            bar = _ascii_bar(value, max_value, width=bar_width, char="#")
            lines.append(
                f"{str(covar):<{covar_width}.{covar_width}} | {value:>10.4g} | {bar}"
            )
    else:
        lines.append(
            f"{'Covariate':<{covar_width}} | {'Unweighted':>10} | {'Weighted':>10} | Change"
        )
        lines.append("-" * (covar_width + bar_width + 40))
        for covar, row in display_data.iterrows():
            before_value = float(row["Unweighted"])
            after_value = float(row["Weighted"])
            before_bar = _ascii_bar(before_value, max_value, width=bar_width, char=".")
            after_bar = _ascii_bar(after_value, max_value, width=bar_width, char="#")
            direction = "improved" if after_value <= before_value else "worse"
            lines.append(
                f"{str(covar):<{covar_width}.{covar_width}} | "
                f"{before_value:>10.4g} | {after_value:>10.4g} | "
                f"{before_bar} -> {after_bar} ({direction})"
            )
    return "\n".join(lines)


def love_plot(
    before: pd.Series,
    after: pd.Series | None = None,
    *,
    xlabel: str = "ASMD",
    threshold: float | None = 0.1,
    ax: matplotlib.axes.Axes | None = None,
    library: LovePlotLibrary = "seaborn",
    line: bool = False,
    order_by: LovePlotOrderBy = "before",
    show: bool = False,
    bar_width: int = 30,
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
        ax: Optional matplotlib ``Axes`` for ``library="seaborn"`` or
            ``library="matplotlib"``.
        library: One of ``"seaborn"``/``"matplotlib"`` (static matplotlib
            axes), ``"plotly"`` (interactive ``plotly.graph_objects.Figure``),
            or ``"balance"`` (ASCII string).
        line: If ``True`` and both series are supplied, connect each
            before/after pair with a horizontal line.
        order_by: Covariate sorting. ``"before"`` (default) orders by the
            first series, ``"after"`` by weighted values, ``"max"`` by the
            larger of the two values, ``"alphabetical"`` by covariate name,
            and ``"none"`` keeps input order.
        show: For ``library="plotly"``, whether to call ``fig.show()``.
        bar_width: Width of ASCII bars for ``library="balance"``.
        **layout_kwargs: Additional Plotly layout options when
            ``library="plotly"``.

    Returns:
        ``matplotlib.axes.Axes`` for static output, ``plotly`` ``Figure`` for
        plotly output, or ``str`` for ASCII output.
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
    if ax is not None and library not in ("seaborn", "matplotlib"):
        raise ValueError(
            "ax can only be used with library='seaborn' or library='matplotlib'."
        )

    data = _prepare_love_plot_data(before, after, order_by=order_by)
    if library in ("seaborn", "matplotlib"):
        if layout_kwargs:
            logger.warning(
                "Ignoring plotly layout kwargs for library=%r: %s",
                library,
                sorted(layout_kwargs.keys()),
            )
        return _matplotlib_love_plot(
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
        data, xlabel=xlabel, threshold=threshold, bar_width=bar_width
    )
