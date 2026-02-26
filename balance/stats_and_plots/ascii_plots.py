# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from balance.stats_and_plots.weighted_comparisons_plots import (
    DataFrameWithWeight,
    naming_legend,
)
from balance.stats_and_plots.weighted_stats import relative_frequency_table
from balance.stats_and_plots.weights_stats import _check_weights_are_valid
from balance.util import choose_variables, rm_mutual_nas

logger: logging.Logger = logging.getLogger(__package__)

# Characters used to distinguish datasets in ASCII bars.
# Each dataset gets a unique character from this list.
BAR_CHARS: List[str] = ["█", "▒", "▐", "░", "▄", "▀"]


def _auto_n_bins(n_samples: int, n_unique: int) -> int:
    """Pick a number of bins using Sturges' rule, capped at unique values."""
    import math

    if n_samples <= 1:
        return 1
    sturges = math.ceil(math.log2(n_samples) + 1)
    # Don't exceed the number of unique values, and clamp to [2, 50]
    return max(2, min(sturges, n_unique, 50))


def _auto_bar_width(label_width: int) -> int:
    """Pick bar_width to fit within terminal width.

    Used by grouped barplots and histograms where each dataset gets its own
    line within a row (single bar per line).
    """
    import shutil

    term_width = shutil.get_terminal_size((80, 24)).columns
    # Each line: label_width + " | " (3) + bar + " (XX.X%)" (9)
    available = term_width - label_width - 3 - 9
    return max(10, available)


def _auto_bar_width_columnar(range_width: int, n_columns: int) -> int:
    """Pick per-column bar_width for a columnar (side-by-side) layout.

    Used by :func:`ascii_comparative_hist` where all datasets are rendered as
    columns on the same line.  Each column needs space for the bar, a
    percentage string (~6 chars), and inter-column separators (`` | ``, 3
    chars each).
    """
    import shutil

    term_width = shutil.get_terminal_size((80, 24)).columns
    # "Range  | col1 | col2 | ..."
    # range_width + " | " (3+1 for padding) consumed by the label column
    available = term_width - range_width - 4
    per_col = max(10, (available - (n_columns - 1) * 3) // n_columns - 6)
    return per_col


def _weighted_histogram(
    values: pd.Series,
    weights: Optional[pd.Series],
    bin_edges: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Computes a weighted histogram and normalizes counts to proportions.

    Args:
        values: The numeric data values.
        weights: Optional weights. If None, uniform weights are used.
        bin_edges: Pre-computed bin edges (length n_bins + 1).

    Returns:
        Array of proportions for each bin (sums to 1.0, or all zeros if empty).
    """
    _check_weights_are_valid(weights)
    weights_arr: Optional[npt.NDArray[np.floating]] = None
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=float)
    counts, _ = np.histogram(values, bins=bin_edges, weights=weights_arr)
    total = counts.sum()
    if total > 0:
        return counts / total
    return np.zeros_like(counts, dtype=float)


def _render_horizontal_bars(
    label: str,
    proportions: Dict[str, float],
    legend_names: List[str],
    bar_width: int,
    max_value: float,
    label_width: int,
) -> str:
    """Renders a group of horizontal bars for one category or bin.

    Each dataset gets its own line with a distinct character and a percentage
    label at the end.  When a proportion is non-zero but too small to render
    even one bar character, a single dot (``.``) is shown so that the reader
    can distinguish "present but tiny" from "truly zero".

    Args:
        label: The category label or bin range string.
        proportions: Dict mapping dataset legend name to its proportion value.
        legend_names: Ordered list of legend names for consistent ordering.
        bar_width: Maximum character width of the longest bar.
        max_value: The maximum proportion value across all bars (used for scaling).
        label_width: Character width reserved for the label column.

    Returns:
        Multi-line string of the grouped bars for this label.
    """
    lines: List[str] = []
    for i, name in enumerate(legend_names):
        prop = proportions.get(name, 0.0)
        char = BAR_CHARS[i % len(BAR_CHARS)]
        if max_value > 0:
            bar_len = int(round((prop / max_value) * bar_width))
        else:
            bar_len = 0
        if bar_len > 0:
            bar = char * bar_len
        elif prop > 0:
            # Non-zero proportion too small to render — show a dot
            bar = "."
        else:
            bar = ""
        if i == 0:
            prefix = label.ljust(label_width)
        else:
            prefix = " " * label_width
        lines.append(f"{prefix} | {bar} ({prop:.1%})")
    return "\n".join(lines)


def _build_legend(legend_names: List[str]) -> str:
    """Builds a legend string mapping characters to dataset names.

    Args:
        legend_names: Ordered list of dataset legend names.

    Returns:
        A two-line legend string: the first line maps bar characters to
        dataset names; the second explains how to interpret bar lengths.
    """
    parts: List[str] = []
    for i, name in enumerate(legend_names):
        char = BAR_CHARS[i % len(BAR_CHARS)]
        parts.append(f"{char} {name}")
    return (
        "Legend: "
        + "  ".join(parts)
        + "\nBar lengths are proportional to weighted frequency within each dataset."
    )


def ascii_plot_bar(
    dfs: List[DataFrameWithWeight],
    names: List[str],
    column: str,
    weighted: bool = True,
    bar_width: Optional[int] = None,
    dist_type: Optional[str] = None,
    separate_categories: bool = True,
) -> str:
    """Produces an ASCII grouped barplot for a single categorical variable.

    Uses :func:`relative_frequency_table` to compute weighted proportions for
    each dataset, then renders grouped horizontal bars.

    How to read the output:
        Each row is a category value. Within a row, each dataset gets its
        own bar drawn with a distinct fill character (``█``, ``▓``, etc.).

        - The percentage at the end of each bar is the weighted proportion
          of that category within its dataset (i.e., proportions within
          each dataset sum to 100%).
        - Bar lengths are scaled so that the longest bar across all
          datasets spans the full ``bar_width``.

    Args:
        dfs: List of DataFrameWithWeight dicts.
        names: Names for each DataFrame (e.g., ["self", "target"]).
        column: The categorical column name to plot.
        weighted: Whether to use weights. Defaults to True.
        bar_width: Maximum character width for bars. Defaults to None,
            which auto-detects based on terminal width.
        dist_type: Accepted for compatibility but only "hist_ascii" is supported.
            A warning is logged if any other value is passed.
        separate_categories: If True, insert a blank line between categories
            for readability. Defaults to True.

    Returns:
        ASCII barplot text for this variable.

    Example:
        ::

            >>> df_a = pd.DataFrame({"color": ["red", "blue", "blue", "green"]})
            >>> df_b = pd.DataFrame({"color": ["red", "red", "blue", "green"]})
            >>> dfs = [
            ...     {"df": df_a, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
            ...     {"df": df_b, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
            ... ]
            >>> print(ascii_plot_bar(dfs, names=["self", "target"],
            ...       column="color", bar_width=20))
            === color (categorical) ===
            <BLANKLINE>
            Category | sample  population
                     |
            blue     | ████████████████████ (50.0%)
                     | ▒▒▒▒▒▒▒▒▒▒ (25.0%)
            <BLANKLINE>
            green    | ██████████ (25.0%)
                     | ▒▒▒▒▒▒▒▒▒▒ (25.0%)
            <BLANKLINE>
            red      | ██████████ (25.0%)
                     | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (50.0%)
            <BLANKLINE>
            Legend: █ sample  ▒ population
            Bar lengths are proportional to weighted frequency within each dataset.
            <BLANKLINE>
    """
    if dist_type is not None and dist_type != "hist_ascii":
        logger.warning(
            f"ASCII plots only support dist_type='hist_ascii'. "
            f"Ignoring dist_type='{dist_type}' and using 'hist_ascii'."
        )
    legend_names: List[str] = [naming_legend(n, names) for n in names]

    # Compute proportions per dataset
    all_props: List[pd.DataFrame] = []
    for ii, d in enumerate(dfs):
        a_series = d["df"][column]
        _w = d["weight"]
        if weighted and _w is not None:
            a_series, _w = rm_mutual_nas(a_series, _w)
            a_series.name = column
        freq_table = relative_frequency_table(a_series, w=_w if weighted else None)
        freq_table["dataset"] = legend_names[ii]
        all_props.append(freq_table)

    combined = pd.concat(all_props, ignore_index=True)

    # Get all unique categories in stable order
    categories: List[str] = list(combined[column].unique())

    # Find max proportion for bar scaling
    max_value: float = float(combined["prop"].max()) if len(combined) > 0 else 1.0

    # Compute label width
    label_width = max(len(str(c)) for c in categories) if categories else 8
    label_width = max(label_width, 8)  # minimum width for "Category"

    if bar_width is None:
        bar_width = _auto_bar_width(label_width)

    # Build output
    lines: List[str] = []
    lines.append(f"=== {column} (categorical) ===")
    lines.append("")

    # Header
    header_label = "Category".ljust(label_width)
    lines.append(f"{header_label} | {'  '.join(legend_names)}")
    lines.append(f"{' ' * label_width} |")

    for ci, cat in enumerate(categories):
        if separate_categories and ci > 0:
            lines.append("")
        cat_data = combined[combined[column] == cat]
        proportions: Dict[str, float] = {}
        for _, row in cat_data.iterrows():
            proportions[row["dataset"]] = float(row["prop"])
        lines.append(
            _render_horizontal_bars(
                str(cat), proportions, legend_names, bar_width, max_value, label_width
            )
        )

    lines.append("")
    lines.append(_build_legend(legend_names))
    lines.append("")

    return "\n".join(lines)


def ascii_plot_hist(
    dfs: List[DataFrameWithWeight],
    names: List[str],
    column: str,
    weighted: bool = True,
    n_bins: Optional[int] = None,
    bar_width: Optional[int] = None,
    dist_type: Optional[str] = None,
) -> str:
    """Produces an ASCII histogram for a single numeric variable.

    Computes weighted histogram bins across all datasets using a shared
    bin range, then renders grouped horizontal bars for each bin.

    How to read the output:
        Each row is a numeric bin range. Within a row, each dataset gets
        its own bar drawn with a distinct fill character (``█``, ``▓``, etc.).

        - The percentage at the end of each bar is the weighted proportion
          of observations falling in that bin within its dataset (i.e.,
          proportions within each dataset sum to 100%).
        - Bar lengths are scaled so that the longest bar across all
          datasets spans the full ``bar_width``.

    Args:
        dfs: List of DataFrameWithWeight dicts.
        names: Names for each DataFrame (e.g., ["self", "target"]).
        column: The numeric column name to plot.
        weighted: Whether to use weights. Defaults to True.
        n_bins: Number of histogram bins. Defaults to None, which
            auto-detects using Sturges' rule.
        bar_width: Maximum character width for bars. Defaults to None,
            which auto-detects based on terminal width.
        dist_type: Accepted for compatibility but only "hist_ascii" is supported.
            A warning is logged if any other value is passed.

    Returns:
        ASCII histogram text for this variable.

    Example:
        ::

            >>> df_a = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0]})
            >>> df_b = pd.DataFrame({"age": [10.0, 10.0, 10.0, 40.0]})
            >>> dfs = [
            ...     {"df": df_a, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
            ...     {"df": df_b, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
            ... ]
            >>> print(ascii_plot_hist(dfs, names=["self", "target"],
            ...       column="age", n_bins=2, bar_width=20))
            === age (numeric) ===
            <BLANKLINE>
            Bin            | sample  population
                           |
            [10.00, 25.00) | █████████████ (50.0%)
                           | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (75.0%)
            [25.00, 40.00] | █████████████ (50.0%)
                           | ▒▒▒▒▒▒▒ (25.0%)
            <BLANKLINE>
            Legend: █ sample  ▒ population
            Bar lengths are proportional to weighted frequency within each dataset.
            <BLANKLINE>
    """
    if dist_type is not None and dist_type != "hist_ascii":
        logger.warning(
            f"ASCII plots only support dist_type='hist_ascii'. "
            f"Ignoring dist_type='{dist_type}' and using 'hist_ascii'."
        )
    legend_names: List[str] = [naming_legend(n, names) for n in names]

    # Collect all values to determine shared bin range
    all_values: List[pd.Series] = []
    all_weights: List[Optional[pd.Series]] = []
    for d in dfs:
        a_series = d["df"][column]
        _w = d["weight"]
        if weighted and _w is not None:
            a_series, _w = rm_mutual_nas(a_series, _w)
        else:
            a_series = a_series.dropna()
        all_values.append(a_series)
        all_weights.append(_w if weighted else None)

    # Compute shared bin edges
    combined_values = pd.concat(all_values, ignore_index=True)
    if len(combined_values) == 0:
        return f"=== {column} (numeric) ===\n\nNo data available.\n"

    if n_bins is None:
        n_bins = _auto_n_bins(len(combined_values), combined_values.nunique())

    global_min = float(combined_values.min())
    global_max = float(combined_values.max())

    # Handle edge case where all values are the same
    if global_min == global_max:
        global_min = global_min - 0.5
        global_max = global_max + 0.5

    bin_edges = np.linspace(global_min, global_max, n_bins + 1)

    # Compute histograms per dataset
    hist_data: List[npt.NDArray[np.floating]] = []
    for vals, wts in zip(all_values, all_weights):
        hist_data.append(_weighted_histogram(vals, wts, bin_edges))

    # Build bin labels
    bin_labels: List[str] = []
    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        bracket_right = "]" if i == n_bins - 1 else ")"
        bin_labels.append(f"[{left:,.2f}, {right:,.2f}{bracket_right}")

    # Find max proportion for bar scaling
    max_value: float = max(float(h.max()) for h in hist_data) if hist_data else 1.0

    # Compute label width
    label_width = max(len(lbl) for lbl in bin_labels) if bin_labels else 8
    label_width = max(label_width, 3)  # minimum width for "Bin"

    if bar_width is None:
        bar_width = _auto_bar_width(label_width)

    # Build output
    lines: List[str] = []
    lines.append(f"=== {column} (numeric) ===")
    lines.append("")

    # Header
    header_label = "Bin".ljust(label_width)
    lines.append(f"{header_label} | {'  '.join(legend_names)}")
    lines.append(f"{' ' * label_width} |")

    for bi, lbl in enumerate(bin_labels):
        proportions: Dict[str, float] = {}
        for di, name in enumerate(legend_names):
            proportions[name] = float(hist_data[di][bi])
        lines.append(
            _render_horizontal_bars(
                lbl, proportions, legend_names, bar_width, max_value, label_width
            )
        )

    lines.append("")
    lines.append(_build_legend(legend_names))
    lines.append("")

    return "\n".join(lines)


def ascii_comparative_hist(
    dfs: List[DataFrameWithWeight],
    names: List[str],
    column: str,
    weighted: bool = True,
    n_bins: Optional[int] = None,
    bar_width: Optional[int] = None,
) -> str:
    """Produces a columnar, baseline-relative ASCII histogram.

    The first dataset is the baseline. Subsequent datasets show bars split
    into segments that indicate how each bin compares to the baseline.

    How to read the output:
        Each row is a bin range. The first column is the baseline dataset,
        shown with solid ``█`` bars. For every other column:

        - ``█`` (solid fill) = the portion of the bar that matches the
          baseline proportion. This is the "common" part.
        - ``▒`` (medium shade) = the portion that **exceeds** the baseline.
          The bin has more mass than the baseline in this range.
        - ``   ]`` (right bracket) = a **deficit** relative to the baseline.
          The gap before the bracket shows how much mass is missing compared to
          the baseline in this range.
        - A number without any bar means the percentage is too small to
          render at the chosen ``bar_width``.

        All percentages are normalized so each column sums to 100%.

    Args:
        dfs: List of DataFrameWithWeight dicts. The first entry is used as
            the baseline for comparison.
        names: Names for each DataFrame (e.g., ["Target", "Sample"]).
        column: The numeric column name to plot.
        weighted: Whether to use weights. Defaults to True.
        n_bins: Number of histogram bins. Defaults to None, which
            auto-detects using Sturges' rule.
        bar_width: Maximum character width for bars. Defaults to None,
            which auto-detects based on terminal width.

    Returns:
        ASCII comparative histogram text.

    Example:
        ::

            >>> print(ascii_comparative_hist(dfs, names=["Target", "Sample"],
            ...       column="income", n_bins=2, bar_width=20))
            === income (numeric, comparative) ===
            <BLANKLINE>
            Range          | Target (%)         | Sample (%)
            ---------------------------------------------------------------
            [10.00, 25.00) | █████████████ 50.0 | █████████████▒▒▒▒▒▒▒ 75.0
            [25.00, 40.00] | █████████████ 50.0 | ███████     ] 25.0
            ---------------------------------------------------------------
            Total          | 100.0              | 100.0

        In the Sample column above, bin [10, 25) shows ``▒`` excess
        (75% vs 50% baseline) while bin [25, 40] shows ``     ]``
        deficit (25% vs 50% baseline).
    """
    legend_names: List[str] = [naming_legend(n, names) for n in names]

    # Collect all values to determine shared bin range
    all_values: List[pd.Series] = []
    all_weights: List[Optional[pd.Series]] = []
    for d in dfs:
        a_series = d["df"][column]
        _w = d["weight"]
        if weighted and _w is not None:
            a_series, _w = rm_mutual_nas(a_series, _w)
        else:
            a_series = a_series.dropna()
        all_values.append(a_series)
        all_weights.append(_w if weighted else None)

    # Compute shared bin edges
    combined_values = pd.concat(all_values, ignore_index=True)
    if len(combined_values) == 0:
        return "No data available."

    if n_bins is None:
        n_bins = _auto_n_bins(len(combined_values), combined_values.nunique())

    global_min = float(combined_values.min())
    global_max = float(combined_values.max())

    if global_min == global_max:
        global_min = global_min - 0.5
        global_max = global_max + 0.5

    bin_edges = np.linspace(global_min, global_max, n_bins + 1)

    # Compute histograms per dataset (as percentages)
    hist_pcts: List[List[float]] = []
    for vals, wts in zip(all_values, all_weights):
        props = _weighted_histogram(vals, wts, bin_edges)
        hist_pcts.append([float(p) * 100.0 for p in props])

    # Find max percentage across all datasets and bins for bar scaling
    max_pct: float = max(
        (pct for pcts in hist_pcts for pct in pcts),
        default=0.0,
    )

    # Build bin labels
    bin_labels: List[str] = []
    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        bracket_right = "]" if i == n_bins - 1 else ")"
        bin_labels.append(f"[{left:,.2f}, {right:,.2f}{bracket_right}")

    # Range column width (computed early so bar_width auto-detection can use it)
    range_header = "Range"
    range_width = max(len(range_header), max(len(lbl) for lbl in bin_labels))

    if bar_width is None:
        bar_width = _auto_bar_width_columnar(range_width, len(legend_names))

    # Baseline percentages (first dataset)
    baseline_pcts = hist_pcts[0]

    # Build cell strings for each dataset x bin
    # cell_strings[dataset_idx][bin_idx] = "bar_chars pct"
    cell_strings: List[List[str]] = []
    for di in range(len(hist_pcts)):
        cells: List[str] = []
        for bi in range(n_bins):
            pct = hist_pcts[di][bi]
            if max_pct > 0:
                bar_len = round((pct / max_pct) * bar_width)
            else:
                bar_len = 0

            if di == 0:
                # Baseline: simple filled bars
                bar = "█" * bar_len
            else:
                base_pct = baseline_pcts[bi]
                if max_pct > 0:
                    baseline_len = round((base_pct / max_pct) * bar_width)
                else:
                    baseline_len = 0

                if bar_len >= baseline_len:
                    bar = "█" * baseline_len + "▒" * (bar_len - baseline_len)
                else:
                    deficit = baseline_len - bar_len
                    if deficit >= 2:
                        bar = "█" * bar_len + " " * (deficit - 1) + "]"
                    else:
                        bar = "█" * bar_len + "]"

            pct_str = f"{pct:.1f}"
            if bar:
                cells.append(f"{bar} {pct_str}")
            else:
                cells.append(pct_str)
        cell_strings.append(cells)

    # Compute column widths
    col_widths: List[int] = []
    for di in range(len(legend_names)):
        header_w = len(f"{legend_names[di]} (%)")
        max_cell_w = max(len(cell_strings[di][bi]) for bi in range(n_bins))
        col_widths.append(max(header_w, max_cell_w))

    # Build output
    lines: List[str] = []
    lines.append(f"=== {column} (numeric, comparative) ===")
    lines.append("")

    # Header row
    header_parts = [range_header.ljust(range_width)]
    for di in range(len(legend_names)):
        header_parts.append(f"{legend_names[di]} (%)".ljust(col_widths[di]))
    lines.append(" | ".join(header_parts))

    # Separator
    sep_width = range_width + sum(col_widths) + 3 * len(col_widths)
    lines.append("-" * sep_width)

    # Data rows
    for bi in range(n_bins):
        row_parts = [bin_labels[bi].ljust(range_width)]
        for di in range(len(legend_names)):
            row_parts.append(cell_strings[di][bi].ljust(col_widths[di]))
        lines.append(" | ".join(row_parts))

    # Separator
    lines.append("-" * sep_width)

    # Total row
    total_parts = ["Total".ljust(range_width)]
    for di in range(len(hist_pcts)):
        total_val = sum(hist_pcts[di])
        total_parts.append(f"{total_val:.1f}".ljust(col_widths[di]))
    lines.append(" | ".join(total_parts))

    # Legend (only when there are non-baseline columns)
    if len(legend_names) > 1:
        lines.append("")
        lines.append(
            f"Key: █ = shared with {legend_names[0]}," " ▒ = excess,    ] = deficit"
        )

    return "\n".join(lines)


def ascii_plot_dist(
    dfs: List[DataFrameWithWeight],
    names: Optional[List[str]] = None,
    variables: Optional[List[str]] = None,
    numeric_n_values_threshold: int = 15,
    weighted: bool = True,
    n_bins: Optional[int] = None,
    bar_width: Optional[int] = None,
    dist_type: Optional[str] = None,
    separate_categories: bool = True,
) -> str:
    """Produces ASCII text comparing weighted distributions across datasets.

    Iterates over variables, classifying each as categorical or numeric
    (using the same logic as :func:`seaborn_plot_dist`), then delegates to
    :func:`ascii_plot_bar` or :func:`ascii_comparative_hist` respectively.

    The output is both printed to stdout and returned as a string.

    Args:
        dfs: List of DataFrameWithWeight dicts.
        names: Names for each DataFrame (e.g., ["self", "unadjusted", "target"]).
            If None, defaults to "df_0", "df_1", etc.
        variables: Subset of variables to plot. None means all.
        numeric_n_values_threshold: Columns with fewer unique values than this
            are treated as categorical. Defaults to 15.
        weighted: Whether to use weights. Defaults to True.
        n_bins: Number of bins for numeric histograms. Defaults to None,
            which auto-detects using Sturges' rule.
        bar_width: Maximum character width for the longest bar. Defaults to
            None, which auto-detects based on terminal width.
        dist_type: Accepted for compatibility but only "hist_ascii" is supported.
            A warning is logged if any other value is passed.
        separate_categories: If True, insert a blank line between categories
            in barplots for readability. Defaults to True.

    Returns:
        The full ASCII output text.

    Examples:
        ::

            >>> import pandas as pd
            >>> from balance.stats_and_plots.ascii_plots import ascii_plot_dist
            >>> df_a = pd.DataFrame({
            ...     "color": ["red", "blue", "blue", "green"],
            ...     "age": [10.0, 20.0, 30.0, 40.0],
            ... })
            >>> df_b = pd.DataFrame({
            ...     "color": ["red", "red", "blue", "green"],
            ...     "age": [10.0, 10.0, 10.0, 40.0],
            ... })
            >>> dfs = [
            ...     {"df": df_a, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
            ...     {"df": df_b, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
            ... ]
            >>> print(ascii_plot_dist(dfs, names=["self", "target"],
            ...       numeric_n_values_threshold=0, n_bins=2, bar_width=20))
            === color (categorical) ===
            <BLANKLINE>
            Category | sample  population
                     |
            blue     | ████████████████████ (50.0%)
                     | ▒▒▒▒▒▒▒▒▒▒ (25.0%)
            <BLANKLINE>
            green    | ██████████ (25.0%)
                     | ▒▒▒▒▒▒▒▒▒▒ (25.0%)
            <BLANKLINE>
            red      | ██████████ (25.0%)
                     | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (50.0%)
            <BLANKLINE>
            Legend: █ sample  ▒ population
            Bar lengths are proportional to weighted frequency within each dataset.
            <BLANKLINE>
            === age (numeric, comparative) ===
            <BLANKLINE>
            Range          | sample (%)         | population (%)
            ---------------------------------------------------------------
            [10.00, 25.00) | █████████████ 50.0 | █████████████▒▒▒▒▒▒▒ 75.0
            [25.00, 40.00] | █████████████ 50.0 | ███████     ] 25.0
            ---------------------------------------------------------------
            Total          | 100.0              | 100.0
            <BLANKLINE>
            Key: █ = shared with sample, ▒ = excess,    ] = deficit
    """
    if dist_type is not None and dist_type != "hist_ascii":
        logger.warning(
            f"ASCII plots only support dist_type='hist_ascii'. "
            f"Ignoring dist_type='{dist_type}' and using 'hist_ascii'."
        )
    if names is None:
        names = [f"df_{i}" for i in range(len(dfs))]

    variables = choose_variables(*(d["df"] for d in dfs), variables=variables)
    logger.debug(f"ASCII plotting variables {variables}")

    numeric_variables = dfs[0]["df"].select_dtypes(exclude=["object"]).columns.values

    output_parts: List[str] = []

    for o in variables:
        # Find the maximum number of non-missing unique values across all dfs
        n_values = max(len(set(rm_mutual_nas(d["df"].loc[:, o].values))) for d in dfs)
        if n_values == 0:
            logger.warning(f"No nonmissing values for variable '{o}', skipping")
            continue

        categorical = (o not in numeric_variables) or (
            n_values < numeric_n_values_threshold
        )

        if categorical:
            output_parts.append(
                ascii_plot_bar(
                    dfs,
                    names,
                    o,
                    weighted=weighted,
                    bar_width=bar_width,
                    separate_categories=separate_categories,
                )
            )
        else:
            output_parts.append(
                ascii_comparative_hist(
                    dfs,
                    names,
                    o,
                    weighted=weighted,
                    n_bins=n_bins,
                    bar_width=bar_width,
                )
            )

    result = "\n".join(output_parts)
    print(result)
    return result
