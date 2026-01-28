# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import numpy as np
import pandas as pd
from balance.stats_and_plots.weights_stats import _check_weights_are_valid
from balance.utils.input_validation import _coerce_to_numeric_and_validate
from scipy import stats

if TYPE_CHECKING:
    from balance.sample_class import Sample


def _prepare_outcome_and_weights(
    y: Iterable[float] | pd.Series | np.ndarray,
    w0: Iterable[float] | pd.Series | np.ndarray,
    w1: Iterable[float] | pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_series = pd.Series(y)
    w0_series = pd.Series(w0)
    w1_series = pd.Series(w1)

    if (
        y_series.shape[0] != w0_series.shape[0]
        or y_series.shape[0] != w1_series.shape[0]
    ):
        raise ValueError(
            "Outcome and weights must have the same number of observations."
        )

    _check_weights_are_valid(w0_series)
    _check_weights_are_valid(w1_series)

    not_null_mask = y_series.notna() & w0_series.notna() & w1_series.notna()
    y_series = y_series[not_null_mask]
    w0_series = w0_series[not_null_mask]
    w1_series = w1_series[not_null_mask]

    y_numeric, w0_values = _coerce_to_numeric_and_validate(
        y_series, w0_series.to_numpy(), "outcome"
    )
    _, w1_values = _coerce_to_numeric_and_validate(
        y_series, w1_series.to_numpy(), "outcome"
    )

    finite_mask = (
        np.isfinite(w0_values) & np.isfinite(w1_values) & np.isfinite(y_numeric)
    )
    if not np.any(finite_mask):
        raise ValueError("Outcome and weights must contain at least one finite value.")

    return y_numeric[finite_mask], w0_values[finite_mask], w1_values[finite_mask]


def weights_impact_on_outcome_ss(
    y: Iterable[float] | pd.Series | np.ndarray,
    w0: Iterable[float] | pd.Series | np.ndarray,
    w1: Iterable[float] | pd.Series | np.ndarray,
    method: str = "t_test",
    conf_level: float = 0.95,
) -> pd.Series:
    """
    Evaluate whether weighting changes the outcome by testing y*w0 vs y*w1.

    Args:
        y: Outcome values.
        w0: Baseline weights.
        w1: Alternative weights.
        method: Statistical test to use ("t_test").
        conf_level: Confidence level for the mean difference interval.

    Returns:
        pd.Series: Summary statistics for the weighted outcome comparison.

    Examples:
    .. code-block:: python

            import pandas as pd

            from balance.stats_and_plots.impact_of_weights_on_outcome import (
                weights_impact_on_outcome_ss,
            )

            result = weights_impact_on_outcome_ss(
                y=pd.Series([1.0, 2.0, 3.0, 4.0]),
                w0=pd.Series([1.0, 1.0, 1.0, 1.0]),
                w1=pd.Series([1.0, 2.0, 1.0, 2.0]),
                method="t_test",
            )
            print(result.round(3).to_string())

    .. code-block:: text

        mean_yw0         2.500
        mean_yw1         4.000
        mean_diff        1.500
        diff_ci_lower   -1.547
        diff_ci_upper    4.547
        t_stat           1.567
        p_value          0.215
        n                4.000
    """
    if method != "t_test":
        raise ValueError(f"Unsupported method: {method}")
    if conf_level <= 0 or conf_level >= 1:
        raise ValueError("conf_level must be between 0 and 1.")

    y_values, w0_values, w1_values = _prepare_outcome_and_weights(y, w0, w1)

    yw0 = y_values * w0_values
    yw1 = y_values * w1_values
    diff = yw1 - yw0
    n_obs = int(diff.shape[0])
    diff_std = float(np.std(diff, ddof=1)) if n_obs > 1 else 0.0

    mean_yw0 = float(np.mean(yw0))
    mean_yw1 = float(np.mean(yw1))
    mean_diff = float(np.mean(diff))

    if n_obs < 2:
        t_stat, p_value = np.nan, np.nan
        ci_lower, ci_upper = np.nan, np.nan
    elif np.isclose(diff_std, 0.0):
        t_stat, p_value = np.nan, np.nan
        ci_lower, ci_upper = mean_diff, mean_diff
    else:
        t_stat, p_value = stats.ttest_rel(yw1, yw0, nan_policy="omit")
        t_crit = stats.t.ppf((1 + conf_level) / 2, df=n_obs - 1)
        margin = t_crit * diff_std / np.sqrt(n_obs)
        ci_lower, ci_upper = mean_diff - margin, mean_diff + margin

    return pd.Series(
        {
            "mean_yw0": mean_yw0,
            "mean_yw1": mean_yw1,
            "mean_diff": mean_diff,
            "diff_ci_lower": ci_lower,
            "diff_ci_upper": ci_upper,
            "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
            "n": n_obs,
        }
    )


def compare_adjusted_weighted_outcome_ss(
    adjusted0: "Sample",
    adjusted1: "Sample",
    method: str = "t_test",
    conf_level: float = 0.95,
    round_ndigits: int | None = 3,
) -> pd.DataFrame:
    """
    Compare two adjusted Samples by testing outcomes under each set of weights.

    Args:
        adjusted0: First adjusted Sample (w0).
        adjusted1: Second adjusted Sample (w1).
        method: Statistical test to use ("t_test").
        conf_level: Confidence level for the mean difference interval.
        round_ndigits: Optional rounding for numeric outputs.

    Returns:
        pd.DataFrame: Outcome-by-statistic table comparing weighted outcomes.

    Examples:
    .. code-block:: python

            import pandas as pd

            from balance.sample_class import Sample
            from balance.stats_and_plots.impact_of_weights_on_outcome import (
                compare_adjusted_weighted_outcome_ss,
            )

            sample = Sample.from_frame(
                pd.DataFrame(
                    {
                        "id": [1, 2, 3],
                        "x": [0.1, 0.2, 0.3],
                        "weight": [1.0, 1.0, 1.0],
                        "outcome": [1.0, 2.0, 3.0],
                    }
                ),
                id_column="id",
                weight_column="weight",
                outcome_columns=("outcome",),
            )
            target = Sample.from_frame(
                pd.DataFrame(
                    {
                        "id": [4, 5, 6],
                        "x": [0.1, 0.2, 0.3],
                        "weight": [1.0, 1.0, 1.0],
                        "outcome": [1.0, 2.0, 3.0],
                    }
                ),
                id_column="id",
                weight_column="weight",
                outcome_columns=("outcome",),
            )
            adjusted_a = sample.set_target(target).adjust(method="null")
            adjusted_b = sample.set_target(target).adjust(method="null")
            adjusted_b.set_weights(pd.Series([1.0, 2.0, 3.0], index=adjusted_b.df.index))

            impact = compare_adjusted_weighted_outcome_ss(
                adjusted_a, adjusted_b, round_ndigits=3
            )
            print(impact.to_string())

    .. code-block:: text

            mean_yw0  mean_yw1  mean_diff  diff_ci_lower  diff_ci_upper  t_stat  p_value    n
    outcome
    outcome       2.0     4.667      2.667         -4.922         10.256   1.512     0.27  3.0
    """
    from balance.sample_class import Sample

    if not isinstance(adjusted0, Sample) or not isinstance(adjusted1, Sample):
        raise ValueError("compare_adjusted_weighted_outcome_ss expects Sample inputs.")

    adjusted0._check_if_adjusted()
    adjusted1._check_if_adjusted()

    outcomes0 = adjusted0.outcomes()
    outcomes1 = adjusted1.outcomes()
    if outcomes0 is None or outcomes1 is None:
        raise ValueError("Both Samples must include outcomes.")

    y0 = outcomes0.model_matrix()
    y1 = outcomes1.model_matrix()
    if list(y0.columns) != list(y1.columns):
        raise ValueError("Outcome columns must match between adjusted Samples.")

    ids0 = adjusted0.id_column.to_numpy()
    ids1 = adjusted1.id_column.to_numpy()
    if pd.Index(ids0).has_duplicates or pd.Index(ids1).has_duplicates:
        raise ValueError("Samples must have unique ids to compare outcomes.")

    y0 = y0.set_index(adjusted0.id_column)
    y1 = y1.set_index(adjusted1.id_column)

    common_ids = y0.index.intersection(y1.index)
    if common_ids.empty:
        raise ValueError("Samples do not share any common ids.")

    y0 = y0.loc[common_ids]
    y1 = y1.loc[common_ids]
    if not y0.equals(y1):
        raise ValueError(
            "Outcome values differ between adjusted Samples for common ids."
        )

    weights0 = adjusted0.weight_column.to_numpy()
    weights0_series = pd.Series(weights0, index=ids0)
    weights0 = weights0_series.reindex(common_ids).to_numpy()

    weights1 = pd.Series(
        adjusted1.weight_column.to_numpy(),
        index=ids1,
    ).reindex(common_ids)

    mask = weights1.notna().to_numpy()
    if not np.any(mask):
        raise ValueError(
            "Samples do not share any common ids with non-missing weights in adjusted1."
        )

    y0 = y0.loc[mask]
    weights0 = weights0[mask]
    weights1 = weights1.to_numpy()[mask]

    results = {}
    for column in y0.columns:
        results[column] = weights_impact_on_outcome_ss(
            y0[column].to_numpy(),
            w0=weights0,
            w1=weights1,
            method=method,
            conf_level=conf_level,
        )

    impact_df = pd.DataFrame(results).T
    impact_df.index.name = "outcome"
    if round_ndigits is not None:
        numeric_cols = impact_df.select_dtypes(include=["number"]).columns
        impact_df[numeric_cols] = impact_df[numeric_cols].round(round_ndigits)
    return impact_df
