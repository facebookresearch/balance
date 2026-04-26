# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Shared summary and diagnostics utilities for the Balance library.

Provides ``_build_summary()`` and ``_build_diagnostics()``, which are shared by
both the legacy ``Sample`` class and the new ``BalanceFrame`` workflow.

These functions accept plain DataFrames / Series instead of ``Sample`` objects,
so callers can produce the same output without tight coupling to the ``Sample``
class hierarchy.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from balance.typing import DiagnosticScalar
from balance.util import _coerce_scalar

logger: logging.Logger = logging.getLogger(__package__)


def _resolve_adjustment_failure_metadata(
    model_dict: Optional[dict[str, Any]],
) -> Tuple[int, Optional[str]]:
    """Normalize adjustment failure metadata from ``model_dict``.

    Args:
        model_dict: Adjustment model metadata dictionary.

    Returns:
        Tuple of ``(adjustment_failure, adjustment_failure_reason)`` where:
            * ``adjustment_failure`` is always ``0`` or ``1``.
            * ``adjustment_failure_reason`` is a stripped non-empty string only
              when a failure is recorded, otherwise ``None``.
    """

    failure: int = 0
    reason: Optional[str] = None
    if model_dict is None:
        return failure, reason

    raw_failure = model_dict.get("adjustment_failure", 0)

    if isinstance(raw_failure, bool):
        failure = int(raw_failure)
    elif isinstance(raw_failure, (int, np.integer)):
        failure = int(raw_failure != 0)
    elif isinstance(raw_failure, (float, np.floating)):
        if np.isnan(raw_failure):
            failure = 0
        else:
            failure = int(raw_failure != 0.0)
    elif isinstance(raw_failure, str):
        normalized = raw_failure.strip().lower()
        if normalized in {"", "0", "false", "no", "n", "success", "succeeded"}:
            failure = 0
        elif normalized in {"1", "true", "yes", "y", "failure", "failed"}:
            failure = 1
        else:
            failure = int(bool(normalized))
    else:
        failure = int(bool(raw_failure))

    raw_reason = model_dict.get("adjustment_failure_reason")
    if failure == 1 and isinstance(raw_reason, str):
        stripped_reason = raw_reason.strip()
        if stripped_reason:
            reason = stripped_reason

    return failure, reason


def _concat_metric_val_var(
    diagnostics: pd.DataFrame,
    metric: str,
    vals: list[DiagnosticScalar],
    vars: list[DiagnosticScalar],
) -> pd.DataFrame:
    """Append metric/val/var rows to a diagnostics table.

    This helper centralizes concatenating new diagnostics rows from separate
    val and var lists. The function internally zips these lists together,
    so callers only need to provide aligned sequences.

    Args:
        diagnostics: Existing diagnostics table.
        metric: Name for the ``metric`` column to repeat for each appended row.
        vals: List of values for the ``val`` column. Must have the same length
            as ``vars``.
        vars: List of variable names for the ``var`` column. Must have the same
            length as ``vals``.

    Returns:
        A new DataFrame with the appended rows (input is not modified).

    Raises:
        ValueError: If ``vals`` and ``vars`` have different lengths.

    Examples:
        A typical usage pattern when both ``val`` and ``var`` are sequences::

            >>> import pandas as pd
            >>> from balance.summary_utils import _concat_metric_val_var
            >>> diag = pd.DataFrame(columns=["metric", "val", "var"])
            >>> result = _concat_metric_val_var(
            ...     diag, "size", [100, 5], ["sample_obs", "sample_covars"]
            ... )
            >>> result["var"].tolist()
            ['sample_obs', 'sample_covars']

        With a single row::

            >>> result = _concat_metric_val_var(
            ...     pd.DataFrame(columns=["metric", "val", "var"]),
            ...     "adjustment_method", [0], ["ipw"],
            ... )
            >>> result["var"].tolist()
            ['ipw']
    """
    if len(vals) != len(vars):
        raise ValueError(
            f"vals and vars must have the same length, got {len(vals)} and {len(vars)}"
        )

    if len(vals) == 0:
        return diagnostics.copy()

    rows = pd.DataFrame(
        {"metric": [metric] * len(vals), "val": list(vals), "var": list(vars)}
    )

    # Append rows to diagnostics with column alignment
    if diagnostics.empty:
        if diagnostics.columns.empty:
            return rows.reset_index(drop=True)
        # pyrefly: ignore [bad-argument-type]
        rows = rows.reindex(columns=diagnostics.columns, fill_value=pd.NA)
        return rows.reset_index(drop=True)

    # pyrefly: ignore [bad-argument-type]
    rows = rows.reindex(columns=diagnostics.columns, fill_value=pd.NA)
    return pd.concat((diagnostics, rows), ignore_index=True)


def _build_summary(
    *,
    is_adjusted: bool,
    has_target: bool,
    covars_asmd: pd.DataFrame | None,
    covars_kld: pd.DataFrame | None,
    asmd_improvement_pct: float | None,
    quick_adjustment_details: list[str],
    design_effect: float | None,
    effective_sample_size: float | None,
    effective_sample_proportion: float | None,
    model_dict: dict[str, Any] | None,
    outcome_means: pd.DataFrame | None,
) -> str:
    """Build a human-readable summary string from pre-computed diagnostics.

    This is the logic previously embedded in ``Sample.summary()``, extracted so
    it can be reused by other APIs (e.g. ``BalanceFrame``) without depending on
    a ``Sample`` instance.

    All parameters are pre-computed values that the caller must supply. The
    function does **not** access any ``Sample`` object.

    Args:
        is_adjusted: Whether the data has been adjusted.
        has_target: Whether a target population is set.
        covars_asmd: Result of ``covars().asmd()`` or ``None``.
        covars_kld: Result of ``covars().kld(aggregate_by_main_covar=True)``
            or ``None``.
        asmd_improvement_pct: ``100 * covars().asmd_improvement()`` or ``None``.
        quick_adjustment_details: Lines from ``_quick_adjustment_details()``.
        design_effect: Design effect value or ``None``.
        effective_sample_size: Effective sample size or ``None``.
        effective_sample_proportion: Effective sample proportion or ``None``.
        model_dict: The adjustment model dictionary or ``None``.
        outcome_means: Result of ``outcomes().mean()`` or ``None``.

    Returns:
        A multi-line summary string.

    Examples:
        >>> from balance.summary_utils import _build_summary
        >>> _build_summary(
        ...     is_adjusted=False,
        ...     has_target=False,
        ...     covars_asmd=None,
        ...     covars_kld=None,
        ...     asmd_improvement_pct=None,
        ...     quick_adjustment_details=[],
        ...     design_effect=None,
        ...     effective_sample_size=None,
        ...     effective_sample_proportion=None,
        ...     model_dict=None,
        ...     outcome_means=None,
        ... )
        ''
    """
    # Initialize variables
    n_asmd_covars: int = 0
    asmd_before: float = 0.0
    asmd_now: float = 0.0
    n_kld_covars: int = 0
    kld_before: float = 0.0
    kld_now: float = 0.0
    kld_reduction: float = 0.0

    if (is_adjusted or has_target) and covars_asmd is not None:
        n_asmd_covars = len(
            # pyrefly: ignore [bad-index]
            covars_asmd.columns.values[covars_asmd.columns.values != "mean(asmd)"]
        )

    if (is_adjusted or has_target) and covars_kld is not None:
        n_kld_covars = len(
            # pyrefly: ignore [bad-index]
            covars_kld.columns.values[covars_kld.columns.values != "mean(kld)"]
        )

    if is_adjusted and covars_asmd is not None:
        asmd_before = covars_asmd.loc["unadjusted", "mean(asmd)"]

    if is_adjusted and covars_kld is not None:
        kld_before = covars_kld.loc["unadjusted", "mean(kld)"]

    if has_target and covars_asmd is not None:
        asmd_now = covars_asmd.loc["self", "mean(asmd)"]

    if has_target and covars_kld is not None:
        kld_now = covars_kld.loc["self", "mean(kld)"]
        if is_adjusted and kld_before > 0:
            kld_reduction = 100 * (kld_before - kld_now) / kld_before

    # model performance
    model_summary = None
    if model_dict is not None:
        if model_dict["method"] == "ipw":
            model_summary = "Model proportion deviance explained: {dev_exp:.3f}".format(
                dev_exp=model_dict["perf"]["prop_dev_explained"]
            )

    sections: list[str] = []

    adjustment_lines = [
        d
        for d in quick_adjustment_details
        if not d.startswith(
            (
                "design effect",
                "effective sample size proportion",
                "effective sample size (ESS)",
            )
        )
    ]
    if adjustment_lines:
        sections.append("Adjustment details:\n    " + "\n    ".join(adjustment_lines))

    covar_lines: list[str] = []
    if has_target:
        if is_adjusted and asmd_improvement_pct is not None:
            covar_lines.append(f"Covar ASMD reduction: {asmd_improvement_pct:.1f}%")
        covar_lines.append(
            f"Covar ASMD ({n_asmd_covars} variables): "
            + (f"{asmd_before:.3f} -> " if is_adjusted else "")
            + f"{asmd_now:.3f}"
        )

        if is_adjusted and kld_before > 0:
            covar_lines.append(f"Covar mean KLD reduction: {kld_reduction:.1f}%")
        covar_lines.append(
            f"Covar mean KLD ({n_kld_covars} variables): "
            + (f"{kld_before:.3f} -> " if is_adjusted else "")
            + f"{kld_now:.3f}"
        )

    if covar_lines:
        sections.append("Covariate diagnostics:\n    " + "\n    ".join(covar_lines))

    if is_adjusted:
        weights_lines: list[str] = []
        if design_effect is not None:
            weights_lines.append(f"design effect (Deff): {design_effect:.3f}")
            if effective_sample_proportion is not None:
                weights_lines.append(
                    f"effective sample size proportion (ESSP): {effective_sample_proportion:.3f}"
                )
            if effective_sample_size is not None:
                weights_lines.append(
                    f"effective sample size (ESS): {effective_sample_size:.1f}"
                )
        else:
            weights_lines.append("design effect (Deff): unavailable")

        sections.append("Weight diagnostics:\n    " + "\n    ".join(weights_lines))

    if outcome_means is not None:
        sections.append(
            "Outcome weighted means:\n"
            + outcome_means.to_string(float_format="{:.3f}".format)
        )

    if model_summary is not None:
        sections.append(f"Model performance: {model_summary}")

    return "\n".join(filter(None, sections))


def _build_diagnostics(
    *,
    covars_df: pd.DataFrame,
    target_covars_df: pd.DataFrame,
    weights_summary: pd.DataFrame,
    model_dict: dict[str, Any] | None,
    covars_asmd: pd.DataFrame,
    covars_asmd_main: pd.DataFrame,
    outcome_columns: pd.DataFrame | None = None,
    weights_impact_on_outcome_method: str | None = "t_test",
    weights_impact_on_outcome_conf_level: float = 0.95,
    outcome_impact: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a diagnostics DataFrame from pre-computed data.

    This is the logic previously embedded in ``Sample.diagnostics()``, extracted
    so it can be reused by other APIs without depending on a ``Sample`` instance.

    Args:
        covars_df: The covariates DataFrame for the sample (``covars().df``).
        target_covars_df: The covariates DataFrame for the target
            (``_links["target"].covars().df``).
        weights_summary: Result of ``weights().summary()``.
        model_dict: The adjustment model dictionary, or ``None``. If ``None``,
            the model diagnostics section is skipped (this can happen for
            methods like rake/poststratify that don't produce a model dict).
        covars_asmd: Result of ``covars().asmd()``.
        covars_asmd_main: Result of
            ``covars().asmd(aggregate_by_main_covar=True)``.
        outcome_columns: The outcome columns DataFrame, or ``None`` if no
            outcomes are set.
        weights_impact_on_outcome_method: Method for outcome-weight impact
            (passed through; only used to decide whether to include impact
            rows).
        weights_impact_on_outcome_conf_level: Confidence level for outcome
            impact.
        outcome_impact: Pre-computed outcome impact DataFrame, or ``None``.

    Returns:
        A diagnostics DataFrame with columns ``["metric", "val", "var"]``.

    Examples:
        >>> import pandas as pd
        >>> from balance.summary_utils import _build_diagnostics
        >>> # Typical usage: call after adjusting a Sample, passing
        >>> # pre-computed covars, weights, and model data.
        >>> # See Sample.diagnostics() for a complete example.
    """
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

    # ----------------------------------------------------
    # Properties of the Sample object (dimensions of the data)
    # ----------------------------------------------------
    n_sample_obs, n_sample_covars = covars_df.shape
    n_target_obs, n_target_covars = target_covars_df.shape

    diagnostics = _concat_metric_val_var(
        diagnostics,
        "size",
        [n_sample_obs, n_sample_covars, n_target_obs, n_target_covars],
        ["sample_obs", "sample_covars", "target_obs", "target_covars"],
    )

    # ----------------------------------------------------
    # Diagnostics on the weights
    # ----------------------------------------------------
    diagnostics = _concat_metric_val_var(
        diagnostics,
        "weights_diagnostics",
        list(weights_summary["val"]),
        list(weights_summary["var"]),
    )

    # ----------------------------------------------------
    # Diagnostics on the impact of weights on outcomes
    # ----------------------------------------------------
    if (
        weights_impact_on_outcome_method is not None
        and outcome_columns is not None
        and outcome_impact is not None
    ):
        impact_rows = outcome_impact.reset_index().melt(
            id_vars="outcome", var_name="stat", value_name="val"
        )
        for stat_name in impact_rows["stat"].unique():
            stat_rows = impact_rows[impact_rows["stat"] == stat_name]
            diagnostics = _concat_metric_val_var(
                diagnostics,
                f"weights_impact_on_outcome_{stat_name}",
                stat_rows["val"].tolist(),
                stat_rows["outcome"].tolist(),
            )

    # ----------------------------------------------------
    # Diagnostics on the model
    # ----------------------------------------------------
    if model_dict is None:
        # Some adjustment methods (e.g. null, rake, poststratify) don't
        # produce a model dict.  Skip the model diagnostics section.
        diagnostics = _concat_metric_val_var(
            diagnostics,
            "adjustment_method",
            [0],
            ["unknown"],
        )
    else:
        model = model_dict
        diagnostics = _concat_metric_val_var(
            diagnostics,
            "adjustment_method",
            [0],
            [model["method"]],
        )
        if model["method"] == "ipw":
            fit = model["fit"]
            params = fit.get_params(deep=False)

            fit_list: list[pd.DataFrame] = []

            for array_key in ("n_iter_", "intercept_"):
                array_val = getattr(fit, array_key, None)
                if array_val is None:
                    continue

                array_as_np = np.asarray(array_val)
                if array_as_np.size == 1:
                    fit_list.append(
                        _concat_metric_val_var(
                            pd.DataFrame(),
                            "ipw_model_glance",
                            [array_as_np.item()],
                            [array_key],
                        )
                    )

            for param_key, metric_name in (
                ("penalty", "ipw_penalty"),
                ("solver", "ipw_solver"),
            ):
                param_val = params.get(param_key, getattr(fit, param_key, None))
                if isinstance(param_val, str):
                    fit_list.append(
                        _concat_metric_val_var(
                            pd.DataFrame(), metric_name, [0], [param_val]
                        )
                    )

            for scalar_key in ("tol", "l1_ratio"):
                scalar_value = _coerce_scalar(
                    params.get(scalar_key, getattr(fit, scalar_key, None))
                )
                fit_list.append(
                    _concat_metric_val_var(
                        pd.DataFrame(), "model_glance", [scalar_value], [scalar_key]
                    )
                )

            multi_class = params.get("multi_class", getattr(fit, "multi_class", None))
            if multi_class is None:
                multi_class = "auto"
            elif not isinstance(multi_class, str):
                multi_class = str(multi_class)

            fit_list.append(
                _concat_metric_val_var(
                    pd.DataFrame(), "ipw_multi_class", [0], [multi_class]
                )
            )

            if len(fit_list) > 0:
                fit_single_values = pd.concat(fit_list, ignore_index=True)
                fit_single_values = fit_single_values.drop_duplicates(
                    subset=["metric", "var"], keep="first"
                )
                diagnostics = pd.concat((diagnostics, fit_single_values))

            #  Extract info about the regularisation parameter
            lambda_value = _coerce_scalar(model["lambda"])
            diagnostics = _concat_metric_val_var(
                diagnostics, "model_glance", [lambda_value], ["lambda"]
            )

            #  Scalar values from 'perf' key of dictionary
            perf_entries: list[pd.DataFrame] = []
            for k, v in model["perf"].items():
                if np.isscalar(v) and k != "coefs":
                    perf_entries.append(
                        _concat_metric_val_var(
                            pd.DataFrame(),
                            "model_glance",
                            [_coerce_scalar(v)],
                            [k],
                        )
                    )

            if perf_entries:
                diagnostics = pd.concat([diagnostics] + perf_entries, ignore_index=True)

            # Model coefficients
            coefs = (
                model["perf"]["coefs"]
                .reset_index()
                .rename({0: "val", "index": "var"}, axis=1)
                .assign(metric="model_coef")
            )
            diagnostics = pd.concat((diagnostics, coefs))

        elif model["method"] == "cbps":
            beta_opt = pd.DataFrame(
                {"val": model["beta_optimal"], "var": model["X_matrix_columns"]}
            ).assign(metric="beta_optimal")
            diagnostics = pd.concat((diagnostics, beta_opt))

            metric = [
                "rescale_initial_result",
                "balance_optimize_result",
                "gmm_optimize_result_glm_init",
                "gmm_optimize_result_bal_init",
            ]
            metric = [x for x in metric for _ in range(2)]
            var = ["success", "message"] * 4
            val = [model[x][y] for (x, y) in zip(metric, var)]

            optimizations = pd.DataFrame({"metric": metric, "var": var, "val": val})
            diagnostics = pd.concat((diagnostics, optimizations))

        # TODO: add model diagnostics for other models

    # ----------------------------------------------------
    # Diagnostics on the covariates correction
    # ----------------------------------------------------
    #  Per-covariate ASMDs
    covar_asmds = (
        covars_asmd.transpose()
        .rename(
            {
                "self": "covar_asmd_adjusted",
                "unadjusted": "covar_asmd_unadjusted",
                "unadjusted - self": "covar_asmd_improvement",
            },
            axis=1,
        )
        .reset_index()
        .melt(id_vars="index")
        .rename({"source": "metric", "value": "val", "index": "var"}, axis=1)
    )
    diagnostics = pd.concat((diagnostics, covar_asmds))

    #  Per-main-covariate ASMDs
    covar_asmds_main = (
        covars_asmd_main.transpose()
        .rename(
            {
                "self": "covar_main_asmd_adjusted",
                "unadjusted": "covar_main_asmd_unadjusted",
                "unadjusted - self": "covar_main_asmd_improvement",
            },
            axis=1,
        )
        .reset_index()
        # TODO:
        # column index name is different here.
        # think again if that's the best default or not for
        # asmd(aggregate_by_main_covar = True)
        .rename({"main_covar_names": "index"}, axis=1)
        .melt(id_vars="index")
        .rename({"source": "metric", "value": "val", "index": "var"}, axis=1)
    )
    # sort covar_asmds_main to have mean(asmd) at the end of it (for when doing quick checks)
    covar_asmds_main = (
        covar_asmds_main.assign(has_mean_asmd=(covar_asmds_main["var"] == "mean(asmd)"))
        .sort_values(by=["has_mean_asmd", "var"])
        .drop(columns="has_mean_asmd")
    )
    diagnostics = pd.concat((diagnostics, covar_asmds_main))

    # ----------------------------------------------------
    # Diagnostics if there was an adjustment_failure
    # ----------------------------------------------------
    (
        resolved_adjustment_failure,
        resolved_adjustment_failure_reason,
    ) = _resolve_adjustment_failure_metadata(model_dict)

    diagnostics = _concat_metric_val_var(
        diagnostics,
        "adjustment_failure",
        [resolved_adjustment_failure],
        [None],
    )

    if resolved_adjustment_failure_reason is not None:
        diagnostics = _concat_metric_val_var(
            diagnostics,
            "adjustment_failure_reason",
            [resolved_adjustment_failure_reason],
            [None],
        )

    diagnostics = diagnostics.reset_index(drop=True)

    return diagnostics
