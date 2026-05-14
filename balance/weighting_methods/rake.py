# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import itertools
import logging
import math
import numbers
import pickle
from fractions import Fraction
from functools import reduce
from typing import Any, Callable, cast, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from balance import adjustment as balance_adjustment, util as balance_util
from balance.util import _safe_fillna_and_infer

logger: logging.Logger = logging.getLogger(__package__)


# TODO: Add options for only marginal distributions input
def _run_ipf_numpy(
    original: np.ndarray,
    target_margins: List[np.ndarray],
    convergence_rate: float,
    max_iteration: int,
    rate_tolerance: float,
) -> Tuple[np.ndarray, int, pd.DataFrame]:
    """Run iterative proportional fitting on a NumPy array.

    This reimplements the minimal subset of the :mod:`ipfn` package that is
    required for balance's usage.  The original implementation spends most of
    its time looping in pure Python over every slice of the contingency table,
    which is prohibitively slow for the high-cardinality problems we test
    against.  The logic here mirrors the algorithm used by ``ipfn.ipfn`` but
    applies the adjustments in a vectorised manner, yielding identical
    numerical results with a fraction of the runtime.

    The caller is expected to pass ``target_margins`` that correspond to
    single-axis marginals (which is how :func:`rake` constructs the inputs).
    """

    if original.ndim == 0:
        raise ValueError("`original` must have at least one dimension")

    table = np.asarray(original, dtype=np.float64)
    margins = [np.asarray(margin, dtype=np.float64) for margin in target_margins]

    # Pre-compute shapes and axes that are repeatedly required during the
    # iterative updates.  Each entry in ``axis_shapes`` represents how a
    # one-dimensional scaling factor should be reshaped in order to broadcast
    # along the appropriate axis of ``table``.
    axis_shapes: List[Tuple[int, ...]] = []
    sum_axes: List[Tuple[int, ...]] = []
    for axis in range(table.ndim):
        shape = [1] * table.ndim
        shape[axis] = table.shape[axis]
        axis_shapes.append(tuple(shape))
        sum_axes.append(tuple(i for i in range(table.ndim) if i != axis))

    conv = np.inf
    old_conv = -np.inf
    conv_history: List[float] = []
    iteration = 0

    while (
        iteration <= max_iteration
        and conv > convergence_rate
        and abs(conv - old_conv) > rate_tolerance
    ):
        old_conv = conv

        # Sequentially update the table for each marginal.  Because the
        # marginals correspond to single axes we can compute all scaling
        # factors at once, avoiding the expensive Python loops present in the
        # reference implementation.
        for axis, margin in enumerate(margins):
            current = table.sum(axis=sum_axes[axis])
            factors = np.ones_like(margin, dtype=np.float64)
            np.divide(margin, current, out=factors, where=current != 0)
            table *= factors.reshape(axis_shapes[axis])

        # Measure convergence using the same criterion as ``ipfn.ipfn``.  The
        # implementation there keeps the maximum absolute proportional
        # difference while naturally ignoring NaNs (which arise for 0/0).  We
        # match that behaviour by treating NaNs as zero deviation.
        conv = 0.0
        for axis, margin in enumerate(margins):
            current = table.sum(axis=sum_axes[axis])
            with np.errstate(divide="ignore", invalid="ignore"):
                diff = np.abs(np.divide(current, margin) - 1.0)
            current_conv = float(np.nanmax(diff)) if diff.size else 0.0
            if math.isnan(current_conv):
                current_conv = 0.0
            if current_conv > conv:
                conv = current_conv

        conv_history.append(conv)
        iteration += 1

    converged = int(iteration <= max_iteration)
    iterations_df = pd.DataFrame(
        {"iteration": range(len(conv_history)), "conv": conv_history}
    ).set_index("iteration")

    return table, converged, iterations_df


def rake(
    sample_df: pd.DataFrame,
    sample_weights: pd.Series,
    target_df: pd.DataFrame,
    target_weights: pd.Series,
    variables: Union[List[str], None] = None,
    transformations: Union[Dict[str, Callable[..., Any]], str, None] = "default",
    na_action: str = "add_indicator",
    max_iteration: int = 1000,
    convergence_rate: float = 0.0005,
    rate_tolerance: float = 1e-8,
    weight_trimming_mean_ratio: Union[float, int, None] = None,
    weight_trimming_percentile: Union[float, None] = None,
    keep_sum_of_weights: bool = True,
    *args: Any,
    store_fit_metadata: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Perform raking (using the iterative proportional fitting algorithm).
    See: https://en.wikipedia.org/wiki/Iterative_proportional_fitting

    Returns weights normalised to sum of target weights

    Arguments:
    sample_df --- (pandas dataframe) a dataframe representing the sample.
    sample_weights --- (pandas series) design weights for sample.
    target_df ---  (pandas dataframe) a dataframe representing the target.
    target_weights --- (pandas series) design weights for target.
    variables ---  (list of strings) list of variables to include in the model.
                   If None all joint variables of sample_df and target_df are used.
    transformations --- (dict) what transformations to apply to data before fitting the model.
                               Default is "default" (see apply_transformations function).
    na_action --- (string) what to do with NAs. Default is "add_indicator", which adds NaN as a
                  group (called "__NaN__") for each weighting variable (post-transformation);
                  "drop" removes rows with any missing values on any variable from both sample
                  and target.
    max_iteration --- (int) maximum number of iterations for iterative proportional fitting algorithm
    convergence_rate --- (float) convergence criteria; the maximum difference in proportions between
                          sample and target marginal distribution on any covariate in order
                          for algorithm to converge.
    rate_tolerance --- (float) convergence criteria; if convergence rate does not move more
                               than this amount than the algorithm is also considered to
                               have converged.
    weight_trimming_mean_ratio --- (float, int, optional) upper bound for weights expressed as a
                                   multiple of the mean weight. Delegated to
                                   :func:`balance.adjustment.trim_weights`.
    weight_trimming_percentile --- (float, optional) percentile limit(s) for winsorisation.
                                   Delegated to :func:`balance.adjustment.trim_weights`.
    keep_sum_of_weights --- (bool, optional) preserve the sum of weights during trimming before
                            rescaling to the target total. Defaults to True.
    store_fit_metadata --- (bool, optional, keyword-only)
                           when True, persist fit-time artifacts in
                           ``model`` for `BalanceFrame.predict_weights()`
                           replay/transfer workflows. Defaults to False.

    Returns:
    A dictionary including:
    "weight" --- The weights for the sample.
    "model" --- parameters of the model: iterations (dataframe with iteration numbers and
                convergence rate information at all steps), converged (Flag with the output
                status: 0 for failure and 1 for success).
                When ``store_fit_metadata=True`` it also includes fit-time
                artifacts for ``BalanceFrame.predict_weights()`` reconstruction.

    Notes:
    When exactly one adjustment variable is selected (either explicitly via
    ``variables=[...]`` or implicitly because only one common variable exists),
    this function delegates to :func:`balance.weighting_methods.poststratify.poststratify`.
    In that fallback path, the returned model metadata records
    ``method='poststratify'`` and the returned weight series is renamed to
    ``rake_weight`` for API consistency. Because
    ``BalanceFrame.predict_weights(data=...)`` dispatches by
    ``model['method']``, delegated fits follow poststratify's transfer-scoring
    capabilities/limitations rather than rake's.

    ``BalanceFrame.predict_weights()`` for rake reuses the fitted cell-ratio
    surface from this function (effectively ``m_fit / m_sample`` per joint
    cell) and applies it to design weights in the scoring sample. This is
    exact in-place replay (same sample rows as fit), but for ``data=...`` it is
    a transfer operation whose validity depends on the new sample having a
    similar joint distribution over rake variables as the training sample.
    If the joint distribution diverges, transferred rake weights can fail to
    recover target marginals even though the same fitted model artifacts are
    used. In that case, re-fit rake on the new sample against the same target.
    For this reason, balance emits an unconditional warning on transferred
    scoring (``predict_weights(data=...)``) when the fit is otherwise
    replayable, and raises for known unreplayable cases — currently
    ``transformations='default'`` and explicit dicts containing the known
    data-dependent helpers (``quantize``, ``fct_lump``).

    Examples:
    .. code-block:: python

        import pandas as pd
        from balance.weighting_methods.rake import rake
        sample_df = pd.DataFrame({"x": ["a", "b"]})
        target_df = pd.DataFrame({"x": ["a", "b"]})
        sample_weights = pd.Series([1.0, 1.0])
        target_weights = pd.Series([1.0, 1.0])
        result = rake(sample_df, sample_weights, target_df, target_weights, variables=["x"])
        result["weight"].tolist()
        # [1.0, 1.0]
    """
    assert (
        "weight" not in sample_df.columns.values
    ), "weight shouldn't be a name for covariate in the sample data"
    assert (
        "weight" not in target_df.columns.values
    ), "weight shouldn't be a name for covariate in the target data"

    # TODO: move the input checks into separate funnction for rake, ipw, poststratify
    assert isinstance(sample_df, pd.DataFrame), "sample_df must be a pandas DataFrame"
    assert isinstance(target_df, pd.DataFrame), "target_df must be a pandas DataFrame"
    assert isinstance(
        sample_weights, pd.Series
    ), "sample_weights must be a pandas Series"
    assert isinstance(
        target_weights, pd.Series
    ), "target_weights must be a pandas Series"
    assert sample_df.shape[0] == sample_weights.shape[0], (
        "sample_weights must be the same length as sample_df"
        f"{sample_df.shape[0]}, {sample_weights.shape[0]}"
    )
    assert target_df.shape[0] == target_weights.shape[0], (
        "target_weights must be the same length as target_df"
        f"{target_df.shape[0]}, {target_weights.shape[0]}"
    )
    if not isinstance(store_fit_metadata, bool):
        raise TypeError("`store_fit_metadata` must be a bool.")
    variables = balance_util.choose_variables(sample_df, target_df, variables=variables)

    logger.debug(f"Join variables for sample and target: {variables}")

    sample_df = sample_df.loc[:, variables]
    target_df = target_df.loc[:, variables]

    if len(variables) == 0:
        raise ValueError(
            "No shared weighting variables were found between sample and target. "
            "Pass `variables=[...]` with at least one common column present in both."
        )

    # Keep single-variable fallback behavior aligned with poststratify:
    # when variables are explicitly provided, out-of-scope transformation
    # entries are ignored.
    single_variable_transformations = transformations
    if len(variables) == 1 and isinstance(transformations, dict):
        single_variable_transformations = {
            key: value for key, value in transformations.items() if key in variables
        }
        if len(single_variable_transformations) == 0:
            single_variable_transformations = None

    transformations_for_pickle = single_variable_transformations
    if len(variables) > 1:
        if transformations == "default":
            transformations_for_pickle = balance_adjustment.default_transformations(
                (sample_df, target_df)
            )
        else:
            transformations_for_pickle = transformations

    if store_fit_metadata:
        # Fail fast: persisting non-pickleable callables (e.g. lambdas,
        # closures) would break `pickle.dumps(adjusted_bf)` workflows
        # downstream. Check here, before long-running fit work. Matches the
        # poststratify pattern.
        try:
            # @lint-ignore PYTHONPICKLEISBAD - serializability check only; no untrusted deserialization
            pickle.dumps(transformations_for_pickle)
        except Exception as exc:
            raise ValueError(
                "`transformations` must be pickleable when "
                "store_fit_metadata=True. Pass store_fit_metadata=False to "
                "disable fit-artifact persistence for this run."
            ) from exc

    if len(variables) == 1:
        logger.warning(
            "rake() received a single adjustment variable (%s); "
            "delegating to poststratify(). Returned model metadata will "
            "record method='poststratify'.",
            variables[0],
        )
        from balance.weighting_methods.poststratify import poststratify

        poststratified = poststratify(
            sample_df=sample_df,
            sample_weights=sample_weights,
            target_df=target_df,
            target_weights=target_weights,
            variables=variables,
            transformations=single_variable_transformations,
            na_action=na_action,
            weight_trimming_mean_ratio=weight_trimming_mean_ratio,
            weight_trimming_percentile=weight_trimming_percentile,
            keep_sum_of_weights=keep_sum_of_weights,
            store_fit_metadata=store_fit_metadata,
        )
        poststratified["weight"] = poststratified["weight"].rename("rake_weight")
        return poststratified

    if store_fit_metadata and transformations == "default":
        # `transformations='default'` resolves to data-dependent helpers
        # (`quantize`/`fct_lump`) which recompute bins/levels from the input
        # data. The fitted model can be replayed in-place via
        # `BalanceFrame.predict_weights()` but cannot be transferred to new
        # data via `predict_weights(data=...)` — that path will raise. To
        # enable transfer scoring, pass deterministic transformations at fit
        # time (e.g. wrappers built around stored fit-time bin edges).
        # TODO: replace this warning with a shared
        # `_freeze_data_dependent_transformations(transformations, dfs)`
        # helper that captures fit-time parameters (bin edges for `quantize`,
        # kept levels for `fct_lump`) and replays them as a deterministic
        # closure, so transfer scoring works out of the box for all four
        # weighting methods (rake/cbps/poststratify/ipw).
        logger.warning(
            "rake(store_fit_metadata=True) is being used together with "
            "transformations='default'. The fitted model can be replayed "
            "in-place via BalanceFrame.predict_weights(), but transfer "
            "scoring via predict_weights(data=...) will raise. Pass "
            "deterministic transformations at fit time to enable transfer."
        )

    transformations_to_apply = transformations
    if transformations == "default":
        if store_fit_metadata:
            transformations_to_apply = transformations_for_pickle
        else:
            transformations_to_apply = balance_adjustment.default_transformations(
                (sample_df, target_df)
            )

    sample_df, target_df = balance_adjustment.apply_transformations(
        (sample_df, target_df), transformations_to_apply
    )

    # TODO: separate into a function that handles NA (for rake, ipw, poststratify)
    if na_action == "drop":
        (sample_df, sample_weights) = balance_util.drop_na_rows(
            sample_df, sample_weights, "sample"
        )
        (target_df, target_weights) = balance_util.drop_na_rows(
            target_df, target_weights, "target"
        )
    elif na_action == "add_indicator":
        # pyrefly: ignore [bad-assignment]
        target_df = _safe_fillna_and_infer(target_df, "__NaN__")
        # pyrefly: ignore [bad-assignment]
        sample_df = _safe_fillna_and_infer(sample_df, "__NaN__")
    else:
        raise ValueError("`na_action` must be 'add_indicator' or 'drop'")

    # Alphabetize variables to ensure consistency across covariate order
    # (ipfn algorithm is iterative and variable order can matter on the margins)
    alphabetized_variables = list(variables)
    alphabetized_variables.sort()

    logger.debug(
        f"Alphabetized variable order is as follows: {alphabetized_variables}."
    )

    # Cast all data types as string to be explicit about each unique value
    # being its own group and to handle that `fillna()` above creates
    # series of type Object, which won't work for the ipfn script
    categories = []
    for variable in alphabetized_variables:
        target_df[variable] = target_df[variable].astype(str)
        sample_df[variable] = sample_df[variable].astype(str)

        sample_var_set = set(sample_df[variable].unique())
        target_var_set = set(target_df[variable].unique())
        sample_over_set = sample_var_set - target_var_set
        target_over_set = target_var_set - sample_var_set
        if len(sample_over_set):
            raise ValueError(
                "All variable levels in sample must be present in target. "
                f"'{variable}' in target is missing these levels: {sample_over_set}."
            )
        if len(target_over_set):
            logger.warning(
                f"'{variable}' has more levels in target than in sample. "
                f"'{variable}' in sample is missing these levels: {target_over_set}. "
                "These levels are treated as if they do not exist for that variable."
            )
        categories.append(sorted(sample_var_set.intersection(target_var_set)))

    logger.info(
        f"Final covariates and levels that will be used in raking: {dict(zip(alphabetized_variables, categories))}."
    )

    target_df = target_df.assign(weight=target_weights)
    sample_df = sample_df.assign(weight=sample_weights)

    sample_sum_weights = sample_df["weight"].sum()
    target_sum_weights = target_df["weight"].sum()

    # Calculate {# covariates}-dimensional array representation of the sample
    # for the ipfn algorithm

    grouped_sample_series = sample_df.groupby(alphabetized_variables)["weight"].sum()
    index = pd.MultiIndex.from_product(categories, names=alphabetized_variables)
    grouped_sample_full = grouped_sample_series.reindex(index, fill_value=0)
    m_sample = grouped_sample_full.to_numpy().reshape([len(c) for c in categories])
    m_fit_input = m_sample.copy()

    # Calculate target margins for ipfn
    target_margins = []
    for col, cats in zip(alphabetized_variables, categories):
        sums = (
            target_df.groupby(col)["weight"].sum()
            / target_sum_weights
            * sample_sum_weights
        )
        sums = sums.reindex(cats, fill_value=0)
        target_margins.append(sums.values)

    logger.debug(
        "Raking algorithm running following settings: "
        f" convergence_rate: {convergence_rate}; max_iteration: {max_iteration}; rate_tolerance: {rate_tolerance}"
    )

    # returns array with joint distribution of covariates and total weight
    # for that specific set of covariates
    # no longer uses the dataframe version of the ipfn algorithm
    # due to incompatability with latest Python versions
    m_fit, converged, iterations = _run_ipf_numpy(
        m_fit_input,
        target_margins,
        convergence_rate,
        max_iteration,
        rate_tolerance,
    )

    logger.debug(
        f"Raking algorithm terminated with following convergence: {converged}; "
        f"and iteration meta data: {iterations}."
    )

    if not converged:
        logger.warning("Maximum iterations reached, convergence was not achieved")

    combos = list(itertools.product(*categories))
    fit = pd.DataFrame(combos, columns=alphabetized_variables)
    fit["rake_weight"] = m_fit.flatten()

    raked = pd.merge(
        sample_df.reset_index(),
        fit,
        how="left",
        on=alphabetized_variables,
    )

    raked_rescaled = pd.merge(
        raked,
        grouped_sample_series.reset_index().rename(
            columns={"weight": "total_survey_weight"}
        ),
        how="left",
        on=alphabetized_variables,
    ).set_index("index")

    raked_rescaled["rake_weight"] = (
        raked_rescaled["rake_weight"]
        * raked_rescaled["weight"]
        / raked_rescaled["total_survey_weight"]
    )

    w = balance_adjustment.trim_weights(
        raked_rescaled["rake_weight"],
        target_sum_weights=target_sum_weights,
        weight_trimming_mean_ratio=weight_trimming_mean_ratio,
        weight_trimming_percentile=weight_trimming_percentile,
        keep_sum_of_weights=keep_sum_of_weights,
    ).rename("rake_weight")
    model: Dict[str, Any] = {
        "method": "rake",
        "iterations": iterations,
        "converged": converged,
        "perf": {"prop_dev_explained": np.array([np.nan])},
        # TODO: fix functions that use the perf and remove it from here
    }
    if store_fit_metadata:
        # Pickleability of `transformations_to_apply` was already validated
        # earlier (before the IPF compute), so persistence here is safe.
        model.update(
            {
                "store_fit_metadata": True,
                "variables": alphabetized_variables,
                "variables_before_transformations": list(variables),
                "categories": categories,
                "m_fit": m_fit,
                "m_sample": m_sample,
                "na_action": na_action,
                "transformations": transformations_to_apply,
                "transformations_origin": transformations,
                "training_sample_weights": sample_weights.copy(),
                "training_target_weights": target_weights.copy(),
                "weight_trimming_mean_ratio": weight_trimming_mean_ratio,
                "weight_trimming_percentile": weight_trimming_percentile,
                "keep_sum_of_weights": keep_sum_of_weights,
            }
        )
    return {"weight": w, "model": model}


# Private to ``balance``: callers should reach this through
# ``BalanceFrame.predict_weights()`` rather than importing it directly.
#
# TODO: decompose into smaller helpers (validate metadata, apply stored
# transformations, NA handling, code mapping, cell-ratio compute, target-sum
# determination, na_action='drop' index restoration). Best done alongside
# the cbps / poststratify / ipw extractions, where the same helpers can be
# shared across all four methods to reduce duplication.
#
# TODO: replace the ``transformations='default'`` and explicit
# data-dependent-dict transfer guards (below) with a shared helper
# ``_freeze_data_dependent_transformations(transformations, dfs) -> dict[str, Callable]``
# that captures fit-time parameters (bin edges for ``quantize``, kept
# levels for ``fct_lump``) and replays them as deterministic closures.
# Once that exists, transfer scoring works out of the box for
# ``transformations='default'`` and for explicit dicts containing
# ``quantize``/``fct_lump`` (or wrappers thereof such as
# ``functools.partial(fct_lump, prop=0.1)``), removing the entire
# transfer-rejection branch and its breaking-change burden on users. The
# helper should live alongside ``apply_transformations`` so all four
# weighting methods (rake/cbps/poststratify/ipw) can share it.
#
# NOTE: this guard logic is now duplicated in
# ``weighting_methods/poststratify.py::_predict_weights_from_model``
# (D105128469). CBPS and IPW will need the same guard when they gain
# transfer-mode support. Extracting the shared helper is increasingly
# load-bearing as the duplication grows.
def _predict_weights_from_model(
    model: dict[str, Any],
    sample_df: pd.DataFrame,
    sample_weights_full: pd.Series,
    target_df: pd.DataFrame,
    target_weights: pd.Series,
    is_transfer: bool,
) -> pd.Series:
    """Reconstruct rake weights from a stored fit-time model dict.

    Internal helper. The pure-math/model-driven core of
    ``BalanceFrame.predict_weights()`` for rake — callers should reach
    this through that public API rather than importing the helper
    directly. It takes plain DataFrames and Series for the scoring sample
    and target, plus the ``model`` dict persisted by :func:`rake` when
    ``store_fit_metadata=True``, and replays the cell-ratio surface
    (``m_fit / m_sample``) on the scoring sample.

    Two modes:
    - ``is_transfer=False`` (in-place replay): the scoring frames must be
      the same rows that were fit; ``training_sample_weights`` and
      ``training_target_weights`` from ``model`` are used as the canonical
      fit-time weights.
    - ``is_transfer=True`` (``data=...`` transfer): the scoring frames are
      a different sample/target. A warning is emitted unconditionally
      because transferred rake weights are only guaranteed to recover
      target marginals when the new sample's joint distribution over the
      rake variables is similar to the training sample's.

    Args:
        model: Fitted rake model dict produced with ``store_fit_metadata=True``.
        sample_df: Scoring sample's covariate frame (full index, pre-transform).
            Must contain every column listed in
            ``model['variables_before_transformations']``.
        sample_weights_full: Scoring sample's design weights aligned to
            ``sample_df``'s full index.
        target_df: Scoring target's covariate frame (pre-transform).
            Must contain every column listed in
            ``model['variables_before_transformations']``.
        target_weights: Scoring target's design weights aligned to
            ``target_df``'s full index.
        is_transfer: ``True`` when called from
            ``BalanceFrame.predict_weights(data=...)``; ``False`` for
            in-place replay (``data=None``).

    Returns:
        Predicted weights as a ``pd.Series`` with ``sample_weights_full``'s
        index. For ``na_action='drop'`` rake models, rows dropped during
        scoring are returned as ``NaN``.

    Raises:
        ValueError: If model metadata is missing/malformed, if the scoring
            frames are missing required covariates, if scoring rows fall in
            joint cells with zero fit-time sample mass, or if in-place
            replay is requested without compatible fit-time design weights.
    """
    required = (
        "variables",
        "variables_before_transformations",
        "categories",
        "m_fit",
        "m_sample",
        "na_action",
        "transformations",
    )
    if not is_transfer:
        required = required + ("training_sample_weights", "training_target_weights")
    missing = [key for key in required if key not in model]
    if missing:
        raise ValueError(
            "Rake model is missing fit-time metadata "
            f"({missing}) for predict_weights(). "
            "Call BalanceFrame.fit(method='rake') or run rake(..., "
            "store_fit_metadata=True)."
        )

    variables = model.get("variables")
    input_variables = model.get("variables_before_transformations")
    categories = model.get("categories")
    m_fit = model.get("m_fit")
    m_sample = model.get("m_sample")
    transformations_origin = model.get("transformations_origin")
    if (
        not isinstance(variables, list)
        or not isinstance(input_variables, list)
        or not isinstance(categories, list)
    ):
        raise ValueError("Rake model metadata is malformed for predict_weights().")
    if not isinstance(m_fit, np.ndarray) or not isinstance(m_sample, np.ndarray):
        raise ValueError("Rake model is missing stored contingency tables.")
    if is_transfer and transformations_origin == "default":
        raise ValueError(
            "Rake predict_weights(data=...) is unsupported for models fitted "
            "with transformations='default' because those transformations are "
            "data-dependent and not replayable across new samples. "
            "BalanceFrame.fit(method='rake') uses transformations='default' "
            "out of the box; to enable transfer scoring, pass deterministic "
            "transformations explicitly at fit time (for example "
            "transformations={'age': partial(quantize_with_edges, edges=...)} "
            "where the wrapper closes over fit-time bin edges) or re-fit rake "
            "on the scoring data."
        )
    if is_transfer and isinstance(transformations_origin, dict):
        # Best-effort guard: reject explicit dicts that directly
        # reference balance's known data-dependent helpers
        # (quantize, fct_lump). These recompute bins/levels from the
        # scoring data, so stored cell ratios no longer line up with
        # the transformed scoring cells and transfer would silently
        # return incorrect weights.
        #
        # This guard does NOT catch indirect uses such as
        # ``functools.partial(fct_lump, prop=0.1)``, top-level wrapper
        # functions, or user-defined data-dependent transformations.
        # The general invariant is: any callable whose output for a
        # row depends on other rows in the input is unsafe to replay
        # on a different sample. Users supplying such transformations
        # are responsible for either (a) wrapping them as
        # deterministic functions of stored fit-time parameters or
        # (b) re-fitting rake on the scoring data.
        from balance.utils.data_transformation import fct_lump, quantize

        data_dependent_helpers = {quantize, fct_lump}
        offenders = sorted(
            {
                getattr(fn, "__name__", repr(fn))
                for fn in transformations_origin.values()
                if fn in data_dependent_helpers
            }
        )
        if offenders:
            raise ValueError(
                "Rake predict_weights(data=...) is unsupported for models "
                f"fitted with data-dependent transformations ({', '.join(offenders)}). "
                "These recompute bins/levels from the scoring data, so "
                "stored cell ratios no longer line up with the transformed "
                "cells. To enable transfer scoring, replace each "
                "data-dependent helper with a deterministic wrapper that "
                "closes over fit-time parameters (e.g. partial(quantize, "
                "edges=fit_time_edges) or partial(fct_lump, kept_levels="
                "fit_time_levels)) — the wrappers must depend only on the "
                "stored fit-time state, not on the scoring data — or re-fit "
                "rake on the scoring data."
            )

    for column in input_variables:
        if column not in sample_df.columns or column not in target_df.columns:
            raise ValueError(
                "Rake predict_weights() cannot find required covariate "
                f"'{column}' in both sample and target."
            )
    sample_df = sample_df.loc[:, input_variables]
    target_df = target_df.loc[:, input_variables]

    na_action = cast(str, model.get("na_action", "add_indicator"))

    sample_df, target_df = balance_adjustment.apply_transformations(
        (sample_df, target_df), transformations=model.get("transformations")
    )
    for column in variables:
        if column not in sample_df.columns:
            raise ValueError(
                "Rake transform output is missing stored variable "
                f"'{column}' required for predict_weights()."
            )
    sample_df = sample_df.loc[:, variables]
    target_df = target_df.loc[:, variables]

    if len(variables) == 0:
        raise ValueError(
            "Rake predict_weights() model metadata is missing stored weighting "
            "variables. Re-fit the model (with store_fit_metadata=True) before "
            "calling predict_weights()."
        )

    sample_weights = sample_weights_full
    if not is_transfer:
        training_sample_weights = model.get("training_sample_weights")
        if isinstance(training_sample_weights, pd.Series):
            if na_action == "drop":
                sample_weights = training_sample_weights
            elif training_sample_weights.index.equals(sample_weights_full.index):
                sample_weights = training_sample_weights
            else:
                raise ValueError(
                    "Rake predict_weights() requires compatible fit-time sample design "
                    "weights for in-place replay. This can happen because "
                    "store_fit_metadata is missing/incompatible, or because you're "
                    "scoring a different sample; use predict_weights(data=...) "
                    "for different samples."
                )
        else:
            raise ValueError(
                "Rake predict_weights() requires compatible fit-time sample design "
                "weights for in-place replay. This can happen because "
                "store_fit_metadata is missing/incompatible, or because you're "
                "scoring a different sample; use predict_weights(data=...) "
                "for different samples."
            )

    dropped_target_weights: pd.Series | None = None
    if na_action == "drop":
        sample_df, sample_weights = balance_util.drop_na_rows(
            sample_df, sample_weights, "sample"
        )
        target_df, dropped_target_weights = balance_util.drop_na_rows(
            target_df, target_weights, "target"
        )
    elif na_action == "add_indicator":
        sample_df = pd.DataFrame(_safe_fillna_and_infer(sample_df, "__NaN__"))
        target_df = pd.DataFrame(_safe_fillna_and_infer(target_df, "__NaN__"))
    else:
        raise ValueError(
            f"Rake model has invalid na_action metadata '{na_action}' for predict_weights()."
        )
    sample_df = sample_df.astype(str)
    if m_fit.shape != m_sample.shape:
        raise ValueError(
            "Rake model metadata has incompatible fitted and sample table shapes."
        )

    category_maps = [{cat: i for i, cat in enumerate(cats)} for cats in categories]
    code_columns = []
    for column, cat_map in zip(variables, category_maps):
        codes = sample_df[column].map(cat_map)
        if bool(codes.isna().any()):
            raise ValueError(
                "Rake predict_weights() found rows that do not map to stored fit-time "
                "categories. Re-fit with compatible covariates."
            )
        code_columns.append(codes.astype(int).to_numpy())
    code_index = tuple(code_columns)

    if is_transfer:
        logger.warning(
            "Rake predict_weights(data=...): replaying fitted rake "
            "artifacts on a different sample is a transfer operation, "
            "not an exact fit. The stored cell-ratio surface "
            "(m_fit / m_sample) encodes the *training* target's "
            "marginal distribution; applying it to a new sample only "
            "produces weights calibrated to that *training* target, "
            "rescaled to the new target's total weight. The new "
            "target's marginals are NOT re-balanced. Predictions are "
            "therefore only valid when (a) the scoring sample's joint "
            "distribution over the rake variables is similar to the "
            "training sample's, AND (b) the scoring target's marginal "
            "distribution is similar to the training target's. "
            "Re-fit rake on the scoring sample/target for exact "
            "marginal matching."
        )
    m_sample_at_cells = m_sample[code_index]
    if bool((m_sample_at_cells <= 0).any()):
        raise ValueError(
            "Rake predict_weights() encountered sample rows in joint cells with "
            "zero fit-time sample mass (m_sample==0). Re-fit rake on data with "
            "compatible joint support."
        )
    # Compute ratios only for the scored cells. Avoids materializing a
    # dense array the size of the full contingency table on every call
    # (relevant for high-cardinality rakes).
    cell_ratios = m_fit[code_index] / m_sample_at_cells
    raw = pd.Series(
        sample_weights.to_numpy() * cell_ratios,
        index=sample_weights.index,
        dtype=float,
    )

    training_target_weights = model.get("training_target_weights")
    if not is_transfer and not isinstance(training_target_weights, pd.Series):
        raise ValueError(
            "Rake predict_weights() requires compatible fit-time target design "
            "weights for in-place replay. This can happen because "
            "store_fit_metadata is missing/incompatible, or because you're "
            "scoring a different sample; use predict_weights(data=...) "
            "for different samples."
        )
    if is_transfer:
        if na_action == "drop" and isinstance(dropped_target_weights, pd.Series):
            target_sum = float(dropped_target_weights.sum())
        else:
            target_sum = float(target_weights.sum())
    elif isinstance(training_target_weights, pd.Series):
        target_sum = float(training_target_weights.sum())
    elif na_action == "drop" and isinstance(dropped_target_weights, pd.Series):
        target_sum = float(dropped_target_weights.sum())
    else:
        target_sum = float(target_weights.sum())

    predicted = balance_adjustment.trim_weights(
        raw,
        target_sum_weights=target_sum,
        weight_trimming_mean_ratio=model.get("weight_trimming_mean_ratio"),
        weight_trimming_percentile=model.get("weight_trimming_percentile"),
        keep_sum_of_weights=bool(model.get("keep_sum_of_weights", True)),
    )
    if na_action == "drop":
        predicted_full = pd.Series(
            np.nan, index=sample_weights_full.index, dtype=float
        ).rename(predicted.name)
        predicted_full.loc[predicted.index] = predicted.to_numpy()
        predicted = predicted_full
    return predicted


def _lcm(a: int, b: int) -> int:
    """
    Calculates the least common multiple (LCM) of two integers.

    The least common multiple (LCM) of two or more numbers is the smallest positive integer that is divisible by each of the given numbers.
    In other words, it is the smallest multiple that the numbers have in common.
    The LCM is useful when you need to find a common denominator for fractions or synchronize repeating events with different intervals.

    For example, let's find the LCM of 4 and 6:

    The multiples of 4 are: 4, 8, 12, 16, 20, ...
    The multiples of 6 are: 6, 12, 18, 24, 30, ...
    The smallest multiple that both numbers have in common is 12, so the LCM of 4 and 6 is 12.

    The calculation is based on the property that the product of two numbers is equal to the product of their LCM and GCD: a * b = LCM(a, b) * GCD(a, b).
    (proof: https://math.stackexchange.com/a/589299/1406)

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The least common multiple of the two integers.

    Implementation note:
        Uses the mathematical relationship between LCM and GCD to efficiently compute the result.
        The absolute value ensures the LCM is always positive, even for negative inputs.
    """
    # NOTE: this function uses math.gcd which calculates the greatest common divisor (GCD) of two integers.
    # The greatest common divisor (GCD) of two or more integers is the largest positive integer that divides each of the given integers without leaving a remainder.
    # In other words, it is the largest common factor of the given integers.
    # For example, the GCD of 12 and 18 is 6, since 6 is the largest integer that divides both 12 and 18 without leaving a remainder.
    # Similarly, the GCD of 24, 36, and 48 is 12, since 12 is the largest integer that divides all three of these numbers without leaving a remainder.
    return abs(a * b) // math.gcd(a, b)


def _proportional_array_from_dict(
    input_dict: Dict[str, float], max_length: int = 10000
) -> List[str]:
    """
    Generates a proportional array based on the input dictionary.

    Args:
        input_dict (Dict[str, float]): A dictionary where keys are strings and values are their proportions (float).
        max_length (int): check if the length of the output exceeds the max_length. If it does, it will be scaled down to that length. Default is 10k.

    Returns:
        A list of strings where each key is repeated according to its proportion.

    Examples:
    .. code-block:: python

            _proportional_array_from_dict({"a":0.2, "b":0.8})
                # ['a', 'b', 'b', 'b', 'b']
            _proportional_array_from_dict({"a":0.5, "b":0.5})
                # ['a', 'b']
            _proportional_array_from_dict({"a":1/3, "b":1/3, "c": 1/3})
                # ['a', 'b', 'c']
            _proportional_array_from_dict({"a": 3/8, "b": 5/8})
                # ['a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']
            _proportional_array_from_dict({"a":3/5, "b":1/5, "c": 2/10})
                # ['a', 'a', 'a', 'b', 'c']
    """
    # Filter out items with zero value
    filtered_dict = {k: v for k, v in input_dict.items() if v > 0}

    # Normalize the values if they don't sum to 1
    sum_values = sum(filtered_dict.values())
    if sum_values != 1:
        filtered_dict = {k: v / sum_values for k, v in filtered_dict.items()}

    # Convert the proportions to fractions
    fraction_dict = {
        k: Fraction(v).limit_denominator() for k, v in filtered_dict.items()
    }

    # Find the least common denominator (LCD) of the fractions
    lcd = 1
    for fraction in fraction_dict.values():
        lcd *= fraction.denominator // math.gcd(lcd, fraction.denominator)

    # Calculate the scaling factor based on max_length
    scaling_factor = min(1, max_length / lcd)

    result = []
    for key, fraction in fraction_dict.items():
        # Calculate the count for each key based on its proportion and scaling_factor
        k_count = round((fraction * lcd).numerator * scaling_factor)
        # Extend the result array with the key repeated according to its count
        result.extend([key] * k_count)

    return result


def _hare_niemeyer_allocation(
    proportions: Dict[str, float],
    n: int,
) -> List[str]:
    """Allocate *n* slots to categories using the Hare-Niemeyer (largest-remainder) method.

    This avoids rounding bias by first assigning floor counts then distributing
    the remaining slots to the categories with the largest fractional remainders.

    See https://en.wikipedia.org/wiki/Hare_quota for background on the method.

    .. note::
       This is an internal helper.  Input validation (type checks on
       proportions, NaN / inf / negative guards) is performed by the caller
       :func:`_realize_dicts_of_proportions`; this function assumes its inputs
       are already validated.

    Args:
        proportions: Mapping of category labels to their (non-negative, finite)
            proportions.  Zero-valued categories are skipped.  Values need not
            sum to 1; they are normalised internally.
        n: Total number of slots to allocate (positive ``int``).

    Returns:
        A list of length *n* with each label repeated according to its allocated
        count.  Ties in fractional remainders are broken deterministically by
        category label (alphabetical).

    Examples:
        >>> _hare_niemeyer_allocation({"a": 0.2, "b": 0.8}, 5)
        ['a', 'b', 'b', 'b', 'b']
        >>> _hare_niemeyer_allocation({"a": 0.5, "b": 0.5}, 3)
        ['a', 'a', 'b']
    """
    # Filter zeros and normalise
    filtered = {k: float(v) for k, v in proportions.items() if float(v) > 0}
    if not filtered:
        raise ValueError("All proportions are zero or empty; cannot allocate slots.")
    total = sum(filtered.values())
    normalised = {k: v / total for k, v in filtered.items()}

    # Ideal (real-valued) counts and floor counts
    ideals = {k: p * n for k, p in normalised.items()}
    floors = {k: int(v) for k, v in ideals.items()}

    # Distribute remaining slots to largest fractional remainders;
    # secondary sort on label for deterministic tie-breaking.
    remaining = n - sum(floors.values())
    sorted_keys = sorted(
        ideals.keys(),
        key=lambda k: (-(ideals[k] - floors[k]), k),
    )
    counts = dict(floors)
    for i in range(int(remaining)):
        counts[sorted_keys[i]] += 1

    # Build result preserving original dict insertion order
    result: List[str] = []
    for k in proportions:
        if k in counts:
            result.extend([k] * counts[k])
    return result


def _find_lcm_of_array_lengths(arrays: Dict[str, List[str]]) -> int:
    """
    Finds the least common multiple (LCM) of the lengths of arrays in the input dictionary.

    Args:
        arrays: A dictionary where keys are strings and values are lists of strings.

    Returns:
        The LCM of the lengths of the arrays in the input dictionary.

    Example:
    .. code-block:: python

            arrays = {
                        "v1": ["a", "b", "b", "c"],
                        "v2": ["aa", "bb"]
                    }
            _find_lcm_of_array_lengths(arrays)
                # 4

            arrays = {
                        "v1": ["a", "b", "b", "c"],
                        "v2": ["aa", "bb"],
                        "v3": ["a1", "a2", "a3"]
                    }
            _find_lcm_of_array_lengths(arrays)
                # 12
    """
    array_lengths = [len(arr) for arr in arrays.values()]
    lcm_length = reduce(lambda x, y: _lcm(x, y), array_lengths)
    return lcm_length


def _realize_dicts_of_proportions(
    dict_of_dicts: Dict[str, Dict[str, float]],
    max_length: int = 10000,
) -> Dict[str, List[str]]:
    """
    Generates proportional arrays of equal length for each input dictionary.

    This can be used to get an input dict of proportions of values, and produce a dict with arrays that realizes these proportions.
    It can be used as input to the Sample object so it could be used for running raking.

    Args:
        dict_of_dicts: A dictionary of dictionaries, where each key is a string and
                   each value is a dictionary with keys as strings and values as their
                   proportions as real-valued numeric types implementing ``numbers.Real``
                   (e.g., Python floats, NumPy or pandas scalar types).
        max_length: Maximum number of rows in the output arrays. When the least
                   common multiple (LCM) of the
                   individual array lengths exceeds this value the output is capped at
                   ``max_length`` rows and each variable is re-allocated using the
                   Hare-Niemeyer (largest remainder) method. Default is 10000.

    Returns:
        A dictionary with the same keys as the input and equal length arrays as values.

    Examples:
    .. code-block:: python

            dict_of_dicts = {
                "v1": {"a": 0.2, "b": 0.6, "c": 0.2},
                "v2": {"aa": 0.5, "bb": 0.5}
            }

            realize_dicts_of_proportions(dict_of_dicts)
            # {'v1': ['a', 'b', 'b', 'b', 'c', 'a', 'b', 'b', 'b', 'c'], 'v2': ['aa', 'bb', 'aa', 'bb', 'aa', 'bb', 'aa', 'bb', 'aa', 'bb']}


            dict_of_dicts = {
                "v1": {"a": 0.2, "b": 0.6, "c": 0.2},
                "v2": {"aa": 0.5, "bb": 0.5},
                "v3": {"A": 0.2, "B": 0.8},
            }
            realize_dicts_of_proportions(dict_of_dicts)
                # {'v1': ['a', 'b', 'b', 'b', 'c', 'a', 'b', 'b', 'b', 'c'],
                #  'v2': ['aa', 'bb', 'aa', 'bb', 'aa', 'bb', 'aa', 'bb', 'aa', 'bb'],
                #  'v3': ['A', 'B', 'B', 'B', 'B', 'A', 'B', 'B', 'B', 'B']}
            # The above example could have been made shorter. But that's a limitation of the function.

            dict_of_dicts = {
                "v1": {"a": 0.2, "b": 0.6, "c": 0.2},
                "v2": {"aa": 0.5, "bb": 0.5},
                "v3": {"A": 0.2, "B": 0.8},
                "v4": {"A": 0.1, "B": 0.9},
            }
            realize_dicts_of_proportions(dict_of_dicts)
                # {'v1': ['a', 'b', 'b', 'b', 'c', 'a', 'b', 'b', 'b', 'c'],
                #  'v2': ['aa', 'bb', 'aa', 'bb', 'aa', 'bb', 'aa', 'bb', 'aa', 'bb'],
                #  'v3': ['A', 'B', 'B', 'B', 'B', 'A', 'B', 'B', 'B', 'B'],
                #  'v4': ['A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']}
    """
    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or max_length < 1
    ):
        raise ValueError(f"max_length must be a positive integer, got {max_length!r}.")
    if not dict_of_dicts:
        raise ValueError("dict_of_dicts must be non-empty; got an empty dictionary.")
    # Validate all inner proportion values before any float coercion so that
    # invalid inputs (bool, NaN, inf, negative) are always caught and reported
    # with the variable name, regardless of whether the LCM-capping path is taken.
    for var_name, inner_dict in dict_of_dicts.items():
        for cat_name, v in inner_dict.items():
            if isinstance(v, bool) or not isinstance(v, numbers.Real):
                raise ValueError(
                    f"Variable '{var_name}', category '{cat_name}': proportion must be "
                    f"a real number (not bool), got {type(v).__name__}."
                )
            try:
                fv = float(v)
            except (TypeError, OverflowError) as exc:
                raise ValueError(
                    f"Variable '{var_name}', category '{cat_name}': proportion must be "
                    f"convertible to float, got {v!r}."
                ) from exc
            if math.isnan(fv) or math.isinf(fv):
                raise ValueError(
                    f"Variable '{var_name}', category '{cat_name}': proportion must be "
                    f"finite, got {v}."
                )
            if fv < 0:
                raise ValueError(
                    f"Variable '{var_name}', category '{cat_name}': proportion must be "
                    f"non-negative, got {v}."
                )
    # Generate proportional arrays for each dictionary.  We pass max_length so
    # that the per-variable array length is roughly bounded, but individual arrays
    # can still exceed max_length when there are many small-weight categories that
    # each round up to 1.  The LCM check below catches the common case; the
    # additional lcm_length == 0 guard handles the edge case where one array is
    # empty (all counts round to zero).  When either condition holds, we discard
    # these arrays and switch to Hare-Niemeyer.
    arrays = {
        k: _proportional_array_from_dict(
            {category: float(weight) for category, weight in v.items()},
            max_length=max_length,
        )
        for k, v in dict_of_dicts.items()
    }

    # Find the LCM over the lengths of all the arrays
    lcm_length = _find_lcm_of_array_lengths(arrays)

    if lcm_length > max_length or lcm_length == 0:
        # The LCM of the individual array lengths exceeds the cap, or one or more
        # arrays are empty (lcm_length == 0 when any length is 0).  Rather than
        # producing a DataFrame with tens-of-millions of rows or crashing with a
        # ZeroDivisionError, reallocate every variable independently with
        # Hare-Niemeyer (largest remainder) rounding against the fixed target of
        # max_length rows.
        logger.warning(
            f"The LCM of array lengths ({lcm_length:,}) exceeds max_length "
            f"({max_length:,}) or an array is empty. Capping output at "
            f"{max_length:,} rows and reallocating counts using the "
            f"Hare-Niemeyer (largest remainder) method."
        )
        return {
            k: _hare_niemeyer_allocation(d, max_length)
            for k, d in dict_of_dicts.items()
        }

    # Extend each array to have the same LCM length while maintaining proportions
    result = {}
    for k, arr in arrays.items():
        factor = lcm_length // len(arr)
        extended_arr = arr * factor
        result[k] = extended_arr

    return result


def prepare_marginal_dist_for_raking(
    dict_of_dicts: Dict[str, Dict[str, float]],
    max_length: int = 10000,
) -> pd.DataFrame:
    """
    Realizes a nested dictionary of proportions into a DataFrame.

    Args:
        dict_of_dicts: A nested dictionary where the outer keys are column names
                   and the inner dictionaries have keys as category labels
                   and values as their proportions (real numbers).
        max_length: Maximum number of rows in the resulting DataFrame. Must be
                   an integer. When the natural least common multiple (LCM) based
                   row count would exceed
                   this value the output is capped using Hare-Niemeyer (largest
                   remainder) allocation. Default is 10000.

    Returns:
        A DataFrame with columns specified by the outer keys of the input dictionary
        and rows containing the category labels according to their proportions.
        An additional "id" column is added with integer values as row identifiers.

    Examples:
    .. code-block:: python

        from balance.weighting_methods.rake import prepare_marginal_dist_for_raking
        df = prepare_marginal_dist_for_raking(
            {"A": {"a": 0.5, "b": 0.5}, "B": {"x": 0.2, "y": 0.8}}
        )
        df.columns.tolist()
        # ['A', 'B', 'id']
    """
    target_dict_from_marginals = _realize_dicts_of_proportions(
        dict_of_dicts, max_length=max_length
    )
    target_df_from_marginals = pd.DataFrame.from_dict(target_dict_from_marginals)
    # Add an id column:
    target_df_from_marginals["id"] = range(target_df_from_marginals.shape[0])

    return target_df_from_marginals
