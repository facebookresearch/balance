# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Bridge between :mod:`balance` and :mod:`diff_diff`.

This module is the single reviewed handoff between balance's reweighting
output (a :class:`balance.Sample` with an active ``weight_column``) and
diff-diff's survey-aware DiD estimators (which all consume a
``diff_diff.SurveyDesign``).

Design choices (rationale at the cited line):

* ``weight_type="pweight"`` is the documented default. It is the only
  choice compatible with CallawaySantAnna (``staggered.py:1500-1506``),
  StackedDiD (``survey.py:1051-1085``), HAD-continuous, ImputationDiD,
  TwoStageDiD, StaggeredTripleDifference, EfficientDiD, WooldridgeDiD,
  TROP, and dCDH. ``aweight`` is supported by a smaller subset
  (DifferenceInDifferences, TwoWayFixedEffects, MultiPeriodDiD,
  SunAbraham, ContinuousDiD, EfficientDiD) but balance does not produce
  precision weights, so exposing it would only be a footgun.
* ``lonely_psu="adjust"`` default — the safest of the three documented
  values (``"remove"`` drops the row, ``"certainty"`` zeroes its variance
  contribution, ``"adjust"`` rescales survivors). Documented at
  ``survey.py:80-84`` in diff-diff.
* No expensive copies — all functions pass column NAMES (str), never
  Series, into ``SurveyDesign``, matching balance's documented
  ``weight_column: str | None`` contract at ``sample_frame.py:570-600``.
* Lazy import of ``diff_diff`` so that bare ``import balance`` works
  even when diff-diff isn't installed; the rewritten ``ImportError``
  points users at ``pip install balance[did]``.

Usage::

    import balance
    from balance.interop import diff_diff as bd

    s = balance.Sample.from_frame(
        df, id_column="id", weight_column="w", outcome_columns=["y"],
    )
    s_target = balance.Sample.from_frame(target_df, id_column="id")
    adj = s.set_target(s_target).adjust(method="ipw")

    # End-to-end fit (attaches `_balance_adjustment` for provenance):
    res = bd.fit_did(
        adj, estimator="CallawaySantAnna",
        outcome="y", time="period", unit="geo",
        treatment_first="first_treat",
    )

    # Or build the SurveyDesign yourself for finer control:
    design = bd.to_survey_design(adj, design_columns={"strata": "region"})

    # Cross-package diagnostics in one place:
    diag = bd.as_balance_diagnostic(adj, res)
"""

from __future__ import annotations

import inspect
import logging
import warnings
from types import ModuleType
from typing import Any, Mapping

import pandas as pd
from balance.interop._common import (
    active_weight_column,
    attach_balance_provenance,
    drop_history_columns,
    drop_nan_weight_rows,
)
from balance.interop.conventions import DEFAULT_DESIGN_COLUMNS, WEIGHT_TYPE_DEFAULT
from balance.sample_class import Sample

# @manual=fbsource//third-party/pypi/diff-diff:diff-diff
try:
    import diff_diff as _dd
except ImportError as _e:  # pragma: no cover
    _IMPORT_ERROR: ImportError | None = ImportError(
        "balance.interop.diff_diff requires the diff-diff package "
        "(>=3.3.0,<4). Install via `pip install 'balance[did]'` or "
        f"`pip install diff-diff`. Original error: {_e}"
    )
    _dd = None  # type: ignore[assignment]
else:
    _IMPORT_ERROR = None

# Route through the package logger ("balance"), matching the convention used
# by ``balance/sample_class.py`` and ``balance/balance_frame.py`` so adapter
# log output inherits the configuration set up in ``balance/__init__.py``.
logger: logging.Logger = logging.getLogger(__package__)

#: Field NAMES on ``diff_diff.SurveyDesign`` that adapter callers may forward
#: via ``design_columns``. Source of truth: ``survey.py:27-72`` in diff-diff.
_ALLOWED_DESIGN_FIELDS: frozenset[str] = frozenset(
    {
        "strata",
        "psu",
        "ssu",
        "fpc",
        "replicate_weights",
        "replicate_method",
        "replicate_strata",
        "fay_rho",
        "combined_weights",
        "replicate_scale",
        "replicate_rscales",
        "mse",
        "nest",
    }
)

#: SurveyDesign / convention keys whose values are literal configuration
#: (method names, floats, booleans) rather than column references on
#: ``sample.df``. The column-existence validation in
#: ``_resolve_design_columns`` must skip these keys -- otherwise passing a
#: perfectly valid value like ``replicate_method="JK1"`` or ``mse=True``
#: would be rejected with a misleading "column not found" error.
#:
#: Note: ``weight_type`` is included here even though it is NOT in
#: ``_ALLOWED_DESIGN_FIELDS`` (the allowed-via-design_columns set). It is
#: separately surfaced as the ``weight_type=`` kwarg on the public adapter
#: functions and is also a default in ``DEFAULT_DESIGN_COLUMNS``; rejecting
#: ``design_columns={"weight_type": ...}`` happens earlier in the validator
#: (line 145) but defensively listing it here keeps the literal-value rule
#: consistent. The other keys (``replicate_method``, ``fay_rho``, ...) ARE
#: subsets of ``_ALLOWED_DESIGN_FIELDS``.
_LITERAL_VALUE_DESIGN_FIELDS: frozenset[str] = frozenset(
    {
        "weight_type",
        "replicate_method",
        "fay_rho",
        "replicate_scale",
        "replicate_rscales",
        "mse",
        "nest",
    }
)


def _require_diff_diff() -> ModuleType:
    """Return the diff-diff module, or raise the rewritten ImportError.

    Returning the module (instead of just raising on failure) lets Pyre
    narrow ``_dd`` from ``ModuleType | None`` to ``ModuleType`` at every
    call site without a separate ``assert`` line.
    """
    if _IMPORT_ERROR is not None:
        raise _IMPORT_ERROR
    assert _dd is not None  # narrows for Pyre
    return _dd


def _resolve_design_columns(
    sample: Sample,
    design_columns: Mapping[str, object] | None,
) -> dict[str, object]:
    """Merge user-supplied design columns with the conventions defaults.

    ``weights`` is always overridden with the active balance weight
    column name — ``design_columns={"weights": ...}`` is rejected up
    front because balance is the source of truth for which column is
    "live".

    Convention auto-population (``stratum`` -> ``"strata"``, ``psu`` ->
    ``"psu"`` etc.) only runs when ``design_columns`` is ``None``.
    Passing an explicit mapping -- including the empty mapping ``{}`` --
    is the supported opt-out for callers whose dataframe has covariates
    that happen to share names with convention design fields and should
    NOT be reinterpreted as survey-design metadata.
    """
    user: Mapping[str, object] = design_columns if design_columns is not None else {}
    if "weights" in user:
        raise ValueError(
            "design_columns cannot set 'weights' — balance's active "
            "weight_column is the source of truth. Use Sample.adjust(...) "
            "or Sample.from_frame(weight_column=...) to control which "
            "column is active."
        )
    if "weight_type" in user:
        # The caller-facing ``to_survey_design(..., weight_type=...)`` kwarg is
        # the only supported channel for this; setting it via design_columns
        # used to be silently overridden by the kwarg, which produced a
        # confusing API where ``design_columns={'weight_type': 'aweight'}``
        # was accepted at validation time but ignored at use-time.
        raise ValueError(
            "design_columns cannot set 'weight_type' — pass it via the "
            "``weight_type=`` kwarg on ``to_survey_design``/``to_panel_for_did``/"
            "``fit_did`` instead. Earlier versions silently discarded a "
            "``weight_type`` value supplied through design_columns."
        )
    extra: set[str] = set(user) - _ALLOWED_DESIGN_FIELDS
    if extra:
        raise ValueError(
            f"design_columns has unknown SurveyDesign fields: {sorted(extra)}. "
            f"Allowed fields are: {sorted(_ALLOWED_DESIGN_FIELDS)} "
            "(see survey.py:27-72 in diff-diff)."
        )
    weights_col: str = active_weight_column(sample)
    # Validate user-supplied column NAMES exist in ``sample.df`` against the
    # raw column index (no ``str()`` coercion): if the dataframe has the
    # integer column ``1`` and the caller passes ``{"strata": "1"}``, the
    # adapter must reject that reference rather than silently letting it
    # through and forwarding a string-typed name that diff-diff will then
    # fail to look up on an int-indexed frame. Skip
    # ``_LITERAL_VALUE_DESIGN_FIELDS`` because their values are configuration
    # literals (e.g. ``replicate_method="JK1"``, ``fay_rho=0.5``,
    # ``mse=True``), not column references on ``sample.df``.
    df_columns_raw: set[object] = set(sample.df.columns)
    column_ref_keys: dict[str, object] = {
        k: v for k, v in user.items() if k not in _LITERAL_VALUE_DESIGN_FIELDS
    }
    non_string_column_refs: dict[str, object] = {
        k: v for k, v in column_ref_keys.items() if not isinstance(v, str)
    }
    if non_string_column_refs:
        raise ValueError(
            "design_columns values that reference dataframe columns must "
            f"be strings. Got non-string values for: "
            f"{sorted(non_string_column_refs.items(), key=lambda kv: kv[0])}. "
            "(Literal-value fields like `mse`, `nest`, `fay_rho`, "
            "`replicate_method`, `replicate_scale`, `replicate_rscales` "
            "are exempt — those accept their natural Python types.)"
        )
    missing_user_cols: dict[str, object] = {
        k: v for k, v in column_ref_keys.items() if v not in df_columns_raw
    }
    if missing_user_cols:
        # Format column labels via ``repr`` so int / str columns are
        # distinguishable in the error message (``1`` vs ``"1"``).
        missing_repr: list[str] = sorted(
            f"{k}={v!r}" for k, v in missing_user_cols.items()
        )
        available_repr: list[str] = sorted(repr(c) for c in df_columns_raw)
        raise ValueError(
            "design_columns references column(s) that do not exist in "
            f"sample.df: {missing_repr}. "
            f"Available columns: {available_repr}."
        )
    # Convention auto-population is opt-in: only when the caller passed
    # ``design_columns=None`` do we layer ``DEFAULT_DESIGN_COLUMNS`` entries
    # whose target columns happen to exist in ``sample.df``. Passing any
    # explicit mapping (including ``{}``) suppresses this entirely, so a
    # caller whose frame has a covariate literally named ``"psu"`` /
    # ``"stratum"`` / ``"fpc"`` can opt out of having it silently promoted
    # into ``SurveyDesign``.
    merged: dict[str, object] = dict(user)
    auto_populate: bool = design_columns is None
    for key, default_col in DEFAULT_DESIGN_COLUMNS.items():
        if key in merged or key == "weights":
            continue
        if key == "weight_type":
            # ``weight_type`` is a literal value (e.g. ``"pweight"``), not a
            # column reference. Always populate when the caller did not
            # supply one — it is a config default, not a column inference.
            merged[key] = default_col
            continue
        if auto_populate and default_col in df_columns_raw:
            merged[key] = default_col
            logger.info(
                "balance.interop.diff_diff: auto-populating SurveyDesign "
                "field %r from sample.df column %r (matched the default "
                "convention name). Pass an explicit design_columns mapping "
                "(or design_columns={}) to suppress this.",
                key,
                default_col,
            )
    merged["weights"] = weights_col
    return merged


def to_survey_design(
    sample: Sample,
    *,
    design_columns: Mapping[str, object] | None = None,
    weight_type: str = WEIGHT_TYPE_DEFAULT,
    lonely_psu: str = "adjust",
) -> Any:
    """Build a ``diff_diff.SurveyDesign`` from a balance ``Sample``.

    Parameters
    ----------
    sample :
        A balance ``Sample`` (typically post-``adjust()``).
    design_columns :
        Optional mapping of ``SurveyDesign`` field name to either a
        ``sample.df`` column name (``str``) or a literal config value.
        Keys are validated against the documented ``SurveyDesign``
        field set (``survey.py:27-72`` in diff-diff). Most keys
        (``strata``, ``psu``, ``ssu``, ``fpc``, ``replicate_weights``,
        ``replicate_strata``, ``combined_weights``) take column names;
        the literal-value keys ``mse`` / ``nest`` (bool), ``fay_rho``
        / ``replicate_scale`` (float), ``replicate_method`` (str),
        and ``replicate_rscales`` (list/array) accept their natural
        Python types directly. The ``weights`` slot is always sourced
        from ``sample.weight_column`` and may NOT be overridden — this
        is enforced. When ``design_columns`` is ``None`` the adapter
        opportunistically auto-populates convention defaults (e.g.
        ``"strata"`` from ``sample.df["stratum"]`` when that column
        exists). Pass an explicit mapping — including the empty
        mapping ``{}`` — to suppress that behaviour entirely; this is
        the supported escape hatch for frames whose covariates happen
        to share names with convention design fields.
    weight_type :
        Forwarded to ``SurveyDesign.weight_type``. Default ``"pweight"``
        is REQUIRED for CallawaySantAnna, StackedDiD, HAD-continuous,
        ImputationDiD, TwoStageDiD, EfficientDiD, WooldridgeDiD, TROP,
        and dCDH (per ``staggered.py:1500-1506`` and the shared
        ``_resolve_pweight_only`` guard at ``survey.py:1051-1085``).
    lonely_psu :
        Forwarded; one of ``"remove" | "certainty" | "adjust"``
        (validated at ``survey.py:80-84`` in diff-diff). Default
        ``"adjust"`` is the safest — it re-scales surviving PSUs in
        singleton strata rather than dropping them.

    Returns
    -------
    diff_diff.SurveyDesign
        Ready to feed into any diff-diff estimator's ``survey_design=``.

    Raises
    ------
    ImportError
        If diff-diff is not installed.
    ValueError
        If ``sample.weight_column`` is ``None``, or if ``design_columns``
        sets the reserved ``"weights"`` key, or contains a key that is
        not a valid ``SurveyDesign`` field.
    """
    dd = _require_diff_diff()
    merged: dict[str, object] = _resolve_design_columns(sample, design_columns)
    # Strip the "weight_type" slot before forwarding — conventions stores it
    # as a default but SurveyDesign takes it as a separate kwarg.
    merged.pop("weight_type", None)
    return dd.SurveyDesign(
        weight_type=weight_type,
        lonely_psu=lonely_psu,
        **merged,
    )


def to_panel_for_did(
    sample: Sample,
    *,
    by: list[str] | str,
    outcomes: list[str] | str,
    covariates: list[str] | None = None,
    second_stage_weights: str = WEIGHT_TYPE_DEFAULT,
    min_n: int = 2,
    design_columns: Mapping[str, object] | None = None,
    weight_type: str = WEIGHT_TYPE_DEFAULT,
    lonely_psu: str = "adjust",
) -> tuple[pd.DataFrame, Any]:
    """Aggregate a respondent-level ``balance.Sample`` into a geo-period
    panel via ``diff_diff.aggregate_survey``.

    Use this when balance has produced respondent-level (microdata)
    weights and the user wants to feed a downstream estimator that
    operates on the geo-period panel scale (e.g. CallawaySantAnna with
    ``panel=True``).

    The first element of ``by`` is the geographic unit per
    ``prep.py:1820-1821`` in diff-diff; the remaining ``by`` columns
    define the period axis.

    Parameters
    ----------
    sample :
        A balance ``Sample`` whose active weight column is the
        first-stage survey weight.
    by, outcomes, covariates, min_n :
        Forwarded verbatim to ``diff_diff.aggregate_survey``.
        ``covariates`` are produced as ``{covariate}_mean`` cells (per
        ``prep.py:1750-1758``).
    second_stage_weights :
        Forwarded to ``aggregate_survey``. Default ``"pweight"`` is
        compatible with all survey-aware estimators.
    design_columns :
        Same semantics as in :func:`to_survey_design` — forwarded to the
        first-stage ``SurveyDesign``.
    weight_type :
        Forwarded to the first-stage :func:`to_survey_design` call.
        Default ``"pweight"`` (matches ``WEIGHT_TYPE_DEFAULT``).
    lonely_psu :
        Forwarded to the first-stage :func:`to_survey_design` call.
        Default ``"adjust"``.

    Returns
    -------
    (panel_df, second_stage_design) :
        Two-tuple as returned by ``diff_diff.aggregate_survey``
        (``prep.py:1498-1507``). ``second_stage_design`` is a fresh
        ``SurveyDesign`` whose ``weights`` is the auto-generated
        ``{first_outcome}_weight`` column on ``panel_df`` and whose
        ``psu`` is the geographic unit column.
    """
    dd = _require_diff_diff()
    first_stage_design: Any = to_survey_design(
        sample,
        design_columns=design_columns,
        weight_type=weight_type,
        lonely_psu=lonely_psu,
    )
    weights_col: str = active_weight_column(sample)
    # Filter NaN-weight rows BEFORE dropping balance's history columns so
    # the warning's "of N rows" denominator reflects the user's view of
    # the sample, not the post-history-strip view.
    df: pd.DataFrame = drop_nan_weight_rows(
        sample.df, weights_col, ctx="to_panel_for_did"
    )
    df = drop_history_columns(df)
    return dd.aggregate_survey(
        df,
        by=by,
        outcomes=outcomes,
        survey_design=first_stage_design,
        covariates=covariates,
        min_n=min_n,
        second_stage_weights=second_stage_weights,
    )


def fit_did(
    sample: Sample,
    *,
    estimator: str = "CallawaySantAnna",
    outcome: str,
    time: str,
    unit: str,
    treatment_first: str | None = None,
    treatment: str | None = None,
    covariates: list[str] | None = None,
    design_columns: Mapping[str, object] | None = None,
    weight_type: str = WEIGHT_TYPE_DEFAULT,
    lonely_psu: str = "adjust",
    preserve_adjustment: bool = True,
    drop_history: bool = True,
    **estimator_kwargs: object,
) -> object:
    """Build a SurveyDesign and fit a diff-diff estimator in one call.

    Parameters
    ----------
    sample :
        A balance ``Sample`` (typically post-``adjust()``).
    estimator :
        Class NAME or short alias of a diff-diff estimator class.
        Resolved via ``getattr(diff_diff, estimator)`` against the
        public alias block at ``diff_diff.__init__.py:271-288``.
        Examples: ``"CallawaySantAnna"``, ``"CS"``,
        ``"DifferenceInDifferences"``, ``"DiD"``, ``"SunAbraham"``,
        ``"ImputationDiD"``, ``"BJS"``, ``"StackedDiD"``,
        ``"HeterogeneousAdoptionDiD"``, ``"HAD"``.
    outcome, time, unit, treatment_first, treatment, covariates :
        Forwarded to the estimator's ``fit()``. Names cover the most
        common kwargs across estimators; estimators that disagree
        (e.g. dCDH uses ``group``/``treatment`` and ``controls``)
        receive their own names via ``**estimator_kwargs``.
    design_columns :
        Same semantics as in :func:`to_survey_design`.
    weight_type :
        Forwarded to the internal :func:`to_survey_design` call. Default
        ``"pweight"`` (matches ``WEIGHT_TYPE_DEFAULT``). Use ``"aweight"``
        on estimators that explicitly support it (e.g. when
        ``aggregate_survey(second_stage_weights="aweight")`` is upstream
        of the panel and the estimator has a precision-weighted variance
        path).
    lonely_psu :
        Forwarded to the internal :func:`to_survey_design` call. Default
        ``"adjust"``.
    preserve_adjustment :
        If ``True`` (the default), attach the source ``sample`` to the
        returned result via :func:`attach_balance_provenance` under the
        attribute name ``_balance_adjustment``. Lets downstream tooling
        recover the BalanceFrame for diagnostics.
    drop_history :
        Strip ``weight_pre_adjust``, ``weight_adjusted_*``, and
        ``weight_trimmed_*`` columns from the DataFrame before fitting
        (default ``True``). These columns would silently be treated as
        covariates by diff-diff.
    **estimator_kwargs :
        Forwarded to both the estimator's ``__init__`` and its
        ``fit()`` — split by inspecting the ``__init__`` signature.

    Returns
    -------
    A diff-diff results dataclass (e.g. ``CallawaySantAnnaResults``).
    The exact schema depends on the chosen estimator -- inspect the
    returned object's type / dataclass fields, or refer to the upstream
    diff-diff documentation at <https://github.com/igerber/diff-diff>.

    The result also carries ``_balance_adjustment`` (when
    ``preserve_adjustment=True``) for provenance.
    """
    dd = _require_diff_diff()
    if not hasattr(dd, estimator):
        # List the actual class-typed names exposed by the installed
        # diff-diff so the user's error message is grounded in their
        # environment rather than a stale internal-line-number reference.
        valid_estimators: list[str] = sorted(
            name
            for name in getattr(dd, "__all__", dir(dd))
            if not name.startswith("_") and inspect.isclass(getattr(dd, name, None))
        )
        raise ValueError(
            f"Unknown diff-diff estimator: {estimator!r}. Valid estimator "
            f"names exposed by the installed diff-diff are: "
            f"{valid_estimators}. See "
            "<https://github.com/igerber/diff-diff> for the upstream API "
            "documentation."
        )
    cls: Any = getattr(dd, estimator)
    design: Any = to_survey_design(
        sample,
        design_columns=design_columns,
        weight_type=weight_type,
        lonely_psu=lonely_psu,
    )
    # Filter NaN-weight rows before everything else. ``adjust(na_action='drop')``
    # retains rows with missing covariates and sets their active weight to NaN
    # rather than removing them; passing those rows to diff-diff would either
    # raise far from the seam or silently contaminate the fit. ``drop_nan_weight_rows``
    # warns when it removes rows so the loss is visible.
    weights_col: str = active_weight_column(sample)
    pre_history_df: pd.DataFrame = drop_nan_weight_rows(
        sample.df, weights_col, ctx="fit_did"
    )
    # Always hand the estimator a fresh copy: ``drop_history_columns`` already
    # guarantees a copy in BOTH branches (history present → ``df.drop`` returns
    # a copy; history absent → explicit ``.copy()``). The ``drop_history=False``
    # path needs a parallel guarantee, otherwise an estimator that mutates the
    # frame in-place (e.g. by adding a column) would silently corrupt
    # ``Sample.df``. The asymmetric earlier behaviour was flagged by an
    # AI-reviewer warning. Note: ``drop_nan_weight_rows`` returns a fresh
    # DataFrame whenever rows are dropped, so the `.copy()` here is only
    # load-bearing in the no-NaN-rows branch.
    df: pd.DataFrame = (
        drop_history_columns(pre_history_df) if drop_history else pre_history_df.copy()
    )

    # Split estimator_kwargs by inspecting __init__'s signature. We need to
    # also peek at fit() so we can warn about names that exist on BOTH; the
    # `if k in init_params` rule sends overlapping names exclusively to
    # __init__ and silently drops them from fit(), which would surprise a
    # user who passed a fit-time name expecting it to land in fit().
    init_sig: inspect.Signature = inspect.signature(cls.__init__)
    init_params: set[str] = set(init_sig.parameters)
    # Eagerly resolve the fit signature too -- ``cls(**init_kwargs)`` below
    # will not yet exist, but ``cls.fit`` is a regular unbound method on
    # the class, so we can introspect it without instantiating first.
    try:
        fit_method: Any = getattr(cls, "fit", None)
        fit_params_for_overlap: set[str] = (
            set(inspect.signature(fit_method).parameters) if fit_method else set()
        )
    except (TypeError, ValueError):  # pragma: no cover
        fit_params_for_overlap = set()
    overlap: set[str] = set(estimator_kwargs) & init_params & fit_params_for_overlap
    if overlap:
        warnings.warn(
            f"The following kwarg(s) appear in BOTH "
            f"{cls.__name__}.__init__ and {cls.__name__}.fit() signatures: "
            f"{sorted(overlap)}. fit_did routes overlapping names to "
            "__init__ only; the fit() call will not see them. If you "
            "intended these for fit(), pass them via estimator-specific "
            "API directly, or open a feature request to widen the split "
            "rule for this estimator class.",
            UserWarning,
            stacklevel=2,
        )
    init_kwargs: dict[str, object] = {
        k: v for k, v in estimator_kwargs.items() if k in init_params
    }
    fit_kwargs: dict[str, object] = {
        k: v for k, v in estimator_kwargs.items() if k not in init_params
    }
    instance: Any = cls(**init_kwargs)

    # Most fit() signatures take (data, outcome, unit, time, first_treat,
    # covariates=...). DiD/TWFE/MultiPeriodDiD use 'treatment' instead of
    # 'first_treat'. We pass under the most common names; estimators that
    # disagree should be invoked via **estimator_kwargs.
    candidate_fit_kwargs: dict[str, object] = {
        "outcome": outcome,
        "time": time,
        "unit": unit,
        "first_treat": treatment_first,
        "treatment": treatment,
        "covariates": covariates,
        "survey_design": design,
    }
    fit_sig: inspect.Signature = inspect.signature(instance.fit)
    fit_params: set[str] = set(fit_sig.parameters)
    # If the chosen estimator's ``fit()`` does not accept ``survey_design``,
    # the line below will silently drop it -- meaning the
    # carefully-constructed survey design (the whole point of this adapter)
    # is never consumed and the estimator effectively runs unweighted /
    # without the cluster-stratification structure. Warn loudly so the user
    # is not silently misled by an apparently-successful fit.
    if "survey_design" not in fit_params:
        warnings.warn(
            f"{type(instance).__name__}.fit() does not accept "
            "`survey_design`; the SurveyDesign built from balance's weights "
            "will NOT be forwarded to this estimator. The fit will run "
            "without survey-design variance / clustering. If you need a "
            "design-aware estimator, choose one whose fit() takes "
            "`survey_design` (e.g. CallawaySantAnna, StackedDiD, "
            "ImputationDiD).",
            UserWarning,
            stacklevel=2,
        )
    common_fit_kwargs: dict[str, object] = {
        k: v
        for k, v in candidate_fit_kwargs.items()
        if k in fit_params and v is not None
    }
    # Guard against silent design override: this adapter's contract is that
    # the SurveyDesign comes from ``sample.weight_column`` /
    # ``design_columns=``. If a caller routes ``survey_design=...`` via
    # ``**estimator_kwargs`` it would land in ``fit_kwargs`` here and the
    # ``.update()`` below would silently replace the design we just built.
    # Reject that explicitly with a TypeError so the misuse is visible.
    if "survey_design" in fit_kwargs:
        raise TypeError(
            "fit_did(): `survey_design` cannot be passed via "
            "`**estimator_kwargs`. The adapter derives the SurveyDesign from "
            "the Sample's active weight column (and any `design_columns=` "
            "override) so the handoff stays consistent with the balance "
            "Sample. Customise the design via `design_columns=` / "
            "`weight_type=` / `lonely_psu=`, or call `to_survey_design()` and "
            "the chosen estimator directly."
        )
    common_fit_kwargs.update(fit_kwargs)
    results: object = instance.fit(df, **common_fit_kwargs)

    if preserve_adjustment:
        attach_balance_provenance(results, sample)
    return results


def as_balance_diagnostic(
    sample: Sample,
    did_results: object,
) -> dict[str, object]:
    """Combine balance's ASMD / Kish ESS with diff-diff's design-effect /
    event-study output into a single flat diagnostic dict.

    Cross-package diagnostics in one place: balance owns the pre-fit
    ASMD / r-indicator / DEFF view (``stats_and_plots/weights_stats.py``);
    diff-diff owns the post-fit design-effect / per-coefficient SE
    breakdown (``DEFFDiagnostics``, ``SurveyMetadata`` in
    ``survey.py``). Without this helper users rebuild the same dict by
    hand on every notebook.

    Returns
    -------
    dict
        Flat dict suitable for tabulation. Keys cover the most common
        per-fit summary scalars from both packages; missing values are
        ``None`` rather than raising.
    """
    out: dict[str, object] = {}

    # diff-diff side — read defensively via getattr() since the result
    # protocol is structural across ~10 estimator dataclasses. Different
    # estimator classes name the headline ATT differently:
    # CallawaySantAnnaResults exposes ``overall_att`` (used by the new
    # tutorial), while DifferenceInDifferencesResults / TwoWayFixedEffects
    # expose ``att``. Try ``att`` first to preserve the existing contract,
    # then fall back to ``overall_att`` so callers using either result
    # family see a populated key rather than ``None``.
    att_value: object = getattr(did_results, "att", None)
    if att_value is None:
        att_value = getattr(did_results, "overall_att", None)
    out["att"] = att_value
    out["se"] = getattr(did_results, "se", None)
    out["conf_int"] = getattr(did_results, "conf_int", None)
    out["n_obs"] = getattr(did_results, "n_obs", None)

    sm: object | None = getattr(did_results, "survey_metadata", None)
    out["diff_diff_design_effect"] = (
        getattr(sm, "design_effect", None) if sm is not None else None
    )
    out["diff_diff_effective_n"] = (
        getattr(sm, "effective_n", None) if sm is not None else None
    )
    out["diff_diff_sum_weights"] = (
        getattr(sm, "sum_weights", None) if sm is not None else None
    )

    # balance side — diagnostics() returns a DataFrame; we extract the
    # relevant scalars defensively (older balance versions had a
    # slightly different schema).
    diag_df: pd.DataFrame
    try:
        diag_df = sample.diagnostics()
    except Exception as e:  # pragma: no cover
        logger.debug("sample.diagnostics() failed: %s", e)
        diag_df = pd.DataFrame()

    # Schema source of truth: ``summary_utils.py`` builds the diagnostics
    # DataFrame and ``balancedf_class.py`` defines the var names used in the
    # ``weights_diagnostics`` metric. Per-covariate ASMDs are emitted under
    # ``covar_main_asmd_adjusted`` (the post-adjustment view) with one row
    # per covariate plus a final ``mean(asmd)`` summary row.
    out["balance_kish_ess"] = _scalar_from_diag(
        diag_df, "weights_diagnostics", "effective_sample_size"
    )
    out["balance_design_effect"] = _scalar_from_diag(
        diag_df, "weights_diagnostics", "design_effect"
    )
    out["balance_asmd_mean_post"] = _scalar_from_diag(
        diag_df, "covar_main_asmd_adjusted", "mean(asmd)"
    )
    out["balance_asmd_max_post"] = _max_per_covariate_asmd_post(diag_df)
    return out


def _scalar_from_diag(
    diag_df: pd.DataFrame, metric_kind: str, var_name: str
) -> float | None:
    """Pull a single scalar out of balance's diagnostics() DataFrame.

    The diagnostics frame layout is documented in
    ``stats_and_plots/weights_stats.py``. We tolerate schema drift
    across balance versions by returning ``None`` on any miss rather
    than raising — diagnostic dicts should never block the user's
    notebook.
    """
    if diag_df.empty or "metric" not in diag_df.columns:
        return None
    try:
        sub: pd.DataFrame = diag_df[
            (diag_df["metric"] == metric_kind) & (diag_df["var"] == var_name)
        ]
        if len(sub) == 0:
            return None
        return float(sub["val"].iloc[0])
    except Exception:  # pragma: no cover
        return None


def _max_per_covariate_asmd_post(diag_df: pd.DataFrame) -> float | None:
    """Maximum post-adjustment per-covariate ASMD (excluding the summary row).

    Balance's diagnostics DataFrame stores one row per main covariate under
    ``metric == "covar_main_asmd_adjusted"`` plus a final ``mean(asmd)``
    summary row. The "max ASMD" is not pre-computed in the schema, so we
    derive it here -- excluding the summary row -- so that callers get the
    advertised ``balance_asmd_max_post`` field rather than ``None``.
    """
    if diag_df.empty or "metric" not in diag_df.columns:
        return None
    try:
        sub: pd.DataFrame = diag_df[
            (diag_df["metric"] == "covar_main_asmd_adjusted")
            & (diag_df["var"] != "mean(asmd)")
        ]
        if len(sub) == 0:
            return None
        return float(sub["val"].max())
    except Exception:  # pragma: no cover
        return None


__all__: list[str] = [
    "to_survey_design",
    "to_panel_for_did",
    "fit_did",
    "as_balance_diagnostic",
]
