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
from types import ModuleType
from typing import Any

import pandas as pd
from balance.interop._common import (
    active_weight_column,
    attach_balance_provenance,
    drop_history_columns,
)
from balance.interop.conventions import DEFAULT_DESIGN_COLUMNS, WEIGHT_TYPE_DEFAULT
from balance.sample_class import Sample

# @manual=fbsource//third-party/pypi/diff-diff:diff-diff
try:
    import diff_diff as _dd
except ImportError as _e:  # pragma: no cover
    _IMPORT_ERROR: ImportError | None = ImportError(
        "balance.interop.diff_diff requires the diff-diff package "
        "(>=3.3.0,<4). Install via `pip install balance[did]` or "
        f"`pip install diff-diff`. Original error: {_e}"
    )
    _dd = None  # type: ignore[assignment]
else:
    _IMPORT_ERROR = None

logger: logging.Logger = logging.getLogger(__name__)

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
    design_columns: dict[str, str] | None,
) -> dict[str, str]:
    """Merge user-supplied design columns with the conventions defaults.

    ``weights`` is always overridden with the active balance weight
    column name — ``design_columns={"weights": ...}`` is rejected up
    front because balance is the source of truth for which column is
    "live".
    """
    user: dict[str, str] = dict(design_columns or {})
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
    # Start from user-supplied keys only. We INTENTIONALLY do NOT pre-populate
    # from ``DEFAULT_DESIGN_COLUMNS`` here: forwarding ``strata="stratum"`` /
    # ``psu="psu"`` / ``ssu="ssu"`` / ``fpc="fpc"`` to ``SurveyDesign`` when
    # those columns do not exist in ``sample.df`` would produce a survey
    # design that references missing data, leading to errors at variance
    # estimation time or — worse — silently incorrect estimation if diff-diff
    # ever resolves missing columns to NA. We layer in convention defaults
    # ONLY for keys whose target column actually exists in ``sample.df``.
    merged: dict[str, str] = dict(user)
    df_columns: set[str] = set(map(str, sample.df.columns))
    for key, default_col in DEFAULT_DESIGN_COLUMNS.items():
        if key in merged or key == "weights":
            continue
        if key == "weight_type":
            # ``weight_type`` is a literal value (e.g. ``"pweight"``), not a
            # column reference — pass through if the user has not overridden.
            merged[key] = default_col
            continue
        if default_col in df_columns:
            merged[key] = default_col
    merged["weights"] = weights_col
    return merged


def to_survey_design(
    sample: Sample,
    *,
    design_columns: dict[str, str] | None = None,
    weight_type: str = WEIGHT_TYPE_DEFAULT,
    lonely_psu: str = "adjust",
) -> Any:
    """Build a ``diff_diff.SurveyDesign`` from a balance ``Sample``.

    Parameters
    ----------
    sample :
        A balance ``Sample`` (typically post-``adjust()``).
    design_columns :
        Optional mapping of design column NAMES to forward into
        ``SurveyDesign``. Keys are validated against the documented
        ``SurveyDesign`` field set (``survey.py:27-72`` in diff-diff).
        The ``weights`` slot is always sourced from
        ``sample.weight_column`` and may NOT be overridden — this is
        enforced.
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
    merged: dict[str, str] = _resolve_design_columns(sample, design_columns)
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
    design_columns: dict[str, str] | None = None,
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
    df: pd.DataFrame = drop_history_columns(sample.df)
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
    design_columns: dict[str, str] | None = None,
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
    A diff-diff results dataclass (e.g. ``CallawaySantAnnaResults``);
    see ``RESEARCH_diff_diff_api.md`` §1.3 for the schema.

    The result also carries ``_balance_adjustment`` (when
    ``preserve_adjustment=True``) for provenance.
    """
    dd = _require_diff_diff()
    if not hasattr(dd, estimator):
        raise ValueError(
            f"Unknown diff-diff estimator: {estimator!r}. Valid names "
            "are documented in diff_diff.__init__.py:271-288 (see "
            "RESEARCH_diff_diff_api.md §1.1)."
        )
    cls: Any = getattr(dd, estimator)
    design: Any = to_survey_design(
        sample,
        design_columns=design_columns,
        weight_type=weight_type,
        lonely_psu=lonely_psu,
    )
    # Always hand the estimator a fresh copy: ``drop_history_columns`` already
    # guarantees a copy in BOTH branches (history present → ``df.drop`` returns
    # a copy; history absent → explicit ``.copy()``). The ``drop_history=False``
    # path needs a parallel guarantee, otherwise an estimator that mutates the
    # frame in-place (e.g. by adding a column) would silently corrupt
    # ``Sample.df``. The asymmetric earlier behaviour was flagged by an
    # AI-reviewer warning.
    df: pd.DataFrame = (
        drop_history_columns(sample.df) if drop_history else sample.df.copy()
    )

    # Split estimator_kwargs by inspecting __init__'s signature.
    init_sig: inspect.Signature = inspect.signature(cls.__init__)
    init_params: set[str] = set(init_sig.parameters)
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
    common_fit_kwargs: dict[str, object] = {
        k: v
        for k, v in candidate_fit_kwargs.items()
        if k in fit_params and v is not None
    }
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
    # protocol is structural across ~10 estimator dataclasses.
    out["att"] = getattr(did_results, "att", None)
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

    out["balance_kish_ess"] = _scalar_from_diag(diag_df, "weights_diagnostics", "ess")
    out["balance_design_effect"] = _scalar_from_diag(
        diag_df, "weights_diagnostics", "design_effect"
    )
    out["balance_asmd_max_post"] = _scalar_from_diag(
        diag_df, "covariates_asmd", "asmd_post_max"
    )
    out["balance_asmd_mean_post"] = _scalar_from_diag(
        diag_df, "covariates_asmd", "asmd_post_mean"
    )
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


__all__: list[str] = [
    "to_survey_design",
    "to_panel_for_did",
    "fit_did",
    "as_balance_diagnostic",
]
