# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""BalanceFrame: workflow orchestrator for survey/observational data reweighting.

Pairs a responder SampleFrame with a target SampleFrame and exposes an
immutable adjust() method that returns a new, weight-augmented BalanceFrame.
"""

from __future__ import annotations

import collections
import copy
import logging
from copy import deepcopy
from typing import Any, Callable, cast, Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
from balance import util as balance_util
from balance.adjustment import _find_adjustment_method
from balance.csv_utils import to_csv_with_defaults
from balance.sample_frame import SampleFrame
from balance.stats_and_plots import weights_stats
from balance.summary_utils import _build_diagnostics, _build_summary
from balance.typing import FilePathOrBuffer
from balance.util import (
    _assert_type,
    _detect_high_cardinality_features,
    HighCardinalityFeature,
)
from balance.utils.file_utils import _to_download
from balance.weighting_methods.ipw import weights_from_link
from scipy.sparse import spmatrix

if TYPE_CHECKING:
    from typing import Self

    from balance.balancedf_class import BalanceDFSource  # noqa: F401

# The set of string method names accepted by _find_adjustment_method.
_AdjustmentMethodStr = Literal["cbps", "ipw", "null", "poststratify", "rake"]

logger: logging.Logger = logging.getLogger(__package__)


class _CallableBool:
    """A bool-like value that is also callable, for backward-compatible property migration.

    This allows properties like ``has_target`` and ``is_adjusted`` to work
    both as a property and as a method call::

        # Both forms are equivalent:
        if bf.has_target:     # property-style (preferred)
            ...
        if bf.has_target():   # method-call-style (backward compat)
            ...

    This dual-use pattern was introduced so that code written against the
    old ``Sample.has_target()`` method continues to work after the migration
    to a property on ``BalanceFrame``.

    Args:
        value: The boolean value to wrap.

    Examples:
        >>> cb = _CallableBool(True)
        >>> bool(cb)
        True
        >>> cb()
        True
    """

    __slots__ = ("_value",)

    def __init__(self, value: bool) -> None:
        self._value: bool = value

    def __bool__(self) -> bool:
        return self._value

    def __call__(self) -> bool:
        return self._value

    def __repr__(self) -> str:
        return repr(self._value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, bool):
            return self._value == other
        if isinstance(other, _CallableBool):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)

    def __mul__(self, other: object) -> object:
        return self._value * other  # pyre-ignore[58]

    def __rmul__(self, other: object) -> object:
        return other * self._value  # pyre-ignore[58]


class BalanceFrame:
    """A pairing of responder and target SampleFrames for survey weighting.

    BalanceFrame holds two :class:`SampleFrame` instances — *responders*
    (the sample to be reweighted) and *target* (the population benchmark) —
    and provides methods for adjusting responder weights and computing
    diagnostics.

    BalanceFrame is **immutable by convention**: :meth:`adjust` returns a
    *new* BalanceFrame rather than modifying the existing one.  This makes
    it safe to keep a reference to the pre-adjustment state.

    Must be constructed via the public constructor
    ``BalanceFrame(sample=..., target=...)`` which delegates to the
    internal :meth:`_create` factory.

    Attributes:
        responders (SampleFrame): The responder sample.
        target (SampleFrame): The target population.
        is_adjusted (bool): Whether :meth:`adjust` has been called.

    Examples:
        >>> import pandas as pd
        >>> from balance.sample_frame import SampleFrame
        >>> from balance.balance_frame import BalanceFrame
        >>> resp = SampleFrame.from_frame(
        ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 1.0]}))
        >>> tgt = SampleFrame.from_frame(
        ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
        >>> bf = BalanceFrame(sample=resp, target=tgt)
        >>> bf.is_adjusted
        False
        >>> adjusted = bf.adjust(method="ipw")
        >>> adjusted.is_adjusted
        True
        >>> bf.is_adjusted  # original unchanged
        False
    """

    # pyre-fixme[13]: Attributes are initialized in _create() / from_frame()
    _sf_sample_pre_adjust: SampleFrame
    # pyre-fixme[13]: Attributes are initialized in _create() / from_frame()
    _sf_sample: SampleFrame
    # pyre-fixme[13]: Attributes are initialized in _create() / from_frame()
    _sf_target: SampleFrame | None
    # pyre-fixme[13]: Attributes are initialized in _create() / from_frame()
    _adjustment_model: dict[str, Any] | None
    # pyre-fixme[4]: Attributes are initialized in from_frame() / _create()
    # _links is a defaultdict(list) but by convention stores single objects
    # (not lists) for the "target" and "unadjusted" keys.  The defaultdict
    # type is kept for BalanceDF compatibility which expects .get() semantics.
    # Values: _links["target"] → SampleFrame | BalanceFrame
    #         _links["unadjusted"] → BalanceFrame
    _links = None

    def _sync_sampleframe_state_from_responder(self, responder: SampleFrame) -> None:
        """Sync inherited SampleFrame fields from a responder SampleFrame.

        This is only needed when ``self`` is also a ``SampleFrame`` (e.g.
        ``Sample`` via multiple inheritance), so inherited SampleFrame
        properties stay consistent with ``_sf_sample``.
        """
        if isinstance(self, SampleFrame):
            self._df = responder._df
            self._id_column_name = responder._id_column_name
            self._column_roles = responder._column_roles
            self._weight_column_name = responder._weight_column_name
            self._weight_metadata = responder._weight_metadata
            self._df_dtypes = responder._df_dtypes

    @property
    def _df_dtypes(self) -> pd.Series | None:
        """Original dtypes, delegated to ``_sf_sample._df_dtypes``."""
        return self._sf_sample._df_dtypes

    @_df_dtypes.setter
    def _df_dtypes(self, value: pd.Series | None) -> None:
        self._sf_sample._df_dtypes = value

    @property
    def id_column(self) -> pd.Series | None:  # pyre-ignore[3]
        """The id column as a Series, delegated to ``_sf_sample``."""
        return self._sf_sample.id_column

    @property
    def weight_series(self) -> pd.Series | None:  # pyre-ignore[3]
        """The active weight as a Series, delegated to ``_sf_sample``."""
        try:
            return self._sf_sample.weight_series
        except ValueError:
            return None

    # --- Property descriptors backed by _sf_sample ---

    @property
    def _df(self) -> pd.DataFrame:  # pyre-ignore[3]
        """The internal DataFrame, delegated to ``_sf_sample._df``."""
        return self._sf_sample._df

    @_df.setter
    def _df(self, value: pd.DataFrame | None) -> None:  # pyre-ignore[2,3]
        if value is None:
            raise ValueError(
                "Cannot set _df to None. A BalanceFrame must always have a "
                "backing DataFrame."
            )
        self._sf_sample._df = value

    @property
    def _outcome_columns(self) -> pd.DataFrame | None:
        """Outcome columns as a DataFrame, delegated to ``_sf_sample``."""
        outcome_cols = self._sf_sample._column_roles.get("outcomes", [])
        if not outcome_cols:
            return None
        return self._sf_sample._df[outcome_cols]

    @_outcome_columns.setter
    def _outcome_columns(self, value: pd.DataFrame | None) -> None:
        if value is None:
            self._sf_sample._column_roles["outcomes"] = []
        else:
            self._sf_sample._column_roles["outcomes"] = value.columns.tolist()

    @property
    def _ignored_column_names(self) -> list[str]:  # pyre-ignore[3]
        """Ignored column names, delegated to ``_sf_sample.ignored_columns``."""
        return self._sf_sample._column_roles.get("ignored", [])

    @_ignored_column_names.setter
    def _ignored_column_names(
        self, value: list[str] | None
    ) -> None:  # pyre-ignore[2,3]
        self._sf_sample._column_roles["ignored"] = list(value) if value else []

    @property
    def df_ignored(self) -> pd.DataFrame | None:
        """Ignored columns from the responder SampleFrame, or None."""
        return self._sf_sample.df_ignored

    # -----------------------------------------------------------------------
    # Design note: Why __new__ + no-op __init__?
    #
    # copy.deepcopy() allocates the new instance by calling __new__(cls)
    # with NO arguments.  If __init__ held the real construction logic
    # (with required *responders* and *target* parameters), deepcopy would
    # raise TypeError before it could copy any attributes.
    #
    # The solution used here:
    #   - __new__ handles BOTH construction paths:
    #       1. Public constructor: BalanceFrame(sample=sf1, target=sf2)
    #          → validates args and delegates to _create().
    #       2. deepcopy path: BalanceFrame() (no args)
    #          → returns a bare object via object.__new__(cls); deepcopy
    #          then copies attributes onto it directly.
    #   - __init__ is intentionally a no-op (all state is set by _create()).
    # -----------------------------------------------------------------------
    def __new__(
        cls,
        sample: SampleFrame | None = None,
        target: SampleFrame | None = None,
    ) -> BalanceFrame:
        """Create a BalanceFrame from responder and target SampleFrames.

        This uses ``__new__`` so that the natural constructor syntax
        ``BalanceFrame(sample=..., target=...)`` works while still
        routing through the validated :meth:`_create` factory.

        Args:
            sample: The responder / sample data.
            target: The target / population data.

        Returns:
            A new BalanceFrame pairing the two samples.

        Raises:
            TypeError: If *sample* or *target* is not a SampleFrame.
            ValueError: If *sample* and *target* share no covariate
                columns.
        """
        if sample is None:
            # Allow object.__new__(cls) for copy.deepcopy() support.
            if target is None:
                return object.__new__(cls)
            raise TypeError(
                "BalanceFrame requires at least a 'sample' argument. "
                "Usage: BalanceFrame(sample=sf1) or "
                "BalanceFrame(sample=sf1, target=sf2)"
            )
        return cls._create(sample=sample, target=target)

    def __init__(
        self,
        sample: SampleFrame | None = None,
        target: SampleFrame | None = None,
    ) -> None:
        # All initialisation happens in _create(); __init__ is intentionally
        # empty so that __new__ + _create() handles everything.
        pass

    @classmethod
    def _create(
        cls,
        sample: SampleFrame,
        target: SampleFrame | None = None,
    ) -> Self:
        """Internal factory method.

        Validates covariate overlap and builds the BalanceFrame instance.
        Prefer the public constructor ``BalanceFrame(sample=..., target=...)``.

        Args:
            sample: The responder sample.
            target: The target population. If None, creates a target-less
                BalanceFrame that can be completed later via :meth:`set_target`.

        Returns:
            A validated BalanceFrame.

        Raises:
            TypeError: If *sample* or *target* is not a SampleFrame.
            ValueError: If they share no covariate columns.
        """
        if not isinstance(sample, SampleFrame):
            raise TypeError(
                f"'sample' must be a SampleFrame, got {type(sample).__name__}"
            )
        if target is not None and not isinstance(target, SampleFrame):
            raise TypeError(
                f"'target' must be a SampleFrame, got {type(target).__name__}"
            )

        instance = object.__new__(cls)
        instance._sf_sample_pre_adjust = sample
        instance._sf_sample = sample  # same object initially
        instance._sf_target = target
        instance._adjustment_model = None
        instance._links = collections.defaultdict(list)
        if target is not None:
            instance._links["target"] = target

        # When the instance is also a SampleFrame (e.g., Sample inherits
        # from both BalanceFrame and SampleFrame), copy SampleFrame state
        # so that inherited SampleFrame properties work on the instance.
        instance._sync_sampleframe_state_from_responder(sample)

        # Validate covariate overlap using public properties
        if target is not None:
            cls._validate_covariate_overlap(sample, target)

        return instance

    @staticmethod
    def _validate_covariate_overlap(
        responders: SampleFrame, target: SampleFrame
    ) -> None:
        """Check that responders and target share at least one covariate.

        When both have no covariates (outcome-only comparison), a warning
        is issued instead of raising.

        Raises:
            ValueError: If both have covariates but share none.
        """
        resp_covars = set(responders.covar_columns)
        target_covars = set(target.covar_columns)
        overlap = resp_covars & target_covars
        if len(overlap) == 0:
            if len(resp_covars) == 0 and len(target_covars) == 0:
                # Both have no covariates — legitimate for outcome-only use.
                logger.warning(
                    "Both responders and target have no covariate columns. "
                    "adjust() will not be available."
                )
                return
            raise ValueError(
                "Responders and target share no covariate columns. "
                f"Responder covariates: {sorted(resp_covars)}, "
                f"target covariates: {sorted(target_covars)}"
            )
        if overlap != resp_covars or overlap != target_covars:
            logger.warning(
                "Responders and target have different covariate columns. "
                f"Using {len(overlap)} common variable(s): {sorted(overlap)}. "
                f"Responder-only: {sorted(resp_covars - overlap)}, "
                f"target-only: {sorted(target_covars - overlap)}."
            )

    # --- Properties ---

    @property
    def df_responders(self) -> pd.DataFrame:
        """The responder data as a DataFrame."""
        return self._sf_sample.df

    @property
    def df_target(self) -> pd.DataFrame | None:
        """The target data as a DataFrame, or None if not yet set."""
        if self._sf_target is None:
            return None
        return self._sf_target.df

    @property
    def df_responders_unadjusted(self) -> pd.DataFrame:
        """The original (pre-adjustment) responder data as a DataFrame."""
        return self._sf_sample_pre_adjust.df

    # --- Backward-compat aliases (to be removed in a future diff) ---

    @property
    def responders(self) -> SampleFrame:
        """Alias for ``_sf_sample`` (backward compat, will be removed)."""
        return self._sf_sample

    @property
    def target(self) -> SampleFrame | None:
        """Alias for ``_sf_target`` (backward compat, will be removed)."""
        return self._sf_target

    @property
    def unadjusted(self) -> SampleFrame | None:
        """Alias for ``_sf_sample_pre_adjust`` if adjusted, else None (backward compat)."""
        if self.is_adjusted:
            return self._sf_sample_pre_adjust
        return None

    @property
    def has_target(self) -> _CallableBool:
        """Whether this BalanceFrame has a target population set.

        Returns a dual-use ``_CallableBool``: both ``bf.has_target`` and
        ``bf.has_target()`` work (the latter for backward compatibility).

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 1.0]}))
            >>> bf = BalanceFrame(sample=resp)
            >>> bf.has_target
            False
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
            >>> bf.set_target(tgt)
            >>> bf.has_target
            True
        """
        return _CallableBool(
            self._sf_target is not None
            or (self._links is not None and "target" in self._links)
        )

    def set_target(
        self, target: BalanceFrame | SampleFrame, in_place: bool | None = None
    ) -> Self:
        """Set or replace the target population.

        When *target* is a BalanceFrame (or subclass such as Sample), a deep
        copy of ``self`` is returned with the target set (immutable pattern).
        When *target* is a raw SampleFrame, the behaviour depends on
        *in_place*: True mutates self, False returns a new BalanceFrame.

        Args:
            target: The target population — a BalanceFrame/Sample or a
                SampleFrame.
            in_place: If True, mutates self (only valid for SampleFrame
                targets). If False, returns a new copy. Defaults to None
                which auto-selects: copy for BalanceFrame targets, in-place
                for SampleFrame targets.

        Returns:
            BalanceFrame with the new target set.

        Raises:
            TypeError / ValueError: If *target* is not a BalanceFrame or
                SampleFrame, or if they share no covariate columns.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
            >>> bf = BalanceFrame(sample=resp)
            >>> bf.set_target(tgt)
            >>> bf.has_target()
            True
        """
        if isinstance(target, BalanceFrame):
            # BalanceFrame / Sample path: return a deep copy (immutable)
            new_copy = deepcopy(self)
            new_copy._links["target"] = target
            BalanceFrame._validate_covariate_overlap(
                new_copy._sf_sample, target._sf_sample
            )
            new_copy._sf_target = target._sf_sample
            return new_copy

        if isinstance(target, SampleFrame):
            # SampleFrame path: default in_place=True for backward compat
            if in_place is None:
                in_place = True
            BalanceFrame._validate_covariate_overlap(self._sf_sample, target)

            if in_place:
                if self.is_adjusted:
                    logger.warning(
                        "Replacing target on an adjusted object resets responder "
                        "weights to pre-adjust values and discards current "
                        "adjustment results. Pass in_place=False to return a new "
                        "object and keep the current adjusted state on this "
                        "instance."
                    )
                self._sf_target = target
                self._links["target"] = target
                # Reset adjustment state — old adjustment is no longer valid.
                self._sf_sample = self._sf_sample_pre_adjust
                self._adjustment_model = None
                self._links.pop("unadjusted", None)
                self._sync_sampleframe_state_from_responder(self._sf_sample)
                return self
            else:
                return type(self)._create(
                    sample=copy.deepcopy(self._sf_sample_pre_adjust),
                    target=target,
                )

        raise TypeError("A target, a Sample object, must be specified")

    def set_as_pre_adjust(self, *, in_place: bool = False) -> Self:
        """Set the current responder state as the new pre-adjust baseline.

        This "locks in" the current responder weights (which may already be
        adjusted and/or trimmed) as the baseline for future diagnostics and
        subsequent adjustments.

        Args:
            in_place: If True, mutate this object and return it. If False
                (default), return a new object with a deep-copied responder
                frame and reset baseline.

        Returns:
            BalanceFrame with ``_sf_sample_pre_adjust`` reset to the current
            responder SampleFrame state. In copy mode (``in_place=False``),
            only the responder frame is deep-copied and used to construct a new
            object (the full ``_links`` graph is not deep-copied). In in-place
            mode, the baseline is set to the existing responder
            frame object so baseline/current share identity, matching
            unadjusted-object semantics elsewhere in the API.
            Any current adjustment model is cleared because the object is no
            longer considered adjusted after this operation.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
            >>> adjusted = BalanceFrame(sample=resp, target=tgt).adjust(method="null")
            >>> baseline_locked = adjusted.set_as_pre_adjust()  # copy mode
            >>> baseline_locked.is_adjusted
            False
            >>> _ = adjusted.set_as_pre_adjust(in_place=True)  # in-place mode
        """
        if in_place:
            bf = self
            frozen = bf._sf_sample
        else:
            frozen = copy.deepcopy(self._sf_sample)
            bf = type(self)._create(sample=frozen, target=self._sf_target)
            # Preserve a richer target link (e.g., BalanceFrame/Sample object)
            # when present on the original.
            if "target" in self._links:
                bf._links["target"] = self._links["target"]
        bf._sf_sample_pre_adjust = frozen
        bf._sf_sample = frozen
        bf._adjustment_model = None
        bf._links.pop("unadjusted", None)
        bf._sync_sampleframe_state_from_responder(frozen)
        return bf

    @property
    def is_adjusted(self) -> _CallableBool:
        """Whether this BalanceFrame has been adjusted.

        Returns a ``_CallableBool`` so both ``bf.is_adjusted`` (property)
        and ``bf.is_adjusted()`` (legacy call) work.

        For compound adjustments (calling ``adjust()`` multiple times),
        ``is_adjusted`` is True after the first adjustment and remains True
        for all subsequent adjustments.  The original unadjusted baseline is
        always preserved in ``_sf_sample_pre_adjust``.
        """
        return _CallableBool(self._sf_sample is not self._sf_sample_pre_adjust)

    # --- Adjustment ---

    def _resolve_adjustment_function(
        self, method: str | Callable[..., Any]
    ) -> Callable[..., Any]:
        """Resolve a weighting method string or callable to a function.

        Args:
            method: A string naming a built-in method or a callable.

        Returns:
            The resolved adjustment function.

        Raises:
            ValueError: If *method* is not a valid string or callable.
        """
        if isinstance(method, str):
            return _find_adjustment_method(cast(_AdjustmentMethodStr, method))
        if callable(method):
            return method
        raise ValueError(
            "'method' must be a string naming a weighting method or a callable"
        )

    def _get_covars(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get covariate DataFrames for responders and target.

        Returns:
            A (responder_covars, target_covars) tuple of DataFrames.

        Raises:
            ValueError: If no target is set.
        """
        if self._sf_target is None:
            raise ValueError("Cannot get covars without a target population.")
        return self._sf_sample.df_covars, self._sf_target.df_covars

    def _build_adjusted_frame(
        self,
        result: dict[str, Any],
        method: str | Callable[..., Any],
    ) -> Self:
        """Construct a new BalanceFrame with adjusted weights.

        Args:
            result: The dict returned by the weighting function, containing
                at least ``"weight"`` and optionally ``"model"``.
            method: The original method argument (string or callable).

        Returns:
            A new, adjusted BalanceFrame (or subclass instance if called on
            a subclass).
        """
        new_responders = copy.deepcopy(self._sf_sample)
        method_name = (
            method
            if isinstance(method, str)
            else getattr(method, "__name__", str(method))
        )

        # --- Unified weight history tracking ---
        #
        # After each adjustment the SampleFrame accumulates weight columns:
        #
        #   | After        | Weight columns                                    | Active   |
        #   |--------------|---------------------------------------------------|----------|
        #   | Before adj.  | weight                                            | weight   |
        #   | 1st adjust   | weight, weight_pre_adjust, weight_adjusted_1      | weight   |
        #   | 2nd adjust   | weight, weight_pre_adjust, weight_adjusted_1, _2  | weight   |
        #   | 3rd adjust   | ... weight_adjusted_1, _2, _3                     | weight   |
        #
        # * weight_pre_adjust — frozen copy of original design weights (first adj only)
        # * weight_adjusted_N — output of the Nth adjustment step
        # * weight — always overwritten with the latest adjusted values
        original_weight_name = str(_assert_type(self.weight_series).name)

        # On first adjustment: freeze the original design weights as
        # "weight_pre_adjust" so the full history is in one SampleFrame.
        if "weight_pre_adjust" not in new_responders._df.columns:
            new_responders.add_weight_column(
                "weight_pre_adjust",
                new_responders._df[original_weight_name].copy(),
            )

        # Find next global action number (shared counter across adjusted/trimmed).
        n = new_responders._next_weight_action_number()
        adj_col_name = f"weight_adjusted_{n}"

        # Add the new adjusted weights as weight_adjusted_N
        new_responders.add_weight_column(
            adj_col_name,
            result["weight"],
            metadata={
                "method": method_name,
                "adjusted": True,
                "model": result.get("model", {}),
            },
        )

        # Overwrite the original weight column with the new adjusted values,
        # so the active weight column always keeps its original name.
        # use_index=True lets na_action="drop" (which returns fewer weights)
        # fill dropped rows with NaN; set_weights warns about missing indices.
        new_responders.set_weights(result["weight"], use_index=True)

        # TODO: The weight history columns (weight_pre_adjust, weight_adjusted_1,
        # weight_adjusted_2, ...) make _sf_sample_pre_adjust redundant.  Once all
        # consumers are updated to read weight_pre_adjust instead,
        # _sf_sample_pre_adjust can be removed entirely.

        # Use type(self) so subclasses (e.g. Sample) get their own type back.
        new_bf = type(self)._create(
            sample=new_responders,
            target=self._sf_target,
        )
        # Point _sf_sample_pre_adjust to the original (pre-adjustment) data.
        # For compound adjustments this is always the *very first* baseline,
        # so diagnostics (asmd_improvement, summary) show total improvement.
        new_bf._sf_sample_pre_adjust = self._sf_sample_pre_adjust
        # Always link back to the original unadjusted BalanceFrame so that
        # 3-way comparisons (adjusted vs original vs target) span the full
        # adjustment chain, not just the last step.
        if "unadjusted" in self._links:
            new_bf._links["unadjusted"] = self._links["unadjusted"]
        else:
            new_bf._links["unadjusted"] = self
        if "target" in self._links:
            new_bf._links["target"] = self._links["target"]

        raw_model = result.get("model")
        # Defensive copy: the weighting function may retain a reference to the
        # dict it returned, so mutating it here could cause surprising side effects.
        # TODO: Track adjustment history — currently only the latest model is
        # stored. A future enhancement should maintain a list of
        # (method, model_dict) tuples for each adjustment step.
        new_bf._adjustment_model = (
            dict(raw_model) if isinstance(raw_model, dict) else raw_model
        )
        # Preserve the raw model's method name (e.g. "null_adjustment") when
        # present; only set a fallback when the model doesn't include one.
        if isinstance(new_bf._adjustment_model, dict):
            new_bf._adjustment_model.setdefault("method", method_name)
            if new_bf._adjustment_model.get("method") == "ipw":
                # Preserve training-time design weights so predict_weights can
                # reproduce IPW weights from fitted links.
                fit_sample_weights = new_bf._adjustment_model.get("fit_sample_weights")
                new_bf._adjustment_model.setdefault(
                    "training_sample_weights",
                    (
                        fit_sample_weights
                        if isinstance(fit_sample_weights, pd.Series)
                        else self._sf_sample.df_weights.iloc[:, 0].copy()
                    ),
                )
                if self._sf_target is not None:
                    fit_target_weights = new_bf._adjustment_model.get(
                        "fit_target_weights"
                    )
                    new_bf._adjustment_model.setdefault(
                        "training_target_weights",
                        (
                            fit_target_weights
                            if isinstance(fit_target_weights, pd.Series)
                            else self._sf_target.df_weights.iloc[:, 0].copy()
                        ),
                    )
        return new_bf

    def adjust(
        self,
        target: BalanceFrame | None = None,
        method: str | Callable[..., Any] = "ipw",
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Adjust responder weights to match the target. Returns a NEW BalanceFrame.

        The original BalanceFrame is not modified (immutable pattern).  The
        returned BalanceFrame has ``is_adjusted == True`` and the pre-adjustment
        responders stored in :attr:`unadjusted`.

        The active weight column always keeps its original name (e.g.,
        ``"weight"``).  Its values are overwritten with the new adjusted
        weights.  The full weight history is tracked via additional columns:

        .. list-table:: Weight columns after each adjustment
           :header-rows: 1

           * - After
             - Weight columns in ``responders``
             - Active (``"weight"``)
           * - Before adjust
             - ``weight``
             - original design weights
           * - 1st adjust
             - ``weight``, ``weight_pre_adjust``, ``weight_adjusted_1``
             - = ``weight_adjusted_1`` values
           * - 2nd adjust
             - + ``weight_adjusted_2``
             - = ``weight_adjusted_2`` values
           * - 3rd adjust
             - + ``weight_adjusted_3``
             - = ``weight_adjusted_3`` values

        **Compound / sequential adjustments:** ``adjust()`` can be called
        multiple times.  Each call uses the *current* (previously adjusted)
        weights as design weights, so adjustments compound.  For example, run
        IPW first to correct broad imbalances, then rake on a specific variable
        for fine-tuning::

            adjusted_ipw = bf.adjust(method="ipw", max_de=2)
            adjusted_final = adjusted_ipw.adjust(method="rake")

        The original unadjusted baseline is always preserved:

        * ``_sf_sample_pre_adjust`` always points to the **original**
          (pre-first-adjustment) SampleFrame.
        * ``_links["unadjusted"]`` always points to the **original**
          unadjusted BalanceFrame, so 3-way comparisons
          (adjusted vs original vs target) and ``asmd_improvement()`` show
          **total** improvement across all adjustment steps.
        * ``model`` stores only the **latest** adjustment's model dict.

        Args:
            target: Optional target BalanceFrame/Sample. If provided, calls
                ``set_target(target)`` first, then adjusts. If None, uses the
                already-set target.
            method: The weighting method to use.  Built-in options:
                ``"ipw"``, ``"cbps"``, ``"rake"``, ``"poststratify"``,
                ``"null"``.  A callable with the same signature as the
                built-in methods is also accepted.
            *args: Positional arguments (forwarded on recursive call only).
            **kwargs: Additional keyword arguments forwarded to the adjustment
                function (e.g. ``max_de``, ``transformations``).

        Returns:
            A new, adjusted BalanceFrame.

        Raises:
            ValueError: If *method* is a string that doesn't match any
                registered adjustment method.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2, 3], "x": [10.0, 20.0, 30.0],
            ...                   "weight": [1.0, 1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [4, 5, 6], "x": [15.0, 25.0, 35.0],
            ...                   "weight": [1.0, 1.0, 1.0]}))
            >>> bf = BalanceFrame(sample=resp, target=tgt)
            >>> adjusted = bf.adjust(method="ipw")
            >>> adjusted.is_adjusted
            True
            >>> adjusted2 = adjusted.adjust(method="null")
            >>> adjusted2.is_adjusted
            True
        """
        if target is not None:
            # Inline target: set it first, then recurse
            self_with_target = self.set_target(target)
            return self_with_target.adjust(*args, method=method, **kwargs)

        self._require_target()

        sf_target = self._sf_target
        assert sf_target is not None  # guaranteed by _require_target() above

        adjustment_function = self._resolve_adjustment_function(method)
        resp_covars, target_covars = self._get_covars()

        # Detect high-cardinality features in both responder and target covariates
        num_rows_sample = resp_covars.shape[0]
        num_rows_target = target_covars.shape[0]
        if (
            num_rows_sample > 0
            and num_rows_target > 100_000
            and num_rows_target >= 10 * num_rows_sample
        ):
            logger.warning(
                "Large target detected: %s target rows vs %s sample rows. "
                "When the target is much larger than the sample (here >10x and >100k rows), "
                "the target's contribution to variance becomes negligible. "
                "Standard errors will be driven almost entirely by the sample, "
                "similar to a one-sample inference setting.",
                num_rows_target,
                num_rows_sample,
            )

        sample_high_card = _detect_high_cardinality_features(resp_covars)
        target_high_card = _detect_high_cardinality_features(target_covars)

        # Merge the results, taking the maximum unique_count for each column
        high_cardinality_dict: dict[str, HighCardinalityFeature] = {}
        for feature in sample_high_card + target_high_card:
            if (
                feature.column not in high_cardinality_dict
                or feature.unique_count
                > high_cardinality_dict[feature.column].unique_count
            ):
                high_cardinality_dict[feature.column] = feature

        high_cardinality_features = sorted(
            high_cardinality_dict.values(),
            key=lambda f: f.unique_count,
            reverse=True,
        )

        if high_cardinality_features:
            formatted_details = ", ".join(
                f"{feature.column} (unique={feature.unique_count}; "
                f"unique_ratio={feature.unique_ratio:.2f}"
                f"{'; missing values present' if feature.has_missing else ''}"
                f")"
                for feature in high_cardinality_features
            )
            logger.warning(
                "High-cardinality features detected that may not provide signal: "
                + formatted_details
            )

        result = adjustment_function(
            sample_df=resp_covars,
            sample_weights=self._sf_sample.df_weights.iloc[:, 0],
            target_df=target_covars,
            target_weights=sf_target.df_weights.iloc[:, 0],
            **kwargs,
        )

        return self._build_adjusted_frame(result, method)

    def fit(
        self,
        target: BalanceFrame | None = None,
        method: str | Callable[..., Any] = "ipw",
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Sklearn-style alias for :meth:`adjust`.

        In ``balance``, fitting the survey-weight model also produces the adjusted
        sample weights, so ``fit(...)`` maps directly to ``adjust(...)`` and
        returns an adjusted object.

        Args:
            target: Optional target population to set before fitting. If
                provided, this method behaves like ``set_target(target)``
                followed by fitting.
            method: Adjustment method name (``"ipw"``, ``"cbps"``, ``"rake"``,
                ``"poststratify"``, ``"null"``) or a custom callable with the
                weighting-method signature.
            *args: Positional arguments forwarded to :meth:`adjust`.
            **kwargs: Keyword arguments forwarded to :meth:`adjust`.

        Returns:
            A new adjusted BalanceFrame (or subclass instance).

        Raises:
            ValueError: If no target is available and none is provided, or if
                ``method`` is invalid.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [0.0, 1.0], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [0.2, 0.8], "weight": [1.0, 1.0]}))
            >>> adjusted = BalanceFrame(sample=resp, target=tgt).fit(method="null")
            >>> bool(adjusted.is_adjusted)
            True
        """
        resolved_method = self._resolve_adjustment_function(method)
        if getattr(resolved_method, "__name__", None) == "ipw":
            kwargs.setdefault("store_fit_matrices", True)
        return self.adjust(target=target, method=method, *args, **kwargs)

    def fit_transform(
        self,
        target: BalanceFrame | None = None,
        method: str | Callable[..., Any] = "ipw",
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Sklearn-style alias combining ``fit`` and ``transform`` in one step.

        For weighting methods this is equivalent to :meth:`fit` / :meth:`adjust`,
        because weights are fitted and applied in a single operation.

        Args:
            target: Optional target population to set before fitting.
            method: Adjustment method name or callable.
            *args: Positional arguments forwarded to :meth:`fit`.
            **kwargs: Keyword arguments forwarded to :meth:`fit`.

        Returns:
            A new adjusted BalanceFrame.

        Raises:
            ValueError: If no target is available and none is provided, or if
                ``method`` is invalid.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [0.0, 1.0], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [0.2, 0.8], "weight": [1.0, 1.0]}))
            >>> adjusted = BalanceFrame(sample=resp, target=tgt).fit_transform(method="null")
            >>> bool(adjusted.is_adjusted)
            True
        """
        return self.fit(target=target, method=method, *args, **kwargs)

    def _require_ipw_model(self) -> dict[str, Any]:
        self._require_adjusted()
        model = _assert_type(self._adjustment_model)
        if not isinstance(model, dict) or model.get("method") != "ipw":
            raise ValueError(
                "predict/transform currently support only IPW-adjusted objects."
            )
        fit = model.get("fit")
        columns = model.get("X_matrix_columns")
        if fit is None or not isinstance(columns, list):
            raise ValueError("IPW model metadata is missing fitted model information.")
        return model

    def _matrix_to_dataframe(
        self,
        matrix: Any,
        index: pd.Index,
        columns: list[str],
    ) -> pd.DataFrame:
        if isinstance(matrix, pd.DataFrame):
            return matrix.reindex(index=index, columns=columns)
        if isinstance(matrix, np.ndarray):
            return pd.DataFrame(matrix, index=index, columns=columns)
        if isinstance(matrix, spmatrix):
            return pd.DataFrame.sparse.from_spmatrix(
                matrix,
                index=index,
                columns=columns,
            )
        raise ValueError(
            "Stored IPW fit-time model matrix is unavailable for this configuration."
        )

    def transform(
        self,
        on: Literal["sample", "target", "both"] = "both",
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Return fitted-model feature matrices (IPW only).

        This applies the same model-matrix transformation to sample/target
        covariates and aligns columns to the fitted IPW model.

        Args:
            on: Which population to transform. ``"sample"`` returns the responder
                matrix, ``"target"`` returns the target matrix, and ``"both"``
                returns ``(sample_matrix, target_matrix)``.

        Returns:
            A transformed model matrix DataFrame, or a tuple of two DataFrames
            when ``on="both"``.

        Raises:
            ValueError: If the object is not IPW-adjusted, if target is missing
                for ``on in {"target", "both"}``, or if ``on`` is invalid.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [0.0, 1.0], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [0.2, 0.8], "weight": [1.0, 1.0]}))
            >>> adjusted = BalanceFrame(sample=resp, target=tgt).fit(method="ipw")
            >>> x_s, x_t = adjusted.transform(on="both")
            >>> x_s.shape[0], x_t.shape[0]
            (2, 2)
        """
        model = self._require_ipw_model()
        columns = _assert_type(model.get("X_matrix_columns"))
        sample_idx = pd.Index(model.get("sample_index", self._sf_sample.df.index))
        sample_matrix = model.get("model_matrix_sample")
        if sample_matrix is None:
            raise ValueError(
                "IPW model is missing fit-time matrices. transform() cannot "
                "reconstruct arbitrary preprocessing reliably."
            )
        sample_df = self._matrix_to_dataframe(
            sample_matrix,
            sample_idx,
            columns,
        ).reindex(self._sf_sample.df.index)
        if on == "sample":
            return sample_df
        if on == "target":
            self._require_target()
            target_idx = pd.Index(
                model.get("target_index", _assert_type(self._sf_target).df.index)
            )
            target_matrix = model.get("model_matrix_target")
            if target_matrix is None:
                raise ValueError(
                    "IPW model is missing fit-time target matrix for transform()."
                )
            return self._matrix_to_dataframe(
                target_matrix,
                target_idx,
                columns,
            ).reindex(_assert_type(self._sf_target).df.index)
        if on == "both":
            self._require_target()
            target_idx = pd.Index(
                model.get("target_index", _assert_type(self._sf_target).df.index)
            )
            target_matrix = model.get("model_matrix_target")
            if target_matrix is None:
                raise ValueError(
                    "IPW model is missing fit-time target matrix for transform()."
                )
            return (
                sample_df,
                self._matrix_to_dataframe(
                    target_matrix,
                    target_idx,
                    columns,
                ).reindex(_assert_type(self._sf_target).df.index),
            )
        raise ValueError("on must be one of: 'sample', 'target', 'both'")

    def predict(
        self,
        on: Literal["sample", "target", "both"] = "target",
        output: Literal["probability", "link"] = "probability",
    ) -> pd.Series | tuple[pd.Series, pd.Series]:
        """Predict IPW propensity outputs on sample/target covariates.

        Args:
            on: Which population to predict on (``"sample"``, ``"target"``,
                or ``"both"``).
            output: Output scale. ``"probability"`` returns class-1 propensity
                probabilities. ``"link"`` returns logit-transformed values.

        Returns:
            A prediction Series, or a tuple of two Series when ``on="both"``.

        Raises:
            ValueError: If the object is not IPW-adjusted, if target is missing
                for ``on in {"target", "both"}``, or if ``on`` is invalid.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [0.0, 1.0], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [0.2, 0.8], "weight": [1.0, 1.0]}))
            >>> adjusted = BalanceFrame(sample=resp, target=tgt).fit(method="ipw")
            >>> p = adjusted.predict(on="target", output="probability")
            >>> int(p.shape[0])
            2
        """
        if output not in ("probability", "link"):
            raise ValueError("output must be one of: 'probability', 'link'")
        model = self._require_ipw_model()
        if output == "probability":
            sample_values = model.get("sample_probability")
            target_values = model.get("target_probability")
        else:
            sample_values = model.get("sample_link")
            target_values = model.get("target_link")
        if not isinstance(sample_values, np.ndarray):
            raise ValueError(
                "IPW model is missing fit-time sample predictions for predict()."
            )
        sample_idx = pd.Index(model.get("sample_index", self._sf_sample.df.index))
        sample_series = pd.Series(sample_values, index=sample_idx).reindex(
            self._sf_sample.df.index
        )
        if on == "sample":
            return sample_series
        if on == "target":
            self._require_target()
            if not isinstance(target_values, np.ndarray):
                raise ValueError(
                    "IPW model is missing fit-time target predictions for predict()."
                )
            target_idx = pd.Index(
                model.get("target_index", _assert_type(self._sf_target).df.index)
            )
            return pd.Series(target_values, index=target_idx).reindex(
                _assert_type(self._sf_target).df.index
            )
        if on == "both":
            self._require_target()
            if not isinstance(target_values, np.ndarray):
                raise ValueError(
                    "IPW model is missing fit-time target predictions for predict()."
                )
            target_idx = pd.Index(
                model.get("target_index", _assert_type(self._sf_target).df.index)
            )
            return (
                sample_series,
                pd.Series(target_values, index=target_idx).reindex(
                    _assert_type(self._sf_target).df.index
                ),
            )
        raise ValueError("on must be one of: 'sample', 'target', 'both'")

    def predict_weights(self) -> pd.Series:
        """Predict responder weights from a fitted IPW model.

        Uses stored fit-time IPW metadata (links, class balancing, trimming,
        and design weights) when available to reproduce fitted responder
        weights.

        Returns:
            A Series of predicted responder weights.

        Raises:
            ValueError: If the object is not IPW-adjusted or has no target.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [0.0, 1.0], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [0.2, 0.8], "weight": [1.0, 1.0]}))
            >>> adjusted = BalanceFrame(sample=resp, target=tgt).fit(method="ipw")
            >>> w = adjusted.predict_weights()
            >>> int(w.shape[0])
            2
        """
        self._require_target()
        model = self._require_ipw_model()
        model_link = model.get("sample_link")
        if isinstance(model_link, np.ndarray):
            link = model_link
        else:
            link = _assert_type(self.predict(on="sample", output="link")).to_numpy()
        sample_weights = model.get("training_sample_weights")
        target_weights = model.get("training_target_weights")
        if not isinstance(sample_weights, pd.Series):
            sample_weights = self._sf_sample.df_weights.iloc[:, 0]
        if not isinstance(target_weights, pd.Series):
            target_weights = _assert_type(self._sf_target).df_weights.iloc[:, 0]

        predicted = weights_from_link(
            link=link,
            balance_classes=bool(model.get("balance_classes", True)),
            sample_weights=sample_weights,
            target_weights=target_weights,
            weight_trimming_mean_ratio=model.get("weight_trimming_mean_ratio"),
            weight_trimming_percentile=model.get("weight_trimming_percentile"),
        )
        sample_idx = pd.Index(model.get("sample_index", sample_weights.index))
        weight_name = getattr(_assert_type(self.weight_series), "name", None)
        return (
            pd.Series(predicted.values, index=sample_idx)
            .reindex(self._sf_sample.df.index)
            .rename(weight_name)
        )

    @property
    def model(self) -> dict[str, Any] | None:
        """The adjustment model dictionary, or None if not adjusted.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
            >>> bf = BalanceFrame(sample=resp, target=tgt)
            >>> bf.model is None
            True
        """
        return self._adjustment_model

    # --- Conversion ---

    @classmethod
    def from_sample(cls, sample: Any) -> BalanceFrame:
        """Convert a :class:`~balance.sample_class.Sample` to a BalanceFrame.

        The Sample must have a target set (via ``Sample.set_target``).  If
        the Sample is adjusted, the adjustment state (unadjusted responders,
        model) is preserved.

        Args:
            sample: A :class:`~balance.sample_class.Sample` instance with
                a target.

        Returns:
            BalanceFrame: A new BalanceFrame mirroring the Sample's data,
                target, and adjustment state.

        Raises:
            TypeError: If *sample* is not a Sample instance.
            ValueError: If *sample* does not have a target set.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_class import Sample
            >>> from balance.balance_frame import BalanceFrame
            >>> s = Sample.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 1.0]}))
            >>> t = Sample.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
            >>> bf = BalanceFrame.from_sample(s.set_target(t))
            >>> bf.is_adjusted
            False
        """
        # Lazy import: sample_class ↔ balance_frame have a circular dependency.
        from balance.sample_class import Sample

        if not isinstance(sample, Sample):
            raise TypeError(
                f"'sample' must be a Sample instance, got {type(sample).__name__}"
            )
        if not sample.has_target():
            raise ValueError(
                "Sample must have a target set. "
                "Use sample.set_target(target) before calling BalanceFrame.from_sample()."
            )

        responders_sf = SampleFrame.from_sample(sample)
        target_sf = SampleFrame.from_sample(sample._links["target"])

        bf = cls._create(sample=responders_sf, target=target_sf)

        if sample.is_adjusted():
            # Set unadjusted to a DIFFERENT SampleFrame so is_adjusted returns True
            bf._sf_sample_pre_adjust = SampleFrame.from_sample(
                sample._links["unadjusted"]
            )
            bf._adjustment_model = sample.model

        return bf

    def to_sample(self) -> Any:
        """Convert this BalanceFrame back to a :class:`~balance.sample_class.Sample`.

        Reconstructs a Sample with the responder data and target set.  If
        this BalanceFrame is adjusted, the returned Sample will also be
        adjusted — ``is_adjusted()`` returns True, ``has_target()`` returns
        True, and the original (unadjusted) weights are preserved via the
        ``"unadjusted"`` link.

        Returns:
            Sample: A Sample mirroring this BalanceFrame's data, target,
                and adjustment state.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2, 3], "x": [10.0, 20.0, 30.0],
            ...                   "weight": [1.0, 1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [4, 5, 6], "x": [15.0, 25.0, 35.0],
            ...                   "weight": [1.0, 1.0, 1.0]}))
            >>> bf = BalanceFrame(sample=resp, target=tgt)
            >>> s = bf.to_sample()
            >>> s.has_target()
            True
        """
        # Lazy import: sample_class ↔ balance_frame have a circular dependency.
        from balance.sample_class import Sample

        target = self._sf_target
        if target is None:
            raise ValueError(
                "Cannot convert to Sample: BalanceFrame has no target set."
            )

        resp_sample = Sample.from_frame(
            self._sf_sample._df,
            id_column=self._sf_sample.id_column_name,
            weight_column=self._sf_sample.weight_column,
            outcome_columns=self._sf_sample.outcome_columns or None,
            ignored_columns=self._sf_sample.ignored_columns or None,
            standardize_types=False,
        )
        target_sample = Sample.from_frame(
            target._df,
            id_column=target.id_column_name,
            weight_column=target.weight_column,
            outcome_columns=target.outcome_columns or None,
            ignored_columns=target.ignored_columns or None,
            standardize_types=False,
        )
        result = resp_sample.set_target(target_sample)

        if self.is_adjusted and self._sf_sample_pre_adjust is not None:
            unadj_sf = SampleFrame.from_frame(
                self._sf_sample_pre_adjust._df,
                id_column=self._sf_sample_pre_adjust.id_column_name,
                weight_column=self._sf_sample_pre_adjust.weight_column,
                outcome_columns=self._sf_sample_pre_adjust.outcome_columns or None,
                ignored_columns=self._sf_sample_pre_adjust.ignored_columns or None,
                standardize_types=False,
            )
            # pyre-ignore[16]: Sample gains this attr via BalanceFrame inheritance (diff 14.3)
            result._sf_sample_pre_adjust = unadj_sf
            # pyre-ignore[16]: Sample gains _links via BalanceFrame inheritance (diff 14.3)
            result._links["unadjusted"] = unadj_sf
            # pyre-ignore[16]: Sample gains this attr via BalanceFrame inheritance (diff 14.3)
            result._adjustment_model = self._adjustment_model

        return result

    # --- BalanceDF integration ---

    def _build_links_dict(self) -> dict[str, BalanceDFSource]:
        """Build a ``_links`` dict matching Sample._links structure.

        Creates a dict mapping link names to SampleFrame instances for the
        target and (if adjusted) the unadjusted responders so that
        ``BalanceDF._balancedf_child_from_linked_samples`` can walk the
        links just as it does for the old ``Sample`` class.

        Returns:
            dict: Mapping of link names to BalanceDFSource instances.
        """
        links: dict[str, BalanceDFSource] = {}
        if self._sf_target is not None:
            links["target"] = self._sf_target
        if self.is_adjusted:
            links["unadjusted"] = self._sf_sample_pre_adjust
        return links

    def covars(self, formula: str | list[str] | None = None) -> Any:
        """Return a :class:`~balance.balancedf_class.BalanceDFCovars` for the responders.

        The returned object carries linked target (and unadjusted, if
        adjusted) views so that methods like ``.mean()`` and ``.asmd()``
        automatically include comparisons across sources.

        Args:
            formula: Optional formula string (or list) for model matrix
                construction. Passed through to BalanceDFCovars.

        Returns:
            BalanceDFCovars: Covariate view with linked sources.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
            >>> bf = BalanceFrame(sample=resp, target=tgt)
            >>> bf.covars().df.columns.tolist()
            ['x']
        """
        from balance.balancedf_class import BalanceDFCovars, BalanceDFSource

        return BalanceDFCovars(
            cast(BalanceDFSource, self),
            links=self._build_links_dict(),
            formula=formula,
        )

    def weights(self) -> Any:
        """Return a :class:`~balance.balancedf_class.BalanceDFWeights` for the responders.

        The returned object carries linked target (and unadjusted, if
        adjusted) views for comparative weight analysis.

        Returns:
            BalanceDFWeights: Weight view with linked sources.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 2.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
            >>> bf = BalanceFrame(sample=resp, target=tgt)
            >>> bf.weights().df.columns.tolist()
            ['weight']
        """
        from balance.balancedf_class import BalanceDFSource, BalanceDFWeights

        # Pass self (not _sf_sample) so that r_indicator and other methods
        # that access self._sample._links find the BalanceFrame's _links.
        return BalanceDFWeights(
            cast(BalanceDFSource, self), links=self._build_links_dict()
        )

    def outcomes(self) -> Any | None:
        """Return a :class:`~balance.balancedf_class.BalanceDFOutcomes`, or None.

        Returns ``None`` if the responder SampleFrame has no outcome columns.

        Returns:
            BalanceDFOutcomes or None: Outcome view with linked sources,
                or ``None`` if no outcomes are defined.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0],
            ...                   "y": [1.0, 0.0], "weight": [1.0, 1.0]}),
            ...     outcome_columns=["y"])
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
            >>> bf = BalanceFrame(sample=resp, target=tgt)
            >>> bf.outcomes().df.columns.tolist()
            ['y']
        """
        if not self._sf_sample.outcome_columns:
            return None
        from balance.balancedf_class import BalanceDFOutcomes, BalanceDFSource

        return BalanceDFOutcomes(
            cast(BalanceDFSource, self), links=self._build_links_dict()
        )

    # --- Summary & diagnostics ---

    def _design_effect_diagnostics(
        self,
        n_rows: int | None = None,
    ) -> tuple[float | None, float | None, float | None]:
        """Compute design effect, ESS, and ESSP from the responder weights.

        Args:
            n_rows: Optional row count to use for scaling. Defaults to the
                sample size when not provided.

        Returns:
            tuple: ``(design_effect, effective_sample_size,
                effective_sample_proportion)``.  All ``None`` if the design
                effect cannot be computed.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [0, 1], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [0, 1], "weight": [1.0, 1.0]}))
            >>> bf = BalanceFrame(sample=resp, target=tgt)
            >>> bf._design_effect_diagnostics()
            (1.0, 2.0, 1.0)
        """
        if n_rows is None:
            n_rows = len(self._sf_sample)
        try:
            de = weights_stats.design_effect(self._sf_sample.df_weights.iloc[:, 0])
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            logger.debug("Unable to compute design effect: %s", exc)
            return None, None, None

        if de is None or not np.isfinite(de):
            return None, None, None

        effective_sample_size = None
        effective_sample_proportion = None
        if n_rows and de != 0:
            effective_sample_size = n_rows / de
            effective_sample_proportion = effective_sample_size / n_rows

        return float(de), effective_sample_size, effective_sample_proportion

    def _quick_adjustment_details(
        self,
        n_rows: int | None = None,
        de: float | None = None,
        ess: float | None = None,
        essp: float | None = None,
    ) -> list[str]:
        """Collect quick-to-compute adjustment diagnostics for display.

        Args:
            de: Pre-computed design effect, or ``None`` to compute lazily.
            ess: Pre-computed effective sample size, or ``None`` to compute
                lazily.
            essp: Pre-computed effective sample proportion, or ``None`` to
                compute lazily.

        Returns:
            list[str]: Human-readable lines describing adjustment method,
                trimming configuration, and weight diagnostics.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [0, 1], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [0, 1], "weight": [1.0, 1.0]}))
            >>> bf = BalanceFrame(sample=resp, target=tgt)
            >>> adjusted = bf.adjust(method="null")
            >>> "method: null" in adjusted._quick_adjustment_details()
            True
        """
        details: list[str] = []
        model = self.model
        if isinstance(model, dict):
            method = model.get("method")
            if isinstance(method, str):
                details.append(f"method: {method}")
            trimming_mean_ratio = model.get("weight_trimming_mean_ratio")
            if trimming_mean_ratio is not None:
                details.append(f"weight trimming mean ratio: {trimming_mean_ratio}")
            trimming_percentile = model.get("weight_trimming_percentile")
            if trimming_percentile is not None:
                details.append(f"weight trimming percentile: {trimming_percentile}")

        if de is None:
            de, ess, essp = self._design_effect_diagnostics(n_rows)
        if de is not None:
            details.append(f"design effect (Deff): {de:.3f}")
            if essp is not None:
                details.append(f"effective sample size proportion (ESSP): {essp:.3f}")
            if ess is not None:
                details.append(f"effective sample size (ESS): {ess:.1f}")

        return details

    def summary(self) -> str:
        """Consolidated summary of covariate balance, weight health, and outcomes.

        Produces a multi-line summary combining covariate ASMD / KLD
        diagnostics, weight design effect, and outcome means.  Delegates to
        :func:`~balance.summary_utils._build_summary` after computing the
        necessary intermediate values.

        When no target is set, returns a minimal summary with weight
        diagnostics and outcome means only.

        Returns:
            str: A human-readable multi-line summary string.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2, 3, 4], "x": [0, 1, 1, 0],
            ...                   "weight": [1.0, 2.0, 1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [5, 6, 7, 8], "x": [0, 0, 1, 1],
            ...                   "weight": [1.0, 1.0, 1.0, 1.0]}))
            >>> bf = BalanceFrame(sample=resp, target=tgt)
            >>> adjusted = bf.adjust(method="null")
            >>> "Covariate diagnostics:" in adjusted.summary()
            True
        """
        if not self.has_target() and not self.is_adjusted:
            # No target: minimal summary (weight diagnostics + outcomes only)
            de, ess, essp = self._design_effect_diagnostics(self._df.shape[0])
            outcome_means = None
            if self._outcome_columns is not None:
                outcome_means = self.outcomes().mean()
            return _build_summary(
                is_adjusted=False,
                has_target=False,
                covars_asmd=None,
                covars_kld=None,
                asmd_improvement_pct=None,
                quick_adjustment_details=[],
                design_effect=de,
                effective_sample_size=ess,
                effective_sample_proportion=essp,
                model_dict=self.model,
                outcome_means=outcome_means,
            )

        covars_asmd = self.covars().asmd()
        covars_kld = self.covars().kld(aggregate_by_main_covar=True)

        asmd_improvement_pct = None
        if self.is_adjusted:
            asmd_improvement_pct = 100 * self.covars().asmd_improvement()

        de, ess, essp = self._design_effect_diagnostics()

        quick_adjustment_details: list[str] = []
        if self.is_adjusted:
            quick_adjustment_details = self._quick_adjustment_details(
                de=de, ess=ess, essp=essp
            )

        outcome_means = None
        outcomes = self.outcomes()
        if outcomes is not None:
            outcome_means = outcomes.mean()

        return _build_summary(
            is_adjusted=bool(self.is_adjusted),
            has_target=True,
            covars_asmd=covars_asmd,
            covars_kld=covars_kld,
            asmd_improvement_pct=asmd_improvement_pct,
            quick_adjustment_details=quick_adjustment_details,
            design_effect=de,
            effective_sample_size=ess,
            effective_sample_proportion=essp,
            model_dict=self.model,
            outcome_means=outcome_means,
        )

    def diagnostics(
        self,
        weights_impact_on_outcome_method: str | None = "t_test",
        weights_impact_on_outcome_conf_level: float = 0.95,
    ) -> pd.DataFrame:
        """Table of diagnostics about the adjusted BalanceFrame.

        Produces a DataFrame with columns ``["metric", "val", "var"]``
        containing size information, weight diagnostics, model details,
        covariate ASMD, and optionally outcome-weight impact statistics.
        Delegates to :func:`~balance.summary_utils._build_diagnostics`.

        Args:
            weights_impact_on_outcome_method: Method for
                computing outcome-weight impact.  Pass ``None`` to skip.
                Defaults to ``"t_test"``.
            weights_impact_on_outcome_conf_level: Confidence level
                for outcome impact intervals.  Defaults to ``0.95``.

        Returns:
            pd.DataFrame: Diagnostics table with columns
                ``["metric", "val", "var"]``.

        Raises:
            ValueError: If this BalanceFrame has not been adjusted.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": ["1", "2"], "x": [0, 1],
            ...                   "weight": [1.0, 2.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": ["3", "4"], "x": [0, 1],
            ...                   "weight": [1.0, 1.0]}))
            >>> bf = BalanceFrame(sample=resp, target=tgt)
            >>> adjusted = bf.adjust(method="null")
            >>> adjusted.diagnostics().columns.tolist()
            ['metric', 'val', 'var']
        """
        logger.info("Starting computation of diagnostics of the fitting")
        self._require_adjusted()

        outcome_columns = self._sf_sample.df_outcomes
        outcome_impact = None
        if weights_impact_on_outcome_method is not None and outcome_columns is not None:
            outcome_impact = self.outcomes().weights_impact_on_outcome_ss(
                method=weights_impact_on_outcome_method,
                conf_level=weights_impact_on_outcome_conf_level,
                round_ndigits=None,
            )

        target = self._sf_target
        assert target is not None, "diagnostics() requires a target"
        result = _build_diagnostics(
            covars_df=self.covars().df,
            target_covars_df=target.df_covars,
            weights_summary=self.weights().summary(),
            model_dict=self.model,
            covars_asmd=self.covars().asmd(),
            covars_asmd_main=self.covars().asmd(aggregate_by_main_covar=True),
            outcome_columns=outcome_columns,
            weights_impact_on_outcome_method=weights_impact_on_outcome_method,
            weights_impact_on_outcome_conf_level=weights_impact_on_outcome_conf_level,
            outcome_impact=outcome_impact,
        )
        logger.info("Done computing diagnostics")
        return result

    # --- Parity helpers ---

    # --- DataFrame / export ---

    @property
    def df_all(self) -> pd.DataFrame:
        """Combined DataFrame with all samples, distinguished by a ``"source"`` column.

        Concatenates the responder, target, and (if adjusted) unadjusted
        DataFrames vertically, adding a ``"source"`` column with values
        ``"self"``, ``"target"``, and ``"unadjusted"`` respectively.

        Returns:
            pd.DataFrame: A DataFrame with all rows from responder, target,
                and optionally unadjusted SampleFrames, plus a ``"source"``
                column.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
            >>> bf = BalanceFrame(sample=resp, target=tgt)
            >>> bf.df_all["source"].unique().tolist()
            ['self', 'target']
        """
        parts: list[pd.DataFrame] = []

        resp_df = self._sf_sample._df.copy()
        resp_df["source"] = "self"
        parts.append(resp_df)

        if self._sf_target is not None:
            tgt_df = self._sf_target._df.copy()
            tgt_df["source"] = "target"
            parts.append(tgt_df)

        if self.is_adjusted:
            unadj_df = self._sf_sample_pre_adjust._df.copy()
            unadj_df["source"] = "unadjusted"
            parts.append(unadj_df)

        return pd.concat(parts, ignore_index=True)

    @property
    def df(self) -> pd.DataFrame:
        """Flat user-facing DataFrame from the responders.

        Returns the responder data with columns ordered as:
        id → covariates → outcomes → weight → ignored.

        Returns:
            pd.DataFrame: Ordered copy of the responder's data.
        """
        covars = self.covars()
        outcomes = self.outcomes()
        ignored = self._sf_sample.df_ignored
        return pd.concat(
            (
                self.id_column,
                covars.df if covars is not None else None,
                outcomes.df if outcomes is not None else None,
                (
                    pd.DataFrame(self.weight_series)
                    if self.weight_series is not None
                    else None
                ),
                ignored if ignored is not None else None,
            ),
            axis=1,
        )

    def keep_only_some_rows_columns(
        self,
        rows_to_keep: str | None = None,
        columns_to_keep: list[str] | None = None,
    ) -> BalanceFrame:
        """Return a new BalanceFrame with filtered rows and/or columns.

        Returns a deep copy with the requested subset applied to the
        responder, target, and (if adjusted) unadjusted SampleFrames.
        The original BalanceFrame is unchanged (immutable pattern).

        Args:
            rows_to_keep: A boolean expression string evaluated via
                ``pd.DataFrame.eval`` to select rows. Applied to each
                SampleFrame's underlying DataFrame. For example:
                ``'x > 10'`` or ``'gender == "Female"'``.
                Defaults to None (all rows kept).
            columns_to_keep: Covariate column names to retain. Special
                columns (id, weight) are always kept. Defaults to None
                (all columns kept).

        Returns:
            BalanceFrame: A new BalanceFrame with the filters applied.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2, 3], "x": [10.0, 20.0, 30.0],
            ...                   "weight": [1.0, 1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [4, 5, 6], "x": [15.0, 25.0, 35.0],
            ...                   "weight": [1.0, 1.0, 1.0]}))
            >>> bf = BalanceFrame(sample=resp, target=tgt)
            >>> filtered = bf.keep_only_some_rows_columns(rows_to_keep="x > 15")
            >>> len(filtered._sf_sample._df)
            2
        """
        if rows_to_keep is None and columns_to_keep is None:
            return self

        new_bf = copy.deepcopy(self)

        if new_bf.has_target():
            # With target: filter all SampleFrames
            new_bf._sf_sample = BalanceFrame._filter_sf(
                new_bf._sf_sample, rows_to_keep, columns_to_keep
            )
            if new_bf._sf_target is not None:
                new_bf._sf_target = BalanceFrame._filter_sf(
                    new_bf._sf_target, rows_to_keep, columns_to_keep
                )
            if new_bf._sf_sample_pre_adjust is not None:
                new_bf._sf_sample_pre_adjust = BalanceFrame._filter_sf(
                    new_bf._sf_sample_pre_adjust, rows_to_keep, columns_to_keep
                )
        else:
            # No target: filter the responder SampleFrame
            if columns_to_keep is not None:
                if not (set(columns_to_keep) <= set(new_bf.df.columns)):
                    logger.warning(
                        "Note that not all columns_to_keep are in Sample. "
                        "Only those that exist are removed"
                    )
            new_bf._sf_sample = BalanceFrame._filter_sf(
                new_bf._sf_sample, rows_to_keep, columns_to_keep
            )
            if (
                new_bf._sf_sample_pre_adjust is not None
                and new_bf._sf_sample_pre_adjust is not new_bf._sf_sample
            ):
                new_bf._sf_sample_pre_adjust = BalanceFrame._filter_sf(
                    new_bf._sf_sample_pre_adjust, rows_to_keep, columns_to_keep
                )

        # Also filter linked BF/Sample objects in _links
        if new_bf._links:
            for k, v in list(new_bf._links.items()):
                if isinstance(v, BalanceFrame):
                    try:
                        new_bf._links[k] = v.keep_only_some_rows_columns(
                            rows_to_keep=rows_to_keep,
                            columns_to_keep=columns_to_keep,
                        )
                    except (TypeError, ValueError, AttributeError, KeyError) as exc:
                        logger.warning(
                            "couldn't filter _links['%s'] using provided filters: %s",
                            k,
                            exc,
                        )

        return new_bf

    @staticmethod
    def _filter_sf(
        sf: SampleFrame,
        rows_to_keep: str | None,
        columns_to_keep: list[str] | None,
    ) -> SampleFrame:
        """Apply row and column filtering to a SampleFrame in place.

        Used internally by :meth:`keep_only_some_rows_columns` to filter
        each SampleFrame (responders, target, unadjusted) consistently.

        Args:
            sf: The SampleFrame to filter (mutated in place).
            rows_to_keep: A pandas ``eval()`` expression for row filtering,
                or ``None`` to skip row filtering.
            columns_to_keep: Column names to retain, or ``None`` to skip
                column filtering. ID and weight columns are always retained.

        Returns:
            SampleFrame: The same *sf* instance, mutated.

        Note:
            If ``rows_to_keep`` references a column that does not exist in
            *sf*, the ``UndefinedVariableError`` is caught, a warning is
            logged, and row filtering is skipped for that SampleFrame.  This
            is intentional: linked frames (target, unadjusted) may not have
            the same columns as the responder, so a filter expression valid
            for the responder may fail on linked frames.  This matches the
            ``Sample.keep_only_some_rows_columns`` behaviour.
        """
        df = sf._df

        if rows_to_keep is not None:
            try:
                mask = df.eval(rows_to_keep)
                logger.info(f"(rows_filtered/total_rows) = ({mask.sum()}/{len(mask)})")
                df = df[mask].reset_index(drop=True)
            except pd.errors.UndefinedVariableError:
                logger.warning(f"couldn't filter SampleFrame using {rows_to_keep}")

        if columns_to_keep is not None:
            keep_set = set(columns_to_keep)
            keep_set.add(sf._id_column_name)
            if sf._weight_column_name is not None:
                keep_set.add(sf._weight_column_name)
            for wc in sf.weight_columns_all:
                keep_set.add(wc)
            # Always preserve outcome columns (matching Sample behavior)
            for oc in sf._column_roles.get("outcomes", []):
                keep_set.add(oc)
            df = df.loc[:, df.columns.isin(keep_set)]

            new_covars = [c for c in sf._column_roles["covars"] if c in keep_set]
            sf._column_roles = dict(sf._column_roles)
            sf._column_roles["covars"] = new_covars
            if sf._column_roles["outcomes"]:
                sf._column_roles["outcomes"] = [
                    c for c in sf._column_roles["outcomes"] if c in keep_set
                ]
            if sf._column_roles["predicted"]:
                sf._column_roles["predicted"] = [
                    c for c in sf._column_roles["predicted"] if c in keep_set
                ]
            if sf._column_roles["ignored"]:
                sf._column_roles["ignored"] = [
                    c for c in sf._column_roles["ignored"] if c in keep_set
                ]

        sf._df = df
        return sf

    def to_csv(
        self, path_or_buf: FilePathOrBuffer | None = None, **kwargs: Any
    ) -> str | None:
        """Write the combined DataFrame to CSV.

        Writes the output of :attr:`df` (responder + target + unadjusted
        rows with a ``"source"`` column) to a CSV file or string.
        Delegates to :func:`~balance.csv_utils.to_csv_with_defaults`.

        Args:
            path_or_buf: Destination. If ``None``, returns the CSV as a string.
            **kwargs: Additional keyword arguments passed to
                :func:`pd.DataFrame.to_csv`.

        Returns:
            str or None: CSV string if ``path_or_buf`` is None, else None.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
            >>> bf = BalanceFrame(sample=resp, target=tgt)
            >>> "id" in bf.to_csv()
            True
        """
        return to_csv_with_defaults(self.df, path_or_buf, **kwargs)

    def to_download(self, tempdir: str | None = None) -> Any:
        """Create a downloadable file link of the combined DataFrame.

        Writes :attr:`df` to a temporary CSV file and returns an IPython
        :class:`~IPython.lib.display.FileLink` for interactive download.

        Args:
            tempdir: Directory for the temp file. If None, uses
                :func:`tempfile.gettempdir`.

        Returns:
            FileLink: An IPython file link for downloading the CSV.

        Examples:
            >>> import tempfile
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
            >>> bf = BalanceFrame(sample=resp, target=tgt)
            >>> link = bf.to_download(tempdir=tempfile.gettempdir())
        """
        return _to_download(self.df, tempdir)

    # --- Methods moved from Sample ---

    def model_matrix(self) -> pd.DataFrame:
        """Return the model matrix of the responder covariates.

        Constructs a model matrix using :func:`balance.util.model_matrix`,
        adding NA indicators for null values.

        Returns:
            pd.DataFrame: The model matrix.
        """
        res = _assert_type(
            balance_util.model_matrix(self, add_na=True)["sample"], pd.DataFrame
        )
        return res

    # TODO: Add a trim() method directly on BalanceFrame that delegates
    # to trim_weights() and calls set_weights(), providing a convenient
    # path for permanent weight trimming without going through
    # weights().trim().

    def set_weights(
        self,
        weights: pd.Series | float | None,
        *,
        use_index: bool = False,
    ) -> None:
        """Set or replace the responder weights.

        Delegates to the underlying SampleFrame's ``set_weights``.

        When called on an unadjusted BalanceFrame (``is_adjusted`` is False),
        ``_sf_sample`` and ``_sf_sample_pre_adjust`` share the same DataFrame,
        so the change is visible to both automatically — changing base weights
        is not an adjustment.

        Args:
            weights: New weights. A Series, a scalar (broadcast to all rows),
                or ``None`` (sets all to 1.0).
            use_index: If True, align *weights* by index instead of requiring
                matching length.  See :meth:`SampleFrame.set_weights`.
        """
        self._sf_sample.set_weights(weights, use_index=use_index)

    def trim(
        self,
        ratio: float | int | None = None,
        percentile: float | tuple[float, float] | None = None,
        keep_sum_of_weights: bool = True,
        target_sum_weights: float | int | np.floating | None = None,
        *,
        in_place: bool = False,
    ) -> Self:
        """Trim extreme weights using mean-ratio clipping or percentile winsorization.

        Delegates to :meth:`SampleFrame.trim` for computation and weight
        history tracking, then wraps the result in a new BalanceFrame
        (preserving target, pre-adjust baseline, and links).

        Args:
            ratio: Mean-ratio upper bound.  Mutually exclusive with
                *percentile*.
            percentile: Percentile(s) for winsorization.  Mutually exclusive
                with *ratio*.
            keep_sum_of_weights: Whether to rescale after trimming to
                preserve the original sum of weights.
            target_sum_weights: If provided, rescale trimmed weights so
                their sum equals this target.
            in_place: If True, mutate this BalanceFrame's weights and
                return it.  If False (default), return a new BalanceFrame.

        Returns:
            The BalanceFrame with trimmed weights (self if *in_place*,
            else a new instance).
        """
        if in_place:
            self._sf_sample.trim(
                ratio=ratio,
                percentile=percentile,
                keep_sum_of_weights=keep_sum_of_weights,
                target_sum_weights=target_sum_weights,
                in_place=True,
            )
            return self

        new_sf = self._sf_sample.trim(
            ratio=ratio,
            percentile=percentile,
            keep_sum_of_weights=keep_sum_of_weights,
            target_sum_weights=target_sum_weights,
            in_place=False,
        )
        new_bf = type(self)._create(
            sample=new_sf,
            target=self._sf_target,
        )
        new_bf._sf_sample_pre_adjust = self._sf_sample_pre_adjust
        # Preserve existing links (target, unadjusted).
        for key, val in self._links.items():
            new_bf._links[key] = val
        return new_bf

    def set_unadjusted(self, second: BalanceFrame) -> Self:
        """Set the unadjusted link for comparative analysis.

        Returns a deep copy with ``_sf_sample_pre_adjust`` pointing at
        *second*'s responder SampleFrame, and ``_links["unadjusted"]``
        pointing at *second*.

        Args:
            second: A BalanceFrame (or subclass) whose responder data
                becomes the unadjusted baseline.

        Returns:
            A new BalanceFrame with the unadjusted link set.

        Raises:
            TypeError: If *second* is not a BalanceFrame.
        """
        if not isinstance(second, BalanceFrame):
            raise TypeError(
                f"set_unadjusted must be called with a BalanceFrame argument, got {type(second).__name__}"
            )
        new_bf = deepcopy(self)
        new_bf._links["unadjusted"] = second
        new_bf._sf_sample_pre_adjust = second._sf_sample
        return new_bf

    # --- Column accessors (moved from Sample) ---

    def _special_columns_names(self) -> list[str]:
        """Return names of all special columns (id, weight, outcome, ignored)."""
        return (
            [str(i.name) for i in [self.id_column, self.weight_series] if i is not None]
            + (
                self._outcome_columns.columns.tolist()
                if self._outcome_columns is not None
                else []
            )
            + getattr(self, "_ignored_column_names", [])
        )

    def _special_columns(self) -> pd.DataFrame:
        """Return a DataFrame of all special columns."""
        return self._df[self._special_columns_names()]

    def _covar_columns_names(self) -> list[str]:
        """Return names of all covariate columns."""
        return [
            c for c in self._df.columns.values if c not in self._special_columns_names()
        ]

    def _covar_columns(self) -> pd.DataFrame:
        """Return a DataFrame of all covariate columns."""
        return self._sf_sample._covar_columns()

    # --- Error checks (moved from Sample) ---

    def _require_adjusted(self) -> None:
        """Raise ValueError if not adjusted."""
        if not self.is_adjusted:
            raise ValueError(
                f"This {type(self).__name__} is not adjusted. "
                "Use .adjust() to adjust to target."
            )

    def _require_target(self) -> None:
        """Raise ValueError if no target is set."""
        if not self.has_target():
            raise ValueError(
                f"This {type(self).__name__} does not have a target set. "
                "Use .set_target() to add a target."
            )

    def _require_outcomes(self) -> None:
        """Raise ValueError if no outcome columns are specified."""
        if self.outcomes() is None:
            raise ValueError(
                f"This {type(self).__name__} does not have outcome columns specified."
            )

    def __repr__(self) -> str:
        return (
            f"({self.__class__.__module__}.{self.__class__.__qualname__})\n"
            f"{self.__str__()}"
        )

    def __str__(self, pkg_source: str | None = None) -> str:
        """Return a readable summary of the sample and any applied adjustment.

        Args:
            pkg_source: Package namespace used in the header. Defaults to
                the module's ``__package__``.

        Returns:
            str: Multi-line description highlighting key structure and
                adjustment details.
        """
        if pkg_source is None:
            pkg_source = __package__

        is_adjusted = self.is_adjusted() * "Adjusted "
        n_rows = self._df.shape[0]
        n_variables = self._covar_columns().shape[1]
        has_target = self.has_target() * " with target set"
        adjustment_method = (
            " using " + _assert_type(self.model)["method"]
            if self.model is not None
            else ""
        )
        variables = ",".join(self._covar_columns_names())
        id_column_name = self.id_column.name if self.id_column is not None else "None"
        weight_column_name = (
            self.weight_series.name if self.weight_series is not None else "None"
        )
        outcome_column_names = (
            ",".join(self._outcome_columns.columns.tolist())
            if self._outcome_columns is not None
            else "None"
        )

        desc = f"""
        {is_adjusted}{pkg_source} Sample object{has_target}{adjustment_method}
        {n_rows} observations x {n_variables} variables: {variables}
        id_column: {id_column_name}, weight_column: {weight_column_name},
        outcome_columns: {outcome_column_names}
        """

        if self.is_adjusted():
            adjustment_details = self._quick_adjustment_details(n_rows)
            if len(adjustment_details) > 0:
                desc += """
        adjustment details:
            {details}
                """.format(
                    details="\n            ".join(adjustment_details)
                )

        if self.has_target():
            common_variables = balance_util.choose_variables(
                self, self._links["target"], variables=None
            )
            target_str = self._links["target"].__str__().replace("\n", "\n\t")
            n_common = len(common_variables)
            common_variables = ",".join(common_variables)
            desc += f"""
            target:
                 {target_str}
            {n_common} common variables: {common_variables}
            """
        return desc
