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
import warnings
from copy import deepcopy
from typing import Any, Callable, cast, Literal, overload, TYPE_CHECKING

import numpy as np
import pandas as pd
from balance import adjustment as balance_adjustment, util as balance_util
from balance.adjustment import _find_adjustment_method
from balance.csv_utils import to_csv_with_defaults
from balance.sample_frame import SampleFrame
from balance.stats_and_plots import weights_stats
from balance.summary_utils import _build_diagnostics, _build_summary
from balance.typing import FilePathOrBuffer
from balance.util import (
    _assert_type,
    _detect_high_cardinality_features,
    _safe_fillna_and_infer,
    HighCardinalityFeature,
)
from balance.utils.file_utils import _to_download
from balance.utils.model_matrix import build_design_matrix

if TYPE_CHECKING:
    from typing import Self

    from balance.balancedf_class import BalanceDFSource  # noqa: F401

# The set of string method names accepted by _find_adjustment_method.
_AdjustmentMethodStr = Literal["cbps", "ipw", "null", "poststratify", "rake"]

logger: logging.Logger = logging.getLogger(__package__)


def _rake_joint_distribution_divergence(
    training_joint: np.ndarray, scoring_joint: np.ndarray
) -> float:
    """Compute total-variation divergence between two rake joint tables.

    Both inputs are interpreted as non-negative contingency tables over the
    same joint support (same shape). They are normalised to probability
    distributions and compared via total variation distance:

    ``TV(P, Q) = 0.5 * sum(abs(P - Q))``.

    Returns a value in ``[0, 1]``:
    - ``0`` means identical normalised joint distributions.
    - ``1`` means disjoint support in the normalised distributions.

    Args:
        training_joint: Fit-time joint table (e.g., stored ``m_sample``).
        scoring_joint: Scoring-time joint table on the same cell grid.

    Raises:
        ValueError: If table shapes differ or either table has non-positive sum.
    """
    train = np.asarray(training_joint, dtype=float)
    score = np.asarray(scoring_joint, dtype=float)
    if train.shape != score.shape:
        raise ValueError("Rake divergence requires joint tables with matching shapes.")
    train_sum = float(train.sum())
    score_sum = float(score.sum())
    if train_sum <= 0 or score_sum <= 0:
        raise ValueError("Rake divergence requires positive table sums.")
    return float(0.5 * np.abs(train / train_sum - score / score_sum).sum())


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
            # pyrefly: ignore [missing-attribute]
            self._df = responder._df
            # pyrefly: ignore [missing-attribute]
            self._id_column_name = responder._id_column_name
            # pyrefly: ignore [missing-attribute]
            self._column_roles = responder._column_roles
            # pyrefly: ignore [missing-attribute]
            self._weight_column_name = responder._weight_column_name
            # pyrefly: ignore [missing-attribute]
            self._weight_metadata = responder._weight_metadata
            # pyrefly: ignore [missing-attribute]
            self._df_dtypes = responder._df_dtypes

    @property
    def _df_dtypes(self) -> pd.Series | None:
        """Original dtypes, delegated to ``_sf_sample._df_dtypes``."""
        return self._sf_sample._df_dtypes

    @_df_dtypes.setter
    def _df_dtypes(self, value: pd.Series | None) -> None:
        # pyrefly: ignore [missing-attribute]
        self._sf_sample._df_dtypes = value

    @property
    def id_series(self) -> pd.Series | None:
        """The id column as a Series, delegated to ``_sf_sample``."""
        return self._sf_sample.id_series

    @property
    def id_column(self) -> str | None:
        """The id column name, delegated to ``_sf_sample``.

        Changed in 0.20.0 to return the name (str) instead of data (pd.Series).
        Use :attr:`id_series` for data.
        """
        # TODO: remove this warning after 2026-06-01
        warnings.warn(
            "Note: id_column now returns the column name (str) since "
            "balance 0.20.0. It previously returned ID data (pd.Series). "
            "Use id_series for ID data.",
            FutureWarning,
            stacklevel=2,
        )
        return self._sf_sample._id_column_name

    @property
    def weight_series(self) -> pd.Series | None:
        """The active weight as a Series, delegated to ``_sf_sample``."""
        try:
            return self._sf_sample.weight_series
        except ValueError:
            return None

    # --- Property descriptors backed by _sf_sample ---

    @property
    def _df(self) -> pd.DataFrame:
        """The internal DataFrame, delegated to ``_sf_sample._df``."""
        return self._sf_sample._df

    @_df.setter
    def _df(self, value: pd.DataFrame | None) -> None:
        if value is None:
            raise ValueError(
                "Cannot set _df to None. A BalanceFrame must always have a "
                "backing DataFrame."
            )
        # pyrefly: ignore [missing-attribute]
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
    def _ignored_column_names(self) -> list[str]:
        """Ignored column names, delegated to ``_sf_sample.ignored_columns``."""
        return self._sf_sample._column_roles.get("ignored", [])

    @_ignored_column_names.setter
    def _ignored_column_names(self, value: list[str] | None) -> None:
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
            # pyrefly: ignore [unsupported-operation]
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
        self, target: BalanceFrame | SampleFrame, inplace: bool | None = None
    ) -> Self:
        """Set or replace the target population.

        When *target* is a BalanceFrame (or subclass such as Sample), a deep
        copy of ``self`` is returned with the target set (immutable pattern).
        When *target* is a raw SampleFrame, the behaviour depends on
        *inplace*: True mutates self, False returns a new BalanceFrame.

        Args:
            target: The target population — a BalanceFrame/Sample or a
                SampleFrame.
            inplace: If True, mutates self (only valid for SampleFrame
                targets). If False, returns a new copy. Defaults to None
                which auto-selects: copy for BalanceFrame targets, inplace
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
            # pyrefly: ignore [unsupported-operation]
            new_copy._links["target"] = target
            BalanceFrame._validate_covariate_overlap(
                new_copy._sf_sample, target._sf_sample
            )
            new_copy._sf_target = target._sf_sample
            return new_copy

        if isinstance(target, SampleFrame):
            # SampleFrame path: default inplace=True for backward compat
            if inplace is None:
                inplace = True
            BalanceFrame._validate_covariate_overlap(self._sf_sample, target)

            if inplace:
                if self.is_adjusted:
                    logger.warning(
                        "Replacing target on an adjusted object resets responder "
                        "weights to pre-adjust values and discards current "
                        "adjustment results. Pass inplace=False to return a new "
                        "object and keep the current adjusted state on this "
                        "instance."
                    )
                self._sf_target = target
                # pyrefly: ignore [unsupported-operation]
                self._links["target"] = target
                # Reset adjustment state — old adjustment is no longer valid.
                self._sf_sample = self._sf_sample_pre_adjust
                self._adjustment_model = None
                # pyrefly: ignore [missing-attribute]
                self._links.pop("unadjusted", None)
                self._sync_sampleframe_state_from_responder(self._sf_sample)
                return self
            else:
                return type(self)._create(
                    sample=copy.deepcopy(self._sf_sample_pre_adjust),
                    target=target,
                )

        raise TypeError("A target, a Sample object, must be specified")

    def set_as_pre_adjust(self, *, inplace: bool = False) -> Self:
        """Set the current responder state as the new pre-adjust baseline.

        This "locks in" the current responder weights (which may already be
        adjusted and/or trimmed) as the baseline for future diagnostics and
        subsequent adjustments.

        Args:
            inplace: If True, mutate this object and return it. If False
                (default), return a new object with a deep-copied responder
                frame and reset baseline.

        Returns:
            BalanceFrame with ``_sf_sample_pre_adjust`` reset to the current
            responder SampleFrame state. In copy mode (``inplace=False``),
            only the responder frame is deep-copied and used to construct a new
            object (the full ``_links`` graph is not deep-copied). In inplace
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
            >>> _ = adjusted.set_as_pre_adjust(inplace=True)  # inplace mode
        """
        if inplace:
            bf = self
            frozen = bf._sf_sample
        else:
            frozen = copy.deepcopy(self._sf_sample)
            bf = type(self)._create(sample=frozen, target=self._sf_target)
            # Preserve a richer target link (e.g., BalanceFrame/Sample object)
            # when present on the original.
            # pyrefly: ignore [not-iterable]
            if "target" in self._links:
                # pyrefly: ignore [unsupported-operation]
                bf._links["target"] = self._links["target"]
        bf._sf_sample_pre_adjust = frozen
        bf._sf_sample = frozen
        bf._adjustment_model = None
        # pyrefly: ignore [missing-attribute]
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
        # pyrefly: ignore [not-iterable]
        if "unadjusted" in self._links:
            # pyrefly: ignore [unsupported-operation]
            new_bf._links["unadjusted"] = self._links["unadjusted"]
        else:
            # pyrefly: ignore [unsupported-operation]
            new_bf._links["unadjusted"] = self
        # pyrefly: ignore [not-iterable]
        if "target" in self._links:
            # pyrefly: ignore [unsupported-operation]
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
        adj_model = new_bf._adjustment_model
        if isinstance(adj_model, dict):
            adj_model.setdefault("method", method_name)
            if adj_model.get("method") == "ipw":
                # Preserve training-time design weights only when the weighting
                # method already opted into fit metadata.
                fit_sample_weights = adj_model.get("fit_sample_weights")
                if isinstance(fit_sample_weights, pd.Series):
                    adj_model.setdefault(
                        "training_sample_weights",
                        fit_sample_weights,
                    )
                if new_bf._sf_target is not None:
                    fit_target_weights = adj_model.get("fit_target_weights")
                    if isinstance(fit_target_weights, pd.Series):
                        adj_model.setdefault(
                            "training_target_weights",
                            fit_target_weights,
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
        *,
        target: BalanceFrame | SampleFrame | None = None,
        method: str | Callable[..., Any] = "ipw",
        inplace: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Fit a weighting model and return the fitted BalanceFrame.

        This is the sklearn-style entry point for survey weight adjustment.
        Like sklearn's ``fit()``, it learns model parameters, mutates
        ``self`` (by default), and returns ``self``.  In survey weighting,
        fitting the propensity model inherently produces adjusted weights
        (the two are inseparable), so the returned object contains both the
        fitted model and the adjusted weights — analogous to how
        ``KMeans.fit()`` stores ``labels_`` on the fitted object.

        **Workflow — basic fitting (sklearn-style, inplace=True):**

        .. code-block:: python

            bf = BalanceFrame(sample=respondents, target=population)
            bf.fit(method="ipw")       # mutates bf, returns bf
            bf.weights().df            # the adjusted weights

        **Workflow — functional style (inplace=False):**

        .. code-block:: python

            adjusted = bf.fit(method="ipw", inplace=False)

        **Workflow — fit on subset, apply to holdout:**

        .. code-block:: python

            fitted = train_bf.fit(method="ipw")
            scored = holdout_bf.set_fitted_model(fitted, inplace=False)
            holdout_weights = scored.predict_weights()

        Alternatively, ``design_matrix()``, ``predict_proba()``, and
        ``predict_weights()`` accept a ``data=`` argument so the holdout
        workflow becomes a single line:
        ``fitted.predict_weights(data=holdout_bf)``.

        Args:
            target: Optional target population to set before fitting.  If
                provided, this method calls ``set_target(target, inplace=False)``
                first, preserving immutability.
            method: Adjustment method name (``"ipw"``, ``"cbps"``, ``"rake"``,
                ``"poststratify"``, ``"null"``) or a custom callable with the
                weighting-method signature.
            inplace: If True (default), mutate this object with the fitted
                state and return ``self`` — matching sklearn's ``fit()``
                convention.  If False, return a new adjusted BalanceFrame
                without modifying ``self``.
            **kwargs: Keyword arguments forwarded to :meth:`adjust`.

        Returns:
            The fitted BalanceFrame — ``self`` when ``inplace=True``,
            a new object when ``inplace=False``.

        Raises:
            ValueError: If no target is available and none is provided, if
                ``method`` is invalid, or if ``na_action='drop'`` is combined
                with stored fit artifacts.

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

        Notes:
            For the built-in IPW method, ``fit()`` enables
            ``store_fit_metadata=True`` and ``store_fit_matrices=True`` by
            default so ``design_matrix()``/``predict_proba()``/``predict_weights()``
            can consume fit-time artifacts.  This may increase memory usage
            for large inputs; pass these kwargs explicitly as ``False`` to
            opt out.
            For the built-in CBPS method, ``fit()`` enables
            ``store_fit_metadata=True`` by default so ``predict_weights()``
            can reconstruct CBPS scoring artifacts.  Pass
            ``store_fit_metadata=False`` to opt out.
            For the built-in poststratify method, ``fit()`` enables
            ``store_fit_metadata=True`` by default so ``predict_weights()``
            can reconstruct poststratification cell-ratio artifacts, while
            direct ``adjust(method='poststratify')`` remains metadata-light
            unless ``store_fit_metadata=True`` is passed explicitly.
            For the built-in rake method, ``fit()`` enables
            ``store_fit_metadata=True`` by default so ``predict_weights()``
            can replay/transfer fitted rake artifacts. This may increase
            memory usage because contingency tables and fit metadata are
            stored; pass ``store_fit_metadata=False`` to opt out.
        """
        from balance.weighting_methods.cbps import cbps as built_in_cbps
        from balance.weighting_methods.ipw import ipw as built_in_ipw
        from balance.weighting_methods.poststratify import (
            poststratify as built_in_poststratify,
        )
        from balance.weighting_methods.rake import rake as built_in_rake

        resolved_method = self._resolve_adjustment_function(method)
        if resolved_method is built_in_ipw:
            kwargs.setdefault("store_fit_matrices", True)
            kwargs.setdefault("store_fit_metadata", True)
            na_action = kwargs.get("na_action", "add_indicator")
            store_fit_matrices = bool(kwargs.get("store_fit_matrices"))
            store_fit_metadata = bool(kwargs.get("store_fit_metadata"))
            if na_action == "drop" and (store_fit_matrices or store_fit_metadata):
                raise ValueError(
                    "BalanceFrame.fit(method='ipw', na_action='drop') is incompatible "
                    "with stored fit artifacts because dropped rows break index/shape "
                    "alignment for design_matrix/predict_proba. Use na_action='add_indicator', "
                    "or disable store_fit_matrices/store_fit_metadata."
                )
        if resolved_method is built_in_cbps:
            user_set_store_fit_metadata = "store_fit_metadata" in kwargs
            kwargs.setdefault("store_fit_metadata", True)
            na_action = kwargs.get("na_action", "add_indicator")
            store_fit_metadata = bool(kwargs.get("store_fit_metadata"))
            if na_action == "drop" and store_fit_metadata:
                if user_set_store_fit_metadata:
                    raise ValueError(
                        "BalanceFrame.fit(method='cbps', na_action='drop') is incompatible "
                        "with stored fit metadata because dropped rows break index/shape "
                        "alignment for predict_weights(). Use na_action='add_indicator', "
                        "or disable store_fit_metadata."
                    )
                warnings.warn(
                    "BalanceFrame.fit(method='cbps', na_action='drop') disables "
                    "store_fit_metadata by default because dropped rows are incompatible "
                    "with predict_weights() reconstruction. Pass "
                    "store_fit_metadata=True to receive an explicit error.",
                    UserWarning,
                    stacklevel=2,
                )
                kwargs["store_fit_metadata"] = False
        if resolved_method is built_in_poststratify:
            kwargs.setdefault("store_fit_metadata", True)
        if resolved_method is built_in_rake:
            kwargs.setdefault("store_fit_metadata", True)

        if isinstance(target, (SampleFrame, BalanceFrame)):
            result = self.set_target(target, inplace=False).adjust(
                method=method, **kwargs
            )
        else:
            result = self.adjust(target=target, method=method, **kwargs)

        if not inplace:
            return result

        # Copy fitted state from result into self.
        self._sf_sample = result._sf_sample
        self._sf_sample_pre_adjust = result._sf_sample_pre_adjust
        self._sf_target = result._sf_target
        self._adjustment_model = result._adjustment_model
        self._links = result._links
        self._sync_sampleframe_state_from_responder(self._sf_sample)
        return self

    def set_fitted_model(self, fitted: BalanceFrame, *, inplace: bool = True) -> Self:
        """Apply a fitted model from another BalanceFrame, producing a fully adjusted result.

        This enables fit-then-apply workflows: fit on one BalanceFrame (e.g.,
        a 20k subset) and apply the fitted model to another BalanceFrame
        (e.g., the remaining 980k) with the same covariate schema.  The
        returned object is fully adjusted (``is_adjusted`` is True,
        ``model`` is set, ``summary()`` works with 3-way comparison).

        **Workflow (inplace=False — returns new adjusted object):**

        .. code-block:: python

            fitted = train_bf.fit(method="ipw")
            scored = holdout_bf.set_fitted_model(fitted, inplace=False)
            scored.summary()  # full diagnostics on holdout

        **Workflow (inplace=True, default — mutates self):**

        .. code-block:: python

            holdout_bf.set_fitted_model(fitted)
            holdout_bf.summary()

        Currently ``set_fitted_model`` applies fitted IPW models directly.
        Other methods may have fit artifacts but are routed through their
        dedicated ``predict_weights(data=...)`` workflows instead of this
        method-specific application path.

        Args:
            fitted: A BalanceFrame already adjusted with a supported method.
                Its fitted model is used to compute holdout weights.
            inplace: If True (default), mutate this object and return ``self``.
                If False, return a new BalanceFrame with computed weights,
                leaving ``self`` unchanged.

        Returns:
            A fully adjusted BalanceFrame with holdout weights applied.
            ``self`` when ``inplace=True``, a new object when ``inplace=False``.

        Raises:
            ValueError: If ``fitted`` has no stored model, if the model method
                is not yet supported, or if covariate column names differ
                between ``self`` and ``fitted``.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> train_resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [0.0, 1.0], "weight": [1.0, 1.0]}))
            >>> train_tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [0.2, 0.8], "weight": [1.0, 1.0]}))
            >>> holdout_resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [5, 6], "x": [0.1, 0.9], "weight": [1.0, 1.0]}))
            >>> holdout_tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [7, 8], "x": [0.3, 0.7], "weight": [1.0, 1.0]}))
            >>> train_bf = BalanceFrame(sample=train_resp, target=train_tgt)
            >>> holdout_bf = BalanceFrame(sample=holdout_resp, target=holdout_tgt)
            >>> fitted = train_bf.fit(method="ipw")
            >>> scored = holdout_bf.set_fitted_model(fitted, inplace=False)
            >>> scored.is_adjusted
            True
            >>> scored.model is not None
            True
        """
        from balance.weighting_methods.ipw import link_transform, weights_from_link

        # --- Validation ---
        if fitted.model is None:
            raise ValueError(
                "fitted must be an adjusted BalanceFrame with a stored model."
            )
        model = fitted.model
        if not isinstance(model, dict):
            raise ValueError("fitted must contain a valid adjustment model dict.")
        method = model.get("method")
        if method != "ipw":
            raise ValueError(
                "set_fitted_model() currently supports only IPW models. "
                f"The fitted model uses method '{method}'. "
                "Other methods (CBPS, rake, poststratify) will be supported "
                "once they store fit-time artifacts."
            )
        if model.get("fit") is None:
            raise ValueError("fitted IPW model is missing estimator information.")
        if set(self._sf_sample.covars().df.columns) != set(
            fitted._sf_sample.covars().df.columns
        ):
            raise ValueError(
                "self and fitted must have matching sample covariate column names."
            )
        if self._sf_target is not None and fitted._sf_target is not None:
            if set(_assert_type(self._sf_target).covars().df.columns) != set(
                _assert_type(fitted._sf_target).covars().df.columns
            ):
                raise ValueError(
                    "self and fitted must have matching target covariate column names."
                )

        # --- Compute holdout weights ---
        sample_matrix, _target_matrix = self._compute_ipw_matrices(model, source=self)
        fit_model = _assert_type(model.get("fit"))
        class_index = self._ipw_class_index(fit_model)
        prob = np.asarray(fit_model.predict_proba(sample_matrix)[:, class_index])
        link = link_transform(prob)

        # Use holdout's own design weights
        current_sample_weights = self._sf_sample.df_weights.iloc[:, 0]
        training_sample_weights = model.get("training_sample_weights")
        sample_weights = (
            training_sample_weights
            if isinstance(training_sample_weights, pd.Series)
            and training_sample_weights.index.equals(current_sample_weights.index)
            else current_sample_weights
        )
        target_weights = _assert_type(self._sf_target).df_weights.iloc[:, 0]

        # Warn if holdout target weight sum differs >1% from training
        training_target_weights = model.get("training_target_weights")
        if isinstance(training_target_weights, pd.Series):
            train_sum = training_target_weights.sum()
            data_sum = target_weights.sum()
            if train_sum > 0 and abs(train_sum - data_sum) / train_sum > 0.01:
                logger.warning(
                    "set_fitted_model(): holdout target weights sum (%.2f) "
                    "differs from training target weights sum (%.2f). The "
                    "balance_classes correction and weight normalization will "
                    "use holdout's weights, which may produce different results "
                    "than the training fit.",
                    data_sum,
                    train_sum,
                )

        predicted = weights_from_link(
            link=link,
            balance_classes=bool(model.get("balance_classes", True)),
            sample_weights=sample_weights,
            target_weights=target_weights,
            weight_trimming_mean_ratio=model.get("weight_trimming_mean_ratio"),
            weight_trimming_percentile=model.get("weight_trimming_percentile"),
        )

        # --- Build the result ---
        if inplace:
            bf = self
        else:
            bf = type(self)._create(
                sample=deepcopy(self._sf_sample),
                target=deepcopy(self._sf_target),
            )
            # pyrefly: ignore [not-iterable]
            if "target" in self._links:
                # pyrefly: ignore [unsupported-operation]
                bf._links["target"] = deepcopy(self._links["target"])

        # Separate _sf_sample from _sf_sample_pre_adjust BEFORE mutating weights,
        # so the unadjusted baseline preserves original design weights.
        if not bf.is_adjusted and bf._sf_sample is bf._sf_sample_pre_adjust:
            bf._sf_sample_pre_adjust = deepcopy(bf._sf_sample_pre_adjust)

        # Apply computed weights
        bf._sf_sample.set_weights(
            pd.Series(predicted.values, index=bf._sf_sample.df.index),
            use_index=True,
        )

        # Store the model and set adjustment state
        bf._adjustment_model = dict(model)
        # pyrefly: ignore [unsupported-operation]
        bf._links["unadjusted"] = type(self)._create(
            sample=bf._sf_sample_pre_adjust,
            target=bf._sf_target,
        )

        bf._sync_sampleframe_state_from_responder(bf._sf_sample)
        return bf

    def _require_fitted_model(self) -> dict[str, Any]:
        """Return the adjustment model dict, or raise.

        The model dict serves triple duty: configuration (formula, na_action,
        one_hot_encoding), fitted artifacts (fit estimator, scaler, column
        names), and cache (model_matrix_sample, sample_probability, etc.).
        Treat nested values as read-only; cache updates must replace dict
        keys, not mutate shared objects.
        """
        model = self._adjustment_model
        if model is None or not isinstance(model, dict):
            raise ValueError(
                "This operation requires an adjusted model. "
                "Call fit()/adjust() first, or apply a model via "
                "set_fitted_model()."
            )
        return model

    def _require_ipw_model(self) -> dict[str, Any]:
        """Return the model dict, raising if it is not an IPW model with fit info."""
        model = self._require_fitted_model()
        if model.get("method") != "ipw":
            raise ValueError(
                "design_matrix() and predict_proba() currently support only IPW models. "
                f"The current model uses method '{model.get('method')}'."
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
            return matrix.reindex(columns=columns)
        if isinstance(matrix, np.ndarray):
            return pd.DataFrame(matrix, index=index, columns=columns)
        from scipy.sparse import spmatrix

        if isinstance(matrix, spmatrix):
            return pd.DataFrame.sparse.from_spmatrix(
                matrix,
                index=index,
                columns=columns,
            )
        raise ValueError(
            "Stored IPW fit-time model matrix is unavailable for this configuration."
        )

    def _align_to_index(
        self,
        data: pd.DataFrame | pd.Series,
        index: pd.Index,
        caller: str,
        method_name: str = "ipw",
    ) -> pd.DataFrame | pd.Series:
        """Align a DataFrame or Series to the given index.

        Handles unique indices (via reindex), reordered indices (via reindex),
        and non-unique indices of equal length (via set_axis).  Raises if
        indices are incompatible.
        """
        if data.index.is_unique and index.is_unique:
            if len(data.index) == len(index) and not data.index.equals(index):
                if not (data.index.isin(index).all() and index.isin(data.index).all()):
                    raise ValueError(
                        f"Stored {method_name.upper()} {caller} output index does not "
                        "match the current data index. Re-fit with "
                        f"BalanceFrame.fit(method='{method_name}') or "
                        "attach a model trained on matching rows."
                    )
            return data.reindex(index)
        if len(data.index) != len(index):
            raise ValueError(
                f"Stored {method_name.upper()} {caller} output cannot be aligned to "
                "the current index because lengths differ. Re-fit with "
                f"BalanceFrame.fit(method='{method_name}') to refresh stored artifacts."
            )
        return data.set_axis(index, axis=0)

    @staticmethod
    def _ipw_class_index(fit_model: Any) -> int:
        classes_attr = getattr(fit_model, "classes_", None)
        if classes_attr is None:
            raise ValueError(
                "Stored IPW estimator is missing classes_ needed for predict_proba()."
            )
        classes = list(classes_attr)
        if 1 not in classes:
            raise ValueError("Stored IPW estimator is missing class label 1.")
        return classes.index(1)

    def _compute_ipw_matrices(
        self,
        model: dict[str, Any],
        source: BalanceFrame | None = None,
    ) -> tuple[Any, Any]:
        """Compute IPW design matrices using stored model config.

        Args:
            model: The fitted model dict containing preprocessing config.
            source: BalanceFrame to extract covariates from.  When ``None``
                (default), uses ``self``.  When provided, uses ``source``'s
                covariates — for the ``data=`` holdout path.  Results are
                only cached when ``source is None`` (via the caller).
        """
        bf = source if source is not None else self
        if source is None:
            self._require_target()
        elif bf._sf_target is None:
            raise ValueError(
                "data must have a target set when computing design matrices."
            )

        sample_covars = bf._sf_sample.covars().df.copy()
        target_covars = _assert_type(bf._sf_target).covars().df.copy()

        transformations = model.get("transformations", "default")
        sample_covars, target_covars = balance_adjustment.apply_transformations(
            (sample_covars, target_covars),
            transformations=transformations,
        )
        columns: list[str] = _assert_type(model.get("X_matrix_columns"), list)

        na_action = cast(str, model.get("na_action", "add_indicator"))

        # Infer matrix_type from stored artifacts when not explicitly stored.
        from scipy.sparse import spmatrix

        matrix_type = model.get("fit_matrix_type")
        if matrix_type is None:
            fit_sample_matrix = model.get("model_matrix_sample")
            if isinstance(fit_sample_matrix, spmatrix):
                matrix_type = "sparse"
            elif isinstance(fit_sample_matrix, np.ndarray):
                matrix_type = "dense"
            elif isinstance(fit_sample_matrix, pd.DataFrame):
                matrix_type = "dataframe"

        result = build_design_matrix(
            sample_covars,
            target_covars,
            use_model_matrix=bool(model.get("use_model_matrix", True)),
            formula=model.get("formula"),
            one_hot_encoding=bool(model.get("one_hot_encoding", False)),
            na_action=na_action,
            project_to_columns=columns,
            fit_scaler=model.get("fit_scaler"),
            fit_penalties_skl=model.get("fit_penalties_skl"),
            matrix_type=matrix_type,
        )
        combined_matrix = result["combined_matrix"]
        sample_n = result["sample_n"]
        return combined_matrix[:sample_n], combined_matrix[sample_n:]

    @staticmethod
    def _is_artifact_stale(
        artifact: Any,
        stored_idx: pd.Index,
        current_idx: pd.Index,
    ) -> bool:
        """Check whether a stored artifact's index is stale vs current data."""
        if artifact is None:
            return True
        artifact_len = getattr(artifact, "shape", [0])[0]
        if artifact_len != len(current_idx):
            return True
        if len(stored_idx) == len(current_idx):
            same_set = (
                stored_idx.isin(current_idx).all()
                and current_idx.isin(stored_idx).all()
            )
            if not same_set:
                return True
        return False

    def _ensure_fresh_ipw_artifacts(
        self,
        model: dict[str, Any],
        side: Literal["sample", "target"],
    ) -> None:
        """Recompute and cache IPW matrices + predictions for one side if stale.

        Checks whether the stored matrix and predictions for the given side
        match the current data index. If stale, recomputes both matrices
        (sample + target — required for correct one-hot encoding) and
        predictions, then caches the results in the model dict.
        """
        if side == "sample":
            current_idx = self._sf_sample.df.index
        else:
            self._require_target()
            current_idx = _assert_type(self._sf_target).df.index

        stored_idx = pd.Index(model.get(f"{side}_index", current_idx))
        matrix = model.get(f"model_matrix_{side}")
        probability = model.get(f"{side}_probability")

        matrix_stale = self._is_artifact_stale(matrix, stored_idx, current_idx)
        prob_stale = self._is_artifact_stale(probability, stored_idx, current_idx)

        if not matrix_stale and not prob_stale:
            return

        # Recompute matrices (always produces both sides).
        sample_matrix, target_matrix = self._compute_ipw_matrices(model)
        # Cache both matrices — we already paid the cost of computing them.
        model["model_matrix_sample"] = sample_matrix
        model["model_matrix_target"] = target_matrix
        # Only update the index for the side being refreshed. Writing the
        # other side's index here would cause the staleness check to skip
        # recomputation on the next call, returning stale predictions.
        model[f"{side}_index"] = current_idx
        recomputed_matrix = sample_matrix if side == "sample" else target_matrix

        # Recompute predictions if stale.
        if prob_stale:
            from balance.weighting_methods.ipw import link_transform

            fit_model = _assert_type(model.get("fit"))
            class_index = self._ipw_class_index(fit_model)
            prob = np.asarray(
                fit_model.predict_proba(recomputed_matrix)[:, class_index]
            )
            link = link_transform(prob)
            model[f"{side}_probability"] = prob
            model[f"{side}_link"] = link
            model[f"{side}_index"] = current_idx

    def _validate_data_covariates(self, data: BalanceFrame) -> None:
        """Validate that ``data`` has matching covariate columns to self."""
        if set(data._sf_sample.covars().df.columns) != set(
            self._sf_sample.covars().df.columns
        ):
            raise ValueError(
                "data and self must have matching sample covariate column names."
            )
        if data._sf_target is not None and self._sf_target is not None:
            if set(_assert_type(data._sf_target).covars().df.columns) != set(
                _assert_type(self._sf_target).covars().df.columns
            ):
                raise ValueError(
                    "data and self must have matching target covariate column names."
                )

    @overload
    def design_matrix(  # noqa: E704
        self,
        on: Literal["sample"],
        *,
        data: BalanceFrame | None = ...,
    ) -> pd.DataFrame: ...

    @overload
    def design_matrix(  # noqa: E704
        self,
        on: Literal["target"],
        *,
        data: BalanceFrame | None = ...,
    ) -> pd.DataFrame: ...

    @overload
    def design_matrix(  # noqa: E704
        self,
        on: Literal["both"] = ...,
        *,
        data: BalanceFrame | None = ...,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    def design_matrix(
        self,
        on: Literal["sample", "target", "both"] = "both",
        *,
        data: BalanceFrame | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Return the IPW model's design matrices.

        Returns the model matrices (feature matrices) built by the stored
        preprocessing pipeline — after formula expansion, one-hot encoding,
        NA indicator addition, scaling, and penalty weighting.

        When ``data`` is provided, the stored preprocessing is applied to
        ``data``'s covariates and the result is returned without caching.
        When ``data`` is None (default), stored/cached matrices for this
        object's own data are returned (original behavior).

        Args:
            on: Which population's matrix to return.  ``"sample"`` returns the
                respondent matrix, ``"target"`` returns the target matrix, and
                ``"both"`` returns ``(sample_matrix, target_matrix)``.
            data: An optional BalanceFrame whose covariates are transformed
                using this object's stored preprocessing pipeline.  The
                ``data`` BalanceFrame does not need to be adjusted — it just
                provides covariates.  Must have matching covariate column names.

        Returns:
            A model-matrix DataFrame, or a tuple of two DataFrames when
            ``on="both"``.

        Raises:
            ValueError: If the object is not IPW-adjusted, if target is missing
                for ``on in {"target", "both"}``, if recomputation of sample-side
                artifacts is required but no target is available, if ``on`` is
                invalid, or if ``data`` has mismatched covariate columns.

        Notes:
            When ``data`` is None and stored fit artifacts are stale for the
            current rows (e.g., after ``set_fitted_model()``), this method
            recomputes and caches refreshed matrices.  That cache update is an
            intentional in-memory mutation.  When ``data`` is provided, no
            caching occurs.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [0.0, 1.0], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [0.2, 0.8], "weight": [1.0, 1.0]}))
            >>> adjusted = BalanceFrame(sample=resp, target=tgt).fit(method="ipw")
            >>> x_s, x_t = adjusted.design_matrix(on="both")
            >>> x_s.shape[0], x_t.shape[0]
            (2, 2)
        """
        if on not in ("sample", "target", "both"):
            raise ValueError("on must be one of: 'sample', 'target', 'both'")
        model = self._require_ipw_model()
        columns: list[str] = _assert_type(model.get("X_matrix_columns"), list)

        # --- data= path: compute from external data, no caching ---
        if data is not None:
            self._validate_data_covariates(data)
            if on in ("target", "both") and data._sf_target is None:
                raise ValueError(
                    "data must have a target set for on='target' or on='both'."
                )
            sample_matrix, target_matrix = self._compute_ipw_matrices(
                model, source=data
            )
            sample_df_result = self._matrix_to_dataframe(
                sample_matrix, data._sf_sample.df.index, columns
            )
            target_df_result = (
                self._matrix_to_dataframe(
                    target_matrix,
                    _assert_type(data._sf_target).df.index,
                    columns,
                )
                if data._sf_target is not None
                else None
            )
            if on == "sample":
                return sample_df_result
            if on == "target":
                return _assert_type(target_df_result)
            return (sample_df_result, _assert_type(target_df_result))

        # --- default path: use stored/cached artifacts ---
        sample_df: pd.DataFrame | None = None
        target_df: pd.DataFrame | None = None

        if on in ("sample", "both"):
            if model.get("model_matrix_sample") is None:
                raise ValueError(
                    "IPW model is missing fit-time sample matrix. Call "
                    "BalanceFrame.fit(method='ipw') or run ipw(..., "
                    "store_fit_matrices=True) before using design_matrix(on='sample'/'both')."
                )
            self._ensure_fresh_ipw_artifacts(model, "sample")
            sample_idx = pd.Index(model.get("sample_index", self._sf_sample.df.index))
            sample_df = cast(
                pd.DataFrame,
                self._align_to_index(
                    self._matrix_to_dataframe(
                        model["model_matrix_sample"], sample_idx, columns
                    ),
                    self._sf_sample.df.index,
                    caller="design_matrix()",
                ),
            )

        if on in ("target", "both"):
            self._require_target()
            if model.get("model_matrix_target") is None:
                raise ValueError(
                    "IPW model is missing fit-time target matrix. Call "
                    "BalanceFrame.fit(method='ipw') or run ipw(..., "
                    "store_fit_matrices=True) before using design_matrix(on='target'/'both')."
                )
            self._ensure_fresh_ipw_artifacts(model, "target")
            current_target_idx = _assert_type(self._sf_target).df.index
            target_idx = pd.Index(model.get("target_index", current_target_idx))
            target_df = cast(
                pd.DataFrame,
                self._align_to_index(
                    self._matrix_to_dataframe(
                        model["model_matrix_target"], target_idx, columns
                    ),
                    current_target_idx,
                    caller="design_matrix()",
                ),
            )

        if on == "sample":
            return _assert_type(sample_df)
        if on == "target":
            return _assert_type(target_df)
        return (_assert_type(sample_df), _assert_type(target_df))

    @overload
    def predict_proba(  # noqa: E704
        self,
        on: Literal["sample"],
        output: Literal["probability", "link"] = ...,
        *,
        data: BalanceFrame | None = ...,
    ) -> pd.Series: ...

    @overload
    def predict_proba(  # noqa: E704
        self,
        on: Literal["target"],
        output: Literal["probability", "link"] = ...,
        *,
        data: BalanceFrame | None = ...,
    ) -> pd.Series: ...

    @overload
    def predict_proba(  # noqa: E704
        self,
        on: Literal["both"] = ...,
        output: Literal["probability", "link"] = ...,
        *,
        data: BalanceFrame | None = ...,
    ) -> tuple[pd.Series, pd.Series]: ...

    def predict_proba(
        self,
        on: Literal["sample", "target", "both"] = "both",
        output: Literal["probability", "link"] = "probability",
        *,
        data: BalanceFrame | None = None,
    ) -> pd.Series | tuple[pd.Series, pd.Series]:
        """Return IPW propensity scores.

        Returns the propensity scores (predicted probabilities of being in
        the sample vs target) from the fitted IPW model.  A target row with
        high propensity is well-represented in the sample; a low score
        indicates underrepresentation.

        When ``data`` is provided, the stored model is applied to ``data``'s
        covariates and fresh predictions are returned without caching.  When
        ``data`` is None (default), stored/cached predictions for this
        object's own data are returned (original behavior).

        Args:
            on: Which population to predict on (``"sample"``, ``"target"``,
                or ``"both"``).
            output: Output scale. ``"probability"`` returns class-1 propensity
                probabilities. ``"link"`` returns logit-transformed values.
            data: An optional BalanceFrame whose covariates are scored using
                this object's stored model.  Must have matching covariate
                column names.  The ``data`` BalanceFrame needs a target for
                ``on="target"`` or ``on="both"``.

        Returns:
            A prediction Series, or a tuple of two Series when ``on="both"``.

        Raises:
            ValueError: If the object is not IPW-adjusted, if target is missing
                for ``on in {"target", "both"}``, if recomputation of sample-side
                predictions is required but no target is available, if ``on`` is
                invalid, or if ``data`` has mismatched covariate columns.

        Notes:
            When ``data`` is None and stored fit-time predictions are stale
            for the current rows, this method may recompute and cache refreshed
            probabilities/links.  When ``data`` is provided, no caching occurs.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [0.0, 1.0], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [0.2, 0.8], "weight": [1.0, 1.0]}))
            >>> adjusted = BalanceFrame(sample=resp, target=tgt).fit(method="ipw")
            >>> p = adjusted.predict_proba(on="target", output="probability")
            >>> int(p.shape[0])
            2
        """
        if output not in ("probability", "link"):
            raise ValueError("output must be one of: 'probability', 'link'")
        if on not in ("sample", "target", "both"):
            raise ValueError("on must be one of: 'sample', 'target', 'both'")
        model = self._require_ipw_model()

        # --- data= path: compute from external data, no caching ---
        if data is not None:
            from balance.weighting_methods.ipw import link_transform

            self._validate_data_covariates(data)
            if on in ("target", "both") and data._sf_target is None:
                raise ValueError(
                    "data must have a target set for on='target' or on='both'."
                )
            sample_matrix, target_matrix = self._compute_ipw_matrices(
                model, source=data
            )
            fit_model = _assert_type(model.get("fit"))
            class_index = self._ipw_class_index(fit_model)

            sample_series_result: pd.Series | None = None
            target_series_result: pd.Series | None = None

            if on in ("sample", "both"):
                prob = np.asarray(
                    fit_model.predict_proba(sample_matrix)[:, class_index]
                )
                if output == "link":
                    values_arr = link_transform(prob)
                else:
                    values_arr = prob
                sample_series_result = pd.Series(
                    values_arr, index=data._sf_sample.df.index
                )

            if on in ("target", "both"):
                prob = np.asarray(
                    fit_model.predict_proba(target_matrix)[:, class_index]
                )
                if output == "link":
                    values_arr = link_transform(prob)
                else:
                    values_arr = prob
                target_series_result = pd.Series(
                    values_arr, index=_assert_type(data._sf_target).df.index
                )

            if on == "sample":
                return _assert_type(sample_series_result)
            if on == "target":
                return _assert_type(target_series_result)
            return (
                _assert_type(sample_series_result),
                _assert_type(target_series_result),
            )

        # --- default path: use stored/cached artifacts ---
        sample_series: pd.Series | None = None
        target_series: pd.Series | None = None

        for side in ("sample", "target"):
            if side == "sample" and on not in ("sample", "both"):
                continue
            if side == "target" and on not in ("target", "both"):
                continue
            if side == "target":
                self._require_target()

            stored_key = (
                f"{side}_probability" if output == "probability" else f"{side}_link"
            )
            values = model.get(stored_key)
            if not isinstance(values, np.ndarray):
                raise ValueError(
                    f"IPW model is missing fit-time {side} predictions for predict_proba(). "
                    "Call BalanceFrame.fit(method='ipw') or run ipw(..., "
                    f"store_fit_metadata=True) before using predict_proba(on='{side}'/'both')."
                )

            self._ensure_fresh_ipw_artifacts(model, side)  # type: ignore[arg-type]
            # Re-read after potential refresh.
            values = model.get(stored_key)

            if side == "sample":
                current_idx = self._sf_sample.df.index
            else:
                current_idx = _assert_type(self._sf_target).df.index
            stored_idx = pd.Index(model.get(f"{side}_index", current_idx))

            series = cast(
                pd.Series,
                self._align_to_index(
                    pd.Series(_assert_type(values), index=stored_idx),
                    current_idx,
                    caller="predict_proba()",
                ),
            )
            if side == "sample":
                sample_series = series
            else:
                target_series = series

        if on == "sample":
            return _assert_type(sample_series)
        if on == "target":
            return _assert_type(target_series)
        return (_assert_type(sample_series), _assert_type(target_series))

    def predict_weights(
        self,
        *,
        data: BalanceFrame | None = None,
    ) -> pd.Series:
        """Predict responder weights from the fitted model's artifacts.

        Reconstructs adjusted survey weights from stored fit-time artifacts
        (propensity links, design weights, class balancing, trimming
        parameters).  On the fitted object itself, the result is numerically
        equivalent to ``self.weights().df`` (within floating-point tolerance)
        and serves as a validation that the stored artifacts are sufficient
        to reproduce the adjustment.

        When ``data`` is provided, computes weights for ``data``'s sample
        using the stored model, without caching.  This is the one-liner
        alternative to the ``set_fitted_model`` workflow::

            fitted.predict_weights(data=holdout_bf)

        When ``data`` is None (default), uses this object's own data
        (original behavior).

        Dispatches by the adjustment method stored in the model dict:

        - **IPW**: uses stored fit-time metadata (links, class balancing,
          trimming, and design weights) to reproduce fitted responder weights.
        - **CBPS**: rebuilds the CBPS scoring artifacts from stored metadata
          and supports both in-place and ``data=...`` holdout scoring.
        - **Poststratify**: reconstructs in-place responder weights from
          stored cell-ratio metadata; ``data=...`` holdout scoring is not yet
          supported for poststratify.
        - **Rake**: replays fitted cell-ratio artifacts in-place and supports
          ``data=...`` transfer scoring. See :func:`balance.weighting_methods.rake.rake`
          Notes for validity constraints and interpretation.
        - **Other methods**: not yet supported — will raise with guidance.

        Args:
            data: An optional BalanceFrame whose sample covariates are scored
                using this object's stored model.  Must have matching covariate
                column names and a target set. Supported for IPW, CBPS, and rake;
                poststratify currently supports only in-place scoring
                (``data=None``).

        Returns:
            A Series of predicted responder weights.

        Raises:
            ValueError: If no fitted model is available, if the method is
                unsupported, if required target data is missing, or if
                ``data`` has mismatched covariate columns.

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
        # NOTE: The data= path intentionally duplicates the weight computation
        # from set_fitted_model() rather than calling it. set_fitted_model()
        # could be reused (scored = data.set_fitted_model(self, inplace=False);
        # return scored.weight_series), but:
        # 1. It does unnecessary work (deepcopy, _links, _adjustment_model)
        #    that would be immediately discarded.
        # 2. design_matrix(data=...) and predict_proba(data=...) cannot use
        #    the same pattern — they need intermediate results (matrices,
        #    probabilities), not final weights. Keeping all three data= paths
        #    parallel is clearer.
        if data is not None:
            self._validate_data_covariates(data)
            model = self._require_fitted_model()
            method = model.get("method")
            if method == "cbps":
                return self._predict_weights_cbps(model, source=data)
            if method == "rake":
                return self._predict_weights_rake(model, source=data)
            if method != "ipw":
                raise ValueError(
                    f"predict_weights(data=...) is not yet supported for method '{method}'. "
                    "Currently only 'ipw', 'cbps', and 'rake' are supported."
                )
            from balance.weighting_methods.ipw import link_transform, weights_from_link

            fit_obj = model.get("fit")
            columns = model.get("X_matrix_columns")
            if fit_obj is None or not isinstance(columns, list):
                raise ValueError(
                    "IPW model metadata is missing fitted model information."
                )

            sample_matrix, _target_matrix = self._compute_ipw_matrices(
                model, source=data
            )
            class_index = self._ipw_class_index(fit_obj)
            prob = np.asarray(fit_obj.predict_proba(sample_matrix)[:, class_index])
            link = link_transform(prob)

            if data._sf_target is None:
                raise ValueError("data must have a target set for predict_weights().")
            sample_weights = data._sf_sample.df_weights.iloc[:, 0]
            target_weights = _assert_type(data._sf_target).df_weights.iloc[:, 0]

            training_target_weights = model.get("training_target_weights")
            if isinstance(training_target_weights, pd.Series):
                train_sum = training_target_weights.sum()
                data_sum = target_weights.sum()
                if train_sum > 0 and abs(train_sum - data_sum) / train_sum > 0.01:
                    logger.warning(
                        "predict_weights(data=...): data's target weights sum (%.2f) "
                        "differs from training target weights sum (%.2f). The "
                        "balance_classes correction and weight normalization will "
                        "use data's weights, which may produce different results "
                        "than the training fit.",
                        target_weights.sum(),
                        training_target_weights.sum(),
                    )

            predicted = weights_from_link(
                link=link,
                balance_classes=bool(model.get("balance_classes", True)),
                sample_weights=sample_weights,
                target_weights=target_weights,
                weight_trimming_mean_ratio=model.get("weight_trimming_mean_ratio"),
                weight_trimming_percentile=model.get("weight_trimming_percentile"),
            )
            weight_name = getattr(data._sf_sample.weight_series, "name", None)
            return pd.Series(predicted.values, index=data._sf_sample.df.index).rename(
                weight_name
            )

        self._require_target()
        model = self._require_fitted_model()
        method = model.get("method")

        if method == "ipw":
            return self._predict_weights_ipw(model)
        if method == "cbps":
            return self._predict_weights_cbps(model)
        if method == "poststratify":
            return self._predict_weights_poststratify(model)
        if method == "rake":
            return self._predict_weights_rake(model)

        raise ValueError(
            f"predict_weights() is not yet supported for method '{method}'. "
            "Currently only 'ipw', 'cbps', 'poststratify', and 'rake' are supported. "
            "Use adjust() to obtain weights directly for other methods."
        )

    def _predict_weights_rake(
        self,
        model: dict[str, Any],
        source: BalanceFrame | None = None,
    ) -> pd.Series:
        required = (
            "variables",
            "variables_before_transformations",
            "categories",
            "m_fit",
            "m_sample",
            "na_action",
            "transformations",
        )
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
        if (
            not isinstance(variables, list)
            or not isinstance(input_variables, list)
            or not isinstance(categories, list)
        ):
            raise ValueError("Rake model metadata is malformed for predict_weights().")
        if not isinstance(m_fit, np.ndarray) or not isinstance(m_sample, np.ndarray):
            raise ValueError("Rake model is missing stored contingency tables.")

        bf = source if source is not None else self
        if source is not None and bf._sf_target is None:
            raise ValueError(
                "data must have a target set for rake predict_weights(data=...)."
            )
        sample_covars = bf._sf_sample.df_covars
        target_covars = _assert_type(bf._sf_target).df_covars
        for column in input_variables:
            if column not in sample_covars.columns:
                raise ValueError(
                    f"Rake predict_weights() cannot find required covariate '{column}'."
                )
        sample_df = sample_covars.loc[:, input_variables]
        target_df = target_covars.loc[:, input_variables]
        sample_weights_full = bf._sf_sample.df_weights.iloc[:, 0]
        training_sample_weights = model.get("training_sample_weights")
        sample_weights = sample_weights_full
        na_action = cast(str, model.get("na_action", "add_indicator"))
        if source is None:
            if isinstance(
                training_sample_weights, pd.Series
            ) and training_sample_weights.index.equals(sample_weights_full.index):
                sample_weights = training_sample_weights
            elif na_action != "drop":
                raise ValueError(
                    "Rake predict_weights() requires compatible fit-time sample design "
                    "weights for in-place replay. This can happen because "
                    "store_fit_metadata is missing/incompatible, or because you're "
                    "scoring a different sample; use predict_weights(data=...) "
                    "for different samples."
                )

        if na_action == "drop":
            sample_df, sample_weights = balance_util.drop_na_rows(
                sample_df, sample_weights, "sample"
            )
            target_df, _target_weights = balance_util.drop_na_rows(
                target_df,
                _assert_type(bf._sf_target).df_weights.iloc[:, 0],
                "target",
            )
        elif na_action == "add_indicator":
            sample_df = pd.DataFrame(_safe_fillna_and_infer(sample_df, "__NaN__"))
            target_df = pd.DataFrame(_safe_fillna_and_infer(target_df, "__NaN__"))
        else:
            raise ValueError(
                f"Rake model has invalid na_action metadata '{na_action}' for predict_weights()."
            )

        sample_df, _target_df = balance_adjustment.apply_transformations(
            (sample_df, target_df), transformations=model.get("transformations")
        )
        for column in variables:
            if column not in sample_df.columns:
                raise ValueError(
                    "Rake transform output is missing stored variable "
                    f"'{column}' required for predict_weights()."
                )
        sample_df = sample_df.loc[:, variables].astype(str)
        if m_fit.shape != m_sample.shape:
            raise ValueError(
                "Rake model metadata has incompatible fitted and sample table shapes."
            )

        ratio = np.divide(
            m_fit,
            m_sample,
            out=np.zeros_like(m_fit, dtype=float),
            where=m_sample != 0,
        )
        index = pd.MultiIndex.from_product(categories, names=variables)
        if source is not None:
            score_joint = (
                sample_df.assign(_w=sample_weights)
                .groupby(variables)["_w"]
                .sum()
                .reindex(index, fill_value=0)
                .to_numpy()
                .reshape(m_sample.shape)
            )
            divergence = _rake_joint_distribution_divergence(m_sample, score_joint)
            if divergence > 0.02:
                logger.warning(
                    "Rake predict_weights(data=...): scoring sample joint distribution "
                    "diverges from training joint distribution (TV=%.4f). "
                    "Transferred rake weights may fail to recover target marginals; "
                    "re-fit rake on the scoring sample for exact marginal matching.",
                    divergence,
                )
        ratio_series = pd.Series(ratio.flatten(), index=index, name="_rake_ratio")
        joined = sample_df.join(ratio_series, on=variables)
        if bool(joined["_rake_ratio"].isna().any()):
            raise ValueError(
                "Rake predict_weights() found rows that do not map to stored fit-time "
                "categories. Re-fit with compatible covariates."
            )
        raw = sample_weights * joined["_rake_ratio"]
        target_weights = model.get("training_target_weights")
        if source is None and not isinstance(target_weights, pd.Series):
            raise ValueError(
                "Rake predict_weights() requires compatible fit-time target design "
                "weights for in-place replay. This can happen because "
                "store_fit_metadata is missing/incompatible, or because you're "
                "scoring a different sample; use predict_weights(data=...) "
                "for different samples."
            )
        target_sum = (
            float(target_weights.sum())
            if isinstance(target_weights, pd.Series)
            else float(_assert_type(bf._sf_target).df_weights.iloc[:, 0].sum())
        )
        predicted = balance_adjustment.trim_weights(
            raw,
            target_sum_weights=target_sum,
            weight_trimming_mean_ratio=model.get("weight_trimming_mean_ratio"),
            weight_trimming_percentile=model.get("weight_trimming_percentile"),
            keep_sum_of_weights=bool(model.get("keep_sum_of_weights", True)),
        )
        weight_name = getattr(_assert_type(bf.weight_series), "name", None)
        if na_action == "drop":
            predicted_full = pd.Series(
                np.nan, index=sample_weights_full.index, dtype=float
            ).rename(predicted.name)
            predicted_full.loc[predicted.index] = predicted.to_numpy()
            predicted = predicted_full
        return cast(
            pd.Series,
            self._align_to_index(
                predicted.rename(weight_name),
                bf._sf_sample.df.index,
                caller="predict_weights()",
                method_name="rake",
            ),
        )

    def _resolve_ipw_link(self, model: dict[str, Any]) -> np.ndarray:
        """Resolve sample link values from stored artifacts or recomputation."""
        model_link = model.get("sample_link")
        current_sample_idx = self._sf_sample.df.index
        model_sample_idx = pd.Index(model.get("sample_index", current_sample_idx))
        if (
            isinstance(model_link, np.ndarray)
            and model_link.shape[0] == len(current_sample_idx)
            and model_sample_idx.equals(current_sample_idx)
        ):
            return model_link
        return self.predict_proba(on="sample", output="link").to_numpy()

    def _resolve_design_weights(
        self,
        model: dict[str, Any],
        link: np.ndarray,
    ) -> tuple[pd.Series, pd.Series]:
        """Resolve sample and target design weights for predict_weights().

        Uses stored training weights when available and compatible; falls
        back to current design weights with a warning otherwise.
        """
        current_sample_weights = self._sf_sample.df_weights.iloc[:, 0]
        current_target_weights = _assert_type(self._sf_target).df_weights.iloc[:, 0]

        sample_weights = model.get("training_sample_weights")
        if (
            not isinstance(sample_weights, pd.Series)
            or len(sample_weights) != len(link)
            or not sample_weights.index.equals(current_sample_weights.index)
        ):
            logger.warning(
                "Falling back to current sample design weights in predict_weights(); "
                "stored training_sample_weights are unavailable or incompatible."
            )
            sample_weights = current_sample_weights

        target_weights = model.get("training_target_weights")
        if (
            not isinstance(target_weights, pd.Series)
            or len(target_weights) != len(current_target_weights)
            or not target_weights.index.equals(current_target_weights.index)
        ):
            logger.warning(
                "Falling back to current target design weights in predict_weights(); "
                "stored training_target_weights are unavailable or incompatible."
            )
            target_weights = current_target_weights

        return sample_weights, target_weights

    def _predict_weights_ipw(self, model: dict[str, Any]) -> pd.Series:
        """IPW-specific weight prediction from stored fit artifacts."""
        from balance.weighting_methods.ipw import weights_from_link

        fit = model.get("fit")
        columns = model.get("X_matrix_columns")
        if fit is None or not isinstance(columns, list):
            raise ValueError("IPW model metadata is missing fitted model information.")

        link = self._resolve_ipw_link(model)
        sample_weights, target_weights = self._resolve_design_weights(model, link)

        predicted = weights_from_link(
            link=link,
            balance_classes=bool(model.get("balance_classes", True)),
            sample_weights=sample_weights,
            target_weights=target_weights,
            weight_trimming_mean_ratio=model.get("weight_trimming_mean_ratio"),
            weight_trimming_percentile=model.get("weight_trimming_percentile"),
        )
        current_sample_idx = self._sf_sample.df.index
        model_sample_idx = pd.Index(model.get("sample_index", current_sample_idx))
        sample_idx = model_sample_idx
        if len(sample_idx) != len(predicted) or not sample_idx.equals(
            current_sample_idx
        ):
            sample_idx = current_sample_idx
        weight_name = getattr(_assert_type(self.weight_series), "name", None)
        return cast(
            pd.Series,
            self._align_to_index(
                pd.Series(predicted.values, index=sample_idx),
                self._sf_sample.df.index,
                caller="predict_weights()",
            ),
        ).rename(weight_name)

    @staticmethod
    def _validate_cbps_metadata(
        model: dict[str, Any],
    ) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, list[str], str]:
        """Validate CBPS fit metadata and return the artifacts needed for scoring."""
        required_metadata = (
            "variables",
            "na_action",
            "model_matrix_mean",
            "model_matrix_std",
            "formula",
            "transformations",
            "beta_optimal_model_space",
            "svd_s",
            "svd_Vh",
            "X_matrix_columns",
        )
        missing = [key for key in required_metadata if key not in model]
        if missing:
            raise ValueError(
                "CBPS model is missing fit-time metadata "
                f"({missing}) for predict_weights(). "
                "Call BalanceFrame.fit(method='cbps') or run cbps(..., "
                "store_fit_metadata=True)."
            )
        beta_opt_model_space = model.get("beta_optimal_model_space")
        fit_columns = model.get("X_matrix_columns")
        svd_s = model.get("svd_s")
        svd_Vh = model.get("svd_Vh")
        variables = model.get("variables")
        if (
            not isinstance(beta_opt_model_space, np.ndarray)
            or not isinstance(fit_columns, list)
            or not isinstance(svd_s, np.ndarray)
            or not isinstance(svd_Vh, np.ndarray)
        ):
            raise ValueError(
                "CBPS model metadata is malformed for predict_weights() scoring."
            )
        if not isinstance(variables, list):
            raise ValueError(
                "CBPS model is missing fit-time variables for predict_weights(). "
                "Call BalanceFrame.fit(method='cbps') or run cbps(..., "
                "store_fit_metadata=True)."
            )
        na_action = cast(str, model.get("na_action", "add_indicator"))
        if na_action == "drop":
            raise ValueError(
                "predict_weights() is unsupported for CBPS models fitted with "
                "na_action='drop' because dropped rows cannot be reconstructed "
                "reliably. Re-fit with na_action='add_indicator'."
            )
        return (
            beta_opt_model_space,
            fit_columns,
            svd_s,
            svd_Vh,
            cast(list[str], variables),
            na_action,
        )

    def _build_cbps_scoring_matrix(
        self,
        bf: BalanceFrame,
        model: dict[str, Any],
        fit_columns: list[str],
        variables: list[str],
        na_action: str,
        svd_s: np.ndarray,
        svd_Vh: np.ndarray,
        beta_opt_model_space: np.ndarray,
    ) -> tuple[np.ndarray, int, int]:
        """Rebuild the CBPS scoring matrix ``U`` from stored fit artifacts."""
        sample_covars = bf._sf_sample.covars().df.copy()
        target_covars = _assert_type(bf._sf_target).covars().df.copy()
        sample_covars, target_covars = balance_adjustment.apply_transformations(
            (sample_covars, target_covars),
            transformations=model.get("transformations", "default"),
        )
        sample_covars = sample_covars.loc[:, variables]
        target_covars = target_covars.loc[:, variables]

        projected_columns = [c for c in fit_columns if c != "Intercept"]
        matrix_out = build_design_matrix(
            sample_covars,
            target_covars,
            use_model_matrix=True,
            formula=model.get("formula"),
            one_hot_encoding=False,
            na_action=na_action,
            project_to_columns=projected_columns,
            matrix_type="dense",
        )
        combined_matrix = np.asarray(matrix_out["combined_matrix"])
        sample_n = cast(int, matrix_out["sample_n"])
        target_n = combined_matrix.shape[0] - sample_n

        model_matrix_mean = np.asarray(model.get("model_matrix_mean"), dtype=float)
        model_matrix_std = np.asarray(model.get("model_matrix_std"), dtype=float)
        if (
            model_matrix_mean.shape[0] != combined_matrix.shape[1]
            or model_matrix_std.shape[0] != combined_matrix.shape[1]
        ):
            raise ValueError(
                "CBPS model metadata has incompatible standardization vectors for "
                "predict_weights()."
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            combined_matrix = (combined_matrix - model_matrix_mean) / model_matrix_std
        X_matrix = np.c_[np.ones(combined_matrix.shape[0]), combined_matrix]
        if np.any(~np.isfinite(X_matrix)):
            raise ValueError(
                "CBPS predict_weights() produced non-finite standardized design "
                "matrix values."
            )
        if svd_Vh.shape[1] != X_matrix.shape[1]:
            raise ValueError(
                "CBPS model metadata has incompatible SVD components for "
                "predict_weights()."
            )
        if svd_s.shape[0] != svd_Vh.shape[0]:
            raise ValueError(
                "CBPS model metadata has inconsistent SVD dimensions for "
                "predict_weights()."
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            s_inv = np.where(svd_s > 1e-10, 1.0 / svd_s, 0.0)
        U_matrix = np.matmul(X_matrix, svd_Vh.T * s_inv)
        if U_matrix.shape[1] != beta_opt_model_space.shape[0]:
            raise ValueError(
                "CBPS model metadata has incompatible coefficient dimensions for "
                "predict_weights()."
            )
        return U_matrix, sample_n, target_n

    @staticmethod
    def _resolve_cbps_training_weights(
        bf: BalanceFrame,
        model: dict[str, Any],
        use_training_weights: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (sample_weights, target_weights) for CBPS scoring.

        For in-place scoring (``use_training_weights=True``) the stored
        training weights are required: falling back to the active
        ``df_weights`` would silently use the *adjusted* weights after
        ``fit()`` and produce incorrect predictions.
        """
        if use_training_weights:
            training_sample_weights = model.get("training_sample_weights")
            training_target_weights = model.get("training_target_weights")
            if not (
                isinstance(training_sample_weights, pd.Series)
                and isinstance(training_target_weights, pd.Series)
                and training_sample_weights.index.equals(bf._sf_sample.df.index)
                and training_target_weights.index.equals(
                    _assert_type(bf._sf_target).df.index
                )
            ):
                raise ValueError(
                    "CBPS predict_weights() requires stored training weights that "
                    "align with the current sample/target indices. Re-fit with "
                    "BalanceFrame.fit(method='cbps') so the training weights are "
                    "persisted, or pass data=... to score a holdout frame."
                )
            return (
                training_sample_weights.to_numpy(),
                training_target_weights.to_numpy(),
            )
        sample_weights = bf._sf_sample.df_weights.iloc[:, 0].to_numpy()
        target_weights = _assert_type(bf._sf_target).df_weights.iloc[:, 0].to_numpy()
        return sample_weights, target_weights

    @staticmethod
    def _compute_cbps_design_weights(
        sample_weights: np.ndarray,
        target_weights: np.ndarray,
        balance_classes: bool,
    ) -> np.ndarray:
        """Compute CBPS design weights with explicit zero-sum guards."""
        sample_sum = float(np.sum(sample_weights))
        target_sum = float(np.sum(target_weights))
        if sample_sum <= 0 or target_sum <= 0:
            raise ValueError(
                "CBPS predict_weights() requires positive sample and target weight "
                "sums; got sample_sum={sample_sum}, target_sum={target_sum}.".format(
                    sample_sum=sample_sum, target_sum=target_sum
                )
            )
        sample_n = sample_weights.shape[0]
        target_n = target_weights.shape[0]
        if balance_classes:
            total_n = sample_n + target_n
            return np.asarray(
                total_n
                / 2
                * np.concatenate(
                    (sample_weights / sample_sum, target_weights / target_sum)
                )
            )
        design_weights = np.concatenate((sample_weights, target_weights))
        design_mean = float(np.mean(design_weights))
        if design_mean <= 0:
            raise ValueError(
                "CBPS predict_weights() requires a positive mean of combined "
                "sample+target weights when balance_classes=False."
            )
        return design_weights / design_mean

    def _predict_weights_cbps(
        self,
        model: dict[str, Any],
        source: BalanceFrame | None = None,
    ) -> pd.Series:
        """CBPS-specific weight prediction from stored fit artifacts."""
        from balance.weighting_methods.cbps import (
            compute_pseudo_weights_from_logit_probs,
            logit_truncated,
        )

        bf = source if source is not None else self
        if bf._sf_target is None:
            raise ValueError("data must have a target set for predict_weights().")

        (
            beta_opt_model_space,
            fit_columns,
            svd_s,
            svd_Vh,
            variables,
            na_action,
        ) = self._validate_cbps_metadata(model)

        U_matrix, sample_n, _ = self._build_cbps_scoring_matrix(
            bf,
            model,
            fit_columns,
            variables,
            na_action,
            svd_s,
            svd_Vh,
            beta_opt_model_space,
        )

        sample_weights, target_weights = self._resolve_cbps_training_weights(
            bf, model, use_training_weights=source is None
        )
        sample_n_actual = sample_weights.shape[0]
        target_n_actual = target_weights.shape[0]
        if sample_n != sample_n_actual:
            raise ValueError(
                "CBPS predict_weights() failed due to sample row misalignment while "
                "rebuilding the model matrix."
            )
        if (sample_n_actual + target_n_actual) != U_matrix.shape[0]:
            raise ValueError(
                "CBPS predict_weights() failed because rebuilt CBPS matrix rows do "
                "not match sample+target weights."
            )

        design_weights = self._compute_cbps_design_weights(
            sample_weights,
            target_weights,
            balance_classes=bool(model.get("balance_classes", True)),
        )
        in_pop = np.concatenate(
            (np.zeros(sample_n_actual), np.ones(target_n_actual))
        ).astype(float)

        probs = logit_truncated(U_matrix, beta_opt_model_space)
        pseudo = np.abs(
            compute_pseudo_weights_from_logit_probs(probs, design_weights, in_pop)
        )
        weights = design_weights[:sample_n_actual] * pseudo[:sample_n_actual]
        if np.any(~np.isfinite(weights)):
            raise ValueError(
                "CBPS predict_weights() produced non-finite intermediate weights."
            )
        weights = balance_adjustment.trim_weights(
            weights,
            model.get("weight_trimming_mean_ratio"),
            model.get("weight_trimming_percentile"),
        )
        weights = np.asarray(weights, dtype=float)
        if np.any(~np.isfinite(weights)):
            raise ValueError(
                "CBPS predict_weights() produced non-finite trimmed weights."
            )
        original_sum = float(np.sum(weights))
        if original_sum <= 0:
            raise ValueError("CBPS predict_weights() produced non-positive weight sum.")
        weights = weights / original_sum * float(np.sum(target_weights))
        return pd.Series(weights, index=bf._sf_sample.df.index).rename(
            getattr(bf._sf_sample.weight_series, "name", None)
        )

    def _predict_weights_poststratify(self, model: dict[str, Any]) -> pd.Series:
        """Poststratify-specific weight reconstruction from stored fit artifacts."""
        required = (
            "variables",
            "variables_before_transformations",
            "na_action",
            "strict_matching",
            "transformations",
            "transformations_drop",
            "weight_trimming_mean_ratio",
            "weight_trimming_percentile",
            "keep_sum_of_weights",
            "cell_weight_ratio",
        )
        missing = [key for key in required if key not in model]
        if missing:
            raise ValueError(
                "Poststratify model is missing fit-time metadata "
                f"({missing}) for predict_weights(). "
                "Call BalanceFrame.fit(method='poststratify') or run "
                "poststratify(..., store_fit_metadata=True)."
            )

        variables = model.get("variables")
        input_variables = model.get("variables_before_transformations")
        if not isinstance(variables, list) or not all(
            isinstance(v, str) for v in variables
        ):
            raise ValueError(
                "Poststratify model has invalid 'variables' metadata for "
                "predict_weights()."
            )
        if not isinstance(input_variables, list) or not all(
            isinstance(v, str) for v in input_variables
        ):
            raise ValueError(
                "Poststratify model has invalid "
                "'variables_before_transformations' metadata for predict_weights()."
            )

        ratio_series = model.get("cell_weight_ratio")
        if not isinstance(ratio_series, pd.Series):
            raise ValueError(
                "Poststratify model is missing cell-weight ratio metadata for "
                "predict_weights()."
            )

        current_sample_weights = self._sf_sample.df_weights.iloc[:, 0]
        current_target_weights = _assert_type(self._sf_target).df_weights.iloc[:, 0]

        sample_weights = model.get("training_sample_weights")
        if (
            not isinstance(sample_weights, pd.Series)
            or len(sample_weights) != len(current_sample_weights)
            or not sample_weights.index.equals(current_sample_weights.index)
        ):
            raise ValueError(
                "Poststratify predict_weights() requires compatible "
                "fit-time sample design weights in model['training_sample_weights']. "
                "Re-fit with BalanceFrame.fit(method='poststratify') and "
                "store_fit_metadata=True."
            )

        target_weights = model.get("training_target_weights")
        if (
            not isinstance(target_weights, pd.Series)
            or len(target_weights) != len(current_target_weights)
            or not target_weights.index.equals(current_target_weights.index)
        ):
            raise ValueError(
                "Poststratify predict_weights() requires compatible "
                "fit-time target design weights in model['training_target_weights']. "
                "Re-fit with BalanceFrame.fit(method='poststratify') and "
                "store_fit_metadata=True."
            )

        sample_covars = self._sf_sample.df_covars
        target_covars = _assert_type(self._sf_target).df_covars
        for column in input_variables:
            if (
                column not in sample_covars.columns
                or column not in target_covars.columns
            ):
                raise ValueError(
                    "Poststratify predict_weights() cannot find required covariate "
                    f"'{column}' in both sample and target."
                )
        sample_df = sample_covars.loc[:, input_variables]
        target_df = target_covars.loc[:, input_variables]

        na_action = cast(str, model.get("na_action", "add_indicator"))
        if na_action == "drop":
            sample_df, sample_weights = balance_util.drop_na_rows(
                sample_df, sample_weights, "sample"
            )
            target_df, target_weights = balance_util.drop_na_rows(
                target_df, target_weights, "target"
            )
        elif na_action == "add_indicator":
            sample_df = pd.DataFrame(_safe_fillna_and_infer(sample_df, "__NaN__"))
            target_df = pd.DataFrame(_safe_fillna_and_infer(target_df, "__NaN__"))
        else:
            raise ValueError(
                "Poststratify model has invalid na_action metadata "
                f"'{na_action}' for predict_weights()."
            )

        sample_df, _target_df = balance_adjustment.apply_transformations(
            (sample_df, target_df),
            transformations=model["transformations"],
            drop=bool(model["transformations_drop"]),
        )
        for column in variables:
            if column not in sample_df.columns:
                raise ValueError(
                    "Poststratify transform output is missing stored variable "
                    f"'{column}' required for predict_weights()."
                )
        sample_df = sample_df.loc[:, variables]

        ratio_name = "_cell_ratio"
        while ratio_name in sample_df.columns:
            ratio_name = f"{ratio_name}_tmp"
        sample_with_ratio = sample_df.join(
            ratio_series.rename(ratio_name), on=variables
        )
        missing_ratio = sample_with_ratio[ratio_name].isna()
        if bool(missing_ratio.any()):
            if bool(model.get("strict_matching", True)):
                raise ValueError(
                    "Poststratify predict_weights() found sample cells missing from "
                    "stored fit-time cell ratios. Re-fit with compatible cells or "
                    "fit with strict_matching=False."
                )
            logger.warning(
                "Poststratify predict_weights() encountered sample cells missing from "
                "stored fit-time cell ratios; assigning zero weights to those rows."
            )
            sample_with_ratio[ratio_name] = _safe_fillna_and_infer(
                sample_with_ratio[ratio_name], 0
            )

        raw_weights = sample_with_ratio[ratio_name] * sample_weights
        target_total = raw_weights.sum()
        trimmed = balance_adjustment.trim_weights(
            raw_weights,
            target_sum_weights=target_total,
            weight_trimming_mean_ratio=model["weight_trimming_mean_ratio"],
            weight_trimming_percentile=model["weight_trimming_percentile"],
            keep_sum_of_weights=bool(model["keep_sum_of_weights"]),
        )
        if na_action == "drop":
            # Align back to the full fit-time sample index so rows dropped for
            # missing covariates retain NaN weights, matching adjust() behavior.
            full_index = current_sample_weights.index
            trimmed_full = pd.Series(np.nan, index=full_index, dtype=float).rename(
                trimmed.name
            )
            trimmed_full.loc[trimmed.index] = trimmed.to_numpy()
            trimmed = trimmed_full
        weight_name = getattr(_assert_type(self.weight_series), "name", None)
        return cast(
            pd.Series,
            self._align_to_index(
                trimmed,
                self._sf_sample.df.index,
                caller="predict_weights()",
                method_name="poststratify",
            ).rename(weight_name),
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
        # pyrefly: ignore [unsupported-operation]
        target_sf = SampleFrame.from_sample(sample._links["target"])

        bf = cls._create(sample=responders_sf, target=target_sf)

        if sample.is_adjusted():
            # Set unadjusted to a DIFFERENT SampleFrame so is_adjusted returns True
            bf._sf_sample_pre_adjust = SampleFrame.from_sample(
                # pyrefly: ignore [unsupported-operation]
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
            id_column=self._sf_sample._id_column_name,
            weight_column=self._sf_sample._weight_column_name,
            outcome_columns=self._sf_sample.outcome_columns or None,
            ignored_columns=self._sf_sample.ignored_columns or None,
            standardize_types=False,
        )
        target_sample = Sample.from_frame(
            target._df,
            id_column=target._id_column_name,
            weight_column=target._weight_column_name,
            outcome_columns=target.outcome_columns or None,
            ignored_columns=target.ignored_columns or None,
            standardize_types=False,
        )
        result = resp_sample.set_target(target_sample)

        if self.is_adjusted and self._sf_sample_pre_adjust is not None:
            unadj_sf = SampleFrame.from_frame(
                self._sf_sample_pre_adjust._df,
                id_column=self._sf_sample_pre_adjust._id_column_name,
                weight_column=self._sf_sample_pre_adjust._weight_column_name,
                outcome_columns=self._sf_sample_pre_adjust.outcome_columns or None,
                ignored_columns=self._sf_sample_pre_adjust.ignored_columns or None,
                standardize_types=False,
            )
            result._sf_sample_pre_adjust = unadj_sf
            # pyrefly: ignore [unsupported-operation]
            result._links["unadjusted"] = unadj_sf
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
                # pyrefly: ignore [missing-attribute]
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
            # pyrefly: ignore [missing-attribute]
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
        # pyrefly: ignore [no-matching-overload]
        return pd.concat(
            (
                self.id_series,
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
            # pyrefly: ignore [missing-attribute]
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

        # pyrefly: ignore [missing-attribute]
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

        .. warning::

            If this BalanceFrame has already been fitted (i.e., ``adjust()``
            has been called), calling ``set_weights()`` changes the design
            weights but does **not** invalidate the stored fit artifacts
            (``_adjustment_model``).  The link values
            in those artifacts were computed using the old weights, so
            ``predict_weights()`` will use new ``current_sample_weights``
            with stale links, producing a mathematical inconsistency.
            Users should re-fit (call ``adjust()`` again) after changing
            weights on an already-fitted BalanceFrame.

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
        inplace: bool = False,
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
            inplace: If True, mutate this BalanceFrame's weights and
                return it.  If False (default), return a new BalanceFrame.

        Returns:
            The BalanceFrame with trimmed weights (self if *inplace*,
            else a new instance).
        """
        if inplace:
            self._sf_sample.trim(
                ratio=ratio,
                percentile=percentile,
                keep_sum_of_weights=keep_sum_of_weights,
                target_sum_weights=target_sum_weights,
                inplace=True,
            )
            return self

        new_sf = self._sf_sample.trim(
            ratio=ratio,
            percentile=percentile,
            keep_sum_of_weights=keep_sum_of_weights,
            target_sum_weights=target_sum_weights,
            inplace=False,
        )
        new_bf = type(self)._create(
            sample=new_sf,
            target=self._sf_target,
        )
        new_bf._sf_sample_pre_adjust = self._sf_sample_pre_adjust
        new_bf._adjustment_model = self._adjustment_model
        # Preserve existing links (target, unadjusted).
        # pyrefly: ignore [missing-attribute]
        for key, val in self._links.items():
            # pyrefly: ignore [unsupported-operation]
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
        # pyrefly: ignore [unsupported-operation]
        new_bf._links["unadjusted"] = second
        new_bf._sf_sample_pre_adjust = second._sf_sample
        return new_bf

    # --- Column accessors (moved from Sample) ---

    def _special_columns_names(self) -> list[str]:
        """Return names of all special columns (id, weight, outcome, ignored)."""
        return (
            [str(i.name) for i in [self.id_series, self.weight_series] if i is not None]
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
        id_column_name = self.id_series.name if self.id_series is not None else "None"
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
                # pyrefly: ignore [unsupported-operation]
                self,
                # pyrefly: ignore [unsupported-operation]
                self._links["target"],
                variables=None,
            )
            # pyrefly: ignore [unsupported-operation]
            target_str = self._links["target"].__str__().replace("\n", "\n\t")
            n_common = len(common_variables)
            common_variables = ",".join(common_variables)
            desc += f"""
            target:
                 {target_str}
            {n_common} common variables: {common_variables}
            """
        return desc
