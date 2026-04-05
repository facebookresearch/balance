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

if TYPE_CHECKING:
    from typing import Self

    from balance.balancedf_class import BalanceDFSource  # noqa: F401

# The set of string method names accepted by _find_adjustment_method.
_AdjustmentMethodStr = Literal["cbps", "ipw", "null", "poststratify", "rake"]

logger: logging.Logger = logging.getLogger(__package__)


class _CallableBool:
    """A bool-like value that is also callable, for backward-compatible property migration.

    This allows ``is_adjusted`` to work both as a property (``sample.is_adjusted``)
    and as a method call (``sample.is_adjusted()``) for backward compatibility.

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
    ``BalanceFrame(sf_with_outcomes=..., sf_target=...)`` which delegates to the
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
        >>> bf = BalanceFrame(sf_with_outcomes=resp, sf_target=tgt)
        >>> bf.is_adjusted
        False
        >>> adjusted = bf.adjust(method="ipw")
        >>> adjusted.is_adjusted
        True
        >>> bf.is_adjusted  # original unchanged
        False
    """

    # pyre-fixme[13]: Attributes are initialized in _create() / from_frame()
    _sf_with_outcomes_pre_adjust: SampleFrame
    # pyre-fixme[13]: Attributes are initialized in _create() / from_frame()
    _sf_with_outcomes: SampleFrame
    # pyre-fixme[13]: Attributes are initialized in _create() / from_frame()
    _sf_target: SampleFrame | None
    # pyre-fixme[13]: Attributes are initialized in _create() / from_frame()
    _adjustment_model: dict[str, Any] | None
    # pyre-fixme[4]: Attributes are initialized in from_frame() / _create()
    _links = None
    _id_column: pd.Series | None = None
    _weight_column: pd.Series | None = None

    @property
    def _df_dtypes(self) -> pd.Series | None:
        """Original dtypes, delegated to ``_sf_with_outcomes._df_dtypes``."""
        return self._sf_with_outcomes._df_dtypes

    @_df_dtypes.setter
    def _df_dtypes(self, value: pd.Series | None) -> None:
        self._sf_with_outcomes._df_dtypes = value

    @property
    def id_column(self) -> pd.Series | None:  # pyre-ignore[3]
        """The id column Series."""
        return self._id_column

    @id_column.setter
    def id_column(self, value: pd.Series | None) -> None:  # pyre-ignore[2,3]
        self._id_column = value

    @property
    def weight_column(self) -> pd.Series | None:  # pyre-ignore[3]
        """The weight column Series."""
        return self._weight_column

    @weight_column.setter
    def weight_column(self, value: pd.Series | None) -> None:  # pyre-ignore[2,3]
        self._weight_column = value

    # --- Property descriptors backed by _sf_with_outcomes ---

    @property
    def _df(self) -> pd.DataFrame:  # pyre-ignore[3]
        """The internal DataFrame, delegated to ``_sf_with_outcomes._df``."""
        return self._sf_with_outcomes._df

    @_df.setter
    def _df(self, value: pd.DataFrame | None) -> None:  # pyre-ignore[2,3]
        if value is not None:
            self._sf_with_outcomes._df = value

    @property
    def _outcome_columns(self) -> pd.DataFrame | None:
        """Outcome columns as a DataFrame, delegated to ``_sf_with_outcomes``."""
        outcome_cols = self._sf_with_outcomes._column_roles.get("outcomes", [])
        if not outcome_cols:
            return None
        return self._sf_with_outcomes._df[outcome_cols]

    @_outcome_columns.setter
    def _outcome_columns(self, value: pd.DataFrame | None) -> None:
        if value is None:
            self._sf_with_outcomes._column_roles["outcomes"] = []
        else:
            self._sf_with_outcomes._column_roles["outcomes"] = value.columns.tolist()

    @property
    def _ignored_column_names(self) -> list[str]:  # pyre-ignore[3]
        """Ignored column names, delegated to ``_sf_with_outcomes.ignored_columns``."""
        return self._sf_with_outcomes._column_roles.get("ignored", [])

    @_ignored_column_names.setter
    def _ignored_column_names(
        self, value: list[str] | None
    ) -> None:  # pyre-ignore[2,3]
        self._sf_with_outcomes._column_roles["ignored"] = list(value) if value else []

    @property
    def df_ignored(self) -> pd.DataFrame | None:
        """Ignored columns from the responder SampleFrame, or None."""
        return self._sf_with_outcomes.df_ignored

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
    #       1. Public constructor: BalanceFrame(sf_with_outcomes=sf1, sf_target=sf2)
    #          → validates args and delegates to _create().
    #       2. deepcopy path: BalanceFrame() (no args)
    #          → returns a bare object via object.__new__(cls); deepcopy
    #          then copies attributes onto it directly.
    #   - __init__ is intentionally a no-op (all state is set by _create()).
    # -----------------------------------------------------------------------
    def __new__(
        cls,
        sf_with_outcomes: SampleFrame | None = None,
        sf_target: SampleFrame | None = None,
    ) -> BalanceFrame:
        """Create a BalanceFrame from responder and target SampleFrames.

        This uses ``__new__`` so that the natural constructor syntax
        ``BalanceFrame(sf_with_outcomes=..., sf_target=...)`` works while still
        routing through the validated :meth:`_create` factory.

        Args:
            sf_with_outcomes: The responder / sample data.
            sf_target: The target / population data.

        Returns:
            A new BalanceFrame pairing the two samples.

        Raises:
            TypeError: If *sf_with_outcomes* or *sf_target* is not a SampleFrame.
            ValueError: If *sf_with_outcomes* and *sf_target* share no covariate
                columns.
        """
        if sf_with_outcomes is None:
            # Allow object.__new__(cls) for copy.deepcopy() support.
            if sf_target is None:
                return object.__new__(cls)
            raise TypeError(
                "BalanceFrame requires at least a 'sf_with_outcomes' argument. "
                "Usage: BalanceFrame(sf_with_outcomes=sf1) or "
                "BalanceFrame(sf_with_outcomes=sf1, sf_target=sf2)"
            )
        return cls._create(sf_with_outcomes=sf_with_outcomes, sf_target=sf_target)

    def __init__(
        self,
        sf_with_outcomes: SampleFrame | None = None,
        sf_target: SampleFrame | None = None,
    ) -> None:
        # All initialisation happens in _create(); __init__ is intentionally
        # empty so that __new__ + _create() handles everything.
        pass

    @classmethod
    def _create(
        cls,
        sf_with_outcomes: SampleFrame,
        sf_target: SampleFrame | None = None,
    ) -> Self:
        """Internal factory method.

        Validates covariate overlap and builds the BalanceFrame instance.
        Prefer the public constructor ``BalanceFrame(sf_with_outcomes=..., sf_target=...)``.

        Args:
            sf_with_outcomes: The responder sample.
            sf_target: The target population. If None, creates a target-less
                BalanceFrame that can be completed later via :meth:`set_target`.

        Returns:
            A validated BalanceFrame.

        Raises:
            TypeError: If *sf_with_outcomes* or *sf_target* is not a SampleFrame.
            ValueError: If they share no covariate columns.
        """
        if not isinstance(sf_with_outcomes, SampleFrame):
            raise TypeError(
                f"'sf_with_outcomes' must be a SampleFrame, got {type(sf_with_outcomes).__name__}"
            )
        if sf_target is not None and not isinstance(sf_target, SampleFrame):
            raise TypeError(
                f"'sf_target' must be a SampleFrame, got {type(sf_target).__name__}"
            )

        instance = object.__new__(cls)
        instance._sf_with_outcomes_pre_adjust = sf_with_outcomes
        instance._sf_with_outcomes = sf_with_outcomes  # same object initially
        instance._sf_target = sf_target
        instance._adjustment_model = None
        instance.id_column = sf_with_outcomes.id_column
        try:
            instance.weight_column = sf_with_outcomes.weight_column
        except ValueError:
            instance.weight_column = None
        instance._links = collections.defaultdict(list)
        if sf_target is not None:
            instance._links["target"] = sf_target

        # Validate covariate overlap using public properties
        if sf_target is not None:
            cls._validate_covariate_overlap(sf_with_outcomes, sf_target)

        return instance

    @staticmethod
    def _validate_covariate_overlap(
        responders: SampleFrame, target: SampleFrame
    ) -> None:
        """Check that responders and target share at least one covariate.

        Raises:
            ValueError: If they share no covariate columns.
        """
        resp_covars = set(responders.covar_columns)
        target_covars = set(target.covar_columns)
        overlap = resp_covars & target_covars
        if len(overlap) == 0:
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
        return self._sf_with_outcomes.df

    @property
    def df_target(self) -> pd.DataFrame | None:
        """The target data as a DataFrame, or None if not yet set."""
        if self._sf_target is None:
            return None
        return self._sf_target.df

    @property
    def df_responders_unadjusted(self) -> pd.DataFrame:
        """The original (pre-adjustment) responder data as a DataFrame."""
        return self._sf_with_outcomes_pre_adjust.df

    # --- Backward-compat aliases (to be removed in a future diff) ---

    @property
    def responders(self) -> SampleFrame:
        """Alias for ``_sf_with_outcomes`` (backward compat, will be removed)."""
        return self._sf_with_outcomes

    @property
    def target(self) -> SampleFrame | None:
        """Alias for ``_sf_target`` (backward compat, will be removed)."""
        return self._sf_target

    @property
    def unadjusted(self) -> SampleFrame | None:
        """Alias for ``_sf_with_outcomes_pre_adjust`` if adjusted, else None (backward compat)."""
        if self.is_adjusted:
            return self._sf_with_outcomes_pre_adjust
        return None

    def has_target(self) -> bool:
        """Check if this BalanceFrame has a target population set.

        Returns:
            bool: True if a target population has been set, False otherwise.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 1.0]}))
            >>> bf = BalanceFrame(sf_with_outcomes=resp)
            >>> bf.has_target()
            False
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
            >>> bf.set_target(tgt)
            >>> bf.has_target()
            True
        """
        return self._sf_target is not None or (
            self._links is not None and "target" in self._links
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
            >>> bf = BalanceFrame(sf_with_outcomes=resp)
            >>> bf.set_target(tgt)
            >>> bf.has_target()
            True
        """
        if isinstance(target, BalanceFrame):
            # BalanceFrame / Sample path: return a deep copy (immutable)
            new_copy = deepcopy(self)
            new_copy._links["target"] = target
            # Validation may fail if sample and target share no covariates
            # (e.g., outcome-only targets). In that case, skip validation —
            # adjust() will report the error when called.
            try:
                BalanceFrame._validate_covariate_overlap(
                    new_copy._sf_with_outcomes, target._sf_with_outcomes
                )
            except (ValueError, TypeError):
                pass
            new_copy._sf_target = target._sf_with_outcomes
            return new_copy

        if isinstance(target, SampleFrame):
            # SampleFrame path: default in_place=True for backward compat
            if in_place is None:
                in_place = True
            BalanceFrame._validate_covariate_overlap(self._sf_with_outcomes, target)

            if in_place:
                self._sf_target = target
                self._links["target"] = target
                # Reset adjustment state — old adjustment is no longer valid.
                self._sf_with_outcomes = self._sf_with_outcomes_pre_adjust
                self._adjustment_model = None
                return self
            else:
                return type(self)._create(
                    sf_with_outcomes=copy.deepcopy(self._sf_with_outcomes_pre_adjust),
                    sf_target=target,
                )

        raise TypeError("A target, a Sample object, must be specified")

    @property
    def is_adjusted(self) -> _CallableBool:
        """Whether this BalanceFrame has been adjusted.

        Returns a ``_CallableBool`` so both ``bf.is_adjusted`` (property)
        and ``bf.is_adjusted()`` (legacy call) work.
        """
        return _CallableBool(
            self._sf_with_outcomes is not self._sf_with_outcomes_pre_adjust
        )

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
        return self._sf_with_outcomes.df_covars, self._sf_target.df_covars

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
        new_responders = copy.deepcopy(self._sf_with_outcomes)
        method_name = (
            method
            if isinstance(method, str)
            else getattr(method, "__name__", str(method))
        )
        new_responders.add_weight_column(
            "weight_adjusted",
            result["weight"],
            metadata={
                "method": method_name,
                "adjusted": True,
                "model": result.get("model", {}),
            },
        )
        new_responders.set_active_weight("weight_adjusted")

        # For Sample subclasses: rename "weight_adjusted" → original weight
        # column name so the public API always sees the original name.
        # For direct BalanceFrame: keep both columns (weight + weight_adjusted).
        original_weight_name = _assert_type(self.weight_column).name
        if type(self) is not BalanceFrame:
            # Sample (or other subclass) path: rename weight_adjusted → original
            if (
                original_weight_name in new_responders._df.columns
                and original_weight_name != "weight_adjusted"
            ):
                new_responders._df = new_responders._df.drop(
                    columns=[original_weight_name]
                )
                if original_weight_name in new_responders._column_roles["weights"]:
                    new_responders._column_roles["weights"].remove(original_weight_name)
            new_responders.rename_weight_column("weight_adjusted", original_weight_name)

        # Use type(self) so subclasses (e.g. Sample) get their own type back.
        new_bf = type(self)._create(
            sf_with_outcomes=new_responders,
            sf_target=self._sf_target,
        )
        # Point _sf_with_outcomes_pre_adjust to the original (pre-adjustment) data
        new_bf._sf_with_outcomes_pre_adjust = self._sf_with_outcomes_pre_adjust
        new_bf.id_column = new_responders.id_column
        new_bf.weight_column = new_responders.weight_column
        # Set _links for __str__() and BalanceDF integration
        new_bf._links["unadjusted"] = self
        if "target" in self._links:
            new_bf._links["target"] = self._links["target"]

        raw_model = result.get("model")
        # Defensive copy: the weighting function may retain a reference to the
        # dict it returned, so mutating it here could cause surprising side effects.
        new_bf._adjustment_model = (
            dict(raw_model) if isinstance(raw_model, dict) else raw_model
        )
        # Preserve the raw model's method name (e.g. "null_adjustment") when
        # present; only set a fallback when the model doesn't include one.
        if isinstance(new_bf._adjustment_model, dict):
            new_bf._adjustment_model.setdefault("method", method_name)
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
        returned BalanceFrame has ``is_adjusted == True``, a new weight column
        (``"weight_adjusted"``), and the pre-adjustment responders stored in
        :attr:`unadjusted`.

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
                registered adjustment method, or if the BalanceFrame has
                already been adjusted.

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
            >>> bf = BalanceFrame(sf_with_outcomes=resp, sf_target=tgt)
            >>> adjusted = bf.adjust(method="ipw")
            >>> adjusted.is_adjusted
            True
        """
        if target is not None:
            # Inline target: set it first, then recurse
            self_with_target = self.set_target(target)
            return self_with_target.adjust(*args, method=method, **kwargs)

        self._no_target_error()

        if self.is_adjusted:
            raise ValueError(
                "Cannot adjust an already-adjusted BalanceFrame. "
                "Use the original (unadjusted) BalanceFrame instead."
            )

        sf_target = self._sf_target
        assert sf_target is not None  # guaranteed by _no_target_error() above

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
            sample_weights=self._sf_with_outcomes.df_weights.iloc[:, 0],
            target_df=target_covars,
            target_weights=sf_target.df_weights.iloc[:, 0],
            **kwargs,
        )

        return self._build_adjusted_frame(result, method)

    def model(self) -> dict[str, Any] | None:
        """Return the adjustment model dictionary, or None if not adjusted.

        Returns:
            The model dict from the weighting method, or None.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
            >>> bf = BalanceFrame(sf_with_outcomes=resp, sf_target=tgt)
            >>> bf.model() is None
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

        bf = cls._create(sf_with_outcomes=responders_sf, sf_target=target_sf)

        if sample.is_adjusted():
            # Set unadjusted to a DIFFERENT SampleFrame so is_adjusted returns True
            bf._sf_with_outcomes_pre_adjust = SampleFrame.from_sample(
                sample._links["unadjusted"]
            )
            bf._adjustment_model = sample.model()

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
            >>> bf = BalanceFrame(sf_with_outcomes=resp, sf_target=tgt)
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
            self._sf_with_outcomes._df,
            id_column=self._sf_with_outcomes.id_column_name,
            weight_column=self._sf_with_outcomes.active_weight_column,
            outcome_columns=self._sf_with_outcomes.outcome_columns or None,
            ignored_columns=self._sf_with_outcomes.ignored_columns or None,
            standardize_types=False,
        )
        target_sample = Sample.from_frame(
            target._df,
            id_column=target.id_column_name,
            weight_column=target.active_weight_column,
            outcome_columns=target.outcome_columns or None,
            ignored_columns=target.ignored_columns or None,
            standardize_types=False,
        )
        result = resp_sample.set_target(target_sample)

        if self.is_adjusted and self._sf_with_outcomes_pre_adjust is not None:
            unadj_sf = SampleFrame.from_frame(
                self._sf_with_outcomes_pre_adjust._df,
                id_column=self._sf_with_outcomes_pre_adjust.id_column_name,
                weight_column=self._sf_with_outcomes_pre_adjust.active_weight_column,
                outcome_columns=self._sf_with_outcomes_pre_adjust.outcome_columns
                or None,
                ignored_columns=self._sf_with_outcomes_pre_adjust.ignored_columns
                or None,
                standardize_types=False,
            )
            # pyre-ignore[16]: Sample gains this attr via BalanceFrame inheritance (diff 14.3)
            result._sf_with_outcomes_pre_adjust = unadj_sf
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
        ``BalanceDF._BalanceDF_child_from_linked_samples`` can walk the
        links just as it does for the old ``Sample`` class.

        Returns:
            dict: Mapping of link names to BalanceDFSource instances.
        """
        links: dict[str, BalanceDFSource] = {}
        if self._sf_target is not None:
            links["target"] = self._sf_target
        if self.is_adjusted:
            links["unadjusted"] = self._sf_with_outcomes_pre_adjust
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
            >>> bf = BalanceFrame(sf_with_outcomes=resp, sf_target=tgt)
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
            >>> bf = BalanceFrame(sf_with_outcomes=resp, sf_target=tgt)
            >>> bf.weights().df.columns.tolist()
            ['weight']
        """
        from balance.balancedf_class import BalanceDFSource, BalanceDFWeights

        # Pass self (not _sf_with_outcomes) so that r_indicator and other methods
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
            >>> bf = BalanceFrame(sf_with_outcomes=resp, sf_target=tgt)
            >>> bf.outcomes().df.columns.tolist()
            ['y']
        """
        if not self._sf_with_outcomes.outcome_columns:
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
            >>> bf = BalanceFrame(sf_with_outcomes=resp, sf_target=tgt)
            >>> bf._design_effect_diagnostics()
            (1.0, 2.0, 1.0)
        """
        if n_rows is None:
            n_rows = len(self._sf_with_outcomes)
        try:
            de = weights_stats.design_effect(
                self._sf_with_outcomes.df_weights.iloc[:, 0]
            )
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
            >>> bf = BalanceFrame(sf_with_outcomes=resp, sf_target=tgt)
            >>> adjusted = bf.adjust(method="null")
            >>> "method: null" in adjusted._quick_adjustment_details()
            True
        """
        details: list[str] = []
        model = self.model()
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
            >>> bf = BalanceFrame(sf_with_outcomes=resp, sf_target=tgt)
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
                model_dict=self.model(),
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
            model_dict=self.model(),
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
            >>> bf = BalanceFrame(sf_with_outcomes=resp, sf_target=tgt)
            >>> adjusted = bf.adjust(method="null")
            >>> adjusted.diagnostics().columns.tolist()
            ['metric', 'val', 'var']
        """
        logger.info("Starting computation of diagnostics of the fitting")
        self._check_if_adjusted()

        outcome_columns = self._sf_with_outcomes.df_outcomes
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
            model_dict=self.model(),
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
            >>> bf = BalanceFrame(sf_with_outcomes=resp, sf_target=tgt)
            >>> bf.df_all["source"].unique().tolist()
            ['self', 'target']
        """
        parts: list[pd.DataFrame] = []

        resp_df = self._sf_with_outcomes._df.copy()
        resp_df["source"] = "self"
        parts.append(resp_df)

        if self._sf_target is not None:
            tgt_df = self._sf_target._df.copy()
            tgt_df["source"] = "target"
            parts.append(tgt_df)

        if self.is_adjusted:
            unadj_df = self._sf_with_outcomes_pre_adjust._df.copy()
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
        ignored = self._sf_with_outcomes.df_ignored
        return pd.concat(
            (
                self.id_column,
                covars.df if covars is not None else None,
                outcomes.df if outcomes is not None else None,
                (
                    pd.DataFrame(self.weight_column)
                    if self.weight_column is not None
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
            >>> bf = BalanceFrame(sf_with_outcomes=resp, sf_target=tgt)
            >>> filtered = bf.keep_only_some_rows_columns(rows_to_keep="x > 15")
            >>> len(filtered._sf_with_outcomes._df)
            2
        """
        if rows_to_keep is None and columns_to_keep is None:
            return self

        new_bf = copy.deepcopy(self)

        if new_bf.has_target():
            # With target: filter all SampleFrames
            new_bf._sf_with_outcomes = BalanceFrame._filter_sf(
                new_bf._sf_with_outcomes, rows_to_keep, columns_to_keep
            )
            if new_bf._sf_target is not None:
                new_bf._sf_target = BalanceFrame._filter_sf(
                    new_bf._sf_target, rows_to_keep, columns_to_keep
                )
            if new_bf._sf_with_outcomes_pre_adjust is not None:
                new_bf._sf_with_outcomes_pre_adjust = BalanceFrame._filter_sf(
                    new_bf._sf_with_outcomes_pre_adjust, rows_to_keep, columns_to_keep
                )
        else:
            # No target: filter the responder SampleFrame
            if columns_to_keep is not None:
                if not (set(columns_to_keep) <= set(new_bf.df.columns)):
                    logger.warning(
                        "Note that not all columns_to_keep are in Sample. "
                        "Only those that exist are removed"
                    )
            new_bf._sf_with_outcomes = BalanceFrame._filter_sf(
                new_bf._sf_with_outcomes, rows_to_keep, columns_to_keep
            )
            if (
                new_bf._sf_with_outcomes_pre_adjust is not None
                and new_bf._sf_with_outcomes_pre_adjust is not new_bf._sf_with_outcomes
            ):
                new_bf._sf_with_outcomes_pre_adjust = BalanceFrame._filter_sf(
                    new_bf._sf_with_outcomes_pre_adjust, rows_to_keep, columns_to_keep
                )

        # Sync id_column / weight_column from filtered responders
        new_bf.id_column = new_bf._sf_with_outcomes.id_column
        try:
            new_bf.weight_column = new_bf._sf_with_outcomes.weight_column
        except ValueError:
            # Weight column was filtered out — leave it unchanged
            pass

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
            if sf._active_weight_column is not None:
                keep_set.add(sf._active_weight_column)
            for wc in sf.weight_columns:
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
            >>> bf = BalanceFrame(sf_with_outcomes=resp, sf_target=tgt)
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
            >>> bf = BalanceFrame(sf_with_outcomes=resp, sf_target=tgt)
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

    def set_weights(self, weights: pd.Series | float | None) -> None:
        """Set or replace the responder weights.

        Overwrites the weight column on the underlying DataFrame.
        When *weights* is a ``pd.Series``, values are aligned by index.

        Args:
            weights: New weights. A Series (aligned by index), a scalar
                (broadcast to all rows), or ``None`` (sets all to NaN).
        """
        # Break the shared reference so we don't corrupt the pre-adjustment
        # baseline (both point to the same SampleFrame initially).
        if self._sf_with_outcomes is self._sf_with_outcomes_pre_adjust:
            self._sf_with_outcomes_pre_adjust = copy.deepcopy(self._sf_with_outcomes)

        if isinstance(weights, pd.Series):
            if not all(idx in weights.index for idx in self.df.index):
                logger.warning(
                    "Note that not all Sample units will be assigned weights, "
                    "since weights are missing some of the indices in Sample.df"
                )

        wc_name = _assert_type(self.weight_column).name
        if isinstance(weights, pd.Series):
            if not pd.api.types.is_float_dtype(self._df[wc_name]):
                self._df[wc_name] = self._df[wc_name].astype("float64")
            if not pd.api.types.is_float_dtype(weights):
                weights = weights.astype("float64")
            self._df.loc[:, wc_name] = weights
        else:
            if not pd.api.types.is_float_dtype(self._df[wc_name]):
                self._df[wc_name] = self._df[wc_name].astype("float64")
            weights_value = np.nan if weights is None else weights
            self._df.loc[:, wc_name] = weights_value

        self.weight_column = self._df[wc_name]

    def set_unadjusted(self, second: BalanceFrame) -> Self:
        """Set the unadjusted link for comparative analysis.

        Returns a deep copy with ``_sf_with_outcomes_pre_adjust`` pointing at
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
        new_bf._sf_with_outcomes_pre_adjust = second._sf_with_outcomes
        return new_bf

    # --- Column accessors (moved from Sample) ---

    def _special_columns_names(self) -> list[str]:
        """Return names of all special columns (id, weight, outcome, ignored)."""
        return (
            [str(i.name) for i in [self.id_column, self.weight_column] if i is not None]
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
        return self._sf_with_outcomes._covar_columns()

    # --- Error checks (moved from Sample) ---

    def _check_if_adjusted(self) -> None:
        """Raise ValueError if not adjusted."""
        if not self.is_adjusted:
            raise ValueError(
                "This is not an adjusted Sample. Use sample.adjust to adjust the sample to target"
            )

    def _no_target_error(self) -> None:
        """Raise ValueError if no target is set."""
        if not self.has_target():
            raise ValueError(
                "This Sample does not have a target set. Use sample.set_target to add target"
            )

    def _check_outcomes_exists(self) -> None:
        """Raise ValueError if no outcome columns are specified."""
        if self.outcomes() is None:
            raise ValueError("This Sample does not have outcome columns specified")

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
            " using " + _assert_type(self.model())["method"]
            if self.model() is not None
            else ""
        )
        variables = ",".join(self._covar_columns_names())
        id_column_name = self.id_column.name if self.id_column is not None else "None"
        weight_column_name = (
            self.weight_column.name if self.weight_column is not None else "None"
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
