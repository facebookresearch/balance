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

import copy
import logging
from typing import Any, Callable, cast, Literal

import pandas as pd
from balance.adjustment import _find_adjustment_method
from balance.sample_frame import SampleFrame

# The set of string method names accepted by _find_adjustment_method.
_AdjustmentMethodStr = Literal["cbps", "ipw", "null", "poststratify", "rake"]

logger: logging.Logger = logging.getLogger(__package__)


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
    ``BalanceFrame(responders=..., target=...)`` which delegates to the
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
        >>> bf = BalanceFrame(responders=resp, target=tgt)
        >>> bf.is_adjusted
        False
        >>> adjusted = bf.adjust(method="ipw")
        >>> adjusted.is_adjusted
        True
        >>> bf.is_adjusted  # original unchanged
        False
    """

    # pyre-fixme[13]: Attributes are initialized in _create()
    _responders: SampleFrame
    # pyre-fixme[13]: Attributes are initialized in _create()
    _target: SampleFrame | None
    # pyre-fixme[13]: Attributes are initialized in _create()
    _unadjusted: SampleFrame | None
    # pyre-fixme[13]: Attributes are initialized in _create()
    _adjustment_model: dict[str, Any] | None

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
    #       1. Public constructor: BalanceFrame(responders=sf1, target=sf2)
    #          → validates args and delegates to _create().
    #       2. deepcopy path: BalanceFrame() (no args)
    #          → returns a bare object via object.__new__(cls); deepcopy
    #          then copies attributes onto it directly.
    #   - __init__ is intentionally a no-op (all state is set by _create()).
    # -----------------------------------------------------------------------
    def __new__(
        cls,
        responders: SampleFrame | None = None,
        target: SampleFrame | None = None,
    ) -> BalanceFrame:
        """Create a BalanceFrame from responder and target SampleFrames.

        This uses ``__new__`` so that the natural constructor syntax
        ``BalanceFrame(responders=..., target=...)`` works while still
        routing through the validated :meth:`_create` factory.

        Args:
            responders: The responder / sample data.
            target: The target / population data.

        Returns:
            A new BalanceFrame pairing the two samples.

        Raises:
            TypeError: If *responders* or *target* is not a SampleFrame.
            ValueError: If *responders* and *target* share no covariate
                columns.
        """
        if responders is None:
            # Allow object.__new__(cls) for copy.deepcopy() support.
            if target is None:
                return object.__new__(cls)
            raise TypeError(
                "BalanceFrame requires at least a 'responders' argument. "
                "Usage: BalanceFrame(responders=sf1) or "
                "BalanceFrame(responders=sf1, target=sf2)"
            )
        return cls._create(responders=responders, target=target)

    def __init__(
        self,
        responders: SampleFrame | None = None,
        target: SampleFrame | None = None,
    ) -> None:
        # All initialisation happens in _create(); __init__ is intentionally
        # empty so that __new__ + _create() handles everything.
        pass

    @classmethod
    def _create(
        cls,
        responders: SampleFrame,
        target: SampleFrame | None = None,
    ) -> BalanceFrame:
        """Internal factory method.

        Validates covariate overlap and builds the BalanceFrame instance.
        Prefer the public constructor ``BalanceFrame(responders=..., target=...)``.

        Args:
            responders: The responder sample.
            target: The target population. If None, creates a target-less
                BalanceFrame that can be completed later via :meth:`set_target`.

        Returns:
            A validated BalanceFrame.

        Raises:
            TypeError: If *responders* or *target* is not a SampleFrame.
            ValueError: If they share no covariate columns.
        """
        if not isinstance(responders, SampleFrame):
            raise TypeError(
                f"'responders' must be a SampleFrame, got {type(responders).__name__}"
            )
        if target is not None and not isinstance(target, SampleFrame):
            raise TypeError(
                f"'target' must be a SampleFrame, got {type(target).__name__}"
            )

        instance = object.__new__(cls)
        instance._responders = responders
        instance._target = target
        instance._unadjusted = None
        instance._adjustment_model = None

        # Validate covariate overlap using public properties
        if target is not None:
            cls._validate_covariate_overlap(responders, target)

        return instance

    @staticmethod
    def _validate_covariate_overlap(
        responders: SampleFrame, target: SampleFrame
    ) -> None:
        """Check that responders and target share at least one covariate.

        Raises:
            ValueError: If they share no covariate columns.
        """
        resp_covars = set(responders.covars_columns)
        target_covars = set(target.covars_columns)
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
    def responders(self) -> SampleFrame:
        """The responder SampleFrame."""
        return self._responders

    @property
    def target(self) -> SampleFrame | None:
        """The target SampleFrame, or None if not yet set."""
        return self._target

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
            >>> bf = BalanceFrame(responders=resp)
            >>> bf.has_target()
            False
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
            >>> bf.set_target(tgt)
            >>> bf.has_target()
            True
        """
        return self._target is not None

    def set_target(self, target: SampleFrame, in_place: bool = True) -> BalanceFrame:
        """Set or replace the target population.

        Args:
            target: The new target SampleFrame.
            in_place: If True (default), modifies this BalanceFrame in place and
                returns self. If False, returns a new BalanceFrame with the new
                target.

        Returns:
            BalanceFrame with the new target set.

        Raises:
            TypeError: If *target* is not a SampleFrame.
            ValueError: If responders and target share no covariate columns.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
            >>> bf = BalanceFrame(responders=resp)
            >>> bf.set_target(tgt)
            >>> bf.has_target()
            True
        """
        if not isinstance(target, SampleFrame):
            raise TypeError(
                f"'target' must be a SampleFrame, got {type(target).__name__}"
            )
        BalanceFrame._validate_covariate_overlap(self._responders, target)

        if in_place:
            self._target = target
            # Reset adjustment state — old adjustment is no longer valid
            self._unadjusted = None
            self._adjustment_model = None
            return self
        else:
            return BalanceFrame._create(
                responders=copy.deepcopy(self._responders), target=target
            )

    @property
    def unadjusted(self) -> SampleFrame | None:
        """The pre-adjustment responder SampleFrame, or None if not adjusted."""
        return self._unadjusted

    @property
    def is_adjusted(self) -> bool:
        """Whether this BalanceFrame has been adjusted."""
        return self._unadjusted is not None

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
        if self._target is None:
            raise ValueError("Cannot get covars without a target population.")
        return self._responders.df_covars, self._target.df_covars

    def _build_adjusted_frame(
        self,
        result: dict[str, Any],
        method: str | Callable[..., Any],
    ) -> BalanceFrame:
        """Construct a new BalanceFrame with adjusted weights.

        Args:
            result: The dict returned by the weighting function, containing
                at least ``"weight"`` and optionally ``"model"``.
            method: The original method argument (string or callable).

        Returns:
            A new, adjusted BalanceFrame.
        """
        new_responders = copy.deepcopy(self._responders)
        method_name = method if isinstance(method, str) else "custom"
        new_responders.add_weight_column(
            "weight_adjusted",
            result["weight"],
            metadata={
                "method": method_name,
                "model": result.get("model", {}),
            },
        )
        new_responders.set_active_weight("weight_adjusted")

        new_bf = BalanceFrame._create(
            responders=new_responders,
            target=self._target,
        )
        new_bf._unadjusted = copy.deepcopy(self._responders)
        raw_model = result.get("model")
        # Defensive copy: the weighting function may retain a reference to the
        # dict it returned, so mutating it here could cause surprising side effects.
        new_bf._adjustment_model = (
            dict(raw_model) if isinstance(raw_model, dict) else raw_model
        )
        if isinstance(new_bf._adjustment_model, dict):
            if isinstance(method, str):
                new_bf._adjustment_model["method"] = method_name
            else:
                new_bf._adjustment_model.setdefault("method", method_name)
        return new_bf

    def adjust(
        self,
        method: str | Callable[..., Any] = "ipw",
        **kwargs: Any,
    ) -> BalanceFrame:
        """Adjust responder weights to match the target. Returns a NEW BalanceFrame.

        The original BalanceFrame is not modified (immutable pattern).  The
        returned BalanceFrame has ``is_adjusted == True``, a new weight column
        (``"weight_adjusted"``), and the pre-adjustment responders stored in
        :attr:`unadjusted`.

        Args:
            method: The weighting method to use.  Built-in options:
                ``"ipw"``, ``"cbps"``, ``"rake"``, ``"poststratify"``,
                ``"null"``.  A callable with the same signature as the
                built-in methods is also accepted.
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
            >>> bf = BalanceFrame(responders=resp, target=tgt)
            >>> adjusted = bf.adjust(method="ipw")
            >>> adjusted.is_adjusted
            True
        """
        target = self._target
        if target is None:
            raise ValueError(
                "Cannot adjust a BalanceFrame without a target population. "
                "Use set_target() to set one first."
            )
        if self.is_adjusted:
            raise ValueError(
                "Cannot adjust an already-adjusted BalanceFrame. "
                "Use the original (unadjusted) BalanceFrame instead."
            )

        adjustment_function = self._resolve_adjustment_function(method)
        resp_covars, target_covars = self._get_covars()

        result = adjustment_function(
            sample_df=resp_covars,
            sample_weights=self._responders.df_weights.iloc[:, 0],
            target_df=target_covars,
            target_weights=target.df_weights.iloc[:, 0],
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
            >>> bf = BalanceFrame(responders=resp, target=tgt)
            >>> bf.model() is None
            True
        """
        return self._adjustment_model

    def __repr__(self) -> str:
        status = "adjusted" if self.is_adjusted else "unadjusted"
        n_resp = len(self._responders)
        n_tgt = len(self._target) if self._target is not None else 0
        tgt_info = (
            f"{n_tgt} target observations" if self._target is not None else "no target"
        )
        resp_covars = self._responders.covars_columns
        return (
            f"BalanceFrame ({status}): "
            f"{n_resp} responders, {tgt_info}, "
            f"{len(resp_covars)} covariates: {','.join(resp_covars)}"
        )

    def __str__(self) -> str:
        return self.__repr__()
