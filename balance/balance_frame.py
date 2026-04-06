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
from typing import Any, Callable, cast, Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
from balance.adjustment import _find_adjustment_method
from balance.csv_utils import to_csv_with_defaults
from balance.sample_frame import SampleFrame
from balance.stats_and_plots import weights_stats
from balance.summary_utils import _build_diagnostics, _build_summary
from balance.typing import FilePathOrBuffer
from balance.utils.file_utils import _to_download

if TYPE_CHECKING:
    from balance.balancedf_class import BalanceDFSource

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

        bf = cls._create(responders=responders_sf, target=target_sf)

        if sample.is_adjusted():
            bf._unadjusted = SampleFrame.from_sample(sample._links["unadjusted"])
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
            >>> bf = BalanceFrame(responders=resp, target=tgt)
            >>> s = bf.to_sample()
            >>> s.has_target()
            True
        """
        # Lazy import: sample_class ↔ balance_frame have a circular dependency.
        from balance.sample_class import Sample

        target = self._target
        if target is None:
            raise ValueError(
                "Cannot convert to Sample: BalanceFrame has no target set."
            )

        resp_sample = Sample.from_frame(
            self._responders._df,
            id_column=self._responders.id_column_name,
            weight_column=self._responders.active_weight_column,
            outcome_columns=self._responders.outcome_columns or None,
            ignore_columns=self._responders.ignore_columns or None,
            standardize_types=False,
        )
        target_sample = Sample.from_frame(
            target._df,
            id_column=target.id_column_name,
            weight_column=target.active_weight_column,
            outcome_columns=target.outcome_columns or None,
            ignore_columns=target.ignore_columns or None,
            standardize_types=False,
        )
        result = resp_sample.set_target(target_sample)

        if self.is_adjusted and self._unadjusted is not None:
            unadj_sample = Sample.from_frame(
                self._unadjusted._df,
                id_column=self._unadjusted.id_column_name,
                weight_column=self._unadjusted.active_weight_column,
                outcome_columns=self._unadjusted.outcome_columns or None,
                ignore_columns=self._unadjusted.ignore_columns or None,
                standardize_types=False,
            )
            result._links["unadjusted"] = unadj_sample
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
        if self._target is not None:
            links["target"] = self._target
        if self._unadjusted is not None:
            links["unadjusted"] = self._unadjusted
        return links

    def covars(self) -> Any:
        """Return a :class:`~balance.balancedf_class.BalanceDFCovars` for the responders.

        The returned object carries linked target (and unadjusted, if
        adjusted) views so that methods like ``.mean()`` and ``.asmd()``
        automatically include comparisons across sources.

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
            >>> bf = BalanceFrame(responders=resp, target=tgt)
            >>> bf.covars().df.columns.tolist()
            ['x']
        """
        from balance.balancedf_class import BalanceDFCovars

        return BalanceDFCovars(self._responders, links=self._build_links_dict())

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
            >>> bf = BalanceFrame(responders=resp, target=tgt)
            >>> bf.weights().df.columns.tolist()
            ['weight']
        """
        from balance.balancedf_class import BalanceDFWeights

        return BalanceDFWeights(self._responders, links=self._build_links_dict())

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
            >>> bf = BalanceFrame(responders=resp, target=tgt)
            >>> bf.outcomes().df.columns.tolist()
            ['y']
        """
        if not self._responders.outcome_columns:
            return None
        from balance.balancedf_class import BalanceDFOutcomes

        return BalanceDFOutcomes(self._responders, links=self._build_links_dict())

    # --- Summary & diagnostics ---

    def _design_effect_diagnostics(
        self,
    ) -> tuple[float | None, float | None, float | None]:
        """Compute design effect, ESS, and ESSP from the responder weights.

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
            >>> bf = BalanceFrame(responders=resp, target=tgt)
            >>> bf._design_effect_diagnostics()
            (1.0, 2.0, 1.0)
        """
        n_rows = len(self._responders)
        try:
            de = weights_stats.design_effect(self._responders.df_weights.iloc[:, 0])
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
            >>> bf = BalanceFrame(responders=resp, target=tgt)
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
            de, ess, essp = self._design_effect_diagnostics()
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
            >>> bf = BalanceFrame(responders=resp, target=tgt)
            >>> adjusted = bf.adjust(method="null")
            >>> "Covariate diagnostics:" in adjusted.summary()
            True
        """
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
            is_adjusted=self.is_adjusted,
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
            >>> bf = BalanceFrame(responders=resp, target=tgt)
            >>> adjusted = bf.adjust(method="null")
            >>> adjusted.diagnostics().columns.tolist()
            ['metric', 'val', 'var']
        """
        if not self.is_adjusted:
            raise ValueError(
                "diagnostics() requires an adjusted BalanceFrame. "
                "Call bf.adjust() first."
            )

        outcome_columns = self._responders.df_outcomes
        outcome_impact = None
        if weights_impact_on_outcome_method is not None and outcome_columns is not None:
            outcome_impact = self.outcomes().weights_impact_on_outcome_ss(
                method=weights_impact_on_outcome_method,
                conf_level=weights_impact_on_outcome_conf_level,
                round_ndigits=None,
            )

        target = self._target
        assert target is not None, "diagnostics() requires a target"
        return _build_diagnostics(
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

    # --- Parity helpers ---

    def ignored_columns(self) -> pd.DataFrame | None:
        """Return ignored (misc) columns from the responder SampleFrame, or None.

        Delegates to :meth:`SampleFrame.ignored_columns` on the responders.
        Provided for API parity with :class:`~balance.sample_class.Sample`.

        Returns:
            pd.DataFrame | None: A copy of the responder's miscellaneous
                columns, or None if no misc columns are registered.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0],
            ...                   "weight": [1.0, 1.0], "region": ["US", "UK"]}),
            ...     ignore_columns=["region"])
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
            >>> bf = BalanceFrame(responders=resp, target=tgt)
            >>> bf.ignored_columns()
               region
            0     US
            1     UK
        """
        return self._responders.ignored_columns()

    @property
    def id_column(self) -> pd.Series:
        """The ID column of the responder SampleFrame as a Series.

        Delegates to :attr:`SampleFrame.id_column` on the responders.
        Provided for API parity with :class:`~balance.sample_class.Sample`.

        Returns:
            pd.Series: A copy of the responder's ID column.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> from balance.balance_frame import BalanceFrame
            >>> resp = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 1.0]}))
            >>> tgt = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
            >>> bf = BalanceFrame(responders=resp, target=tgt)
            >>> bf.id_column.tolist()
            ['1', '2']
        """
        return self._responders.id_column

    # --- DataFrame / export ---

    @property
    def df(self) -> pd.DataFrame:
        """Combined DataFrame with all samples, distinguished by a ``"source"`` column.

        Concatenates the responder, target, and (if adjusted) unadjusted
        DataFrames vertically, adding a ``"source"`` column with values
        ``"self"``, ``"target"``, and ``"unadjusted"`` respectively.
        Mirrors the ``Sample.df`` property behaviour.

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
            >>> bf = BalanceFrame(responders=resp, target=tgt)
            >>> bf.df["source"].unique().tolist()
            ['self', 'target']
        """
        parts: list[pd.DataFrame] = []

        resp_df = self._responders._df.copy()
        resp_df["source"] = "self"
        parts.append(resp_df)

        if self._target is not None:
            tgt_df = self._target._df.copy()
            tgt_df["source"] = "target"
            parts.append(tgt_df)

        if self._unadjusted is not None:
            unadj_df = self._unadjusted._df.copy()
            unadj_df["source"] = "unadjusted"
            parts.append(unadj_df)

        return pd.concat(parts, ignore_index=True)

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
            >>> bf = BalanceFrame(responders=resp, target=tgt)
            >>> filtered = bf.keep_only_some_rows_columns(rows_to_keep="x > 15")
            >>> len(filtered.responders._df)
            2
        """
        if rows_to_keep is None and columns_to_keep is None:
            return self

        new_bf = copy.deepcopy(self)

        new_bf._responders = BalanceFrame._filter_sf(
            new_bf._responders, rows_to_keep, columns_to_keep
        )
        if new_bf._target is not None:
            new_bf._target = BalanceFrame._filter_sf(
                new_bf._target, rows_to_keep, columns_to_keep
            )
        if new_bf._unadjusted is not None:
            new_bf._unadjusted = BalanceFrame._filter_sf(
                new_bf._unadjusted, rows_to_keep, columns_to_keep
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
            >>> bf = BalanceFrame(responders=resp, target=tgt)
            >>> "source" in bf.to_csv()
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
            >>> bf = BalanceFrame(responders=resp, target=tgt)
            >>> link = bf.to_download(tempdir=tempfile.gettempdir())
        """
        return _to_download(self.df, tempdir)

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
