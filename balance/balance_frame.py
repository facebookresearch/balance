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
from balance.sample_frame import SampleFrame
from balance.stats_and_plots import weights_stats
from balance.summary_utils import _build_diagnostics, _build_summary

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
