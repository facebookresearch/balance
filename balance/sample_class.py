# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import inspect
import logging
from copy import deepcopy
from typing import Any, List

from balance.balance_frame import _CallableBool, BalanceFrame  # noqa: F401
from balance.sample_frame import SampleFrame
from balance.summary_utils import _concat_metric_val_var

logger: logging.Logger = logging.getLogger(__package__)

# Re-export _concat_metric_val_var so existing imports from this module still work
__all__ = ["Sample", "_concat_metric_val_var", "_CallableBool"]


class Sample(BalanceFrame, SampleFrame):
    """
    A class used to represent a sample.

    Sample is the main object of balance. It contains a dataframe of unit's observations,
    associated with id and weight.

    Sample inherits from both :class:`~balance.balance_frame.BalanceFrame` and
    :class:`~balance.sample_frame.SampleFrame`.  Without a target it behaves
    like a SampleFrame; with a target (after ``set_target()``) it behaves
    like a BalanceFrame.

    MRO: Sample → BalanceFrame → SampleFrame → object

    Attributes
    ----------
    id_column : pd.Series
        a column representing the ids of the units in sample
    weight_column : pd.Series
        a column representing the weights of the units in sample
    """

    def __new__(
        cls,
        responders: SampleFrame | None = None,
        target: SampleFrame | None = None,
    ) -> "Sample":
        """Override __new__ to raise NotImplementedError for direct construction.

        Sample should be constructed via ``Sample.from_frame()``.  Direct
        ``Sample()`` calls raise ``NotImplementedError``.  Internal paths
        (``from_frame``, ``_create``, ``deepcopy``) are allowed through by
        checking the call stack.
        """
        if responders is None and target is None:
            # Check if called from an internal path (deepcopy, from_frame, _create)
            try:
                caller_func = inspect.stack()[1].function
            except Exception:
                raise NotImplementedError(
                    "Sample should not be constructed directly. "
                    "Use Sample.from_frame() instead."
                )
            if caller_func not in (
                "__deepcopy__",
                "__newobj__",
                "__newobj_ex__",
                "deepcopy",
                "_reconstruct",
                "from_frame",
                "_create",
            ):
                raise NotImplementedError(
                    "Sample should not be constructed directly. "
                    "Use Sample.from_frame() instead."
                )
        return object.__new__(cls)

    @classmethod
    def from_frame(
        cls,
        df: Any,
        id_column: str | None = None,
        covars_columns: list[str] | None = None,
        weight_column: str | None = None,
        outcome_columns: List[str] | tuple[str, ...] | str | None = None,
        predicted_outcome_columns: List[str] | tuple[str, ...] | str | None = None,
        ignore_columns: List[str] | tuple[str, ...] | str | None = None,
        check_id_uniqueness: bool = True,
        standardize_types: bool = True,
        use_deepcopy: bool = True,
        id_column_candidates: List[str] | tuple[str, ...] | str | None = None,
    ) -> Sample:
        """Create a Sample from a pandas DataFrame.

        Thin wrapper around :meth:`SampleFrame.from_frame` that builds a
        SampleFrame and then wraps it in a Sample via :meth:`_create`.

        Args:
            df: DataFrame containing the sample data.
            id_column: Column name for respondent ids (must be unique).
            covars_columns: Explicit covariate column names. If None,
                covariates are inferred by exclusion.
            weight_column: Column to treat as weight.
            outcome_columns: Columns to treat as outcomes.
            predicted_outcome_columns: Columns to treat as predicted outcomes.
            ignore_columns: Columns to ignore (excluded from covariates).
            check_id_uniqueness: Whether to verify id uniqueness.
            standardize_types: Whether to convert int types to float.
            use_deepcopy: Whether to deepcopy the input DataFrame.
            id_column_candidates: Candidate id column names when ``id_column``
                is not provided.

        Returns:
            A new Sample.
        """
        sf = SampleFrame.from_frame(
            df=df,
            id_column=id_column,
            covars_columns=covars_columns,
            weight_column=weight_column,
            outcome_columns=outcome_columns,
            predicted_outcome_columns=predicted_outcome_columns,
            ignore_columns=ignore_columns,
            check_id_uniqueness=check_id_uniqueness,
            standardize_types=standardize_types,
            use_deepcopy=use_deepcopy,
            id_column_candidates=id_column_candidates,
        )
        return cls._create(sf_with_outcomes=sf, sf_target=None)

    def __deepcopy__(self, memo: dict[int, Any]) -> Sample:
        """Return an independent deep copy of this Sample.

        Copies all instance attributes, preserving both BalanceFrame and
        SampleFrame state.
        """
        new = object.__new__(type(self))
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, deepcopy(v, memo))
        return new

    # _sample_frame is kept as an alias for _sf_with_outcomes for backward compat.
    # _balance_frame is no longer needed since Sample IS a BalanceFrame.

    @property
    def _sample_frame(self) -> SampleFrame:
        """Alias for ``_sf_with_outcomes`` (backward compat for internal code)."""
        return self._sf_with_outcomes

    @_sample_frame.setter
    def _sample_frame(self, value: SampleFrame | None) -> None:
        if value is not None:
            self._sf_with_outcomes = value
            self._sf_with_outcomes_pre_adjust = value

    @property
    def _balance_frame(self) -> BalanceFrame | None:
        """Returns self since Sample IS a BalanceFrame now."""
        return self if self.has_target() else None

    @_balance_frame.setter
    def _balance_frame(self, value: Any) -> None:
        pass  # no-op: Sample IS the BalanceFrame

    # All public API methods (df, covars, outcomes, weights,
    # adjust, set_target, has_target, set_unadjusted, is_adjusted, summary,
    # diagnostics, keep_only_some_rows_columns, to_csv, to_download,
    # model_matrix, set_weights, __str__, __repr__, ignored_columns, model)
    # are inherited from BalanceFrame.

    # --- Conversion to new API (kept temporarily) ---

    def to_sample_frame(self) -> Any:
        """Convert this Sample to a :class:`~balance.sample_frame.SampleFrame`.

        Preserves all data and column roles (id, weight, outcomes, ignored
        columns).  The returned SampleFrame is independent of the original
        Sample.

        Returns:
            SampleFrame: A new SampleFrame mirroring this Sample.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_class import Sample
            >>> s = Sample.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 2.0]}))
            >>> sf = s.to_sample_frame()
            >>> list(sf.df_covars.columns)
            ['x']
        """
        from balance.sample_frame import SampleFrame

        return SampleFrame.from_sample(self)

    def to_balance_frame(self) -> Any:
        """Convert this Sample (with target) to a :class:`~balance.balance_frame.BalanceFrame`.

        The Sample must have a target set.  If the Sample is adjusted, the
        adjustment state is preserved in the BalanceFrame.

        Returns:
            BalanceFrame: A new BalanceFrame mirroring this Sample's data,
                target, and adjustment state.

        Raises:
            ValueError: If this Sample does not have a target set.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_class import Sample
            >>> s = Sample.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 1.0]}))
            >>> t = Sample.from_frame(
            ...     pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]}))
            >>> bf = s.set_target(t).to_balance_frame()
            >>> bf.is_adjusted
            False
        """
        from balance.balance_frame import BalanceFrame

        return BalanceFrame.from_sample(self)
