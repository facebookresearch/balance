# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""SampleFrame: an explicit-role DataFrame container for the Balance library.

Stores covariates, weights, outcomes, predicted, and misc columns with
explicit role metadata, replacing the inference-by-exclusion pattern used
in the legacy Sample class.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, cast

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__package__)


class SampleFrame:
    """A DataFrame container with explicit column-role metadata.

    SampleFrame stores data as a single internal pd.DataFrame but with
    explicit metadata tracking which columns belong to which role:
    covars (X), weights (W), outcomes (Y), predicted_outcomes (Y_hat), misc.

    Must be constructed via SampleFrame.from_frame() or SampleFrame.from_sample().

    Mutability:
        SampleFrame is mostly-immutable at the data level.  The underlying
        DataFrame and column-role assignments are set at construction time and
        are not replaced afterwards.  All data-access properties (e.g.
        ``df_covars``, ``df_weights``) return *copies*, so callers cannot
        mutate internal state through the returned objects.

        Controlled mutation points (methods that intentionally modify the
        instance in-place):

        * ``set_active_weight()`` — changes which weight column is active.
        * ``add_weight_column()`` — appends a new weight column to the frame.
        * ``set_weight_metadata()`` — updates weight provenance metadata.

        These mutations are intentional and expected as part of normal usage
        (e.g. after calling ``BalanceFrame.adjust()``).  Outside of these
        methods the object behaves as immutable.
    """

    # pyre-fixme[13]: Initialized in _create() which bypasses __init__
    _df: pd.DataFrame
    # pyre-fixme[13]: Initialized in _create() which bypasses __init__
    _id_column_name: str
    # pyre-fixme[13]: Initialized in _create() which bypasses __init__
    _column_roles: dict[str, list[str]]
    # pyre-fixme[13]: Initialized in _create() which bypasses __init__
    _active_weight_column: str | None
    # pyre-fixme[13]: Initialized in _create() which bypasses __init__
    _weight_metadata: dict[str, Any]
    # SampleFrame is a single-DataFrame container and does NOT manage
    # multi-sample relationships.  _links is exposed as a read-only property
    # returning an empty dict to satisfy the BalanceDFSource protocol.
    # Link management belongs in BalanceDF/BalanceFrame, which can pass
    # explicit links via the BalanceDF(links=...) parameter.

    def __init__(self) -> None:
        raise NotImplementedError(
            "SampleFrame must be constructed via from_frame() or from_sample(). "
            "Direct construction is not supported."
        )

    @property
    def _links(self) -> dict[str, "SampleFrame"]:
        """Return an empty links dict (satisfies BalanceDFSource protocol).

        SampleFrame is a single-DataFrame container and does not manage
        multi-sample relationships.  Link management belongs in
        BalanceDF/BalanceFrame.
        """
        return {}

    def __len__(self) -> int:
        """Return the number of rows in the SampleFrame.

        Returns:
            int: Number of rows in the underlying DataFrame.

        Examples:
            >>> sf = SampleFrame.from_frame(df, id_column="id", weight_column="w")
            >>> len(sf)
            3
        """
        return len(self._df)

    def __deepcopy__(self, memo: dict[int, Any]) -> SampleFrame:
        """Return an independent deep copy of this SampleFrame.

        Uses :meth:`_create` to produce a new instance with deep-copied
        data.  All column-role metadata, weight metadata, and prediction
        metadata are also copied.

        Args:
            memo: The memoisation dictionary passed by :func:`copy.deepcopy`.

        Returns:
            SampleFrame: A new SampleFrame that shares no mutable state with
            the original.

        Examples:
            >>> import copy
            >>> sf2 = copy.deepcopy(sf)
            >>> sf2._df is sf._df
            False
        """
        new_instance = object.__new__(SampleFrame)
        memo[id(self)] = new_instance
        new_instance._df = self._df.copy()
        new_instance._id_column_name = self._id_column_name
        new_instance._column_roles = deepcopy(self._column_roles, memo)
        new_instance._active_weight_column = self._active_weight_column
        new_instance._weight_metadata = deepcopy(self._weight_metadata, memo)
        return new_instance

    @classmethod
    def _create(
        cls,
        df: pd.DataFrame,
        id_column: str,
        covars_columns: list[str],
        weight_columns: list[str],
        outcome_columns: list[str] | None = None,
        predicted_outcome_columns: list[str] | None = None,
        misc_columns: list[str] | None = None,
        _skip_copy: bool = False,
    ) -> SampleFrame:
        """Internal factory method. Use from_frame() instead."""
        instance = object.__new__(cls)
        instance._df = df if _skip_copy else df.copy()
        instance._id_column_name = id_column
        instance._column_roles = {
            "covars": list(covars_columns),
            "weights": list(weight_columns),
            "outcomes": list(outcome_columns or []),
            "predicted": list(predicted_outcome_columns or []),
            "misc": list(misc_columns or []),
        }
        instance._active_weight_column = weight_columns[0] if weight_columns else None
        # Defaults; set via set_weight_metadata() etc.
        instance._weight_metadata = {}
        return instance

    # --- Construction ---

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        id_column: str | None = None,
        covars_columns: list[str] | None = None,
        weight_column: str | None = None,
        outcome_columns: list[str] | tuple[str, ...] | str | None = None,
        predicted_outcome_columns: list[str] | tuple[str, ...] | str | None = None,
        misc_columns: list[str] | tuple[str, ...] | str | None = None,
        check_id_uniqueness: bool = True,
        standardize_types: bool = True,
        use_deepcopy: bool = True,
        id_column_candidates: list[str] | tuple[str, ...] | str | None = None,
    ) -> SampleFrame:
        """Create a SampleFrame from a pandas DataFrame with auto-detection.

        Infers id, weight, and covariate columns from column names when not
        explicitly provided.  Validates the data (e.g., unique IDs,
        non-negative weights) and standardizes dtypes (Int64 -> float64,
        pd.NA -> np.nan).

        Args:
            df (pd.DataFrame): The input DataFrame containing survey or
                observational data.
            id_column (str, optional): Name of the column to use as row
                identifier. If None, guessed from common names
                (``"id"``, ``"ID"``, etc.).
            covars_columns (list of str, optional): Explicit list of covariate
                column names. If None, inferred by exclusion (all columns
                minus id, weight, outcome, predicted, and misc columns).
            weight_column (str, optional): Name of the column containing
                sampling weights. If None, guesses ``"weight"``/``"weights"``
                or creates one filled with 1.0.
            outcome_columns (list of str or str, optional): Column names to
                treat as outcome variables.
            predicted_outcome_columns (list of str or str, optional): Column
                names to treat as predicted outcome variables.
            misc_columns (list of str or str, optional): Column names to treat
                as miscellaneous (excluded from covariates).
            check_id_uniqueness (bool): Whether to verify id uniqueness.
                Defaults to True.
            standardize_types (bool): Whether to standardize dtypes.
                Defaults to True.
            use_deepcopy (bool): Whether to deep-copy the input DataFrame.
                Defaults to True.
            id_column_candidates (list of str, optional): Candidate id column
                names to try when ``id_column`` is None.

        Returns:
            SampleFrame: A validated SampleFrame with standardized dtypes.

        Raises:
            ValueError: If the id column contains nulls or duplicates, if the
                weight column contains nulls or negative values, or if
                specified outcome/predicted/misc columns are missing from the
                DataFrame.

        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame({"id": [1, 2, 3], "weight": [1.0, 2.0, 1.5],
            ...                    "age": [25, 30, 35], "income": [50000, 60000, 70000]})
            >>> sf = SampleFrame.from_frame(df)
            >>> list(sf.df_covars.columns)
            ['age', 'income']
        """
        from balance.util import (
            _pd_convert_all_types,
            _safe_fillna_and_infer,
            _warn_of_df_dtypes_change,
            guess_id_column,
        )

        # Normalize string inputs to lists
        if isinstance(outcome_columns, str):
            outcome_columns = [outcome_columns]
        if isinstance(predicted_outcome_columns, str):
            predicted_outcome_columns = [predicted_outcome_columns]
        if isinstance(misc_columns, str):
            misc_columns = [misc_columns]

        # Deep copy
        df_dtypes = df.dtypes
        if use_deepcopy:
            _df = deepcopy(df)
        else:
            _df = df

        # --- ID column ---
        try:
            id_col_name = guess_id_column(
                df, id_column, possible_id_columns=id_column_candidates
            )
        except (ValueError, TypeError) as exc:
            raise type(exc)(
                "Error while inferring id_column from DataFrame. Specify a valid "
                "'id_column' or provide 'id_column_candidates'. Original error: "
                f"{exc}"
            ) from exc

        if any(_df[id_col_name].isnull()):
            raise ValueError("Null values are not allowed in the id_column")

        if not all(isinstance(x, str) for x in _df[id_col_name].tolist()):
            logger.warning("Casting id column to string")
            _df[id_col_name] = _df[id_col_name].astype(str)

        if check_id_uniqueness and (
            _df[id_col_name].nunique() != len(_df[id_col_name])
        ):
            raise ValueError("Values in the id_column must be unique")

        # --- Type standardization ---
        if standardize_types:
            input_type = ["Int64", "Int32", "int64", "int32", "int16", "int8"]
            output_type = [
                "float64",
                "float32",
                "float64",
                "float32",
                "float16",
                "float16",
            ]
            from importlib.metadata import version as importlib_version

            from packaging.version import Version

            if Version(importlib_version("pandas")) < Version("3.0"):
                input_type.append("string")
                output_type.append("object")
            for i_input, i_output in zip(input_type, output_type):
                _df = _pd_convert_all_types(_df, i_input, i_output)

            _df = cast(pd.DataFrame, _safe_fillna_and_infer(_df, np.nan))

            _warn_of_df_dtypes_change(
                df_dtypes,
                _df.dtypes,
                "df",
                "SampleFrame._df",
            )

        # --- Weight column ---
        if weight_column is None:
            if "weight" in _df.columns:
                logger.warning("Guessing weight column is 'weight'")
                weight_column = "weight"
            elif "weights" in _df.columns:
                logger.warning("Guessing weight column is 'weights'")
                weight_column = "weights"
            else:
                logger.warning(
                    "No weights passed. Adding a 'weight' column and setting all values to 1"
                )
                weight_column = "weight"
                if standardize_types:
                    _df.loc[:, weight_column] = 1.0
                else:
                    _df.loc[:, weight_column] = 1

        # Validate weights
        null_weights = _df[weight_column].isnull()
        if any(null_weights):
            null_weight_rows = _df.loc[null_weights].head()
            null_weight_rows_count = int(null_weights.sum())
            raise ValueError(
                "Null values (including None) are not allowed in the weight_column. "
                "If you wish to remove an observation, either remove it from the df, or use a weight of 0. "
                f"Found {null_weight_rows_count} row(s) with null weights. Preview (up to 5 rows):\n"
                + null_weight_rows.to_string(index=False)
            )

        if not np.issubdtype(_df[weight_column].dtype, np.number):
            raise ValueError("Weights must be numeric")

        if any(_df[weight_column] < 0):
            raise ValueError("Weights must be non-negative")

        # --- Outcome columns validation ---
        outcome_list: list[str] | None = None
        if outcome_columns is not None:
            outcome_list = list(outcome_columns)
            missing_outcome = set(outcome_list).difference(_df.columns)
            if missing_outcome:
                raise ValueError(
                    f"outcome columns {list(missing_outcome)} not in df columns {_df.columns.values.tolist()}"
                )

        # --- Predicted outcome columns validation ---
        predicted_list: list[str] | None = None
        if predicted_outcome_columns is not None:
            predicted_list = list(predicted_outcome_columns)
            missing_predicted = set(predicted_list).difference(_df.columns)
            if missing_predicted:
                raise ValueError(
                    f"predicted outcome columns {list(missing_predicted)} not in df columns {_df.columns.values.tolist()}"
                )

        # --- Misc columns validation ---
        misc_list: list[str] | None = None
        if misc_columns is not None:
            misc_list = list(misc_columns)
            missing_misc = set(misc_list).difference(_df.columns)
            if missing_misc:
                raise ValueError(
                    f"misc columns {list(missing_misc)} not in df columns {_df.columns.values.tolist()}"
                )

        # --- Covariate columns ---
        if covars_columns is not None:
            covar_list = list(covars_columns)
            missing_covars = set(covar_list).difference(_df.columns)
            if missing_covars:
                raise ValueError(
                    f"covariate columns {list(missing_covars)} not in df columns {_df.columns.values.tolist()}"
                )
        else:
            # Infer by exclusion
            ignored = (predicted_list or []) + (misc_list or [])
            special = {id_col_name, weight_column}
            special.update(outcome_list or [])
            special.update(ignored or [])
            covar_list = [c for c in _df.columns if c not in special]

        # --- Column role overlap validation ---
        role_to_columns: dict[str, list[str]] = {
            "covars": covar_list,
            "outcomes": outcome_list or [],
            "predicted": predicted_list or [],
            "misc": misc_list or [],
        }
        roles = list(role_to_columns.keys())
        for i in range(len(roles)):
            for j in range(i + 1, len(roles)):
                role_a, role_b = roles[i], roles[j]
                overlap = set(role_to_columns[role_a]) & set(role_to_columns[role_b])
                if overlap:
                    raise ValueError(
                        f"Column(s) {sorted(overlap)!r} appear in both '{role_a}' and "
                        f"'{role_b}' roles. Each column must have exactly one role."
                    )

        return cls._create(
            df=_df,
            id_column=id_col_name,
            covars_columns=covar_list,
            weight_columns=[weight_column],
            outcome_columns=outcome_list,
            predicted_outcome_columns=predicted_list,
            misc_columns=misc_list,
            _skip_copy=True,
        )

    # --- Column role accessors ---

    @property
    def covars_columns(self) -> list[str]:
        """Names of the covariate columns.

        Returns a copy so that callers cannot accidentally mutate the
        internal column-role registry.

        Returns:
            list[str]: Covariate column names.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "age": [25, 30],
            ...                    "income": [50000, 60000], "weight": [1.0, 1.0]})
            >>> sf = SampleFrame.from_frame(df)
            >>> sf.covars_columns
            ['age', 'income']
        """
        return list(self._column_roles["covars"])

    @property
    def weight_columns(self) -> list[str]:
        """Names of all registered weight columns.

        Returns a copy so that callers cannot accidentally mutate the
        internal column-role registry.

        Returns:
            list[str]: Weight column names.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> sf = SampleFrame._create(
            ...     df=pd.DataFrame({"id": [1], "x": [10], "w1": [1.0], "w2": [2.0]}),
            ...     id_column="id", covars_columns=["x"],
            ...     weight_columns=["w1", "w2"])
            >>> sf.weight_columns
            ['w1', 'w2']
        """
        return list(self._column_roles["weights"])

    @property
    def outcome_columns(self) -> list[str]:
        """Names of the outcome columns.

        Returns a copy so that callers cannot accidentally mutate the
        internal column-role registry.

        Returns:
            list[str]: Outcome column names (empty list if none).

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20],
            ...                    "weight": [1.0, 1.0], "y": [5, 6]})
            >>> sf = SampleFrame.from_frame(df, outcome_columns=["y"])
            >>> sf.outcome_columns
            ['y']
        """
        return list(self._column_roles["outcomes"])

    @property
    def predicted_outcome_columns(self) -> list[str]:
        """Names of the predicted outcome columns.

        Returns a copy so that callers cannot accidentally mutate the
        internal column-role registry.

        Returns:
            list[str]: Predicted outcome column names (empty list if none).

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20],
            ...                    "weight": [1.0, 1.0], "p_y": [0.3, 0.7]})
            >>> sf = SampleFrame.from_frame(df, predicted_outcome_columns=["p_y"])
            >>> sf.predicted_outcome_columns
            ['p_y']
        """
        return list(self._column_roles["predicted"])

    @property
    def misc_columns(self) -> list[str]:
        """Names of the miscellaneous columns.

        Returns a copy so that callers cannot accidentally mutate the
        internal column-role registry.

        Returns:
            list[str]: Misc column names (empty list if none).

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20],
            ...                    "weight": [1.0, 1.0], "region": ["US", "UK"]})
            >>> sf = SampleFrame.from_frame(df, misc_columns=["region"])
            >>> sf.misc_columns
            ['region']
        """
        return list(self._column_roles["misc"])

    @property
    def active_weight_column(self) -> str | None:
        """Name of the currently active weight column, or None.

        Returns:
            str | None: The active weight column name.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20],
            ...                    "weight": [1.0, 2.0]})
            >>> sf = SampleFrame.from_frame(df)
            >>> sf.active_weight_column
            'weight'
        """
        return self._active_weight_column

    @property
    def id_column_name(self) -> str:
        """Name of the ID column.

        Returns:
            str: The ID column name.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20],
            ...                    "weight": [1.0, 2.0]})
            >>> sf = SampleFrame.from_frame(df)
            >>> sf.id_column_name
            'id'
        """
        return self._id_column_name

    # --- DataFrame properties ---

    @property
    def df_covars(self) -> pd.DataFrame:
        """Covariate columns as a DataFrame.

        Returns a copy so that callers cannot accidentally mutate the
        internal data.

        Returns:
            pd.DataFrame: A copy of the covariate columns.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "age": [25, 30],
            ...                    "income": [50000, 60000], "weight": [1.0, 1.0]})
            >>> sf = SampleFrame.from_frame(df)
            >>> covars = sf.df_covars
            >>> covars["age"] = [999, 999]
            >>> list(sf.df_covars["age"])  # internal data unchanged
            [25.0, 30.0]
        """
        cols = self._column_roles["covars"]
        return self._df[cols].copy() if cols else pd.DataFrame(index=self._df.index)

    @property
    def df_weights(self) -> pd.DataFrame:
        """Active weight column as a single-column DataFrame.

        Returns a copy so that callers cannot accidentally mutate the
        internal data.

        Returns:
            pd.DataFrame: A copy of the active weight column, or an empty
                DataFrame if no active weight is set.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20],
            ...                    "weight": [1.0, 2.0]})
            >>> sf = SampleFrame.from_frame(df)
            >>> w = sf.df_weights
            >>> w["weight"] = [999.0, 999.0]
            >>> list(sf.df_weights["weight"])  # internal data unchanged
            [1.0, 2.0]
        """
        if self._active_weight_column:
            return self._df[[self._active_weight_column]].copy()
        return pd.DataFrame(index=self._df.index)

    @property
    def df_outcomes(self) -> pd.DataFrame | None:
        """Outcome columns, or None if no outcomes.

        Returns a copy so that callers cannot accidentally mutate the
        internal data.

        Returns:
            pd.DataFrame | None: A copy of outcome columns, or None if
                no outcome columns are registered.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20],
            ...                    "weight": [1.0, 1.0], "y": [5, 6]})
            >>> sf = SampleFrame.from_frame(df, outcome_columns=["y"])
            >>> out = sf.df_outcomes
            >>> out["y"] = [999, 999]
            >>> list(sf.df_outcomes["y"])  # internal data unchanged
            [5.0, 6.0]
        """
        cols = self._column_roles["outcomes"]
        return self._df[cols].copy() if cols else None

    @property
    def df_misc(self) -> pd.DataFrame | None:
        """Misc columns, or None.

        Returns a copy so that callers cannot accidentally mutate the
        internal data.

        Returns:
            pd.DataFrame | None: A copy of miscellaneous columns, or
                None if no misc columns are registered.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20],
            ...                    "weight": [1.0, 1.0], "region": ["US", "UK"]})
            >>> sf = SampleFrame.from_frame(df, misc_columns=["region"])
            >>> m = sf.df_misc
            >>> m["region"] = ["XX", "XX"]
            >>> list(sf.df_misc["region"])  # internal data unchanged
            ['US', 'UK']
        """
        cols = self._column_roles["misc"]
        return self._df[cols].copy() if cols else None

    def ignored_columns(self) -> pd.DataFrame | None:
        """Return ignored (misc) columns as a DataFrame, or None.

        This is an alias for :attr:`df_misc`, provided for API parity with
        :class:`~balance.sample_class.Sample` which uses ``ignored_columns()``
        to access miscellaneous/ignored columns.

        Returns:
            pd.DataFrame | None: A copy of miscellaneous columns, or
                None if no misc columns are registered.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20],
            ...                    "weight": [1.0, 1.0], "region": ["US", "UK"]})
            >>> sf = SampleFrame.from_frame(df, misc_columns=["region"])
            >>> sf.ignored_columns()
                   region
            0     US
            1     UK
        """
        return self.df_misc

    @property
    def id_column(self) -> pd.Series:
        """The ID column as a Series.

        Returns a copy so that callers cannot accidentally mutate the
        internal data.

        Returns:
            pd.Series: A copy of the ID column.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20],
            ...                    "weight": [1.0, 1.0]})
            >>> sf = SampleFrame.from_frame(df)
            >>> ids = sf.id_column
            >>> ids.iloc[0] = "MUTATED"
            >>> sf.id_column.iloc[0]  # internal data unchanged
            '1'
        """
        return self._df[self._id_column_name].copy()

    @property
    def weight_column(self) -> pd.Series:
        """Active weight column as a Series (BalanceDFSource protocol).

        Returns the active weight column values as a ``pd.Series``.  This is
        the thin protocol-level accessor used by ``BalanceDF`` and its
        subclasses.  Unlike :attr:`df_weights` which returns a single-column
        DataFrame, this returns a plain Series.

        Returns:
            pd.Series: The active weight column values.

        Raises:
            ValueError: If no active weight column is set.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": [1, 2], "x": [10, 20],
            ...                    "weight": [1.0, 2.0]})
            >>> sf = SampleFrame.from_frame(df)
            >>> sf.weight_column.tolist()
            [1.0, 2.0]
        """
        if not self._active_weight_column:
            raise ValueError("No active weight column is set.")
        return self._df[self._active_weight_column].copy()

    def _covar_columns(self) -> pd.DataFrame:
        """Return the covariate DataFrame (BalanceDFSource protocol).

        This method satisfies the ``BalanceDFSource`` protocol and is used
        by ``BalanceDFCovars`` to obtain the covariate columns.  It returns
        the same data as :attr:`df_covars`.

        Returns:
            pd.DataFrame: A copy of the covariate columns.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": [1, 2], "x": [10, 20],
            ...                    "weight": [1.0, 2.0]})
            >>> sf = SampleFrame.from_frame(df)
            >>> list(sf._covar_columns().columns)
            ['x']
        """
        return self.df_covars

    @property
    def _outcome_columns(self) -> pd.DataFrame | None:
        """Outcome columns as a DataFrame, or None (BalanceDFSource protocol).

        This property satisfies the ``BalanceDFSource`` protocol and is used
        by ``BalanceDFOutcomes`` to obtain the outcome columns.  It returns
        the same data as :attr:`df_outcomes`.

        Returns:
            pd.DataFrame | None: A copy of outcome columns, or None if no
                outcome columns are registered.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": [1, 2], "x": [10, 20],
            ...                    "weight": [1.0, 1.0], "y": [5, 6]})
            >>> sf = SampleFrame.from_frame(df, outcome_columns=["y"])
            >>> sf._outcome_columns.columns.tolist()
            ['y']
        """
        return self.df_outcomes

    def set_weights(self, weights: pd.Series | float | None) -> None:
        """Replace the active weight column values (BalanceDFSource protocol).

        This method satisfies the ``BalanceDFSource`` protocol and is used
        by ``BalanceDFWeights.trim()`` to update weight values after trimming.

        If *weights* is a float, all rows are set to that value.  If None,
        all rows are set to 1.0.  If a Series, its values are used directly
        (must match the DataFrame length).

        Args:
            weights (pd.Series | float | None): New weight values.

        Raises:
            ValueError: If no active weight column is set, or if *weights*
                is a Series with a different length than the DataFrame.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": [1, 2], "x": [10, 20],
            ...                    "weight": [1.0, 2.0]})
            >>> sf = SampleFrame.from_frame(df)
            >>> sf.set_weights(pd.Series([3.0, 4.0]))
            >>> sf.weight_column.tolist()
            [3.0, 4.0]
        """
        if not self._active_weight_column:
            raise ValueError("No active weight column is set.")
        if weights is None:
            self._df[self._active_weight_column] = 1.0
        elif isinstance(weights, (int, float)):
            self._df[self._active_weight_column] = float(weights)
        else:
            if len(weights) != len(self._df):
                raise ValueError(
                    f"'weights' length ({len(weights)}) doesn't match "
                    f"DataFrame length ({len(self._df)})"
                )
            self._df[self._active_weight_column] = weights.to_numpy()

    @property
    def df(self) -> pd.DataFrame:
        """Full DataFrame reconstruction."""
        return self._df.copy()

    # --- Weight & prediction provenance ---

    def set_weight_metadata(self, column: str, metadata: dict[str, Any]) -> None:
        """Store provenance metadata for a weight column.

        Metadata is an arbitrary dict that can track adjustment method,
        hyperparameters, timestamps, or any other provenance information
        relevant to how the weight column was computed.

        Args:
            column (str): Name of the weight column.
            metadata (dict): Arbitrary metadata dict (e.g. method name,
                hyperparameters, timestamp).

        Raises:
            ValueError: If *column* is not a registered weight column.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20],
            ...                    "weight": [1.0, 2.0]})
            >>> sf = SampleFrame.from_frame(df)
            >>> sf.set_weight_metadata("weight", {"method": "ipw"})
            >>> sf.weight_metadata()
            {'method': 'ipw'}
        """
        if column not in self._column_roles["weights"]:
            raise ValueError(
                f"'{column}' is not a weight column. "
                f"Weight columns: {self._column_roles['weights']}"
            )
        self._weight_metadata[column] = metadata

    def weight_metadata(self, column: str | None = None) -> dict[str, Any]:
        """Retrieve metadata for a weight column.

        Args:
            column (str, optional): Weight column name. Defaults to the
                active weight column.

        Returns:
            dict: The metadata dict, or an empty dict if none was set.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20],
            ...                    "weight": [1.0, 2.0]})
            >>> sf = SampleFrame.from_frame(df)
            >>> sf.weight_metadata()
            {}
        """
        if column is None:
            column = self._active_weight_column
        return self._weight_metadata.get(column, {}) if column is not None else {}

    def set_active_weight(self, column_name: str) -> None:
        """Set which weight column is the active one.

        The active weight column is the one returned by :attr:`df_weights`.

        Args:
            column_name (str): Must be a registered weight column.

        Raises:
            ValueError: If *column_name* is not a weight column.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> sf = SampleFrame._create(
            ...     df=pd.DataFrame({"id": [1], "x": [10], "w1": [1.0], "w2": [2.0]}),
            ...     id_column="id", covars_columns=["x"],
            ...     weight_columns=["w1", "w2"])
            >>> sf.set_active_weight("w2")
            >>> list(sf.df_weights.columns)
            ['w2']
        """
        if column_name not in self._column_roles["weights"]:
            raise ValueError(
                f"'{column_name}' is not a weight column. "
                f"Weight columns: {self._column_roles['weights']}"
            )
        self._active_weight_column = column_name

    def add_weight_column(
        self,
        name: str,
        values: pd.Series,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a new weight column to the SampleFrame.

        The column is appended to the internal DataFrame and registered
        as a weight column.  Optionally associates provenance metadata.

        Args:
            name (str): Name for the new weight column.
            values (pd.Series): Weight values, must be the same length as
                the DataFrame.
            metadata (dict, optional): Provenance metadata for the new
                column.

        Raises:
            ValueError: If *name* is already a registered weight column,
                if *name* already exists in the DataFrame as a non-weight
                column, or if *values* has a different length than the
                DataFrame.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20],
            ...                    "weight": [1.0, 2.0]})
            >>> sf = SampleFrame.from_frame(df)
            >>> sf.add_weight_column("w_adj", pd.Series([1.5, 1.5]),
            ...                      metadata={"method": "rake"})
            >>> sf._column_roles["weights"]
            ['weight', 'w_adj']
        """
        if name in self._column_roles["weights"]:
            raise ValueError(
                f"'{name}' is already a weight column. "
                f"Use set_weight_metadata() to update metadata."
            )
        if name in self._df.columns:
            raise ValueError(
                f"'{name}' already exists in the DataFrame as a non-weight column. "
                f"Choose a different name."
            )
        if len(values) != len(self._df):
            raise ValueError(
                f"'values' length ({len(values)}) doesn't match "
                f"DataFrame length ({len(self._df)})"
            )
        self._df[name] = values.to_numpy()
        self._column_roles["weights"].append(name)
        if metadata is not None:
            self._weight_metadata[name] = metadata

    def __repr__(self) -> str:
        n_obs = len(self._df)
        n_covars = len(self._column_roles["covars"])
        covar_names = ",".join(self._column_roles["covars"])
        outcome_info = (
            ",".join(self._column_roles["outcomes"])
            if self._column_roles["outcomes"]
            else "None"
        )
        return (
            f"SampleFrame: {n_obs} observations x {n_covars} covariates: {covar_names}\n"
            f"  id_column: {self._id_column_name}, "
            f"weight_columns: {self._column_roles['weights']}, "
            f"outcome_columns: {outcome_info}"
        )

    def __str__(self) -> str:
        return self.__repr__()
