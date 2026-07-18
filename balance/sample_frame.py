# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""SampleFrame: an explicit-role DataFrame container for the Balance library.

Stores covariates, weights, outcomes, outcomes_hat, and ignored columns with
explicit role metadata, replacing the inference-by-exclusion pattern used
in the legacy Sample class.
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import Any, cast, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from typing import Self

    from balance.balancedf_class import BalanceDFSource  # noqa: F401
    from balance.typing import OutcomeLearner  # noqa: F401


logger: logging.Logger = logging.getLogger(__package__)


def _is_float64_dtype(series: pd.Series) -> bool:
    """Return True when the series dtype is exactly NumPy float64."""
    return series.dtype == np.dtype("float64")


class SampleFrame:
    """A DataFrame container with explicit column-role metadata.

    SampleFrame stores data as a single internal pd.DataFrame but with
    explicit metadata tracking which columns belong to which role:
    covars (X), weights (W), outcomes (Y), outcomes_hat (Y_hat), ignored.

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
    _weight_column_name: str | None
    # pyre-fixme[13]: Initialized in _create() which bypasses __init__
    _weight_metadata: dict[str, Any]
    # pyre-fixme[13]: Initialized in _create() which bypasses __init__
    _prediction_metadata: dict[str, Any]
    # pyre-fixme[13]: Initialized in _create() which bypasses __init__
    _outcome_model: dict[str, Any] | None
    # pyre-fixme[13]: Initialized in _create() which bypasses __init__
    _links: dict[str, Any]
    # pyre-fixme[13]: Initialized in _create() which bypasses __init__
    _df_dtypes: pd.Series | None
    # SampleFrame is a single-DataFrame container and does NOT manage
    # multi-sample relationships.  _links is initialised to an empty dict
    # in _create() to satisfy the BalanceDFSource protocol.  BalanceFrame
    # overrides _links with a defaultdict(list) in its own _create().

    def __init__(self) -> None:
        pass

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
        new_instance = object.__new__(type(self))
        memo[id(self)] = new_instance
        new_instance._df = self._df.copy()
        new_instance._id_column_name = self._id_column_name
        new_instance._column_roles = deepcopy(self._column_roles, memo)
        new_instance._weight_column_name = self._weight_column_name
        new_instance._weight_metadata = deepcopy(self._weight_metadata, memo)
        new_instance._prediction_metadata = deepcopy(
            getattr(self, "_prediction_metadata", {}), memo
        )
        # Reference-share the fitted estimators (immutable post-fit and possibly
        # large, e.g. boosting models) but deep-copy the mutable metadata, so an
        # in-place edit of one copy's metadata can't leak into another.  getattr
        # default keeps older pickles loadable.
        outcome_model = getattr(self, "_outcome_model", None)
        if outcome_model is None:
            new_instance._outcome_model = None
        else:
            fit = outcome_model["fit"]
            new_instance._outcome_model = {
                key: (fit if key == "fit" else deepcopy(value, memo))
                for key, value in outcome_model.items()
            }
        new_instance._links = deepcopy(getattr(self, "_links", {}), memo)
        _df_dtypes = getattr(self, "_df_dtypes", None)
        new_instance._df_dtypes = _df_dtypes.copy() if _df_dtypes is not None else None
        return new_instance

    @classmethod
    def _create(
        cls,
        df: pd.DataFrame,
        id_column: str,
        covar_columns: list[str],
        weight_columns: list[str],
        outcome_columns: list[str] | None = None,
        outcomes_hat_columns: list[str] | None = None,
        ignored_columns: list[str] | None = None,
        _skip_copy: bool = False,
        _df_dtypes: pd.Series | None = None,
    ) -> SampleFrame:
        """Internal factory method. Use from_frame() instead."""
        instance = object.__new__(cls)
        instance._df = df if _skip_copy else df.copy()
        instance._id_column_name = id_column
        instance._column_roles = {
            "covars": list(covar_columns),
            "weights": list(weight_columns),
            "outcomes": list(outcome_columns or []),
            "outcomes_hat": list(outcomes_hat_columns or []),
            "ignored": list(ignored_columns or []),
        }
        instance._weight_column_name = weight_columns[0] if weight_columns else None
        # Defaults; set via set_weight_metadata() etc.
        instance._weight_metadata = {}
        # Per-outcomes_hat-column provenance; set via add_outcomes_hat_column().
        instance._prediction_metadata = {}
        # Fitted outcome-model dict; set via fit_outcome_model().
        instance._outcome_model = None
        instance._links = {}
        instance._df_dtypes = _df_dtypes
        return instance

    # --- Construction ---

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        id_column: str | None = None,
        covar_columns: list[str] | None = None,
        weight_column: str | None = None,
        outcome_columns: list[str] | tuple[str, ...] | str | None = None,
        outcomes_hat_columns: list[str] | tuple[str, ...] | str | None = None,
        ignored_columns: list[str] | tuple[str, ...] | str | None = None,
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
            covar_columns (list of str, optional): Explicit list of covariate
                column names. If None, inferred by exclusion (all columns
                minus id, weight, outcome, outcomes_hat, and ignored columns).
            weight_column (str, optional): Name of the column containing
                sampling weights. If None, guesses ``"weight"``/``"weights"``
                or creates one filled with 1.0.
            outcome_columns (list of str or str, optional): Column names to
                treat as outcome variables.
            outcomes_hat_columns (list of str or str, optional): Column
                names to treat as predicted-outcome (Y_hat) variables.
            ignored_columns (list of str or str, optional): Column names to
                ignore (excluded from covariates).
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
                specified outcome/outcomes_hat/ignore columns are missing from the
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
        if isinstance(outcomes_hat_columns, str):
            outcomes_hat_columns = [outcomes_hat_columns]
        if isinstance(ignored_columns, str):
            ignored_columns = [ignored_columns]

        # Deep copy
        df_dtypes = df.dtypes
        if use_deepcopy:
            _df = deepcopy(df)
        else:
            _df = df

        # --- Duplicate column check ---
        dup_mask = _df.columns.duplicated()
        if dup_mask.any():
            dup_names = sorted(set(_df.columns[dup_mask].tolist()))
            raise ValueError(
                f"DataFrame has duplicate column names: {dup_names}. "
                "Please rename columns to be unique before creating a SampleFrame."
            )

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

        try:
            is_numeric = np.issubdtype(_df[weight_column].dtype, np.number)
        except TypeError:
            # Extension dtypes (e.g. pandas StringDtype) can't be interpreted
            # by np.issubdtype — treat them as non-numeric.
            is_numeric = False
        if not is_numeric:
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

        # --- outcomes_hat columns validation ---
        outcomes_hat_list: list[str] | None = None
        if outcomes_hat_columns is not None:
            outcomes_hat_list = list(outcomes_hat_columns)
            missing_outcomes_hat = set(outcomes_hat_list).difference(_df.columns)
            if missing_outcomes_hat:
                raise ValueError(
                    f"outcomes_hat columns {list(missing_outcomes_hat)} not in df columns {_df.columns.values.tolist()}"
                )

        # --- Ignore columns validation ---
        ignore_list: list[str] | None = None
        if ignored_columns is not None:
            ignore_list = list(dict.fromkeys(ignored_columns))  # deduplicate
            missing_ignore = set(ignore_list).difference(_df.columns)
            if missing_ignore:
                raise ValueError(
                    f"ignore columns {list(missing_ignore)} not in df columns {_df.columns.values.tolist()}"
                )
            # ignored_columns must not overlap with id/weight columns
            reserved = {id_col_name, weight_column} - {None}
            overlap_reserved = set(ignore_list).intersection(reserved)
            if overlap_reserved:
                raise ValueError(
                    f"ignore columns cannot include id/weight columns: {overlap_reserved}"
                )

        # --- Covariate columns ---
        if covar_columns is not None:
            covar_list = list(covar_columns)
            missing_covars = set(covar_list).difference(_df.columns)
            if missing_covars:
                raise ValueError(
                    f"covariate columns {list(missing_covars)} not in df columns {_df.columns.values.tolist()}"
                )
        else:
            # Infer by exclusion
            ignored = (outcomes_hat_list or []) + (ignore_list or [])
            special = {id_col_name, weight_column}
            special.update(outcome_list or [])
            special.update(ignored or [])
            covar_list = [c for c in _df.columns if c not in special]
            # Round-trip leak guard: an undeclared "<outcome>_hat" column would
            # be inferred as a covariate and silently corrupt a later adjust()'s
            # propensity model. Warn (do not raise) — it is a naming convention.
            stray_hat = [str(c) for c in covar_list if str(c).endswith("_hat")]
            if stray_hat:
                logger.warning(
                    "Inferred covariate column(s) %r end with '_hat' but were "
                    "not declared as outcomes_hat_columns; they will be treated "
                    "as covariates. If these are predicted-outcome (Y_hat) "
                    "columns, pass them via outcomes_hat_columns= to keep them "
                    "out of the covariate/propensity model.",
                    stray_hat,
                )

        # --- Column role overlap validation ---
        role_to_columns: dict[str, list[str]] = {
            "covars": covar_list,
            "outcomes": outcome_list or [],
            "outcomes_hat": outcomes_hat_list or [],
            "ignored": ignore_list or [],
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

        # M4: weight column must not overlap with outcome columns
        if outcome_list and weight_column in outcome_list:
            raise ValueError(
                f"Weight column '{weight_column}' is also listed as an "
                "outcome column. The weight column must be separate from outcomes."
            )

        # M5: warn if explicitly-provided covariates include id or weight
        if covar_columns is not None:
            special_in_covars = [
                c for c in covar_list if c == id_col_name or c == weight_column
            ]
            if special_in_covars:
                logger.warning(
                    "covar_columns contains column(s) %r that are also used as "
                    "id or weight columns. This is likely unintentional.",
                    special_in_covars,
                )

        return cls._create(
            df=_df,
            id_column=id_col_name,
            covar_columns=covar_list,
            weight_columns=[weight_column],
            outcome_columns=outcome_list,
            outcomes_hat_columns=outcomes_hat_list,
            ignored_columns=ignore_list,
            _skip_copy=True,
            _df_dtypes=df_dtypes,
        )

    # --- Column role accessors ---

    @property
    def covar_columns(self) -> list[str]:
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
            >>> sf.covar_columns
            ['age', 'income']
        """
        return list(self._column_roles["covars"])

    @property
    def weight_columns_all(self) -> list[str]:
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
            ...     id_column="id", covar_columns=["x"],
            ...     weight_columns=["w1", "w2"])
            >>> sf.weight_columns_all
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
    def outcomes_hat_columns(self) -> list[str]:
        """Names of the predicted-outcome (Y_hat) columns.

        Returns a copy so that callers cannot accidentally mutate the
        internal column-role registry.

        Returns:
            list[str]: outcomes_hat column names (empty list if none).

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20],
            ...                    "weight": [1.0, 1.0], "p_y": [0.3, 0.7]})
            >>> sf = SampleFrame.from_frame(df, outcomes_hat_columns=["p_y"])
            >>> sf.outcomes_hat_columns
            ['p_y']
        """
        return list(self._column_roles["outcomes_hat"])

    @property
    def outcome_model(self) -> dict[str, Any] | None:
        """The fitted outcome-model dictionary, or None if not fit.

        Mirrors :attr:`~balance.balance_frame.BalanceFrame.model` (which holds
        the *weighting* model) for the outcome-modelling axis.  The dict is
        produced by :meth:`fit_outcome_model` and consumed by
        :meth:`predict_outcomes`; its keys are documented on
        :func:`balance.outcome_models.fit_outcome_model` (``"method"``,
        ``"fit"``, ``"X_matrix_columns"``, ``"perf"``, …).

        Returns:
            dict[str, Any] | None: The stored ``_outcome_model`` dict, or
                ``None`` when no outcome model has been fit.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2", "3", "4"],
            ...                    "age": [25.0, 30.0, 35.0, 40.0],
            ...                    "happiness": [50.0, 55.0, 65.0, 80.0],
            ...                    "weight": [1.0, 1.0, 1.0, 1.0]})
            >>> sf = SampleFrame.from_frame(df, outcome_columns=["happiness"])
            >>> sf.outcome_model is None
            True
            >>> sf.fit_outcome_model()  # doctest: +ELLIPSIS
            <balance.sample_frame.SampleFrame object at ...>
            >>> sf.outcome_model["method"]
            'outcome_model'
        """
        return self._outcome_model

    @property
    def ignored_columns(self) -> list[str]:
        """Names of the ignored columns.

        Returns a copy so that callers cannot accidentally mutate the
        internal column-role registry.

        Returns:
            list[str]: Ignored column names (empty list if none).

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20],
            ...                    "weight": [1.0, 1.0], "region": ["US", "UK"]})
            >>> sf = SampleFrame.from_frame(df, ignored_columns=["region"])
            >>> sf.ignored_columns
            ['region']
        """
        return list(self._column_roles["ignored"])

    @property
    def weight_column(self) -> str | None:
        """Name of the currently active weight column, or None.

        .. note::
            In balance 0.19.0, ``weight_column`` was changed from returning
            weight *data* (``pd.Series``) to returning the column *name*
            (``str``).  If you need weight data, use :attr:`weight_series`.

        Returns:
            str | None: The active weight column name.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20],
            ...                    "weight": [1.0, 2.0]})
            >>> sf = SampleFrame.from_frame(df)
            >>> sf.weight_column
            'weight'
        """
        return self._weight_column_name

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
        if self._weight_column_name:
            return self._df[[self._weight_column_name]].copy()
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
    def df_outcomes_hat(self) -> pd.DataFrame | None:
        """Predicted-outcome (Y_hat) columns, or None if none.

        Mirrors :attr:`df_outcomes` for the ``outcomes_hat`` role.  Returns a
        copy so that callers cannot accidentally mutate the internal data.

        Returns:
            pd.DataFrame | None: A copy of the outcomes_hat columns, or None
                if no outcomes_hat columns are registered.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2", "3", "4"],
            ...                    "age": [25, 30, 35, 40],
            ...                    "weight": [1.0, 1.0, 1.0, 1.0]})
            >>> sf = SampleFrame.from_frame(df)
            >>> sf.add_outcomes_hat_column("happiness_hat",
            ...                            pd.Series([52., 58., 68., 79.]))
            >>> sf.df_outcomes_hat["happiness_hat"].tolist()
            [52.0, 58.0, 68.0, 79.0]
        """
        cols = self._column_roles["outcomes_hat"]
        return self._df[cols].copy() if cols else None

    @property
    def df_ignored(self) -> pd.DataFrame | None:
        """Ignored columns, or None.

        Returns a copy so that callers cannot accidentally mutate the
        internal data.

        Returns:
            pd.DataFrame | None: A copy of ignored columns, or
                None if no ignored columns are registered.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20],
            ...                    "weight": [1.0, 1.0], "region": ["US", "UK"]})
            >>> sf = SampleFrame.from_frame(df, ignored_columns=["region"])
            >>> m = sf.df_ignored
            >>> m["region"] = ["XX", "XX"]
            >>> list(sf.df_ignored["region"])  # internal data unchanged
            ['US', 'UK']
        """
        cols = self._column_roles["ignored"]
        return self._df[cols].copy() if cols else None

    @property
    def id_series(self) -> pd.Series:
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
            >>> ids = sf.id_series
            >>> ids.iloc[0] = "MUTATED"
            >>> sf.id_series.iloc[0]  # internal data unchanged
            '1'
        """
        return self._df[self._id_column_name].copy()

    @property
    def id_column(self) -> str:
        """Name of the ID column.

        .. note::
            In balance 0.20.0, ``id_column`` was changed from returning
            ID *data* (``pd.Series``) to returning the column *name*
            (``str``), for consistency with :attr:`weight_column`.
            If you need ID data, use :attr:`id_series`.

        Returns:
            str: The ID column name.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20],
            ...                    "weight": [1.0, 1.0]})
            >>> sf = SampleFrame.from_frame(df)
            >>> sf.id_column
            'id'
        """
        return self._id_column_name

    @property
    def weight_series(self) -> pd.Series:
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
            >>> sf.weight_series.tolist()
            [1.0, 2.0]
        """
        if not self._weight_column_name:
            raise ValueError("No active weight column is set.")
        return self._df[self._weight_column_name].copy()

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

    @property
    def _outcomes_hat_columns(self) -> pd.DataFrame | None:
        """outcomes_hat columns as a DataFrame, or None (BalanceDFSource protocol).

        This property satisfies the ``BalanceDFSource`` protocol.  Note the
        codebase quirk it mirrors from :attr:`_outcome_columns`: this ``_*``
        protocol accessor returns the *data* (a DataFrame), not the column
        *names* (the names accessor is :attr:`outcomes_hat_columns`).

        Returns:
            pd.DataFrame | None: A copy of the outcomes_hat columns, or None
                if no outcomes_hat columns are registered.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": [1, 2], "x": [10, 20],
            ...                    "weight": [1.0, 1.0], "p_y": [0.3, 0.7]})
            >>> sf = SampleFrame.from_frame(df, outcomes_hat_columns=["p_y"])
            >>> sf._outcomes_hat_columns.columns.tolist()
            ['p_y']
        """
        return self.df_outcomes_hat

    def set_weights(
        self,
        weights: pd.Series | float | None,
        *,
        use_index: bool = False,
    ) -> None:
        """Replace the active weight column values.

        This is the canonical weight-update method for balance objects.
        Both ``SampleFrame`` and ``BalanceFrame`` use this implementation
        (BalanceFrame delegates here).  It also satisfies the
        ``BalanceDFSource`` protocol and is used by
        ``BalanceDFWeights.trim()`` to update weight values after trimming.

        If *weights* is a float, all rows are set to that value.  If None,
        all rows are set to 1.0.  If a Series, behavior depends on
        *use_index*:

        - ``use_index=False`` (default): the Series must have the same
          length as the DataFrame; values are assigned positionally.
        - ``use_index=True``: values are aligned by index.  Rows whose
          index is missing from *weights* are set to NaN (pandas
          index-alignment semantics), and a warning is emitted.

        All weight values are cast to float64.

        Args:
            weights: New weight values — a Series, scalar, or None.
            use_index: If True, align a Series by index instead of
                requiring an exact length match.

        Raises:
            ValueError: If no active weight column is set, or if
                ``use_index=False`` and a Series has a different length
                than the DataFrame.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": [1, 2], "x": [10, 20],
            ...                    "weight": [1.0, 2.0]})
            >>> sf = SampleFrame.from_frame(df)
            >>> sf.set_weights(pd.Series([3.0, 4.0]))
            >>> sf.weight_series.tolist()
            [3.0, 4.0]
        """
        if not self._weight_column_name:
            raise ValueError("No active weight column is set.")

        wc = self._weight_column_name

        # Ensure the column is float64 before any assignment.
        if not _is_float64_dtype(self._df[wc]):
            self._df[wc] = self._df[wc].astype("float64")

        if weights is None:
            self._df[wc] = 1.0
        elif isinstance(weights, (int, float)):
            self._df[wc] = float(weights)
        elif use_index:
            self._set_weights_by_index(wc, weights)
        else:
            self._set_weights_positional(wc, weights)

    def _set_weights_by_index(self, wc: str, weights: pd.Series | Any) -> None:
        """Assign *weights* to column *wc* aligned by DataFrame index."""
        if not isinstance(weights, pd.Series):
            raise TypeError(
                f"use_index=True requires a pandas Series (got {type(weights).__name__}). "
                "Pass a Series with an appropriate index, or use use_index=False."
            )
        if not _is_float64_dtype(weights):
            weights = weights.astype("float64")
        if not all(idx in weights.index for idx in self._df.index):
            logger.warning(
                "Not all units will be assigned weights — the weights "
                "Series is missing some of the indices in the DataFrame."
            )
        self._df.loc[:, wc] = weights

    def _set_weights_positional(self, wc: str, weights: pd.Series | Any) -> None:
        """Assign *weights* to column *wc* by position (length must match)."""
        if len(weights) != len(self._df):
            raise ValueError(
                f"'weights' length ({len(weights)}) doesn't match "
                f"DataFrame length ({len(self._df)})"
            )
        if isinstance(weights, pd.Series):
            if not _is_float64_dtype(weights):
                weights = weights.astype("float64")
            self._df[wc] = weights.to_numpy()
        else:
            # numpy array or other array-like
            self._df[wc] = np.asarray(weights, dtype="float64")

    def _next_weight_action_number(self) -> int:
        """Return the next global action number for weight history columns.

        Scans existing columns for ``weight_adjusted_N`` and
        ``weight_trimmed_N`` patterns and returns ``max(N) + 1``, or ``1``
        if no history columns exist yet.
        """
        pattern = re.compile(r"^weight_(?:adjusted|trimmed)_(\d+)$")
        max_n = 0
        for col in self._df.columns:
            m = pattern.match(str(col))
            if m:
                max_n = max(max_n, int(m.group(1)))
        return max_n + 1

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

        Delegates to :func:`~balance.adjustment.trim_weights` for the
        computation, then writes the result back via :meth:`set_weights`.
        A weight history column (``weight_trimmed_N``) is added so the
        pre-trim values are preserved.

        Args:
            ratio: Mean-ratio upper bound.  Mutually exclusive with
                *percentile*.
            percentile: Percentile(s) for winsorization.  Mutually exclusive
                with *ratio*.
            keep_sum_of_weights: Whether to rescale after trimming to
                preserve the original sum of weights.
            target_sum_weights: If provided, rescale trimmed weights so
                their sum equals this numeric target value.  (This is a
                general-purpose rescaling parameter — not related to
                the "target population" concept in BalanceFrame.)
            inplace: If True, mutate this SampleFrame and return it.
                If False (default), return a new SampleFrame with trimmed
                weights and the original left untouched.

        Returns:
            The SampleFrame with trimmed weights (self if *inplace*,
            else a new copy).

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> sf = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2, 3], "weight": [1.0, 2.0, 100.0]}))
            >>> sf2 = sf.trim(ratio=2)
            >>> sf2.weight_series.max() < 100.0
            True
            >>> "weight_trimmed_1" in sf2._df.columns
            True
        """
        from balance.adjustment import trim_weights

        target = self if inplace else deepcopy(self)

        original_weight_name = str(
            target._weight_column_name if target._weight_column_name else "weight"
        )

        # Freeze original weights on first action (adjust or trim).
        if "weight_pre_adjust" not in target._df.columns:
            target.add_weight_column(
                "weight_pre_adjust",
                target._df[original_weight_name].copy(),
            )

        # Compute trimmed weights.
        trimmed = trim_weights(
            target._df[original_weight_name],
            weight_trimming_mean_ratio=ratio,
            weight_trimming_percentile=percentile,
            keep_sum_of_weights=keep_sum_of_weights,
            target_sum_weights=target_sum_weights,
        )

        # Record in weight history.
        n = target._next_weight_action_number()
        col_name = f"weight_trimmed_{n}"
        target.add_weight_column(
            col_name,
            trimmed,
            metadata={
                "method": "trim",
                "trimmed": True,
                "ratio": ratio,
                "percentile": percentile,
                "keep_sum_of_weights": keep_sum_of_weights,
                "target_sum_weights": target_sum_weights,
            },
        )

        # Overwrite active weight column.
        target.set_weights(trimmed, use_index=True)

        return target

    # --- BalanceDF integration ---

    def covars(self, formula: str | list[str] | None = None) -> Any:
        """Return a :class:`~balance.balancedf_class.BalanceDFCovars` for this SampleFrame.

        Creates a covariate analysis view backed by this SampleFrame,
        inheriting any linked sources set via ``_links``.

        Args:
            formula: Optional formula string (or list) for model matrix
                construction. Passed through to BalanceDFCovars.

        Returns:
            BalanceDFCovars: Covariate view backed by this SampleFrame.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> sf = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0],
            ...                   "weight": [1.0, 1.0]}))
            >>> sf.covars().df.columns.tolist()
            ['x']
        """
        from balance.balancedf_class import BalanceDFCovars

        return BalanceDFCovars(cast("BalanceDFSource", self), formula=formula)

    def weights(self) -> Any:
        """Return a :class:`~balance.balancedf_class.BalanceDFWeights` for this SampleFrame.

        Creates a weight analysis view backed by this SampleFrame,
        inheriting any linked sources set via ``_links``.

        Returns:
            BalanceDFWeights: Weight view backed by this SampleFrame.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> sf = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0],
            ...                   "weight": [1.0, 2.0]}))
            >>> sf.weights().df.columns.tolist()
            ['weight']
        """
        from balance.balancedf_class import BalanceDFWeights

        return BalanceDFWeights(cast("BalanceDFSource", self))

    def outcomes(self) -> Any | None:
        """Return a :class:`~balance.balancedf_class.BalanceDFOutcomes`, or None.

        Returns ``None`` if this SampleFrame has no outcome columns.

        Returns:
            BalanceDFOutcomes or None: Outcome view backed by this SampleFrame,
                or ``None`` if no outcomes are defined.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> sf = SampleFrame.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0],
            ...                   "y": [1.0, 0.0], "weight": [1.0, 1.0]}),
            ...     outcome_columns=["y"])
            >>> sf.outcomes().df.columns.tolist()
            ['y']
        """
        if not self._column_roles["outcomes"]:
            return None
        # Deferred import to avoid circular dependency with balancedf_class
        from balance.balancedf_class import BalanceDFOutcomes

        return BalanceDFOutcomes(cast("BalanceDFSource", self))

    def outcomes_hat(self) -> Any | None:
        """Return a :class:`~balance.balancedf_class.BalanceDFOutcomesHat`, or None.

        Returns ``None`` if this SampleFrame has no predicted-outcome
        (``outcomes_hat`` / Y_hat) columns.  When present, the returned view
        exposes the weighted mean / CI machinery over the Y_hat columns (one
        column per predicted outcome), mirroring :meth:`outcomes`.

        Returns:
            BalanceDFOutcomesHat or None: Predicted-outcome view backed by this
                SampleFrame, or ``None`` if no outcomes_hat columns are defined.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2", "3", "4"],
            ...                    "age": [25, 30, 35, 40],
            ...                    "weight": [1.0, 1.0, 1.0, 1.0]})
            >>> sf = SampleFrame.from_frame(df)
            >>> sf.add_outcomes_hat_column("happiness_hat",
            ...                            pd.Series([52., 58., 68., 79.]))
            >>> sf.outcomes_hat().df.columns.tolist()
            ['happiness_hat']
            >>> SampleFrame.from_frame(df).outcomes_hat() is None
            True
        """
        if not self._column_roles["outcomes_hat"]:
            return None
        # Deferred import to avoid circular dependency with balancedf_class
        from balance.balancedf_class import BalanceDFOutcomesHat

        return BalanceDFOutcomesHat(cast("BalanceDFSource", self))

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
            column = self._weight_column_name
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
            ...     id_column="id", covar_columns=["x"],
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
        self._weight_column_name = column_name

    def rename_weight_column(self, old_name: str, new_name: str) -> None:
        """Rename a weight column in-place.

        Renames the column in the DataFrame, updates the column roles list,
        active weight pointer, and weight metadata.

        Args:
            old_name: Current name of the weight column.
            new_name: New name for the weight column.

        Raises:
            ValueError: If *old_name* is not a registered weight column,
                or if *new_name* already exists in the DataFrame.
        """
        if old_name not in self._column_roles["weights"]:
            raise ValueError(
                f"'{old_name}' is not a weight column. "
                f"Weight columns: {self._column_roles['weights']}"
            )
        if new_name in self._df.columns:
            raise ValueError(
                f"'{new_name}' already exists in the DataFrame. "
                "Choose a different name."
            )
        # Rename in DataFrame
        self._df = self._df.rename(columns={old_name: new_name})
        # Update column roles
        idx = self._column_roles["weights"].index(old_name)
        self._column_roles["weights"][idx] = new_name
        # Update active weight pointer
        if self._weight_column_name == old_name:
            self._weight_column_name = new_name
        # Migrate metadata
        if old_name in self._weight_metadata:
            self._weight_metadata[new_name] = self._weight_metadata.pop(old_name)

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
            values (pd.Series): Weight values.  Must match the DataFrame
                length, unless it is a shorter ``pd.Series`` — in which case
                values are aligned by index and missing rows are filled with
                NaN (this supports adjustment functions that drop rows
                internally, e.g., ``na_action="drop"``).  Note: this column
                is a *history* column, not the active weight — the active
                weight is set separately via ``set_weights()``.
            metadata (dict, optional): Provenance metadata for the new
                column.

        Raises:
            ValueError: If *name* is already a registered weight column,
                if *name* already exists in the DataFrame as a non-weight
                column, or if *values* is longer than the DataFrame or is
                a non-Series with a different length.

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
            if isinstance(values, pd.Series) and len(values) < len(self._df):
                # Align by index, padding missing rows with NaN.
                # This supports adjustment functions that drop rows internally
                # (e.g., na_action="drop") and return fewer weights.
                values = values.reindex(self._df.index)
            else:
                raise ValueError(
                    f"'values' length ({len(values)}) doesn't match "
                    f"DataFrame length ({len(self._df)})"
                )
        self._df[name] = values.to_numpy()
        self._column_roles["weights"].append(name)
        if metadata is not None:
            self._weight_metadata[name] = metadata

    def add_outcomes_hat_column(
        self,
        name: str,
        values: pd.Series,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Attach a predicted-outcome (Y_hat) column to the SampleFrame.

        Mirrors :meth:`add_weight_column`: the column is appended to the
        internal DataFrame in place, registered under the ``outcomes_hat``
        role, and optionally associated with provenance metadata (stored in
        ``_prediction_metadata``).  Because the column takes the
        ``outcomes_hat`` role, it is *not* treated as a covariate.

        By convention, predicted-outcome columns are named ``"<outcome>_hat"``
        (e.g. ``"happiness_hat"``).  A ``logging.warning`` is emitted — but
        no error raised — when *name* does not end in ``_hat``.

        Args:
            name (str): Name for the new outcomes_hat column.
            values (pd.Series): Predicted values.  Must match the DataFrame
                length, unless it is a shorter ``pd.Series`` — in which case
                values are aligned by index and missing rows are filled with
                NaN.
            metadata (dict, optional): Provenance metadata for the new column
                (e.g. the fitting method or learner).

        Raises:
            ValueError: If *name* is already an outcomes_hat column, if *name*
                already exists in the DataFrame, or if *values* is longer than
                the DataFrame (or is a non-Series with a different length).

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2", "3", "4"],
            ...                    "age": [25, 30, 35, 40],
            ...                    "weight": [1.0, 1.0, 1.0, 1.0]})
            >>> sf = SampleFrame.from_frame(df)
            >>> sf.add_outcomes_hat_column("happiness_hat",
            ...                            pd.Series([52., 58., 68., 79.]))
            >>> sf.df_outcomes_hat["happiness_hat"].tolist()
            [52.0, 58.0, 68.0, 79.0]
            >>> list(sf.df_covars.columns)
            ['age']
        """
        if name in self._column_roles["outcomes_hat"]:
            raise ValueError(
                f"'{name}' is already an outcomes_hat column. "
                "Choose a different name."
            )
        if name in self._df.columns:
            raise ValueError(
                f"'{name}' already exists in the DataFrame. " "Choose a different name."
            )
        if not name.endswith("_hat"):
            logger.warning(
                "outcomes_hat column %r does not follow the '<outcome>_hat' "
                "naming convention (e.g. 'happiness_hat'). It will still be "
                "registered, but the convention keeps predicted columns "
                "self-describing and discernible from covariates.",
                name,
            )
        if len(values) != len(self._df):
            if isinstance(values, pd.Series) and len(values) < len(self._df):
                # Align by index, padding missing rows with NaN (mirrors
                # add_weight_column, supporting learners that drop rows).
                values = values.reindex(self._df.index)
            else:
                raise ValueError(
                    f"'values' length ({len(values)}) doesn't match "
                    f"DataFrame length ({len(self._df)})"
                )
        self._df[name] = values.to_numpy()
        self._column_roles["outcomes_hat"].append(name)
        if metadata is not None:
            self._prediction_metadata[name] = metadata

    # --- Outcome-model fit / predict (sklearn-style trio) ---

    def _resolve_outcome_columns_for_fit(
        self, outcome_columns: list[str] | str | None
    ) -> list[str]:
        """Resolve the outcome column(s) to fit, defaulting to all outcomes.

        Raises an actionable ``ValueError`` when the resolved set is empty (no
        outcome column to fit a model on) or when a requested column is not a
        registered outcome column.
        """
        if outcome_columns is None:
            resolved = list(self._column_roles["outcomes"])
        elif isinstance(outcome_columns, str):
            resolved = [outcome_columns]
        else:
            resolved = list(outcome_columns)

        if not resolved:
            if outcome_columns is not None:
                raise ValueError(
                    "outcome_columns= was given as an empty selection; pass None to "
                    "use all registered outcome columns, or a non-empty subset."
                )
            raise ValueError(
                "fit_outcome_model requires at least one outcome column, but this "
                "SampleFrame has no registered outcome columns. Construct the frame "
                "with outcome_columns=, or pass outcome_columns= explicitly."
            )
        known = set(self._column_roles["outcomes"])
        missing = [c for c in resolved if c not in known]
        if missing:
            raise ValueError(
                f"outcome_columns {missing} are not registered outcome columns. "
                f"Available outcome columns: {sorted(known)}."
            )
        return resolved

    def _resolve_variables_for_fit(
        self, variables: list[str] | str | None
    ) -> list[str] | None:
        """Resolve the covariate subset to model on, or ``None`` for all covars.

        Validates that every requested variable is a registered covariate column.
        """
        if variables is None:
            return None
        resolved = [variables] if isinstance(variables, str) else list(variables)
        if not resolved:
            raise ValueError(
                "variables= was given as an empty list; pass None to use all "
                "covariates, or a non-empty subset of the covariate columns."
            )
        known = set(self._column_roles["covars"])
        missing = [c for c in resolved if c not in known]
        if missing:
            raise ValueError(
                f"variables {missing} are not registered covariate columns. "
                f"Available covariate columns: {sorted(known)}."
            )
        return resolved

    def fit_outcome_model(
        self,
        *,
        model: OutcomeLearner = "auto",
        outcome_columns: list[str] | str | None = None,
        variables: list[str] | str | None = None,
        formula: str | list[str] | None = None,
        transformations: str | dict[str, Any] | None = None,
        na_action: str = "add_indicator",
        use_model_matrix: bool | str = "auto",
        weighted: bool = False,
        calibrate: bool = False,
        inplace: bool = True,
    ) -> Self:
        """Fit an outcome model ``ĝ(X) ≈ E[Y|X]`` on the responders and store it.

        This is the outcome-modelling counterpart to
        :meth:`~balance.balance_frame.BalanceFrame.fit` (which fits the
        *weighting* model).  It fits a regressor (continuous outcome) or
        classifier (binary outcome) per resolved outcome column on this frame's
        covariates and observed outcome(s), and stores the resulting model dict
        (fitted estimators + preprocessing) on :attr:`outcome_model`.  Like
        sklearn's ``fit()``, it does **not** produce predictions — call
        :meth:`predict_outcomes` (or :meth:`fit_predict_outcomes`) to write the
        ``<outcome>_hat`` columns.

        The fit is **unweighted by default** (``weighted=False``): the outcome
        model estimates ``E[Y|X]`` and is usually best left unbiased by the
        design weights.  Pass ``weighted=True`` to fit with the frame's
        currently-applied weight (weighted least squares / weighted boosting),
        aligned to the covariate index before it is passed through.  Rows whose
        outcome ``Y`` is missing (NaN) are dropped before fitting — the
        covariates and weights are aligned to the retained rows, so the fit uses
        only complete outcome observations.  Re-fitting drops any
        ``<outcome>_hat`` columns left by a previous :meth:`predict_outcomes` so
        a stale prediction cannot linger against a newly-fit model.

        Args:
            model: ``"auto"`` (a ``HistGradientBoosting`` regressor/classifier
                chosen by outcome type), a single sklearn estimator (cloned per
                outcome column — one estimator type only when the outcomes are
                mixed), a ``{"_discrete": clf, "_continuous": reg}`` type map, or
                a ``{outcome_column: estimator}`` column map.
            outcome_columns: Outcome column name(s) to model.  Defaults to all
                of this frame's :attr:`outcome_columns`; raises if none exist.
            variables: Covariate column name(s) to use as the model inputs ``X``.
                Defaults to ``None`` (all of this frame's covariates); pass a
                subset (validated against the covariate role) to restrict the
                model to those columns.
            formula: Optional patsy formula(s) forwarded to the one-hot path.
            transformations: Reserved for parity with IPW; must be ``None`` (the
                replay-safe default) — a non-``None`` value raises.
            na_action: Missing-value handling for the design matrix; only
                ``"add_indicator"`` (default) is supported (``"drop"`` raises).
            use_model_matrix: ``"auto"`` (default) picks the native-categorical
                path for tree/boosting learners on scikit-learn >= 1.4 and the
                one-hot + scaler path otherwise; pass ``True``/``False`` to force.
            weighted: When ``True``, fit with the frame's currently-applied
                weight (weighted least squares / weighted boosting).  Defaults to
                ``False`` — the outcome model predicts ``E[Y|X]`` and is usually
                best left unweighted.  A weighted fit whose estimator's ``fit``
                does not accept ``sample_weight`` raises ``TypeError``.
            calibrate: When ``True``, wrap each classifier in
                ``CalibratedClassifierCV`` (binary outcomes only).
            inplace: If ``True`` (default), mutate this frame (store the model)
                and return ``self``; if ``False``, return a new copy with the
                model stored and leave ``self`` untouched — mirroring
                :meth:`~balance.balance_frame.BalanceFrame.fit`.

        Returns:
            The frame with the fitted outcome model stored (``self`` when
            ``inplace``, else a new copy).

        Raises:
            ValueError: If there are no outcome columns to fit (and none are
                passed), if a requested outcome/variable column is not registered,
                or for the underlying ``fit_outcome_model`` errors
                (``na_action="drop"``, non-``None`` ``transformations``, a
                ``model`` column map missing an outcome column, a single estimator
                for mixed-type outcomes, etc.).
            TypeError: If ``weighted=True`` but the resolved estimator's ``fit``
                does not accept ``sample_weight``.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2", "3", "4"],
            ...                    "age": [25.0, 30.0, 35.0, 40.0],
            ...                    "happiness": [50.0, 55.0, 65.0, 80.0],
            ...                    "weight": [1.0, 1.0, 1.0, 1.0]})
            >>> sf = SampleFrame.from_frame(df, outcome_columns=["happiness"])
            >>> sf.fit_outcome_model()  # doctest: +ELLIPSIS
            <balance.sample_frame.SampleFrame object at ...>
            >>> sf.outcome_model["method"]
            'outcome_model'
            >>> sf.df_outcomes_hat is None  # fit does NOT persist Ŷ
            True
        """
        from balance.outcome_models.outcome_model import (
            fit_outcome_model as _fit_outcome_model,
        )

        target = self if inplace else deepcopy(self)

        resolved_outcomes = target._resolve_outcome_columns_for_fit(outcome_columns)
        resolved_variables = target._resolve_variables_for_fit(variables)

        covars = target.df_covars
        if resolved_variables is not None:
            covars = covars[resolved_variables]
        outcomes = target._df[resolved_outcomes].copy()

        # Drop rows whose outcome Y is missing so the weighted fit uses only
        # complete outcome observations; align covars (and weights) to match.
        complete = outcomes.notna().all(axis=1)
        covars = covars.loc[complete]
        outcomes = outcomes.loc[complete]

        sample_weight: pd.Series | None = None
        if weighted:
            # Align the active weight to the (post-drop) covariate index — the
            # weighting-methods input validator requires the weight index to
            # match the covariate index exactly.
            sample_weight = target.weight_series.loc[covars.index]

        fitted_model = _fit_outcome_model(
            covars,
            outcomes,
            sample_weight=sample_weight,
            model=model,
            formula=formula,
            transformations=transformations,
            na_action=na_action,
            use_model_matrix=use_model_matrix,
            calibrate=calibrate,
        )

        # Drop any superseded <outcome>_hat columns from a prior
        # predict_outcomes so a stale Ŷ can't linger against the new model.
        target._drop_outcomes_hat_columns()
        target._outcome_model = fitted_model
        return target

    def _drop_outcomes_hat_columns(self) -> None:
        """Drop every registered ``outcomes_hat`` column from the frame in place."""
        hat_columns = list(self._column_roles["outcomes_hat"])
        if not hat_columns:
            return
        self._df = self._df.drop(columns=hat_columns)
        self._column_roles["outcomes_hat"] = []
        for col in hat_columns:
            self._prediction_metadata.pop(col, None)

    def predict_outcomes(
        self,
        *,
        data: SampleFrame | None = None,
        populate: bool | None = None,
    ) -> pd.DataFrame:
        """Predict ``outcomes_hat`` from the stored model, optionally persisting.

        Requires a model fit by :meth:`fit_outcome_model`.  Predictions are
        produced on this frame's covariates (or on ``data``'s covariates when a
        ``SampleFrame`` is passed via ``data=``) by replaying the model's stored
        preprocessing.  The returned DataFrame has one column per fitted
        outcome, named ``"<outcome>_hat"``.

        When ``populate`` is ``True``, the predicted columns are written
        onto **this** frame via :meth:`add_outcomes_hat_column` (a same-named Ŷ
        column is dropped and re-added).

        Args:
            data: Optional ``SampleFrame`` whose covariates to score.  Defaults
                to ``None``, meaning predict on this frame's own covariates.
                When scoring a *different* frame, prefer ``populate=False``:
                persisting predictions row-indexed by ``data`` onto this frame
                would row-misalign (they are reindexed to this frame, NaN-padding
                or dropping non-matching rows).
            populate: Whether to persist the predictions onto this frame as
                ``<outcome>_hat`` columns. Defaults to ``True`` when scoring this
                frame (``data=None``) and ``False`` when ``data=`` is given, since a
                different frame's row index would misalign onto this one.

        Returns:
            pd.DataFrame: The predictions, one ``"<outcome>_hat"`` column per
                fitted outcome, indexed by the scored covariate rows.

        Raises:
            ValueError: If no outcome model has been fit on this frame.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2", "3", "4"],
            ...                    "age": [25.0, 30.0, 35.0, 40.0],
            ...                    "happiness": [50.0, 55.0, 65.0, 80.0],
            ...                    "weight": [1.0, 1.0, 1.0, 1.0]})
            >>> sf = SampleFrame.from_frame(df, outcome_columns=["happiness"])
            >>> _ = sf.fit_outcome_model()
            >>> preds = sf.predict_outcomes()
            >>> list(preds.columns)
            ['happiness_hat']
            >>> "happiness_hat" in sf.outcomes_hat_columns
            True
        """
        from balance.outcome_models.outcome_model import (
            predict_outcome as _predict_outcome,
        )

        if self._outcome_model is None:
            raise ValueError(
                "no outcome model has been fit; call fit_outcome_model(...) "
                "(or fit_predict_outcomes(...)) before predict_outcomes()."
            )

        # Default: persist when scoring this frame, but not when scoring a
        # different frame via data= (whose row index would misalign onto self).
        if populate is None:
            populate = data is None

        source: SampleFrame = data if data is not None else self
        covars = source.df_covars

        predictions = _predict_outcome(self._outcome_model, covars)
        result = pd.DataFrame(
            {f"{col}_hat": values for col, values in predictions.items()},
            index=covars.index,
        )

        if populate:
            for column in result.columns:
                column_name = str(column)
                if column_name in self._column_roles["outcomes_hat"]:
                    self._drop_single_outcomes_hat_column(column_name)
                self.add_outcomes_hat_column(column_name, result[column_name])

        return result

    def _drop_single_outcomes_hat_column(self, name: str) -> None:
        """Drop one registered ``outcomes_hat`` column from the frame in place."""
        if name not in self._column_roles["outcomes_hat"]:
            return
        self._df = self._df.drop(columns=[name])
        self._column_roles["outcomes_hat"].remove(name)
        self._prediction_metadata.pop(name, None)

    def fit_predict_outcomes(
        self,
        *,
        populate: bool = True,
        **fit_kwargs: Any,
    ) -> pd.DataFrame:
        """Fit an outcome model then predict on this frame, in one call.

        Convenience wrapper mirroring sklearn's ``fit_predict``: it calls
        :meth:`fit_outcome_model` (in place) with ``**fit_kwargs`` and then
        :meth:`predict_outcomes` on this frame, persisting ``<outcome>_hat`` when
        ``populate=True`` (the default).

        Args:
            populate: When ``True`` (default), persist the predictions onto this
                frame as ``<outcome>_hat`` columns.
            **fit_kwargs: Keyword arguments forwarded to
                :meth:`fit_outcome_model` (e.g. ``model``, ``outcome_columns``,
                ``variables``, ``weighted``).  ``inplace`` is always ``True``
                here.

        Returns:
            pd.DataFrame: The predictions, one ``"<outcome>_hat"`` column per
                fitted outcome.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_frame import SampleFrame
            >>> df = pd.DataFrame({"id": ["1", "2", "3", "4"],
            ...                    "age": [25.0, 30.0, 35.0, 40.0],
            ...                    "happiness": [50.0, 55.0, 65.0, 80.0],
            ...                    "weight": [1.0, 1.0, 1.0, 1.0]})
            >>> sf = SampleFrame.from_frame(df, outcome_columns=["happiness"])
            >>> preds = sf.fit_predict_outcomes()
            >>> list(preds.columns)
            ['happiness_hat']
            >>> sf.outcome_model["method"]
            'outcome_model'
        """
        fit_kwargs.pop("inplace", None)
        self.fit_outcome_model(inplace=True, **fit_kwargs)
        return self.predict_outcomes(populate=populate)

    @classmethod
    def from_sample(cls, sample: Any) -> SampleFrame:
        """Convert a :class:`~balance.sample_class.Sample` to a SampleFrame.

        Preserves the Sample's tabular data and column role assignments:
        id column, weight column, outcome columns, and ignored columns.
        Covariate columns are inferred by exclusion,
        matching the Sample's own logic.

        The internal DataFrame is deep-copied so that the resulting
        SampleFrame is fully independent of the original Sample.

        .. warning:: **Data not preserved in the conversion**

           The following Sample attributes are **not** carried over:

           * ``_adjustment_model`` — the fitted model dictionary stored by
             :meth:`~balance.sample_class.Sample.adjust`.
           * ``_links`` — references to ``target``, ``unadjusted``, and other
             linked Samples (used by :class:`~balance.balancedf_class.BalanceDF`
             for comparative display).
           * **Column ordering** may differ after a round-trip
             (``Sample → SampleFrame → Sample``), since SampleFrame stores
             columns grouped by role rather than preserving the original
             DataFrame column order.

        Args:
            sample: A :class:`~balance.sample_class.Sample` instance.

        Returns:
            SampleFrame: A new SampleFrame mirroring the Sample's data and
                column roles.

        Raises:
            TypeError: If *sample* is not a Sample instance.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_class import Sample
            >>> from balance.sample_frame import SampleFrame
            >>> s = Sample.from_frame(
            ...     pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 2.0]}))
            >>> sf = SampleFrame.from_sample(s)
            >>> list(sf.df_covars.columns)
            ['x']
        """
        # Lazy import: sample_class ↔ sample_frame have a circular dependency.
        from balance.sample_class import Sample

        if not isinstance(sample, Sample):
            raise TypeError(
                f"'sample' must be a Sample instance, got {type(sample).__name__}"
            )

        _id_col = sample.id_series
        _weight_col = sample.weight_series
        if _id_col is None:
            raise ValueError(
                "Sample must have an id_column before converting to SampleFrame."
            )
        if _weight_col is None:
            raise ValueError(
                "Sample must have a weight_column before converting to SampleFrame."
            )
        id_col_name: str = str(_id_col.name)
        weight_col_name: str = str(_weight_col.name)

        outcome_cols: list[str] | None = None
        if sample._outcome_columns is not None:
            outcome_cols = sample._outcome_columns.columns.tolist()

        outcomes_hat_cols: list[str] = getattr(sample, "outcomes_hat_columns", []) or []

        ignored_cols: list[str] = getattr(sample, "_ignored_column_names", []) or []

        df = sample._df
        if df is None:
            raise ValueError("Sample has no DataFrame set.")

        # ``_covar_columns_names()`` derives covariates by excluding special
        # columns (id/weight/outcome/outcomes_hat/ignored); defensively drop
        # any outcomes_hat columns again so each column keeps exactly one role.
        covar_cols = sample._covar_columns_names()
        if outcomes_hat_cols:
            hat_set = set(outcomes_hat_cols)
            covar_cols = [c for c in covar_cols if c not in hat_set]

        return cls._create(
            df=df,
            id_column=id_col_name,
            covar_columns=covar_cols,
            weight_columns=[weight_col_name],
            outcome_columns=outcome_cols,
            outcomes_hat_columns=outcomes_hat_cols if outcomes_hat_cols else None,
            ignored_columns=ignored_cols if ignored_cols else None,
        )

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
            f"weight_columns_all: {self._column_roles['weights']}, "
            f"outcome_columns: {outcome_info}"
        )

    def __str__(self) -> str:
        return self.__repr__()
