# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import collections
import inspect
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal

import numpy as np
import pandas as pd
from balance import adjustment as balance_adjustment, util as balance_util

from balance.csv_utils import to_csv_with_defaults
from balance.stats_and_plots import weights_stats

from balance.stats_and_plots.weighted_comparisons_stats import outcome_variance_ratio
from balance.typing import DiagnosticScalar, FilePathOrBuffer
from balance.util import (
    _coerce_scalar,
    _detect_high_cardinality_features,
    _verify_value_type,
    HighCardinalityFeature,
)

from IPython.lib.display import FileLink


logger: logging.Logger = logging.getLogger(__package__)


def _concat_metric_val_var(
    diagnostics: pd.DataFrame,
    metric: str,
    vals: list[DiagnosticScalar],
    vars: list[DiagnosticScalar],
) -> pd.DataFrame:
    """Append metric/val/var rows to a diagnostics table.

    This helper centralizes concatenating new diagnostics rows from separate
    val and var lists. The function internally zips these lists together,
    so callers only need to provide aligned sequences.

    Args:
        diagnostics: Existing diagnostics table.
        metric: Name for the ``metric`` column to repeat for each appended row.
        vals: List of values for the ``val`` column. Must have the same length
            as ``vars``.
        vars: List of variable names for the ``var`` column. Must have the same
            length as ``vals``.

    Returns:
        A new DataFrame with the appended rows (input is not modified).

    Raises:
        ValueError: If ``vals`` and ``vars`` have different lengths.

    Examples
    --------
    A typical usage pattern when both ``val`` and ``var`` are sequences::

        diagnostics = _concat_metric_val_var(
            diagnostics,
            "weights_diagnostics",
            the_weights_summary["val"].tolist(),
            the_weights_summary["var"].tolist(),
        )

    With a single row::

        diagnostics = _concat_metric_val_var(
            diagnostics,
            "adjustment_method",
            [0],
            [model["method"]],
        )
    """
    if len(vals) != len(vars):
        raise ValueError(
            f"vals and vars must have the same length, got {len(vals)} and {len(vars)}"
        )

    if len(vals) == 0:
        return diagnostics.copy()

    rows = pd.DataFrame(
        {"metric": [metric] * len(vals), "val": list(vals), "var": list(vars)}
    )

    # Append rows to diagnostics with column alignment
    if diagnostics.empty:
        if diagnostics.columns.empty:
            return rows.reset_index(drop=True)
        rows = rows.reindex(columns=diagnostics.columns, fill_value=pd.NA)
        return rows.reset_index(drop=True)

    rows = rows.reindex(columns=diagnostics.columns, fill_value=pd.NA)
    return pd.concat((diagnostics, rows), ignore_index=True)


class Sample:
    """
    A class used to represent a sample.

    Sample is the main object of balance. It contains a dataframe of unit's observations,
    associated with id and weight.

    Attributes
    ----------
    id_column : pd.Series
        a column representing the ids of the units in sample
    weight_column : pd.Series
        a column representing the weights of the units in sample
    """

    # TODO: fix the following missing pyre-strict issues:
    # The following attributes are updated when initiating Sample using Sample.from_frame
    # pyre-fixme[4]: Attributes are initialized in from_frame()
    _df = None
    # pyre-fixme[4]: Attributes are initialized in from_frame()
    id_column = None
    # pyre-fixme[4]: Attributes are initialized in from_frame()
    _outcome_columns = None
    # pyre-fixme[4]: Attributes are initialized in from_frame()
    _ignored_column_names = None
    # pyre-fixme[4]: Attributes are initialized in from_frame()
    weight_column = None
    # pyre-fixme[4]: Attributes are initialized in from_frame()
    _links = None
    # pyre-fixme[4]: Attributes are initialized in from_frame()
    _adjustment_model = None
    # pyre-fixme[4]: Attributes are initialized in from_frame()
    _df_dtypes = None

    def __init__(self) -> None:
        # The following checks if the call to Sample() was initiated inside the class itself using from_frame, or outside of it
        # If the call was made internally, it will enable the creation of an instance of the class.
        # This is used when from_frame calls `sample = Sample()`. Keeping the full stack allows this also to work by a child of Sample.
        # If Sample() is called outside of the class structure, it will return the NotImplementedError error.
        try:
            calling_functions = [x.function for x in inspect.stack()]
        except Exception:
            raise NotImplementedError(
                "cannot construct Sample class directly... yet (only by invoking Sample.from_frame(...)"
            )

        if "from_frame" not in calling_functions:
            raise NotImplementedError(
                "cannot construct Sample class directly... yet (only by invoking Sample.from_frame(...)"
            )
        pass

    def __repr__(self: "Sample") -> str:
        return (
            f"({self.__class__.__module__}.{self.__class__.__qualname__})\n"
            f"{self.__str__()}"
        )

    def __str__(self: "Sample", pkg_source: str = __package__) -> str:
        """Return a readable summary of the sample and any applied adjustment.

        The summary reports the number of observations, covariate names, id and
        weight columns, available outcome columns, and whether a target has been
        set. When an adjustment is present, quick diagnostics are included such
        as the adjustment method, trimming configuration, and weight summaries
        (design effect and implied effective sample size when available).

        Args:
            pkg_source: Package namespace used in the header of the printed
                object. Defaults to ``balance`` and is primarily useful for
                subclasses that wish to identify their own module.

        Returns:
            str: Multi-line description of the ``Sample`` highlighting key
                structure and adjustment details.

        Examples:
            >>> from balance import Sample
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     "gender": ["f", "m"],
            ...     "age_group": ["18-24", "25-34"],
            ...     "id": [1, 2],
            ...     "weight": [1.2, 0.8],
            ... })
            >>> sample = Sample.from_frame(df, id_column="id", weight_column="weight")
            >>> adjusted = sample.set_target(sample).adjust(method="ipw")
            >>> print(adjusted)
            Adjusted balance Sample object with target set using ipw
            2 observations x 2 variables: gender,age_group
            id_column: id, weight_column: weight,
            outcome_columns: None
            adjustment details:
                method: ipw
                design effect (Deff): 1.013
                effective sample size (ESS): 2.0
            target:
                 balance Sample object
                2 observations x 2 variables: gender,age_group
                id_column: id, weight_column: weight,
                outcome_columns: None
                2 common variables: age_group,gender
        """
        is_adjusted = self.is_adjusted() * "Adjusted "
        n_rows = self._df.shape[0]
        n_variables = self._covar_columns().shape[1]
        has_target = self.has_target() * " with target set"
        adjustment_method = (
            " using " + _verify_value_type(self.model())["method"]
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
                """.format(details="\n            ".join(adjustment_details))

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

    def _quick_adjustment_details(
        self: "Sample", n_rows: int | None = None
    ) -> List[str]:
        """Collect quick-to-compute adjustment diagnostics for display.

        This helper centralizes the lightweight adjustment-related statistics
        surfaced in ``__str__`` so they can be reused by other presentation
        helpers (for example, :meth:`summary`) without duplicating logic.

        Note: This method formats diagnostics for human-readable output.

        Args:
            n_rows: Optional row count to use for effective sample size
                calculations. Defaults to the current sample's row count.

        Returns:
            List[str]: Human-readable lines describing adjustment method,
            trimming configuration, and weight diagnostics when available.

        Examples:
            The ``__str__`` example above shows these details rendered as part
            of the adjusted sample printout. You can also retrieve them
            directly for custom displays:

            >>> sample._quick_adjustment_details()  # doctest: +SKIP
            ['method: ipw', 'design effect (Deff): 1.013', 'effective sample size (ESS): 2.0']
        """

        adjustment_details: List[str] = []
        model = self.model()
        if isinstance(model, dict):
            method = model.get("method")
            if isinstance(method, str):
                adjustment_details.append(f"method: {method}")

            trimming_mean_ratio = model.get("weight_trimming_mean_ratio")
            if trimming_mean_ratio is not None:
                adjustment_details.append(
                    f"weight trimming mean ratio: {trimming_mean_ratio}"
                )

            trimming_percentile = model.get("weight_trimming_percentile")
            if trimming_percentile is not None:
                adjustment_details.append(
                    f"weight trimming percentile: {trimming_percentile}"
                )

        if n_rows is None:
            n_rows = self._df.shape[0]

        design_effect, effective_n, effective_sample_proportion = (
            self._design_effect_diagnostics(n_rows)
        )
        if design_effect is not None:
            adjustment_details.append(f"design effect (Deff): {design_effect:.3f}")
            if effective_sample_proportion is not None:
                adjustment_details.append(
                    f"effective sample size proportion (ESSP): {effective_sample_proportion:.3f}"
                )
            if effective_n is not None:
                adjustment_details.append(
                    f"effective sample size (ESS): {effective_n:.1f}"
                )

        return adjustment_details

    def _design_effect_diagnostics(
        self, n_rows: int | None = None
    ) -> tuple[float | None, float | None, float | None]:
        """
        Compute design effect and related effective sample statistics.

        Args:
            n_rows: Optional number of rows to use for scaling. Defaults to the
                sample size when not provided.

        Returns:
            Tuple of (design_effect, effective_sample_size, effective_sample_proportion).
            If the design effect is unavailable or invalid, all values are ``None``.
        """

        if n_rows is None:
            n_rows = self._df.shape[0]

        if self.weight_column is None:
            return None, None, None

        try:
            design_effect = self.design_effect()
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            logger.debug("Unable to compute design effect: %s", exc)
            return None, None, None

        if design_effect is None or not np.isfinite(design_effect):
            return None, None, None

        effective_sample_size = None
        effective_sample_proportion = None
        if n_rows and design_effect != 0:
            effective_sample_size = n_rows / design_effect
            effective_sample_proportion = effective_sample_size / n_rows

        return float(design_effect), effective_sample_size, effective_sample_proportion

    ################################################################################
    #  Public API
    ################################################################################

    # TODO: add examples to the docstring
    @classmethod
    def from_frame(
        cls: type["Sample"],
        df: pd.DataFrame,
        id_column: str | None = None,
        outcome_columns: List[str] | tuple[str, ...] | str | None = None,
        weight_column: str | None = None,
        ignore_columns: List[str] | tuple[str, ...] | str | None = None,
        check_id_uniqueness: bool = True,
        standardize_types: bool = True,
        use_deepcopy: bool = True,
    ) -> "Sample":
        """
        Create a new Sample object.

        NOTE that all integer columns will be converted by defaults into floats. This behavior can be turned off
        by setting standardize_types argument to False.
        The reason this is done by default is because of missing value handling combined with balance current lack of support
        for pandas Integer types:
            1. Native numpy integers do not support missing values (NA), while pandas Integers do,
            as well numpy floats. Also,
            2. various functions in balance do not support pandas Integers, while they do support numpy floats.
            3. Hence, since some columns might have missing values, the safest solution is to just convert all integers into numpy floats.

        The id_column is stored as a string, even if the input is an integer.

        Args:
            df (pd.DataFrame): containing the sample's data
            id_column (str | None): the column of the df which contains the respondent's id
            (should be unique). Defaults to None.
            outcome_columns (list | tuple | str | None): names of columns to treat as outcome
            weight_column (str | None): name of column to treat as weight. If not specified, will
                be guessed (either "weight" or "weights"). If not found, a new column will be created ("weight") and filled with 1.0.
            ignore_columns (list | tuple | str | None): names of columns to keep on the
                underlying dataframe but ignore in covariate/outcome handling. These columns
                are excluded from outcome statistics and covariate selections. Defaults to None.
            check_id_uniqueness (bool): Whether to check if ids are unique. Defaults to True.
            standardize_types (bool): Whether to standardize types. Defaults to True.
                Int64/int64 -> float64
                Int32/int32 -> float64
                string -> object
                pandas.NA -> numpy.nan (within each cell)
                This is slightly memory intensive (since it copies the data twice),
                but helps keep various functions working for both Int64 and Int32 input columns.
            use_deepcopy (Optional, bool): Whether to have a new df copy inside the sample object.
                If False, then when the sample methods update the internal df then the original df will also be updated.
                Defaults to True.

        Returns:
            Sample: a sample object
        """
        # Inititate a Sample() class, inside a from_frame constructor
        sample = cls()

        sample._df_dtypes = df.dtypes

        if use_deepcopy:
            sample._df = deepcopy(df)
        else:
            sample._df = df

        # id column
        id_column = balance_util.guess_id_column(df, id_column)
        if any(sample._df[id_column].isnull()):
            raise ValueError("Null values are not allowed in the id_column")
        if not all(isinstance(x, str) for x in sample._df[id_column].tolist()):
            logger.warning("Casting id column to string")
            sample._df[id_column] = sample._df[id_column].astype(str)

        if (check_id_uniqueness) and (
            sample._df[id_column].nunique() != len(sample._df[id_column])
        ):
            raise ValueError("Values in the id_column must be unique")
        sample.id_column = sample._df[id_column]

        # TODO: in the future, if we could have all functions work with the original data types, that would be better.
        if standardize_types:
            # Move from some pandas Integer types to numpy float types.
            # NOTE: The rationale is that while pandas integers support missing values,
            #       numpy float types do (storing it as np.nan).
            #       Furthermore, other functions in the package don't handle pandas Integer objects well, so
            #       they must be converted to numpy integers (if they have no missing values).
            #       But since we can't be sure that none of the various objects with the same column will not have NAs,
            #       we just convert them all to float (either 32 or 64).
            #       For more details, see: https://stackoverflow.com/a/53853351
            # This line is after the id_column is set, so to make sure that the conversion happens after it is stored as a string.
            # Move from Int64Dtype() to dtype('int64'):

            # TODO: convert all numeric values (no matter what the original is) to "float64"?
            #       (Instead of mentioning all different types)
            #       using is_numeric_dtype: https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_numeric_dtype.html?highlight=is_numeric_dtype#pandas.api.types.is_numeric_dtype
            #       Also, consider using
            #       https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_string_dtype.html
            #       or https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_object_dtype.html
            #       from non-numeric.
            #       e.d.: balance/util.py?lines=512.
            #           for x in df.columns:
            #               if (is_numeric_dtype(df[x])) and (not is_bool_dtype(df[x])):
            #                   df[x] = df[x].astype("float64")
            input_type = ["Int64", "Int32", "int64", "int32", "int16", "int8", "string"]
            output_type = [
                "float64",
                "float32",  # This changes Int32Dtype() into dtype('int32') (from pandas to numpy)
                "float64",
                "float32",
                "float16",
                "float16",  # Using float16 since float8 doesn't exist, see: https://stackoverflow.com/a/40507235/256662
                "object",
            ]
            for i_input, i_output in zip(input_type, output_type):
                sample._df = balance_util._pd_convert_all_types(
                    sample._df, i_input, i_output
                )

            # Replace any pandas.NA with numpy.nan avoiding downcasting warnings
            # By explicitly setting the result of fillna and using infer_objects
            from balance.util import _safe_fillna_and_infer

            sample._df = _safe_fillna_and_infer(sample._df, np.nan)

            balance_util._warn_of_df_dtypes_change(
                sample._df_dtypes,
                sample._df.dtypes,
                "df",
                "sample._df",
            )

        # weight column
        if weight_column is None:
            if "weight" in sample._df.columns:
                logger.warning("Guessing weight column is 'weight'")
                weight_column = "weight"
            elif "weights" in sample._df.columns:
                logger.warning("Guessing weight column is 'weights'")
                weight_column = "weights"
            else:
                # TODO: The current default when weights are not available is "weight", while the method in balanceDF is called "weights",
                #       and the subclass is called BalanceDFWeights (with and 's' at the end)
                #       In the future, it would be better to be more consistent and use the same name for all variations (e.g.: "weight").
                #       Unless, we move to use more weights columns, and this method could be used to get all of them.
                logger.warning(
                    "No weights passed. Adding a 'weight' column and setting all values to 1"
                )
                weight_column = "weight"
                if standardize_types:
                    sample._df.loc[:, weight_column] = (
                        1.0  # Use 1.0 to ensure float64 type
                    )
                else:
                    sample._df.loc[:, weight_column] = (
                        1  # Use 1 to preserve int64 type when standardize_types=False
                    )

        # verify that the weights are not null
        null_weights = sample._df[weight_column].isnull()
        if any(null_weights):
            null_weight_rows = sample._df.loc[null_weights].head()
            null_weight_rows_count = int(null_weights.sum())
            raise ValueError(
                "Null values (including None) are not allowed in the weight_column. "
                + "If you wish to remove an observation, either remove it from the df, or use a weight of 0. "
                + f"Found {null_weight_rows_count} row(s) with null weights. Preview (up to 5 rows):\n"
                + null_weight_rows.to_string(index=False)
            )

        # verify that the weights are numeric
        if not np.issubdtype(sample._df[weight_column].dtype, np.number):
            raise ValueError("Weights must be numeric")

        # verify that the weights are not negative
        if any(sample._df[weight_column] < 0):
            raise ValueError("Weights must be non-negative")

        sample.weight_column = sample._df[weight_column]

        # ignore columns
        if ignore_columns is None:
            sample._ignored_column_names = []
        else:
            if isinstance(ignore_columns, str):
                ignore_columns = [ignore_columns]

            if not all(isinstance(col, str) for col in ignore_columns):
                raise ValueError("ignore_columns must be strings")

            missing_ignore = set(ignore_columns).difference(sample._df.columns)
            if missing_ignore:
                raise ValueError(
                    f"ignore columns {missing_ignore} not in df columns {sample._df.columns.values.tolist()}"
                )

            duplicate_preserving_order = list(dict.fromkeys(ignore_columns))
            reserved_columns = {id_column, weight_column} - {None}
            overlap_reserved = set(duplicate_preserving_order).intersection(
                reserved_columns
            )
            if overlap_reserved:
                raise ValueError(
                    f"ignore columns cannot include id/weight columns: {overlap_reserved}"
                )
            sample._ignored_column_names = duplicate_preserving_order

        # outcome columns
        if outcome_columns is None:
            sample._outcome_columns = None
        else:
            if isinstance(outcome_columns, str):
                outcome_columns = [outcome_columns]

            overlapping_columns = set(outcome_columns).intersection(
                getattr(sample, "_ignored_column_names", [])
            )
            if overlapping_columns:
                raise ValueError(
                    f"Columns cannot be both ignored and outcomes: {overlapping_columns}"
                )
            try:
                sample._outcome_columns = sample._df.loc[:, outcome_columns]
            except KeyError:
                _all_columns = sample._df.columns.values.tolist()
                raise ValueError(
                    f"outcome columns {outcome_columns} not in df columns {_all_columns}"
                )

        sample._links = collections.defaultdict(list)
        return sample

    ####################
    # Class base methods
    ####################
    @property
    def df(self: "Sample") -> pd.DataFrame:
        """Produce a DataFrame (of the self) from a Sample object.

        Args:
            self (Sample): Sample object.

        Returns:
            pd.DataFrame: with id_columns, and the df values of covars(), outcome() and weights() of the self in the Sample object.
        """
        return pd.concat(
            (
                self.id_column,
                self.covars().df if self.covars() is not None else None,
                self.outcomes().df if self.outcomes() is not None else None,
                self.weights().df if self.weights() is not None else None,
                self.ignored_columns() if self.ignored_columns() is not None else None,
            ),
            axis=1,
        )

    def outcomes(
        self: "Sample",
    ) -> Any:  # -> "Optional[Type[BalanceDFOutcomes]]" (not imported due to circular dependency)
        """
        Produce a BalanceOutcomeDF from a Sample object.
        See :class:BalanceDFOutcomes.

        Args:
            self (Sample): Sample object.

        Returns:
            BalanceDFOutcomes or None
        """
        if self._outcome_columns is not None:
            # NOTE: must import here so to avoid circular dependency
            from balance.balancedf_class import BalanceDFOutcomes

            return BalanceDFOutcomes(self)
        else:
            return None

    def weights(
        self: "Sample",
    ) -> Any:  # -> "Optional[Type[BalanceDFWeights]]" (not imported due to circular dependency)
        """
        Produce a BalanceDFWeights from a Sample object.
        See :class:BalanceDFWeights.

        Args:
            self (Sample): Sample object.

        Returns:
            BalanceDFWeights
        """
        # NOTE: must import here so to avoid circular dependency
        from balance.balancedf_class import BalanceDFWeights

        return BalanceDFWeights(self)

    def covars(
        self: "Sample",
    ) -> (
        Any
    ):  # -> "Optional[Type[BalanceDFCovars]]" (not imported due to circular dependency)
        """
        Produce a BalanceDFCovars from a Sample object.
        See :class:BalanceDFCovars.

        Args:
            self (Sample): Sample object.

        Returns:
            BalanceDFCovars
        """
        # NOTE: must import here so to avoid circular dependency
        from balance.balancedf_class import BalanceDFCovars

        return BalanceDFCovars(self)

    def ignored_columns(self: "Sample") -> pd.DataFrame | None:
        """Return columns marked as ignored on the sample.

        These columns stay on the underlying dataframe for tracking purposes
        but are excluded from covariate selection and outcome statistics.

        Returns:
            Optional[pd.DataFrame]: DataFrame of ignored columns, or None when
            no ignored columns are defined.

        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame(
            ...     {"id": [1, 2], "note": ["x", "y"], "age": [20, 21], "out": [0, 1]}
            ... )
            >>> sample = Sample.from_frame(
            ...     df, id_column="id", outcome_columns="out", ignore_columns=["note"], weight_column=None
            ... )
            >>> sample.ignored_columns().columns.tolist()
            ['note']
        """

        if len(getattr(self, "_ignored_column_names", [])) == 0:
            return None
        return self._df[self._ignored_column_names]

    def model(
        self: "Sample",
    ) -> Dict[str, Any] | None:
        """
        Returns the name of the model used to adjust Sample if adjusted.
        Otherwise returns None.

        Args:
            self (Sample): Sample object.

        Returns:
            str or None: name of model used for adjusting Sample
        """
        if hasattr(self, "_adjustment_model"):
            return self._adjustment_model
        else:
            return None

    def model_matrix(self: "Sample") -> pd.DataFrame:
        """
        Returns the model matrix of sample using :func:`model_matrix`,
        while adding na indicator for null values (see :func:`add_na_indicator`).

        Returns:
            pd.DataFrame: model matrix of sample
        """
        res = _verify_value_type(
            balance_util.model_matrix(self, add_na=True)["sample"], pd.DataFrame
        )
        return res

    ############################################
    # Adjusting and adapting weights of a sample
    ############################################
    def adjust(
        self: "Sample",
        target: "Sample" | None = None,
        method: Literal["cbps", "ipw", "null", "poststratify", "rake"]
        | Callable[..., Any] = "ipw",
        *args: Any,
        **kwargs: Any,
    ) -> "Sample":
        """
        Perform adjustment of one sample to match another.
        This function returns a new sample.

        Args:
            target ("Sample" | None): Second sample object which should be matched.
                If None, the set target of the object is used for matching.
            method (str): method for adjustment: cbps, ipw, null, poststratify, rake

        Returns:
            Sample: an adjusted Sample object

        Note:
            During adjustment, the method automatically detects and warns about high-cardinality
            categorical features (features where most values are unique). These features
            typically don't provide meaningful signal for adjustment and may lead to
            unstable or uniform weights. The warning helps identify features that should be
            reviewed or excluded from the adjustment.

        Examples:
            ::

                import balance
                from sklearn.ensemble import RandomForestClassifier
                from balance import Sample
                from balance import load_data

                # Load simulated data
                target_df, sample_df = load_data()

                sample = Sample.from_frame(sample_df, outcome_columns=["happiness"])
                # Often times we don'y have the outcome for the target. In this case we've added it just to validate later that the weights indeed help us reduce the bias
                target = Sample.from_frame(target_df, outcome_columns=["happiness"])

                sample_with_target = sample.set_target(target)
                adjusted = sample_with_target.adjust()

                rf = RandomForestClassifier(n_estimators=200, random_state=0)
                adjusted_rf = sample_with_target.adjust(model = rf)

                # Print ASMD tables for both adjusted and adjusted_rf
                print("\\n=== Adjusted ASMD ===")
                print(adjusted.covars().asmd().T)

                print("\\n=== Adjusted_RF ASMD ===")
                print(adjusted_rf.covars().asmd().T)

                # output
                #
                # === Adjusted ASMD ===
                # source                  self  unadjusted  unadjusted - self
                # age_group[T.25-34]  0.021777    0.005688          -0.016090
                # age_group[T.35-44]  0.055884    0.312711           0.256827
                # age_group[T.45+]    0.169816    0.378828           0.209013
                # gender[Female]      0.097916    0.375699           0.277783
                # gender[Male]        0.103989    0.379314           0.275324
                # gender[_NA]         0.010578    0.006296          -0.004282
                # income              0.205469    0.494217           0.288748
                # mean(asmd)          0.119597    0.326799           0.207202
                #
                # === Adjusted_RF ASMD ===
                # source                  self  unadjusted  unadjusted - self
                # age_group[T.25-34]  0.074491    0.005688          -0.068804
                # age_group[T.35-44]  0.022383    0.312711           0.290328
                # age_group[T.45+]    0.145628    0.378828           0.233201
                # gender[Female]      0.037700    0.375699           0.337999
                # gender[Male]        0.067392    0.379314           0.311922
                # gender[_NA]         0.051718    0.006296          -0.045422
                # income              0.140655    0.494217           0.353562
                # mean(asmd)          0.091253    0.326799           0.235546
        """
        if target is None:
            self._no_target_error()
            target = self._links["target"]

        new_sample = deepcopy(self)
        if isinstance(method, str):
            adjustment_function = balance_adjustment._find_adjustment_method(method)
        elif callable(method):
            adjustment_function = method
        else:
            raise ValueError("Method should be one of existing weighting methods")

        # Detect high cardinality features in both sample and target covariates
        sample_covars_df = self.covars().df
        target_covars_df = target.covars().df

        sample_high_card = _detect_high_cardinality_features(sample_covars_df)
        target_high_card = _detect_high_cardinality_features(target_covars_df)

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

        adjusted = adjustment_function(
            *args,
            sample_df=sample_covars_df,
            sample_weights=self.weight_column,
            target_df=target_covars_df,
            target_weights=target.weight_column,
            **kwargs,
        )
        new_sample.set_weights(adjusted["weight"])
        new_sample._adjustment_model = adjusted["model"]
        new_sample._links["unadjusted"] = self
        new_sample._links["target"] = target

        return new_sample

    def set_weights(self, weights: pd.Series | float | None) -> None:
        """
        Adjusting the weights of a Sample object.
        This will overwrite the weight_column of the Sample.
        Note that the weights are assigned by index if weights is a pd.Series
        (of Sample.df and weights series)

        Args:
            weights (pd.Series | float | None): Series of weights to add to sample.
                If None or float values, the same weight (or None) will be assigned to all units.

        Returns:
            None, but adapting the Sample weight column to weights
        """
        if isinstance(weights, pd.Series):
            if not all(idx in weights.index for idx in self.df.index):
                logger.warning(
                    """Note that not all Sample units will be assigned weights,
                    since weights are missing some of the indices in Sample.df"""
                )

        if isinstance(weights, pd.Series):
            # For Series weights, always ensure weight column is float64 for proper weighting operations
            if not pd.api.types.is_float_dtype(self._df[self.weight_column.name]):
                self._df[self.weight_column.name] = self._df[
                    self.weight_column.name
                ].astype("float64")

            # Convert weights to float64 if not already
            if not pd.api.types.is_float_dtype(weights):
                weights = weights.astype("float64")

            # Now assign the weights
            self._df.loc[:, self.weight_column.name] = weights
        else:
            # For scalar weights, always ensure weight column is float64 for proper weighting operations
            if not pd.api.types.is_float_dtype(self._df[self.weight_column.name]):
                self._df[self.weight_column.name] = self._df[
                    self.weight_column.name
                ].astype("float64")

            # Now assign the weights
            self._df.loc[:, self.weight_column.name] = weights

        self.weight_column = self._df[self.weight_column.name]

    ####################################
    # Handling links to other dataframes
    ####################################
    def set_unadjusted(self, second_sample: "Sample") -> "Sample":
        """
        Used to set the unadjusted link to Sample.
        This is useful in case one wants to compare two samples.

        Args:
            second_sample (Sample): A second Sample to be set as unadjusted of Sample.

        Returns:
            Sample: a new copy of Sample with unadjusted link attached to the self object.
        """
        if isinstance(second_sample, Sample):
            newsample = deepcopy(self)
            newsample._links["unadjusted"] = second_sample
            return newsample
        else:
            raise TypeError(
                "set_unadjusted must be called with second_sample argument of type Sample"
            )

    def is_adjusted(self) -> bool:
        """Check if a Sample object is adjusted and has target attached

        Returns:
            bool: whether the Sample is adjusted or not.
        """
        return ("unadjusted" in self._links) and ("target" in self._links)

    def set_target(self, target: "Sample") -> "Sample":
        """
        Used to set the target linked to Sample.

        Args:
            target (Sample): A Sample object to be linked as target

        Returns:
            Sample: new copy of Sample with target link attached
        """
        if isinstance(target, Sample):
            newsample = deepcopy(self)
            newsample._links["target"] = target
            return newsample
        else:
            raise ValueError("A target, a Sample object, must be specified")

    def has_target(self) -> bool:
        """
        Check if a Sample object has target attached.

        Returns:
            bool: whether the Sample has target attached
        """
        return "target" in self._links

    ##############################
    # Metrics for adjusted samples
    ##############################
    def covar_means(self: "Sample") -> pd.DataFrame:
        """
        Compare the means of covariates (after using :func:`BalanceDF.model_matrix`) before and after adjustment as compared with target.

        Args:
            self (Sample): A Sample object produces after running :func:`Sample.adjust`.
                It should include 3 components: "unadjusted", "adjusted", "target".

        Returns:
            pd.DataFrame: A DataFrame with 3 columns ("unadjusted", "adjusted", "target"),
            and a row for each feature of the covariates.
            The cells show the mean value. For categorical features, they are first transformed into the one-hot encoding.
            For these columns, since they are all either 0 or 1, their means should be interpreted as proportions.

        Examples:
            ::

                from balance import Sample
                import pandas as pd

                s = Sample.from_frame(
                    pd.DataFrame(
                        {"a": (0, 1, 2), "c": ("a", "b", "c"), "o": (1,3,5), "id": (1, 2, 3)}
                    ),
                    outcome_columns=("o"),
                )
                s_adjusted = s.set_target(s).adjust(method = 'null')
                print(s_adjusted.covar_means())

                    # source  unadjusted  adjusted    target
                    # a         1.000000  1.000000  1.000000
                    # c[a]      0.333333  0.333333  0.333333
                    # c[b]      0.333333  0.333333  0.333333
                    # c[c]      0.333333  0.333333  0.333333
        """
        self._check_if_adjusted()

        means = self.covars().mean()
        means = (
            means.rename(index={"self": "adjusted"})
            .reindex(["unadjusted", "adjusted", "target"])
            .transpose()
        )

        return means

    def design_effect(self) -> np.float64:
        """
        Return the design effect of the weights of Sample. Uses :func:`weights_stats.design_effect`.

        Args:
            self (Sample): A Sample object

        Returns:
            np.float64: Design effect
        """
        return weights_stats.design_effect(self.weight_column)

    def design_effect_prop(self) -> np.float64:
        """
        Return the relative difference in design effect of the weights of the unadjusted sample and the adjusted sample.
        I.e. (Deff of adjusted - Deff of unadjusted) / Deff of unadjusted.
        Uses :func:`weights_stats.design_effect`.

        Args:
            self (Sample): A Sample object produces after running :func:`Sample.adjust`.
                It should include 3 components: "unadjusted", "adjusted", "target".

        Returns:
            np.float64: relative difference in design effect.
        """
        self._check_if_adjusted()
        deff_unadjusted = self._links["unadjusted"].design_effect()
        deff_adjusted = self.design_effect()

        return (deff_adjusted - deff_unadjusted) / deff_unadjusted

    # TODO: add unittest for this function
    def plot_weight_density(self) -> None:
        """Plot the density of weights of Sample.

        Examples:
            ::

                import numpy as np
                import pandas as pd
                from balance.sample_class import Sample


                np.random.seed(123)
                df = pd.DataFrame(
                    {
                        "a": np.random.uniform(size=100),
                        "c": np.random.choice(
                            ["a", "b", "c", "d"],
                            size=100,
                            replace=True,
                            p=[0.01, 0.04, 0.5, 0.45],
                        ),
                        "id": range(100),
                        "weight": np.random.uniform(size=100) + 0.5,
                    }
                )

                a = Sample.from_frame(df)
                sample.weights().plot()
                # The same as:
                sample.plot_weight_density()
        """
        self.weights().plot()

    ##########################################
    # Metrics for outcomes of adjusted samples
    ##########################################
    def outcome_sd_prop(self) -> pd.Series:
        """
        Return the difference in outcome weighted standard deviation (sd) of the unadjusted
        sample and the adjusted sample, relative to the unadjusted weighted sd.
        I.e. (weighted sd of adjusted - weighted sd of unadjusted) / weighted sd  of unadjusted.
        Uses :func:`BalanceDF.weighted_stats.weighted_sd`.

        Args:
            self (Sample): A Sample object produces after running :func:`Sample.adjust`.
                It should include 3 components: "unadjusted", "adjusted", "target".

        Returns:
            pd.Series: (np.float64) relative difference in outcome weighted standard deviation.
        """
        self._check_if_adjusted()
        self._check_outcomes_exists()

        outcome_std = self.outcomes().std()
        adjusted_outcome_sd = outcome_std.loc["self"]
        unadjusted_outcome_sd = outcome_std.loc["unadjusted"]

        return (adjusted_outcome_sd - unadjusted_outcome_sd) / unadjusted_outcome_sd

    def outcome_variance_ratio(self: "Sample") -> pd.Series:
        """The empirical ratio of variance of the outcomes before and after weighting.

        See :func:`outcome_variance_ratio` for details.

        Args:
            self (Sample): A Sample object produces after running :func:`Sample.adjust`.
                It should include 3 components: "unadjusted", "adjusted", "target".

        Returns:
             pd.Series: (np.float64) A series of calculated ratio of variances for each outcome.
        """
        return outcome_variance_ratio(
            self.outcomes().df,
            self._links["unadjusted"].outcomes().df,
            self.weights().df["weight"],
            self._links["unadjusted"].weights().df["weight"],
        )

    # TODO: Add a method that plots the distribution of the outcome (adjusted v.s. unadjusted
    #       if adjusted, and only unadjusted otherwise)

    ##############################################
    # Summary of metrics and diagnostics of Sample
    ##############################################
    def summary(self) -> str:
        """
        Provides a summary of covariate balance, design effect and model properties (if applicable)
        of a sample.

        The summary consolidates high-level diagnostics that are most helpful after adjustment,
        including covariate balance, weight health, outcome behavior, and basic model fit when
        available.

        For more details see: :func:`BalanceDF.asmd`, :func:`BalanceDF.asmd_improvement`
        and :func:`weights_stats.design_effect`

        Returns:
            str: a summary description of properties of an adjusted sample.

        Examples:
            >>> import pandas as pd
            >>> from balance.sample_class import Sample
            >>> survey = Sample.from_frame(
            ...     pd.DataFrame(
            ...         {"x": (0, 1, 1, 0), "id": range(4), "y": (0.1, 0.5, 0.4, 0.9), "w": (1, 2, 1, 1)}
            ...     ),
            ...     id_column="id",
            ...     outcome_columns="y",
            ...     weight_column="w",
            ... )
            >>> target = Sample.from_frame(
            ...     pd.DataFrame({"x": (0, 0, 1, 1), "id": range(4)}),
            ...     id_column="id",
            ... )
            >>> adjusted = survey.set_target(target).adjust(method="null")
            >>> print(adjusted.summary())
            Adjustment details:
                method: null_adjustment
            Covariate diagnostics:
                Covar ASMD reduction: 0.0%
                Covar ASMD (1 variables): 0.173 -> 0.173
                Covar mean KLD reduction: 0.0%
                Covar mean KLD (1 variables): 0.020 -> 0.020
            Weight diagnostics:
                design effect (Deff): 1.120
                effective sample proportion (ESSP): 0.893
                effective sample size (ESS): 3.6
            Outcome weighted means:
                           y
            source
            self       0.480
            unadjusted 0.480
        """
        # Initialize variables
        n_asmd_covars: int = 0
        asmd_before: float = 0.0
        asmd_improvement: float = 0.0
        asmd_now: float = 0.0
        n_kld_covars: int = 0
        kld_before: float = 0.0
        kld_now: float = 0.0
        kld_reduction: float = 0.0

        # asmd
        if self.is_adjusted() or self.has_target():
            asmd = self.covars().asmd()
            n_asmd_covars = len(
                asmd.columns.values[asmd.columns.values != "mean(asmd)"]
            )

            kld = self.covars().kld(aggregate_by_main_covar=True)
            n_kld_covars = len(kld.columns.values[kld.columns.values != "mean(kld)"])

        # asmd improvement
        if self.is_adjusted():
            asmd_before = asmd.loc["unadjusted", "mean(asmd)"]
            asmd_improvement = 100 * self.covars().asmd_improvement()
            kld_before = kld.loc["unadjusted", "mean(kld)"]

        if self.has_target():
            asmd_now = asmd.loc["self", "mean(asmd)"]
            kld_now = kld.loc["self", "mean(kld)"]
            if self.is_adjusted() and kld_before > 0:
                kld_reduction = 100 * (kld_before - kld_now) / kld_before

        # quick, lightweight adjustment details reused with __str__
        quick_adjustment_details: List[str] = []
        if self.is_adjusted():
            quick_adjustment_details = self._quick_adjustment_details(self._df.shape[0])

        # design effect and effective sample diagnostics
        design_effect, effective_sample_size, effective_sample_proportion = (
            self._design_effect_diagnostics(self._df.shape[0])
        )

        # model performance

        if self.model() is not None:
            self_model = _verify_value_type(self.model())
            if self_model["method"] == "ipw":
                model_summary = (
                    "Model proportion deviance explained: {dev_exp:.3f}".format(
                        dev_exp=self_model["perf"]["prop_dev_explained"]
                    )
                )
            else:
                # TODO: add model performance for other types of models
                model_summary = None
        else:
            model_summary = None

        sections: List[str] = []

        adjustment_lines = [
            d
            for d in quick_adjustment_details
            if not d.startswith(
                ("design effect", "effective sample size proportion", "effective sample size (ESS)")
            )
        ]
        if adjustment_lines:
            sections.append(
                "Adjustment details:\n    " + "\n    ".join(adjustment_lines)
            )

        covar_lines: List[str] = []
        if self.has_target():
            if self.is_adjusted():
                covar_lines.append(f"Covar ASMD reduction: {asmd_improvement:.1f}%")
            covar_lines.append(
                f"Covar ASMD ({n_asmd_covars} variables): "
                + (f"{asmd_before:.3f} -> " if self.is_adjusted() else "")
                + f"{asmd_now:.3f}"
            )

            if self.is_adjusted() and kld_before > 0:
                covar_lines.append(f"Covar mean KLD reduction: {kld_reduction:.1f}%")
            covar_lines.append(
                f"Covar mean KLD ({n_kld_covars} variables): "
                + (f"{kld_before:.3f} -> " if self.is_adjusted() else "")
                + f"{kld_now:.3f}"
            )

        if covar_lines:
            sections.append("Covariate diagnostics:\n    " + "\n    ".join(covar_lines))

        if self.is_adjusted():
            weights_lines: List[str] = []
            if design_effect is not None:
                weights_lines.append(f"design effect (Deff): {design_effect:.3f}")
                if effective_sample_proportion is not None:
                    weights_lines.append(
                        f"effective sample size proportion (ESSP): {effective_sample_proportion:.3f}"
                    )
                if effective_sample_size is not None:
                    weights_lines.append(
                        f"effective sample size (ESS): {effective_sample_size:.1f}"
                    )
            else:
                weights_lines.append("design effect (Deff): unavailable")

            sections.append("Weight diagnostics:\n    " + "\n    ".join(weights_lines))

        if self._outcome_columns is not None:
            outcome_means = self.outcomes().mean()
            sections.append(
                "Outcome weighted means:\n"
                + outcome_means.to_string(float_format="{:.3f}".format)
            )

        if model_summary is not None:
            sections.append(f"Model performance: {model_summary}")

        return "\n".join(filter(None, sections))

    def diagnostics(self: "Sample") -> pd.DataFrame:
        # TODO: mention the other diagnostics
        # TODO: update/improve the wiki pages doc is linking to.
        # TODO: move explanation on weights normalization to some external page
        """
        Output a table of diagnostics about adjusted Sample object.

        size
        ======================
        All values in the "size" metrics are AFTER any rows/columns were filtered.
        So, for example, if we use respondents from previous days but filter them for diagnostics purposes, then
        sample_obs and target_obs will NOT include them in the counting. The same is true for sample_covars and target_covars.
        In the "size" metrics we have the following 'var's:
        - sample_obs - number of respondents
        - sample_covars -  number of covariates (main covars, before any transformations were used)
        - target_obs - number of users used to represent the target pop
        - target_covars - like sample_covars, but for target.

        weights_diagnostics
        ======================
        In the "weights_diagnostics" metric we have the following 'var's:
        - design effect (de), effective sample size (n/de), effective sample ratio (1/de). See also:
            - https://en.wikipedia.org/wiki/Design_effect
            - https://en.wikipedia.org/wiki/Effective_sample_size
        - sum
        - describe of the (normalized to sample size) weights (mean, median, std, etc.)
        - prop of the (normalized to sample size) weights that are below or above some numbers (1/2, 1, 2, etc.)
        - nonparametric_skew and weighted_median_breakdown_point

        Why is the diagnostics focused on weights normalized to sample size
        -------------------------------------------------------------------
        There are 3 well known normalizations of weights:
        1. to sum to 1
        2. to sum to target population
        3. to sum to n (sample size)

        Each one has their own merits:
        1. is good if wanting to easily calculate avg of some response var (then we just use sum(w*y) and no need for /sum(w))
        2. is good for sum of stuff. For example, how many people in the US use android? For this we'd like the weight of
            each person to represent their share of the population and then we just sum the weights of the people who use android in the survey.
        3. is good for understanding relative "importance" of a respondent as compared to the weights of others in the survey.
            So if someone has a weight that is >1 it means that this respondent (conditional on their covariates) was 'rare' in the survey,
            so the model we used decided to give them a larger weight to account for all the people like him/her that didn't answer.

        For diagnostics purposes, option 3 is most useful for discussing the distribution of the weights
        (e.g.: how many respondents got a weight >2 or smaller <0.5).
        This is a method (standardized  across surveys) to helping us identify how many of the respondents are "dominating"
        and have a large influence on the conclusion we draw from the survey.

        model_glance
        ======================
        Properties of the model fitted, depends on the model used for weighting.

        covariates ASMD
        ======================
        Includes covariates ASMD before and after adjustment (per level of covariate and aggregated) and the ASMD improvement.

        Args:
            self (Sample): only after running an adjustment with Sample.adjust.

        Returns:
            pd.DataFrame: with 3 columns: ("metric", "val", "var"),
                indicating various tracking metrics on the model.
        """
        logger.info("Starting computation of diagnostics of the fitting")
        self._check_if_adjusted()
        diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

        # ----------------------------------------------------
        # Properties of the Sample object (dimensions of the data)
        # ----------------------------------------------------
        n_sample_obs, n_sample_covars = self.covars().df.shape
        n_target_obs, n_target_covars = self._links["target"].covars().df.shape

        diagnostics = _concat_metric_val_var(
            diagnostics,
            "size",
            [n_sample_obs, n_sample_covars, n_target_obs, n_target_covars],
            ["sample_obs", "sample_covars", "target_obs", "target_covars"],
        )

        # ----------------------------------------------------
        # Diagnostics on the weights
        # ----------------------------------------------------
        the_weights_summary = self.weights().summary()

        # Add all the weights_diagnostics to diagnostics
        diagnostics = _concat_metric_val_var(
            diagnostics,
            "weights_diagnostics",
            list(the_weights_summary["val"]),  # passing a list instead of a pd.Series
            list(the_weights_summary["var"]),
        )

        # ----------------------------------------------------
        # Diagnostics on the model
        # ----------------------------------------------------
        model = _verify_value_type(self.model())
        diagnostics = _concat_metric_val_var(
            diagnostics,
            "adjustment_method",
            [0],
            [model["method"]],
        )
        if model["method"] == "ipw":
            fit = model["fit"]
            params = fit.get_params(deep=False)

            fit_list: List[pd.DataFrame] = []

            # TODO: add tests checking these values
            for array_key in ("n_iter_", "intercept_"):
                array_val = getattr(fit, array_key, None)
                if array_val is None:
                    continue

                array_as_np = np.asarray(array_val)
                if array_as_np.size == 1:
                    fit_list.append(
                        _concat_metric_val_var(
                            pd.DataFrame(),
                            "ipw_model_glance",
                            [array_as_np.item()],
                            [array_key],
                        )
                    )

            # TODO: add tests checking these values
            for param_key, metric_name in (
                ("penalty", "ipw_penalty"),
                ("solver", "ipw_solver"),
            ):
                param_val = params.get(param_key, getattr(fit, param_key, None))
                if isinstance(param_val, str):
                    fit_list.append(
                        _concat_metric_val_var(
                            pd.DataFrame(), metric_name, [0], [param_val]
                        )
                    )

            for scalar_key in ("tol", "l1_ratio"):
                scalar_value = _coerce_scalar(
                    params.get(scalar_key, getattr(fit, scalar_key, None))
                )
                fit_list.append(
                    _concat_metric_val_var(
                        pd.DataFrame(), "model_glance", [scalar_value], [scalar_key]
                    )
                )

            # TODO: add tests checking these values
            multi_class = params.get("multi_class", getattr(fit, "multi_class", None))
            if multi_class is None:
                multi_class = "auto"
            elif not isinstance(multi_class, str):
                multi_class = str(multi_class)

            fit_list.append(
                _concat_metric_val_var(
                    pd.DataFrame(), "ipw_multi_class", [0], [multi_class]
                )
            )

            if len(fit_list) > 0:
                fit_single_values = pd.concat(fit_list, ignore_index=True)
                fit_single_values = fit_single_values.drop_duplicates(
                    subset=["metric", "var"], keep="first"
                )
                diagnostics = pd.concat((diagnostics, fit_single_values))

            #  Extract info about the regularisation parameter
            lambda_value = _coerce_scalar(model["lambda"])
            diagnostics = _concat_metric_val_var(
                diagnostics, "model_glance", [lambda_value], ["lambda"]
            )

            #  Scalar values from 'perf' key of dictionary
            perf_entries: List[pd.DataFrame] = []
            for k, v in model["perf"].items():
                if np.isscalar(v) and k != "coefs":
                    perf_entries.append(
                        _concat_metric_val_var(
                            pd.DataFrame(), "model_glance", [_coerce_scalar(v)], [k]
                        )
                    )

            if perf_entries:
                diagnostics = pd.concat([diagnostics] + perf_entries, ignore_index=True)

            # Model coefficients
            coefs = (
                model["perf"]["coefs"]
                .reset_index()
                .rename({0: "val", "index": "var"}, axis=1)
                .assign(metric="model_coef")
            )
            diagnostics = pd.concat((diagnostics, coefs))

        elif model["method"] == "cbps":
            beta_opt = pd.DataFrame(
                {"val": model["beta_optimal"], "var": model["X_matrix_columns"]}
            ).assign(metric="beta_optimal")
            diagnostics = pd.concat((diagnostics, beta_opt))

            metric = [
                "rescale_initial_result",
                "balance_optimize_result",
                "gmm_optimize_result_glm_init",
                "gmm_optimize_result_bal_init",
            ]
            metric = [x for x in metric for _ in range(2)]
            var = ["success", "message"] * 4
            val = [model[x][y] for (x, y) in zip(metric, var)]

            optimizations = pd.DataFrame({"metric": metric, "var": var, "val": val})
            diagnostics = pd.concat((diagnostics, optimizations))

        # TODO: add model diagnostics for other models

        # ----------------------------------------------------
        # Diagnostics on the covariates correction
        # ----------------------------------------------------
        asmds = self.covars().asmd()

        #  Per-covariate ASMDs
        covar_asmds = (
            asmds.transpose()
            .rename(
                {
                    "self": "covar_asmd_adjusted",
                    "unadjusted": "covar_asmd_unadjusted",
                    "unadjusted - self": "covar_asmd_improvement",
                },
                axis=1,
            )
            .reset_index()
            .melt(id_vars="index")
            .rename({"source": "metric", "value": "val", "index": "var"}, axis=1)
        )
        diagnostics = pd.concat((diagnostics, covar_asmds))

        #  Per-main-covariate ASMDs
        asmds_main = self.covars().asmd(aggregate_by_main_covar=True)
        covar_asmds_main = (
            asmds_main.transpose()
            .rename(
                {
                    "self": "covar_main_asmd_adjusted",
                    "unadjusted": "covar_main_asmd_unadjusted",
                    "unadjusted - self": "covar_main_asmd_improvement",
                },
                axis=1,
            )
            .reset_index()
            # TODO:
            # column index name is different here.
            # think again if that's the best default or not for
            # asmd(aggregate_by_main_covar = True)
            .rename({"main_covar_names": "index"}, axis=1)
            .melt(id_vars="index")
            .rename({"source": "metric", "value": "val", "index": "var"}, axis=1)
        )
        # sort covar_asmds_main to have mean(asmd) at the end of it (for when doing quick checks)
        covar_asmds_main = (
            covar_asmds_main.assign(
                has_mean_asmd=(covar_asmds_main["var"] == "mean(asmd)")
            )
            .sort_values(by=["has_mean_asmd", "var"])
            .drop(columns="has_mean_asmd")
        )
        diagnostics = pd.concat((diagnostics, covar_asmds_main))

        # ----------------------------------------------------
        # Diagnostics if there was an adjustment_failure
        # ----------------------------------------------------
        # This field is used in the cli and filled with an alternative value if needed.
        diagnostics = pd.concat(
            (
                diagnostics,
                pd.DataFrame({"metric": ("adjustment_failure",), "val": (0,)}),
            )
        )

        diagnostics = diagnostics.reset_index(drop=True)

        logger.info("Done computing diagnostics")
        return diagnostics

    ############################################
    # Column and rows modifiers - use carefully!
    ############################################
    def keep_only_some_rows_columns(
        self: "Sample",
        rows_to_keep: str | None = None,
        columns_to_keep: List[str] | None = None,
    ) -> "Sample":
        # TODO: split this into two functions (one for rows and one for columns)
        """
        This function returns the sample object after filtering rows and/or columns from _df and _links objects
        (which includes unadjusted and target objects).

        This function is useful when wanting to calculate metrics, such as ASMD, but only on some of the features,
        or part of the observations.

        Args:
            self (Sample): a sample object (preferably after adjustment)
            rows_to_keep (str | None, optional): A string with a condition to eval (on some of the columns).
                This will run df.eval(rows_to_keep) which will return a pd.Series of bool by which
                we will filter the Sample object.
                This effects both the df of covars AND the weights column (weight_column)
                AND the outcome column (_outcome_columns), AND the id_column column.
                Input should be a boolean feature, or a condition such as: 'gender == "Female" & age >= 18'.
                Defaults to None.
            columns_to_keep (List[str] | None, optional): the covariates of interest.
                Defaults to None, which returns all columns.

        Returns:
            Sample: If both rows and columns to keep are None, returns the original object unchanged.
                Otherwise, returns a copy of the original object with filtering applied - first the rows, then the columns.
                This performs the transformation on both the sample's df and its linked dfs (unadjusted, target).
        """
        if (rows_to_keep is None) and (columns_to_keep is None):
            return self

        # Let's make sure to not ruin our old object:
        self = deepcopy(self)

        if rows_to_keep is not None:
            # let's filter the weights Series and then the df rows
            ss = self.df.eval(rows_to_keep)  # rows to keep after the subset # noqa
            logger.info(
                f"From self -> (rows_filtered/total_rows) = ({ss.sum()}/{len(ss)})"
            )
            # filter ids
            self.id_column = self.id_column[ss]
            # filter weights
            self.weight_column = self.weight_column[ss]
            # filter _df
            self._df = self._df[ss]
            # filter outcomes
            if self._outcome_columns is not None:
                self._outcome_columns = self._outcome_columns[ss]
            # filter links
            for k, v in self._links.items():
                try:
                    ss = v.df.eval(rows_to_keep)  # rows to keep after the subset # noqa
                    logger.info(
                        f"From {k} -> (rows_filtered/total_rows) = ({ss.sum()}/{len(ss)})"
                    )
                    v.id_column = v.id_column[ss]
                    v.weight_column = v.weight_column[ss]
                    v._df = v._df[ss]
                    if v._outcome_columns is not None:
                        v._outcome_columns = v._outcome_columns[ss]
                except pd.errors.UndefinedVariableError:
                    # This can happen, for example, if the row filtering condition depends somehow on a feature that is
                    # in the sample but not in the _links. For example, if filtering over one of the
                    # outcome variables, it would filter out these rows from sample, but it wouldn't have this column to
                    # use in target. So this is meant to capture that when this happens the function won't fail but simply
                    # report it to the user.
                    logger.warning(
                        f"couldn't filter _links['{k}'] using {rows_to_keep}"
                    )

        if columns_to_keep is not None:
            if not (set(columns_to_keep) <= set(self.df.columns)):
                logger.warning(
                    "Note that not all columns_to_keep are in Sample. Only those exists are removed"
                )
            # let's remove columns...
            self._df = self._df.loc[:, self._df.columns.isin(columns_to_keep)]
            for v in self._links.values():
                v._df = v._df.loc[:, v._df.columns.isin(columns_to_keep)]

        return self

    ################
    # Saving results
    ################
    def to_download(self, tempdir: str | None = None) -> FileLink:
        """Creates a downloadable link of the DataFrame of the Sample object.

        File name starts with tmp_balance_out_, and some random file name (using :func:`uuid.uuid4`).

        Args:
            self (Sample): Object.
            tempdir (str | None, optional): Defaults to None (which then uses a temporary folder using :func:`tempfile.gettempdir`).

        Returns:
            FileLink: Embedding a local file link in an IPython session, based on path. Using :func:FileLink.
        """
        return balance_util._to_download(self.df, tempdir)

    def to_csv(
        self, path_or_buf: FilePathOrBuffer | None = None, **kwargs: Any
    ) -> str | None:
        """Write df with ids from BalanceDF to a comma-separated values (csv) file.

        Uses :func:`pd.DataFrame.to_csv`.

        If an 'index' argument is not provided then it defaults to False.

        Args:
            self: Object.
            path_or_buf (FilePathOrBuffer | None, optional): location where to save the csv.

        Returns:
            str | None: If path_or_buf is None, returns the resulting csv format as a string. Otherwise returns None.
        """
        return to_csv_with_defaults(self.df, path_or_buf, **kwargs)

    ################################################################################
    #  Private API
    ################################################################################

    ##################
    # Column accessors
    ##################
    def _special_columns_names(self: "Sample") -> List[str]:
        """
        Returns names of all special columns (id column,
        weight column, outcome columns, and ignored columns) in Sample.

        Returns:
            List[str]: names of special columns
        """
        return (
            [i.name for i in [self.id_column, self.weight_column] if i is not None]
            + (
                self._outcome_columns.columns.tolist()
                if self._outcome_columns is not None
                else []
            )
            + getattr(self, "_ignored_column_names", [])
        )

    # TODO: _special_columns is just like df. Unclear if we need both or not, and why.
    def _special_columns(self: "Sample") -> pd.DataFrame:
        """
        Returns dataframe of all special columns (id column,
        weight column and outcome columns) in Sample.

        Returns:
            pd.DataFrame: special columns
        """
        return self._df[self._special_columns_names()]

    def _covar_columns_names(self: "Sample") -> List[str]:
        """
        Returns names of all covars in Sample.

        Returns:
            List[str]: names of covars
        """
        return [
            c for c in self._df.columns.values if c not in self._special_columns_names()
        ]

    def _covar_columns(self: "Sample") -> pd.DataFrame:
        """
        Returns dataframe of all covars columns in Sample.

        Returns:
            pd.DataFrame: covars columns
        """
        return self._df[self._covar_columns_names()]

    ################
    #  Errors checks
    ################
    def _check_if_adjusted(self) -> None:
        """
        Raises a ValueError if sample is not adjusted
        """
        if not self.is_adjusted():
            raise ValueError(
                "This is not an adjusted Sample. Use sample.adjust to adjust the sample to target"
            )

    def _no_target_error(self: "Sample") -> None:
        """
        Raises a ValueError if sample doesn't have target
        """
        if not self.has_target():
            raise ValueError(
                "This Sample does not have a target set. Use sample.set_target to add target"
            )

    def _check_outcomes_exists(self) -> None:
        """
        Raises a ValueError if sample doesn't have outcome_columns specified.
        """
        if self.outcomes() is None:
            raise ValueError("This Sample does not have outcome columns specified")
