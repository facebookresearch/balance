# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from __future__ import annotations

import collections
import inspect
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from balance import adjustment as balance_adjustment, util as balance_util
from balance.stats_and_plots import weights_stats

from balance.stats_and_plots.weighted_comparisons_stats import outcome_variance_ratio
from balance.typing import FilePathOrBuffer

from IPython.lib.display import FileLink


logger: logging.Logger = logging.getLogger(__package__)


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

    # The following attributes are updated when initiating Sample using Sample.fram_frame
    _df = None
    id_column = None
    _outcome_columns = None
    weight_column = None
    _links = None
    _adjustment_model = None

    def __init__(self) -> None:
        # The following checks if the call to Sample() was initiated inside the class itself using from_frame, or outside of it
        # If the call was made internally, it will enable the creation of an instance of the class.
        # This is used when from_frame calls `sample = Sample()`. Keeping the full stack allows this also to work by a child of Sample.
        # If Sample() is called outside of the class structure, it will return the NotImplementedError error.
        try:
            calling_functions = [x.function for x in inspect.stack()]
        except Exception:
            raise NotImplementedError(
                "cannot construct Sample class directly... yet (only by invoking Sample.from_fram(...)"
            )

        if "from_frame" not in calling_functions:
            raise NotImplementedError(
                "cannot construct Sample class directly... yet (only by invoking Sample.from_fram(...)"
            )
        pass

    def __repr__(self: "Sample") -> str:
        return (
            f"({self.__class__.__module__}.{self.__class__.__qualname__})\n"
            f"{self.__str__()}"
        )

    def __str__(self: "Sample", pkg_source: str = __package__) -> str:
        is_adjusted = self.is_adjusted() * "Adjusted "
        n_rows = self._df.shape[0]
        n_variables = self._covar_columns().shape[1]
        has_target = self.has_target() * " with target set"
        adjustment_method = (
            " using " + self.model()["method"]  # pyre-ignore[16]
            # (None is eliminated by if statement)
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

    ################################################################################
    #  Public API
    ################################################################################

    @classmethod
    def from_frame(
        cls: type["Sample"],
        df: pd.DataFrame,
        id_column: Optional[str] = None,
        outcome_columns: Optional[Union[list, tuple, str]] = None,
        weight_column: Optional[str] = None,
        check_id_uniqueness: bool = True,
        standardize_types: bool = True,
    ) -> "Sample":
        """Create a new Sample object.

        NOTE that all integer columns will be converted by defaults into floats. This behavior can be turned off
        by setting standardize_types argument to False.
        The reason this is done by default is because of missing value handeling combined with balance current lack of support
        for pandas Integer types:
            1. Native numpy integers do not support missing values (NA), while pandas Intgers do,
            as well numpy floats. Also,
            2. various functions in balance do not support pandas Integers, while they do support numpy floats.
            3. Hence, since some columns might have missing values, the safest solution is to just convert all integers into numpy floats.

        The id_column is stored as a string, even if the input is an integer.

        Args:
            df (pd.DataFrame): containing the sample's data
            id_column (Optional, Optional[str]): the column of the df which contains the respondent's id
            (should be unique). Defaults to None.
            outcome_columns (Optional, Optional[Union[list, tuple, str]]): names of columns to treat as outcome
            weight_column (Optional, Optional[str]): name of column to treat as weight. If not specified, will
                be guessed. If not found, will be filled with 1.0.
            check_id_uniqueness (Optional, bool): Whether to check if ids are unique. Defaults to True.
            standardize_types (Optional, bool): Whether to standardize types. Defaults to True.
                Int64/int64 -> float64
                Int32/int32 -> float64
                string -> object
                pandas.NA -> numpy.nan (within each cell)
                This is slightly memory intensive (since it copies the data twice),
                but helps keep various functions working for both Int64 and Int32 input columns.

        Returns:
            Sample: a sample object
        """
        # Inititate a Sample() class, inside a from_frame constructor
        sample = cls()

        sample._df = df

        # id column
        id_column = balance_util.guess_id_column(df, id_column)
        if any(sample._df[id_column].isnull()):
            raise ValueError("Null values are not allowed in the id_column")
        if not set(map(type, sample._df[id_column].tolist())) == {  # pyre-fixme[6] ???
            str
        }:
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
            #       we just convert them all to np.float (either 32 or 64).
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
            sample._df = balance_util._pd_convert_all_types(
                sample._df, "Int64", "float64"
            )
            # Move from Int32Dtype() to dtype('int32'):
            sample._df = balance_util._pd_convert_all_types(
                sample._df, "Int32", "float32"
            )
            sample._df = balance_util._pd_convert_all_types(
                sample._df, "int64", "float64"
            )
            sample._df = balance_util._pd_convert_all_types(
                sample._df, "int32", "float32"
            )
            sample._df = balance_util._pd_convert_all_types(
                sample._df, "string", "object"
            )
            # Replace any pandas.NA with numpy.nan:
            sample._df = sample._df.fillna(np.nan)

        # weight column
        if weight_column is None:
            if "weight" in sample._df.columns:
                logger.warning(f"Guessing weight column '{'weight'}'")
            else:
                logger.warning("No weights passed, setting all weights to 1")
                sample._df["weight"] = 1
            sample.weight_column = sample._df["weight"]
        else:
            sample.weight_column = sample._df[weight_column]

        # outcome columns
        if outcome_columns is None:
            sample._outcome_columns = None
        else:
            if isinstance(outcome_columns, str):
                outcome_columns = [outcome_columns]
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
            ),
            axis=1,
        )

    def outcomes(
        self: "Sample",
    ):  # -> "Optional[Type[BalanceOutcomesDF]]" (not imported due to circular dependency)
        """
        Produce a BalanceOutcomeDF from a Sample object.
        See :class:BalanceOutcomesDF.

        Args:
            self (Sample): Sample object.

        Returns:
            BalanceOutcomesDF or None
        """
        if self._outcome_columns is not None:
            # NOTE: must import here so to avoid circular dependency
            from balance.balancedf_class import BalanceOutcomesDF

            return BalanceOutcomesDF(self)
        else:
            return None

    def weights(
        self: "Sample",
    ):  # -> "Optional[Type[BalanceWeightsDF]]" (not imported due to circular dependency)
        """
        Produce a BalanceWeightsDF from a Sample object.
        See :class:BalanceWeightsDF.

        Args:
            self (Sample): Sample object.

        Returns:
            BalanceWeightsDF
        """
        # NOTE: must import here so to avoid circular dependency
        from balance.balancedf_class import BalanceWeightsDF

        return BalanceWeightsDF(self)

    def covars(
        self: "Sample",
    ):  # -> "Optional[Type[BalanceCovarsDF]]" (not imported due to circular dependency)
        """
        Produce a BalanceCovarsDF from a Sample object.
        See :class:BalanceCovarsDF.

        Args:
            self (Sample): Sample object.

        Returns:
            BalanceCovarsDF
        """
        # NOTE: must import here so to avoid circular dependency
        from balance.balancedf_class import BalanceCovarsDF

        return BalanceCovarsDF(self)

    def model(
        self: "Sample",
    ) -> Optional[Dict]:
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
        res = balance_util.model_matrix(self, add_na=True)["sample"]
        return res  # pyre-ignore[7]: ["sample"] only chooses the DataFrame

    ############################################
    # Adjusting and adapting weights of a sample
    ############################################
    def adjust(
        self: "Sample",
        target: Optional["Sample"] = None,
        method: Union[Literal["cbps", "ipw", "null", "poststratify"], Callable] = "ipw",
        *args,
        **kwargs,
    ) -> "Sample":
        """
        Perform adjustment of one sample to match another.
        This function returns a new sample.

        Args:
            target (Optional["Sample"]): Second sample object which should be matched.
                If None, the set traget of the object is used for matching.
            method (str): method for adjustment: cbps, ipw, null, poststratify

        Returns:
            Sample: an adjusted Sample object
        """
        if target is None:
            self._no_target_error()
            target = self._links["target"]

        new_sample = deepcopy(self)
        if type(method) == str:
            adjustment_function = balance_adjustment._find_adjustment_method(method)
        elif callable(method):
            adjustment_function = method
        else:
            raise ValueError("Method should be one of existing weighting methods")

        adjusted = adjustment_function(
            sample_df=self.covars().df,
            sample_weights=self.weight_column,
            target_df=target.covars().df,
            target_weights=target.weight_column,
            *args,
            **kwargs,
        )
        new_sample.set_weights(adjusted["weights"])
        new_sample._adjustment_model = adjusted["model"]
        new_sample._links["unadjusted"] = self
        new_sample._links["target"] = target

        return new_sample

    def set_weights(self, weights: Optional[Union[pd.Series, float]]) -> None:
        """
        Adjusteing the weights of a Sample object.
        This will overwrite the weight_column of the Sample.
        Note that the weights are assigned by index if weights is a pd.Series
        (of Sample.df and weights series)

        Args:
            weights (Optional[Union[pd.Series, float]]): Seiers of weights to add to sample.
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
        self._df[self.weight_column.name] = weights
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
            bool: whetehr the Sample is adjusted or not.
        """
        return ("unadjusted" in self._links) and ("target" in self._links)

    def set_target(self, target: "Sample") -> "Sample":
        """
        Used to set the tagret linked to Sample.

        Args:
            target (Sample): A Sample pbject to be linked as target

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
                It should include 3 componants: "unadjusted", "adjusted", "target".

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
                It should include 3 componants: "unadjusted", "adjusted", "target".

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
                It should include 3 componants: "unadjusted", "adjusted", "target".

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
                It should include 3 componants: "unadjusted", "adjusted", "target".

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

        For more detials see: :func:`BalanceDF.asmd`, :func:`BalanceDF.asmd_improvement`
        and :func:`weights_stats.design_effect`

        Returns:
            str: a summary description of properties of an adjusted sample.
        """
        # asmd
        if self.is_adjusted() or self.has_target():
            asmd = self.covars().asmd()
            n_asmd_covars = len(
                asmd.columns.values[asmd.columns.values != "mean(asmd)"]
            )

        # asmd improvement
        if self.is_adjusted():
            asmd_before = asmd.loc["unadjusted", "mean(asmd)"]
            asmd_improvement = 100 * self.covars().asmd_improvement()

        if self.has_target():
            asmd_now = asmd.loc["self", "mean(asmd)"]

        # design effect
        design_effect = self.design_effect()

        # model performance
        if self.model() is not None:
            if (
                self.model()["method"]  # pyre-ignore[16]
                # (None is eliminated by if statement)
                == "ipw"
            ):
                model_summary = (
                    "Model proportion deviance explained: {dev_exp:.3f}".format(
                        dev_exp=self.model()["perf"]["prop_dev_explained"][0]
                    )
                )
            else:
                # TODO: add model performance for other types of models
                model_summary = None
        else:
            model_summary = None

        out = (
            (
                f"Covar ASMD reduction: {asmd_improvement:.1f}%, design effect: {design_effect:.3f}\n"
                if self.is_adjusted()
                else ""
            )
            + (f"Covar ASMD ({n_asmd_covars} variables): " if self.has_target() else "")
            + (f"{asmd_before:.3f} -> " if self.is_adjusted() else "")
            + (f"{asmd_now:.3f}\n" if self.has_target() else "")
            + (
                f"Model performance: {model_summary}"
                if (model_summary is not None)
                else ""
            )
        )
        return out

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
        - sample_covars -  number of covariates (main covars, before any transofrmation was used)
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
        diagnostics = pd.DataFrame(columns=("metric", "val", "var"))

        # ----------------------------------------------------
        # Properties of the Sample object (dimensions of the data)
        # ----------------------------------------------------
        n_sample_obs, n_sample_covars = self.covars().df.shape
        n_target_obs, n_target_covars = self._links["target"].covars().df.shape

        diagnostics = pd.concat(
            (
                diagnostics,
                pd.DataFrame(
                    {
                        "metric": "size",
                        "val": [
                            n_sample_obs,
                            n_sample_covars,
                            n_target_obs,
                            n_target_covars,
                        ],
                        "var": [
                            "sample_obs",
                            "sample_covars",
                            "target_obs",
                            "target_covars",
                        ],
                    }
                ),
            )
        )

        # ----------------------------------------------------
        # Diagnostics on the weights
        # ----------------------------------------------------
        the_weights = self.weights().df.iloc[
            :, 0
        ]  # should be ['weight'], but this is more robust in case a user uses other names
        weights_diag_var = []
        weights_diag_value = []

        # adding design_effect and variations
        the_weights_de = weights_stats.design_effect(the_weights)
        weights_diag_var.extend(
            ["design_effect", "effective_sample_ratio", "effective_sample_size"]
        )
        weights_diag_value.extend(
            [the_weights_de, 1 / the_weights_de, len(the_weights) / the_weights_de]
        )

        # adding sum of weights, and then normalizing them to n (sample size)
        weights_diag_var.append("sum")
        weights_diag_value.append(the_weights.sum())

        the_weights = the_weights / the_weights.mean()  # normalize weights to sum to n.

        # adding basic summary statistics from describe:
        tmp_describe = the_weights.describe()
        weights_diag_var.extend(["describe_" + i for i in tmp_describe.index])
        weights_diag_value.extend(tmp_describe.to_list())
        # TODO: decide if we want more quantiles of the weights.

        # adding prop_above_and_below
        tmp_props = weights_stats.prop_above_and_below(the_weights)
        weights_diag_var.extend(
            tmp_props.index.to_list()  # pyre-ignore[16]: existing defaults make sure this output is pd.Series with relevant methods.
        )
        weights_diag_value.extend(
            tmp_props.to_list()  # pyre-ignore[16]: existing defaults make sure this output is pd.Series with relevant methods.
        )
        # TODO: decide if we want more numbers (e.g.: 2/3 and 3/2)

        # adding nonparametric_skew and weighted_median_breakdown_point
        weights_diag_var.append("nonparametric_skew")
        weights_diag_value.append(weights_stats.nonparametric_skew(the_weights))

        weights_diag_var.append("weighted_median_breakdown_point")
        weights_diag_value.append(
            weights_stats.weighted_median_breakdown_point(the_weights)
        )

        # Add all the weights_diagnostics to diagnostics
        diagnostics = pd.concat(
            (
                diagnostics,
                pd.DataFrame(
                    {
                        "metric": "weights_diagnostics",
                        "val": weights_diag_value,
                        "var": weights_diag_var,
                    }
                ),
            )
        )

        # ----------------------------------------------------
        # Diagnostics on the model
        # ----------------------------------------------------
        model = self.model()
        diagnostics = pd.concat(
            (
                diagnostics,
                pd.DataFrame(
                    {
                        "metric": "adjustment_method",
                        "val": (0,),
                        "var": model["method"],  # pyre-ignore[16]
                        # (None is eliminated by if statement)
                    }
                ),
            )
        )
        if model["method"] == "ipw":
            #  Scalar values from 'perf' key of dictionary
            fit_single_values = pd.concat(
                [
                    pd.DataFrame({"metric": "model_glance", "val": v, "var": k})
                    for k, v in model["fit"].items()
                    if (isinstance(v, np.ndarray) and v.shape == (1,))
                ]
            )
            diagnostics = pd.concat((diagnostics, fit_single_values))

            #  Extract glmnet output about this regularisation parameter
            lambda_ = model["lambda"]
            lambda_index = model["fit"]["lambdau"] == lambda_
            fit_values = pd.concat(
                [
                    pd.DataFrame(
                        {"metric": "model_glance", "val": v[lambda_index], "var": k}
                    )
                    for k, v in self.model()["fit"].items()
                    if (isinstance(v, np.ndarray) and v.shape) == lambda_index.shape
                ]
            )
            diagnostics = pd.concat((diagnostics, fit_values))

            #  Scalar values from 'perf' key of dictionary
            perf_single_values = pd.concat(
                [
                    pd.DataFrame({"metric": "model_glance", "val": v, "var": k})
                    for k, v in model["perf"].items()
                    if (isinstance(v, np.ndarray) and v.shape == (1,))
                ]
            )
            diagnostics = pd.concat((diagnostics, perf_single_values))

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
        rows_to_keep: Optional[str] = None,
        columns_to_keep: Optional[List[str]] = None,
    ) -> "Sample":
        # TODO: split this into two functions (one for rows and one for columns)
        """
        This function returns a **copy** of the sample object
        after removing ALL columns from _df and _links objects
        (which includes unadjusted and target objects).

        This function is useful when wanting to calculate metrics, such as ASMD, but only on some of the features,
        or part of the observations.

        Args:
            self (Sample): a sample object (preferably after adjustment)
            rows_to_keep (Optional[str], optional): A string with a condition to eval (on some of the columns).
                This will run df.eval(rows_to_keep) which will return a pd.Series of bool by which
                we will filter the Sample object.
                This effects both the df of covars AND the weights column (weight_column)
                AND the outcome column (_outcome_columns), AND the id_column column.
                Input should be a boolean feature, or a condition such as: 'gender == "Female" & age >= 18'.
                Defaults to None.
            columns_to_keep (Optional[List[str]], optional): the covariates of interest.
                Defaults to None, which returns all columns.

        Returns:
            Sample: A copy of the original object. If both rows and columns to keep are None,
                returns the copied object unchanged.
                If some are not None, will update - first the rows - then the columns.
                This performs the transformation on both the sameple's df and its linkes dfs (unadjusted, target).
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
                except pd.core.computation.ops.UndefinedVariableError:
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
    def to_download(self, tempdir: Optional[str] = None) -> FileLink:
        """Creates a downloadable link of the DataFrame of the Sample object.

        File name starts with tmp_balance_out_, and some random file name (using :func:`uuid.uuid4`).

        Args:
            self (Sample): Object.
            tempdir (Optional[str], optional): Defaults to None (which then uses a temporary folder using :func:`tempfile.gettempdir`).

        Returns:
            FileLink: Embedding a local file link in an IPython session, based on path. Using :func:FileLink.
        """
        return balance_util._to_download(self.df, tempdir)

    def to_csv(
        self, path_or_buf: Optional[FilePathOrBuffer] = None, **kwargs
    ) -> Optional[str]:
        """Write df with ids from BalanceDF to a comma-separated values (csv) file.

        Uses :func:`pd.DataFrame.to_csv`.

        If an 'index' argument is not provided then it defaults to False.

        Args:
            self: Object.
            path_or_buf (Optional[FilePathOrBuffer], optional): location where to save the csv.

        Returns:
            Optional[str]: If path_or_buf is None, returns the resulting csv format as a string. Otherwise returns None.
        """
        if "index" not in kwargs:
            kwargs["index"] = False
        return self.df.to_csv(path_or_buf=path_or_buf, **kwargs)

    ################################################################################
    #  Private API
    ################################################################################

    ##################
    # Column accessors
    ##################
    def _special_columns_names(self: "Sample") -> List[str]:
        """
        Returns names of all special columns (id column,
        wegiht column and outcome columns) in Sample.

        Returns:
            List[str]: names of special columns
        """
        return [
            i.name for i in [self.id_column, self.weight_column] if i is not None
        ] + (
            self._outcome_columns.columns.tolist()
            if self._outcome_columns is not None
            else []
        )

    def _special_columns(self: "Sample") -> pd.DataFrame:
        """
        Returns dataframe of all special columns (id column,
        wegiht column and outcome columns) in Sample.

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
