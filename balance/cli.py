# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import inspect
import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import balance
import pandas as pd
from balance import __version__  # @manual
from balance.sample_class import Sample as balance_sample_cls  # @manual
from balance.util import _float_or_none
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression

logger: logging.Logger = logging.getLogger(__package__)


class BalanceCLI:
    """Helper class that encapsulates CLI argument handling and execution.

    Args:
        None.

    Returns:
        None.

    Examples:
        .. code-block:: python
            from argparse import Namespace

            cli = BalanceCLI(Namespace(method="ipw"))
            cli.method()
            # 'ipw'
    """

    def __init__(self, args: Namespace) -> None:
        """Initialize CLI helpers and store parsed arguments.

        Args:
            args: Parsed argparse namespace containing CLI configuration.

        Returns:
            None.

        Examples:
            .. code-block:: python
                from argparse import Namespace

                cli = BalanceCLI(Namespace(method="ipw"))
                cli.method()
                # 'ipw'
        """
        self.args: Namespace = args

        # Create attributes (to be populated later, which will be used in main)
        self._transformations: Dict[str, Any] | str | None = None
        self._formula: str | None = None
        self._penalty_factor: None = None
        self._one_hot_encoding: bool = False
        self._max_de: float | None = None
        self._lambda_min: float | None = None
        self._lambda_max: float | None = None
        self._num_lambdas: int | None = None
        self._weight_trimming_mean_ratio: float = 20.0
        self._sample_cls: Type[balance_sample_cls] = balance_sample_cls
        self._sample_package_name: str = __package__
        self._sample_package_version: str = __version__

    def check_input_columns(self, columns: List[str] | pd.Index) -> None:
        """Validate the input frame includes required columns.

        Args:
            columns: Available column names in the input data.

        Returns:
            None.

        Examples:
            .. code-block:: python
                from argparse import Namespace

                import pandas as pd

                cli = BalanceCLI(
                    Namespace(
                        sample_column="is_respondent",
                        id_column="id",
                        weight_column="weight",
                        covariate_columns="x",
                        batch_columns=None,
                        keep_columns=None,
                        keep_row_column=None,
                        outcome_columns=None,
                    )
                )
                cli.check_input_columns(
                    pd.Index(["is_respondent", "id", "weight", "x"])
                )
                # None
        """
        needed_columns = []
        needed_columns.append(self.sample_column())
        needed_columns.append(self.id_column())
        needed_columns.append(self.weight_column())
        needed_columns.extend(self.covariate_columns())
        if self.has_batch_columns():
            needed_columns.extend(self.batch_columns())
        if self.has_keep_columns():
            keep_cols = self.keep_columns()
            if keep_cols is not None:
                needed_columns.extend(keep_cols)
        if self.has_keep_row_column():
            needed_columns.append(self.keep_row_column())
        if self.has_outcome_columns():
            outcome_columns = self.outcome_columns()
            if outcome_columns is not None:
                needed_columns.extend(outcome_columns)

        for nc in needed_columns:
            assert nc in columns, f"{nc} not in input colums"

    # TODO: decide if to explicitly mention/check here the option of methods or not
    def method(self) -> str:
        """Return the adjustment method name.

        Args:
            None.

        Returns:
            The adjustment method string (for example, ``"ipw"``).

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(method="ipw")).method()
                # 'ipw'
        """
        return self.args.method

    def sample_column(self) -> str:
        """Return the column indicating sample membership.

        Args:
            None.

        Returns:
            Name of the sample indicator column.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(sample_column="is_respondent")).sample_column()
                # 'is_respondent'
        """
        return self.args.sample_column

    def id_column(self) -> str:
        """Return the identifier column name.

        Args:
            None.

        Returns:
            Name of the ID column.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(id_column="id")).id_column()
                # 'id'
        """
        return self.args.id_column

    def weight_column(self) -> str:
        """Return the weight column name.

        Args:
            None.

        Returns:
            Name of the weight column.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(weight_column="weight")).weight_column()
                # 'weight'
        """
        return self.args.weight_column

    def covariate_columns(self) -> str:
        """Return the list of covariate column names.

        Args:
            None.

        Returns:
            Covariate column names parsed from the CLI argument.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(covariate_columns="x,y")).covariate_columns()
                # ['x', 'y']
        """
        return self.args.covariate_columns.split(",")

    def covariate_columns_for_diagnostics(self) -> List[str]:
        """Return covariate columns used for diagnostics reporting.

        Args:
            None.

        Returns:
            List of columns to keep in diagnostics or ``None``.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(
                Namespace(covariate_columns_for_diagnostics="x,y")
                ).covariate_columns_for_diagnostics()
                # ['x', 'y']
        """
        out = self.args.covariate_columns_for_diagnostics
        return None if out is None else out.split(",")

    def rows_to_keep_for_diagnostics(self) -> str:
        """Return the diagnostics row-filter expression.

        Args:
            None.

        Returns:
            The pandas expression string used to filter rows.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(
                Namespace(rows_to_keep_for_diagnostics="age >= 18")
                ).rows_to_keep_for_diagnostics()
                # 'age >= 18'
        """
        return self.args.rows_to_keep_for_diagnostics

    def has_batch_columns(self) -> bool:
        """Return True when batch columns are supplied.

        Args:
            None.

        Returns:
            ``True`` if batch columns are set, otherwise ``False``.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(batch_columns="region")).has_batch_columns()
                # True
        """
        return self.args.batch_columns is not None

    def batch_columns(self) -> str:
        """Return the list of batch column names.

        Args:
            None.

        Returns:
            Batch column names parsed from the CLI argument.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(batch_columns="region,team")).batch_columns()
                # ['region', 'team']
        """
        return self.args.batch_columns.split(",")

    def has_keep_columns(self) -> bool:
        """Return True when output keep columns are supplied.

        Args:
            None.

        Returns:
            ``True`` if keep columns are set, otherwise ``False``.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(keep_columns="id,weight")).has_keep_columns()
                # True
        """
        return self.args.keep_columns is not None

    def keep_columns(self) -> list[str] | None:
        """Return the subset of columns to keep in outputs.

        Args:
            None.

        Returns:
            List of columns to keep or ``None`` if unspecified.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(keep_columns="id,weight")).keep_columns()
                # ['id', 'weight']
        """
        if self.args.keep_columns:
            return self.args.keep_columns.split(",")
        return None

    def has_keep_row_column(self) -> bool:
        """Return True when a keep-row column is supplied.

        Args:
            None.

        Returns:
            ``True`` if a keep-row column is set, otherwise ``False``.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(keep_row_column="keep")).has_keep_row_column()
                # True
        """
        return self.args.keep_row_column is not None

    def keep_row_column(self) -> str:
        """Return the keep-row indicator column name.

        Args:
            None.

        Returns:
            Name of the keep-row indicator column.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(keep_row_column="keep")).keep_row_column()
                # 'keep'
        """
        return self.args.keep_row_column

    def has_outcome_columns(self) -> bool:
        """Return True when outcome columns are explicitly supplied.

        Args:
            None.

        Returns:
            ``True`` if outcome columns are set, otherwise ``False``.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(outcome_columns="y")).has_outcome_columns()
                # True
        """
        return self.args.outcome_columns is not None

    def outcome_columns(self) -> list[str] | None:
        """Return the list of outcome columns if provided.

        Args:
            None.

        Returns:
            List of outcome columns or ``None`` if unset.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(outcome_columns="y,z")).outcome_columns()
                # ['y', 'z']
        """
        if self.args.outcome_columns:
            return self.args.outcome_columns.split(",")
        return None

    def max_de(self) -> float | None:
        """Return the max design effect setting.

        Args:
            None.

        Returns:
            Maximum design effect or ``None`` if unset.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(max_de=1.5)).max_de()
                # 1.5
        """
        return self.args.max_de

    def lambda_min(self) -> float | None:
        """Return the minimum L1 penalty setting.

        Args:
            None.

        Returns:
            Minimum L1 penalty value or ``None``.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(lambda_min=1e-5)).lambda_min()
                # 1e-05
        """
        return self.args.lambda_min

    def lambda_max(self) -> float | None:
        """Return the maximum L1 penalty setting.

        Args:
            None.

        Returns:
            Maximum L1 penalty value or ``None``.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(lambda_max=10.0)).lambda_max()
                # 10.0
        """
        return self.args.lambda_max

    def num_lambdas(self) -> int | None:
        """Return the number of lambda values to search over.

        Args:
            None.

        Returns:
            Number of lambdas as an integer or ``None``.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(num_lambdas=250)).num_lambdas()
                # 250
        """
        if self.args.num_lambdas is None:
            return None
        return int(self.args.num_lambdas)

    def transformations(self) -> str | None:
        """Return the transformations config for adjustment.

        Args:
            None.

        Returns:
            Transformations setting or ``None`` if disabled.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(transformations="default")).transformations()
                # 'default'
        """
        if (self.args.transformations is None) or (self.args.transformations == "None"):
            return None
        else:
            return self.args.transformations

    def formula(self) -> str | None:
        """Return the formula string used for model matrices.

        Args:
            None.

        Returns:
            Formula string or ``None`` if unset.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(formula="age + gender")).formula()
                # 'age + gender'
        """
        return self.args.formula

    def one_hot_encoding(self) -> bool | None:
        """Return the parsed one-hot encoding flag.

        Args:
            None.

        Returns:
            ``True``/``False`` for one-hot encoding, or ``None`` if unset.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(one_hot_encoding="False")).one_hot_encoding()
                # False
        """
        return balance.util._true_false_str_to_bool(self.args.one_hot_encoding)

    def standardize_types(self) -> bool:
        """Return whether to standardize input types in Sample.from_frame.

        Args:
            None.

        Returns:
            ``True`` if standardization is enabled, otherwise ``False``.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(standardize_types="True")).standardize_types()
                # True
        """
        return balance.util._true_false_str_to_bool(self.args.standardize_types)

    def weight_trimming_mean_ratio(self) -> float:
        """Return the mean ratio used for trimming weights.

        Args:
            None.

        Returns:
            Weight trimming ratio.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                BalanceCLI(Namespace(weight_trimming_mean_ratio=20.0)).weight_trimming_mean_ratio()
                # 20.0
        """
        return self.args.weight_trimming_mean_ratio

    def logistic_regression_kwargs(self) -> Dict[str, Any] | None:
        """Parse JSON keyword arguments for the IPW logistic regression model.

        Args:
            None.

        Returns:
            Parsed keyword arguments dictionary or ``None``.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                cli = BalanceCLI(
                Namespace(ipw_logistic_regression_kwargs='{\"max_iter\": 100}')
                )
                cli.logistic_regression_kwargs()
                # {'max_iter': 100}
        """
        raw_kwargs = self.args.ipw_logistic_regression_kwargs
        if raw_kwargs is None:
            return None
        if isinstance(raw_kwargs, dict):
            return raw_kwargs
        try:
            parsed = json.loads(raw_kwargs)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "--ipw_logistic_regression_kwargs must be a JSON object string"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                "--ipw_logistic_regression_kwargs must decode to a JSON object"
            )
        return parsed

    def logistic_regression_model(self) -> ClassifierMixin | None:
        """Build a LogisticRegression model when IPW kwargs are supplied.

        Args:
            None.

        Returns:
            Configured LogisticRegression instance or ``None``.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                cli = BalanceCLI(
                Namespace(ipw_logistic_regression_kwargs='{\"max_iter\": 100}')
                )
                cli.logistic_regression_model().__class__.__name__
                # 'LogisticRegression'
        """
        kwargs = self.logistic_regression_kwargs()
        if kwargs is None:
            return None
        return LogisticRegression(**kwargs)

    def split_sample(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the input frame into sample and target partitions.

        Args:
            df: Input DataFrame containing sample and target rows.

        Returns:
            A tuple of (sample_df, target_df).

        Examples:
            .. code-block:: python
                import pandas as pd
                from argparse import Namespace
                cli = BalanceCLI(Namespace(sample_column="is_respondent"))
                frame = pd.DataFrame({"is_respondent": [1, 0], "x": [1, 2]})
                sample_df, target_df = cli.split_sample(frame)
                len(sample_df), len(target_df)
                # (1, 1)
        """
        in_sample = df[self.sample_column()] == 1
        sample_df = df[in_sample]
        target_df = df[~in_sample]

        return sample_df, target_df

    def process_batch(
        self,
        batch_df: pd.DataFrame,
        transformations: Dict[str, Any] | str | None = "default",
        formula: str | None = None,
        penalty_factor: None = None,
        one_hot_encoding: bool = False,
        max_de: float | None = 1.5,
        lambda_min: float | None = 1e-05,
        lambda_max: float | None = 10,
        num_lambdas: int | None = 250,
        weight_trimming_mean_ratio: float = 20,
        sample_cls: Type[balance_sample_cls] = balance_sample_cls,
        sample_package_name: str = __package__,
    ) -> Dict[str, pd.DataFrame]:
        """Run adjustment for a batch of data and return outputs.

        Args:
            batch_df: Input data for the current batch.
            transformations: Transformations argument for Sample.adjust.
            formula: Optional formula for model matrices.
            penalty_factor: Optional penalty factor passed to adjust.
            one_hot_encoding: Whether to one-hot encode categorical features.
            max_de: Maximum design effect constraint.
            lambda_min: Minimum penalty value for IPW.
            lambda_max: Maximum penalty value for IPW.
            num_lambdas: Number of penalty values to search.
            weight_trimming_mean_ratio: Mean ratio for trimming weights.
            sample_cls: Sample implementation used to build sample/target.
            sample_package_name: Name used in logging.

        Returns:
            Dict with adjusted data and diagnostics frames.

        Examples:
            .. code-block:: python
                import pandas as pd
                from argparse import Namespace

                cli = BalanceCLI(
                    Namespace(
                        method="ipw",
                        sample_column="is_respondent",
                        id_column="id",
                        weight_column="weight",
                        covariate_columns="x",
                        covariate_columns_for_diagnostics=None,
                        rows_to_keep_for_diagnostics=None,
                        batch_columns=None,
                        keep_columns=None,
                        keep_row_column=None,
                        outcome_columns=None,
                        max_de=1.5,
                        lambda_min=1e-5,
                        lambda_max=10.0,
                        num_lambdas=250,
                        transformations="default",
                        formula=None,
                        one_hot_encoding="True",
                        standardize_types="True",
                        weight_trimming_mean_ratio=20.0,
                        ipw_logistic_regression_kwargs=None,
                        succeed_on_weighting_failure=True,
                        return_df_with_original_dtypes=False,
                    )
                )
                frame = pd.DataFrame(
                    {
                        "is_respondent": [1, 0],
                        "id": [1, 2],
                        "weight": [1.0, 1.0],
                        "x": [0.1, 0.2],
                    }
                )
                result = cli.process_batch(frame)
                set(result.keys()) == {"adjusted", "diagnostics"}
                # True
        """
        # TODO: add unit tests
        sample_df, target_df = self.split_sample(batch_df)

        if sample_df.shape[0] == 0:
            return {
                "adjusted": pd.DataFrame(),
                "diagnostics": pd.DataFrame(
                    {
                        "metric": ("adjustment_failure", "adjustment_failure_reason"),
                        "var": (None, None),
                        "val": (1, "No input data"),
                    }
                ),
            }

        # Stuff everything that is not id, weight, or covariate into outcomes
        outcome_columns = self.outcome_columns()
        ignore_columns = None
        if outcome_columns is None:
            outcome_columns = [
                column
                for column in batch_df.columns
                if column
                not in {
                    self.id_column(),
                    self.weight_column(),
                    *self.covariate_columns(),
                }
            ]
        else:
            ignore_columns = [
                column
                for column in batch_df.columns
                if column
                not in {
                    self.id_column(),
                    self.weight_column(),
                    *self.covariate_columns(),
                    *outcome_columns,
                }
            ]
        outcome_columns = tuple(outcome_columns)

        # definitions for diagnostics
        covariate_columns_for_diagnostics = self.covariate_columns_for_diagnostics()
        rows_to_keep_for_diagnostics = self.rows_to_keep_for_diagnostics()

        sample = sample_cls.from_frame(
            sample_df,
            id_column=self.id_column(),
            weight_column=self.weight_column(),
            outcome_columns=outcome_columns,
            ignore_columns=ignore_columns,
            check_id_uniqueness=False,
            standardize_types=self.standardize_types(),
        )
        logger.info("%s sample object: %s" % (sample_package_name, str(sample)))

        target = sample_cls.from_frame(
            target_df,
            id_column=self.id_column(),
            weight_column=self.weight_column(),
            outcome_columns=outcome_columns,
            ignore_columns=ignore_columns,
            check_id_uniqueness=False,
            standardize_types=self.standardize_types(),
        )
        logger.info("%s target object: %s" % (sample_package_name, str(target)))

        try:
            method = self.method()
            model = self.logistic_regression_model() if method == "ipw" else None

            adjusted_kwargs: Dict[str, Any] = {
                "method": method,
                "transformations": transformations,
                "formula": formula,
                "penalty_factor": penalty_factor,
                "one_hot_encoding": one_hot_encoding,
                "max_de": max_de,
                "lambda_min": lambda_min,
                "lambda_max": lambda_max,
                "num_lambdas": num_lambdas,
                "weight_trimming_mean_ratio": weight_trimming_mean_ratio,
            }

            if model is not None:
                adjusted_kwargs["model"] = model

            adjusted = sample.set_target(target).adjust(**adjusted_kwargs)
            logger.info("Succeeded with adjusting sample to target")
            logger.info("%s adjusted object: %s" % (sample_package_name, str(adjusted)))

            logger.info(
                "Condition on which rows to keep for diagnostics: %s "
                % rows_to_keep_for_diagnostics
            )
            logger.info(
                "Names of columns to keep for diagnostics: %s "
                % covariate_columns_for_diagnostics
            )

            diagnostics = adjusted.keep_only_some_rows_columns(
                rows_to_keep=rows_to_keep_for_diagnostics,
                columns_to_keep=covariate_columns_for_diagnostics,
            ).diagnostics()
            logger.info(
                "%s diagnostics object: %s" % (sample_package_name, str(diagnostics))
            )

            # Update dtypes
            if self.args.return_df_with_original_dtypes:
                df_to_return = balance.util._astype_in_df_from_dtypes(
                    adjusted.df, adjusted._df_dtypes
                )
                balance.util._warn_of_df_dtypes_change(
                    adjusted.df.dtypes,
                    df_to_return.dtypes,
                    "df_after_adjust",
                    "df_returned_by_the_cli",
                )

            else:
                df_to_return = adjusted.df
            rval = {"adjusted": df_to_return, "diagnostics": diagnostics}
        except Exception as e:
            if self.args.succeed_on_weighting_failure:
                logger.error(
                    "Adjustment failed. Because '--succeed_on_weighting_failure' was set: returning empty weights."
                )
                sample.set_weights(None)
                module = inspect.getmodule(inspect.trace()[-1][0])
                module_name = module.__name__ if module is not None else None
                error_message = f"{module_name}: {e}"
                logger.exception("The error message is: " + error_message)
                # Update dtypes
                if self.args.return_df_with_original_dtypes:
                    df_to_return = balance.util._astype_in_df_from_dtypes(
                        sample.df, sample._df_dtypes
                    )
                    balance.util._warn_of_df_dtypes_change(
                        sample.df.dtypes,
                        df_to_return.dtypes,
                        "df_without_adjusting",
                        "df_returned_by_the_cli",
                    )
                else:
                    df_to_return = sample.df
                rval = {
                    "adjusted": df_to_return,
                    "diagnostics": pd.DataFrame(
                        {
                            "metric": (
                                "adjustment_failure",
                                "adjustment_failure_reason",
                            ),
                            "var": (None, None),
                            "val": (1, error_message),
                        }
                    ),
                }
            else:
                raise e
        return rval

    def adapt_output(self, output_df: pd.DataFrame) -> pd.DataFrame:
        """Filter raw output dataframe to user's requested rows/columns.

        - First we filter to the rows we are supposed to keep.
        - Next we select the columns that need to be returned.

        Args:
            output_df: DataFrame produced by the adjustment step.

        Returns:
            Filtered DataFrame containing requested rows and columns.

        Examples:
            .. code-block:: python
                import pandas as pd
                from argparse import Namespace

                cli = BalanceCLI(
                    Namespace(
                        keep_row_column="keep",
                        keep_columns="id,weight",
                    )
                )
                frame = pd.DataFrame(
                    {"id": [1, 2], "weight": [1.0, 2.0], "keep": [1, 0]}
                )
                cli.adapt_output(frame).to_dict(orient="list")
                # {'id': [1], 'weight': [1.0]}
        """
        if output_df.empty:
            return output_df

        if self.has_keep_row_column():
            keep_rows = output_df[self.keep_row_column()] == 1
            output_df = output_df[keep_rows]

        if self.has_keep_columns():
            output_df = output_df[self.keep_columns()]

        return output_df

    def load_and_check_input(self) -> pd.DataFrame:
        """Read the input file and log basic information.

        Args:
            None.

        Returns:
            DataFrame loaded from the input file.

        Examples:
            .. code-block:: python
                import pandas as pd
                import tempfile
                from argparse import Namespace
                with tempfile.NamedTemporaryFile(suffix=".csv") as tmp_file:
                    df = pd.DataFrame({"x": [1], "y": [2]})
                    df.to_csv(tmp_file.name, index=False)
                    cli = BalanceCLI(
                        Namespace(
                            input_file=tmp_file.name,
                            sep_input_file=",",
                            keep_row_column=None,
                        )
                    )
                    loaded = cli.load_and_check_input()
                    loaded.shape
                    # (1, 2)
        """
        # TODO: Add unit tests for function
        # Load and check input
        input_df = pd.read_csv(self.args.input_file, sep=self.args.sep_input_file)
        logger.info("Number of rows in input file: %d" % input_df.shape[0])
        if self.has_keep_row_column():
            logger.info(
                "Number of rows to keep in input file: %d"
                % input_df[input_df[self.keep_row_column()] == 1].shape[0]
            )
        logger.info("Number of columns in input file: %d" % input_df.shape[1])
        return input_df

    def write_outputs(
        self, output_df: pd.DataFrame, diagnostics_df: pd.DataFrame
    ) -> None:
        """Write adjusted output and diagnostics CSV files.

        Args:
            output_df: Adjusted output DataFrame to write.
            diagnostics_df: Diagnostics DataFrame to write.

        Returns:
            None.

        Examples:
            .. code-block:: python
                import pandas as pd
                from argparse import Namespace
                from pathlib import Path
                output_df = pd.DataFrame({"id": [1], "weight": [1.0]})
                diagnostics_df = pd.DataFrame({"metric": ["size"], "var": ["sample"], "val": [1]})
                cli = BalanceCLI(
                    Namespace(
                        output_file=Path("tmp_cli_out.csv"),
                        diagnostics_output_file=Path("tmp_cli_diag.csv"),
                        no_output_header=False,
                        sep_output_file=",",
                        sep_diagnostics_output_file=",",
                    )
                )
                cli.write_outputs(output_df, diagnostics_df)
        """
        # TODO: Add unit tests for function
        # Write output
        output_df.to_csv(
            path_or_buf=self.args.output_file,
            index=False,
            header=(not self.args.no_output_header),
            sep=self.args.sep_output_file,
        )

        if self.args.diagnostics_output_file is not None:
            diagnostics_df.to_csv(
                path_or_buf=self.args.diagnostics_output_file,
                index=False,
                header=(not self.args.no_output_header),
                sep=self.args.sep_diagnostics_output_file,
            )

    def update_attributes_for_main_used_by_adjust(self) -> None:
        """Prepare cached attributes for main to use in adjustment.

        Args:
            None.

        Returns:
            None.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                cli = BalanceCLI(
                    Namespace(
                        method="ipw",
                        transformations="default",
                        formula=None,
                        lambda_min=1e-5,
                        lambda_max=10.0,
                        num_lambdas=250,
                        one_hot_encoding="True",
                        max_de=1.5,
                        weight_trimming_mean_ratio=20.0,
                    )
                )
                cli.update_attributes_for_main_used_by_adjust()
        """
        # TODO: future version might include conditional control over these attributes based on some input
        transformations = self.transformations()
        formula = self.formula()
        penalty_factor = None
        lambda_min = self.lambda_min()
        lambda_max = self.lambda_max()
        num_lambdas = self.num_lambdas()
        one_hot_encoding_result = self.one_hot_encoding()
        one_hot_encoding = (
            one_hot_encoding_result if one_hot_encoding_result is not None else False
        )
        max_de = self.max_de()
        weight_trimming_mean_ratio = self.weight_trimming_mean_ratio()
        sample_cls, sample_package_name, sample_package_version = (
            balance_sample_cls,
            __package__,
            __version__,
        )

        # update all attributes (to be later used in main)
        (
            self._transformations,
            self._formula,
            self._penalty_factor,
            self._one_hot_encoding,
            self._max_de,
            self._lambda_min,
            self._lambda_max,
            self._num_lambdas,
            self._weight_trimming_mean_ratio,
            self._sample_cls,
            self._sample_package_name,
            self._sample_package_version,
        ) = (
            transformations,
            formula,
            penalty_factor,
            one_hot_encoding,
            max_de,
            lambda_min,
            lambda_max,
            num_lambdas,
            weight_trimming_mean_ratio,
            sample_cls,
            sample_package_name,
            sample_package_version,
        )

    def main(self) -> None:
        """Run the CLI workflow from input loading to output writing.

        Args:
            None.

        Returns:
            None.

        Examples:
            .. code-block:: python
                from argparse import Namespace
                cli = BalanceCLI(Namespace(method="ipw"))
                callable(cli.main)
                # True
        """
        # update all the objects from self attributes
        (
            transformations,
            formula,
            penalty_factor,
            one_hot_encoding,
            max_de,
            lambda_min,
            lambda_max,
            num_lambdas,
            weight_trimming_mean_ratio,
            sample_cls,
            sample_package_name,
            sample_package_version,
        ) = (
            self._transformations,
            self._formula,
            self._penalty_factor,
            self._one_hot_encoding,
            self._max_de,
            self._lambda_min,
            self._lambda_max,
            self._num_lambdas,
            self._weight_trimming_mean_ratio,
            self._sample_cls,
            self._sample_package_name,
            self._sample_package_version,
        )

        logger.info(
            "Running cli.main() using %s version %s"
            % (sample_package_name, sample_package_version)
        )

        # Logging arguments used by main:
        keys = (
            "transformations",
            "formula",
            "penalty_factor",
            "one_hot_encoding",
            "max_de",
            "lambda_min",
            "lambda_max",
            "num_lambdas",
            "weight_trimming_mean_ratio",
            "sample_cls",
            "sample_package_name",
            "sample_package_version",
        )
        values = (
            transformations,
            formula,
            penalty_factor,
            one_hot_encoding,
            max_de,
            lambda_min,
            lambda_max,
            num_lambdas,
            weight_trimming_mean_ratio,
            sample_cls,
            sample_package_name,
            sample_package_version,
        )
        main_config = dict(zip(keys, values))
        logger.info("Attributes used by main() for running adjust: %s" % main_config)

        # Load and check input
        input_df = self.load_and_check_input()
        self.check_input_columns(input_df.columns)

        # Run process_batch on input_df, and adjustment arguments
        if self.has_batch_columns():
            results = []
            diagnostics = []
            for batch_name, batch_df in input_df.groupby(self.batch_columns()):
                logger.info("Running weighting for batch = %s " % str(batch_name))
                processed = self.process_batch(
                    batch_df,
                    transformations,
                    formula,
                    penalty_factor,
                    one_hot_encoding,
                    max_de,
                    lambda_min,
                    lambda_max,
                    num_lambdas,
                    weight_trimming_mean_ratio,
                    sample_cls,
                    sample_package_name,
                )
                results.append(processed["adjusted"])
                diagnostics.append(processed["diagnostics"])
                logger.info("Done processing batch %s" % str(batch_name))

            if (len(results) == 0) and len(diagnostics) == 0:
                output_df = pd.DataFrame()
                diagnostics_df = pd.DataFrame()
            else:
                output_df = pd.concat(results)
                diagnostics_df = pd.concat(diagnostics)
        else:
            processed = self.process_batch(
                input_df,
                transformations,
                formula,
                penalty_factor,
                one_hot_encoding,
                max_de,
                lambda_min,
                lambda_max,
                num_lambdas,
                weight_trimming_mean_ratio,
                sample_cls,
                sample_package_name,
            )
            output_df = processed["adjusted"]
            diagnostics_df = processed["diagnostics"]

        logger.info("Done fitting the model, writing output")
        # Remove unneeded rows and columns
        output_df = self.adapt_output(output_df)

        # Write output
        self.write_outputs(output_df, diagnostics_df)


def add_arguments_to_parser(parser: ArgumentParser) -> ArgumentParser:
    """Register CLI arguments on an argparse parser.

    Args:
        parser: Parser to add arguments to.

    Returns:
        The parser instance with CLI arguments registered.

    Examples:
        .. code-block:: python
            from argparse import ArgumentParser
            parser = add_arguments_to_parser(ArgumentParser())
            isinstance(parser, ArgumentParser)
            # True
    """
    # TODO: add checks for validity of input (including None as input)
    # TODO: add arguments for formula when used as a list and for penalty_factor
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Path to input sample/target",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to write output weights",
    )
    parser.add_argument(
        "--diagnostics_output_file",
        type=Path,
        required=False,
        help="Path to write adjustment diagnostics",
    )
    parser.add_argument(
        "--method", default="ipw", help="Method to use for weighting [default=ipw]"
    )
    parser.add_argument(
        "--sample_column",
        default="is_respondent",
        help="Path to target population [default=is_respondent]",
    )
    parser.add_argument(
        "--id_column",
        default="id",
        help="Column that identifies units [default=id]",
    )
    parser.add_argument(
        "--weight_column",
        default="weight",
        help="Column that identifies weights of samples [default=weight]",
    )
    parser.add_argument(
        "--covariate_columns", required=True, help="Set of columns used for adjustment"
    )
    parser.add_argument(
        "--outcome_columns",
        required=False,
        default=None,
        help=(
            "Set of columns used as outcomes. If not supplied, all columns that are "
            "not in id, weight, or covariate columns are treated as outcomes."
        ),
    )
    parser.add_argument(
        "--covariate_columns_for_diagnostics",
        required=False,
        default=None,
        help="Set of columns used for diagnostics reporting (if not supplied the default is None, which means to use all columns from --covariate_columns)",
    )
    parser.add_argument(
        "--rows_to_keep_for_diagnostics",
        required=False,
        default=None,
        help="A string with a condition for filtering rows (to be used for diagnostics reporting). \
             The condition should be based on the list of covariate_columns and result in a bool pd.Series \
             (e.g.: 'is_married' or 'is_married & age >= 18', etc.) \
             (if not supplied the default is None, which means to use all rows without filtering",
    )
    parser.add_argument(
        "--batch_columns",
        required=False,
        help="Set of columns used to indicate batches of data",
    )
    parser.add_argument(
        "--keep_columns",
        type=str,
        required=False,
        help="Set of columns we include in the output csv file",
    )
    parser.add_argument(
        "--keep_row_column",
        type=str,
        required=False,
        help="Column indicating which rows we include in the output csv file",
    )
    parser.add_argument(
        "--sep_input_file",
        type=str,
        required=False,
        default=",",
        help="A 1 character for indicating the delimiter for the output file. If not supplied it defaults to a comma (,)",
    )
    parser.add_argument(
        "--sep_output_file",
        type=str,
        required=False,
        default=",",
        help="A 1 character for indicating the delimiter for the output file. If not supplied it defaults to a comma (,)",
    )
    parser.add_argument(
        "--sep_diagnostics_output_file",
        type=str,
        required=False,
        default=",",
        help="A 1 character for indicating the delimiter for the diagnostics output file. If not supplied it defaults to a comma (,)",
    )
    parser.add_argument(
        "--no_output_header",
        default=False,
        action="store_true",
        help="Turn off header in the output csv file",
    )
    # TODO: Identify conditions for weighting failure or remove this argument entirely
    parser.add_argument(
        "--succeed_on_weighting_failure",
        action="store_true",
        help=(
            "If adjustment fails (e.g. because the input data has too few "
            "rows), then do not fail, but instead return the input data with "
            "all weights null"
        ),
    )
    parser.add_argument(
        "--max_de",
        type=_float_or_none,
        required=False,
        default=1.5,
        help=(
            "Upper bound for the design effect of the computed weights. "
            "If not supplied it defaults to 1.5. If set to 'None', then it uses 'lambda_1se'. "
            "Only used if method is ipw or CBPS."
        ),
    )
    parser.add_argument(
        "--lambda_min",
        type=float,
        required=False,
        default=1e-05,
        help=(
            "Lower bound (least penalized) for the L1 penalty range searched over in ipw."
            "Only used if method is ipw."
            "If not supplied it defaults to 1e-05."
        ),
    )
    parser.add_argument(
        "--lambda_max",
        type=float,
        required=False,
        default=10,
        help=(
            "Upper bound (most penalized) for the L1 penalty range searched over in ipw."
            "Only used if method is ipw."
            "If not supplied it defaults to 10."
        ),
    )
    parser.add_argument(
        "--num_lambdas",
        type=float,
        required=False,
        default=250,
        help=(
            "Number of elements searched over in the L1 penalty range in ipw."
            "Only used if method is ipw."
            "If not supplied it defaults to 250."
        ),
    )
    parser.add_argument(
        "--ipw_logistic_regression_kwargs",
        required=False,
        help=(
            "A valid JSON object string of keyword arguments forwarded to sklearn.linear_model.LogisticRegression "
            'when using the ipw method. For example: \'{"solver": "liblinear", "max_iter": 500}\'. '
            "Ignored for other methods."
        ),
    )
    parser.add_argument(
        "--weight_trimming_mean_ratio",
        type=_float_or_none,
        required=False,
        default=20.0,
        help=(
            "The ratio according to which weights are trimmed from above by mean(weights) * ratio. "
            "Defaults to 20. "
            "Used only for CBPS and ipw."
            "For ipw this is only used if max_de is set to 'None',otherwise, the trimming ratio is chosen by the algorithm."
        ),
    )
    parser.add_argument(
        "--one_hot_encoding",
        type=str,
        default="True",
        required=False,
        help=(
            "Set the value of the one_hot_encoding parameter. Accepts a string with one a value of 'True' or 'False' (treats it as a bool). Default is 'True'"
        ),
    )
    # TODO: Ideally we would like transformations argument to be able to get three types of values: None (for no transformations),
    # "default" for default transformations or a dictionary of transformations.
    # However, as a first step I added the option for "default" (which is also the default) and None (for no transformations).
    parser.add_argument(
        "--transformations",
        default="default",
        required=False,
        help=(
            "Define the transformations for the covariates. Can be set to None for no transformations or"
            "'default' for default transformations."
        ),
    )
    # TODO: we currently support only the option of a string formula (or None), not a list of formulas.
    parser.add_argument(
        "--formula",
        default=None,
        required=False,
        help=(
            "The formula of the model matrix (in ipw or cbps). If None (default), the formula will be setted to an additive formula using all the covariates."
        ),
    )
    parser.add_argument(
        "--return_df_with_original_dtypes",
        action="store_true",
        help=(
            "If the input table has unsupported column types (e.g.: int32), then when using Sample.from_frame it will be changed (e.g.: float32). "
            "The returned df from the cli will use the transformed DataFrame. If this flag is used, then the cli will attempt to restore the original dtypes of the input table"
            "before returning it."
            "WARNING: sometimes the pd.astype command might lead to odd behaviors, so it is generally safer to NOT use this flag but to manually specify the desired dtypes in the input/output tables."
            "For example, dealing with missing values could lead to many issues (e.g.: there is np.nan and pd.NA, and these do not play nicely with type conversions)"
        ),
    )
    parser.add_argument(
        "--standardize_types",
        type=str,
        default="True",
        required=False,
        help=(
            "Control the standardize_types argument in Sample.from_frame (which is used on the files read by the cli)"
            "The default is True. It is generally not advised to use False since this step is needed to deal with converting input types that are not supported by various functions."
            "For example, if a column of Int64 has pandas.NA, it could fail on various functions. Current default (True) will turn that column into float64"
            "(the pandas.NA will be converted into numpy.nan)."
            "Setting the current flag to 'False' might lead to failures. The advantage is that it would keep most columns dtype as is"
            " (which might be helpful for some downstream operations that assume the output dtypes are the same as the input dtypes)."
            "NOTE: regardless if this flag is set to true or false, the weight column will be turned into a float64 type anyway."
        ),
    )

    return parser


def make_parser() -> ArgumentParser:
    """Create and return the CLI argument parser.

    Args:
        None.

    Returns:
        A configured ``ArgumentParser`` for the balance CLI.

    Examples:
        .. code-block:: python
            parser = make_parser()
            isinstance(parser, ArgumentParser)
            # True
    """
    parser = ArgumentParser()
    parser = add_arguments_to_parser(parser)
    return parser


def main() -> None:
    """Entry point for the balance CLI.

    Args:
        None.

    Returns:
        None.

    Examples:
        .. code-block:: python
            callable(main)
            # True
    """
    parser: ArgumentParser = make_parser()
    args: Namespace = parser.parse_args()
    cli = BalanceCLI(args)
    cli.update_attributes_for_main_used_by_adjust()
    cli.main()


if __name__ == "__main__":
    main()
