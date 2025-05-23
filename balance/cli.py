# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import absolute_import, division, print_function, unicode_literals

import inspect
import logging

from argparse import ArgumentParser, FileType, Namespace

from typing import Dict, List, Optional, Tuple, Type, Union

import balance

import pandas as pd

from balance import __version__  # @manual
from balance.sample_class import Sample as balance_sample_cls  # @manual

logger: logging.Logger = logging.getLogger(__package__)


class BalanceCLI:
    def __init__(self, args) -> None:
        self.args = args

        # Create attributes (to be populated later, which will be used in main)
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
        ) = (None, None, None, None, None, None, None, None, None, None, None, None)

    def check_input_columns(self, columns: Union[List[str], pd.Index]) -> None:
        needed_columns = []
        needed_columns.append(self.sample_column())
        needed_columns.append(self.id_column())
        needed_columns.append(self.weight_column())
        needed_columns.extend(self.covariate_columns())
        if self.has_batch_columns():
            needed_columns.extend(self.batch_columns())
        if self.has_keep_columns():
            needed_columns.extend(self.keep_columns())
        if self.has_keep_row_column():
            needed_columns.append(self.keep_row_column())

        for nc in needed_columns:
            assert nc in columns, f"{nc} not in input colums"

    # TODO: decide if to explicitly mention/check here the option of methods or not
    def method(self) -> str:
        return self.args.method

    def sample_column(self) -> str:
        return self.args.sample_column

    def id_column(self) -> str:
        return self.args.id_column

    def weight_column(self) -> str:
        return self.args.weight_column

    def covariate_columns(self) -> str:
        return self.args.covariate_columns.split(",")

    def covariate_columns_for_diagnostics(self) -> List[str]:
        out = self.args.covariate_columns_for_diagnostics
        return None if out is None else out.split(",")

    def rows_to_keep_for_diagnostics(self) -> str:
        return self.args.rows_to_keep_for_diagnostics

    def has_batch_columns(self) -> bool:
        return self.args.batch_columns is not None

    def batch_columns(self) -> str:
        return self.args.batch_columns.split(",")

    def has_keep_columns(self) -> bool:
        return self.args.keep_columns is not None

    def keep_columns(self):  # TODO: figure out how to type hint this one.
        if self.args.keep_columns:
            return self.args.keep_columns.split(",")

    def has_keep_row_column(self) -> bool:
        return self.args.keep_row_column is not None

    def keep_row_column(self) -> str:
        return self.args.keep_row_column

    def max_de(self) -> Optional[float]:
        return self.args.max_de

    def lambda_min(self) -> Optional[float]:
        return self.args.lambda_min

    def lambda_max(self) -> Optional[float]:
        return self.args.lambda_max

    def num_lambdas(self) -> Optional[float]:
        return self.args.num_lambdas

    def transformations(self) -> Optional[str]:
        if (self.args.transformations is None) or (self.args.transformations == "None"):
            return None
        else:
            return self.args.transformations

    def formula(self) -> Optional[str]:
        return self.args.formula

    def one_hot_encoding(self) -> Optional[bool]:
        return balance.util._true_false_str_to_bool(self.args.one_hot_encoding)

    def standardize_types(self) -> bool:
        return balance.util._true_false_str_to_bool(self.args.standardize_types)

    def weight_trimming_mean_ratio(self) -> float:
        return self.args.weight_trimming_mean_ratio

    def split_sample(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        in_sample = df[self.sample_column()] == 1
        sample_df = df[in_sample]
        target_df = df[~in_sample]

        return sample_df, target_df

    def process_batch(
        self,
        batch_df: pd.DataFrame,
        transformations: Union[Dict, str, None] = "default",
        formula=None,
        penalty_factor=None,
        one_hot_encoding: bool = False,
        max_de: Optional[float] = 1.5,
        lambda_min: Optional[float] = 1e-05,
        lambda_max: Optional[float] = 10,
        num_lambdas: Optional[int] = 250,
        weight_trimming_mean_ratio: float = 20,
        sample_cls: Type[balance_sample_cls] = balance_sample_cls,
        sample_package_name: str = __package__,
    ) -> Dict[str, pd.DataFrame]:
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
        outcome_columns = tuple(
            set(batch_df.columns)
            - {self.id_column(), self.weight_column()}
            - set(self.covariate_columns())
        )

        # definitions for diagnostics
        covariate_columns_for_diagnostics = self.covariate_columns_for_diagnostics()
        rows_to_keep_for_diagnostics = self.rows_to_keep_for_diagnostics()

        sample = sample_cls.from_frame(
            sample_df,
            id_column=self.id_column(),
            weight_column=self.weight_column(),
            outcome_columns=outcome_columns,
            check_id_uniqueness=False,
            standardize_types=self.standardize_types(),
        )
        logger.info("%s sample object: %s" % (sample_package_name, str(sample)))

        target = sample_cls.from_frame(
            target_df,
            id_column=self.id_column(),
            weight_column=self.weight_column(),
            outcome_columns=outcome_columns,
            check_id_uniqueness=False,
            standardize_types=self.standardize_types(),
        )
        logger.info("%s target object: %s" % (sample_package_name, str(target)))

        try:
            adjusted = sample.set_target(
                target
            ).adjust(
                method=self.method(),  # pyre-ignore[6] it gets str, but the function will verify internally if it's the str it should be.
                transformations=transformations,
                formula=formula,
                penalty_factor=penalty_factor,
                one_hot_encoding=one_hot_encoding,
                max_de=max_de,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                num_lambdas=num_lambdas,
                weight_trimming_mean_ratio=weight_trimming_mean_ratio,
            )
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

    def write_outputs(self, output_df, diagnostics_df) -> None:
        # TODO: Add unit tests for function
        # Write output
        output_df.to_csv(
            path_or_buf=self.args.output_file,
            index=False,
            header=(not self.args.no_output_header),
            sep=self.args.sep_output_file,
        )
        self.args.output_file.close()

        if self.args.diagnostics_output_file is not None:
            diagnostics_df.to_csv(
                path_or_buf=self.args.diagnostics_output_file,
                index=False,
                header=(not self.args.no_output_header),
                sep=self.args.sep_diagnostics_output_file,
            )
            self.args.diagnostics_output_file.close()

    def update_attributes_for_main_used_by_adjust(self) -> None:
        """
        Prepares all the defaults for main to use.
        """
        # TODO: future version might include conditional control over these attributes based on some input
        transformations = self.transformations()
        formula = self.formula()
        penalty_factor = None
        lambda_min = self.lambda_min()
        lambda_max = self.lambda_max()
        num_lambdas = self.num_lambdas()
        one_hot_encoding = self.one_hot_encoding()
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

    def __del__(self) -> None:
        for handle in [self.args.input_file, self.args.output_file]:
            try:
                handle.close()
            except FileNotFoundError:
                pass


def _float_or_none(value: Union[float, int, str, None]) -> Optional[float]:
    """Return a float (if float or int) or None if it's None or "None"

    This is so as to be clear that some input returned type is float or None.

    Args:
        value (Union[float, int, str, None]): A value to be converted.

    Returns:
        Optional[float]: None or float.
    """
    if (value is None) or (value == "None"):
        return None
    return float(value)


def add_arguments_to_parser(parser: ArgumentParser) -> ArgumentParser:
    # TODO: add checks for validity of input (including None as input)
    # TODO: add arguments for formula when used as a list and for penalty_factor
    parser.add_argument(
        "--input_file",
        type=FileType("r"),
        required=True,
        help="Path to input sample/target",
    )
    parser.add_argument(
        "--output_file",
        type=FileType("w"),
        required=True,
        help="Path to write output weights",
    )
    parser.add_argument(
        "--diagnostics_output_file",
        type=FileType("w"),
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
    (
        parser.add_argument(
            "--transformations",
            default="default",
            required=False,
            help=(
                "Define the transformations for the covariates. Can be set to None for no transformations or"
                "'default' for default transformations."
            ),
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
    parser = ArgumentParser()
    parser = add_arguments_to_parser(parser)
    return parser


def main() -> None:
    parser: ArgumentParser = make_parser()
    args: Namespace = parser.parse_args()
    cli = BalanceCLI(args)
    cli.update_attributes_for_main_used_by_adjust()
    cli.main()


if __name__ == "__main__":
    main()
