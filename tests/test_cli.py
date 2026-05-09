# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import os.path
import tempfile
import warnings
from argparse import ArgumentTypeError, Namespace
from typing import Any

import balance.testutil
import numpy as np
import pandas as pd
from balance.cli import BalanceCLI, make_parser
from balance.util import _assert_type, _float_or_none
from numpy import dtype
from sklearn.linear_model import LogisticRegression

# Test constants — small sizes for fast tests (CLI tests validate argument
# parsing, file I/O, and column handling, not numerical accuracy of adjustment)
SAMPLE_SIZE_SMALL = 100
SAMPLE_SIZE_LARGE = 200
TEST_SEED = 2021


def check_some_flags(
    flag: bool = True, the_flag_str: str = "--skip_standardize_types"
) -> dict[str, pd.DataFrame]:
    """
    Helper function to test CLI flags with standardized input data.

    Args:
        flag: Whether to include the specified flag in CLI arguments
        the_flag_str: The CLI flag string to test

    Returns:
        Dict containing input and output pandas DataFrames for comparison
    """
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as in_file,
    ):
        in_contents = (
            "x,y,is_respondent,id,weight\n"
            + ("1.0,50,1,1,1\n" * SAMPLE_SIZE_SMALL)
            + ("2.0,60,0,1,1\n" * SAMPLE_SIZE_SMALL)
        )
        in_file.write(in_contents)
        in_file.close()
        out_file = os.path.join(temp_dir, "out.csv")

        parser = make_parser()

        args_to_parse = [
            "--input_file",
            in_file.name,
            "--output_file",
            out_file,
            "--covariate_columns",
            "x,y",
            "--num_lambdas=1",
        ]
        if flag:
            args_to_parse.append(the_flag_str)
        args = parser.parse_args(args_to_parse)
        cli = BalanceCLI(args)
        cli.update_attributes_for_main_used_by_adjust()
        cli.main()

        pd_in = pd.read_csv(in_file.name)
        pd_out = pd.read_csv(out_file)
        return {"pd_in": pd_in, "pd_out": pd_out}


def _create_sample_and_target_data() -> pd.DataFrame:
    """
    Helper function to create standardized sample and target datasets for testing.

    Returns:
        pd.DataFrame: Combined dataset with sample and target data, including
                     age, gender, id, weight, and is_respondent columns.
    """
    # pyrefly: ignore [bad-argument-type]
    np.random.seed(TEST_SEED)
    n_sample = SAMPLE_SIZE_SMALL
    n_target = SAMPLE_SIZE_LARGE
    sample_df = pd.DataFrame(
        {
            "age": np.random.uniform(0, 100, n_sample),
            "gender": np.random.choice((1, 2, 3, 4), n_sample),
            "id": range(n_sample),
            "weight": pd.Series((1,) * n_sample),
        }
    )
    sample_df["is_respondent"] = True
    target_df = pd.DataFrame(
        {
            "age": np.random.uniform(0, 100, n_target),
            "gender": np.random.choice((1, 2, 3, 4), n_target),
            "id": range(n_target),
            "weight": pd.Series((1,) * n_target),
        }
    )
    target_df["is_respondent"] = False
    input_dataset = pd.concat([sample_df, target_df])
    return input_dataset


class TestCli(
    balance.testutil.BalanceTestCase,
):
    def _make_cli(self, **overrides: str | None) -> BalanceCLI:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "input.csv")
            output_file = os.path.join(temp_dir, "output.csv")
            parser = make_parser()
            args_to_parse = [
                "--input_file",
                input_file,
                "--output_file",
                output_file,
                "--covariate_columns",
                "covar_a,covar_b",
            ]
            if overrides.get("outcome_columns") is not None:
                args_to_parse.extend(
                    ["--outcome_columns", str(overrides["outcome_columns"])]
                )
            args = parser.parse_args(args_to_parse)
        return BalanceCLI(args)

    def _make_batch_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "is_respondent": [1, 0],
                "id": [1, 2],
                "weight": [1.0, 1.0],
                "covar_a": [1.0, 2.0],
                "covar_b": [3.0, 4.0],
                "outcome_b": [10.0, 20.0],
                "outcome_a": [30.0, 40.0],
                "extra": [5.0, 6.0],
            }
        )

    def _recording_sample_cls(self) -> type[Any]:
        class RecordingSample:
            calls: list[tuple[str, ...] | None] = []
            ignore_calls: list[list[str] | None] = []

            def __init__(self, df: pd.DataFrame) -> None:
                self.df = df
                self._df_dtypes = df.dtypes

            @classmethod
            def from_frame(
                cls,
                df: pd.DataFrame,
                id_column: str | None = None,
                weight_column: str | None = None,
                outcome_columns: tuple[str, ...] | None = None,
                ignored_columns: list[str] | None = None,
                **kwargs: object,
            ) -> "RecordingSample":
                cls.calls.append(outcome_columns)
                cls.ignore_calls.append(ignored_columns)
                return cls(df)

            def set_target(self, target: "RecordingSample") -> "RecordingSample":
                return self

            def adjust(self, **kwargs: object) -> "RecordingSample":
                return self

            def keep_only_some_rows_columns(
                self,
                rows_to_keep: str | None = None,
                columns_to_keep: list[str] | None = None,
            ) -> "RecordingSample":
                return self

            def diagnostics(
                self, weights_impact_on_outcome_method: str | None = "t_test"
            ) -> pd.DataFrame:
                return pd.DataFrame()

        return RecordingSample

    def test_cli_unmentioned_columns_go_to_ignore(self) -> None:
        RecordingSample = self._recording_sample_cls()
        cli = self._make_cli()
        cli.process_batch(
            self._make_batch_df(),
            sample_cls=RecordingSample,
            sample_package_name="recording",
        )
        self.assertEqual(
            RecordingSample.calls,
            [None, None],
        )
        self.assertEqual(
            RecordingSample.ignore_calls,
            [
                ["is_respondent", "outcome_b", "outcome_a", "extra"],
                ["is_respondent", "outcome_b", "outcome_a", "extra"],
            ],
        )

    def test_cli_outcome_columns_explicit_selection(self) -> None:
        RecordingSample = self._recording_sample_cls()
        cli = self._make_cli(outcome_columns="outcome_a,outcome_b")
        cli.process_batch(
            self._make_batch_df(),
            sample_cls=RecordingSample,
            sample_package_name="recording",
        )
        self.assertEqual(
            RecordingSample.calls,
            [("outcome_a", "outcome_b"), ("outcome_a", "outcome_b")],
        )
        self.assertEqual(
            RecordingSample.ignore_calls,
            [["is_respondent", "extra"], ["is_respondent", "extra"]],
        )

    def test_cli_outcome_columns_missing_column_raises(self) -> None:
        cli = self._make_cli(outcome_columns="missing")
        with self.assertRaises(AssertionError):
            cli.check_input_columns(self._make_batch_df().columns)

    def test_cli_weights_impact_on_outcome_method(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "input.csv")
            output_file = os.path.join(temp_dir, "output.csv")
            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    input_file,
                    "--output_file",
                    output_file,
                    "--covariate_columns",
                    "covar_a,covar_b",
                    "--weights_impact_on_outcome_method",
                    "t_test",
                ]
            )
            cli = BalanceCLI(args)
            self.assertEqual(cli.weights_impact_on_outcome_method(), "t_test")

            args_default = parser.parse_args(
                [
                    "--input_file",
                    input_file,
                    "--output_file",
                    output_file,
                    "--covariate_columns",
                    "covar_a,covar_b",
                ]
            )
            cli_default = BalanceCLI(args_default)
            self.assertEqual(cli_default.weights_impact_on_outcome_method(), "t_test")

            args_none = parser.parse_args(
                [
                    "--input_file",
                    input_file,
                    "--output_file",
                    output_file,
                    "--covariate_columns",
                    "covar_a,covar_b",
                    "--weights_impact_on_outcome_method",
                    "none",
                ]
            )
            cli_none = BalanceCLI(args_none)
            self.assertIsNone(cli_none.weights_impact_on_outcome_method())

    def test_process_batch_returns_failure_payload_for_empty_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "input.csv")
            output_file = os.path.join(temp_dir, "output.csv")
            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    input_file,
                    "--output_file",
                    output_file,
                    "--sample_column",
                    "is_respondent",
                    "--covariate_columns",
                    "x",
                ]
            )
            cli = BalanceCLI(args)

            batch_df = pd.DataFrame(
                {
                    "is_respondent": [0, 0],
                    "id": [1, 2],
                    "weight": [1.0, 1.0],
                    "x": [1.0, 2.0],
                }
            )
            result = cli.process_batch(batch_df)

            self.assertTrue(result["adjusted"].empty)
            self.assertEqual(
                result["diagnostics"].to_dict("records"),
                [
                    {
                        "metric": "adjustment_failure",
                        "var": None,
                        "val": 1,
                    },
                    {
                        "metric": "adjustment_failure_reason",
                        "var": None,
                        "val": "No input data",
                    },
                ],
            )

    def test_load_and_check_input_reads_file_and_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "input.csv")
            output_file = os.path.join(temp_dir, "output.csv")
            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    input_file,
                    "--output_file",
                    output_file,
                    "--sample_column",
                    "is_respondent",
                    "--covariate_columns",
                    "x",
                    "--keep_row_column",
                    "keep",
                ]
            )
            cli = BalanceCLI(args)

            input_df = pd.DataFrame(
                {
                    "is_respondent": [1, 0],
                    "id": [1, 2],
                    "weight": [1.0, 1.0],
                    "x": [1.0, 2.0],
                    "keep": [1, 0],
                }
            )
            input_df.to_csv(input_file, index=False)

            loaded = cli.load_and_check_input()
            pd.testing.assert_frame_equal(loaded, input_df)

    def test_write_outputs_skips_diagnostics_when_no_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "input.csv")
            output_file = os.path.join(temp_dir, "output.csv")
            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    input_file,
                    "--output_file",
                    output_file,
                    "--sample_column",
                    "is_respondent",
                    "--covariate_columns",
                    "x",
                ]
            )
            cli = BalanceCLI(args)

            output_df = pd.DataFrame({"id": [1], "weight": [1.25]})
            diagnostics_df = pd.DataFrame(
                {"metric": ["adjustment_failure"], "var": [None], "val": [0]}
            )

            cli.write_outputs(output_df, diagnostics_df)

            pd.testing.assert_frame_equal(
                pd.read_csv(output_file, sep=cli.args.sep_output_file),
                output_df,
            )
            self.assertIsNone(cli.args.diagnostics_output_file)

    def test_write_outputs_writes_adjusted_and_diagnostics_with_custom_seps(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "input.csv")
            output_file = os.path.join(temp_dir, "output.csv")
            diagnostics_output_file = os.path.join(temp_dir, "diagnostics.csv")
            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    input_file,
                    "--output_file",
                    output_file,
                    "--diagnostics_output_file",
                    diagnostics_output_file,
                    "--sample_column",
                    "is_respondent",
                    "--covariate_columns",
                    "x",
                    "--sep_output_file",
                    "\t",
                    "--sep_diagnostics_output_file",
                    ";",
                ]
            )
            cli = BalanceCLI(args)

            output_df = pd.DataFrame({"id": [1], "weight": [1.25]})
            diagnostics_df = pd.DataFrame(
                {
                    "metric": ["adjustment_failure"],
                    "var": [None],
                    "val": [0],
                }
            )

            cli.write_outputs(output_df, diagnostics_df)

            pd.testing.assert_frame_equal(
                pd.read_csv(output_file, sep=cli.args.sep_output_file),
                output_df,
            )

            diagnostics_loaded = pd.read_csv(
                diagnostics_output_file,
                sep=cli.args.sep_diagnostics_output_file,
            )
            self.assertEqual(
                diagnostics_loaded.loc[0, "metric"],
                "adjustment_failure",
            )
            self.assertTrue(pd.isna(diagnostics_loaded.loc[0, "var"]))
            self.assertEqual(diagnostics_loaded.loc[0, "val"], 0)

    def test_cli_help(self) -> None:
        """Test that CLI help command executes without errors."""
        parser = make_parser()

        try:
            parser.parse_args(["--help"])
            # If we get here, something is wrong - help should have exited
            self.fail("Expected SystemExit when parsing --help")
        except SystemExit as e:
            # Help command should exit with code 0
            self.assertEqual(e.code, 0)

    def test_cli_float_or_none(self) -> None:
        """Test the _float_or_none utility function with various inputs."""
        self.assertEqual(_float_or_none(None), None)
        self.assertEqual(_float_or_none("None"), None)
        self.assertEqual(_float_or_none("13.37"), 13.37)

    def test_cli_builds_logistic_regression_model(self) -> None:
        """Ensure CLI JSON kwargs are parsed into a configured LogisticRegression."""
        args = Namespace(
            ipw_logistic_regression_kwargs='{"solver": "liblinear", "max_iter": 321}',
            method="ipw",
        )
        cli = BalanceCLI(args)

        kwargs = cli.logistic_regression_kwargs()
        assert kwargs is not None
        self.assertEqual(kwargs["solver"], "liblinear")
        self.assertEqual(kwargs["max_iter"], 321)

        model = _assert_type(cli.logistic_regression_model())
        self.assertIsInstance(model, LogisticRegression)
        if isinstance(model, LogisticRegression):
            self.assertEqual(model.solver, "liblinear")
            self.assertEqual(model.max_iter, 321)

    def test_cli_omits_logistic_regression_model_without_kwargs(self) -> None:
        """No kwargs should result in no model being constructed."""

        cli = BalanceCLI(Namespace(ipw_logistic_regression_kwargs=None, method="ipw"))

        self.assertIsNone(cli.logistic_regression_model())

    def test_cli_rejects_non_mapping_logistic_regression_kwargs(self) -> None:
        """Invalid JSON structures should raise clear errors."""
        cli = BalanceCLI(Namespace(ipw_logistic_regression_kwargs="[]", method="ipw"))

        with self.assertRaises(ValueError):
            cli.logistic_regression_kwargs()

    def test_cli_accepts_mapping_logistic_regression_kwargs(self) -> None:
        """Dict inputs are forwarded directly without JSON parsing."""
        cli = BalanceCLI(
            Namespace(
                ipw_logistic_regression_kwargs={"solver": "saga", "max_iter": 100},
                method="ipw",
            )
        )

        kwargs = cli.logistic_regression_kwargs()
        assert kwargs is not None
        self.assertEqual(kwargs["solver"], "saga")
        self.assertEqual(kwargs["max_iter"], 100)

    def test_cli_succeed_on_weighting_failure(self) -> None:
        """Test CLI behavior when weighting fails but succeed_on_weighting_failure flag is set."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as in_file,
        ):
            in_contents = "x,y,is_respondent,id,weight\na,b,1,1,1\na,b,0,1,1"
            in_file.write(in_contents)
            in_file.close()
            out_file = os.path.join(temp_dir, "out.csv")

            parser = make_parser()

            args = parser.parse_args(
                [
                    "--input_file",
                    in_file.name,
                    "--output_file",
                    out_file,
                    "--covariate_columns",
                    "x,y",
                    "--succeed_on_weighting_failure",
                ]
            )
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()

            self.assertTrue(os.path.isfile(out_file))

    def test_cli_works(self) -> None:
        """Test basic CLI functionality with sample data and diagnostics output."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as in_file,
        ):
            in_contents = (
                "x,y,is_respondent,id,weight\n"
                + ("a,b,1,1,1\n" * SAMPLE_SIZE_SMALL)
                + ("c,b,0,1,1\n" * SAMPLE_SIZE_SMALL)
            )
            in_file.write(in_contents)
            in_file.close()
            out_file = os.path.join(temp_dir, "out.csv")
            diagnostics_out_file = os.path.join(temp_dir, "diagnostics_out.csv")

            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    in_file.name,
                    "--output_file",
                    out_file,
                    "--diagnostics_output_file",
                    diagnostics_out_file,
                    "--covariate_columns",
                    "x,y",
                    "--num_lambdas=1",
                ]
            )
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()

            self.assertTrue(os.path.isfile(out_file))
            self.assertTrue(os.path.isfile(diagnostics_out_file))

    def test_cli_works_with_row_column_filters(self) -> None:
        """Test CLI functionality with row and column filtering for diagnostics."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as in_file,
        ):
            in_contents = (
                "x,y,z,is_respondent,id,weight\n"
                + ("a,b,g,1,1,1\n" * SAMPLE_SIZE_SMALL)
                + ("c,b,g,1,1,1\n" * SAMPLE_SIZE_SMALL)
                + ("c,b,g,0,1,1\n" * SAMPLE_SIZE_SMALL)
                + ("a,d,n,1,2,1\n" * SAMPLE_SIZE_SMALL)
            )
            in_file.write(in_contents)
            in_file.close()

            out_file = os.path.join(temp_dir, "out.csv")
            diagnostics_out_file = os.path.join(temp_dir, "diagnostics_out.csv")

            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    in_file.name,
                    "--output_file",
                    out_file,
                    "--diagnostics_output_file",
                    diagnostics_out_file,
                    "--covariate_columns",
                    "x,y,z",
                    "--num_lambdas=1",
                    "--rows_to_keep_for_diagnostics",
                    "z == 'g'",  # filtering condition
                    "--covariate_columns_for_diagnostics",
                    "x,y",  # ignoring z
                ]
            )

            # run cli
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()

            # get the files created from cli to pandas to check them
            pd_in_file = pd.read_csv(in_file.name)
            pd_out_file = pd.read_csv(out_file)
            pd_diagnostics_out_file = pd.read_csv(diagnostics_out_file)

            # test stuff
            # Make sure we indeed got all the output files
            self.assertTrue(os.path.isfile(out_file))
            self.assertTrue(os.path.isfile(diagnostics_out_file))

            # the original file had 4*SAMPLE_SIZE_SMALL rows (1x target and 3x sample)
            self.assertEqual(pd_in_file.shape, (4 * SAMPLE_SIZE_SMALL, 6))
            # the cli output includes only the panel (NOT the target)
            #   it also includes all the original columns
            self.assertEqual(pd_out_file.shape, (3 * SAMPLE_SIZE_SMALL, 6))
            self.assertEqual(pd_out_file.is_respondent.mean(), 1)
            # the diagnostics file shows it was calculated on only 2x panelists (as required from the condition)
            ss = pd_diagnostics_out_file.eval(
                "(metric == 'size') & (var == 'sample_obs')"
            )
            self.assertEqual(
                int(pd_diagnostics_out_file[ss]["val"].iloc[0]), 2 * SAMPLE_SIZE_SMALL
            )

            # verify we get diagnostics only for x and y, and not z
            ss = pd_diagnostics_out_file.eval("metric == 'covar_main_asmd_adjusted'")
            output = pd_diagnostics_out_file[ss]["var"].to_list()
            expected = ["x", "y", "mean(asmd)"]
            self.assertEqual(output, expected)

    def test_cli_empty_input(self) -> None:
        """Test CLI behavior with empty input data (header only)."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as in_file,
        ):
            in_contents = "x,y,is_respondent,id,weight\n"
            in_file.write(in_contents)
            in_file.close()
            out_file = os.path.join(temp_dir, "out.csv")
            diagnostics_out_file = os.path.join(temp_dir, "diagnostics_out.csv")

            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    in_file.name,
                    "--output_file",
                    out_file,
                    "--diagnostics_output_file",
                    diagnostics_out_file,
                    "--covariate_columns",
                    "x,y",
                    "--num_lambdas=1",
                ]
            )
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()

            self.assertTrue(os.path.isfile(out_file))
            self.assertTrue(os.path.isfile(diagnostics_out_file))

    def test_cli_empty_input_keep_row(self) -> None:
        """Test CLI behavior with empty input data when using keep_row and batch_columns."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as in_file,
        ):
            in_contents = "x,y,is_respondent,id,weight,keep_row,batch_column\n"
            in_file.write(in_contents)
            in_file.close()
            out_file = os.path.join(temp_dir, "out.csv")
            diagnostics_out_file = os.path.join(temp_dir, "diagnostics_out.csv")

            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    in_file.name,
                    "--output_file",
                    out_file,
                    "--diagnostics_output_file",
                    diagnostics_out_file,
                    "--covariate_columns",
                    "x,y",
                    "--keep_row",
                    "keep_row",
                    "--batch_columns",
                    "batch_column",
                ]
            )
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()

            self.assertTrue(os.path.isfile(out_file))
            self.assertTrue(os.path.isfile(diagnostics_out_file))

    def test_cli_sep_works(self) -> None:
        """Test CLI functionality with custom output file separators."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as in_file,
        ):
            in_contents = (
                "x,y,z,is_respondent,id,weight\n"
                + ("a,b,g,1,1,1\n" * SAMPLE_SIZE_SMALL)
                + ("c,b,g,1,1,1\n" * SAMPLE_SIZE_SMALL)
                + ("c,b,g,0,1,1\n" * SAMPLE_SIZE_SMALL)
                + ("a,d,n,1,2,1\n" * SAMPLE_SIZE_SMALL)
            )
            in_file.write(in_contents)
            in_file.close()
            # pd.read_csv(in_file)

            out_file = os.path.join(temp_dir, "out.csv")
            diagnostics_out_file = os.path.join(temp_dir, "diagnostics_out.csv")

            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    in_file.name,
                    "--output_file",
                    out_file,
                    "--diagnostics_output_file",
                    diagnostics_out_file,
                    "--covariate_columns",
                    "x,y,z",
                    "--num_lambdas=1",
                    "--sep_output_file",
                    ";",
                    "--sep_diagnostics_output_file",
                    ";",
                ]
            )

            # run cli
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()

            # get the files created from cli to pandas to check them
            pd_in_file = pd.read_csv(in_file.name)
            pd_out_file = pd.read_csv(out_file, sep=";")
            pd_diagnostics_out_file = pd.read_csv(diagnostics_out_file, sep=";")

            # test stuff
            # Make sure we indeed got all the output files
            self.assertTrue(os.path.isfile(out_file))
            self.assertTrue(os.path.isfile(diagnostics_out_file))

            # the original file had 4*SAMPLE_SIZE_SMALL rows (1x target and 3x sample)
            self.assertEqual(pd_in_file.shape, (4 * SAMPLE_SIZE_SMALL, 6))
            # the cli output includes only the panel (NOT the target)
            #   it also includes all the original columns
            self.assertEqual(pd_out_file.shape, (3 * SAMPLE_SIZE_SMALL, 6))
            self.assertEqual(pd_out_file.is_respondent.mean(), 1)
            # the diagnostics file shows it was calculated on all 3*SAMPLE_SIZE_SMALL panelists
            ss = pd_diagnostics_out_file.eval(
                "(metric == 'size') & (var == 'sample_obs')"
            )
            self.assertEqual(
                int(pd_diagnostics_out_file[ss]["val"].iloc[0]), 3 * SAMPLE_SIZE_SMALL
            )

    def test_cli_sep_input_works(self) -> None:
        """Test CLI functionality with custom input file separators (TSV)."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as in_file,
        ):
            in_contents = (
                "x\ty\tz\tis_respondent\tid\tweight\n"
                + ("a\tb\tg\t1\t1\t1\n" * SAMPLE_SIZE_SMALL)
                + ("c\tb\tg\t1\t1\t1\n" * SAMPLE_SIZE_SMALL)
                + ("c\tb\tg\t0\t1\t1\n" * SAMPLE_SIZE_SMALL)
                + ("a\td\tn\t1\t2\t1\n" * SAMPLE_SIZE_SMALL)
            )
            in_file.write(in_contents)
            in_file.close()

            out_file = os.path.join(temp_dir, "out.csv")
            diagnostics_out_file = os.path.join(temp_dir, "diagnostics_out.csv")

            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    in_file.name,
                    "--output_file",
                    out_file,
                    "--diagnostics_output_file",
                    diagnostics_out_file,
                    "--covariate_columns",
                    "x,y,z",
                    "--num_lambdas=1",
                    "--sep_input_file",
                    "\t",
                ]
            )

            # run cli
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()

            # get the files created from cli to pandas to check them
            pd_in_file = pd.read_csv(in_file.name, sep="\t")
            pd_out_file = pd.read_csv(out_file)
            pd_diagnostics_out_file = pd.read_csv(diagnostics_out_file)

            # test stuff
            # Make sure we indeed got all the output files
            self.assertTrue(os.path.isfile(out_file))
            self.assertTrue(os.path.isfile(diagnostics_out_file))

            # the original file had 4*SAMPLE_SIZE_SMALL rows (1x target and 3x sample)
            self.assertEqual(pd_in_file.shape, (4 * SAMPLE_SIZE_SMALL, 6))
            # the cli output includes only the panel (NOT the target)
            #   it also includes all the original columns
            self.assertEqual(pd_out_file.shape, (3 * SAMPLE_SIZE_SMALL, 6))
            self.assertEqual(pd_out_file.is_respondent.mean(), 1)
            # the diagnostics file shows it was calculated on all 3*SAMPLE_SIZE_SMALL panelists
            ss = pd_diagnostics_out_file.eval(
                "(metric == 'size') & (var == 'sample_obs')"
            )
            self.assertEqual(
                int(pd_diagnostics_out_file[ss]["val"].iloc[0]), 3 * SAMPLE_SIZE_SMALL
            )

    def test_cli_short_arg_names_works(self) -> None:
        """
        Test CLI backward compatibility with partial argument names.

        Some users used only partial arg names for their pipelines.
        This test verifies new arguments would still be backward compatible.
        """
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as in_file,
        ):
            in_contents = (
                "x,y,z,is_respondent,id,weight\n"
                + ("a,b,g,1,1,1\n" * SAMPLE_SIZE_SMALL)
                + ("c,b,g,1,1,1\n" * SAMPLE_SIZE_SMALL)
                + ("c,b,g,0,1,1\n" * SAMPLE_SIZE_SMALL)
                + ("a,d,n,1,2,1\n" * SAMPLE_SIZE_SMALL)
            )
            in_file.write(in_contents)
            in_file.close()

            out_file = os.path.join(temp_dir, "out.csv")
            diagnostics_out_file = os.path.join(temp_dir, "diagnostics_out.csv")

            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input",
                    in_file.name,
                    "--output",  # instead of output_file
                    out_file,
                    "--diagnostics_output_file",
                    diagnostics_out_file,
                    "--covariate_columns",
                    "x,y,z",
                    "--num_lambdas=1",
                    "--sep_output",
                    ";",
                    "--sep_diagnostics",
                    ";",
                ]
            )

            # run cli
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()

            # get the files created from cli to pandas to check them
            pd_in_file = pd.read_csv(in_file.name)
            pd_out_file = pd.read_csv(out_file, sep=";")
            pd_diagnostics_out_file = pd.read_csv(diagnostics_out_file, sep=";")

            # test stuff
            # Make sure we indeed got all the output files
            self.assertTrue(os.path.isfile(out_file))
            self.assertTrue(os.path.isfile(diagnostics_out_file))

            # the original file had 4*SAMPLE_SIZE_SMALL rows (1x target and 3x sample)
            self.assertEqual(pd_in_file.shape, (4 * SAMPLE_SIZE_SMALL, 6))
            # the cli output includes only the panel (NOT the target)
            #   it also includes all the original columns
            self.assertEqual(pd_out_file.shape, (3 * SAMPLE_SIZE_SMALL, 6))
            self.assertEqual(pd_out_file.is_respondent.mean(), 1)
            # the diagnostics file shows it was calculated on all 3*SAMPLE_SIZE_SMALL panelists
            ss = pd_diagnostics_out_file.eval(
                "(metric == 'size') & (var == 'sample_obs')"
            )
            self.assertEqual(
                int(pd_diagnostics_out_file[ss]["val"].iloc[0]), 3 * SAMPLE_SIZE_SMALL
            )

    def test_method_works(self) -> None:
        """Test CLI functionality with different weighting methods (CBPS and IPW)."""
        # pyrefly: ignore [bad-argument-type]
        np.random.seed(TEST_SEED)
        n_sample = SAMPLE_SIZE_SMALL
        n_target = SAMPLE_SIZE_LARGE
        sample_df = pd.DataFrame(
            {
                "age": np.random.uniform(0, 100, n_sample),
                "gender": np.random.choice((1, 2, 3, 4), n_sample),
                "id": range(n_sample),
                "weight": pd.Series((1,) * n_sample),
            }
        )
        sample_df["is_respondent"] = True
        target_df = pd.DataFrame(
            {
                "age": np.random.uniform(0, 100, n_target),
                "gender": np.random.choice((1, 2, 3, 4), n_target),
                "id": range(n_target),
                "weight": pd.Series((1,) * n_target),
            }
        )
        target_df["is_respondent"] = False
        input_dataset = pd.concat([sample_df, target_df])

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as input_file,
        ):
            input_dataset.to_csv(path_or_buf=input_file)
            input_file.close()
            output_file = os.path.join(temp_dir, "weights_out.csv")
            diagnostics_output_file = os.path.join(temp_dir, "diagnostics_out.csv")
            features = "age,gender"

            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    input_file.name,
                    "--output_file",
                    output_file,
                    "--diagnostics_output_file",
                    diagnostics_output_file,
                    "--covariate_columns",
                    features,
                    "--num_lambdas=1",
                    "--method=cbps",
                ]
            )
            # run cli
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()
            # get the files created from cli to pandas to check them
            diagnostics_output = pd.read_csv(diagnostics_output_file, sep=",")
            self.assertEqual(
                diagnostics_output[diagnostics_output["metric"] == "adjustment_method"][
                    "var"
                ].values,
                np.array(["cbps"]),
            )

            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    input_file.name,
                    "--output_file",
                    output_file,
                    "--diagnostics_output_file",
                    diagnostics_output_file,
                    "--covariate_columns",
                    features,
                    "--num_lambdas=1",
                    "--method=ipw",
                    "--max_de=1.5",
                ]
            )
            # run cli
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()
            # get the files created from cli to pandas to check them
            diagnostics_output = pd.read_csv(diagnostics_output_file, sep=",")
            self.assertEqual(
                diagnostics_output[diagnostics_output["metric"] == "adjustment_method"][
                    "var"
                ].values,
                np.array(["ipw"]),
            )

    def test_method_works_with_rake(self) -> None:
        """Test CLI functionality with raking weighting method."""
        # pyrefly: ignore [bad-argument-type]
        np.random.seed(TEST_SEED)
        n_sample = SAMPLE_SIZE_SMALL
        n_target = SAMPLE_SIZE_LARGE
        sample_df = pd.DataFrame(
            {
                "age": np.random.uniform(0, 100, n_sample),
                "gender": np.random.choice((1, 2, 3, 4), n_sample),
                "id": range(n_sample),
                "weight": pd.Series((1,) * n_sample),
            }
        )
        sample_df["is_respondent"] = True
        target_df = pd.DataFrame(
            {
                "age": np.random.uniform(0, 100, n_target),
                "gender": np.random.choice((1, 2, 3, 4), n_target),
                "id": range(n_target),
                "weight": pd.Series((1,) * n_target),
            }
        )
        target_df["is_respondent"] = False
        input_dataset = pd.concat([sample_df, target_df])

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as input_file,
        ):
            input_dataset.to_csv(path_or_buf=input_file)
            input_file.close()
            output_file = os.path.join(temp_dir, "weights_out.csv")
            diagnostics_output_file = os.path.join(temp_dir, "diagnostics_out.csv")
            features = "age,gender"

            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    input_file.name,
                    "--output_file",
                    output_file,
                    "--diagnostics_output_file",
                    diagnostics_output_file,
                    "--covariate_columns",
                    features,
                    "--method=rake",
                ]
            )
            # run cli
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()
            # get the files created from cli to pandas to check them
            diagnostics_output = pd.read_csv(diagnostics_output_file, sep=",")
            self.assertEqual(
                diagnostics_output[diagnostics_output["metric"] == "adjustment_method"][
                    "var"
                ].values,
                np.array(["rake"]),
            )

    def test_one_hot_encoding_works(self) -> None:
        """Test CLI one-hot encoding parameter with various boolean values."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as in_file,
        ):
            # Assert value is False when "False" is passed
            out_file = os.path.join(temp_dir, "out.csv")
            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    in_file.name,
                    "--output_file",
                    out_file,
                    "--covariate_columns",
                    "x,y",
                    "--one_hot_encoding",
                    "False",
                ]
            )
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            self.assertFalse(cli.one_hot_encoding())
            self.assertEqual(type(cli.one_hot_encoding()), bool)

            # Assert value is True by default
            args2 = parser.parse_args(
                [
                    "--input_file",
                    in_file.name,
                    "--output_file",
                    out_file,
                    "--covariate_columns",
                    "x,y",
                ]
            )
            cli2 = BalanceCLI(args2)
            cli2.update_attributes_for_main_used_by_adjust()
            self.assertTrue(cli2.one_hot_encoding())
            self.assertEqual(type(cli2.one_hot_encoding()), bool)

            # Assert invalid value raises an error
            with self.assertRaises(ValueError):
                args3 = parser.parse_args(
                    [
                        "--input_file",
                        in_file.name,
                        "--output_file",
                        out_file,
                        "--covariate_columns",
                        "x,y",
                        "--one_hot_encoding",
                        "Invalid Value",
                    ]
                )
                cli3 = BalanceCLI(args3)
                cli3.update_attributes_for_main_used_by_adjust()

    def test_transformations_works(self) -> None:
        """Test CLI functionality with different transformation options (None and default)."""
        input_dataset = _create_sample_and_target_data()

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as input_file,
        ):
            input_dataset.to_csv(path_or_buf=input_file)
            input_file.close()
            output_file = os.path.join(temp_dir, "weights_out.csv")
            diagnostics_output_file = os.path.join(temp_dir, "diagnostics_out.csv")
            features = "age,gender"

            # test transformations=None
            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    input_file.name,
                    "--output_file",
                    output_file,
                    "--diagnostics_output_file",
                    diagnostics_output_file,
                    "--covariate_columns",
                    features,
                    "--num_lambdas=1",
                    "--transformations=None",
                ]
            )
            # run cli
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()
            # get the files created from cli to pandas to check them
            diagnostics_output = pd.read_csv(diagnostics_output_file, sep=",")
            self.assertEqual(
                diagnostics_output[diagnostics_output["metric"] == "model_coef"][
                    "var"
                ].values,
                np.array(["intercept", "age", "gender"]),
            )

            # test transformations='default'
            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    input_file.name,
                    "--output_file",
                    output_file,
                    "--diagnostics_output_file",
                    diagnostics_output_file,
                    "--covariate_columns",
                    features,
                    "--num_lambdas=1",
                    "--transformations=default",
                ]
            )
            # run cli
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()
            # get the files created from cli to pandas to check them
            diagnostics_output = pd.read_csv(diagnostics_output_file, sep=",")
            self.assertEqual(
                diagnostics_output[diagnostics_output["metric"] == "model_coef"][
                    "var"
                ].values,
                np.array(
                    [
                        "intercept",
                        "C(age, one_hot_encoding_greater_2)[(0.503, 12.698]]",
                        "C(age, one_hot_encoding_greater_2)[(12.698, 24.046]]",
                        "C(age, one_hot_encoding_greater_2)[(24.046, 34.075]]",
                        "C(age, one_hot_encoding_greater_2)[(34.075, 42.568]]",
                        "C(age, one_hot_encoding_greater_2)[(42.568, 52.977]]",
                        "C(age, one_hot_encoding_greater_2)[(52.977, 61.757]]",
                        "C(age, one_hot_encoding_greater_2)[(61.757, 69.386]]",
                        "C(age, one_hot_encoding_greater_2)[(69.386, 79.639]]",
                        "C(age, one_hot_encoding_greater_2)[(79.639, 88.948]]",
                        "C(age, one_hot_encoding_greater_2)[(88.948, 99.992]]",
                        "C(gender, one_hot_encoding_greater_2)[(0.999, 2.0]]",
                        "C(gender, one_hot_encoding_greater_2)[(2.0, 3.0]]",
                        "C(gender, one_hot_encoding_greater_2)[(3.0, 4.0]]",
                    ]
                ),
            )

            # test default value for transformations
            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    input_file.name,
                    "--output_file",
                    output_file,
                    "--diagnostics_output_file",
                    diagnostics_output_file,
                    "--covariate_columns",
                    features,
                    "--num_lambdas=1",
                ]
            )
            # run cli
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()
            # get the files created from cli to pandas to check them
            diagnostics_output = pd.read_csv(diagnostics_output_file, sep=",")
            self.assertEqual(
                diagnostics_output[diagnostics_output["metric"] == "model_coef"][
                    "var"
                ].values,
                np.array(
                    [
                        "intercept",
                        "C(age, one_hot_encoding_greater_2)[(0.503, 12.698]]",
                        "C(age, one_hot_encoding_greater_2)[(12.698, 24.046]]",
                        "C(age, one_hot_encoding_greater_2)[(24.046, 34.075]]",
                        "C(age, one_hot_encoding_greater_2)[(34.075, 42.568]]",
                        "C(age, one_hot_encoding_greater_2)[(42.568, 52.977]]",
                        "C(age, one_hot_encoding_greater_2)[(52.977, 61.757]]",
                        "C(age, one_hot_encoding_greater_2)[(61.757, 69.386]]",
                        "C(age, one_hot_encoding_greater_2)[(69.386, 79.639]]",
                        "C(age, one_hot_encoding_greater_2)[(79.639, 88.948]]",
                        "C(age, one_hot_encoding_greater_2)[(88.948, 99.992]]",
                        "C(gender, one_hot_encoding_greater_2)[(0.999, 2.0]]",
                        "C(gender, one_hot_encoding_greater_2)[(2.0, 3.0]]",
                        "C(gender, one_hot_encoding_greater_2)[(3.0, 4.0]]",
                    ]
                ),
            )

    def test_formula_works(self) -> None:
        """Test CLI functionality with custom formula specifications."""
        input_dataset = _create_sample_and_target_data()

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as input_file,
        ):
            input_dataset.to_csv(path_or_buf=input_file)
            input_file.close()
            output_file = os.path.join(temp_dir, "weights_out.csv")
            diagnostics_output_file = os.path.join(temp_dir, "diagnostics_out.csv")
            features = "age,gender"

            # test no formula
            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    input_file.name,
                    "--output_file",
                    output_file,
                    "--diagnostics_output_file",
                    diagnostics_output_file,
                    "--covariate_columns",
                    features,
                    "--num_lambdas=1",
                    "--transformations=None",
                ]
            )
            # run cli
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()
            # get the files created from cli to pandas to check them
            diagnostics_output = pd.read_csv(diagnostics_output_file, sep=",")
            self.assertEqual(
                diagnostics_output[diagnostics_output["metric"] == "model_coef"][
                    "var"
                ].values,
                np.array(["intercept", "age", "gender"]),
            )

            # test transformations=age*gender
            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    input_file.name,
                    "--output_file",
                    output_file,
                    "--diagnostics_output_file",
                    diagnostics_output_file,
                    "--covariate_columns",
                    features,
                    "--num_lambdas=1",
                    "--transformations=None",
                    "--formula=age*gender",
                ]
            )
            # run cli
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()
            # get the files created from cli to pandas to check them
            diagnostics_output = pd.read_csv(diagnostics_output_file, sep=",")
            self.assertEqual(
                diagnostics_output[diagnostics_output["metric"] == "model_coef"][
                    "var"
                ].values,
                np.array(["intercept", "age", "age:gender", "gender"]),
            )

    def test_cli_return_df_with_original_dtypes(self) -> None:
        """Test CLI flag for preserving original data types in output DataFrames."""
        out_True = check_some_flags(True, "--return_df_with_original_dtypes")
        out_False = check_some_flags(False, "--return_df_with_original_dtypes")

        # IF the flag 'return_df_with_original_dtypes' is not passed, then the dtypes of the output dataframes are different
        self.assertEqual(
            out_False["pd_in"].dtypes.to_dict(),
            {
                "x": dtype("float64"),
                "y": dtype("int64"),
                "is_respondent": dtype("int64"),
                "id": dtype("int64"),
                "weight": dtype("int64"),
            },
        )
        self.assertEqual(
            out_False["pd_out"].dtypes.to_dict(),
            {
                "id": dtype("int64"),
                "x": dtype("float64"),
                "y": dtype("float64"),
                "is_respondent": dtype("float64"),
                "weight": dtype("float64"),
            },
        )

        # IF the flag 'return_df_with_original_dtypes' IS passed, then the dtypes of the output dataframes are the SAME:
        # The input df dtypes are the same
        self.assertEqual(
            out_True["pd_in"].dtypes.to_dict(), out_False["pd_in"].dtypes.to_dict()
        )
        # But now the output df dtypes are also the same (the order of the columns is different though)
        self.assertEqual(
            out_True["pd_out"].dtypes.to_dict(),
            {
                "id": dtype("int64"),
                "x": dtype("float64"),
                "y": dtype("int64"),
                "is_respondent": dtype("int64"),
                "weight": dtype("int64"),
            },
        )

    def test_cli_standardize_types(self) -> None:
        """Test CLI standardize_types flag with different boolean values."""
        out_False = check_some_flags(True, "--standardize_types=False")
        out_True = check_some_flags(
            False, "--standardize_types=True"
        )  # This is the same as not using the flag at all
        out_True_flag_on = check_some_flags(
            True, "--standardize_types=True"
        )  # This is the same as not using the flag at all

        # IF the flag 'skip_standardize_types' is not passed, then the dtypes of the output dataframes are different
        self.assertEqual(
            out_True["pd_in"].dtypes.to_dict(),
            {
                "x": dtype("float64"),
                "y": dtype("int64"),
                "is_respondent": dtype("int64"),
                "id": dtype("int64"),
                "weight": dtype("int64"),
            },
        )
        self.assertEqual(
            out_True["pd_out"].dtypes.to_dict(),
            {
                "id": dtype("int64"),
                "x": dtype("float64"),
                "y": dtype("float64"),
                "is_respondent": dtype("float64"),
                "weight": dtype("float64"),
            },
        )

        # IF the flag 'skip_standardize_types' IS passed, then the dtypes of the output dataframes are the SAME:
        # The input df dtypes are the same
        self.assertEqual(
            out_False["pd_in"].dtypes.to_dict(), out_True["pd_in"].dtypes.to_dict()
        )
        # But now the output df dtypes are also the same (the order of the columns is different though)
        self.assertEqual(
            out_False["pd_out"].dtypes.to_dict(),
            {
                "id": dtype("int64"),
                "x": dtype("float64"),
                "y": dtype("int64"),
                "is_respondent": dtype("int64"),
                "weight": dtype("float64"),
            },
        )

        self.assertEqual(
            out_True["pd_out"].dtypes.to_dict(),
            out_True_flag_on["pd_out"].dtypes.to_dict(),
        )


class TestBalanceCLI_keep_columns(balance.testutil.BalanceTestCase):
    """Test cases for keep_columns and related methods (lines 113-115, 284-286)."""

    def test_keep_columns_returns_list_when_set(self) -> None:
        """Test keep_columns returns list of column names when set.

        Verifies lines 284-286 in cli.py.
        """
        args = Namespace(keep_columns="id,weight,extra")
        cli = BalanceCLI(args)
        result = cli.keep_columns()
        self.assertEqual(result, ["id", "weight", "extra"])

    def test_keep_columns_returns_none_when_not_set(self) -> None:
        """Test keep_columns returns None when not set.

        Verifies lines 284-286 in cli.py.
        """
        args = Namespace(keep_columns=None)
        cli = BalanceCLI(args)
        result = cli.keep_columns()
        self.assertIsNone(result)

    def test_keep_columns_strips_whitespace(self) -> None:
        """Test keep_columns trims whitespace around comma-separated names."""
        args = Namespace(keep_columns=" id, weight ,extra ")
        cli = BalanceCLI(args)
        self.assertEqual(cli.keep_columns(), ["id", "weight", "extra"])

    def test_keep_columns_raises_for_empty_column_name(self) -> None:
        """Test keep_columns rejects empty names in comma-separated input."""
        args = Namespace(keep_columns="id,,weight")
        cli = BalanceCLI(args)
        with self.assertRaisesRegex(
            ValueError,
            "--keep_columns must be a comma-separated list of non-empty column names",
        ):
            cli.keep_columns()

    def test_keep_columns_raises_for_empty_string(self) -> None:
        """Test keep_columns rejects an explicitly provided empty string."""
        args = Namespace(keep_columns="")
        cli = BalanceCLI(args)
        with self.assertRaisesRegex(
            ValueError,
            "--keep_columns must be a comma-separated list of non-empty column names",
        ):
            cli.keep_columns()

    def test_has_keep_columns_with_keep_columns(self) -> None:
        """Test has_keep_columns returns True when keep_columns is set."""
        args = Namespace(keep_columns="id,weight")
        cli = BalanceCLI(args)
        self.assertTrue(cli.has_keep_columns())

    def test_check_input_columns_with_keep_columns(self) -> None:
        """Test check_input_columns includes keep_columns in validation.

        Verifies lines 112-115 in cli.py.
        """
        args = Namespace(
            id_column="id",
            weight_column="weight",
            covariate_columns="x,y",
            sample_column="is_respondent",
            batch_columns=None,
            keep_columns="keep_col",
            keep_row_column=None,
            outcome_columns=None,
        )
        cli = BalanceCLI(args)

        columns = ["id", "weight", "x", "y", "is_respondent", "keep_col"]
        cli.check_input_columns(columns)

    def test_check_input_columns_raises_when_keep_column_missing(self) -> None:
        """Test check_input_columns raises when keep_column not in input columns.

        Verifies lines 112-115 in cli.py.
        """
        args = Namespace(
            id_column="id",
            weight_column="weight",
            covariate_columns="x,y",
            sample_column="is_respondent",
            batch_columns=None,
            keep_columns="missing_col",
            keep_row_column=None,
            outcome_columns=None,
        )
        cli = BalanceCLI(args)

        columns = ["id", "weight", "x", "y", "is_respondent"]
        with self.assertRaises(AssertionError):
            cli.check_input_columns(columns)


class TestBalanceCLI_csv_column_parsing(balance.testutil.BalanceTestCase):
    """Test normalized parsing of comma-separated column arguments."""

    def test_covariate_columns_strip_whitespace(self) -> None:
        args = Namespace(covariate_columns=" a, b ,c ")
        cli = BalanceCLI(args)
        self.assertEqual(cli.covariate_columns(), ["a", "b", "c"])

    def test_outcome_columns_raise_for_empty_name(self) -> None:
        args = Namespace(outcome_columns="y,,z")
        cli = BalanceCLI(args)
        with self.assertRaisesRegex(
            ValueError,
            "--outcome_columns must be a comma-separated list of non-empty column names",
        ):
            cli.outcome_columns()

    def test_outcome_columns_raise_for_empty_string(self) -> None:
        args = Namespace(outcome_columns="")
        cli = BalanceCLI(args)
        with self.assertRaisesRegex(
            ValueError,
            "--outcome_columns must be a comma-separated list of non-empty column names",
        ):
            cli.outcome_columns()

    def test_batch_columns_raise_for_empty_name(self) -> None:
        args = Namespace(batch_columns="region,")
        cli = BalanceCLI(args)
        with self.assertRaisesRegex(
            ValueError,
            "--batch_columns must be a comma-separated list of non-empty column names",
        ):
            cli.batch_columns()

    def test_covariate_columns_for_diagnostics_strip_whitespace(self) -> None:
        args = Namespace(covariate_columns_for_diagnostics=" x, y ")
        cli = BalanceCLI(args)
        self.assertEqual(cli.covariate_columns_for_diagnostics(), ["x", "y"])

    def test_keep_columns_preserved_in_adjusted_output(self) -> None:
        """Test that --keep_columns columns survive adjustment via ignored_columns.

        A keep column that is not id, weight, covariate, or outcome should be
        routed to ignored_columns by process_batch, carried through the Sample,
        and available for adapt_output to subset without KeyError.
        """
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as in_file,
        ):
            in_contents = (
                "x,y,is_respondent,id,weight,extra_col\n"
                + ("1.0,50,1,1,1,abc\n" * 100)
                + ("2.0,60,0,1,1,def\n" * 100)
            )
            in_file.write(in_contents)
            in_file.close()
            out_file = os.path.join(temp_dir, "out.csv")

            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    in_file.name,
                    "--output_file",
                    out_file,
                    "--covariate_columns",
                    "x,y",
                    "--num_lambdas=1",
                    "--keep_columns",
                    "id,weight,extra_col",
                ]
            )
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()

            self.assertTrue(os.path.isfile(out_file))
            pd_out = pd.read_csv(out_file)
            # adapt_output should have subsetted to exactly these columns
            self.assertEqual(
                sorted(pd_out.columns.tolist()), ["extra_col", "id", "weight"]
            )
            # extra_col values should be preserved from the sample rows
            self.assertTrue((pd_out["extra_col"] == "abc").all())


class TestBalanceCLI_num_lambdas(balance.testutil.BalanceTestCase):
    """Test cases for num_lambdas method (line 401)."""

    def test_num_lambdas_returns_none_when_not_set(self) -> None:
        """Test num_lambdas returns None when args.num_lambdas is None.

        Verifies lines 400-401 in cli.py.
        """
        args = Namespace(num_lambdas=None)
        cli = BalanceCLI(args)
        result = cli.num_lambdas()
        self.assertIsNone(result)

    def test_num_lambdas_returns_positive_int_when_set(self) -> None:
        """Test num_lambdas returns a positive int when set."""
        for value in ("250", "+250", "0250", 250):
            with self.subTest(value=value):
                args = Namespace(num_lambdas=value)
                cli = BalanceCLI(args)
                result = cli.num_lambdas()
                self.assertEqual(result, 250)
                self.assertIsInstance(result, int)

    def test_num_lambdas_rejects_invalid_namespace_values(self) -> None:
        """Test direct Namespace values use the same validation as argparse."""
        invalid_values = ("1.5", 1.5, "0", 0, "-1", -1, "abc", True)
        for value in invalid_values:
            with self.subTest(value=value):
                cli = BalanceCLI(Namespace(num_lambdas=value))
                with self.assertRaises(ArgumentTypeError):
                    cli.num_lambdas()

    def test_parser_accepts_positive_integer_num_lambdas(self) -> None:
        """Test parser accepts positive integer num_lambdas values."""
        parser = make_parser()
        for value in ("250", "+250", "0250"):
            with self.subTest(value=value):
                args = parser.parse_args(
                    [
                        "--input_file",
                        "in.csv",
                        "--output_file",
                        "out.csv",
                        "--covariate_columns",
                        "x",
                        f"--num_lambdas={value}",
                    ]
                )
                self.assertEqual(args.num_lambdas, 250)
                self.assertIsInstance(args.num_lambdas, int)

    def test_parser_rejects_invalid_num_lambdas(self) -> None:
        """Test parser rejects non-positive and non-integer num_lambdas values."""
        parser = make_parser()
        invalid_values = ("1.5", "1.0", "0", "-1", "abc", "")
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parser.parse_args([f"--num_lambdas={value}"])


class TestBalanceCLI_ipw_kwargs(balance.testutil.BalanceTestCase):
    """Test cases for ipw_logistic_regression_kwargs (lines 499-500)."""

    def test_ipw_logistic_regression_kwargs_invalid_json_raises_error(self) -> None:
        """Test that invalid JSON string raises ValueError.

        Verifies lines 499-502 in cli.py.
        """
        args = Namespace(ipw_logistic_regression_kwargs="not valid json", method="ipw")
        cli = BalanceCLI(args)
        with self.assertRaises(ValueError) as context:
            cli.logistic_regression_kwargs()
        self.assertIn("must be a JSON object string", str(context.exception))

    def test_ipw_logistic_regression_kwargs_valid_json(self) -> None:
        """Test that valid JSON object is parsed correctly."""
        args = Namespace(
            ipw_logistic_regression_kwargs='{"C": 0.5, "max_iter": 1000}', method="ipw"
        )
        cli = BalanceCLI(args)
        result = cli.logistic_regression_kwargs()
        self.assertEqual(result, {"C": 0.5, "max_iter": 1000})

    def test_ipw_logistic_regression_kwargs_non_object_raises_error(self) -> None:
        """Test that JSON non-object raises ValueError.

        Verifies lines 503-506 in cli.py.
        """
        args = Namespace(ipw_logistic_regression_kwargs="[1, 2, 3]", method="ipw")
        cli = BalanceCLI(args)
        with self.assertRaises(ValueError) as context:
            cli.logistic_regression_kwargs()
        self.assertIn("must decode to a JSON object", str(context.exception))


class TestBalanceCLI_adapt_output(balance.testutil.BalanceTestCase):
    """Test cases for adapt_output method (lines 830-831, 834)."""

    def test_adapt_output_filters_rows_by_keep_row_column(self) -> None:
        """Test adapt_output filters rows by keep_row_column.

        Verifies lines 829-831 in cli.py.
        """
        args = Namespace(
            keep_row_column="keep",
            keep_columns=None,
        )
        cli = BalanceCLI(args)
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30], "keep": [1, 0, 1]})
        result = cli.adapt_output(df)
        self.assertEqual(result["id"].tolist(), [1, 3])
        self.assertEqual(len(result), 2)

    def test_adapt_output_selects_keep_columns(self) -> None:
        """Test adapt_output selects only keep_columns.

        Verifies lines 833-834 in cli.py.
        """
        args = Namespace(
            keep_row_column=None,
            keep_columns="id,value",
        )
        cli = BalanceCLI(args)
        df = pd.DataFrame({"id": [1, 2], "value": [10, 20], "extra": [100, 200]})
        result = cli.adapt_output(df)
        self.assertEqual(list(result.columns), ["id", "value"])
        self.assertNotIn("extra", result.columns)

    def test_adapt_output_empty_df_returns_empty(self) -> None:
        """Test adapt_output with empty DataFrame returns empty.

        Verifies line 826-827 in cli.py.
        """
        args = Namespace(
            keep_row_column="keep",
            keep_columns="id",
        )
        cli = BalanceCLI(args)
        df = pd.DataFrame()
        result = cli.adapt_output(df)
        self.assertTrue(result.empty)

    def test_cli_succeed_on_weighting_failure_with_return_df_with_original_dtypes(
        self,
    ) -> None:
        """Test succeed_on_weighting_failure flag with return_df_with_original_dtypes.

        Verifies lines 757-794 in cli.py - the exception handling path when
        weighting fails and return_df_with_original_dtypes is True.
        """
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as in_file,
        ):
            in_contents = "x,y,is_respondent,id,weight\na,b,1,1,1\na,b,0,1,1"
            in_file.write(in_contents)
            in_file.close()
            out_file = os.path.join(temp_dir, "out.csv")
            diagnostics_out_file = os.path.join(temp_dir, "diagnostics_out.csv")

            parser = make_parser()

            args = parser.parse_args(
                [
                    "--input_file",
                    in_file.name,
                    "--output_file",
                    out_file,
                    "--diagnostics_output_file",
                    diagnostics_out_file,
                    "--covariate_columns",
                    "x,y",
                    "--num_lambdas=1",
                    "--succeed_on_weighting_failure",
                    "--return_df_with_original_dtypes",
                ]
            )
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()

            self.assertTrue(os.path.isfile(out_file))
            self.assertTrue(os.path.isfile(diagnostics_out_file))

            diagnostics_df = pd.read_csv(diagnostics_out_file)
            self.assertIn("adjustment_failure", diagnostics_df["metric"].values)

    def test_cli_ipw_method_with_model_in_adjusted_kwargs(self) -> None:
        """Test CLI with IPW method to verify model is passed to adjust.

        Verifies line 719 in cli.py where model is added to adjusted_kwargs.
        """
        input_dataset = _create_sample_and_target_data()

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as input_file,
        ):
            input_dataset.to_csv(path_or_buf=input_file)
            input_file.close()
            output_file = os.path.join(temp_dir, "weights_out.csv")
            diagnostics_output_file = os.path.join(temp_dir, "diagnostics_out.csv")
            features = "age,gender"

            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    input_file.name,
                    "--output_file",
                    output_file,
                    "--diagnostics_output_file",
                    diagnostics_output_file,
                    "--covariate_columns",
                    features,
                    "--num_lambdas=1",
                    "--method=ipw",
                    "--ipw_logistic_regression_kwargs",
                    '{"solver": "lbfgs", "max_iter": 200}',
                ]
            )
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()

            self.assertTrue(os.path.isfile(output_file))
            self.assertTrue(os.path.isfile(diagnostics_output_file))

    def test_cli_batch_columns_empty_batches(self) -> None:
        """Test CLI batch processing with empty batches.

        Verifies lines 1082-1099, 1101-1106 in cli.py - batch processing
        path including the empty results case.
        """
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as in_file,
        ):
            in_contents = (
                "x,y,is_respondent,id,weight,batch\n"
                + ("1.0,50,1,1,1,A\n" * 50)
                + ("2.0,60,0,1,1,A\n" * 50)
                + ("1.0,50,1,2,1,B\n" * 50)
                + ("2.0,60,0,2,1,B\n" * 50)
            )
            in_file.write(in_contents)
            in_file.close()
            out_file = os.path.join(temp_dir, "out.csv")
            diagnostics_out_file = os.path.join(temp_dir, "diagnostics_out.csv")

            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    in_file.name,
                    "--output_file",
                    out_file,
                    "--diagnostics_output_file",
                    diagnostics_out_file,
                    "--covariate_columns",
                    "x,y",
                    "--num_lambdas=1",
                    "--batch_columns",
                    "batch",
                ]
            )
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()

            self.assertTrue(os.path.isfile(out_file))
            self.assertTrue(os.path.isfile(diagnostics_out_file))

            output_df = pd.read_csv(out_file)
            self.assertTrue(len(output_df) > 0)


class TestCliMainFunction(balance.testutil.BalanceTestCase):
    """Test cases for CLI main() entry point function (lines 1421-1425)."""

    def test_main_is_callable(self) -> None:
        """Test that main function is callable.

        Verifies lines 1421-1425 in cli.py.
        """
        from balance.cli import main

        self.assertTrue(callable(main))

    def test_main_runs_with_valid_args(self) -> None:
        """Test that main() function executes successfully with valid arguments.

        Verifies lines 1424-1428 in cli.py - the full main() entry point.
        """
        import sys
        from unittest.mock import patch

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as in_file,
        ):
            in_contents = (
                "x,y,is_respondent,id,weight\n"
                + ("1.0,50,1,1,1\n" * 100)
                + ("2.0,60,0,1,1\n" * 100)
            )
            in_file.write(in_contents)
            in_file.close()
            out_file = os.path.join(temp_dir, "out.csv")

            test_args = [
                "balance_cli",
                "--input_file",
                in_file.name,
                "--output_file",
                out_file,
                "--covariate_columns",
                "x,y",
                "--num_lambdas=1",
            ]

            from balance.cli import main

            with patch.object(sys, "argv", test_args):
                main()

            self.assertTrue(os.path.isfile(out_file))


class TestCliExceptionHandling(balance.testutil.BalanceTestCase):
    """Test cases for exception handling in process_batch (lines 760-797)."""

    def test_process_batch_raises_without_succeed_on_weighting_failure(self) -> None:
        """Test that exception is re-raised when succeed_on_weighting_failure is False.

        Verifies lines 796-797 in cli.py (else: raise e).
        """
        from unittest.mock import patch

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as in_file,
        ):
            in_contents = (
                "x,y,is_respondent,id,weight\n"
                + ("1.0,50,1,1,1\n" * 50)
                + ("2.0,60,0,1,1\n" * 50)
            )
            in_file.write(in_contents)
            in_file.close()
            out_file = os.path.join(temp_dir, "out.csv")

            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    in_file.name,
                    "--output_file",
                    out_file,
                    "--covariate_columns",
                    "x,y",
                ]
            )
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()

            # Mock the adjust method to raise an exception to test the error handling path
            with patch(
                "balance.sample_class.Sample.adjust",
                side_effect=ValueError("Simulated weighting failure"),
            ):
                with self.assertRaisesRegex(ValueError, r"Simulated weighting failure"):
                    cli.main()

    def test_succeed_on_weighting_failure_exception_path(self) -> None:
        """Test exception handling when succeed_on_weighting_failure is True.

        Verifies lines 762-783 in cli.py - the exception handling path when
        adjustment fails and succeed_on_weighting_failure is True.
        This tests the else branch (lines 781-782) where return_df_with_original_dtypes is False.
        """
        from unittest.mock import patch

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as in_file,
        ):
            in_contents = (
                "x,y,is_respondent,id,weight\n"
                + ("1.0,50,1,1,1\n" * 50)
                + ("2.0,60,0,1,1\n" * 50)
            )
            in_file.write(in_contents)
            in_file.close()
            out_file = os.path.join(temp_dir, "out.csv")
            diagnostics_out_file = os.path.join(temp_dir, "diagnostics_out.csv")

            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    in_file.name,
                    "--output_file",
                    out_file,
                    "--diagnostics_output_file",
                    diagnostics_out_file,
                    "--covariate_columns",
                    "x,y",
                    "--succeed_on_weighting_failure",
                ]
            )
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()

            # Mock the adjust method to raise an exception
            # Suppress FutureWarning from pandas about setting incompatible dtype
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Setting an item of incompatible dtype",
                    category=FutureWarning,
                )
                with patch(
                    "balance.sample_class.Sample.adjust",
                    side_effect=ValueError("Simulated weighting failure for testing"),
                ):
                    cli.main()

            self.assertTrue(os.path.isfile(out_file))
            self.assertTrue(os.path.isfile(diagnostics_out_file))

            diagnostics_df = pd.read_csv(diagnostics_out_file)
            self.assertIn("adjustment_failure", diagnostics_df["metric"].values)
            self.assertIn("adjustment_failure_reason", diagnostics_df["metric"].values)
            # Check that the error message is captured
            failure_reason = diagnostics_df[
                diagnostics_df["metric"] == "adjustment_failure_reason"
            ]["val"].values[0]
            self.assertIn("Simulated weighting failure", failure_reason)

    def test_succeed_on_weighting_failure_with_return_original_dtypes(self) -> None:
        """Test exception handling with succeed_on_weighting_failure and return_df_with_original_dtypes.

        Verifies lines 771-780 in cli.py - the return_df_with_original_dtypes branch
        inside the exception handler when weighting fails.
        """
        from unittest.mock import patch

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as in_file,
        ):
            # Use float weights (1.0) so that when set_weights(None) is called,
            # the dtype conversion back to float64 can handle None/NaN values
            in_contents = (
                "x,y,is_respondent,id,weight\n"
                + ("1.0,50.0,1,1,1.0\n" * 50)
                + ("2.0,60.0,0,1,1.0\n" * 50)
            )
            in_file.write(in_contents)
            in_file.close()
            out_file = os.path.join(temp_dir, "out.csv")
            diagnostics_out_file = os.path.join(temp_dir, "diagnostics_out.csv")

            parser = make_parser()
            args = parser.parse_args(
                [
                    "--input_file",
                    in_file.name,
                    "--output_file",
                    out_file,
                    "--diagnostics_output_file",
                    diagnostics_out_file,
                    "--covariate_columns",
                    "x,y",
                    "--succeed_on_weighting_failure",
                    "--return_df_with_original_dtypes",
                ]
            )
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()

            # Mock the adjust method to raise an exception
            # Suppress FutureWarning from pandas about setting incompatible dtype
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Setting an item of incompatible dtype",
                    category=FutureWarning,
                )
                with patch(
                    "balance.sample_class.Sample.adjust",
                    side_effect=ValueError(
                        "Simulated weighting failure for dtype test"
                    ),
                ):
                    cli.main()

            self.assertTrue(os.path.isfile(out_file))
            self.assertTrue(os.path.isfile(diagnostics_out_file))

            diagnostics_df = pd.read_csv(diagnostics_out_file)
            self.assertIn("adjustment_failure", diagnostics_df["metric"].values)
            self.assertIn("adjustment_failure_reason", diagnostics_df["metric"].values)


class TestCliParseColumnsNone(balance.testutil.BalanceTestCase):
    """Cover _parse_csv_columns_arg raising ValueError when value is None (line 42)."""

    def test_parse_csv_columns_arg_raises_on_none(self) -> None:
        """_parse_csv_columns_arg(None, ...) raises ValueError."""
        from balance.cli import _parse_csv_columns_arg

        with self.assertRaisesRegex(ValueError, "cannot be None"):
            _parse_csv_columns_arg(None, "test_arg")
