# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os.path
import tempfile

import balance.testutil
import numpy as np
import pandas as pd
from balance.cli import _float_or_none, BalanceCLI, make_parser
from numpy import dtype

# Test constants
SAMPLE_SIZE_SMALL = 1000
SAMPLE_SIZE_LARGE = 2000
TEST_SEED = 2021


def check_some_flags(flag=True, the_flag_str="--skip_standardize_types"):
    """
    Helper function to test CLI flags with standardized input data.

    Args:
        flag: Whether to include the specified flag in CLI arguments
        the_flag_str: The CLI flag string to test

    Returns:
        Dict containing input and output pandas DataFrames for comparison
    """
    with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False
    ) as in_file:
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


def _create_sample_and_target_data():
    """
    Helper function to create standardized sample and target datasets for testing.

    Returns:
        pd.DataFrame: Combined dataset with sample and target data, including
                     age, gender, id, weight, and is_respondent columns.
    """
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
    def test_cli_help(self):
        """Test that CLI help command executes without errors."""
        parser = make_parser()

        try:
            parser.parse_args(["--help"])
            # If we get here, something is wrong - help should have exited
            self.fail("Expected SystemExit when parsing --help")
        except SystemExit as e:
            # Help command should exit with code 0
            self.assertEqual(e.code, 0)

    def test_cli_float_or_none(self):
        """Test the _float_or_none utility function with various inputs."""
        self.assertEqual(_float_or_none(None), None)
        self.assertEqual(_float_or_none("None"), None)
        self.assertEqual(_float_or_none("13.37"), 13.37)

    def test_cli_succeed_on_weighting_failure(self):
        """Test CLI behavior when weighting fails but succeed_on_weighting_failure flag is set."""
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False
        ) as in_file:
            in_contents = "x,y,is_respondent,id,weight\n" "a,b,1,1,1\n" "a,b,0,1,1"
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

    def test_cli_works(self):
        """Test basic CLI functionality with sample data and diagnostics output."""
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False
        ) as in_file:
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
                ]
            )
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()

            self.assertTrue(os.path.isfile(out_file))
            self.assertTrue(os.path.isfile(diagnostics_out_file))

    def test_cli_works_with_row_column_filters(self):
        """Test CLI functionality with row and column filtering for diagnostics."""
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False
        ) as in_file:
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

            # the original file had 4000 rows (1k target and 3k sample)
            self.assertEqual(pd_in_file.shape, (4000, 6))
            # the cli output includes only the panel (NOT the target)
            #   it also includes all the original columns
            self.assertEqual(pd_out_file.shape, (3000, 6))
            self.assertEqual(pd_out_file.is_respondent.mean(), 1)
            # the diagnostics file shows it was calculated on only 2k panelists (as required from the condition)
            ss = pd_diagnostics_out_file.eval(
                "(metric == 'size') & (var == 'sample_obs')"
            )
            self.assertEqual(int(pd_diagnostics_out_file[ss]["val"].iloc[0]), 2000)

            # verify we get diagnostics only for x and y, and not z
            ss = pd_diagnostics_out_file.eval("metric == 'covar_main_asmd_adjusted'")
            output = pd_diagnostics_out_file[ss]["var"].to_list()
            expected = ["x", "y", "mean(asmd)"]
            self.assertEqual(output, expected)

    def test_cli_empty_input(self):
        """Test CLI behavior with empty input data (header only)."""
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False
        ) as in_file:
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
                ]
            )
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()

            self.assertTrue(os.path.isfile(out_file))
            self.assertTrue(os.path.isfile(diagnostics_out_file))

    def test_cli_empty_input_keep_row(self):
        """Test CLI behavior with empty input data when using keep_row and batch_columns."""
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False
        ) as in_file:
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

    def test_cli_sep_works(self):
        """Test CLI functionality with custom output file separators."""
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False
        ) as in_file:
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

            # the original file had 4000 rows (1k target and 3k sample)
            self.assertEqual(pd_in_file.shape, (4000, 6))
            # the cli output includes only the panel (NOT the target)
            #   it also includes all the original columns
            self.assertEqual(pd_out_file.shape, (3000, 6))
            self.assertEqual(pd_out_file.is_respondent.mean(), 1)
            # the diagnostics file shows it was calculated on all 3k panelists
            ss = pd_diagnostics_out_file.eval(
                "(metric == 'size') & (var == 'sample_obs')"
            )
            self.assertEqual(int(pd_diagnostics_out_file[ss]["val"].iloc[0]), 3000)

    def test_cli_sep_input_works(self):
        """Test CLI functionality with custom input file separators (TSV)."""
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            "w", suffix=".tsv", delete=False
        ) as in_file:
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

            # the original file had 4000 rows (1k target and 3k sample)
            self.assertEqual(pd_in_file.shape, (4000, 6))
            # the cli output includes only the panel (NOT the target)
            #   it also includes all the original columns
            self.assertEqual(pd_out_file.shape, (3000, 6))
            self.assertEqual(pd_out_file.is_respondent.mean(), 1)
            # the diagnostics file shows it was calculated on all 3k panelists
            ss = pd_diagnostics_out_file.eval(
                "(metric == 'size') & (var == 'sample_obs')"
            )
            self.assertEqual(int(pd_diagnostics_out_file[ss]["val"].iloc[0]), 3000)

    def test_cli_short_arg_names_works(self):
        """
        Test CLI backward compatibility with partial argument names.

        Some users used only partial arg names for their pipelines.
        This test verifies new arguments would still be backward compatible.
        """
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False
        ) as in_file:
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

            # the original file had 4000 rows (1k target and 3k sample)
            self.assertEqual(pd_in_file.shape, (4000, 6))
            # the cli output includes only the panel (NOT the target)
            #   it also includes all the original columns
            self.assertEqual(pd_out_file.shape, (3000, 6))
            self.assertEqual(pd_out_file.is_respondent.mean(), 1)
            # the diagnostics file shows it was calculated on all 3k panelists
            ss = pd_diagnostics_out_file.eval(
                "(metric == 'size') & (var == 'sample_obs')"
            )
            self.assertEqual(int(pd_diagnostics_out_file[ss]["val"].iloc[0]), 3000)

    def test_method_works(self):
        """Test CLI functionality with different weighting methods (CBPS and IPW)."""
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

        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False
        ) as input_file:
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

    def test_method_works_with_rake(self):
        """Test CLI functionality with raking weighting method."""
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

        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False
        ) as input_file:
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

    def test_one_hot_encoding_works(self):
        """Test CLI one-hot encoding parameter with various boolean values."""
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            "w", suffix=".tsv", delete=False
        ) as in_file:
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

    def test_transformations_works(self):
        """Test CLI functionality with different transformation options (None and default)."""
        input_dataset = _create_sample_and_target_data()

        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False
        ) as input_file:
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
                        "C(age, one_hot_encoding_greater_2)[(0.00264, 10.925]]",
                        "C(age, one_hot_encoding_greater_2)[(10.925, 20.624]]",
                        "C(age, one_hot_encoding_greater_2)[(20.624, 30.985]]",
                        "C(age, one_hot_encoding_greater_2)[(30.985, 41.204]]",
                        "C(age, one_hot_encoding_greater_2)[(41.204, 51.335]]",
                        "C(age, one_hot_encoding_greater_2)[(51.335, 61.535]]",
                        "C(age, one_hot_encoding_greater_2)[(61.535, 71.696]]",
                        "C(age, one_hot_encoding_greater_2)[(71.696, 80.08]]",
                        "C(age, one_hot_encoding_greater_2)[(80.08, 89.446]]",
                        "C(age, one_hot_encoding_greater_2)[(89.446, 99.992]]",
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
                        "C(age, one_hot_encoding_greater_2)[(0.00264, 10.925]]",
                        "C(age, one_hot_encoding_greater_2)[(10.925, 20.624]]",
                        "C(age, one_hot_encoding_greater_2)[(20.624, 30.985]]",
                        "C(age, one_hot_encoding_greater_2)[(30.985, 41.204]]",
                        "C(age, one_hot_encoding_greater_2)[(41.204, 51.335]]",
                        "C(age, one_hot_encoding_greater_2)[(51.335, 61.535]]",
                        "C(age, one_hot_encoding_greater_2)[(61.535, 71.696]]",
                        "C(age, one_hot_encoding_greater_2)[(71.696, 80.08]]",
                        "C(age, one_hot_encoding_greater_2)[(80.08, 89.446]]",
                        "C(age, one_hot_encoding_greater_2)[(89.446, 99.992]]",
                        "C(gender, one_hot_encoding_greater_2)[(0.999, 2.0]]",
                        "C(gender, one_hot_encoding_greater_2)[(2.0, 3.0]]",
                        "C(gender, one_hot_encoding_greater_2)[(3.0, 4.0]]",
                    ]
                ),
            )

    def test_formula_works(self):
        """Test CLI functionality with custom formula specifications."""
        input_dataset = _create_sample_and_target_data()

        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False
        ) as input_file:
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

    def test_cli_return_df_with_original_dtypes(self):
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

    def test_cli_standardize_types(self):
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
