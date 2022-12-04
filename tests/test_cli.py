# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

import os.path

import tempfile

import balance.testutil

import numpy as np
import pandas as pd

from balance.cli import _float_or_none, BalanceCLI, make_parser


class TestCli(
    balance.testutil.BalanceTestCase,
):
    def test_cli_help(self):
        #  Just make sure it doesn't fail
        try:
            parser = make_parser()
            args = parser.parse_args(["--help"])
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()
        except SystemExit as e:
            if e.code == 0:
                pass
            else:
                raise e

    def test_cli_float_or_none(self):
        self.assertEqual(_float_or_none(None), None)
        self.assertEqual(_float_or_none("None"), None)
        self.assertEqual(_float_or_none("13.37"), 13.37)

    def test_cli_succeed_on_weighting_failure(self):
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
                ]
            )
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            self.assertRaisesRegex(Exception, "glmnet", cli.main)

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
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False
        ) as in_file:

            in_contents = (
                "x,y,is_respondent,id,weight\n"
                + ("a,b,1,1,1\n" * 1000)
                + ("c,b,0,1,1\n" * 1000)
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
        import os

        # TODO: remove imports (already imported in the beginning of the file)
        import tempfile

        import pandas as pd

        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False
        ) as in_file:
            # # create temp folder / file
            # in_file = tempfile.NamedTemporaryFile(
            #             "w", suffix=".csv", delete=False
            #         )
            # temp_dir = tempfile.TemporaryDirectory().name

            in_contents = (
                "x,y,z,is_respondent,id,weight\n"
                + ("a,b,g,1,1,1\n" * 1000)
                + ("c,b,g,1,1,1\n" * 1000)
                + ("c,b,g,0,1,1\n" * 1000)
                + ("a,d,n,1,2,1\n" * 1000)
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
            self.assertEqual(int(pd_diagnostics_out_file[ss]["val"]), 2000)

            # verify we get diagnostics only for x and y, and not z
            ss = pd_diagnostics_out_file.eval("metric == 'covar_main_asmd_adjusted'")
            output = pd_diagnostics_out_file[ss]["var"].to_list()
            expected = ["x", "y", "mean(asmd)"]
            self.assertEqual(output, expected)

        # # remove temp folder / file
        # os.unlink(in_file.name)
        # import shutil
        # shutil.rmtree(temp_dir)

    def test_cli_empty_input(self):
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
        import os
        import tempfile

        import pandas as pd

        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False
        ) as in_file:
            # # create temp folder / file
            # in_file = tempfile.NamedTemporaryFile(
            #             "w", suffix=".csv", delete=False
            #         )
            # temp_dir = tempfile.TemporaryDirectory().name

            in_contents = (
                "x,y,z,is_respondent,id,weight\n"
                + ("a,b,g,1,1,1\n" * 1000)
                + ("c,b,g,1,1,1\n" * 1000)
                + ("c,b,g,0,1,1\n" * 1000)
                + ("a,d,n,1,2,1\n" * 1000)
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
            self.assertEqual(int(pd_diagnostics_out_file[ss]["val"]), 3000)

        # # remove temp folder / file
        # os.unlink(in_file.name)
        # import shutil
        # shutil.rmtree(temp_dir)

    def test_cli_sep_input_works(self):
        import os
        import tempfile

        import pandas as pd

        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            "w", suffix=".tsv", delete=False
        ) as in_file:

            in_contents = (
                "x\ty\tz\tis_respondent\tid\tweight\n"
                + ("a\tb\tg\t1\t1\t1\n" * 1000)
                + ("c\tb\tg\t1\t1\t1\n" * 1000)
                + ("c\tb\tg\t0\t1\t1\n" * 1000)
                + ("a\td\tn\t1\t2\t1\n" * 1000)
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
            self.assertEqual(int(pd_diagnostics_out_file[ss]["val"]), 3000)

    def test_cli_short_arg_names_works(self):
        # Some users used only partial arg names for their pipelines
        # This is not good practice, but we'd like to not break these pipelines.
        # Hence, this test verifies new arguments would still be backward compatible
        import os
        import tempfile

        import pandas as pd

        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False
        ) as in_file:

            in_contents = (
                "x,y,z,is_respondent,id,weight\n"
                + ("a,b,g,1,1,1\n" * 1000)
                + ("c,b,g,1,1,1\n" * 1000)
                + ("c,b,g,0,1,1\n" * 1000)
                + ("a,d,n,1,2,1\n" * 1000)
            )
            in_file.write(in_contents)
            in_file.close()
            # pd.read_csv(in_file)

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
            self.assertEqual(int(pd_diagnostics_out_file[ss]["val"]), 3000)

    # TODO: Add tests for max_de
    # TODO: Add tests for weight_trimming_mean_ratio

    def test_method_works(self):
        # TODO: ideally we'll have the example outside, and a different function for each of the methods (ipw, cbps, raking)
        np.random.seed(2021)
        n_sample = 1000
        n_target = 2000
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

    def test_one_hot_encoding_works(self):
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
        # TODO: ideally we'll have the example outside
        np.random.seed(2021)
        n_sample = 1000
        n_target = 2000
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
