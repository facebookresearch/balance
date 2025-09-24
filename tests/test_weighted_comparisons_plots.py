# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import absolute_import, division, print_function, unicode_literals

from typing import Any, Dict

import balance.testutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from balance.stats_and_plots import weighted_comparisons_plots, weighted_stats


class Test_weighted_comparisons_plots(balance.testutil.BalanceTestCase):
    """
    Test suite for weighted_comparisons_plots module.

    This class contains comprehensive tests for plotting functions used in
    weighted statistical comparisons, including color palette generation,
    bar plots, distribution plots, and frequency table calculations.

    The tests cover both matplotlib/seaborn and plotly backends for
    visualization functions.
    """

    def _create_concentrated_weight_test_data(self) -> Dict[str, Any]:
        """Helper method to create test data with concentrated weights."""
        return {
            "df": pd.DataFrame(
                {"numeric_5": pd.Series([0, 0, 0, 0, 0, 1, 1, 2, 3, 4])}
            ),
            "weight": pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 5]),
        }

    def _create_basic_test_dataframe(self) -> pd.DataFrame:
        """Helper method to create basic test DataFrame for frequency table tests."""
        return pd.DataFrame({"a": list("abcd"), "b": list("bbcd")})

    def test_return_sample_palette_standard_combinations(self) -> None:
        """
        Test _return_sample_palette function with standard sample name combinations.

        This function should return appropriate color palettes for common
        sample comparison scenarios like self vs target, adjusted vs unadjusted, etc.
        """
        from balance.stats_and_plots.weighted_comparisons_plots import (
            _return_sample_palette,
        )

        # Test two-sample comparisons
        expected_self_target = {"self": "#de2d26cc", "target": "#9ecae1cc"}
        self.assertEqual(
            _return_sample_palette(["self", "target"]), expected_self_target
        )

        expected_self_unadjusted = {"self": "#34a53080", "unadjusted": "#de2d26cc"}
        self.assertEqual(
            _return_sample_palette(["self", "unadjusted"]), expected_self_unadjusted
        )

        # Test three-sample comparisons
        expected_three_samples = {
            "self": "#34a53080",
            "unadjusted": "#de2d26cc",
            "target": "#9ecae1cc",
        }
        self.assertEqual(
            _return_sample_palette(["self", "unadjusted", "target"]),
            expected_three_samples,
        )
        self.assertEqual(
            _return_sample_palette(["adjusted", "unadjusted", "target"]),
            {"adjusted": "#34a53080", "unadjusted": "#de2d26cc", "target": "#9ecae1cc"},
        )

        # Test non-standard sample names should return default palette
        self.assertEqual(_return_sample_palette(["cat", "dog"]), "muted")

    def test_plotly_marker_color_combinations(self) -> None:
        """
        Test _plotly_marker_color function for various sample types and style combinations.

        This function should return appropriate RGBA color strings for plotly markers
        based on sample name, whether it's a target sample, and color/line style.
        """
        from balance.stats_and_plots.weighted_comparisons_plots import (
            _plotly_marker_color,
        )

        # Test 'self' sample colors
        self.assertEqual(
            _plotly_marker_color("self", True, "color"), "rgba(222,45,38,0.8)"
        )
        self.assertEqual(
            _plotly_marker_color("self", False, "color"), "rgba(52,165,48,0.5)"
        )
        self.assertEqual(
            _plotly_marker_color("self", True, "line"), "rgba(222,45,38,1)"
        )
        self.assertEqual(
            _plotly_marker_color("self", False, "line"), "rgba(52,165,48,1)"
        )

        # Test 'target' sample colors
        self.assertEqual(
            _plotly_marker_color("target", True, "color"),  # pyre-ignore[6]
            "rgb(158,202,225,0.8)",
        )
        self.assertEqual(
            _plotly_marker_color("target", False, "color"),  # pyre-ignore[6]
            "rgb(158,202,225,0.8)",
        )
        self.assertEqual(
            _plotly_marker_color("target", True, "line"),  # pyre-ignore[6]
            "rgb(158,202,225,1)",
        )
        self.assertEqual(
            _plotly_marker_color("target", False, "line"),  # pyre-ignore[6]
            "rgb(158,202,225,1)",
        )

        # Test 'adjusted' sample colors
        self.assertEqual(
            _plotly_marker_color("adjusted", False, "color"), "rgba(52,165,48,0.5)"
        )
        self.assertEqual(
            _plotly_marker_color("adjusted", False, "line"), "rgba(52,165,48,1)"
        )

    def test_plot_bar_weighted_comparison(self) -> None:
        """
        Test plot_bar function for weighted bar plot generation.

        This test verifies that the plot_bar function correctly generates
        weighted bar plots comparing multiple samples with appropriate titles.
        """
        import matplotlib.pyplot as plt
        from balance.stats_and_plots.weighted_comparisons_plots import plot_bar

        # Create test data
        test_df = pd.DataFrame(
            {
                "group": ("a", "b", "c", "c"),
                "v1": (1, 2, 3, 4),
            }
        )

        # Set up matplotlib figure
        plt.figure(1)
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        # Generate the bar plot
        plot_bar(
            [
                {"df": test_df, "weight": pd.Series((1, 1, 1, 1))},
                {"df": test_df, "weight": pd.Series((2, 1, 1, 1))},
            ],
            names=["self", "target"],
            column="group",
            axis=ax,
            weighted=True,
        )

        # Verify the plot title contains expected text
        self.assertIn("barplot of covar ", ax.get_title())

    def test_plot_dist_kde_returns_matplotlib_axes(self) -> None:
        """
        Test that plot_dist with KDE returns a matplotlib Axes object.

        This test verifies that the plot_dist function properly returns
        matplotlib Axes objects when using KDE distribution plots with seaborn.
        Uses edge case data with concentrated weights to test robustness.
        """
        # Create test data with concentrated weights (edge case)
        test_data = self._create_concentrated_weight_test_data()

        # Generate KDE plot and get the axes type
        plot_axes = weighted_comparisons_plots.plot_dist(  # pyre-ignore[16]
            [test_data],
            dist_type="kde",
            numeric_n_values_threshold=0,
            weighted=False,
            library="seaborn",
            return_axes=True,
        )[0]

        # Verify that the returned object is a matplotlib Axes
        # NOTE: AxesSubplot class is created dynamically when invoked
        # See: https://stackoverflow.com/a/11690800/256662
        self.assertTrue(issubclass(type(plot_axes), plt.Axes))

    def test_relative_frequency_table_unweighted(self) -> None:
        """
        Test relative_frequency_table function with unweighted data.

        Verifies that the function correctly calculates proportions for
        categorical data without weights (equal weights assumed).
        """
        test_df = self._create_basic_test_dataframe()

        # Test column 'a' with uniform distribution
        expected_a = pd.DataFrame({"a": list("abcd"), "prop": (0.25,) * 4})
        result_a = weighted_stats.relative_frequency_table(test_df, "a")
        self.assertEqual(result_a, expected_a)

        # Test column 'b' with non-uniform distribution
        expected_b = pd.DataFrame({"b": list("bcd"), "prop": (0.5, 0.25, 0.25)})
        result_b = weighted_stats.relative_frequency_table(test_df, "b")
        self.assertEqual(result_b, expected_b)

    def test_relative_frequency_table_with_unit_weights(self) -> None:
        """
        Test relative_frequency_table function with explicit unit weights.

        Verifies that passing unit weights produces the same results as
        unweighted calculations.
        """
        test_df = self._create_basic_test_dataframe()
        unit_weights = pd.Series((1, 1, 1, 1))

        # Results with unit weights should match unweighted results
        expected_a = pd.DataFrame({"a": list("abcd"), "prop": (0.25,) * 4})
        result_a = weighted_stats.relative_frequency_table(test_df, "a", unit_weights)
        self.assertEqual(result_a, expected_a)

        expected_b = pd.DataFrame({"b": list("bcd"), "prop": (0.5, 0.25, 0.25)})
        result_b = weighted_stats.relative_frequency_table(test_df, "b", unit_weights)
        self.assertEqual(result_b, expected_b)

    def test_relative_frequency_table_with_custom_weights(self) -> None:
        """
        Test relative_frequency_table function with custom weight distributions.

        Verifies that the function correctly applies custom weights to
        calculate weighted proportions.
        """
        test_df = self._create_basic_test_dataframe()

        # Test with non-uniform weights
        custom_weights_1 = pd.Series((1, 2, 1, 1))
        expected_weighted_a = pd.DataFrame(
            {"a": list("abcd"), "prop": (0.2, 0.4, 0.2, 0.2)}
        )
        result_weighted_a = weighted_stats.relative_frequency_table(
            test_df, "a", custom_weights_1
        )
        self.assertEqual(result_weighted_a, expected_weighted_a)

        # Test with fractional weights
        custom_weights_2 = pd.Series((0.5, 0.5, 1, 1))
        expected_weighted_b = pd.DataFrame(
            {"b": list("bcd"), "prop": (1 / 3, 1 / 3, 1 / 3)}
        )
        result_weighted_b = weighted_stats.relative_frequency_table(
            test_df, "b", custom_weights_2
        )
        self.assertEqual(result_weighted_b, expected_weighted_b)

    def test_relative_frequency_table_error_handling(self) -> None:
        """
        Test relative_frequency_table function error handling for invalid inputs.

        Verifies that appropriate errors are raised for invalid weight parameters.
        """
        test_df = self._create_basic_test_dataframe()

        # Should raise TypeError for non-Series weights
        with self.assertRaisesRegex(TypeError, "must be a pandas Series"):
            weighted_stats.relative_frequency_table(test_df, "a", 1)  # pyre-ignore[6]

    def test_relative_frequency_table_with_dataframe_and_series_input(self) -> None:
        """
        Test relative_frequency_table function with DataFrame vs Series input consistency.

        Verifies that using a DataFrame column vs passing a Series directly
        produces identical results.
        """
        group_df = pd.DataFrame(
            {
                "group": ("a", "b", "c", "c"),
                "v1": (1, 2, 3, 4),
            }
        )
        weights = pd.Series((2, 1, 1, 1))
        expected_result = {
            "group": {0: "a", 1: "b", 2: "c"},
            "prop": {0: 0.4, 1: 0.2, 2: 0.4},
        }

        # Test with DataFrame and column name
        result_df = weighted_stats.relative_frequency_table(
            group_df, "group", weights
        ).to_dict()
        self.assertEqual(result_df, expected_result)

        # Test with Series directly
        result_series = weighted_stats.relative_frequency_table(
            df=group_df["group"], w=weights
        ).to_dict()
        self.assertEqual(result_series, expected_result)

    def test_naming_legend_transformations(self) -> None:
        """
        Test naming_legend function for proper legend name transformations.

        This function should transform sample names into appropriate legend labels
        based on the context of available samples (e.g., 'self' becomes 'adjusted'
        when target and unadjusted are present).
        """
        # Test three-sample scenario: self becomes 'adjusted'
        result_three_sample = weighted_comparisons_plots.naming_legend(
            "self", ["self", "target", "unadjusted"]
        )
        self.assertEqual(result_three_sample, "adjusted")

        # Test unadjusted becomes 'sample' in three-sample scenario
        result_unadjusted = weighted_comparisons_plots.naming_legend(
            "unadjusted", ["self", "target", "unadjusted"]
        )
        self.assertEqual(result_unadjusted, "sample")

        # Test two-sample scenario: self becomes 'sample'
        result_two_sample = weighted_comparisons_plots.naming_legend(
            "self", ["self", "target"]
        )
        self.assertEqual(result_two_sample, "sample")

        # Test non-standard names remain unchanged
        result_other = weighted_comparisons_plots.naming_legend(
            "other_name", ["self", "target"]
        )
        self.assertEqual(result_other, "other_name")

    def test_seaborn_plot_dist_returns_matplotlib_axes(self) -> None:
        """
        Test seaborn_plot_dist function returns matplotlib Axes for all distribution types.

        Verifies that the seaborn plotting function correctly returns matplotlib Axes
        objects for all supported distribution types: hist, kde, qq, and ecdf.
        """
        test_data = self._create_concentrated_weight_test_data()

        # Test all distribution types return matplotlib Axes
        distribution_types = ("hist", "kde", "qq", "ecdf")
        axes_types = []
        for dist_type in distribution_types:
            plot_result = weighted_comparisons_plots.seaborn_plot_dist(
                [test_data],
                names=["test"],
                dist_type=dist_type,  # pyre-ignore[6]
                return_axes=True,
            )
            axes_types.append(type(plot_result[0]))

        # Verify all returned objects are matplotlib Axes subclasses
        # NOTE: AxesSubplot class is created dynamically when invoked
        # See: https://stackoverflow.com/a/11690800/256662
        for axes_type in axes_types:
            self.assertTrue(issubclass(axes_type, matplotlib.axes.Axes))

    def test_plot_dist_plotly_functionality(self) -> None:
        """
        Test plot_dist function with plotly backend for various scenarios.

        Verifies that the plot_dist function correctly handles plotly plotting,
        including return value behavior and figure dictionary generation.
        Tests both normal operation and error handling for invalid parameters.
        """
        import plotly.graph_objs as go
        from balance.stats_and_plots.weighted_comparisons_plots import plot_dist
        from numpy import random

        # Create test datasets with varied weights
        random.seed(96483)

        test_df = pd.DataFrame(
            {
                "v1": random.randint(11111, 11115, size=100).astype(str),
                "v2": random.normal(size=100),
                "v3": random.uniform(size=100),
            }
        ).sort_values(by=["v2"])

        test_datasets = [
            {"df": test_df, "weight": pd.Series(np.ones(100))},
            {"df": test_df, "weight": pd.Series(np.ones(99).tolist() + [1000])},
            {"df": test_df, "weight": pd.Series(np.ones(100))},
        ]

        # Test that plot_it=False returns None by default
        result = plot_dist(
            test_datasets,
            names=["self", "unadjusted", "target"],
            library="plotly",
            plot_it=False,
        )
        self.assertTrue(result is None)

        # Test return_dict_of_figures functionality
        dict_of_figures = plot_dist(
            test_datasets,
            names=["self", "unadjusted", "target"],
            library="plotly",
            plot_it=False,
            return_dict_of_figures=True,
        )

        # Verify dictionary structure and contents
        self.assertEqual(type(dict_of_figures), dict)
        self.assertEqual(
            sorted(dict_of_figures.keys()),  # pyre-ignore[16]
            ["v1", "v2", "v3"],
        )
        self.assertEqual(type(dict_of_figures["v1"]), go.Figure)

        # Test error handling for invalid library parameter
        with self.assertRaisesRegex(ValueError, "library must be either*"):
            plot_dist(
                test_datasets,
                names=["self", "unadjusted", "target"],
                library="ploting_library_which_is_not_plotly_or_seaborn",  # pyre-ignore[6]
            )
