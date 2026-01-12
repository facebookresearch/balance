# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List

import balance.testutil
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from balance.stats_and_plots import weighted_comparisons_plots, weighted_stats
from balance.stats_and_plots.weighted_comparisons_plots import DataFrameWithWeight
from balance.util import _verify_value_type


class Test_weighted_comparisons_plots(balance.testutil.BalanceTestCase):
    """
    Test suite for weighted_comparisons_plots module.

    This class contains comprehensive tests for plotting functions used in
    weighted statistical comparisons, including color palette generation,
    bar plots, distribution plots, and frequency table calculations.

    The tests cover both matplotlib/seaborn and plotly backends for
    visualization functions.
    """

    def tearDown(self) -> None:
        """Clean up matplotlib figures after each test to avoid memory warnings."""
        plt.close("all")
        super().tearDown()

    def _create_concentrated_weight_test_data(self) -> DataFrameWithWeight:
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
            _plotly_marker_color("target", True, "color"),
            "rgb(158,202,225,0.8)",
        )
        self.assertEqual(
            _plotly_marker_color("target", False, "color"),
            "rgb(158,202,225,0.8)",
        )
        self.assertEqual(
            _plotly_marker_color("target", True, "line"),
            "rgb(158,202,225,1)",
        )
        self.assertEqual(
            _plotly_marker_color("target", False, "line"),
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

        # Create test data with proper type
        test_data: List[DataFrameWithWeight] = [
            {"df": test_df, "weight": pd.Series((1, 1, 1, 1))},
            {"df": test_df, "weight": pd.Series((2, 1, 1, 1))},
        ]

        # Generate the bar plot
        plot_bar(
            test_data,
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
        test_data_list: List[DataFrameWithWeight] = [test_data]

        # Generate KDE plot and get the axes type
        plot_result = weighted_comparisons_plots.plot_dist(
            test_data_list,
            dist_type="kde",
            numeric_n_values_threshold=0,
            weighted=False,
            library="seaborn",
            return_axes=True,
        )
        plot_result = _verify_value_type(plot_result, list)
        plot_axes = plot_result[0]

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
        # Test error handling for invalid weight parameters
        with self.assertRaisesRegex(TypeError, "must be a pandas Series"):
            weighted_stats.relative_frequency_table(test_df, "a", 1)  # type: ignore[arg-type]

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
        test_data_list: List[DataFrameWithWeight] = [test_data]

        # Test all distribution types return matplotlib Axes
        from typing import cast, Literal

        distribution_types = ("hist", "kde", "qq", "ecdf")
        axes_types = []
        for dist_type_str in distribution_types:
            dist_type = cast(Literal["hist", "kde", "qq", "ecdf"], dist_type_str)
            plot_result = weighted_comparisons_plots.seaborn_plot_dist(
                test_data_list,
                names=["test"],
                dist_type=dist_type,
                return_axes=True,
            )
            plot_result = _verify_value_type(plot_result, list)
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

        test_datasets: List[DataFrameWithWeight] = [
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
        dict_of_figures = _verify_value_type(dict_of_figures, dict)
        self.assertEqual(
            sorted(dict_of_figures.keys()),
            ["v1", "v2", "v3"],
        )
        self.assertEqual(type(dict_of_figures["v1"]), go.Figure)

        # Test error handling for invalid library parameter
        with self.assertRaisesRegex(ValueError, "library must be either*"):
            plot_dist(
                test_datasets,
                names=["self", "unadjusted", "target"],
                library="ploting_library_which_is_not_plotly_or_seaborn",  # type: ignore[arg-type]
            )

    def test_plotly_marker_color_invalid_color_type(self) -> None:
        """
        Test _plotly_marker_color function error handling for invalid color_type.

        This test verifies that the function raises ValueError when an invalid
        color_type parameter is provided (not 'color' or 'line').
        """
        from balance.stats_and_plots.weighted_comparisons_plots import (
            _plotly_marker_color,
        )

        # Test that invalid color_type raises ValueError
        with self.assertRaisesRegex(ValueError, "Invalid value for"):
            _plotly_marker_color("self", True, "invalid_type")  # type: ignore[arg-type]

    def test_plot_bar_with_parameters_and_edge_cases(self) -> None:
        """
        Test plot_bar function with various parameters and edge cases.

        This test verifies ylim, custom title parameters, and NaN weight handling.
        """
        from balance.stats_and_plots.weighted_comparisons_plots import plot_bar

        test_df = pd.DataFrame(
            {
                "group": ("a", "b", "c", "c"),
                "v1": (1, 2, 3, 4),
            }
        )

        # Test 1: ylim parameter
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        test_data: List[DataFrameWithWeight] = [
            {"df": test_df, "weight": pd.Series((1, 1, 1, 1))},
            {"df": test_df, "weight": pd.Series((2, 1, 1, 1))},
        ]
        plot_bar(
            test_data,
            names=["self", "target"],
            column="group",
            axis=ax,
            weighted=True,
            ylim=(0, 1),
        )
        ylim = ax.get_ylim()
        self.assertEqual(ylim[0], 0)
        self.assertEqual(ylim[1], 1)

        # Test 2: custom title
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        custom_title = "Custom Bar Plot Title"
        plot_bar(
            [{"df": test_df, "weight": pd.Series((1, 1, 1, 1))}],
            names=["self"],
            column="group",
            axis=ax,
            weighted=True,
            title=custom_title,
        )
        self.assertEqual(ax.get_title(), custom_title)

        # Test 3: NaN weights handling
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        test_data_nan: List[DataFrameWithWeight] = [
            {"df": test_df, "weight": pd.Series((1, 1, 1, np.nan))},
            {"df": test_df, "weight": pd.Series((2, 1, 1, np.nan))},
        ]
        plot_bar(
            test_data_nan,
            names=["self", "target"],
            column="group",
            axis=ax,
            weighted=True,
        )
        self.assertIsNotNone(ax.get_title())

    def test_plot_bar_unweighted(self) -> None:
        """
        Test plot_bar function with weighted=False parameter.

        This test verifies that the plot_bar function correctly generates
        unweighted bar plots when weighted=False is specified, and that
        the bar values differ appropriately between weighted and unweighted modes.

        The test uses dramatically different weights between datasets to ensure
        a clear distinction between weighted and unweighted behavior.
        """
        from balance.stats_and_plots.weighted_comparisons_plots import plot_bar

        # Create test data where weights would affect proportions if applied
        # Dataset has groups: a(1), b(1), c(2) - so unweighted props are 0.25, 0.25, 0.5
        test_df = pd.DataFrame(
            {
                "group": ("a", "b", "c", "c"),
                "v1": (1, 2, 3, 4),
            }
        )

        # Use dramatically different weights for the two datasets
        # First dataset: uniform weights (1,1,1,1) -> props: a=0.25, b=0.25, c=0.5
        # Second dataset: heavily weight first element (100,1,1,1) -> props: a≈0.97, b≈0.01, c≈0.02
        # With weighted=True, these should produce very different bar heights
        # With weighted=False, these should produce identical bar heights
        test_data: List[DataFrameWithWeight] = [
            {"df": test_df, "weight": pd.Series((1, 1, 1, 1))},
            {"df": test_df, "weight": pd.Series((100, 1, 1, 1))},
        ]

        # Test 1: Generate weighted bar plot (weighted=True) first to establish baseline
        # This proves that the weights DO make a difference when applied
        fig, ax_weighted = plt.subplots(1, 1, figsize=(7.2, 7.2))
        plot_bar(
            test_data,
            names=["self", "target"],
            column="group",
            axis=ax_weighted,
            weighted=True,
        )

        # Extract bar heights from weighted plot
        weighted_containers = ax_weighted.containers
        self.assertEqual(len(weighted_containers), 2)

        weighted_heights_self = [bar.get_height() for bar in weighted_containers[0]]
        weighted_heights_target = [bar.get_height() for bar in weighted_containers[1]]

        # When weighted=True, the bar heights should differ significantly between datasets
        # because the weights are very different (1,1,1,1 vs 100,1,1,1)
        # Calculate the maximum difference between corresponding bars
        max_height_diff_weighted = max(
            abs(h_self - h_target)
            for h_self, h_target in zip(weighted_heights_self, weighted_heights_target)
        )
        # With weights of 100,1,1,1 vs 1,1,1,1, the difference should be substantial (>0.5)
        self.assertGreater(
            max_height_diff_weighted,
            0.5,
            "With weighted=True and very different weights, bar heights should differ significantly",
        )

        # Test 2: Generate unweighted bar plot (weighted=False) with same data
        fig, ax_unweighted = plt.subplots(1, 1, figsize=(7.2, 7.2))
        plot_bar(
            test_data,
            names=["self", "target"],
            column="group",
            axis=ax_unweighted,
            weighted=False,
        )

        # Verify plot was created
        self.assertIn("barplot of covar ", ax_unweighted.get_title())

        # Extract bar heights from unweighted plot
        unweighted_containers = ax_unweighted.containers
        self.assertEqual(len(unweighted_containers), 2)

        unweighted_heights_self = [bar.get_height() for bar in unweighted_containers[0]]
        unweighted_heights_target = [
            bar.get_height() for bar in unweighted_containers[1]
        ]

        # When weighted=False, both datasets should have identical bar heights
        # (weights are ignored, so same DataFrame = same proportions)
        for h_self, h_target in zip(unweighted_heights_self, unweighted_heights_target):
            self.assertAlmostEqual(h_self, h_target, places=5)

        # Additional verification: the unweighted heights should match the expected
        # unweighted proportions (0.25, 0.25, 0.5 for groups a, b, c)
        # This ensures we're actually computing unweighted proportions
        self.assertAlmostEqual(unweighted_heights_self[0], 0.25, places=2)  # group 'a'
        self.assertAlmostEqual(unweighted_heights_self[1], 0.25, places=2)  # group 'b'
        self.assertAlmostEqual(unweighted_heights_self[2], 0.50, places=2)  # group 'c'

    def test_set_xy_axes_to_use_the_same_lim(self) -> None:
        """
        Test set_xy_axes_to_use_the_same_lim function for axis limit synchronization.

        This test verifies that the function correctly sets x and y axes to use
        the same limits, ensuring the limits encompass both the original x and y
        data ranges.
        """
        from balance.stats_and_plots.weighted_comparisons_plots import (
            set_xy_axes_to_use_the_same_lim,
        )

        # Test with positive values where x and y have different ranges
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        ax.scatter(x=[1, 2, 3], y=[3, 4, 5])
        # Before synchronization, get the original limits
        original_xlim = ax.get_xlim()
        original_ylim = ax.get_ylim()
        # Apply synchronization
        set_xy_axes_to_use_the_same_lim(ax)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Verify x and y limits are now equal
        self.assertAlmostEqual(xlim[0], ylim[0], places=5)
        self.assertAlmostEqual(xlim[1], ylim[1], places=5)
        # Verify the new limits encompass both original ranges
        self.assertLessEqual(xlim[0], min(original_xlim[0], original_ylim[0]))
        self.assertGreaterEqual(xlim[1], max(original_xlim[1], original_ylim[1]))

        # Test with negative values and wider range difference
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        ax.scatter(x=[-5, -2, 0], y=[1, 3, 5])
        original_xlim = ax.get_xlim()
        original_ylim = ax.get_ylim()
        set_xy_axes_to_use_the_same_lim(ax)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Verify x and y limits are now equal
        self.assertAlmostEqual(xlim[0], ylim[0], places=5)
        self.assertAlmostEqual(xlim[1], ylim[1], places=5)
        # Verify the synchronized limits cover the full range of both axes
        # The new min should be <= the minimum of both original mins
        # The new max should be >= the maximum of both original maxes
        self.assertLessEqual(xlim[0], min(original_xlim[0], original_ylim[0]))
        self.assertGreaterEqual(xlim[1], max(original_xlim[1], original_ylim[1]))

        # Test with identical ranges (edge case)
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        ax.scatter(x=[0, 1, 2], y=[0, 1, 2])
        set_xy_axes_to_use_the_same_lim(ax)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        self.assertAlmostEqual(xlim[0], ylim[0], places=5)
        self.assertAlmostEqual(xlim[1], ylim[1], places=5)

    def test_plot_hist_kde_with_hist_type(self) -> None:
        """
        Test plot_hist_kde function with histogram distribution type.

        This test verifies that the function correctly generates histogram
        plots for weighted comparisons, comparing actual histogram bar heights
        between weighted and unweighted modes.
        """
        from balance.stats_and_plots.weighted_comparisons_plots import plot_hist_kde

        # Create test data with values that will produce clear histogram bins
        # Values: 1(1x), 2(3x), 3(1x), 4(1x), 5(2x), 7(1x), 8(1x), 9(4x)
        test_df = pd.DataFrame({"v1": [1, 2, 2, 2, 3, 4, 5, 5, 7, 8, 9, 9, 9, 9]})

        # Use dramatically different weights for the two datasets
        # First dataset: uniform weights
        # Second dataset: heavily weight the first few elements
        uniform_weights = pd.Series(np.ones(len(test_df)))
        concentrated_weights = pd.Series([100] * 3 + [1] * (len(test_df) - 3))

        test_data_weighted: List[DataFrameWithWeight] = [
            {"df": test_df, "weight": uniform_weights},
            {"df": test_df, "weight": concentrated_weights},
        ]

        # Test 1: Generate weighted histogram plot (weighted=True)
        fig, ax_weighted = plt.subplots(1, 1, figsize=(7.2, 7.2))
        plot_hist_kde(
            test_data_weighted,
            names=["self", "target"],
            column="v1",
            axis=ax_weighted,
            weighted=True,
            dist_type="hist",
        )
        self.assertIn("distribution plot of covar ", ax_weighted.get_title())

        # Extract histogram bar heights from weighted plot
        # Each dataset creates a set of patches (bars) in the plot
        weighted_patches = ax_weighted.patches
        num_bars_per_dataset = len(weighted_patches) // 2
        weighted_heights_self = [
            p.get_height() for p in weighted_patches[:num_bars_per_dataset]
        ]
        weighted_heights_target = [
            p.get_height() for p in weighted_patches[num_bars_per_dataset:]
        ]

        # When weighted=True, the bar heights should differ between datasets
        # because the weights are very different
        if len(weighted_heights_self) > 0 and len(weighted_heights_target) > 0:
            max_height_diff_weighted = max(
                abs(h_self - h_target)
                for h_self, h_target in zip(
                    weighted_heights_self, weighted_heights_target
                )
            )
            self.assertGreater(
                max_height_diff_weighted,
                0.01,
                "With weighted=True and different weights, histogram heights should differ",
            )

        # Test 2: Generate unweighted histogram plot (weighted=False) with same data
        fig, ax_unweighted = plt.subplots(1, 1, figsize=(7.2, 7.2))
        plot_hist_kde(
            test_data_weighted,
            names=["self", "target"],
            column="v1",
            axis=ax_unweighted,
            weighted=False,
            dist_type="hist",
        )

        # Extract histogram bar heights from unweighted plot
        unweighted_patches = ax_unweighted.patches
        num_bars_per_dataset_unweighted = len(unweighted_patches) // 2
        unweighted_heights_self = [
            p.get_height() for p in unweighted_patches[:num_bars_per_dataset_unweighted]
        ]
        unweighted_heights_target = [
            p.get_height() for p in unweighted_patches[num_bars_per_dataset_unweighted:]
        ]

        # When weighted=False, both datasets should have identical bar heights
        # (weights are ignored, so same DataFrame = same histogram)
        for h_self, h_target in zip(unweighted_heights_self, unweighted_heights_target):
            self.assertAlmostEqual(h_self, h_target, places=5)

        # Test 3: custom title
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        test_df_small = pd.DataFrame({"v1": [1, 2, 3, 4, 5]})
        custom_title = "Custom Distribution Title"
        plot_hist_kde(
            [{"df": test_df_small, "weight": pd.Series(np.ones(len(test_df_small)))}],
            names=["self"],
            column="v1",
            axis=ax,
            weighted=True,
            dist_type="hist",
            title=custom_title,
        )
        self.assertEqual(ax.get_title(), custom_title)

    def test_plot_hist_kde_with_kde_type(self) -> None:
        """
        Test plot_hist_kde function with KDE distribution type.

        This test verifies that the function correctly generates kernel density
        estimate plots for weighted comparisons, comparing actual KDE line data
        between weighted and unweighted modes.
        """
        from balance.stats_and_plots.weighted_comparisons_plots import plot_hist_kde

        # Create test data with values that will produce clear KDE curves
        test_df = pd.DataFrame({"v1": [1, 2, 2, 2, 3, 4, 5, 5, 7, 8, 9, 9, 9, 9]})

        # Use dramatically different weights for the two datasets
        # First dataset: uniform weights
        # Second dataset: heavily weight the first few elements
        uniform_weights = pd.Series(np.ones(len(test_df)))
        concentrated_weights = pd.Series([100] * 3 + [1] * (len(test_df) - 3))

        test_data_weighted: List[DataFrameWithWeight] = [
            {"df": test_df, "weight": uniform_weights},
            {"df": test_df, "weight": concentrated_weights},
        ]

        # Test 1: Generate weighted KDE plot (weighted=True)
        fig, ax_weighted = plt.subplots(1, 1, figsize=(7.2, 7.2))
        plot_hist_kde(
            test_data_weighted,
            names=["self", "target"],
            column="v1",
            axis=ax_weighted,
            weighted=True,
            dist_type="kde",
        )
        self.assertIn("distribution plot of covar ", ax_weighted.get_title())

        # Extract KDE line data from weighted plot
        weighted_lines = ax_weighted.get_lines()
        if len(weighted_lines) >= 2:
            kde_y_values_self_weighted = weighted_lines[0].get_ydata()
            kde_y_values_target_weighted = weighted_lines[1].get_ydata()

            # When weighted=True, the KDE curves should differ between datasets
            # because the weights are very different
            max_y_diff_weighted = max(
                abs(y_self - y_target)
                for y_self, y_target in zip(
                    kde_y_values_self_weighted, kde_y_values_target_weighted
                )
            )
            self.assertGreater(
                max_y_diff_weighted,
                0.01,
                "With weighted=True and different weights, KDE curves should differ",
            )

        # Test 2: Generate unweighted KDE plot (weighted=False) with same data
        fig, ax_unweighted = plt.subplots(1, 1, figsize=(7.2, 7.2))
        plot_hist_kde(
            test_data_weighted,
            names=["self", "target"],
            column="v1",
            axis=ax_unweighted,
            weighted=False,
            dist_type="kde",
        )

        # Extract KDE line data from unweighted plot
        unweighted_lines = ax_unweighted.get_lines()
        if len(unweighted_lines) >= 2:
            kde_y_values_self_unweighted = unweighted_lines[0].get_ydata()
            kde_y_values_target_unweighted = unweighted_lines[1].get_ydata()

            # When weighted=False, both datasets should have identical KDE curves
            # (weights are ignored, so same DataFrame = same KDE)
            for y_self, y_target in zip(
                kde_y_values_self_unweighted, kde_y_values_target_unweighted
            ):
                self.assertAlmostEqual(y_self, y_target, places=5)

    def test_plot_hist_kde_with_ecdf_type(self) -> None:
        """
        Test plot_hist_kde function with ECDF distribution type.

        This test verifies that the function correctly generates empirical
        cumulative distribution function plots, comparing actual ECDF line data
        between weighted and unweighted modes.
        """
        from balance.stats_and_plots.weighted_comparisons_plots import plot_hist_kde

        # Create test data with values that will produce clear ECDF curves
        test_df = pd.DataFrame({"v1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        # Use dramatically different weights for the two datasets
        # First dataset: uniform weights
        # Second dataset: heavily weight the first few elements
        uniform_weights = pd.Series(np.ones(len(test_df)))
        concentrated_weights = pd.Series([100] * 3 + [1] * (len(test_df) - 3))

        test_data_weighted: List[DataFrameWithWeight] = [
            {"df": test_df, "weight": uniform_weights},
            {"df": test_df, "weight": concentrated_weights},
        ]

        # Test 1: Generate weighted ECDF plot (weighted=True)
        fig, ax_weighted = plt.subplots(1, 1, figsize=(7.2, 7.2))
        plot_hist_kde(
            test_data_weighted,
            names=["self", "target"],
            column="v1",
            axis=ax_weighted,
            weighted=True,
            dist_type="ecdf",
        )
        self.assertIn("distribution plot of covar ", ax_weighted.get_title())

        # Extract ECDF line data from weighted plot
        weighted_lines = ax_weighted.get_lines()
        if len(weighted_lines) >= 2:
            ecdf_y_values_self_weighted = weighted_lines[0].get_ydata()
            ecdf_y_values_target_weighted = weighted_lines[1].get_ydata()

            # When weighted=True, the ECDF curves should differ between datasets
            # because the weights are very different
            max_y_diff_weighted = max(
                abs(y_self - y_target)
                for y_self, y_target in zip(
                    ecdf_y_values_self_weighted, ecdf_y_values_target_weighted
                )
            )
            self.assertGreater(
                max_y_diff_weighted,
                0.01,
                "With weighted=True and different weights, ECDF curves should differ",
            )

        # Test 2: Generate unweighted ECDF plot (weighted=False) with same data
        fig, ax_unweighted = plt.subplots(1, 1, figsize=(7.2, 7.2))
        plot_hist_kde(
            test_data_weighted,
            names=["self", "target"],
            column="v1",
            axis=ax_unweighted,
            weighted=False,
            dist_type="ecdf",
        )

        # Extract ECDF line data from unweighted plot
        unweighted_lines = ax_unweighted.get_lines()
        if len(unweighted_lines) >= 2:
            ecdf_y_values_self_unweighted = unweighted_lines[0].get_ydata()
            ecdf_y_values_target_unweighted = unweighted_lines[1].get_ydata()

            # When weighted=False, both datasets should have identical ECDF curves
            # (weights are ignored, so same DataFrame = same ECDF)
            for y_self, y_target in zip(
                ecdf_y_values_self_unweighted, ecdf_y_values_target_unweighted
            ):
                self.assertAlmostEqual(y_self, y_target, places=5)

    def test_plot_hist_kde_unweighted(self) -> None:
        """
        Test plot_hist_kde function with weighted=False parameter.

        This test verifies that the function correctly generates unweighted
        distribution plots.
        """
        from balance.stats_and_plots.weighted_comparisons_plots import plot_hist_kde

        test_df = pd.DataFrame({"v1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        test_data: List[DataFrameWithWeight] = [
            {"df": test_df, "weight": pd.Series(np.ones(len(test_df)) * 100)},
        ]

        # Generate unweighted plot (weights should be ignored)
        plot_hist_kde(
            test_data,
            names=["self"],
            column="v1",
            axis=ax,
            weighted=False,
            dist_type="kde",
        )

        # Verify plot was created
        self.assertIsNotNone(ax.get_title())

    def test_plot_hist_kde_with_none_weights(self) -> None:
        """
        Test plot_hist_kde function when weight is None.

        This test verifies that the function correctly handles None weights
        by defaulting to equal weights.
        """
        from balance.stats_and_plots.weighted_comparisons_plots import plot_hist_kde

        test_df = pd.DataFrame({"v1": [1, 2, 3, 4, 5]})

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        test_data: List[DataFrameWithWeight] = [
            {"df": test_df, "weight": None},
        ]

        # Should handle None weight without error
        plot_hist_kde(
            test_data,
            names=["self"],
            column="v1",
            axis=ax,
            weighted=True,
            dist_type="hist",
        )

        # Verify plot was created
        self.assertIsNotNone(ax.get_title())

    def test_plot_qq_basic_functionality(self) -> None:
        """
        Test plot_qq function for basic QQ plot generation.

        This test verifies that the function correctly generates quantile-quantile
        plots comparing sample distributions against a target distribution,
        comparing actual scatter plot data points between weighted and unweighted modes.
        """
        from balance.stats_and_plots.weighted_comparisons_plots import plot_qq

        # Create test data with uniform distribution
        test_df = pd.DataFrame({"v1": np.linspace(0, 1, 100)})

        # Use dramatically different weights for the two datasets
        # First dataset: uniform weights
        # Second dataset: heavily weight the first few elements
        uniform_weights = pd.Series(np.ones(100))
        concentrated_weights = pd.Series([100] * 10 + [1] * 90)

        test_data_weighted: List[DataFrameWithWeight] = [
            {"df": test_df, "weight": uniform_weights},
            {"df": test_df, "weight": concentrated_weights},
        ]

        # Test 1: Generate weighted QQ plot (weighted=True)
        fig, ax_weighted = plt.subplots(1, 1, figsize=(7.2, 7.2))
        plot_qq(
            test_data_weighted,
            names=["self", "target"],
            column="v1",
            axis=ax_weighted,
            weighted=True,
        )

        # Verify plot was created
        self.assertIsNotNone(ax_weighted.get_xlabel())
        self.assertIsNotNone(ax_weighted.get_ylabel())

        # Extract scatter plot data from weighted plot
        weighted_collections = ax_weighted.collections
        if len(weighted_collections) >= 1:
            # Get the scatter plot offsets (x, y coordinates)
            weighted_offsets = weighted_collections[0].get_offsets()
            if len(weighted_offsets) > 0:
                weighted_x_values = weighted_offsets[:, 0]
                weighted_y_values = weighted_offsets[:, 1]

                # When weighted=True with different weights, x and y should differ
                # (points should not lie on the diagonal y=x line)
                max_diff_weighted = max(
                    abs(x - y) for x, y in zip(weighted_x_values, weighted_y_values)
                )
                self.assertGreater(
                    max_diff_weighted,
                    0.01,
                    "With weighted=True and different weights, QQ points should deviate from diagonal",
                )

        # Test 2: Generate unweighted QQ plot (weighted=False) with same data
        fig, ax_unweighted = plt.subplots(1, 1, figsize=(7.2, 7.2))
        plot_qq(
            test_data_weighted,
            names=["self", "target"],
            column="v1",
            axis=ax_unweighted,
            weighted=False,
        )

        # Extract scatter plot data from unweighted plot
        unweighted_collections = ax_unweighted.collections
        if len(unweighted_collections) >= 1:
            # Get the scatter plot offsets (x, y coordinates)
            unweighted_offsets = unweighted_collections[0].get_offsets()
            if len(unweighted_offsets) > 0:
                unweighted_x_values = unweighted_offsets[:, 0]
                unweighted_y_values = unweighted_offsets[:, 1]

                # When weighted=False, both datasets have same data, so QQ points
                # should lie on or very close to the diagonal (y ≈ x)
                for x_val, y_val in zip(unweighted_x_values, unweighted_y_values):
                    self.assertAlmostEqual(x_val, y_val, places=5)

    def test_plot_qq_unweighted(self) -> None:
        """
        Test plot_qq function with weighted=False parameter.

        This test verifies that the function correctly generates unweighted
        QQ plots.
        """
        from balance.stats_and_plots.weighted_comparisons_plots import plot_qq

        test_df = pd.DataFrame({"v1": np.random.uniform(size=50)})

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        test_data: List[DataFrameWithWeight] = [
            {"df": test_df, "weight": pd.Series(np.ones(50))},
            {"df": test_df, "weight": pd.Series(np.ones(50))},
        ]

        # Generate unweighted QQ plot
        plot_qq(
            test_data,
            names=["self", "target"],
            column="v1",
            axis=ax,
            weighted=False,
        )

        # Verify plot was created
        self.assertIsNotNone(ax)

    def test_plot_qq_categorical_basic_functionality(self) -> None:
        """
        Test plot_qq_categorical function for categorical QQ plots.

        This test verifies that the function correctly generates QQ plots
        for categorical variables using relative frequency comparisons,
        comparing actual scatter plot data points between weighted and unweighted modes.
        """
        from balance.stats_and_plots.weighted_comparisons_plots import (
            plot_qq_categorical,
        )

        # Create categorical test data
        test_df = pd.DataFrame(
            {"category": ["a"] * 10 + ["b"] * 20 + ["c"] * 15 + ["d"] * 5}
        )

        # Use dramatically different weights for the two datasets
        # First dataset: uniform weights
        # Second dataset: heavily weight the first category elements
        uniform_weights = pd.Series(np.ones(len(test_df)))
        concentrated_weights = pd.Series([100] * 10 + [1] * 40)

        test_data_weighted: List[DataFrameWithWeight] = [
            {"df": test_df, "weight": uniform_weights},
            {"df": test_df, "weight": concentrated_weights},
        ]

        # Test 1: Generate weighted categorical QQ plot (weighted=True)
        fig, ax_weighted = plt.subplots(1, 1, figsize=(7.2, 7.2))
        plot_qq_categorical(
            test_data_weighted,
            names=["self", "target"],
            column="category",
            axis=ax_weighted,
            weighted=True,
        )

        # Verify plot was created
        self.assertIsNotNone(ax_weighted.get_xlabel())

        # Extract scatter plot data from weighted plot
        weighted_collections = ax_weighted.collections
        if len(weighted_collections) >= 1:
            # Get the scatter plot offsets (x, y coordinates)
            weighted_offsets = weighted_collections[0].get_offsets()
            if len(weighted_offsets) > 0:
                weighted_x_values = weighted_offsets[:, 0]
                weighted_y_values = weighted_offsets[:, 1]

                # When weighted=True with different weights, x and y should differ
                # (points should not lie on the diagonal y=x line)
                max_diff_weighted = max(
                    abs(x - y) for x, y in zip(weighted_x_values, weighted_y_values)
                )
                self.assertGreater(
                    max_diff_weighted,
                    0.01,
                    "With weighted=True and different weights, categorical QQ points should deviate from diagonal",
                )

        # Test 2: Generate unweighted categorical QQ plot (weighted=False) with same data
        fig, ax_unweighted = plt.subplots(1, 1, figsize=(7.2, 7.2))
        plot_qq_categorical(
            test_data_weighted,
            names=["self", "target"],
            column="category",
            axis=ax_unweighted,
            weighted=False,
        )

        # Extract scatter plot data from unweighted plot
        unweighted_collections = ax_unweighted.collections
        if len(unweighted_collections) >= 1:
            # Get the scatter plot offsets (x, y coordinates)
            unweighted_offsets = unweighted_collections[0].get_offsets()
            if len(unweighted_offsets) > 0:
                unweighted_x_values = unweighted_offsets[:, 0]
                unweighted_y_values = unweighted_offsets[:, 1]

                # When weighted=False, both datasets have same data, so QQ points
                # should lie on or very close to the diagonal (y ≈ x)
                for x_val, y_val in zip(unweighted_x_values, unweighted_y_values):
                    self.assertAlmostEqual(x_val, y_val, places=5)

    def test_plot_qq_categorical_unweighted(self) -> None:
        """
        Test plot_qq_categorical function with weighted=False.

        This test verifies that the function correctly generates unweighted
        categorical QQ plots.
        """
        from balance.stats_and_plots.weighted_comparisons_plots import (
            plot_qq_categorical,
        )

        test_df = pd.DataFrame({"category": ["a", "b", "c", "d"] * 10})

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        test_data: List[DataFrameWithWeight] = [
            {"df": test_df, "weight": pd.Series(np.ones(len(test_df)))},
            {"df": test_df, "weight": pd.Series(np.ones(len(test_df)))},
        ]

        # Generate unweighted categorical QQ plot
        plot_qq_categorical(
            test_data,
            names=["self", "target"],
            column="category",
            axis=ax,
            weighted=False,
        )

        # Verify plot was created
        self.assertIsNotNone(ax)

    def test_plotly_plot_qq_basic_functionality(self) -> None:
        """
        Test plotly_plot_qq function for interactive QQ plot generation.

        This test verifies that the function correctly generates plotly
        interactive QQ plots and returns a dictionary of figures, comparing
        actual trace data between weighted and unweighted modes.
        """
        from balance.stats_and_plots.weighted_comparisons_plots import plotly_plot_qq

        # Create test data with uniform distribution
        np.random.seed(42)
        test_df = pd.DataFrame(
            {
                "v1": np.linspace(0, 1, 100),
            }
        )

        # Use dramatically different weights for the two datasets
        # First dataset (self): uniform weights
        # Second dataset (target): heavily weight the first few elements
        test_df_self = test_df.copy()
        test_df_self["weight"] = np.ones(100)

        test_df_target = test_df.copy()
        test_df_target["weight"] = [100] * 10 + [1] * 90

        dict_of_dfs_weighted = {
            "self": test_df_self,
            "target": test_df_target,
        }

        # Test 1: Generate weighted plotly QQ plot
        result_weighted = plotly_plot_qq(
            dict_of_dfs_weighted,
            variables=["v1"],
            plot_it=False,
            return_dict_of_figures=True,
        )

        # Verify result is a dictionary with expected keys
        self.assertIsNotNone(result_weighted)
        self.assertIsInstance(result_weighted, dict)
        self.assertIn("v1", result_weighted)  # type: ignore[arg-type]

        # Extract trace data from weighted plot
        result_weighted = _verify_value_type(result_weighted, dict)
        fig_weighted = result_weighted["v1"]
        traces_weighted = fig_weighted.data
        if len(traces_weighted) >= 1:
            # Get the scatter trace y-values (quantiles)
            trace_y_weighted = traces_weighted[0].y
            trace_x_weighted = traces_weighted[0].x

            if trace_y_weighted is not None and trace_x_weighted is not None:
                # When weighted=True with different weights, x and y should differ
                # (points should not lie on the diagonal y=x line)
                max_diff_weighted = max(
                    abs(x - y) for x, y in zip(trace_x_weighted, trace_y_weighted)
                )
                self.assertGreater(
                    max_diff_weighted,
                    0.01,
                    "With different weights, plotly QQ points should deviate from diagonal",
                )

        # Test 2: Generate unweighted plotly QQ plot (same weights for both)
        test_df_self_unweighted = test_df.copy()
        test_df_self_unweighted["weight"] = np.ones(100)

        test_df_target_unweighted = test_df.copy()
        test_df_target_unweighted["weight"] = np.ones(100)

        dict_of_dfs_unweighted = {
            "self": test_df_self_unweighted,
            "target": test_df_target_unweighted,
        }

        result_unweighted = plotly_plot_qq(
            dict_of_dfs_unweighted,
            variables=["v1"],
            plot_it=False,
            return_dict_of_figures=True,
        )

        # Extract trace data from unweighted plot
        result_unweighted = _verify_value_type(result_unweighted, dict)
        fig_unweighted = result_unweighted["v1"]
        traces_unweighted = fig_unweighted.data
        if len(traces_unweighted) >= 1:
            # Get the scatter trace y-values (quantiles)
            trace_y_unweighted = traces_unweighted[0].y
            trace_x_unweighted = traces_unweighted[0].x

            if trace_y_unweighted is not None and trace_x_unweighted is not None:
                # When both datasets have same weights, QQ points should lie on diagonal
                for x_val, y_val in zip(trace_x_unweighted, trace_y_unweighted):
                    self.assertAlmostEqual(x_val, y_val, places=5)

    def test_plotly_plot_density_basic_functionality(self) -> None:
        """
        Test plotly_plot_density function for interactive density plot generation.

        This test verifies that the function correctly generates plotly
        interactive density plots and returns a dictionary of figures, comparing
        actual trace data between weighted and unweighted modes.
        """
        from balance.stats_and_plots.weighted_comparisons_plots import (
            plotly_plot_density,
        )

        # Create test data with values that will produce clear density curves
        np.random.seed(42)
        test_df = pd.DataFrame(
            {
                "v1": np.random.normal(size=100),
            }
        )

        # Use dramatically different weights for the two datasets
        # First dataset (self): uniform weights
        # Second dataset (target): heavily weight the first few elements
        test_df_self = test_df.copy()
        test_df_self["weight"] = np.ones(100)

        test_df_target = test_df.copy()
        test_df_target["weight"] = [100] * 10 + [1] * 90

        dict_of_dfs_weighted = {
            "self": test_df_self,
            "target": test_df_target,
        }

        # Test 1: Generate weighted plotly density plot
        result_weighted = plotly_plot_density(
            dict_of_dfs_weighted,
            variables=["v1"],
            plot_it=False,
            return_dict_of_figures=True,
        )

        # Verify result is a dictionary with expected keys
        self.assertIsNotNone(result_weighted)
        self.assertIsInstance(result_weighted, dict)
        self.assertIn("v1", result_weighted)  # type: ignore[arg-type]

        # Extract trace data from weighted plot
        result_weighted = _verify_value_type(result_weighted, dict)
        fig_weighted = result_weighted["v1"]
        traces_weighted = fig_weighted.data
        if len(traces_weighted) >= 2:
            # Get the density trace y-values for self and target
            trace_y_self_weighted = traces_weighted[0].y
            trace_y_target_weighted = traces_weighted[1].y

            if (
                trace_y_self_weighted is not None
                and trace_y_target_weighted is not None
            ):
                # When weights differ, the density curves should differ
                max_diff_weighted = max(
                    abs(y_self - y_target)
                    for y_self, y_target in zip(
                        trace_y_self_weighted, trace_y_target_weighted
                    )
                )
                self.assertGreater(
                    max_diff_weighted,
                    0.01,
                    "With different weights, plotly density curves should differ",
                )

        # Test 2: Generate unweighted plotly density plot (same weights for both)
        test_df_self_unweighted = test_df.copy()
        test_df_self_unweighted["weight"] = np.ones(100)

        test_df_target_unweighted = test_df.copy()
        test_df_target_unweighted["weight"] = np.ones(100)

        dict_of_dfs_unweighted = {
            "self": test_df_self_unweighted,
            "target": test_df_target_unweighted,
        }

        result_unweighted = plotly_plot_density(
            dict_of_dfs_unweighted,
            variables=["v1"],
            plot_it=False,
            return_dict_of_figures=True,
        )

        # Extract trace data from unweighted plot
        result_unweighted = _verify_value_type(result_unweighted, dict)
        fig_unweighted = result_unweighted["v1"]
        traces_unweighted = fig_unweighted.data
        if len(traces_unweighted) >= 2:
            # Get the density trace y-values for self and target
            trace_y_self_unweighted = traces_unweighted[0].y
            trace_y_target_unweighted = traces_unweighted[1].y

            if (
                trace_y_self_unweighted is not None
                and trace_y_target_unweighted is not None
            ):
                # When both datasets have same weights, density curves should be identical
                for y_self, y_target in zip(
                    trace_y_self_unweighted, trace_y_target_unweighted
                ):
                    self.assertAlmostEqual(y_self, y_target, places=5)

    def test_plotly_plot_bar_basic_functionality(self) -> None:
        """
        Test plotly_plot_bar function for interactive bar plot generation.

        This test verifies that the function correctly generates plotly
        interactive bar plots and returns a dictionary of figures, comparing
        actual trace data between weighted and unweighted modes.
        """
        from balance.stats_and_plots.weighted_comparisons_plots import plotly_plot_bar

        # Create categorical test data
        test_df = pd.DataFrame(
            {
                "cat1": ["a", "b", "c", "d"] * 25,
            }
        )

        # Use dramatically different weights for the two datasets
        # First dataset (self): uniform weights
        # Second dataset (target): heavily weight the first category elements
        test_df_self = test_df.copy()
        test_df_self["weight"] = np.ones(100)

        test_df_target = test_df.copy()
        test_df_target["weight"] = [100] * 25 + [1] * 75

        dict_of_dfs_weighted = {
            "self": test_df_self,
            "target": test_df_target,
        }

        # Test 1: Generate weighted plotly bar plot
        result_weighted = plotly_plot_bar(
            dict_of_dfs_weighted,
            variables=["cat1"],
            plot_it=False,
            return_dict_of_figures=True,
        )

        # Verify result is a dictionary with expected keys
        self.assertIsNotNone(result_weighted)
        self.assertIsInstance(result_weighted, dict)
        self.assertIn("cat1", result_weighted)  # type: ignore[arg-type]

        # Extract trace data from weighted plot
        result_weighted = _verify_value_type(result_weighted, dict)
        fig_weighted = result_weighted["cat1"]
        traces_weighted = fig_weighted.data
        if len(traces_weighted) >= 2:
            # Get the bar trace y-values for self and target
            trace_y_self_weighted = traces_weighted[0].y
            trace_y_target_weighted = traces_weighted[1].y

            if (
                trace_y_self_weighted is not None
                and trace_y_target_weighted is not None
            ):
                # When weights differ, the bar heights should differ
                max_diff_weighted = max(
                    abs(y_self - y_target)
                    for y_self, y_target in zip(
                        trace_y_self_weighted, trace_y_target_weighted
                    )
                )
                self.assertGreater(
                    max_diff_weighted,
                    0.01,
                    "With different weights, plotly bar heights should differ",
                )

        # Test 2: Generate unweighted plotly bar plot (same weights for both)
        test_df_self_unweighted = test_df.copy()
        test_df_self_unweighted["weight"] = np.ones(100)

        test_df_target_unweighted = test_df.copy()
        test_df_target_unweighted["weight"] = np.ones(100)

        dict_of_dfs_unweighted = {
            "self": test_df_self_unweighted,
            "target": test_df_target_unweighted,
        }

        result_unweighted = plotly_plot_bar(
            dict_of_dfs_unweighted,
            variables=["cat1"],
            plot_it=False,
            return_dict_of_figures=True,
        )

        # Extract trace data from unweighted plot
        result_unweighted = _verify_value_type(result_unweighted, dict)
        fig_unweighted = result_unweighted["cat1"]
        traces_unweighted = fig_unweighted.data
        if len(traces_unweighted) >= 2:
            # Get the bar trace y-values for self and target
            trace_y_self_unweighted = traces_unweighted[0].y
            trace_y_target_unweighted = traces_unweighted[1].y

            if (
                trace_y_self_unweighted is not None
                and trace_y_target_unweighted is not None
            ):
                # When both datasets have same weights, bar heights should be identical
                for y_self, y_target in zip(
                    trace_y_self_unweighted, trace_y_target_unweighted
                ):
                    self.assertAlmostEqual(y_self, y_target, places=5)

    def test_plotly_functions_return_none_by_default(self) -> None:
        """
        Test that plotly plotting functions return None when return_dict_of_figures=False.

        This test verifies the default return behavior for plotly_plot_qq,
        plotly_plot_density, and plotly_plot_bar functions.
        """
        from balance.stats_and_plots.weighted_comparisons_plots import (
            plotly_plot_bar,
            plotly_plot_density,
            plotly_plot_qq,
        )

        # Create test data
        test_df = pd.DataFrame(
            {
                "v1": np.random.normal(size=50),
                "cat1": ["a", "b", "c"] * 16 + ["a", "b"],
                "weight": np.ones(50),
            }
        )

        dict_of_dfs = {
            "self": test_df.copy(),
            "target": test_df.copy(),
        }

        # Test plotly_plot_qq returns None
        result_qq = plotly_plot_qq(
            dict_of_dfs,
            variables=["v1"],
            plot_it=False,
            return_dict_of_figures=False,
        )
        self.assertIsNone(result_qq)

        # Test plotly_plot_density returns None
        result_density = plotly_plot_density(
            dict_of_dfs,
            variables=["v1"],
            plot_it=False,
            return_dict_of_figures=False,
        )
        self.assertIsNone(result_density)

        # Test plotly_plot_bar returns None
        result_bar = plotly_plot_bar(
            dict_of_dfs,
            variables=["cat1"],
            plot_it=False,
            return_dict_of_figures=False,
        )
        self.assertIsNone(result_bar)
