# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import absolute_import, division, print_function, unicode_literals

import io
import textwrap
from typing import List
from unittest.mock import patch

import balance.testutil
import numpy as np
import pandas as pd
from balance.stats_and_plots.ascii_plots import (
    _build_legend,
    _render_horizontal_bars,
    _weighted_histogram,
    ascii_comparative_hist,
    ascii_plot_bar,
    ascii_plot_dist,
    ascii_plot_hist,
)
from balance.stats_and_plots.weighted_comparisons_plots import (
    DataFrameWithWeight,
    plot_dist,
)


class TestWeightedHistogram(balance.testutil.BalanceTestCase):
    """Tests for _weighted_histogram helper."""

    def test_uniform_weights(self) -> None:
        """Test with uniform weights produces expected proportions."""
        values = pd.Series([0.5, 1.5, 2.5, 3.5])
        bin_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = _weighted_histogram(values, None, bin_edges)
        # Each value falls in a different bin, so each bin should be 0.25
        np.testing.assert_array_almost_equal(result, [0.25, 0.25, 0.25, 0.25])

    def test_nonuniform_weights(self) -> None:
        """Test with non-uniform weights changes proportions."""
        values = pd.Series([0.5, 1.5])
        weights = pd.Series([3.0, 1.0])
        bin_edges = np.array([0.0, 1.0, 2.0])
        result = _weighted_histogram(values, weights, bin_edges)
        np.testing.assert_array_almost_equal(result, [0.75, 0.25])

    def test_proportions_sum_to_one(self) -> None:
        """Test that returned proportions sum to 1.0."""
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        bin_edges = np.linspace(0.5, 5.5, 6)
        result = _weighted_histogram(values, weights, bin_edges)
        self.assertAlmostEqual(float(result.sum()), 1.0)

    def test_empty_values(self) -> None:
        """Test with no values returns zeros."""
        values = pd.Series([], dtype=float)
        bin_edges = np.array([0.0, 1.0, 2.0])
        result = _weighted_histogram(values, None, bin_edges)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_negative_weights_raises(self) -> None:
        """Test that negative weights raise ValueError."""
        values = pd.Series([0.5, 1.5])
        weights = pd.Series([1.0, -1.0])
        bin_edges = np.array([0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            _weighted_histogram(values, weights, bin_edges)

    def test_non_numeric_weights_raises(self) -> None:
        """Test that non-numeric weights raise TypeError."""
        values = pd.Series([0.5, 1.5])
        weights = pd.Series(["a", "b"])
        bin_edges = np.array([0.0, 1.0, 2.0])
        with self.assertRaises(TypeError):
            _weighted_histogram(values, weights, bin_edges)

    def test_none_weights_passes_validation(self) -> None:
        """Test that None weights are accepted without error."""
        values = pd.Series([0.5, 1.5])
        bin_edges = np.array([0.0, 1.0, 2.0])
        result = _weighted_histogram(values, None, bin_edges)
        np.testing.assert_array_almost_equal(result, [0.5, 0.5])


class TestRenderHorizontalBars(balance.testutil.BalanceTestCase):
    """Tests for _render_horizontal_bars helper."""

    def test_basic_rendering(self) -> None:
        """Test basic bar rendering with known proportions."""
        result = _render_horizontal_bars(
            label="cat_a",
            proportions={"sample": 0.5, "population": 0.25},
            legend_names=["sample", "population"],
            bar_width=20,
            max_value=0.5,
            label_width=10,
        )
        lines = result.split("\n")
        self.assertEqual(len(lines), 2)
        # First line has the label
        self.assertIn("cat_a", lines[0])
        self.assertIn("50.0%", lines[0])
        # Second line has blank label
        self.assertIn("25.0%", lines[1])

    def test_zero_proportion(self) -> None:
        """Test that zero proportion produces empty bar with 0.0%."""
        result = _render_horizontal_bars(
            label="x",
            proportions={"sample": 0.0},
            legend_names=["sample"],
            bar_width=20,
            max_value=1.0,
            label_width=5,
        )
        self.assertIn("0.0%", result)

    def test_bar_width_scaling(self) -> None:
        """Test that bars scale correctly to bar_width."""
        result = _render_horizontal_bars(
            label="x",
            proportions={"sample": 1.0},
            legend_names=["sample"],
            bar_width=20,
            max_value=1.0,
            label_width=5,
        )
        # The bar should be exactly 20 '█' characters
        self.assertIn("█" * 20, result)

    def test_multiple_datasets_different_chars(self) -> None:
        """Test that different datasets get different characters."""
        result = _render_horizontal_bars(
            label="x",
            proportions={"a": 0.5, "b": 0.5},
            legend_names=["a", "b"],
            bar_width=10,
            max_value=0.5,
            label_width=5,
        )
        lines = result.split("\n")
        self.assertIn("█", lines[0])
        self.assertIn("▒", lines[1])


class TestBuildLegend(balance.testutil.BalanceTestCase):
    """Tests for _build_legend helper."""

    def test_legend_format(self) -> None:
        """Test that legend has correct format."""
        result = _build_legend(["sample", "population"])
        self.assertIn("Legend:", result)
        self.assertIn("█ sample", result)
        self.assertIn("▒ population", result)
        self.assertIn(
            "Bar lengths are proportional to weighted frequency within each dataset.",
            result,
        )

    def test_three_datasets(self) -> None:
        """Test legend with three datasets."""
        result = _build_legend(["sample", "adjusted", "population"])
        self.assertIn("█ sample", result)
        self.assertIn("▒ adjusted", result)
        self.assertIn("▐ population", result)
        self.assertIn(
            "Bar lengths are proportional to weighted frequency within each dataset.",
            result,
        )


class TestAsciiPlotBar(balance.testutil.BalanceTestCase):
    """Tests for ascii_plot_bar function."""

    def test_basic_categorical_output(self) -> None:
        """Test that basic categorical data produces expected ASCII bars."""
        df = pd.DataFrame({"group": ("a", "b", "c", "c")})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series((1, 1, 1, 1))},
            {"df": df, "weight": pd.Series((1, 1, 1, 1))},
        ]
        result = ascii_plot_bar(dfs, names=["self", "target"], column="group")
        self.assertIn("=== group (categorical) ===", result)
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertIn("c", result)
        self.assertIn("Legend:", result)

    def test_weighted_categorical_output(self) -> None:
        """Test that weights affect proportions correctly."""
        df = pd.DataFrame({"group": ("a", "b", "c", "c")})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series((1, 1, 1, 1))},
            {"df": df, "weight": pd.Series((2, 1, 1, 1))},
        ]
        result = ascii_plot_bar(dfs, names=["self", "target"], column="group")
        # With equal weights: a=25%, b=25%, c=50%
        self.assertIn("25.0%", result)
        self.assertIn("50.0%", result)
        # With weight (2,1,1,1): a=40%, b=20%, c=40%
        self.assertIn("40.0%", result)
        self.assertIn("20.0%", result)

    def test_single_dataset(self) -> None:
        """Test with a single DataFrame (no comparison)."""
        df = pd.DataFrame({"group": ("a", "b")})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series((1, 1))},
        ]
        result = ascii_plot_bar(dfs, names=["self"], column="group")
        self.assertIn("=== group (categorical) ===", result)
        self.assertIn("50.0%", result)

    def test_legend_names_transformation(self) -> None:
        """Test that naming_legend is applied."""
        df = pd.DataFrame({"group": ("a", "b")})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series((1, 1))},
            {"df": df, "weight": pd.Series((1, 1))},
            {"df": df, "weight": pd.Series((1, 1))},
        ]
        result = ascii_plot_bar(
            dfs, names=["self", "unadjusted", "target"], column="group"
        )
        # "self" -> "adjusted", "unadjusted" -> "sample", "target" -> "population"
        self.assertIn("adjusted", result)
        self.assertIn("sample", result)
        self.assertIn("population", result)

    def test_unweighted(self) -> None:
        """Test unweighted mode ignores weights."""
        df = pd.DataFrame({"group": ("a", "b")})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series((100, 1))},
        ]
        result = ascii_plot_bar(dfs, names=["self"], column="group", weighted=False)
        # Without weighting, each row counts equally: a=50%, b=50%
        self.assertIn("50.0%", result)


class TestAsciiPlotHist(balance.testutil.BalanceTestCase):
    """Tests for ascii_plot_hist function."""

    def test_basic_numeric_output(self) -> None:
        """Test that numeric data produces expected bin labels and bars."""
        df = pd.DataFrame({"v1": [1.0, 2.0, 3.0, 4.0, 5.0]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series(np.ones(5))},
        ]
        result = ascii_plot_hist(dfs, names=["self"], column="v1", n_bins=5)
        self.assertIn("=== v1 (numeric) ===", result)
        self.assertIn("Legend:", result)
        # Should have 5 bin labels
        self.assertIn("[", result)
        self.assertIn(")", result)

    def test_weighted_numeric_output(self) -> None:
        """Test that weights change histogram proportions."""
        df = pd.DataFrame({"v1": [0.5, 1.5]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([3.0, 1.0])},
        ]
        result = ascii_plot_hist(dfs, names=["self"], column="v1", n_bins=2)
        # First bin should have ~75%, second ~25%
        self.assertIn("75.0%", result)
        self.assertIn("25.0%", result)

    def test_custom_n_bins(self) -> None:
        """Test that n_bins parameter changes number of bins."""
        df = pd.DataFrame({"v1": np.linspace(0, 10, 100)})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series(np.ones(100))},
        ]
        result_5 = ascii_plot_hist(dfs, names=["self"], column="v1", n_bins=5)
        result_20 = ascii_plot_hist(dfs, names=["self"], column="v1", n_bins=20)
        # More bins = more lines of output
        self.assertGreater(len(result_20.split("\n")), len(result_5.split("\n")))

    def test_handles_single_value(self) -> None:
        """Test edge case where all values are the same."""
        df = pd.DataFrame({"v1": [5.0, 5.0, 5.0]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series(np.ones(3))},
        ]
        result = ascii_plot_hist(dfs, names=["self"], column="v1", n_bins=3)
        self.assertIn("=== v1 (numeric) ===", result)

    def test_two_datasets_comparison(self) -> None:
        """Test comparing two datasets in a histogram."""
        df = pd.DataFrame({"v1": [1.0, 2.0, 3.0, 4.0]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([1, 1, 1, 1])},
            {"df": df, "weight": pd.Series([4, 1, 1, 1])},
        ]
        result = ascii_plot_hist(dfs, names=["self", "target"], column="v1", n_bins=4)
        # Both sample and population should appear in legend
        self.assertIn("sample", result)
        self.assertIn("population", result)

    def test_unweighted_hist(self) -> None:
        """Test ascii_plot_hist with weighted=False (line 316)."""
        df = pd.DataFrame({"v1": [1.0, 2.0, 3.0]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([100.0, 100.0, 100.0])},
        ]
        result = ascii_plot_hist(
            dfs, names=["self"], column="v1", n_bins=2, weighted=False
        )
        # Without weighting, each value counts equally
        self.assertIn("=== v1 (numeric) ===", result)

    def test_empty_after_na_drop(self) -> None:
        """Test ascii_plot_hist with all-NaN data returns empty message (line 323)."""
        df = pd.DataFrame({"v1": [np.nan, np.nan, np.nan]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": None},
        ]
        result = ascii_plot_hist(dfs, names=["self"], column="v1", n_bins=2)
        self.assertIn("No data available.", result)


class TestAsciiPlotDist(balance.testutil.BalanceTestCase):
    """Tests for ascii_plot_dist dispatcher function."""

    def test_dispatches_categorical_and_numeric(self) -> None:
        """Test that mixed data produces both barplots and histograms."""
        df = pd.DataFrame(
            {
                "gender": ["male", "female", "female", "male"],
                "age": [25.0, 35.0, 45.0, 55.0],
            }
        )
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([1, 1, 1, 1])},
            {"df": df, "weight": pd.Series([1, 1, 1, 1])},
        ]
        result = ascii_plot_dist(
            dfs, names=["self", "target"], numeric_n_values_threshold=0
        )
        self.assertIn("(categorical)", result)
        self.assertIn("(numeric)", result)

    def test_respects_numeric_n_values_threshold(self) -> None:
        """Test that low-cardinality numeric columns are treated as categorical."""
        df = pd.DataFrame({"v1": [1, 2, 3, 1, 2, 3]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series(np.ones(6))},
        ]
        # With threshold=10, 3 unique values < 10, so treated as categorical
        result = ascii_plot_dist(dfs, names=["self"], numeric_n_values_threshold=10)
        self.assertIn("(categorical)", result)

        # With threshold=0, treated as numeric
        result = ascii_plot_dist(dfs, names=["self"], numeric_n_values_threshold=0)
        self.assertIn("(numeric)", result)

    def test_returns_string(self) -> None:
        """Test that the function returns a string."""
        df = pd.DataFrame({"v1": [1.0, 2.0, 3.0]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series(np.ones(3))},
        ]
        result = ascii_plot_dist(dfs, names=["self"])
        self.assertIsInstance(result, str)

    def test_prints_to_stdout(self) -> None:
        """Test that the function prints output to stdout."""
        df = pd.DataFrame({"v1": [1.0, 2.0, 3.0]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series(np.ones(3))},
        ]
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            ascii_plot_dist(dfs, names=["self"])
            output = mock_stdout.getvalue()
        self.assertIn("v1", output)

    def test_variables_parameter(self) -> None:
        """Test that the variables parameter filters variables."""
        df = pd.DataFrame(
            {
                "v1": [1.0, 2.0, 3.0],
                "v2": [4.0, 5.0, 6.0],
            }
        )
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series(np.ones(3))},
        ]
        result = ascii_plot_dist(dfs, names=["self"], variables=["v1"])
        self.assertIn("v1", result)
        self.assertNotIn("v2", result)

    def test_default_names(self) -> None:
        """Test that default names are generated when names is None."""
        df = pd.DataFrame({"v1": [1.0, 2.0, 3.0]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series(np.ones(3))},
        ]
        result = ascii_plot_dist(dfs, names=None)
        self.assertIn("df_0", result)


class TestPlotDistBalanceLibrary(balance.testutil.BalanceTestCase):
    """Tests for plot_dist with library='balance'."""

    def test_plot_dist_balance_returns_string(self) -> None:
        """Test that plot_dist with library='balance' returns a string."""
        df = pd.DataFrame({"v1": [1.0, 2.0, 3.0]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series(np.ones(3))},
        ]
        result = plot_dist(
            dfs, names=["self"], library="balance", numeric_n_values_threshold=0
        )
        self.assertIsInstance(result, str)
        assert isinstance(result, str)
        self.assertIn("v1", result)

    def test_plot_dist_invalid_library_message(self) -> None:
        """Test that the error message for invalid library mentions 'balance'."""
        df = pd.DataFrame({"v1": [1.0, 2.0]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series(np.ones(2))},
        ]
        with self.assertRaises(ValueError) as cm:
            plot_dist(dfs, names=["self"], library="invalid")  # pyre-ignore[6]
        self.assertIn("balance", str(cm.exception))


class TestAsciiPlotsEndToEnd(balance.testutil.BalanceTestCase):
    """End-to-end tests comparing full ASCII plot output against expected strings."""

    def _assert_lines_equal(self, actual: str, expected_text: str) -> None:
        """Compare actual output lines against a dedented expected string."""
        expected_lines = textwrap.dedent(expected_text).strip().splitlines()
        actual_lines = [line.rstrip() for line in actual.splitlines()]
        self.assertEqual(actual_lines, expected_lines)

    def test_e2e_ascii_bar_single_dataset(self) -> None:
        """Full barplot output for a single categorical variable, one dataset."""
        df = pd.DataFrame({"color": ["red", "blue", "blue", "green"]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
        ]
        result = ascii_plot_bar(dfs, names=["self"], column="color", bar_width=20)
        self._assert_lines_equal(
            result,
            """\
            === color (categorical) ===

            Category | sample
                     |
            blue     | ████████████████████ (50.0%)
            green    | ██████████ (25.0%)
            red      | ██████████ (25.0%)

            Legend: █ sample
            Bar lengths are proportional to weighted frequency within each dataset.
            """,
        )

    def test_e2e_ascii_bar_two_datasets(self) -> None:
        """Full barplot output for a single categorical variable, two datasets."""
        df_a = pd.DataFrame({"color": ["red", "blue", "blue", "green"]})
        df_b = pd.DataFrame({"color": ["red", "red", "blue", "green"]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df_a, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
            {"df": df_b, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
        ]
        result = ascii_plot_bar(
            dfs, names=["self", "target"], column="color", bar_width=20
        )
        self._assert_lines_equal(
            result,
            """\
            === color (categorical) ===

            Category | sample  population
                     |
            blue     | ████████████████████ (50.0%)
                     | ▒▒▒▒▒▒▒▒▒▒ (25.0%)
            green    | ██████████ (25.0%)
                     | ▒▒▒▒▒▒▒▒▒▒ (25.0%)
            red      | ██████████ (25.0%)
                     | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (50.0%)

            Legend: █ sample  ▒ population
            Bar lengths are proportional to weighted frequency within each dataset.
            """,
        )

    def test_e2e_ascii_hist_single_dataset(self) -> None:
        """Full histogram output for a single numeric variable, one dataset."""
        df = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
        ]
        result = ascii_plot_hist(
            dfs, names=["self"], column="age", n_bins=2, bar_width=20
        )
        self._assert_lines_equal(
            result,
            """\
            === age (numeric) ===

            Bin            | sample
                           |
            [10.00, 25.00) | ████████████████████ (50.0%)
            [25.00, 40.00] | ████████████████████ (50.0%)

            Legend: █ sample
            Bar lengths are proportional to weighted frequency within each dataset.
            """,
        )

    def test_e2e_ascii_hist_two_datasets(self) -> None:
        """Full histogram output for a single numeric variable, two datasets."""
        df_a = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0]})
        df_b = pd.DataFrame({"age": [10.0, 10.0, 10.0, 40.0]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df_a, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
            {"df": df_b, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
        ]
        result = ascii_plot_hist(
            dfs, names=["self", "target"], column="age", n_bins=2, bar_width=20
        )
        self._assert_lines_equal(
            result,
            """\
            === age (numeric) ===

            Bin            | sample  population
                           |
            [10.00, 25.00) | █████████████ (50.0%)
                           | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (75.0%)
            [25.00, 40.00] | █████████████ (50.0%)
                           | ▒▒▒▒▒▒▒ (25.0%)

            Legend: █ sample  ▒ population
            Bar lengths are proportional to weighted frequency within each dataset.
            """,
        )

    def test_e2e_ascii_plot_dist_mixed(self) -> None:
        """Full output for ascii_plot_dist with one categorical and one numeric."""
        df_a = pd.DataFrame(
            {"color": ["red", "blue", "blue", "green"], "age": [10.0, 20.0, 30.0, 40.0]}
        )
        df_b = pd.DataFrame(
            {"color": ["red", "red", "blue", "green"], "age": [10.0, 10.0, 10.0, 40.0]}
        )
        dfs: List[DataFrameWithWeight] = [
            {"df": df_a, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
            {"df": df_b, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
        ]
        with patch("sys.stdout", new_callable=io.StringIO):
            result = ascii_plot_dist(
                dfs,
                names=["self", "target"],
                numeric_n_values_threshold=0,
                n_bins=2,
                bar_width=20,
            )
        self._assert_lines_equal(
            result,
            """\
            === color (categorical) ===

            Category | sample  population
                     |
            blue     | ████████████████████ (50.0%)
                     | ▒▒▒▒▒▒▒▒▒▒ (25.0%)
            green    | ██████████ (25.0%)
                     | ▒▒▒▒▒▒▒▒▒▒ (25.0%)
            red      | ██████████ (25.0%)
                     | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (50.0%)

            Legend: █ sample  ▒ population
            Bar lengths are proportional to weighted frequency within each dataset.

            === age (numeric) ===

            Bin            | sample  population
                           |
            [10.00, 25.00) | █████████████ (50.0%)
                           | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (75.0%)
            [25.00, 40.00] | █████████████ (50.0%)
                           | ▒▒▒▒▒▒▒ (25.0%)

            Legend: █ sample  ▒ population
            Bar lengths are proportional to weighted frequency within each dataset.
            """,
        )

    def test_e2e_plot_dist_balance_library(self) -> None:
        """Full output through the plot_dist(library='balance') public API."""
        df = pd.DataFrame({"color": ["red", "blue", "blue", "green"]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
        ]
        with patch("sys.stdout", new_callable=io.StringIO):
            result = plot_dist(
                dfs,
                names=["self"],
                library="balance",
                bar_width=20,
            )
        assert isinstance(result, str)
        self._assert_lines_equal(
            result,
            """\
            === color (categorical) ===

            Category | sample
                     |
            blue     | ████████████████████ (50.0%)
            green    | ██████████ (25.0%)
            red      | ██████████ (25.0%)

            Legend: █ sample
            Bar lengths are proportional to weighted frequency within each dataset.
            """,
        )


class TestAsciiComparativeHistEndToEnd(balance.testutil.BalanceTestCase):
    """End-to-end tests for ascii_comparative_hist function."""

    def _assert_lines_equal(self, actual: str, expected_text: str) -> None:
        """Compare actual output lines against a dedented expected string."""
        expected_lines = textwrap.dedent(expected_text).strip().splitlines()
        actual_lines = [line.rstrip() for line in actual.splitlines()]
        self.assertEqual(actual_lines, expected_lines)

    def test_e2e_comparative_hist_single_dataset(self) -> None:
        """One dataset, 2 bins — degrades to a regular histogram with filled bars."""
        df = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
        ]
        result = ascii_comparative_hist(
            dfs, names=["Normal"], column="age", n_bins=2, bar_width=20
        )
        self._assert_lines_equal(
            result,
            """\
            Range          | Normal (%)
            ------------------------------------------
            [10.00, 25.00) | ████████████████████ 50.0
            [25.00, 40.00] | ████████████████████ 50.0
            ------------------------------------------
            Total          | 100.0
            """,
        )

    def test_e2e_comparative_hist_two_datasets(self) -> None:
        """Two datasets, 2 bins — verifies common, excess, and missing rendering."""
        df_a = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0]})
        df_b = pd.DataFrame({"age": [10.0, 10.0, 10.0, 40.0]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df_a, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
            {"df": df_b, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
        ]
        result = ascii_comparative_hist(
            dfs, names=["Normal", "Skewed"], column="age", n_bins=2, bar_width=20
        )
        self._assert_lines_equal(
            result,
            """\
            Range          | Normal (%)         | Skewed (%)
            ---------------------------------------------------------------
            [10.00, 25.00) | █████████████ 50.0 | █████████████▒▒▒▒▒▒▒ 75.0
            [25.00, 40.00] | █████████████ 50.0 | ███████     ] 25.0
            ---------------------------------------------------------------
            Total          | 100.0              | 100.0

            Key: █ = shared with Normal, ▒ = excess,    ] = deficit
            """,
        )

    def test_e2e_comparative_hist_three_datasets(self) -> None:
        """Three datasets, 3 bins — full comparative display."""
        df_a = pd.DataFrame({"v": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]})
        df_b = pd.DataFrame({"v": [1.0, 1.0, 1.0, 2.0, 2.0, 3.0]})
        df_c = pd.DataFrame({"v": [1.0, 2.0, 3.0, 3.0, 3.0, 3.0]})
        w_all = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        dfs: List[DataFrameWithWeight] = [
            {"df": df_a, "weight": w_all},
            {"df": df_b, "weight": w_all},
            {"df": df_c, "weight": w_all},
        ]
        result = ascii_comparative_hist(
            dfs, names=["Baseline", "Left", "Right"], column="v", n_bins=3, bar_width=20
        )
        self._assert_lines_equal(
            result,
            """\
            Range        | Baseline (%)    | Left (%)             | Right (%)
            ---------------------------------------------------------------------------------
            [1.00, 1.67) | ██████████ 33.3 | ██████████▒▒▒▒▒ 50.0 | █████    ] 16.7
            [1.67, 2.33) | ██████████ 33.3 | ██████████ 33.3      | █████    ] 16.7
            [2.33, 3.00] | ██████████ 33.3 | █████    ] 16.7      | ██████████▒▒▒▒▒▒▒▒▒▒ 66.7
            ---------------------------------------------------------------------------------
            Total        | 100.0           | 100.0                | 100.0

            Key: █ = shared with Baseline, ▒ = excess,    ] = deficit
            """,
        )

    def test_e2e_comparative_hist_empty_data(self) -> None:
        """Empty data returns 'No data available.' message."""
        df_empty = pd.DataFrame({"v": pd.Series([], dtype=float)})
        dfs: List[DataFrameWithWeight] = [
            {"df": df_empty, "weight": pd.Series([], dtype=float)},
        ]
        result = ascii_comparative_hist(
            dfs, names=["Empty"], column="v", n_bins=2, bar_width=20
        )
        self.assertEqual(result, "No data available.")


class TestAsciiPlotsAdjustmentEndToEnd(balance.testutil.BalanceTestCase):
    """End-to-end test: adjust a biased sample and verify ASCII plot output."""

    def _assert_lines_equal(self, actual: str, expected_text: str) -> None:
        """Compare actual output lines against a dedented expected string."""
        expected_lines = textwrap.dedent(expected_text).strip().splitlines()
        actual_lines = [line.rstrip() for line in actual.splitlines()]
        self.assertEqual(actual_lines, expected_lines)

    def test_full_pipeline_adjust_and_ascii_plot(self) -> None:
        """Create biased sample, adjust to target, and verify ASCII comparison plot."""
        from balance.sample_class import Sample

        # Biased sample: overrepresents "male" (75%) and "young" (62.5%)
        sample_df = pd.DataFrame(
            {
                "gender": ["male"] * 6 + ["female"] * 2,
                "age_group": ["young"] * 5 + ["old"] * 3,
                "id": list(range(1, 9)),
            }
        )
        sample = Sample.from_frame(sample_df, id_column="id")

        # Target population: balanced (50/50 for both)
        target_df = pd.DataFrame(
            {
                "gender": ["male", "male", "female", "female"],
                "age_group": ["young", "old", "young", "old"],
                "id": list(range(1, 5)),
            }
        )
        target = Sample.from_frame(target_df, id_column="id")

        # Adjust to correct the bias
        adjusted = sample.set_target(target).adjust(method="ipw")

        # Generate ASCII plot comparing unadjusted, adjusted, and target
        with patch("sys.stdout", new_callable=io.StringIO):
            result = adjusted.covars().plot(library="balance", bar_width=20)

        assert isinstance(result, str)

        # Verify the full output matches expected ASCII plots.
        #
        # The plot shows three datasets per variable:
        #   █ sample     = unadjusted (original biased sample)
        #   ▒ adjusted   = after IPW bias correction
        #   ▐ population = target population
        #
        # For gender: sample is 75% male / 25% female, population is 50/50.
        #   IPW adjustment shifts adjusted slightly toward the target.
        # For age_group: sample is 62.5% young / 37.5% old, population is 50/50.
        self._assert_lines_equal(
            result,
            """\
            === gender (categorical) ===

            Category | sample  adjusted  population
                     |
            female   | ███████ (25.0%)
                     | ▒▒▒▒▒▒▒ (26.2%)
                     | ▐▐▐▐▐▐▐▐▐▐▐▐▐ (50.0%)
            male     | ████████████████████ (75.0%)
                     | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (73.8%)
                     | ▐▐▐▐▐▐▐▐▐▐▐▐▐ (50.0%)

            Legend: █ sample  ▒ adjusted  ▐ population
            Bar lengths are proportional to weighted frequency within each dataset.

            === age_group (categorical) ===

            Category | sample  adjusted  population
                     |
            old      | ████████████ (37.5%)
                     | ▒▒▒▒▒▒▒▒▒▒▒▒ (38.6%)
                     | ▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐ (50.0%)
            young    | ████████████████████ (62.5%)
                     | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (61.4%)
                     | ▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐ (50.0%)

            Legend: █ sample  ▒ adjusted  ▐ population
            Bar lengths are proportional to weighted frequency within each dataset.
            """,
        )


class TestAsciiPlotDistTypeWarning(balance.testutil.BalanceTestCase):
    """Tests that dist_type warnings are logged for ASCII plot functions."""

    def test_ascii_plot_bar_warns_on_unsupported_dist_type(self) -> None:
        """Test ascii_plot_bar logs warning for non-hist_ascii dist_type."""
        import logging

        df = pd.DataFrame({"g": ["a", "b"]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([1.0, 1.0])},
        ]
        with self.assertLogs("balance.stats_and_plots", level=logging.WARNING) as ctx:
            ascii_plot_bar(dfs, names=["self"], column="g", dist_type="kde")
        self.assertTrue(
            any("only support dist_type='hist_ascii'" in m for m in ctx.output)
        )

    def test_ascii_plot_hist_warns_on_unsupported_dist_type(self) -> None:
        """Test ascii_plot_hist logs warning for non-hist_ascii dist_type."""
        import logging

        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([1.0, 1.0, 1.0])},
        ]
        with self.assertLogs("balance.stats_and_plots", level=logging.WARNING) as ctx:
            ascii_plot_hist(dfs, names=["self"], column="v", dist_type="kde")
        self.assertTrue(
            any("only support dist_type='hist_ascii'" in m for m in ctx.output)
        )

    def test_ascii_plot_dist_warns_on_unsupported_dist_type(self) -> None:
        """Test ascii_plot_dist logs warning for non-hist_ascii dist_type."""
        import logging

        df = pd.DataFrame({"v": ["a", "b"]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([1.0, 1.0])},
        ]
        with self.assertLogs("balance.stats_and_plots", level=logging.WARNING) as ctx:
            with patch("sys.stdout", new_callable=io.StringIO):
                ascii_plot_dist(dfs, names=["self"], dist_type="kde")
        self.assertTrue(
            any("only support dist_type='hist_ascii'" in m for m in ctx.output)
        )

    def test_no_warning_when_dist_type_is_hist_ascii(self) -> None:
        """Test no warning is logged when dist_type='hist_ascii'."""
        import logging

        df = pd.DataFrame({"v": ["a", "b"]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([1.0, 1.0])},
        ]
        logger = logging.getLogger("balance.stats_and_plots")
        with patch.object(logger, "warning") as mock_warn:
            ascii_plot_bar(dfs, names=["self"], column="v", dist_type="hist_ascii")
        mock_warn.assert_not_called()


class TestRenderHorizontalBarsEdgeCases(balance.testutil.BalanceTestCase):
    """Tests for edge cases in _render_horizontal_bars."""

    def test_zero_max_value(self) -> None:
        """Test that max_value=0 produces bars of length 0."""
        result = _render_horizontal_bars(
            label="cat",
            proportions={"sample": 0.0},
            legend_names=["sample"],
            bar_width=20,
            max_value=0.0,
            label_width=3,
        )
        # bar_len should be 0, so no fill characters
        self.assertNotIn("█", result)
        self.assertIn("0.0%", result)


class TestAsciiComparativeHistEdgeCases(balance.testutil.BalanceTestCase):
    """Tests for edge cases in ascii_comparative_hist."""

    def test_all_zero_weights(self) -> None:
        """Test with all-NaN data produces 'No data available.'."""
        df = pd.DataFrame({"v": [np.nan, np.nan]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([1.0, 1.0])},
        ]
        result = ascii_comparative_hist(
            dfs, names=["Target"], column="v", n_bins=2, bar_width=10
        )
        self.assertEqual(result, "No data available.")

    def test_weighted_with_nas(self) -> None:
        """Test ascii_comparative_hist with weighted data containing NAs."""
        df = pd.DataFrame({"v": [1.0, 2.0, np.nan, 4.0]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([1.0, 2.0, 1.0, 1.0])},
            {"df": df, "weight": pd.Series([2.0, 1.0, 1.0, 2.0])},
        ]
        result = ascii_comparative_hist(
            dfs, names=["Target", "Sample"], column="v", n_bins=2, bar_width=10
        )
        # Should produce output without error, with NAs removed
        self.assertIn("Target", result)
        self.assertIn("Sample", result)
        self.assertIn("Total", result)

    def test_all_zero_weights_two_datasets(self) -> None:
        """Test ascii_comparative_hist with all-zero weights (max_pct == 0)."""
        df = pd.DataFrame({"v": [1.0, 2.0]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([0.0, 0.0])},
            {"df": df, "weight": pd.Series([0.0, 0.0])},
        ]
        result = ascii_comparative_hist(
            dfs, names=["Target", "Sample"], column="v", n_bins=2, bar_width=10
        )
        # All bins should show 0.0, totals should be 0.0
        self.assertIn("0.0", result)
        self.assertIn("Total", result)
        # Data rows should have no bars (only the Key line has █)
        data_lines = [
            line
            for line in result.splitlines()
            if not line.startswith("Key:") and not line.startswith("---")
        ]
        for line in data_lines:
            if "0.0" in line and "Total" not in line:
                # Data bin lines should have no fill characters
                self.assertNotIn("▒", line)

    def test_unweighted(self) -> None:
        """Test ascii_comparative_hist with weighted=False."""
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([10.0, 10.0, 10.0, 10.0])},
        ]
        result = ascii_comparative_hist(
            dfs, names=["Base"], column="v", n_bins=2, bar_width=10, weighted=False
        )
        # Without weighting, each value counts equally
        self.assertIn("50.0", result)

    def test_identical_values(self) -> None:
        """Test ascii_comparative_hist when all values are identical (lines 463-464)."""
        df = pd.DataFrame({"v": [5.0, 5.0, 5.0]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([1.0, 1.0, 1.0])},
        ]
        result = ascii_comparative_hist(
            dfs, names=["Base"], column="v", n_bins=2, bar_width=10
        )
        # Should not error; global_min == global_max triggers the ±0.5 adjustment
        self.assertIn("Base", result)
        self.assertIn("100.0", result)

    def test_deficit_of_one(self) -> None:
        """Test ascii_comparative_hist deficit==1 renders ']' (lines 519-520)."""
        # Craft data so that one bin has a deficit of exactly 1 character
        # Baseline: 2 values in bin0, 2 in bin1 => 50%/50%
        # Comparison: 3 values in bin0, 1 in bin1
        # With bar_width=10: baseline_len=5 for 50%, comparison bin1 => 1/4=25% => bar_len=2.5->round(2 or 3)
        # We need precise control: bar_width=4
        # Baseline 50% => baseline_len = round(50/50*4) = 4
        # Comparison bin1: 25% => bar_len = round(25/50*4) = round(2) = 2
        # deficit = 4 - 2 = 2, which is >= 2, not == 1
        # Let's try bar_width=3:
        # Baseline 50% => round(50/50*3) = 3
        # Comparison bin1: 25% => round(25/50*3) = round(1.5) = 2
        # deficit = 3 - 2 = 1 => exactly deficit == 1!
        df_a = pd.DataFrame({"v": [0.5, 0.5, 1.5, 1.5]})
        df_b = pd.DataFrame({"v": [0.5, 0.5, 0.5, 1.5]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df_a, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
            {"df": df_b, "weight": pd.Series([1.0, 1.0, 1.0, 1.0])},
        ]
        result = ascii_comparative_hist(
            dfs, names=["Base", "Test"], column="v", n_bins=2, bar_width=3
        )
        # deficit==1 should render as "██]" (bar_len chars + "]")
        self.assertIn("]", result)

    def test_deficit_of_zero(self) -> None:
        """Test ascii_comparative_hist deficit==0 renders no bracket (lines 521-522)."""
        # Same proportions in both datasets => deficit is 0 for all bins
        df = pd.DataFrame({"v": [0.5, 1.5]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([1.0, 1.0])},
            {"df": df, "weight": pd.Series([1.0, 1.0])},
        ]
        result = ascii_comparative_hist(
            dfs, names=["Base", "Same"], column="v", n_bins=2, bar_width=10
        )
        # No deficit or excess in the "Same" column
        for line in result.splitlines():
            if "Same (%)" in line or line.startswith("---") or line.startswith("Key:"):
                continue
            # Check the Same column portion (after the second |)
            parts = line.split("|")
            if len(parts) >= 3:
                same_col = parts[2]
                self.assertNotIn("▒", same_col)
                # The ] in same_col would only be a deficit bracket, not a bin label
                self.assertNotIn("]", same_col)


class TestAsciiPlotDistAllNanVariable(balance.testutil.BalanceTestCase):
    """Test ascii_plot_dist skips all-NaN variables (lines 680-681)."""

    def test_skips_all_nan_variable(self) -> None:
        """Test that a variable with all NaN values is skipped with a warning."""
        import logging

        df = pd.DataFrame({"good": ["a", "b", "c"], "bad": [np.nan, np.nan, np.nan]})
        dfs: List[DataFrameWithWeight] = [
            {"df": df, "weight": pd.Series([1.0, 1.0, 1.0])},
        ]
        with self.assertLogs("balance.stats_and_plots", level=logging.WARNING) as ctx:
            with patch("sys.stdout", new_callable=io.StringIO):
                result = ascii_plot_dist(dfs, names=["self"])
        # The good variable should be plotted
        self.assertIn("good", result)
        # The warning should mention the bad variable
        self.assertTrue(any("bad" in m for m in ctx.output))
