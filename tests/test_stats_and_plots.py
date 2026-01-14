# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Any, cast

import balance.testutil
import numpy as np
import pandas as pd
from balance.util import _verify_value_type


class TestBalance_weights_stats(
    balance.testutil.BalanceTestCase,
):
    def test__check_weights_are_valid(self) -> None:
        """Test validation of weight arrays for statistical calculations.

        Verifies that _check_weights_are_valid correctly validates different
        weight input formats (list, numpy array, pandas Series/DataFrame)
        and raises appropriate errors for invalid inputs like negative weights
        or non-numeric values.
        """
        from balance.stats_and_plots.weights_stats import _check_weights_are_valid

        w = [np.inf, 1, 2, 1.0, 0]

        # Test various valid input formats
        self.assertEqual(_check_weights_are_valid(w), None)
        self.assertEqual(_check_weights_are_valid(np.array(w)), None)
        self.assertEqual(_check_weights_are_valid(pd.Series(w)), None)
        self.assertEqual(_check_weights_are_valid(pd.DataFrame(w)), None)
        self.assertEqual(_check_weights_are_valid(pd.DataFrame({"a": w})), None)
        self.assertEqual(
            _check_weights_are_valid(pd.DataFrame({"a": w, "b": [str(x) for x in w]})),
            None,
        )  # checking only the first column

        # Test invalid weight types
        with self.assertRaisesRegex(TypeError, "weights \\(w\\) must be a number*"):
            _check_weights_are_valid([str(x) for x in w])

        with self.assertRaisesRegex(TypeError, "weights \\(w\\) must be a number*"):
            invalid_w = ["a", "b"]
            _check_weights_are_valid(invalid_w)

        # Test invalid weight values (negative)
        with self.assertRaisesRegex(
            ValueError, "weights \\(w\\) must all be non-negative values."
        ):
            negative_w = [-1, 0, 1]
            _check_weights_are_valid(negative_w)

    def test_design_effect(self) -> None:
        """Test calculation of design effect for weighted samples.

        Design effect measures the loss of precision due to weighting.
        Tests with equal weights (design effect = 1) and unequal weights
        to verify correct calculation and return type.
        """
        from balance.stats_and_plots.weights_stats import design_effect

        self.assertEqual(design_effect(pd.Series((1, 1, 1, 1))), 1)
        self.assertEqual(
            design_effect(pd.Series((0, 1, 2, 3))),
            1.555_555_555_555_555_6,
        )
        self.assertEqual(type(design_effect(pd.Series((0, 1, 2, 3)))), np.float64)

    def test_nonparametric_skew(self) -> None:
        """Test calculation of nonparametric skewness measure.

        Tests skewness calculation for various distributions including
        symmetric (skew = 0), single values, and right-skewed distributions
        to verify correct nonparametric skewness computation.
        """
        from balance.stats_and_plots.weights_stats import nonparametric_skew

        self.assertEqual(nonparametric_skew(pd.Series((1, 1, 1, 1))), 0)
        self.assertEqual(nonparametric_skew(pd.Series((1))), 0)
        self.assertEqual(nonparametric_skew(pd.Series((1, 2, 3, 4))), 0)
        self.assertEqual(nonparametric_skew(pd.Series((1, 1, 1, 2))), 0.5)

    def test_prop_above_and_below(self) -> None:
        """Test calculation of proportions above and below thresholds.

        Tests the prop_above_and_below function with default thresholds,
        custom thresholds, and different return formats to ensure correct
        proportion calculations and proper handling of edge cases.
        """
        from balance.stats_and_plots.weights_stats import prop_above_and_below

        # Test with identical values
        result1 = prop_above_and_below(pd.Series((1, 1, 1, 1)))
        self.assertIsNotNone(result1)
        result1 = _verify_value_type(result1, pd.Series)  # Type narrowing for pyre
        self.assertEqual(
            result1.astype(int).to_list(),
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        )

        # Test with varying values
        result2 = prop_above_and_below(pd.Series((1, 2, 3, 4)))
        self.assertIsNotNone(result2)
        result2 = _verify_value_type(result2, pd.Series)  # Type narrowing for pyre
        self.assertEqual(
            result2.to_list(),
            [0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        )

        # Test custom thresholds
        result = prop_above_and_below(
            pd.Series((1, 2, 3, 4)), below=(0.1, 0.5), above=(2, 3)
        )
        self.assertIsNotNone(result)
        result = _verify_value_type(result, pd.Series)  # Type narrowing for pyre
        self.assertEqual(result.to_list(), [0.0, 0.25, 0.0, 0.0])
        self.assertEqual(
            result.index.to_list(),
            ["prop(w < 0.1)", "prop(w < 0.5)", "prop(w >= 2)", "prop(w >= 3)"],
        )

        # Test with None parameters
        self.assertEqual(
            prop_above_and_below(pd.Series((1, 2, 3, 4)), above=None, below=None), None
        )

        # Test return_as_series = False
        result_dict = prop_above_and_below(
            pd.Series((1, 2, 3, 4)), return_as_series=False
        )
        self.assertIsNotNone(result_dict)
        result_dict = _verify_value_type(result_dict)
        expected = {
            "below": [0.0, 0.0, 0.0, 0.25, 0.5],
            "above": [0.5, 0.0, 0.0, 0.0, 0.0],
        }
        self.assertEqual({k: v.to_list() for k, v in result_dict.items()}, expected)

    def test_weighted_median_breakdown_point(self) -> None:
        """Test calculation of weighted median breakdown point.

        Tests the weighted median breakdown point calculation which measures
        the robustness of the weighted median. Tests with equal and unequal
        weights to verify correct breakdown point calculations.
        """
        import pandas as pd
        from balance.stats_and_plots.weights_stats import (
            weighted_median_breakdown_point,
        )

        self.assertEqual(weighted_median_breakdown_point(pd.Series((1, 1, 1, 1))), 0.5)
        self.assertEqual(weighted_median_breakdown_point(pd.Series((2, 2, 2, 2))), 0.5)
        self.assertEqual(
            weighted_median_breakdown_point(pd.Series((1, 1, 1, 10))), 0.25
        )
        self.assertEqual(
            weighted_median_breakdown_point(pd.Series((1, 1, 1, 1, 10))), 0.2
        )

    def test_design_effect_with_two_values(self) -> None:
        """Test design effect with two weight values.

        Tests the simplest non-trivial case with two different weights.
        """
        from balance.stats_and_plots.weights_stats import design_effect

        result = design_effect(pd.Series([1, 3]))
        # E[W^2] = ((1^2 + 3^2) / 2) = 5
        # E[W]^2 = ((1 + 3) / 2)^2 = 4
        # Deff = 5 / 4 = 1.25
        self.assertEqual(result, 1.25)

    def test_nonparametric_skew_with_two_elements(self) -> None:
        """Test nonparametric skew with two elements.

        Tests the simplest non-trivial case with two different weights.
        """
        from balance.stats_and_plots.weights_stats import nonparametric_skew

        # For [1, 3]: mean=2, median=2, so skew should be 0
        self.assertEqual(nonparametric_skew(pd.Series([1, 3])), 0.0)

        # For [1, 2]: mean=1.5, median=1.5, so skew should be 0
        self.assertEqual(nonparametric_skew(pd.Series([1, 2])), 0.0)

    def test_nonparametric_skew_with_left_skewed_distribution(self) -> None:
        """Test nonparametric skew with left-skewed (negative) distribution.

        Verifies that left-skewed distributions produce negative skew values.
        For left skew with positive weights, we need mean < median.
        """
        from balance.stats_and_plots.weights_stats import nonparametric_skew

        # Left-skewed: one small value, rest larger values
        # [1, 10, 10, 10] -> mean = 31/4 = 7.75, median = 10, skew = (7.75-10)/std < 0
        result = nonparametric_skew(pd.Series([1, 10, 10, 10]))
        self.assertLess(result, 0)

    def test_nonparametric_skew_with_large_positive_skew(self) -> None:
        """Test nonparametric skew with highly right-skewed distribution.

        Verifies calculation with extreme right skew.
        """
        from balance.stats_and_plots.weights_stats import nonparametric_skew

        # Extreme right skew: many small values, one very large
        result = nonparametric_skew(pd.Series([1, 1, 1, 1, 1, 100]))
        # Result should be positive for right-skewed distribution
        self.assertGreater(result, 0)

    def test_weighted_median_breakdown_point_with_two_weights(self) -> None:
        """Test breakdown point with two equal weights.

        With two equal weights [1,1], normalized is [0.5, 0.5].
        Cumsum of sorted: [0.5, 1.0]. Count <= 0.5 is 1, so 1/2 = 0.5.
        """
        from balance.stats_and_plots.weights_stats import (
            weighted_median_breakdown_point,
        )

        result = weighted_median_breakdown_point(pd.Series([1, 1]))
        self.assertEqual(result, 0.5)

    def test_weighted_median_breakdown_point_with_one_weight_above_50_percent(
        self,
    ) -> None:
        """Test breakdown point when one weight exceeds 50%.

        When single weight is > 50% of total, breakdown point is 1/n.
        """
        from balance.stats_and_plots.weights_stats import (
            weighted_median_breakdown_point,
        )

        # One weight has 60% of total
        result = weighted_median_breakdown_point(pd.Series([60, 20, 20]))
        # 60/(60+20+20) = 60/100 = 0.6 > 0.5, so need 1 observation = 1/3
        self.assertAlmostEqual(result, 1 / 3, places=10)

    def test_weighted_median_breakdown_point_with_all_zeros_except_one(self) -> None:
        """Test breakdown point when only one weight is non-zero.

        When only one weight is non-zero, it has 100% of weight.
        """
        from balance.stats_and_plots.weights_stats import (
            weighted_median_breakdown_point,
        )

        result = weighted_median_breakdown_point(pd.Series([0, 0, 0, 10]))
        # Single non-zero weight has 100% > 50%, need 1 obs out of 4
        self.assertEqual(result, 0.25)

    def test_weighted_median_breakdown_point_with_gradual_distribution(self) -> None:
        """Test breakdown point with gradually increasing weights.

        Verifies calculation with a more complex weight distribution.
        """
        from balance.stats_and_plots.weights_stats import (
            weighted_median_breakdown_point,
        )

        # Weights: [10, 8, 6, 4, 2], total = 30
        # Normalized: [1/3, 4/15, 1/5, 2/15, 1/15]
        # Sorted desc cumsum: [1/3≈0.33, 0.6, 0.8, 0.93, 1.0]
        # Count where cumsum <= 0.5: only first element (1/3) is <= 0.5, so count = 1
        # So breakdown point = 1/5 = 0.2
        result = weighted_median_breakdown_point(pd.Series([10, 8, 6, 4, 2]))
        self.assertEqual(result, 0.2)


class TestBalance_weighted_stats(
    balance.testutil.BalanceTestCase,
):
    def test__prepare_weighted_stat_args(self) -> None:
        """Test preparation of arguments for weighted statistical functions.

        Tests the _prepare_weighted_stat_args function which standardizes
        input formats for weighted statistics, handles different data types,
        manages infinite values, and validates input consistency.
        """
        from balance.stats_and_plots.weighted_stats import _prepare_weighted_stat_args

        v, w = [-1, 0, 1, np.inf], [np.inf, 1, 2, 1.0]
        v2, w2 = _prepare_weighted_stat_args(v, w, False)
        # assert the new types:
        self.assertEqual(type(v2), pd.DataFrame)
        self.assertEqual(type(w2), pd.Series)

        v, w = pd.Series([-1, 0, 1, np.inf]), pd.Series([np.inf, 1, 2, 1.0])
        v2, w2 = _prepare_weighted_stat_args(v, w, False)
        # assert the new types:
        self.assertEqual(type(v2), pd.DataFrame)
        self.assertEqual(type(w2), pd.Series)
        # check the values are the same:
        self.assertEqual(v.to_list(), v2[0].to_list())
        self.assertEqual(w.to_list(), w2.to_list())
        # Do we have any inf or nan?
        self.assertTrue(any(np.isinf(v2[0])))
        self.assertTrue(any(np.isinf(w2)))
        self.assertTrue(not any(np.isnan(v2[0])))
        self.assertTrue(not any(np.isnan(w2)))

        # Checking how it works when inf_rm = True
        v2, w2 = _prepare_weighted_stat_args(v, w, True)
        # assert the new types:
        self.assertEqual(type(v2), pd.DataFrame)
        self.assertEqual(type(w2), pd.Series)
        # check the values are the NOT same (because the Inf was turned to nan):
        self.assertNotEqual(v.to_list(), v2[0].to_list())
        self.assertNotEqual(w.to_list(), w2.to_list())
        # Do we have any inf or nan?
        self.assertTrue(not any(np.isinf(v2[0])))
        self.assertTrue(not any(np.isinf(w2)))
        self.assertTrue(any(np.isnan(v2[0])))
        self.assertTrue(any(np.isnan(w2)))

        # Check that it catches wrong input types
        with self.assertRaises(TypeError):
            v, w = pd.Series([1, 2]), "wrong_type"  # type: ignore[assignment]
            v2, w2 = _prepare_weighted_stat_args(v, w)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            v, w = pd.Series([1, 2]), (1, 2)  # type: ignore[assignment]
            v2, w2 = _prepare_weighted_stat_args(v, w)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            v, w = (1, 2), pd.Series([1, 2])
            v2, w2 = _prepare_weighted_stat_args(v, w)
        with self.assertRaises(ValueError):
            v, w = pd.Series([1, 2, 3]), pd.Series([1, 2])
            v2, w2 = _prepare_weighted_stat_args(v, w)
        with self.assertRaises(TypeError):
            v, w = pd.Series(["a", "b"]), pd.Series([1, 2])
            v2, w2 = _prepare_weighted_stat_args(v, w)

        # check other input types for v and w
        # np.array
        v, w = np.array([-1, 0, 1, np.inf]), np.array([np.inf, 1, 2, 1.0])
        v2, w2 = _prepare_weighted_stat_args(v, w, False)
        # assert the new types:
        self.assertEqual(type(v2), pd.DataFrame)
        self.assertEqual(type(w2), pd.Series)
        # pd.DataFrame
        v, w = pd.DataFrame([-1, 0, 1, np.inf]), np.array([np.inf, 1, 2, 1.0])
        v2, w2 = _prepare_weighted_stat_args(v, w, False)
        # assert the new types:
        self.assertEqual(type(v2), pd.DataFrame)
        self.assertEqual(type(w2), pd.Series)
        # np.array (replacing np.matrix)
        v, w = (
            np.array([-1, 0, 1, np.inf]).reshape(-1, 1),
            np.array([np.inf, 1, 2, 1.0]),
        )
        v2, w2 = _prepare_weighted_stat_args(v, w, False)
        # assert the new types:
        self.assertEqual(type(v2), pd.DataFrame)
        self.assertEqual(type(w2), pd.Series)

        # Dealing with w=None
        v, w = pd.Series([-1, 0, 1, np.inf]), None
        v2, w2 = _prepare_weighted_stat_args(v, w, False)
        self.assertEqual(type(v2), pd.DataFrame)
        self.assertEqual(type(w2), pd.Series)
        self.assertEqual(w2, pd.Series([1.0, 1.0, 1.0, 1.0]))
        self.assertEqual(len(w2), 4)
        # Verify defaults:
        self.assertEqual(
            _prepare_weighted_stat_args(v, None, False)[0],
            _prepare_weighted_stat_args(v)[0],
        )

        with self.assertRaises(ValueError):
            v, w = pd.Series([-1, 0, 1, np.inf]), pd.Series([np.inf, 1, -2, 1.0])
            v2, w2 = _prepare_weighted_stat_args(v, w)

    def test_weighted_mean(self) -> None:
        """Test calculation of weighted mean for various input types.

        Tests weighted mean calculation with different input formats,
        handling of infinite values, None/NaN values, and validates
        proper weight application and index handling.
        """
        from balance.stats_and_plots.weighted_stats import weighted_mean

        # No weights
        self.assertEqual(weighted_mean(pd.Series([-1, 0, 1, 2])), pd.Series(0.5))
        self.assertEqual(weighted_mean(pd.Series([-1, None, 1, 2])), pd.Series(0.5))

        # No weights, with one inf value
        self.assertEqual(
            weighted_mean(pd.Series([-1, np.inf, 1, 2])),
            pd.Series(np.inf),
        )
        self.assertEqual(
            weighted_mean(pd.Series([-1, np.inf, 1, 2]), inf_rm=True),
            pd.Series(0.5),
        )

        # Inf value in weights
        self.assertTrue(
            all(
                np.isnan(
                    weighted_mean(
                        pd.Series([-1, 2, 1, 2]), w=pd.Series([1, np.inf, 1, 1])
                    )
                )
            )
        )
        self.assertEqual(
            weighted_mean(
                pd.Series([-1, 2, 1, 2]), w=pd.Series([1, np.inf, 1, 1]), inf_rm=True
            ),
            pd.Series(2 / 3),
        )

        # With weights
        self.assertEqual(
            weighted_mean(pd.Series([-1, 2, 1, 2]), w=pd.Series([1, 2, 3, 4])),
            pd.Series(1.4),
        )

        self.assertEqual(
            weighted_mean(pd.Series([-1, 2, 1, 2]), w=pd.Series([1, None, 1, 1])),
            pd.Series(2 / 3),
        )

        # Notice that while None values are ignored, their weights will still be used in the denominator
        # Hence, None in values acts like 0
        self.assertEqual(
            weighted_mean(pd.Series([-1, None, 1, 2]), w=pd.Series([1, 0, 1, 1])),
            pd.Series(2 / 3),
        )
        self.assertEqual(
            weighted_mean(pd.Series([-1, np.nan, 1, 2]), w=pd.Series([1, 0, 1, 1])),
            pd.Series(2 / 3),
        )
        self.assertEqual(
            weighted_mean(pd.Series([-1, None, 1, 2]), w=pd.Series([1, 1, 1, 1])),
            pd.Series(1 / 2),
        )
        # None in the values is just like 0
        self.assertEqual(
            weighted_mean(pd.Series([-1, None, 1, 2]), w=pd.Series([1, 1, 1, 1])),
            weighted_mean(pd.Series([-1, 0, 1, 2]), w=pd.Series([1, 1, 1, 1])),
        )

        # pd.DataFrame v
        d = pd.DataFrame([(-1, 2, 1, 2), (1, 2, 3, 4)]).transpose()
        pd.testing.assert_series_equal(weighted_mean(d), pd.Series((1, 2.5)))
        pd.testing.assert_series_equal(
            weighted_mean(d, w=pd.Series((1, 2, 3, 4))),
            pd.Series((1.4, 3)),
        )

        # np.array v (replacing np.matrix)
        d = np.array([(-1, 2, 1, 2), (1, 2, 3, 4)]).transpose()
        pd.testing.assert_series_equal(weighted_mean(d), pd.Series((1, 2.5)))
        pd.testing.assert_series_equal(
            weighted_mean(d, w=pd.Series((1, 2, 3, 4))),
            pd.Series((1.4, 3)),
        )

        #  Test input validation
        with self.assertRaisesRegex(TypeError, "must be numeric"):
            weighted_mean(pd.Series(["a", "b"]))
        with self.assertRaisesRegex(TypeError, "must be numeric"):
            weighted_mean(pd.DataFrame({"a": [1, 2], "b": ["a", "b"]}))

        with self.assertRaisesRegex(ValueError, "must have same number of rows"):
            weighted_mean(pd.Series([1, 2]), pd.Series([1, 2, 3]))
        with self.assertRaisesRegex(ValueError, "must have same number of rows"):
            weighted_mean(pd.DataFrame({"a": [1, 2]}), pd.Series([1]))

        # Make sure that index is ignored
        self.assertEqual(
            weighted_mean(
                pd.Series([-1, 2, 1, 2], index=(0, 1, 2, 3)),
                w=pd.Series([1, 2, 3, 4], index=(5, 6, 7, 8)),
                inf_rm=True,
            ),
            pd.Series((-1 * 1 + 2 * 2 + 1 * 3 + 2 * 4) / (1 + 2 + 3 + 4)),
        )

        self.assertEqual(
            weighted_mean(
                pd.Series([-1, 2, 1, 2], index=(0, 1, 2, 3)),
                w=pd.Series([1, 2, 3, 4], index=(0, 6, 7, 8)),
                inf_rm=True,
            ),
            pd.Series((-1 * 1 + 2 * 2 + 1 * 3 + 2 * 4) / (1 + 2 + 3 + 4)),
        )

    def test_var_of_weighted_mean(self) -> None:
        """Test calculation of variance of weighted mean.

        Tests the variance calculation of weighted means with and without
        weights to verify correct statistical computations and R compatibility.
        """
        from balance.stats_and_plots.weighted_stats import var_of_weighted_mean

        # Test no weights assigned
        #  In R: sum((1:4 - mean(1:4))^2 / 4) / (4)
        #  [1] 0.3125
        self.assertEqual(
            var_of_weighted_mean(pd.Series((1, 2, 3, 4))), pd.Series(0.3125)
        )

        #  For a reproducible R example, see: https://gist.github.com/talgalili/b92cd8cdcbfc287e331a8f27db265c00
        self.assertEqual(
            var_of_weighted_mean(pd.Series((1, 2, 3, 4)), pd.Series((1, 2, 3, 4))),
            pd.Series(0.24),
        )

    def test_ci_of_weighted_mean(self) -> None:
        """Test calculation of confidence intervals for weighted means.

        Tests confidence interval calculations for weighted means with
        different confidence levels and validates proper interval bounds
        for various input data types.
        """
        from balance.stats_and_plots.weighted_stats import ci_of_weighted_mean

        self.assertEqual(
            ci_of_weighted_mean(pd.Series((1, 2, 3, 4)), round_ndigits=3).to_list(),
            [(1.404, 3.596)],
        )

        self.assertEqual(
            ci_of_weighted_mean(
                pd.Series((1, 2, 3, 4)), pd.Series((1, 2, 3, 4)), round_ndigits=3
            ).to_list(),
            [(2.04, 3.96)],
        )

        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 1, 1, 1]})
        w = pd.Series((1, 2, 3, 4))
        self.assertEqual(
            ci_of_weighted_mean(df, w, conf_level=0.99, round_ndigits=3).to_dict(),
            {"a": (1.738, 4.262), "b": (1.0, 1.0)},
        )

    def test_weighted_var(self) -> None:
        """Test calculation of weighted variance.

        Tests weighted variance calculations with and without weights
        to verify correct statistical computations and R compatibility.
        """
        from balance.stats_and_plots.weighted_stats import weighted_var

        #  > var(c(1, 2, 3, 4))
        #  [1] 1.66667
        self.assertEqual(weighted_var(pd.Series((1, 2, 3, 4))), pd.Series(5 / 3))

        #  > SDMTools::wt.var(c(1, 2), c(1, 2))
        #  [1] 0.5
        self.assertEqual(
            weighted_var(pd.Series((1, 2)), pd.Series((1, 2))), pd.Series(0.5)
        )

    def test_weighted_sd(self) -> None:
        """Test calculation of weighted standard deviation.

        Tests weighted standard deviation calculations with various inputs
        and validates against manual calculations and R compatibility.
        """
        from balance.stats_and_plots.weighted_stats import weighted_sd

        #  > sd(c(1, 2, 3, 4))
        #  [1] 1.290994
        self.assertEqual(weighted_sd(pd.Series((1, 2, 3, 4))), pd.Series(1.290_994))

        #  > SDMTools::wt.sd(c(1, 2), c(1, 2))
        #  [1] 0.7071068
        self.assertEqual(
            weighted_sd(pd.Series((1, 2)), pd.Series((1, 2))),
            pd.Series(0.707_106_8),
        )

        x = [1, 2, 3, 4]
        x2 = pd.Series(x)
        manual_std = np.sqrt(np.sum((x2 - x2.mean()) ** 2) / (len(x) - 1))
        self.assertEqual(round(weighted_sd(x)[0], 5), round(manual_std, 5))

    def test_weighted_quantile(self) -> None:
        """Test calculation of weighted quantiles.

        Tests weighted quantile calculations with various input formats
        (pandas DataFrame, numpy matrix, numpy array) and validates
        against R's weighted quantile calculations.
        """
        from balance.stats_and_plots.weighted_stats import weighted_quantile

        self.assertEqual(
            weighted_quantile(np.arange(1, 100, 1), [0.5]).values,
            np.array(((50,),)),
        )

        # In R: reldist::wtd.quantile(c(1, 2, 3), q=c(0.5, 0.75), weight=c(1, 1, 2))
        self.assertEqual(
            weighted_quantile(np.array([1, 2, 3]), [0.5, 0.75]).values,
            np.array(((2,), (3,))),
        )

        self.assertEqual(
            weighted_quantile(np.array([1, 2, 3]), [0.5], np.array([1, 1, 2])).values,
            np.percentile([1, 2, 3, 3], 50),
        )

        # verify it indeed works with pd.DataFrame input
        # no weights
        self.assertEqual(
            weighted_quantile(
                pd.DataFrame([[1, 2, 3, 4], [1, 1, 1, 1]]).transpose(),
                [0.25, 0.5],
            ).values,
            np.array([[1.5, 1.0], [2.5, 1.0]]),
        )
        # with weights
        self.assertEqual(
            weighted_quantile(
                pd.DataFrame([[1, 2, 3, 4], [1, 1, 1, 1]]).transpose(),
                [0.25, 0.5],
                w=pd.Series([1, 1, 0, 0]),
            ).values,
            np.array([[1.0, 1.0], [1.5, 1.0]]),
        )
        self.assertEqual(
            weighted_quantile(
                pd.DataFrame([[1, 2, 3, 4], [1, 1, 1, 1]]).transpose(),
                [0.25, 0.5],
                w=pd.Series([1, 100, 1, 1]),
            ).values,
            np.array([[2.0, 1.0], [2.0, 1.0]]),
        )

        # verify it indeed works with np.array input (replacing np.matrix)
        # Note: np.matrix tests were removed due to NumPy deprecation warnings:
        # "the matrix subclass is not the recommended way to represent matrices"
        # no weights
        self.assertEqual(
            weighted_quantile(
                np.array([[1, 2, 3, 4], [1, 1, 1, 1]]).transpose(),
                [0.25, 0.5],
            ).values,
            np.array([[1.5, 1.0], [2.5, 1.0]]),
        )
        # with weights
        self.assertEqual(
            weighted_quantile(
                np.array([[1, 2, 3, 4], [1, 1, 1, 1]]).transpose(),
                [0.25, 0.5],
                w=pd.Series([1, 1, 0, 0]),
            ).values,
            np.array([[1.0, 1.0], [1.5, 1.0]]),
        )
        self.assertEqual(
            weighted_quantile(
                np.array([[1, 2, 3, 4], [1, 1, 1, 1]]).transpose(),
                [0.25, 0.5],
                w=pd.Series([1, 100, 1, 1]),
            ).values,
            np.array([[2.0, 1.0], [2.0, 1.0]]),
        )

    def test_descriptive_stats(self) -> None:
        """Test calculation of descriptive statistics with weights.

        Tests the descriptive_stats function with various statistics
        (mean, std, var_of_mean, std_mean, ci_of_mean) and validates
        consistency with individual weighted statistic functions.
        """
        from balance.stats_and_plots.weighted_stats import (
            descriptive_stats,
            weighted_mean,
            weighted_sd,
        )
        from statsmodels.stats.weightstats import DescrStatsW

        x = pd.Series([-1, 0, 1, 2])
        self.assertEqual(
            descriptive_stats(pd.DataFrame(x), stat="mean"),
            weighted_mean(x).to_frame().T,
        )
        self.assertEqual(
            descriptive_stats(pd.DataFrame(x), stat="std"), weighted_sd(x).to_frame().T
        )

        x = [1, 2, 3, 4]
        self.assertEqual(
            descriptive_stats(pd.DataFrame(x), stat="var_of_mean").to_dict(),
            {0: {0: 0.3125}},
        )

        # with weights
        x, w = pd.Series([-1, 2, 1, 2]), pd.Series([1, 2, 3, 4])
        self.assertEqual(
            descriptive_stats(pd.DataFrame(x), w, stat="mean"),
            weighted_mean(x, w).to_frame().T,
        )
        self.assertEqual(
            descriptive_stats(pd.DataFrame(x), w, stat="std"),
            weighted_sd(x, w).to_frame().T,
        )
        # show that with/without weights gives different results
        self.assertNotEqual(
            descriptive_stats(pd.DataFrame(x), stat="mean").iloc[0, 0],
            descriptive_stats(pd.DataFrame(x), w, stat="mean").iloc[0, 0],
        )
        self.assertNotEqual(
            descriptive_stats(pd.DataFrame(x), stat="std").iloc[0, 0],
            descriptive_stats(pd.DataFrame(x), w, stat="std").iloc[0, 0],
        )
        # shows that descriptive_stats can calculate std_mean and that it's smaller than std (as expected.)
        self.assertTrue(
            descriptive_stats(pd.DataFrame(x), w, stat="std").iloc[0, 0]
            > descriptive_stats(pd.DataFrame(x), w, stat="std_mean").iloc[0, 0]
        )

        x = [1, 2, 3, 4]
        x2 = pd.Series(x)
        tmp_sd = np.sqrt(np.sum((x2 - x2.mean()) ** 2) / (len(x) - 1))
        tmp_se = tmp_sd / np.sqrt(len(x))
        self.assertEqual(
            round(descriptive_stats(pd.DataFrame(x), stat="std_mean").iloc[0, 0], 5),
            round(tmp_se, 5),
        )

        # verify DescrStatsW can deal with None weights
        self.assertEqual(
            DescrStatsW([1, 2, 3], weights=[1, 1, 1]).mean,
            DescrStatsW([1, 2, 3], weights=None).mean,
        )
        self.assertEqual(
            DescrStatsW([1, 2, 3], weights=None).mean,
            2.0,
        )

        x, w = [1, 2, 3, 4], [1, 2, 3, 4]
        self.assertEqual(
            descriptive_stats(pd.DataFrame(x), w, stat="var_of_mean").to_dict(),
            {0: {0: 0.24}},
        )
        self.assertEqual(
            descriptive_stats(
                pd.DataFrame(x), w, stat="ci_of_mean", conf_level=0.99, round_ndigits=3
            ).to_dict(),
            {0: {0: (1.738, 4.262)}},
        )

        df = pd.DataFrame({"num": [1, 2, 3], "group": ["a", "b", "a"]})
        stats_num_only = descriptive_stats(df, stat="mean", formula="num")
        self.assertEqual(list(stats_num_only.columns), ["num"])
        self.assertEqual(stats_num_only.iloc[0, 0], np.mean([1, 2, 3]))

        stats_group_only = descriptive_stats(df, stat="mean", formula="group")
        self.assertNotIn("num", stats_group_only.columns)
        self.assertTrue(
            all(column.startswith("group") for column in stats_group_only.columns)
        )

        numeric_df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
        stats_single_col = descriptive_stats(numeric_df, stat="mean", formula="a")
        self.assertEqual(list(stats_single_col.columns), ["a"])
        self.assertEqual(stats_single_col.iloc[0, 0], np.mean([1, 2, 3]))

        stats_multi_formula = descriptive_stats(
            df, stat="mean", formula=["num", "group"]
        )
        self.assertIn("num", stats_multi_formula.columns)
        self.assertTrue(
            any(column.startswith("group") for column in stats_multi_formula.columns)
        )

        stats_complex = descriptive_stats(df, stat="mean", formula="num + group")
        self.assertTrue(
            all(
                column in stats_complex.columns
                for column in ("group[a]", "group[b]", "num")
            )
        )

        stats_interactions = descriptive_stats(
            df, stat="mean", formula="num + group + num:group"
        )
        self.assertTrue(
            all(
                column in stats_interactions.columns
                for column in ("num", "group[a]", "group[b]")
            )
        )
        self.assertTrue(
            all(column in stats_interactions.columns for column in ("num:group[T.b]",))
        )


class TestBalance_weighted_comparisons_stats(
    balance.testutil.BalanceTestCase,
):
    def test_outcome_variance_ratio(self) -> None:
        """Test calculation of outcome variance ratios between datasets.

        Tests the outcome_variance_ratio function which compares variance
        ratios between sample and target datasets. Uses reproducible random
        data to verify that identical datasets produce variance ratio of 1.0.
        """
        from balance.stats_and_plots.weighted_comparisons_stats import (
            outcome_variance_ratio,
        )

        # Create reproducible test data
        np.random.seed(876324)
        test_data = pd.DataFrame(np.random.rand(1000, 11))
        test_data["id"] = range(0, test_data.shape[0])
        test_data = test_data.rename(
            columns={i: "abcdefghijk"[i] for i in range(0, 11)}
        )

        self.assertEqual(
            outcome_variance_ratio(test_data[["j", "k"]], test_data[["j", "k"]]),
            pd.Series([1.0, 1.0], index=["j", "k"]),
        )

    def test__weights_per_covars_names(self) -> None:
        """Test calculation of weights per covariate names.

        Tests the _weights_per_covars_names function which assigns weights
        to covariates based on their names, handling categorical variables
        with multiple levels appropriately.
        """
        from balance.stats_and_plots.weighted_comparisons_stats import (
            _weights_per_covars_names,
        )

        asmd_df = pd.DataFrame(
            {
                "age": 0.5,
                "education[T.high_school]": 1,
                "education[T. bachelor]": 1,
                "education[T. masters]": 1,
                "education[T. phd]": 1,
            },
            index=("self",),
        )

        outcome = _weights_per_covars_names(asmd_df.columns.values.tolist()).to_dict()
        # We expect a df with 2 columns: weight and main_covar_names.
        expected = {
            "weight": {
                "age": 1.0,
                "education[T.high_school]": 0.25,
                "education[T. bachelor]": 0.25,
                "education[T. masters]": 0.25,
                "education[T. phd]": 0.25,
            },
            "main_covar_names": {
                "age": "age",
                "education[T.high_school]": "education",
                "education[T. bachelor]": "education",
                "education[T. masters]": "education",
                "education[T. phd]": "education",
            },
        }

        self.assertEqual(outcome, expected)

    def test_asmd(self) -> None:
        """Test calculation of Absolute Standardized Mean Differences (ASMD).

        Tests the asmd function which calculates standardized mean differences
        between sample and target groups, handling various input validation,
        different standardization types, and categorical variables.
        """
        from balance.stats_and_plots.weighted_comparisons_stats import asmd

        with self.assertRaisesRegex(ValueError, "sample_df must be pd.DataFrame, is*"):
            # Using wild card since it will return:
            # "sample_df must be pd.DataFrame, is* <class 'pandas.core.series.Series'>"
            asmd(
                pd.Series((0, 1, 2, 3)),  # type: ignore[arg-type]
                pd.Series((0, 1, 2, 3)),  # type: ignore[arg-type]
                pd.Series((0, 1, 2, 3)),
                pd.Series((0, 1, 2, 3)),
            )

        with self.assertRaisesRegex(ValueError, "target_df must be pd.DataFrame, is*"):
            asmd(
                pd.DataFrame({"a": (0, 1, 2, 3)}),
                pd.Series((0, 1, 2, 3)),  # type: ignore[arg-type]
                pd.Series((0, 1, 2, 3)),
                pd.Series((0, 1, 2, 3)),
            )

        # If column names are different, it will only calculate asmd for
        # the overlapping columns. The rest will be np.nan.
        # The mean(asmd) will be calculated while treating the nan values as 0s.
        self.assertEqual(
            np.isnan(
                asmd(
                    pd.DataFrame({"a": (1, 2), "b": (-1, 12)}),
                    pd.DataFrame({"a": (3, 4), "c": (5, 6)}),
                )
            ).tolist(),
            [False, True, True, False],
        )

        with self.assertRaisesRegex(ValueError, "std_type must be in*"):
            asmd(
                pd.DataFrame({"a": (1, 2), "b": (-1, 12)}),
                pd.DataFrame({"a": (3, 4), "c": (5, 6)}),
                std_type=cast(Any, "magic variance type that doesn't exist"),
            )

        # TODO: (p2) add comparison to the following numbers
        # Benchmark for the numbers:
        # ---------------------------
        #  Test data from R cobalt package:
        #  cobalt::bal.tab(
        #    data.frame(a=c(1, 2, 3, 4), b=c(-1, 12, 0, 42)),
        #    treat=c(1, 1, 0, 0),
        #    weights=c(1, 2, 1, 2),
        #    s.d.denom='control',
        #    method='weighting'
        #  )
        # Output:
        # Balance Measures
        #      Type Diff.Adj
        # a Contin.  -2.8284
        # b Contin.  -0.6847

        # Effective sample sizes
        #            Control Treated
        # Unadjusted     2.      2.
        # Adjusted       1.8     1.8

        # show the default "target" calculation is working as expected
        a1 = pd.Series((1, 2))
        b1 = pd.Series((-1, 1))
        a2 = pd.Series((3, 4))
        b2 = pd.Series((-2, 2))
        w1 = pd.Series((1, 1))
        w2 = w1
        r = asmd(
            pd.DataFrame({"a": a1, "b": b1}),
            pd.DataFrame({"a": a2, "b": b2}),
            w1,
            w2,
        ).to_dict()
        exp_a = np.abs(a1.mean() - a2.mean()) / a2.std()
        exp_b = np.abs(b1.mean() - b2.mean()) / b2.std()
        self.assertEqual((r["a"], r["b"]), (exp_a, exp_b))
        # show that the default is weights equal 1.
        r_no_weights = asmd(
            pd.DataFrame({"a": a1, "b": b1}),
            pd.DataFrame({"a": a2, "b": b2}),
        ).to_dict()
        self.assertEqual(r, r_no_weights)
        # demonstrate that weights effect the outcome (use 0 weights.)
        a1 = pd.Series((1, 2, 100))
        b1 = pd.Series((-1, 1, 100))
        a2 = pd.Series((3, 4, 100))
        b2 = pd.Series((-2, 2, 100))
        w1 = pd.Series((1, 1, 0))
        w2 = w1
        r_with_0_3rd_weight = asmd(
            pd.DataFrame({"a": a1, "b": b1}),
            pd.DataFrame({"a": a2, "b": b2}),
            w1,
            w2,
        ).to_dict()
        self.assertEqual(r, r_with_0_3rd_weight)

        # Not passing weights is the same as weights of all 1s
        r = asmd(
            pd.DataFrame({"a": (1, 2), "b": (-1, 12)}),
            pd.DataFrame({"a": (3, 4), "b": (0, 42)}),
            pd.Series((1, 2)),
            pd.Series((1, 2)),
        )
        e_asmd = pd.Series(
            (2.828_427_1, 0.684_658_9, (2.828_427_1 + 0.684_658_9) / 2),
            index=("a", "b", "mean(asmd)"),
        )
        self.assertEqual(r, e_asmd)
        self.assertEqual(type(r), pd.Series)

        r = asmd(
            pd.DataFrame({"a": (1, 2), "b": (-1, 12)}),
            pd.DataFrame({"a": (3, 4), "b": (0, 42)}),
            pd.Series((1, 2)),
            pd.Series((1, 2)),
        )
        e_asmd = pd.Series(
            (2.828_427_1, 0.684_658_9, (2.828_427_1 + 0.684_658_9) / 2),
            index=("a", "b", "mean(asmd)"),
        )
        self.assertEqual(r, e_asmd)

        # Test different std_types
        args = (
            pd.DataFrame({"a": (1, 2), "b": (-1, 12)}),
            pd.DataFrame({"a": (3, 4), "b": (0, 42)}),
            pd.Series((1, 2)),
            pd.Series((1, 2)),
        )

        r = asmd(*args, std_type="target")
        self.assertEqual(r, e_asmd)

        # TODO: this should be called : test consistency of asmd
        r = asmd(*args, std_type="sample")
        e_asmd_sample = pd.Series(
            (
                2.828_427_124_746_189_4,
                2.211_975_059_096_379,
                (2.828_427_124_746_189_4 + 2.211_975_059_096_379) / 2,
            ),
            index=("a", "b", "mean(asmd)"),
        )
        self.assertEqual(r, e_asmd_sample)

        r = asmd(*args, std_type="pooled")
        e_asmd_pooled = pd.Series(
            (2.828_427_1, 0.924_959_4, (2.828_427_1 + 0.924_959_4) / 2),
            index=("a", "b", "mean(asmd)"),
        )
        self.assertEqual(r, e_asmd_pooled)

        #  Test with categoricals
        #  Categorical variable has missing value in one df
        r = asmd(
            pd.DataFrame({"c": ("x", "y", "x", "x"), "a": (5, 6, 7, 8)}),
            pd.DataFrame({"c": ("x", "y", "x", "z"), "a": (1, 2, 3, 4)}),
            pd.Series((1, 2, 3, 4)),
            pd.Series((1, 1, 1, 1)),
        )
        e_asmd = pd.Series(
            (
                3.485_685_011_586_675_3,
                0.519_615_242_270_663_2,
                0.099_999_999_999_999_98,
                np.nan,
                (
                    3.485_685_011_586_675_3
                    + 0.519_615_242_270_663_2 * (1 / 3)
                    + 0.099_999_999_999_999_98 * (1 / 3)
                    + 0 * (1 / 3)
                )
                / (2),
            ),
            index=("a", "c[x]", "c[y]", "c[z]", "mean(asmd)"),
        )
        self.assertEqual(r, e_asmd)

        # Check that using aggregate_by_main_covar works
        a1 = pd.Series((1, 2))
        b1_A = pd.Series((1, 3))
        b1_B = pd.Series((-1, -3))
        a2 = pd.Series((3, 4))
        b2_A = pd.Series((2, 3))
        b2_B = pd.Series((-2, -3))
        w1 = pd.Series((1, 1))
        w2 = w1

        r1 = asmd(
            pd.DataFrame({"a": a1, "b[A]": b1_A, "b[B]": b1_B}),
            pd.DataFrame({"a": a2, "b[A]": b2_A, "b[B]": b2_B}),
            w1,
            w2,
        ).to_list()

        r2 = asmd(
            pd.DataFrame({"a": a1, "b[A]": b1_A, "b[B]": b1_B}),
            pd.DataFrame({"a": a2, "b[A]": b2_A, "b[B]": b2_B}),
            w1,
            w2,
            "target",
            True,
        ).to_list()

        self.assertEqual(np.round(r1, 5).tolist(), [2.82843, 0.70711, 0.70711, 1.76777])
        self.assertEqual(np.round(r2, 5).tolist(), [2.82843, 0.70711, 1.76777])

    def test__aggregate_statistic_by_main_covar(self) -> None:
        """Test aggregation of statistics values by main covariate.

        Tests the _aggregate_statistic_by_main_covar function which groups
        statistic values by their main covariate names (e.g., combining
        categorical levels) and calculates appropriate aggregations.
        """
        from balance.stats_and_plots.weighted_comparisons_stats import (
            _aggregate_statistic_by_main_covar,
        )

        # toy example
        asmd_series = pd.Series(
            {
                "age": 0.5,
                "education[T.high_school]": 1,
                "education[T. bachelor]": 2,
                "education[T. masters]": 3,
                "education[T. phd]": 4,
            }
        )

        outcome = _aggregate_statistic_by_main_covar(asmd_series).to_dict()
        expected = {"age": 0.5, "education": 2.5}

        self.assertEqual(outcome, expected)

    def test_asmd_improvement(self) -> None:
        """Test calculation of ASMD improvement ratios.

        Tests the asmd_improvement function which measures the improvement
        in balance between unadjusted and adjusted samples by comparing
        ASMD values before and after weighting adjustments.
        """
        from balance.stats_and_plots.weighted_comparisons_stats import asmd_improvement

        r = asmd_improvement(
            pd.DataFrame({"a": (5, 6, 7, 8)}),
            pd.DataFrame({"a": (5, 6, 7, 8)}),
            pd.DataFrame({"a": (1, 2, 3, 4)}),
            pd.Series((1, 1, 1, 1)),
            pd.Series((2, 2, 2, 2)),
            pd.Series((1, 1, 1, 1)),
        )
        self.assertEqual(r, 0)
        self.assertEqual(type(r), np.float64)

        r = asmd_improvement(
            pd.DataFrame({"a": (5, 6, 7, 8)}),
            pd.DataFrame({"a": (1, 2, 3, 4)}),
            pd.DataFrame({"a": (1, 2, 3, 4)}),
            pd.Series((1, 1, 1, 1)),
            pd.Series((1, 1, 1, 1)),
            pd.Series((1, 1, 1, 1)),
        )
        self.assertEqual(r, 1)
        self.assertEqual(type(r), np.float64)

        r = asmd_improvement(
            pd.DataFrame({"a": (3, 4)}),
            pd.DataFrame({"a": (2, 3)}),
            pd.DataFrame({"a": (1, 2)}),
            pd.Series((1, 1)),
            pd.Series((1, 1)),
            pd.Series((1, 1)),
        )
        self.assertEqual(r, 0.5)
        self.assertEqual(type(r), np.float64)


class TestBalance_general_stats(
    balance.testutil.BalanceTestCase,
):
    def test_relative_response_rates(self) -> None:
        """Test calculation of relative response rates across columns.

        Tests the relative_response_rates function which calculates response
        rates for each column in a dataset, handling missing values and
        comparing against target datasets when provided.
        """
        from balance.stats_and_plots.general_stats import relative_response_rates

        df = pd.DataFrame(
            {"o1": (7, 8, 9, 10), "o2": (7, 8, 9, np.nan), "id": (1, 2, 3, 4)}
        )

        self.assertEqual(
            relative_response_rates(df).to_dict(),
            {
                "o1": {"n": 4.0, "%": 100.0},
                "o2": {"n": 3.0, "%": 75.0},
                "id": {"n": 4.0, "%": 100.0},
            },
        )

        df_target = pd.concat([df, df])
        self.assertEqual(
            relative_response_rates(df, df_target).to_dict(),
            {
                "o1": {"n": 4.0, "%": 50.0},
                "o2": {"n": 3.0, "%": 50.0},
                "id": {"n": 4.0, "%": 50.0},
            },
        )

        # verify behavior when per_column is set to False
        self.assertEqual(
            relative_response_rates(df, df_target, per_column=False).round(3).to_dict(),
            {
                "o1": {"n": 4.0, "%": 66.667},
                "o2": {"n": 3.0, "%": 50.0},
                "id": {"n": 4.0, "%": 66.667},
            },
        )

        with self.assertRaisesRegex(
            ValueError, "df and df_target must have the exact same columns*"
        ):
            relative_response_rates(df, df_target.iloc[:, 0:1], per_column=True)

    def test_relative_response_rates_edge_cases(self) -> None:
        """Test relative_response_rates with edge cases like empty and all-NaN DataFrames.

        This consolidated test validates the function's behavior with edge cases that
        represent real-world scenarios where data might be completely missing or empty.
        """
        from balance.stats_and_plots.general_stats import relative_response_rates

        # Test case 1: Empty DataFrame
        df_empty = pd.DataFrame({"a": [], "b": []})
        result_empty = relative_response_rates(df_empty)

        self.assertEqual(result_empty.shape, (2, 2))
        self.assertEqual(result_empty.loc["n", "a"], 0.0)
        self.assertEqual(result_empty.loc["n", "b"], 0.0)
        # When dividing by 0, we get NaN
        self.assertTrue(pd.isna(result_empty.loc["%", "a"]))
        self.assertTrue(pd.isna(result_empty.loc["%", "b"]))

        # Test case 2: All NaN values
        df_all_nan = pd.DataFrame(
            {"a": [np.nan, np.nan, np.nan], "b": [np.nan, np.nan, np.nan]}
        )
        result_nan = relative_response_rates(df_all_nan)

        self.assertEqual(result_nan.loc["n", "a"], 0.0)
        self.assertEqual(result_nan.loc["n", "b"], 0.0)
        self.assertEqual(result_nan.loc["%", "a"], 0.0)
        self.assertEqual(result_nan.loc["%", "b"], 0.0)

    def test_relative_response_rates_validation_errors(self) -> None:
        """Test validation logic that protects against invalid inputs.

        This consolidated test ensures the function properly validates inputs
        and raises appropriate errors when constraints are violated, which is
        crucial for data integrity in statistical analysis.
        """
        from balance.stats_and_plots.general_stats import relative_response_rates

        # Test case 1: More non-null values in df than df_target
        df_more_values = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
        df_target_fewer = pd.DataFrame(
            {"a": [1, 2, np.nan, np.nan], "b": [5, np.nan, np.nan, np.nan]}
        )

        with self.assertRaisesRegex(
            ValueError,
            "The number of \\(notnull\\) rows in df MUST be smaller or equal*",
        ):
            relative_response_rates(df_more_values, df_target_fewer, per_column=True)

        # Test case 2: Different columns between df and df_target
        df_abc = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        df_target_ab = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        with self.assertRaisesRegex(
            ValueError, "df and df_target must have the exact same columns*"
        ):
            relative_response_rates(df_abc, df_target_ab, per_column=True)

        # Test case 3: per_column=False with no complete rows in target
        df_complete = pd.DataFrame({"a": [1], "b": [2]})
        # df_target has no complete rows (each row has at least one NaN)
        df_target_incomplete = pd.DataFrame({"a": [1, np.nan], "b": [np.nan, 2]})

        with self.assertRaisesRegex(
            ValueError,
            "The number of \\(notnull\\) rows in df MUST be smaller or equal*",
        ):
            relative_response_rates(df_complete, df_target_incomplete, per_column=False)

    def test_relative_response_rates_per_column_false_valid_case(self) -> None:
        """Test relative_response_rates with per_column=False and valid complete rows.

        Verifies correct calculation when comparing to total complete rows in target.
        """
        from balance.stats_and_plots.general_stats import relative_response_rates

        # df_target has 2 complete rows
        df_target = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, 6, 7, np.nan]})
        # df has 1 row with both values
        df = pd.DataFrame({"a": [1], "b": [5]})

        result = relative_response_rates(df, df_target, per_column=False)

        # df_target has 2 complete rows (first two)
        # df has 1 complete observation per column
        # So for column "a": 1/2 = 50%, for column "b": 1/2 = 50%
        self.assertEqual(result.loc["n", "a"], 1.0)
        self.assertEqual(result.loc["%", "a"], 50.0)
        self.assertEqual(result.loc["n", "b"], 1.0)
        self.assertEqual(result.loc["%", "b"], 50.0)


class TestKLDivergence(balance.testutil.BalanceTestCase):
    def test_discrete_normalization_and_value(self) -> None:
        from balance.stats_and_plots.weighted_comparisons_stats import (
            _kl_divergence_discrete,
        )

        p = np.array([2.0, 2.0])
        q = np.array([1.0, 3.0])
        expected = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)

        self.assertAlmostEqual(_kl_divergence_discrete(p, q), expected, places=6)

    def test_discrete_invalid_inputs(self) -> None:
        from balance.stats_and_plots.weighted_comparisons_stats import (
            _kl_divergence_discrete,
        )

        cases = (
            ((np.array([[0.1, 0.9]]), np.array([0.5, 0.5])), "1D arrays"),
            ((np.array([]), np.array([])), "must not be empty"),
            ((np.array([0.5]), np.array([0.5, 0.5])), "same length"),
            (
                (np.array([-0.1, 1.1]), np.array([0.5, 0.5])),
                "must not contain negative",
            ),
            (
                (np.array([np.nan, 1.0]), np.array([0.5, 0.5])),
                "NaN or infinite",
            ),
            ((np.array([0.0, 0.0]), np.array([0.5, 0.5])), "must not sum to zero"),
        )

        for (p, q), msg in cases:
            with self.subTest(msg=msg):
                with self.assertRaisesRegex(ValueError, msg):
                    _kl_divergence_discrete(p, q)

    def test_continuous_invalid_inputs(self) -> None:
        from balance.stats_and_plots.weighted_comparisons_stats import (
            _kl_divergence_continuous_quad,
        )

        with self.assertRaisesRegex(ValueError, "p_samples must be a 1D array"):
            _kl_divergence_continuous_quad(np.array([[1.0, 2.0]]), np.array([1.0, 2.0]))

        with self.assertRaisesRegex(ValueError, "q_samples must not be empty"):
            _kl_divergence_continuous_quad(np.array([1.0, 2.0]), np.array([]))

        with self.assertRaisesRegex(
            ValueError, "p_samples must contain at least two samples"
        ):
            _kl_divergence_continuous_quad(np.array([1.0]), np.array([1.0, 2.0]))

        with self.assertRaisesRegex(ValueError, "weights must match"):
            _kl_divergence_continuous_quad(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                p_weights=np.array([1.0, 2.0, 3.0]),
            )

        with self.assertRaisesRegex(ValueError, "weights must sum to a positive value"):
            _kl_divergence_continuous_quad(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                p_weights=np.array([0.0, 0.0]),
            )

        with self.assertRaisesRegex(ValueError, "weights must be non-negative"):
            _kl_divergence_continuous_quad(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                q_weights=np.array([-1.0, 2.0]),
            )

    def test_continuous_matches_identical_distribution(self) -> None:
        from balance.stats_and_plots.weighted_comparisons_stats import (
            _kl_divergence_continuous_quad,
        )

        x = np.linspace(-1, 1, 20)
        weights = np.linspace(1, 2, 20)

        kld = _kl_divergence_continuous_quad(
            x,
            x,
            p_weights=weights,
            q_weights=weights,
        )

        self.assertLess(kld, 1e-6)

    def test_kld_requires_positive_weight_mass(self) -> None:
        from balance.stats_and_plots import weighted_comparisons_stats

        df = pd.DataFrame({"cat": ["a", "a", "b"]})
        zero_weights = np.zeros(df.shape[0])

        with self.assertRaisesRegex(ValueError, "must sum to a positive value"):
            weighted_comparisons_stats.kld(df, df, zero_weights, zero_weights)

    def test_kld_with_categorical_data_basic(self) -> None:
        """Test KLD calculation with basic categorical data.

        This test verifies KLD calculation with known expected values
        for categorical distributions.
        """
        import math

        from balance.stats_and_plots import weighted_comparisons_stats

        # Create test case with known distribution
        # P(A)=0.5, P(B)=0.25, P(C)=0.25 in sample
        # P(A)=0.25, P(B)=0.5, P(C)=0.25 in target
        sample_df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "C"],
            }
        )
        target_df = pd.DataFrame(
            {
                "category": ["A", "B", "B", "C"],
            }
        )

        kld_result = weighted_comparisons_stats.kld(sample_df, target_df)

        # Verify result structure
        self.assertIn("category", kld_result.index)
        self.assertIn("mean(kld)", kld_result.index)

        # Calculate expected KLD manually
        # KLD = 0.5 * log(0.5/0.25) + 0.25 * log(0.25/0.5) + 0.25 * log(0.25/0.25)
        #     = 0.5 * log(2) + 0.25 * log(0.5) + 0.25 * 0
        #     = 0.5 * 0.693 + 0.25 * (-0.693) + 0
        #     ≈ 0.173
        expected_kld = 0.5 * math.log(2) + 0.25 * math.log(0.5)

        # Verify KLD is positive and reasonably close to expected
        self.assertGreater(kld_result["category"], 0)
        # Allow some tolerance for numerical computation
        self.assertAlmostEqual(kld_result["category"], abs(expected_kld), places=2)

    def test_kld_with_numeric_continuous_data(self) -> None:
        """Test KLD calculation with continuous numeric data.

        Tests that continuous distributions are properly handled using
        kernel density estimation and numerical integration.
        """
        from balance.stats_and_plots import weighted_comparisons_stats

        np.random.seed(42)
        sample_df = pd.DataFrame(
            {
                "age": np.random.normal(30, 5, 50),
                "income": np.random.normal(50000, 10000, 50),
            }
        )
        target_df = pd.DataFrame(
            {
                "age": np.random.normal(35, 5, 50),
                "income": np.random.normal(55000, 10000, 50),
            }
        )

        kld_result = weighted_comparisons_stats.kld(sample_df, target_df)

        # Verify result structure
        self.assertIn("age", kld_result.index)
        self.assertIn("income", kld_result.index)
        self.assertIn("mean(kld)", kld_result.index)

        # Verify all values are non-negative
        self.assertTrue(all(kld_result >= 0))

        # Different distributions should have positive KLD
        self.assertGreater(kld_result["age"], 0)
        self.assertGreater(kld_result["income"], 0)

    def test_kld_with_various_data_types(self) -> None:
        """Test KLD calculation with various data types using parameterized approach.

        Tests multiple data type scenarios with meaningful assertions about
        KLD calculation correctness.
        """
        from balance.stats_and_plots import weighted_comparisons_stats

        # Test cases with different data types
        test_cases = [
            {
                "name": "binary_columns",
                "sample_data": {"binary": [0, 0, 1, 1]},
                "target_data": {"binary": [0, 1, 1, 1]},
                "expected_behavior": "positive_kld",  # Different distributions
            },
            {
                "name": "boolean_columns",
                "sample_data": {"is_active": [True, True, False, False]},
                "target_data": {"is_active": [True, False, False, False]},
                "expected_behavior": "positive_kld",  # Different distributions
            },
            {
                "name": "object_dtype_categorical",
                "sample_data": {
                    "country": pd.Series(["US", "UK", "DE", "FR"], dtype=object)
                },
                "target_data": {
                    "country": pd.Series(["US", "US", "UK", "FR"], dtype=object)
                },
                "expected_behavior": "positive_kld",  # Different distributions
            },
            {
                "name": "pandas_categorical",
                "sample_data": {"size": pd.Categorical(["S", "M", "L", "M"])},
                "target_data": {"size": pd.Categorical(["S", "S", "M", "L"])},
                "expected_behavior": "positive_kld",  # Different distributions
            },
            {
                "name": "identical_distributions",
                "sample_data": {"value": [1, 1, 2, 2]},
                "target_data": {"value": [1, 1, 2, 2]},
                "expected_behavior": "zero_kld",  # Identical distributions
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case=test_case["name"]):
                sample_df = pd.DataFrame(test_case["sample_data"])
                target_df = pd.DataFrame(test_case["target_data"])

                kld_result = weighted_comparisons_stats.kld(sample_df, target_df)

                # Verify structure
                sample_data = test_case["sample_data"]
                assert isinstance(sample_data, dict)  # Type narrowing for mypy
                col_name = list(sample_data.keys())[0]
                # For pandas categorical, the column name might be encoded differently
                if test_case["name"] == "pandas_categorical":
                    self.assertTrue(any("size" in str(idx) for idx in kld_result.index))
                else:
                    self.assertIn(col_name, kld_result.index)
                self.assertIn("mean(kld)", kld_result.index)

                # Verify expected behavior
                if test_case["expected_behavior"] == "zero_kld":
                    # For identical distributions, KLD should be very close to zero
                    kld_value = kld_result.get(col_name, kld_result.iloc[0])
                    self.assertLess(
                        kld_value,
                        1e-5,
                        f"KLD should be near zero for identical distributions in {test_case['name']}",
                    )
                elif test_case["expected_behavior"] == "positive_kld":
                    # For different distributions, KLD should be positive
                    kld_value = kld_result.get(col_name, kld_result.iloc[0])
                    self.assertGreater(
                        kld_value,
                        0,
                        f"KLD should be positive for different distributions in {test_case['name']}",
                    )

    def test_kld_with_weights(self) -> None:
        """Test KLD calculation with weighted samples.

        Verifies that sample and target weights are properly incorporated
        into the KLD calculation and produce predictable effects.
        """
        from balance.stats_and_plots import weighted_comparisons_stats

        sample_df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "C"],
            }
        )
        target_df = pd.DataFrame(
            {
                "category": ["A", "B", "B", "C"],
            }
        )

        # Test 1: Weights that make distributions more similar
        # Weight A more heavily in sample to match target's A proportion
        sample_weights_similar = np.array([0.5, 0.5, 2.0, 2.0])  # Makes P(A)≈0.2
        target_weights_uniform = np.array([1.0, 1.0, 1.0, 1.0])

        # Test 2: Weights that make distributions more different
        sample_weights_different = np.array([2.0, 2.0, 0.5, 0.5])  # Makes P(A)≈0.67

        kld_similar = weighted_comparisons_stats.kld(
            sample_df, target_df, sample_weights_similar, target_weights_uniform
        )
        kld_different = weighted_comparisons_stats.kld(
            sample_df, target_df, sample_weights_different, target_weights_uniform
        )
        kld_unweighted = weighted_comparisons_stats.kld(sample_df, target_df)

        # Weights that make distributions more similar should reduce KLD
        self.assertLess(kld_similar["category"], kld_unweighted["category"])

        # Weights that make distributions more different should increase KLD
        self.assertGreater(kld_different["category"], kld_unweighted["category"])

        # The different-weighted KLD should be larger than similar-weighted
        self.assertGreater(kld_different["category"], kld_similar["category"])

    def test_kld_with_aggregate_by_main_covar(self) -> None:
        """Test KLD calculation with aggregate_by_main_covar option.

        Tests that categorical variables with multiple levels are properly
        aggregated into a single KLD value per covariate.
        """
        from balance.stats_and_plots import weighted_comparisons_stats

        sample_df = pd.DataFrame(
            {
                "age": [25, 30, 35, 40],
                "education[T.high_school]": [1, 0, 0, 0],
                "education[T.bachelor]": [0, 1, 0, 0],
                "education[T.masters]": [0, 0, 1, 1],
            }
        )
        target_df = pd.DataFrame(
            {
                "age": [28, 32, 36, 42],
                "education[T.high_school]": [0, 1, 0, 0],
                "education[T.bachelor]": [1, 0, 0, 0],
                "education[T.masters]": [0, 0, 1, 1],
            }
        )

        kld_aggregated = weighted_comparisons_stats.kld(
            sample_df, target_df, aggregate_by_main_covar=True
        )

        # Should have aggregated education levels
        self.assertIn("age", kld_aggregated.index)
        self.assertIn("education", kld_aggregated.index)
        self.assertIn("mean(kld)", kld_aggregated.index)

        # Should NOT have individual education levels
        self.assertNotIn("education[T.high_school]", kld_aggregated.index)
        self.assertNotIn("education[T.bachelor]", kld_aggregated.index)
        self.assertNotIn("education[T.masters]", kld_aggregated.index)

    def test_kld_validation_errors(self) -> None:
        """Test KLD function raises appropriate errors for invalid inputs.

        Verifies that kld properly validates input DataFrame types and
        raises descriptive errors for invalid arguments.
        """
        from typing import Any

        from balance.stats_and_plots import weighted_comparisons_stats

        valid_df = pd.DataFrame({"a": [1, 2, 3]})
        invalid_series: Any = pd.Series(
            [1, 2, 3]
        )  # Use Any to allow testing invalid types

        # Test invalid sample_df type
        with self.assertRaisesRegex(ValueError, "sample_df must be pd.DataFrame"):
            weighted_comparisons_stats.kld(
                invalid_series,
                valid_df,
            )

        # Test invalid target_df type
        with self.assertRaisesRegex(ValueError, "target_df must be pd.DataFrame"):
            weighted_comparisons_stats.kld(
                valid_df,
                invalid_series,
            )

    def test_kld_with_na_indicator_columns(self) -> None:
        """Test KLD handling of NA indicator columns.

        Verifies that columns starting with '_is_na_' are properly excluded
        from the KLD calculation as specified in the function documentation.
        """
        from balance.stats_and_plots import weighted_comparisons_stats

        sample_df = pd.DataFrame(
            {
                "value": [1, 2, 3, 4],
                "_is_na_value": [0, 0, 1, 0],
            }
        )
        target_df = pd.DataFrame(
            {
                "value": [2, 3, 4, 5],
                "_is_na_value": [0, 1, 0, 0],
            }
        )

        kld_result = weighted_comparisons_stats.kld(sample_df, target_df)

        # _is_na_ columns should be excluded
        self.assertNotIn("_is_na_value", kld_result.index)
        self.assertIn("value", kld_result.index)
        self.assertIn("mean(kld)", kld_result.index)

    def test_kld_identical_distributions(self) -> None:
        """Test KLD with identical sample and target distributions.

        When distributions are identical, KLD should be zero or very close
        to zero for all columns.
        """
        from balance.stats_and_plots import weighted_comparisons_stats

        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "C"],
                "numeric": [1.0, 2.0, 3.0, 4.0],
            }
        )

        kld_result = weighted_comparisons_stats.kld(df, df)

        # All KLD values should be very close to zero
        for col in ["category", "numeric"]:
            self.assertLess(kld_result[col], 1e-5)

        # Mean should also be close to zero
        self.assertLess(kld_result["mean(kld)"], 1e-5)

    def test_kld_with_known_values(self) -> None:
        """Test KLD calculation against manually calculated expected values.

        Uses simple distributions where KLD can be calculated by hand
        to verify the implementation correctness.
        """
        import math

        from balance.stats_and_plots import weighted_comparisons_stats

        # Test case 1: Binary distribution with known KLD
        # P = [0.75, 0.25], Q = [0.5, 0.5]
        # KLD = 0.75*log(0.75/0.5) + 0.25*log(0.25/0.5) = 0.75*log(1.5) + 0.25*log(0.5)
        sample_df = pd.DataFrame({"binary": [0, 0, 0, 1]})  # 75% 0s, 25% 1s
        target_df = pd.DataFrame({"binary": [0, 0, 1, 1]})  # 50% 0s, 50% 1s

        kld_result = weighted_comparisons_stats.kld(sample_df, target_df)

        expected_kld = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
        self.assertAlmostEqual(kld_result["binary"], expected_kld, places=3)

        # Test case 2: Three category distribution
        # P = [0.5, 0.25, 0.25], Q = [0.33, 0.33, 0.34]
        sample_df2 = pd.DataFrame({"cat": ["A", "A", "B", "C"]})
        target_df2 = pd.DataFrame({"cat": ["A", "B", "C", "C"]})  # Close to uniform

        kld_result2 = weighted_comparisons_stats.kld(sample_df2, target_df2)

        # KLD should be positive for different distributions
        self.assertGreater(kld_result2["cat"], 0)
        # For this specific case, KLD should be relatively small (< 0.5) since distributions are not too different
        self.assertLess(kld_result2["cat"], 0.5)

    def test_kld_with_single_category_column(self) -> None:
        """Test KLD with categorical column having single unique value.

        Edge case where a categorical variable has only one level should
        still be processed without errors.
        """
        from balance.stats_and_plots import weighted_comparisons_stats

        sample_df = pd.DataFrame(
            {
                "constant": ["A", "A", "A", "A"],
            }
        )
        target_df = pd.DataFrame(
            {
                "constant": ["A", "A", "A", "A"],
            }
        )

        kld_result = weighted_comparisons_stats.kld(sample_df, target_df)

        # Should produce result without error
        self.assertIn("constant", kld_result.index)
        self.assertIn("mean(kld)", kld_result.index)

        # Identical single-category should have zero KLD
        self.assertAlmostEqual(kld_result["constant"], 0.0, places=5)
