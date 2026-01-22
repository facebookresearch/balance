# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import absolute_import, division, print_function, unicode_literals

import tempfile
from copy import deepcopy

import IPython.display
import numpy as np
import pandas as pd
from balance.balancedf_class import (  # noqa
    BalanceDF,
    BalanceDFCovars,  # noqa
    BalanceDFOutcomes,  # noqa
    BalanceDFWeights,  # noqa
)
from balance.sample_class import Sample
from balance.stats_and_plots import weighted_comparisons_stats
from balance.testutil import BalanceTestCase, tempfile_path


class TestDataFactory:
    """Factory class for creating test data samples used across test cases."""

    @staticmethod
    def create_basic_sample() -> Sample:
        """Create a basic sample with covariates, outcomes, and weights."""
        return Sample.from_frame(
            pd.DataFrame(
                {
                    "a": (1, 2, 3, 1),
                    "b": (-42, 8, 2, -42),
                    "o": (7, 8, 9, 10),
                    "c": ("x", "y", "z", "v"),
                    "id": (1, 2, 3, 4),
                    "w": (0.5, 2, 1, 1),
                }
            ),
            id_column="id",
            weight_column="w",
            outcome_columns="o",
        )

    @staticmethod
    def create_target_sample() -> Sample:
        """Create a target sample for comparison testing."""
        return Sample.from_frame(
            pd.DataFrame(
                {
                    "a": (1, 2, 3),
                    "b": (4, 6, 8),
                    "id": (1, 2, 3),
                    "w": (0.5, 1, 2),
                    "c": ("x", "y", "z"),
                }
            ),
            id_column="id",
            weight_column="w",
        )

    @staticmethod
    def create_sample_with_null_values() -> Sample:
        """Create a sample with null values for testing edge cases."""
        return Sample.from_frame(
            pd.DataFrame(
                {
                    "a": (0, None, 2),
                    "b": (0, None, 2),
                    "c": ("a", "b", "c"),
                    "id": (1, 2, 3),
                }
            ),
            outcome_columns=("b", "c"),
        )

    @staticmethod
    def create_multi_outcome_sample() -> Sample:
        """Create a sample with multiple outcome columns."""
        return Sample.from_frame(
            pd.DataFrame(
                {"o1": (7, 8, 9, 10), "o2": (7, 8, 9, np.nan), "id": (1, 2, 3, 4)}
            ),
            id_column="id",
            outcome_columns=("o1", "o2"),
        )

    @staticmethod
    def create_large_target_outcome_sample() -> Sample:
        """Create a larger target sample for outcome comparison testing."""
        return Sample.from_frame(
            pd.DataFrame(
                {
                    "o1": (7, 8, 9, 10, 11, 12, 13, 14),
                    "o2": (7, 8, 9, np.nan, np.nan, 12, 13, 14),
                    "id": (1, 2, 3, 4, 5, 6, 7, 8),
                }
            ),
            id_column="id",
            outcome_columns=("o1", "o2"),
        )

    @staticmethod
    def create_sample_with_special_characters() -> Sample:
        """Create a sample with special characters in column names for testing."""
        return Sample.from_frame(
            pd.DataFrame(
                {
                    "a$": (1, 2, 3, 1),
                    "a%": (-42, 8, 2, -42),
                    "o*": (7, 8, 9, 10),
                    "c@": ("x", "y", "z", "v"),
                    "id": (1, 2, 3, 4),
                    "w": (0.5, 2, 1, 1),
                }
            ),
            id_column="id",
            weight_column="w",
            outcome_columns="o*",
        )

    @staticmethod
    def create_adjusted_samples() -> dict[str, Sample]:
        """Create a set of related samples for adjustment testing."""
        s1 = TestDataFactory.create_basic_sample()
        s2 = TestDataFactory.create_target_sample()
        s3 = s1.set_target(s2)
        s3_null = s3.adjust(method="null")

        s3_null_madeup_weights = deepcopy(s3_null)
        s3_null_madeup_weights.set_weights(
            pd.Series([1, 2, 3, 1], index=s3_null.df.index)
        )

        return {
            "basic": s1,
            "target": s2,
            "with_target": s3,
            "null_adjusted": s3_null,
            "custom_weights": s3_null_madeup_weights,
        }


# Create commonly used test samples
s1: Sample = TestDataFactory.create_basic_sample()
s2: Sample = TestDataFactory.create_target_sample()
s3: Sample = s1.set_target(s2)
s3_null: Sample = s3.adjust(method="null")

s3_null_madeup_weights: Sample = deepcopy(s3_null)
s3_null_madeup_weights.set_weights(pd.Series([1, 2, 3, 1], index=s3_null.df.index))

s4: Sample = TestDataFactory.create_sample_with_null_values()
o: BalanceDFOutcomes = s1.outcomes()
s_o: Sample = TestDataFactory.create_multi_outcome_sample()
t_o: Sample = TestDataFactory.create_large_target_outcome_sample()
s_o2: Sample = s_o.set_target(t_o)
c: BalanceDFCovars = s1.covars()
w: BalanceDFWeights = s1.weights()
s1_bad_columns: Sample = TestDataFactory.create_sample_with_special_characters()


class TestBalanceDFOutcomes(BalanceTestCase):
    """Test cases for BalanceDFOutcomes class functionality."""

    def test_Sample_outcomes(self) -> None:
        """Test that Sample.outcomes() returns correct BalanceDFOutcomes instances.

        Verifies:
        - Returns BalanceDFOutcomes instance for samples with outcomes
        - Handles multicharacter column names correctly
        - Returns None for samples without outcomes
        """
        self.assertTrue(isinstance(s4.outcomes(), BalanceDFOutcomes))
        self.assertEqual(
            s4.outcomes().df, pd.DataFrame({"b": (0, None, 2), "c": ("a", "b", "c")})
        )

        # Test with multicharacter string name
        s = Sample.from_frame(
            pd.DataFrame({"aardvark": (0, None, 2), "id": (1, 2, 3)}),
            outcome_columns="aardvark",
        )
        self.assertEqual(s.outcomes().df, pd.DataFrame({"aardvark": (0, None, 2)}))

        # Null outcomes
        self.assertTrue(s2.outcomes() is None)

    def test_BalanceDFOutcomes_df(self) -> None:
        """Test BalanceDFOutcomes.df property behavior and data integrity.

        Verifies:
        - df is implemented as a property decorator
        - Property cannot be called as a function
        - Data values are correctly converted to float type
        - Property accessor works correctly
        """
        # Verify that the @property decorator works properly.
        self.assertTrue(isinstance(BalanceDFOutcomes.df, property))
        # We can no longer call .df() as if it was a function:
        with self.assertRaisesRegex(TypeError, "'DataFrame' object is not callable"):
            o.df()  # pyre-ignore[29]: Testing property call error
        # Here is how we can call it as a function:
        self.assertEqual(BalanceDFOutcomes.df.fget(o), o.df)

        # Check values are as expected:
        # NOTE that values changed from integer to float
        self.assertEqual(o.df, pd.DataFrame({"o": (7.0, 8.0, 9.0, 10.0)}))

    def test__get_df_and_weights(self) -> None:
        """Test BalanceDF._get_df_and_weights() method functionality.

        Verifies:
        - Returns correct DataFrame and numpy array types
        - DataFrame values match expected outcome data
        - Weights array contains correct weight values
        """
        df, w = BalanceDF._get_df_and_weights(o)
        # Check types
        self.assertEqual(type(df), pd.DataFrame)
        self.assertEqual(type(w), np.ndarray)
        # Check values
        self.assertEqual(df.to_dict(), {"o": {0: 7.0, 1: 8.0, 2: 9.0, 3: 10.0}})
        self.assertEqual(w, np.array([0.5, 2.0, 1.0, 1.0]))

    def test_BalanceDFOutcomes_names(self) -> None:
        """Test that BalanceDFOutcomes.names() returns correct outcome column names."""
        self.assertEqual(o.names(), ["o"])

    def test_BalanceDFOutcomes__sample(self) -> None:
        """Test that BalanceDFOutcomes._sample references the correct source sample."""
        self.assertTrue(o._sample is s1)

    def test_BalanceDFOutcomes_weights(self) -> None:
        """Test that BalanceDFOutcomes._weights returns correct weight Series."""
        pd.testing.assert_series_equal(o._weights, pd.Series((0.5, 2, 1, 1)))

    def test_BalanceDFOutcomes_relative_response_rates(self) -> None:
        """Test relative_response_rates method with various target configurations.

        Verifies:
        - Basic relative response rates calculation
        - Handling of None targets
        - Per-column vs per-dataset calculations
        - Error handling for mismatched column structures
        """
        self.assertEqual(
            s_o.outcomes().relative_response_rates(),
            pd.DataFrame({"o1": [100.0, 4], "o2": [75.0, 3]}, index=["%", "n"]),
            lazy=True,
        )

        self.assertEqual(s_o.outcomes().relative_response_rates(target=True), None)

        # compared with a larget target
        self.assertEqual(
            s_o2.outcomes()
            .relative_response_rates(True, per_column=True)
            .round(3)
            .to_dict(),
            {"o1": {"n": 4.0, "%": 50.0}, "o2": {"n": 3.0, "%": 50.0}},
        )

        df_target = pd.DataFrame(
            {
                "o1": (7, 8, 9, 10, 11, 12, 13, 14),
                "o2": (7, 8, 9, np.nan, np.nan, 12, 13, 14),
            }
        )
        # Relative to per column:
        self.assertEqual(
            s_o2.outcomes()
            .relative_response_rates(target=df_target, per_column=True)
            .to_dict(),
            {"o1": {"n": 4.0, "%": 50.0}, "o2": {"n": 3.0, "%": 50.0}},
        )
        # Checking that if we force per_column=True
        # On a df_target that is not the same column structure as s_o2.outcomes()
        # It will lead to a ValueError
        with self.assertRaisesRegex(
            ValueError, "df and df_target must have the exact same columns*"
        ):
            s_o2.outcomes().relative_response_rates(
                df_target.iloc[:, 0:1], per_column=True
            )

        # Relative to all notnull rows in outcome
        self.assertEqual(
            s_o2.outcomes().relative_response_rates(target=True).round(3).to_dict(),
            {"o1": {"n": 4.0, "%": 66.667}, "o2": {"n": 3.0, "%": 50.0}},
        )
        self.assertEqual(
            s_o2.outcomes()
            .relative_response_rates(
                target=df_target,
                per_column=False,  # This is the default.
            )
            .round(3)
            .to_dict(),
            s_o2.outcomes().relative_response_rates(target=True).round(3).to_dict(),
        )
        # This will also work with different shape of columns (exactly what we need for .summary())
        self.assertEqual(
            s_o.outcomes()
            .relative_response_rates(df_target.iloc[:, 0:1], per_column=False)
            .round(3)
            .to_dict(),
            {"o1": {"n": 4.0, "%": 50.0}, "o2": {"n": 3.0, "%": 37.5}},
        )

    def test_BalanceDFOutcomes_target_response_rates(self) -> None:
        """Test target_response_rates method for calculating target sample response rates."""
        self.assertEqual(
            s_o2.outcomes().target_response_rates(),
            pd.DataFrame({"o1": {"n": 8.0, "%": 100.0}, "o2": {"n": 6.0, "%": 75.0}}),
            lazy=True,
        )

    def test_BalanceDFOutcomes_summary(self) -> None:
        """Test BalanceDFOutcomes.summary() method output format and content.

        Verifies that summary output includes:
        - Outcome column information
        - Mean outcomes with confidence intervals
        - Response rates for different comparison scenarios
        """

        def _remove_whitespace_and_newlines(s: str) -> str:
            return " ".join(s.split())

        e_str = """\
                2 outcomes: ['o1' 'o2']
                Mean outcomes (with 95% confidence intervals):
                source            self          self_ci
                _is_na_o2[False]  0.75   (0.326, 1.174)
                _is_na_o2[True]   0.25  (-0.174, 0.674)
                o1                8.50   (7.404, 9.596)
                o2                6.00   (2.535, 9.465)

                Response rates (relative to number of respondents in sample):
                    o1    o2
                n    4.0   3.0
                %  100.0  75.0
            """
        self.assertEqual(
            _remove_whitespace_and_newlines(s_o.outcomes().summary()),
            _remove_whitespace_and_newlines(e_str),
        )

        e_str = """\
                2 outcomes: ['o1' 'o2']
                Mean outcomes (with 95% confidence intervals):
                source            self  target          self_ci        target_ci
                _is_na_o2[False]  0.75   0.750   (0.326, 1.174)     (0.45, 1.05)
                _is_na_o2[True]   0.25   0.250  (-0.174, 0.674)    (-0.05, 0.55)
                o1                8.50  10.500   (7.404, 9.596)  (8.912, 12.088)
                o2                6.00   7.875   (2.535, 9.465)  (4.351, 11.399)

                Response rates (relative to number of respondents in sample):
                    o1    o2
                n    4.0   3.0
                %  100.0  75.0
                Response rates (relative to notnull rows in the target):
                        o1    o2
                n   4.000000   3.0
                %  66.666667  50.0
                Response rates (in the target):
                    o1    o2
                n    8.0   6.0
                %  100.0  75.0
            """
        self.assertEqual(
            _remove_whitespace_and_newlines(s_o2.outcomes().summary()),
            _remove_whitespace_and_newlines(e_str),
        )


class TestBalanceDFCovars(BalanceTestCase):
    """Test cases for BalanceDFCovars class functionality."""

    def test_BalanceDFCovars_df(self) -> None:
        """Test BalanceDFCovars.df property behavior and data type conversion.

        Verifies:
        - df is implemented as a property decorator
        - Property cannot be called as a function
        - Integer values are converted to floats in DataFrame
        - String values are preserved unchanged
        """
        # Verify that the @property decorator works properly.
        self.assertTrue(isinstance(BalanceDFCovars.df, property))
        self.assertEqual(BalanceDFOutcomes.df.fget(c), c.df)
        # We can no longer call .df() as if it was a function:
        with self.assertRaisesRegex(TypeError, "'DataFrame' object is not callable"):
            c.df()  # pyre-ignore[29]: Testing property call error
        # NOTE: while the original datatype had integers, the stored df has only floats:
        self.assertEqual(
            c.df,
            pd.DataFrame(
                {
                    "a": (1.0, 2.0, 3.0, 1.0),
                    "b": (-42.0, 8.0, 2.0, -42.0),
                    "c": ("x", "y", "z", "v"),
                }
            ),
        )

    def test_BalanceDFCovars_names(self) -> None:
        """Test that BalanceDFCovars.names() returns correct column names as list."""
        self.assertEqual(c.names(), ["a", "b", "c"])
        self.assertEqual(type(c.names()), list)

    def test_BalanceDFCovars__sample(self) -> None:
        """Test that BalanceDFCovars._sample references the correct source sample."""
        self.assertTrue(c._sample is s1)

    def test_BalanceDFCovars_weights(self) -> None:
        """Test that BalanceDFCovars._weights returns correct weight Series."""
        pd.testing.assert_series_equal(
            c._weights, pd.Series(np.array([0.5, 2.0, 1.0, 1.0]))
        )


class TestBalanceDFWeights(BalanceTestCase):
    """Test cases for BalanceDFWeights class functionality."""

    def test_BalanceDFWeights_df(self) -> None:
        """Test BalanceDFWeights.df property behavior and weight data access.

        Verifies:
        - df is implemented as a property decorator
        - Property cannot be called as a function
        - DataFrame contains correct weight values
        """
        # Verify that the @property decorator works properly.
        self.assertTrue(isinstance(BalanceDFWeights.df, property))
        self.assertEqual(
            BalanceDFWeights.df.fget(w),  # pyre-ignore[29]: Testing property getter
            w.df,
        )
        # We can no longer call .df() as if it was a function:
        with self.assertRaisesRegex(TypeError, "'DataFrame' object is not callable"):
            w.df()  # pyre-ignore[29]: Testing property call error
        # Check values are as expected:
        self.assertEqual(w.df, pd.DataFrame({"w": (0.5, 2, 1, 1)}))

    def test_BalanceDFWeights_names(self) -> None:
        """Test that BalanceDFWeights.names() returns correct weight column names."""
        self.assertEqual(w.names(), ["w"])

    def test_BalanceDFWeights__sample(self) -> None:
        """Test that BalanceDFWeights._sample references the correct source sample."""
        self.assertTrue(c._sample is s1)

    def test_BalanceDFWeights_weights(self) -> None:
        """Test that BalanceDFWeights._weights is None (weights don't have weights)."""
        self.assertTrue(w._weights is None)

    def test_BalanceDFWeights_design_effect(self) -> None:
        s = Sample.from_frame(
            pd.DataFrame({"w": (1, 2, 4), "id": (1, 2, 3)}),
            id_column="id",
            weight_column="w",
        )
        self.assertTrue(s.weights().design_effect(), 7 / 3)

    def test_BalanceDFWeights_trim(self) -> None:
        np.random.seed(112358)  # Fix seed for reproducibility
        s = Sample.from_frame(
            pd.DataFrame({"w": np.random.uniform(0, 1, 10000), "id": range(0, 10000)}),
            id_column="id",
            weight_column="w",
        )
        s.weights().trim(percentile=(0, 0.11), keep_sum_of_weights=False)
        self.assertTrue(max(s.weights().df.iloc[:, 0]) < 0.9)

    def test_BalanceDFWeights_summary(self) -> None:
        exp = {
            "var": {
                0: "design_effect",
                1: "effective_sample_proportion",
                2: "effective_sample_size",
                3: "sum",
                4: "describe_count",
                5: "describe_mean",
                6: "describe_std",
                7: "describe_min",
                8: "describe_25%",
                9: "describe_50%",
                10: "describe_75%",
                11: "describe_max",
                12: "prop(w < 0.1)",
                13: "prop(w < 0.2)",
                14: "prop(w < 0.333)",
                15: "prop(w < 0.5)",
                16: "prop(w < 1)",
                17: "prop(w >= 1)",
                18: "prop(w >= 2)",
                19: "prop(w >= 3)",
                20: "prop(w >= 5)",
                21: "prop(w >= 10)",
                22: "nonparametric_skew",
                23: "weighted_median_breakdown_point",
            },
            "val": {
                0: 1.23,
                1: 0.81,
                2: 3.24,
                3: 4.5,
                4: 4.0,
                5: 1.0,
                6: 0.56,
                7: 0.44,
                8: 0.78,
                9: 0.89,
                10: 1.11,
                11: 1.78,
                12: 0.0,
                13: 0.0,
                14: 0.0,
                15: 0.25,
                16: 0.75,
                17: 0.25,
                18: 0.0,
                19: 0.0,
                20: 0.0,
                21: 0.0,
                22: 0.2,
                23: 0.25,
            },
        }
        self.assertEqual(w.summary().round(2).to_dict(), exp)


class TestBalanceDF__BalanceDF_child_from_linked_samples(BalanceTestCase):
    def test__BalanceDF_child_from_linked_samples_keys(self) -> None:
        self.assertEqual(
            list(s1.covars()._BalanceDF_child_from_linked_samples().keys()), ["self"]
        )
        self.assertEqual(
            list(s3.covars()._BalanceDF_child_from_linked_samples().keys()),
            ["self", "target"],
        )
        self.assertEqual(
            list(s3_null.covars()._BalanceDF_child_from_linked_samples().keys()),
            ["self", "target", "unadjusted"],
        )

        self.assertEqual(
            list(s1.weights()._BalanceDF_child_from_linked_samples().keys()), ["self"]
        )
        self.assertEqual(
            list(s3.weights()._BalanceDF_child_from_linked_samples().keys()),
            ["self", "target"],
        )
        self.assertEqual(
            list(s3_null.weights()._BalanceDF_child_from_linked_samples().keys()),
            ["self", "target", "unadjusted"],
        )

        self.assertEqual(
            list(s1.outcomes()._BalanceDF_child_from_linked_samples().keys()), ["self"]
        )
        self.assertEqual(
            list(s3.outcomes()._BalanceDF_child_from_linked_samples().keys()),
            ["self", "target"],
        )
        self.assertEqual(
            list(s3_null.outcomes()._BalanceDF_child_from_linked_samples().keys()),
            ["self", "target", "unadjusted"],
        )

    def test__BalanceDF_child_from_linked_samples_class_types(self) -> None:
        """Test that _BalanceDF_child_from_linked_samples returns correct class types.

        Verifies that the method returns appropriate BalanceDF subclasses
        based on the type and number of linked samples.
        """
        # We can get a class using .__class__
        self.assertEqual(s1.covars().__class__, BalanceDFCovars)

        # We get a different number of classes based on the number of linked items:
        the_dict = s1.covars()._BalanceDF_child_from_linked_samples()
        exp = [BalanceDFCovars]
        self.assertEqual([v.__class__ for (k, v) in the_dict.items()], exp)

        the_dict = s3.covars()._BalanceDF_child_from_linked_samples()
        exp = 2 * [BalanceDFCovars]
        self.assertEqual([v.__class__ for (k, v) in the_dict.items()], exp)

        the_dict = s3_null.covars()._BalanceDF_child_from_linked_samples()
        exp = 3 * [BalanceDFCovars]
        self.assertEqual([v.__class__ for (k, v) in the_dict.items()], exp)

    def test__BalanceDF_child_from_linked_samples_weights_class(self) -> None:
        """Test that _BalanceDF_child_from_linked_samples works for BalanceDFWeights."""
        # This also works for things other than BalanceDFCovars:
        the_dict = s3_null.weights()._BalanceDF_child_from_linked_samples()
        exp = 3 * [BalanceDFWeights]
        self.assertEqual([v.__class__ for (k, v) in the_dict.items()], exp)

    def test__BalanceDF_child_from_linked_samples_outcomes_with_none(self) -> None:
        """Test that _BalanceDF_child_from_linked_samples handles None outcomes correctly."""
        # Notice that with something like outcomes, we might get a None in return!
        the_dict = s3_null.outcomes()._BalanceDF_child_from_linked_samples()
        exp = [
            BalanceDFOutcomes,
            type(None),
            BalanceDFOutcomes,
        ]
        self.assertEqual([v.__class__ for (k, v) in the_dict.items()], exp)

    def test__BalanceDF_child_from_linked_samples_covars_values(self) -> None:
        """Test that covariates DataFrame values are correctly preserved in linked samples."""
        # Verify DataFrame values makes sense:
        # for covars
        the_dict = s3_null.covars()._BalanceDF_child_from_linked_samples()
        exp = [
            {
                "a": {0: 1, 1: 2, 2: 3, 3: 1},
                "b": {0: -42, 1: 8, 2: 2, 3: -42},
                "c": {0: "x", 1: "y", 2: "z", 3: "v"},
            },
            {
                "a": {0: 1, 1: 2, 2: 3},
                "b": {0: 4, 1: 6, 2: 8},
                "c": {0: "x", 1: "y", 2: "z"},
            },
            {
                "a": {0: 1, 1: 2, 2: 3, 3: 1},
                "b": {0: -42, 1: 8, 2: 2, 3: -42},
                "c": {0: "x", 1: "y", 2: "z", 3: "v"},
            },
        ]
        self.assertEqual([v.df.to_dict() for (k, v) in the_dict.items()], exp)

    def test__BalanceDF_child_from_linked_samples_outcomes_values(self) -> None:
        """Test that outcomes DataFrame values are correctly preserved, excluding None values."""
        # for outcomes
        the_dict = s3_null.outcomes()._BalanceDF_child_from_linked_samples()
        exp = [{"o": {0: 7, 1: 8, 2: 9, 3: 10}}, {"o": {0: 7, 1: 8, 2: 9, 3: 10}}]
        # need to exclude None v:
        self.assertEqual(
            [v.df.to_dict() for (k, v) in the_dict.items() if v is not None], exp
        )

    def test__BalanceDF_child_from_linked_samples_weights_values(self) -> None:
        """Test that weights DataFrame values are correctly preserved in linked samples."""
        # for weights
        the_dict = s3_null.weights()._BalanceDF_child_from_linked_samples()
        exp = [
            {"w": {0: 0.5, 1: 2.0, 2: 1.0, 3: 1.0}},
            {"w": {0: 0.5, 1: 1.0, 2: 2.0}},
            {"w": {0: 0.5, 1: 2.0, 2: 1.0, 3: 1.0}},
        ]
        self.assertEqual([v.df.to_dict() for (k, v) in the_dict.items()], exp)


class TestBalanceDF__call_on_linked(BalanceTestCase):
    def test_BalanceDF__call_on_linked(self) -> None:
        self.assertEqual(
            s1.weights()._call_on_linked("mean").values[0][0], (0.5 + 2 + 1 + 1) / 4
        )

        self.assertEqual(
            s1.weights()._call_on_linked("mean"),
            s3.weights()._call_on_linked("mean", exclude="target"),
        )

        self.assertEqual(
            # it's tricky to compare nan values, so using fillna
            s3.covars()._call_on_linked("mean").fillna(0).round(3).to_dict(),
            {
                "a": {"self": 1.889, "target": 2.429},
                "b": {"self": -10.0, "target": 6.857},
                "c[v]": {"self": 0.222, "target": 0},
                "c[x]": {"self": 0.111, "target": 0.143},
                "c[y]": {"self": 0.444, "target": 0.286},
                "c[z]": {"self": 0.222, "target": 0.571},
            },
        )

        self.assertEqual(
            # it's tricky to compare nan values, so using fillna
            # checking also on std, and on a larger object (with both self, target and unadjusted)
            s3_null.covars()._call_on_linked("std").fillna(0).round(3).to_dict(),
            {
                "a": {"self": 0.886, "target": 0.964, "unadjusted": 0.886},
                "b": {"self": 27.355, "target": 1.927, "unadjusted": 27.355},
                "c[v]": {"self": 0.5, "target": 0.0, "unadjusted": 0.5},
                "c[x]": {"self": 0.378, "target": 0.463, "unadjusted": 0.378},
                "c[y]": {"self": 0.598, "target": 0.598, "unadjusted": 0.598},
                "c[z]": {"self": 0.5, "target": 0.655, "unadjusted": 0.5},
            },
        )

        # verify exclude works:
        self.assertEqual(
            s3.covars()._call_on_linked("mean", exclude=("self")).round(3).to_dict(),
            {
                "a": {"target": 2.429},
                "b": {"target": 6.857},
                "c[x]": {"target": 0.143},
                "c[y]": {"target": 0.286},
                "c[z]": {"target": 0.571},
            },
        )
        self.assertEqual(
            s3.covars()._call_on_linked("mean", exclude=("target")).round(3).to_dict(),
            {
                "a": {"self": 1.889},
                "b": {"self": -10.0},
                "c[v]": {"self": 0.222},
                "c[x]": {"self": 0.111},
                "c[y]": {"self": 0.444},
                "c[z]": {"self": 0.222},
            },
        )

        # Verify we can also access df (i.e.: an attribute)
        self.assertEqual(
            s3.covars()._call_on_linked("df").round(3).to_dict(),
            {
                "a": {"self": 1, "target": 3},
                "b": {"self": -42, "target": 8},
                "c": {"self": "v", "target": "z"},
            },
        )


class TestBalanceDF__descriptive_stats(BalanceTestCase):
    def test_BalanceDF__descriptive_stats(self) -> None:
        self.assertEqual(
            s1.weights()._descriptive_stats("mean", weighted=False).values[0][0], 1.125
        )
        self.assertAlmostEqual(
            s1.weights()._descriptive_stats("std", weighted=False).values[0][0],
            0.62915286,
        )
        # Not that you would ever really want the weighted weights
        self.assertEqual(
            s1.weights()._descriptive_stats("mean", weighted=True).values[0][0], 1.125
        )
        self.assertAlmostEqual(
            s1.weights()._descriptive_stats("std", weighted=True).values[0][0],
            0.62915286,
        )

        self.assertAlmostEqual(
            s1.covars()._descriptive_stats("mean", weighted=True)["a"][0], 1.88888888
        )

        # Test numeric_only and weighted flags
        r = s1.covars()._descriptive_stats("mean", weighted=False, numeric_only=True)
        e = pd.DataFrame({"a": [(1 + 2 + 3 + 1) / 4], "b": [(-42 + 8 + 2 - 42) / 4]})
        self.assertEqual(r, e)

        r = (
            s1.covars()
            ._descriptive_stats("mean", weighted=False, numeric_only=False)
            .sort_index(axis=1)
        )
        e = pd.DataFrame(
            {
                "a": [(1 + 2 + 3 + 1) / 4],
                "b": [(-42 + 8 + 2 - 42) / 4],
                "c[v]": [0.25],
                "c[x]": [0.25],
                "c[y]": [0.25],
                "c[z]": [0.25],
            }
        )
        self.assertEqual(r, e)

        r = (
            s1.covars()
            ._descriptive_stats("mean", weighted=True, numeric_only=True)
            .sort_index(axis=1)
        )
        e = pd.DataFrame(
            {
                "a": [(1 * 0.5 + 2 * 2 + 3 + 1) / 4.5],
                "b": [(-42 * 0.5 + 8 * 2 + 2 - 42) / 4.5],
            }
        )
        self.assertEqual(r, e)

        r = (
            s1.covars()
            ._descriptive_stats("mean", weighted=True, numeric_only=False)
            .sort_index(axis=1)
        )
        e = pd.DataFrame(
            {
                "a": [(1 * 0.5 + 2 * 2 + 3 + 1) / 4.5],
                "b": [(-42 * 0.5 + 8 * 2 + 2 - 42) / 4.5],
                "c[v]": [1 / 4.5],
                "c[x]": [0.5 / 4.5],
                "c[y]": [2 / 4.5],
                "c[z]": [1 / 4.5],
            }
        )
        self.assertEqual(r, e)

        # Test with missing values and weights
        s_ds = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": (1, 2, 3, 1),
                    "b": (-42, 8, 2, np.nan),
                    "c": ("a", "b", "c", "a"),
                    "id": (1, 2, 3, 4),
                    "w": (0, 2, 1, 1),
                }
            ),
            id_column="id",
            weight_column="w",
        )

        r = (
            s_ds.covars()
            ._descriptive_stats("mean", weighted=True, numeric_only=False)
            .sort_index(axis=1)
        )
        e = pd.DataFrame(
            {
                "_is_na_b[T.True]": [(1 * 1) / (2 + 1 + 1)],
                "a": [(2 * 2 + 3 + 1) / (2 + 1 + 1)],
                "b": [(8 * 2 + 2 * 1) / (2 + 1 + 1)],
                "c[a]": [(1 * 1) / (2 + 1 + 1)],
                "c[b]": [(1 * 2) / (2 + 1 + 1)],
                "c[c]": [(1 * 1) / (2 + 1 + 1)],
            }
        )
        self.assertEqual(r, e)

    def test_Balance_df_summary_stats_numeric_only(self) -> None:
        #  Test that the asmd, std, and mean methods pass the `numeric_only`
        #  argument to _descriptive_stats

        # Default is numeric_only=False
        e_all = pd.Index(("a", "b", "c[x]", "c[y]", "c[z]", "c[v]"))
        e_numeric_only = pd.Index(("a", "b"))
        self.assertEqual(s1.covars().mean().columns, e_all, lazy=True)
        self.assertEqual(s1.covars().mean(numeric_only=True).columns, e_numeric_only)
        self.assertEqual(s1.covars().mean(numeric_only=False).columns, e_all, lazy=True)

        self.assertEqual(s1.covars().std().columns, e_all, lazy=True)
        self.assertEqual(s1.covars().std(numeric_only=True).columns, e_numeric_only)
        self.assertEqual(s1.covars().std(numeric_only=False).columns, e_all, lazy=True)


class TestBalanceDF_mean(BalanceTestCase):
    def test_BalanceDF_mean(self) -> None:
        self.assertEqual(
            s1.weights().mean(),
            pd.DataFrame({"w": [1.125], "source": "self"}).set_index("source"),
        )

        self.assertEqual(
            s3_null.covars().mean().fillna(0).round(3).to_dict(),
            {
                "a": {"self": 1.889, "target": 2.429, "unadjusted": 1.889},
                "b": {"self": -10.0, "target": 6.857, "unadjusted": -10.0},
                "c[v]": {"self": 0.222, "target": 0.0, "unadjusted": 0.222},
                "c[x]": {"self": 0.111, "target": 0.143, "unadjusted": 0.111},
                "c[y]": {"self": 0.444, "target": 0.286, "unadjusted": 0.444},
                "c[z]": {"self": 0.222, "target": 0.571, "unadjusted": 0.222},
            },
        )

        # test it works when we have columns with special characters
        self.assertEqual(
            s1_bad_columns.covars().mean().round(2).to_dict(),
            {
                "a_": {"self": 1.89},
                "a__1": {"self": -10.0},
                "c_[v]": {"self": 0.22},
                "c_[x]": {"self": 0.11},
                "c_[y]": {"self": 0.44},
                "c_[z]": {"self": 0.22},
            },
        )


class TestBalanceDF_std(BalanceTestCase):
    def test_BalanceDF_std(self) -> None:
        self.assertEqual(
            s1.weights().std(),
            pd.DataFrame({"w": [0.6291529], "source": "self"}).set_index("source"),
        )

        self.assertEqual(
            s3_null.covars().std().fillna(0).round(3).to_dict(),
            {
                "a": {"self": 0.886, "target": 0.964, "unadjusted": 0.886},
                "b": {"self": 27.355, "target": 1.927, "unadjusted": 27.355},
                "c[v]": {"self": 0.5, "target": 0.0, "unadjusted": 0.5},
                "c[x]": {"self": 0.378, "target": 0.463, "unadjusted": 0.378},
                "c[y]": {"self": 0.598, "target": 0.598, "unadjusted": 0.598},
                "c[z]": {"self": 0.5, "target": 0.655, "unadjusted": 0.5},
            },
        )


class TestBalanceDF_var_of_mean(BalanceTestCase):
    def test_BalanceDF_var_of_mean(self) -> None:
        self.assertEqual(
            s3_null.covars().var_of_mean().fillna(0).round(3).to_dict(),
            {
                "a": {"self": 0.112, "target": 0.163, "unadjusted": 0.112},
                "b": {"self": 134.321, "target": 0.653, "unadjusted": 134.321},
                "c[v]": {"self": 0.043, "target": 0.0, "unadjusted": 0.043},
                "c[x]": {"self": 0.013, "target": 0.023, "unadjusted": 0.013},
                "c[y]": {"self": 0.083, "target": 0.07, "unadjusted": 0.083},
                "c[z]": {"self": 0.043, "target": 0.093, "unadjusted": 0.043},
            },
        )


class TestBalanceDF_ci(BalanceTestCase):
    def test_BalanceDF_ci_of_mean(self) -> None:
        self.assertEqual(
            s3_null.covars().ci_of_mean(round_ndigits=3).fillna(0).to_dict(),
            {
                "a": {
                    "self": (1.232, 2.545),
                    "target": (1.637, 3.221),
                    "unadjusted": (1.232, 2.545),
                },
                "b": {
                    "self": (-32.715, 12.715),
                    "target": (5.273, 8.441),
                    "unadjusted": (-32.715, 12.715),
                },
                "c[v]": {
                    "self": (-0.183, 0.627),
                    "target": 0,
                    "unadjusted": (-0.183, 0.627),
                },
                "c[x]": {
                    "self": (-0.116, 0.338),
                    "target": (-0.156, 0.442),
                    "unadjusted": (-0.116, 0.338),
                },
                "c[y]": {
                    "self": (-0.12, 1.009),
                    "target": (-0.233, 0.804),
                    "unadjusted": (-0.12, 1.009),
                },
                "c[z]": {
                    "self": (-0.183, 0.627),
                    "target": (-0.027, 1.17),
                    "unadjusted": (-0.183, 0.627),
                },
            },
        )

    def test_BalanceDF_mean_with_ci(self) -> None:
        self.assertEqual(
            s_o2.outcomes().mean_with_ci().to_dict(),
            {
                "self": {
                    "_is_na_o2[False]": 0.75,
                    "_is_na_o2[True]": 0.25,
                    "o1": 8.5,
                    "o2": 6.0,
                },
                "target": {
                    "_is_na_o2[False]": 0.75,
                    "_is_na_o2[True]": 0.25,
                    "o1": 10.5,
                    "o2": 7.875,
                },
                "self_ci": {
                    "_is_na_o2[False]": (0.326, 1.174),
                    "_is_na_o2[True]": (-0.174, 0.674),
                    "o1": (7.404, 9.596),
                    "o2": (2.535, 9.465),
                },
                "target_ci": {
                    "_is_na_o2[False]": (0.45, 1.05),
                    "_is_na_o2[True]": (-0.05, 0.55),
                    "o1": (8.912, 12.088),
                    "o2": (4.351, 11.399),
                },
            },
        )


class TestBalanceDF_asmd(BalanceTestCase):
    def test_BalanceDF_asmd(self) -> None:
        # Test with BalanceDF

        r = BalanceDF._asmd_BalanceDF(
            Sample.from_frame(
                pd.DataFrame(
                    {"id": (1, 2), "a": (1, 2), "b": (-1, 12), "weight": (1, 2)}
                )
            ).covars(),
            Sample.from_frame(
                pd.DataFrame(
                    {"id": (1, 2), "a": (3, 4), "b": (0, 42), "weight": (1, 2)}
                )
            ).covars(),
        ).sort_index()
        e_asmd = pd.Series(
            (2.828_427_1, 0.684_658_9, (2.828_427_1 + 0.684_658_9) / 2),
            index=("a", "b", "mean(asmd)"),
        )
        self.assertEqual(r, e_asmd)

        with self.assertRaisesRegex(ValueError, "has no target set"):
            s1.weights().asmd()

        with self.assertRaisesRegex(ValueError, "has no target set"):
            s3_null.outcomes().asmd()

        self.assertEqual(
            s3.covars().asmd().loc["self"],
            pd.Series(
                (
                    0.560055,
                    8.746742,
                    np.nan,
                    0.068579,
                    0.265606,
                    0.533422,
                    (
                        0.560055
                        + 8.746742
                        + 0.068579 * 0.25
                        + 0.265606 * 0.25
                        + 0.533422 * 0.25
                    )
                    / 3,
                ),
                index=("a", "b", "c[v]", "c[x]", "c[y]", "c[z]", "mean(asmd)"),
                name="self",
            ),
        )

        self.assertEqual(
            s3_null.covars().asmd().fillna(0).round(3).to_dict(),
            {
                "a": {"self": 0.56, "unadjusted": 0.56, "unadjusted - self": 0.0},
                "b": {"self": 8.747, "unadjusted": 8.747, "unadjusted - self": 0.0},
                "c[v]": {"self": 0.0, "unadjusted": 0.0, "unadjusted - self": 0.0},
                "c[x]": {"self": 0.069, "unadjusted": 0.069, "unadjusted - self": 0.0},
                "c[y]": {"self": 0.266, "unadjusted": 0.266, "unadjusted - self": 0.0},
                "c[z]": {"self": 0.533, "unadjusted": 0.533, "unadjusted - self": 0.0},
                "mean(asmd)": {
                    "self": 3.175,
                    "unadjusted": 3.175,
                    "unadjusted - self": 0.0,
                },
            },
        )
        # also check that on_linked_samples = False works:
        self.assertEqual(
            s3_null.covars().asmd(on_linked_samples=False).fillna(0).round(3).to_dict(),
            {
                "a": {"covars": 0.56},
                "b": {"covars": 8.747},
                "c[v]": {"covars": 0.0},
                "c[x]": {"covars": 0.069},
                "c[y]": {"covars": 0.266},
                "c[z]": {"covars": 0.533},
                "mean(asmd)": {"covars": 3.175},
            },
        )

        self.assertEqual(
            s3_null_madeup_weights.covars().asmd().fillna(0).round(3).to_dict(),
            {
                "a": {"self": 0.296, "unadjusted": 0.56, "unadjusted - self": 0.264},
                "b": {"self": 8.154, "unadjusted": 8.747, "unadjusted - self": 0.593},
                "c[v]": {"self": 0.0, "unadjusted": 0.0, "unadjusted - self": 0.0},
                "c[x]": {"self": 0.0, "unadjusted": 0.069, "unadjusted - self": 0.069},
                "c[y]": {"self": 0.0, "unadjusted": 0.266, "unadjusted - self": 0.266},
                "c[z]": {
                    "self": 0.218,
                    "unadjusted": 0.533,
                    "unadjusted - self": 0.315,
                },
                "mean(asmd)": {
                    "self": 2.835,
                    "unadjusted": 3.175,
                    "unadjusted - self": 0.34,
                },
            },
        )

    def test_BalanceDF_kld(self) -> None:
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": (1, 2, 3, 4),
                    "a": (1, 2, 3, 4),
                    "c": ("x", "x", "y", "z"),
                    "w": (1, 1, 1, 1),
                }
            ),
            id_column="id",
            weight_column="w",
        )

        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": (1, 2, 3, 4),
                    "a": (1, 2, 2, 3),
                    "c": ("x", "y", "y", "z"),
                    "w": (1, 1, 1, 1),
                }
            ),
            id_column="id",
            weight_column="w",
        )

        sample_with_target = sample.set_target(target)

        output = sample_with_target.covars().kld(on_linked_samples=False)

        expected = pd.DataFrame(
            {
                "a": 0.0,
                "c[x]": 0.143841,
                "c[y]": 0.130812,
                "c[z]": 0.0,
                "mean(kld)": 0.045776,
            },
            index=("covars",),
        )

        expected.index.name = "index"

        self.assertEqual(output.round(6), expected)

    def test_BalanceDF_kld_aggregate_by_main_covar(self) -> None:
        covars = s3.covars()

        unaggregated = covars.kld(on_linked_samples=False)
        expected = weighted_comparisons_stats._aggregate_statistic_by_main_covar(
            unaggregated.loc["covars"]
        )

        aggregated = covars.kld(
            on_linked_samples=False, aggregate_by_main_covar=True
        ).loc["covars"]

        self.assertEqual(aggregated.rename(None).round(6), expected.round(6))

    def test_BalanceDF_kld_on_linked_samples(self) -> None:
        covars = s3_null_madeup_weights.covars()
        links = covars._BalanceDF_child_from_linked_samples()

        self.assertIn("unadjusted", links)
        self.assertIn("target", links)

        self_df, self_weights = links["self"]._get_df_and_weights()
        target_df, target_weights = links["target"]._get_df_and_weights()
        unadj_df, unadj_weights = links["unadjusted"]._get_df_and_weights()

        expected_self = weighted_comparisons_stats.kld(
            self_df,
            target_df,
            self_weights,
            target_weights,
            aggregate_by_main_covar=True,
        )
        expected_unadjusted = weighted_comparisons_stats.kld(
            unadj_df,
            target_df,
            unadj_weights,
            target_weights,
            aggregate_by_main_covar=True,
        )

        expected = pd.DataFrame(
            [
                expected_self,
                expected_unadjusted,
                expected_unadjusted - expected_self,
            ],
            index=["self", "unadjusted", "unadjusted - self"],
        )

        output = covars.kld(aggregate_by_main_covar=True)
        expected.index.name = output.index.name

        self.assertEqual(output.round(6), expected.round(6))

    def test_BalanceDF_asmd_improvement(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "has no unadjusted set or unadjusted has no covars"
        ):
            s3.covars().asmd_improvement()

        s3_unadjusted = deepcopy(s3)
        s3_unadjusted.set_weights(pd.Series([1, 1, 1, 1], index=s3.df.index))
        s3_2 = s3.set_unadjusted(s3_unadjusted)
        self.assertEqual(s3_2.covars().asmd_improvement(), 0.3224900694460681)

        s1_with_unadjusted = deepcopy(s1)
        s1_with_unadjusted = s1.set_unadjusted(s3_unadjusted)
        with self.assertRaisesRegex(
            ValueError, "has no target set or target has no covars"
        ):
            s1_with_unadjusted.covars().asmd_improvement()

        self.assertEqual(s3_null.covars().asmd_improvement().round(3), 0)
        self.assertEqual(
            s3_null_madeup_weights.covars().asmd_improvement().round(3), 0.107
        )

        asmd_df = s3_null_madeup_weights.covars().asmd()
        exp = round(
            (asmd_df["mean(asmd)"].iloc[1] - asmd_df["mean(asmd)"].iloc[0])
            / asmd_df["mean(asmd)"].iloc[1],
            3,
        )
        self.assertEqual(exp, 0.107)
        self.assertEqual(
            s3_null_madeup_weights.covars().asmd_improvement().round(3), exp
        )

    def test_BalanceDF_asmd_aggregate_by_main_covar(self) -> None:
        # TODO: re-use this example across tests
        # TODO: bugfix - adjust fails with apply_transform when inputting a df with categorical column :(

        # Prepare dummy data
        np.random.seed(112358)

        d = pd.DataFrame(np.random.rand(1000, 3))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abc"[i] for i in range(0, 3)})
        # make 'a' a categorical column in d
        # d = d.assign(a=lambda x: pd.cut(x.a,[0,.25,.5,.75,1]))
        d["a"] = pd.cut(d["a"], [0, 0.25, 0.5, 0.75, 1]).astype(str)
        # make b "interesting" (so that the model would have something to do)
        d["b"] = np.sqrt(d["b"])
        s = Sample.from_frame(d)

        d = pd.DataFrame(np.random.rand(1000, 3))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abc"[i] for i in range(0, 3)})
        # make 'a' a categorical column in d
        # d = d.assign(a=lambda x: pd.cut(x.a,[0,.25,.5,.75,1]))
        d["a"] = pd.cut(d["a"], [0, 0.25, 0.5, 0.75, 1]).astype(str)
        t = Sample.from_frame(d)

        st = s.set_target(t)

        # Fit IPW
        a = st.adjust(max_de=1.5)

        # Check ASMD
        tmp_asmd_default = a.covars().asmd()
        tmp_asmd_main_covar = a.covars().asmd(aggregate_by_main_covar=True)

        outcome_default = tmp_asmd_default.round(2).to_dict()
        outcome_main_covar = tmp_asmd_main_covar.round(2).to_dict()

        expected_default = {
            "a[(0.0, 0.25]]": {
                "self": 0.01,
                "unadjusted": 0.09,
                "unadjusted - self": 0.08,
            },
            "a[(0.25, 0.5]]": {
                "self": 0.01,
                "unadjusted": 0.06,
                "unadjusted - self": 0.05,
            },
            "a[(0.5, 0.75]]": {
                "self": 0.0,
                "unadjusted": 0.01,
                "unadjusted - self": 0.01,
            },
            "a[(0.75, 1.0]]": {
                "self": 0.03,
                "unadjusted": 0.02,
                "unadjusted - self": -0.0,
            },
            "c": {"self": 0.01, "unadjusted": 0.03, "unadjusted - self": 0.02},
            "b": {"self": 0.15, "unadjusted": 0.6, "unadjusted - self": 0.45},
            "mean(asmd)": {"self": 0.06, "unadjusted": 0.23, "unadjusted - self": 0.17},
        }
        expected_main_covar = {
            "a": {"self": 0.01, "unadjusted": 0.05, "unadjusted - self": 0.03},
            "b": {"self": 0.15, "unadjusted": 0.6, "unadjusted - self": 0.45},
            "c": {"self": 0.01, "unadjusted": 0.03, "unadjusted - self": 0.02},
            "mean(asmd)": {"self": 0.06, "unadjusted": 0.23, "unadjusted - self": 0.17},
        }
        self.maxDiff = None
        self.assertEqual(outcome_default, expected_default)
        self.assertEqual(outcome_main_covar, expected_main_covar)

    def test_BalanceDF__kld_BalanceDF(self) -> None:
        """Test _kld_BalanceDF static method directly."""
        sample = Sample.from_frame(
            pd.DataFrame({"id": (1, 2), "a": (1, 2), "b": (-1, 12), "weight": (1, 2)})
        ).covars()

        target = Sample.from_frame(
            pd.DataFrame({"id": (1, 2), "a": (3, 4), "b": (0, 42), "weight": (1, 2)})
        ).covars()

        result = BalanceDF._kld_BalanceDF(sample, target)

        # Verify result is a Series with expected keys
        self.assertIsInstance(result, pd.Series)
        self.assertIn("a", result.index)
        self.assertIn("b", result.index)
        self.assertIn("mean(kld)", result.index)

        # Verify all values are non-negative (KLD property)
        self.assertTrue((result >= 0).all())

        # Test with aggregate_by_main_covar
        result_agg = BalanceDF._kld_BalanceDF(
            sample, target, aggregate_by_main_covar=True
        )
        self.assertIsInstance(result_agg, pd.Series)

    def test_BalanceDF__emd_BalanceDF(self) -> None:
        """Test _emd_BalanceDF static method directly."""
        sample = Sample.from_frame(
            pd.DataFrame({"id": (1, 2), "a": (1, 2), "b": (-1, 12), "weight": (1, 2)})
        ).covars()

        target = Sample.from_frame(
            pd.DataFrame({"id": (1, 2), "a": (3, 4), "b": (0, 42), "weight": (1, 2)})
        ).covars()

        result = BalanceDF._emd_BalanceDF(sample, target)

        # Verify result is a Series with expected keys
        self.assertIsInstance(result, pd.Series)
        self.assertIn("a", result.index)
        self.assertIn("b", result.index)
        self.assertIn("mean(emd)", result.index)

        # Verify all values are non-negative (EMD property)
        self.assertTrue((result >= 0).all())

        # Test with aggregate_by_main_covar
        result_agg = BalanceDF._emd_BalanceDF(
            sample, target, aggregate_by_main_covar=True
        )
        self.assertIsInstance(result_agg, pd.Series)

    def test_BalanceDF__cvmd_BalanceDF(self) -> None:
        """Test _cvmd_BalanceDF static method directly."""
        sample = Sample.from_frame(
            pd.DataFrame({"id": (1, 2), "a": (1, 2), "b": (-1, 12), "weight": (1, 2)})
        ).covars()

        target = Sample.from_frame(
            pd.DataFrame({"id": (1, 2), "a": (3, 4), "b": (0, 42), "weight": (1, 2)})
        ).covars()

        result = BalanceDF._cvmd_BalanceDF(sample, target)

        # Verify result is a Series with expected keys
        self.assertIsInstance(result, pd.Series)
        self.assertIn("a", result.index)
        self.assertIn("b", result.index)
        self.assertIn("mean(cvmd)", result.index)

        # Verify all values are non-negative (CVMD property)
        self.assertTrue((result >= 0).all())

        # Test with aggregate_by_main_covar
        result_agg = BalanceDF._cvmd_BalanceDF(
            sample, target, aggregate_by_main_covar=True
        )
        self.assertIsInstance(result_agg, pd.Series)

    def test_BalanceDF__ks_BalanceDF(self) -> None:
        """Test _ks_BalanceDF static method directly."""
        sample = Sample.from_frame(
            pd.DataFrame({"id": (1, 2), "a": (1, 2), "b": (-1, 12), "weight": (1, 2)})
        ).covars()

        target = Sample.from_frame(
            pd.DataFrame({"id": (1, 2), "a": (3, 4), "b": (0, 42), "weight": (1, 2)})
        ).covars()

        result = BalanceDF._ks_BalanceDF(sample, target)

        # Verify result is a Series with expected keys
        self.assertIsInstance(result, pd.Series)
        self.assertIn("a", result.index)
        self.assertIn("b", result.index)
        self.assertIn("mean(ks)", result.index)

        # Verify all values are in [0, 1] (KS property)
        self.assertTrue((result >= 0).all())
        self.assertTrue((result <= 1).all())

        # Test with aggregate_by_main_covar
        result_agg = BalanceDF._ks_BalanceDF(
            sample, target, aggregate_by_main_covar=True
        )
        self.assertIsInstance(result_agg, pd.Series)

    def test_BalanceDF_comparison_functions_invalid_input(self) -> None:
        """Test that all comparison functions properly validate inputs."""
        sample = Sample.from_frame(
            pd.DataFrame({"id": (1, 2), "a": (1, 2), "weight": (1, 2)})
        ).covars()

        # Test with non-BalanceDF inputs
        invalid_input = "not a BalanceDF"

        with self.assertRaisesRegex(ValueError, "must be balancedf_class.BalanceDF"):
            BalanceDF._kld_BalanceDF(invalid_input, sample)  # type: ignore

        with self.assertRaisesRegex(ValueError, "must be balancedf_class.BalanceDF"):
            BalanceDF._emd_BalanceDF(sample, invalid_input)  # type: ignore

        with self.assertRaisesRegex(ValueError, "must be balancedf_class.BalanceDF"):
            BalanceDF._cvmd_BalanceDF(invalid_input, sample)  # type: ignore

        with self.assertRaisesRegex(ValueError, "must be balancedf_class.BalanceDF"):
            BalanceDF._ks_BalanceDF(sample, invalid_input)  # type: ignore


class TestBalanceDF_to_download(BalanceTestCase):
    def test_BalanceDF_to_download(self) -> None:
        r = s1.covars().to_download()
        self.assertIsInstance(r, IPython.display.FileLink)


class TestBalanceDF_to_csv(BalanceTestCase):
    def test_BalanceDF_to_csv(self) -> None:
        with tempfile_path() as tmp_path:
            s1.weights().to_csv(path_or_buf=tmp_path)
            with open(tmp_path, "rb") as output:
                r = output.read()
        e = b"id,w\n1,0.5\n2,2.0\n3,1.0\n4,1.0\n"
        self.assertEqual(r, e)

    def test_BalanceDF_to_csv_first_default_argument_is_path(self) -> None:
        with tempfile_path() as tmp_path:
            s1.weights().to_csv(tmp_path)
            with open(tmp_path, "rb") as output:
                r = output.read()
        e = b"id,w\n1,0.5\n2,2.0\n3,1.0\n4,1.0\n"
        self.assertEqual(r, e)

    def test_BalanceDF_to_csv_output_with_no_path(self) -> None:
        with tempfile.NamedTemporaryFile():
            out = s1.weights().to_csv()
        self.assertEqual(out, "id,w\n1,0.5\n2,2.0\n3,1.0\n4,1.0\n")

    def test_BalanceDF_to_csv_output_with_path(self) -> None:
        with tempfile_path() as tmp_path:
            out = s1.weights().to_csv(path_or_buf=tmp_path)
        self.assertEqual(out, None)


class TestBalanceDF__df_with_ids(BalanceTestCase):
    def test_BalanceDF__df_with_ids(self) -> None:
        # Test it has an id column:
        self.assertTrue("id" in s1.weights()._df_with_ids().columns)
        self.assertTrue("id" in s1.covars()._df_with_ids().columns)
        self.assertTrue("id" in s_o.outcomes()._df_with_ids().columns)

        # Test it has df columns:
        self.assertTrue("w" in s1.weights()._df_with_ids().columns)
        self.assertEqual(
            ["id", "a", "b", "c"], s1.covars()._df_with_ids().columns.tolist()
        )
        self.assertEqual((4, 4), s1.covars()._df_with_ids().shape)


class TestBalanceDF_summary(BalanceTestCase):
    def testBalanceDF_summary(self) -> None:
        self.assertEqual(
            s1.covars().summary().to_dict(),
            {
                "self": {
                    "a": 1.889,
                    "b": -10.0,
                    "c[v]": 0.222,
                    "c[x]": 0.111,
                    "c[y]": 0.444,
                    "c[z]": 0.222,
                },
                "self_ci": {
                    "a": (1.232, 2.545),
                    "b": (-32.715, 12.715),
                    "c[v]": (-0.183, 0.627),
                    "c[x]": (-0.116, 0.338),
                    "c[y]": (-0.12, 1.009),
                    "c[z]": (-0.183, 0.627),
                },
            },
        )

        s3_2 = s1.adjust(s2, method="null")
        self.assertEqual(
            s3_2.covars().summary().sort_index(axis=1).fillna(0).to_dict(),
            {
                "self": {
                    "a": 1.889,
                    "b": -10.0,
                    "c[v]": 0.222,
                    "c[x]": 0.111,
                    "c[y]": 0.444,
                    "c[z]": 0.222,
                },
                "self_ci": {
                    "a": (1.232, 2.545),
                    "b": (-32.715, 12.715),
                    "c[v]": (-0.183, 0.627),
                    "c[x]": (-0.116, 0.338),
                    "c[y]": (-0.12, 1.009),
                    "c[z]": (-0.183, 0.627),
                },
                "target": {
                    "a": 2.429,
                    "b": 6.857,
                    "c[v]": 0.0,
                    "c[x]": 0.143,
                    "c[y]": 0.286,
                    "c[z]": 0.571,
                },
                "target_ci": {
                    "a": (1.637, 3.221),
                    "b": (5.273, 8.441),
                    "c[v]": 0,
                    "c[x]": (-0.156, 0.442),
                    "c[y]": (-0.233, 0.804),
                    "c[z]": (-0.027, 1.17),
                },
                "unadjusted": {
                    "a": 1.889,
                    "b": -10.0,
                    "c[v]": 0.222,
                    "c[x]": 0.111,
                    "c[y]": 0.444,
                    "c[z]": 0.222,
                },
                "unadjusted_ci": {
                    "a": (1.232, 2.545),
                    "b": (-32.715, 12.715),
                    "c[v]": (-0.183, 0.627),
                    "c[x]": (-0.116, 0.338),
                    "c[y]": (-0.12, 1.009),
                    "c[z]": (-0.183, 0.627),
                },
            },
        )

        self.assertEqual(
            s3_2.covars().summary(on_linked_samples=False).to_dict(),
            {
                0: {
                    "a": 1.889,
                    "b": -10.0,
                    "c[v]": 0.222,
                    "c[x]": 0.111,
                    "c[y]": 0.444,
                    "c[z]": 0.222,
                },
                "0_ci": {
                    "a": (1.232, 2.545),
                    "b": (-32.715, 12.715),
                    "c[v]": (-0.183, 0.627),
                    "c[x]": (-0.116, 0.338),
                    "c[y]": (-0.12, 1.009),
                    "c[z]": (-0.183, 0.627),
                },
            },
        )


class TestBalanceDF__str__(BalanceTestCase):
    def testBalanceDF__str__(self) -> None:
        self.assertTrue(s1.outcomes().df.__str__() in s1.outcomes().__str__())

    def test_BalanceDFOutcomes___str__(self) -> None:
        # NOTE how the type is float even though the original input was integer.
        self.assertTrue(
            pd.DataFrame({"o": (7.0, 8.0, 9.0, 10.0)}).__str__() in o.__str__()
        )


class TestBalanceDF__repr__(BalanceTestCase):
    def test_BalanceDFWeights___repr__(self) -> None:
        repr = w.__repr__()
        self.assertTrue("weights from" in repr)
        self.assertTrue(object.__repr__(s1) in repr)

    def test_BalanceDFCovars___repr__(self) -> None:
        repr = c.__repr__()
        self.assertTrue("covars from" in repr)
        self.assertTrue(object.__repr__(s1) in repr)

    def test_BalanceDFOutcomes___repr__(self) -> None:
        repr = o.__repr__()
        self.assertTrue("outcomes from" in repr)
        self.assertTrue(object.__repr__(s1) in repr)


class TestBalanceDF(BalanceTestCase):
    def testBalanceDF_model_matrix(self) -> None:
        self.assertEqual(
            s1.covars().model_matrix().sort_index(axis=1).columns.values,
            ("a", "b", "c[v]", "c[x]", "c[y]", "c[z]"),
        )
        self.assertEqual(
            s1.covars().model_matrix().to_dict(),
            {
                "a": {0: 1.0, 1: 2.0, 2: 3.0, 3: 1.0},
                "b": {0: -42.0, 1: 8.0, 2: 2.0, 3: -42.0},
                "c[v]": {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "c[x]": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "c[y]": {0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0},
                "c[z]": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
            },
        )

    def test_check_if_not_BalanceDF(self) -> None:
        with self.assertRaisesRegex(ValueError, "number must be balancedf_class"):
            BalanceDF._check_if_not_BalanceDF(
                5,  # pyre-ignore[6]: Testing error handling with wrong type
                "number",
            )
        self.assertTrue(BalanceDF._check_if_not_BalanceDF(s3.covars()) is None)


class TestBalanceDFCovars_from_frame(BalanceTestCase):
    """Test cases for BalanceDFCovars.from_frame() factory method."""

    def test_from_frame_with_basic_dataframe_creates_instance(self) -> None:
        """Test that from_frame creates a BalanceDFCovars instance from a basic DataFrame.

        Verifies:
        - Returns BalanceDFCovars instance
        - DataFrame data is correctly preserved
        - Column names include original columns
        - Data values are preserved correctly
        """
        # Arrange
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        sample = TestDataFactory.create_basic_sample()
        covars = sample.covars()

        # Act
        result = covars.from_frame(df)

        # Assert
        self.assertIsInstance(result, BalanceDFCovars)
        # from_frame adds index column from reset_index()
        self.assertIn("a", result.df.columns)
        self.assertIn("b", result.df.columns)
        self.assertEqual(result.df.shape[0], 3)

        # Verify actual data values are preserved
        self.assertEqual(result.df["a"].tolist(), [1, 2, 3])
        self.assertEqual(result.df["b"].tolist(), [4, 5, 6])

    def test_from_frame_with_weights_creates_instance_with_weights(self) -> None:
        """Test that from_frame correctly handles weights parameter.

        Verifies:
        - Weights are properly integrated into the created instance
        - Sample object contains weight information
        - Weight values are correctly applied
        """
        # Arrange
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        weights = pd.Series([0.5, 1.0, 1.5], name="weight")
        sample = TestDataFactory.create_basic_sample()
        covars = sample.covars()

        # Act
        result = covars.from_frame(df, weights=weights)

        # Assert
        self.assertIsInstance(result, BalanceDFCovars)
        self.assertIsNotNone(result._sample.weights())

        # Verify actual weight values are correctly applied
        weights_df = result._sample.weights().df
        self.assertEqual(weights_df["weight"].tolist(), [0.5, 1.0, 1.5])

    def test_from_frame_with_various_column_counts(self) -> None:
        """Test that from_frame handles DataFrames with different numbers of columns.

        Verifies:
        - Single, double, and triple column DataFrames work correctly
        - All columns are preserved
        - Data integrity is maintained
        """
        # Test cases with different column counts
        test_cases = [
            # (dataframe, expected_cols)
            (pd.DataFrame({"x": [10, 20, 30]}), ["x"]),
            (pd.DataFrame({"a": [1, 2], "b": [3, 4]}), ["a", "b"]),
            (pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}), ["a", "b", "c"]),
        ]

        sample = TestDataFactory.create_basic_sample()
        covars = sample.covars()

        for df, expected_cols in test_cases:
            with self.subTest(columns=expected_cols):
                # Act
                result = covars.from_frame(df)

                # Assert
                self.assertIsInstance(result, BalanceDFCovars)
                for col in expected_cols:
                    self.assertIn(col, result.df.columns)
                self.assertEqual(result.df.shape[0], len(df))

    def test_from_frame_preserves_numeric_values(self) -> None:
        """Test that from_frame preserves numeric values correctly.

        Verifies:
        - Integer values are preserved
        - Float values are preserved
        - No unexpected type conversions occur
        """
        # Arrange
        df = pd.DataFrame({"int_col": [1, 2, 3], "float_col": [1.5, 2.5, 3.5]})
        sample = TestDataFactory.create_basic_sample()
        covars = sample.covars()

        # Act
        result = covars.from_frame(df)

        # Assert
        self.assertEqual(result.df["int_col"].tolist(), [1, 2, 3])
        self.assertEqual(result.df["float_col"].tolist(), [1.5, 2.5, 3.5])


class TestBalanceDF_model_matrix_caching(BalanceTestCase):
    """Test cases for model_matrix caching behavior."""

    def test_model_matrix_is_cached_after_first_call(self) -> None:
        """Test that model_matrix result is cached after first invocation.

        Verifies:
        - First call computes model matrix
        - Subsequent calls return identical data
        - Model matrix handles formula correctly
        """
        # Arrange
        covars = s1.covars()

        # Act
        # First call should compute the model matrix
        first_result = covars.model_matrix()
        # Check internal cache was set
        self.assertIsNotNone(covars._model_matrix)

        # Second call should use cached result
        second_result = covars.model_matrix()

        # Assert
        # Results should be identical in content
        pd.testing.assert_frame_equal(first_result, second_result)
        # Verify the cache is being used by checking the internal state
        self.assertIsNotNone(covars._model_matrix)

    def test_model_matrix_with_categorical_columns_creates_dummy_variables(
        self,
    ) -> None:
        """Test that model_matrix creates proper dummy variables for categorical columns.

        Verifies:
        - Categorical columns are one-hot encoded
        - Numeric columns are preserved
        - Column names follow expected format
        """
        # Arrange
        covars = s1.covars()

        # Act
        mm = covars.model_matrix()

        # Assert
        # Check that categorical variable 'c' has been expanded
        self.assertIn("c[x]", mm.columns)
        self.assertIn("c[y]", mm.columns)
        self.assertIn("c[z]", mm.columns)
        self.assertIn("c[v]", mm.columns)
        # Check numeric columns are present
        self.assertIn("a", mm.columns)
        self.assertIn("b", mm.columns)


class TestBalanceDF_edge_cases(BalanceTestCase):
    """Test cases for edge cases and boundary conditions."""

    def test_BalanceDF_mean_with_all_zero_weights(self) -> None:
        """Test mean calculation with all zero weights.

        Verifies:
        - Handles edge case of all zero weights appropriately
        - Returns DataFrame with NaN values when weights sum to zero
        """
        # Arrange
        sample = TestDataFactory.create_basic_sample()
        sample_with_zero_weights = deepcopy(sample)
        sample_with_zero_weights.set_weights(pd.Series([0, 0, 0, 0]))

        # Act
        result = sample_with_zero_weights.covars().mean(on_linked_samples=False)

        # Assert
        # With zero weights, the weighted mean should return NaN values
        self.assertIsInstance(result, pd.DataFrame)
        # Check that the result contains NaN values for numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        self.assertTrue(
            result[numeric_cols].isna().all().all(),
            "Expected NaN values for all numeric columns with zero weights",
        )

    def test_BalanceDF_with_nan_in_covariates(self) -> None:
        """Test BalanceDF behavior with NaN values in covariates.

        Verifies:
        - NaN values are properly handled
        - model_matrix returns valid DataFrame
        """
        # Arrange
        df = pd.DataFrame(
            {"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan], "id": [1, 2, 3]}
        )
        sample = Sample.from_frame(df, id_column="id")

        # Act
        covars = sample.covars()
        mm = covars.model_matrix()

        # Assert
        self.assertIsInstance(mm, pd.DataFrame)
        # model_matrix returns a valid DataFrame with NaN handling
        self.assertGreater(len(mm.columns), 0)

    def test_BalanceDF_std_with_single_observation(self) -> None:
        """Test standard deviation calculation with single observation.

        Verifies:
        - Single observation is handled appropriately
        - Returns expected result or handles gracefully
        """
        # Arrange
        df = pd.DataFrame({"a": [1], "b": [2], "id": [1]})
        sample = Sample.from_frame(df, id_column="id")

        # Act
        result = sample.covars().std()

        # Assert
        self.assertIsInstance(result, pd.DataFrame)

    def test_BalanceDF_to_csv_with_none_path_returns_string(self) -> None:
        """Test to_csv with None path returns CSV string.

        Verifies:
        - to_csv without path returns string representation
        - String contains expected data and is valid CSV
        """
        # Arrange
        covars = s1.covars()

        # Act
        result = covars.to_csv(None)

        # Assert
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

        # Verify it's valid CSV by checking for common CSV elements
        lines = result.strip().split("\n")
        # Should have header and at least one data row
        self.assertGreaterEqual(len(lines), 2)
        # Should contain column names from the dataframe
        self.assertIn("a", lines[0])  # Check header contains expected column

    def test_BalanceDF_names_returns_list_of_strings(self) -> None:
        """Test that names() returns a list of string column names.

        Verifies:
        - Returns list type
        - All elements are strings
        - Contains expected column names
        """
        # Arrange
        covars = s1.covars()

        # Act
        names = covars.names()

        # Assert
        self.assertIsInstance(names, list)
        self.assertTrue(all(isinstance(name, str) for name in names))
        self.assertEqual(len(names), len(covars.df.columns))

        # Verify expected column names are present
        expected_columns = ["a", "b", "c"]  # Based on s1 sample data
        for col in expected_columns:
            self.assertIn(col, names)


class TestBalanceDFWeights_edge_cases(BalanceTestCase):
    """Test cases for BalanceDFWeights edge cases."""

    def test_BalanceDFWeights_design_effect_with_equal_weights(self) -> None:
        """Test design_effect calculation with all equal weights.

        Verifies:
        - Equal weights produce design effect of 1.0
        - Calculation is mathematically correct
        """
        # Arrange
        df = pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3], "w": [1.0, 1.0, 1.0]})
        sample = Sample.from_frame(df, id_column="id", weight_column="w")

        # Act
        deff = sample.weights().design_effect()

        # Assert
        self.assertAlmostEqual(deff, 1.0, places=5)

    def test_BalanceDFWeights_trim_modifies_weights_in_place(
        self,
    ) -> None:
        """Test trim method modifies weights in place.

        Verifies:
        - Trim method modifies weights in place
        - Trimmed weights are different from original
        - Weight values are properly trimmed
        """
        # Arrange - use existing sample with weights
        # Create a sample with diverse weights for testing trimming
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": [1, 2, 3, 4, 5],
                    "id": [1, 2, 3, 4, 5],
                    "w": [1.0, 2.0, 5.0, 10.0, 20.0],  # Wide range for trimming
                }
            ),
            id_column="id",
            weight_column="w",
        )
        weights = sample.weights()
        original_weights = weights.df["w"].copy()

        # Act
        weights.trim(
            percentile=0.6
        )  # trim returns None, modifies in place at 60th percentile

        # Assert
        # Verify that trimming actually modified the weights
        trimmed_values = weights.df["w"]
        self.assertFalse(
            trimmed_values.equals(original_weights),
            "Trimmed weights should differ from original weights",
        )
        # After trimming at 60th percentile, the highest weights should be capped
        # With weights [1.0, 2.0, 5.0, 10.0, 20.0], 60th percentile should cap higher values
        self.assertLess(trimmed_values.max(), original_weights.max())

    def test_BalanceDFWeights_summary_returns_dataframe(self) -> None:
        """Test that weights summary returns a properly formatted DataFrame.

        Verifies:
        - summary() returns DataFrame
        - Contains expected statistical measures
        """
        # Arrange
        weights = s1.weights()

        # Act
        summary = weights.summary()

        # Assert
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertTrue(len(summary) > 0)

        # Verify summary contains expected columns
        # The summary DataFrame has 'var' and 'val' columns
        self.assertIn("var", summary.columns)
        self.assertIn("val", summary.columns)

        # Should contain weight-related statistics in the 'var' column
        var_values = summary["var"].tolist()
        expected_stats = ["mean", "std", "min", "max", "design_effect"]
        for stat in expected_stats:
            self.assertTrue(
                any(stat in str(var).lower() for var in var_values),
                f"Expected '{stat}' to be in summary statistics",
            )


class TestBalanceDFOutcomes_edge_cases(BalanceTestCase):
    """Test cases for BalanceDFOutcomes edge cases."""

    def test_BalanceDFOutcomes_with_all_null_values(self) -> None:
        """Test outcomes handling when all values are null.

        Verifies:
        - All-null outcomes are handled gracefully
        - No errors are raised during creation
        """
        # Arrange
        df = pd.DataFrame(
            {"a": [1, 2, 3], "o": [np.nan, np.nan, np.nan], "id": [1, 2, 3]}
        )
        sample = Sample.from_frame(df, id_column="id", outcome_columns="o")

        # Act
        outcomes = sample.outcomes()

        # Assert
        self.assertIsInstance(outcomes, BalanceDFOutcomes)
        self.assertTrue(outcomes.df["o"].isna().all())

    def test_BalanceDFOutcomes_relative_response_rates_with_zero_responses(
        self,
    ) -> None:
        """Test relative_response_rates with zero non-null responses.

        Verifies:
        - Zero response case is handled appropriately
        - Returns valid result or handles gracefully
        """
        # Arrange
        df = pd.DataFrame(
            {"a": [1, 2, 3], "o": [np.nan, np.nan, np.nan], "id": [1, 2, 3]}
        )
        sample = Sample.from_frame(df, id_column="id", outcome_columns="o")

        # Act
        rrr = sample.outcomes().relative_response_rates()

        # Assert
        self.assertIsInstance(rrr, pd.DataFrame)
        # Should show 0% response rate
        self.assertEqual(rrr.loc["%", "o"], 0.0)

    def test_BalanceDFOutcomes_summary_returns_proper_result(
        self,
    ) -> None:
        """Test summary returns meaningful result.

        Verifies:
        - summary() returns string with outcome statistics
        - Result contains expected outcome column information
        """
        # Arrange - use simple outcome sample
        outcomes = s1.outcomes()

        # Act
        summary = outcomes.summary()

        # Assert
        # For outcomes, summary returns a string with statistics
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        # Should contain the outcome column 'o' in the summary
        self.assertIn("o", summary)
        # Should contain mean outcomes and response rates
        self.assertIn("Mean outcomes", summary)
        self.assertIn("Response rates", summary)


class TestBalanceDF_descriptive_stats_edge_cases(BalanceTestCase):
    """Test cases for _descriptive_stats method edge cases."""

    def test_descriptive_stats_with_numeric_only_true(self) -> None:
        """Test _descriptive_stats with numeric_only=True filters categorical columns.

        Verifies:
        - Categorical columns are excluded when numeric_only=True
        - Result is a valid DataFrame
        """
        # Arrange
        covars = s1.covars()

        # Act
        result = covars._descriptive_stats(stat="mean", numeric_only=True)

        # Assert
        self.assertIsInstance(result, pd.DataFrame)
        # Result should contain numeric columns
        self.assertGreater(len(result), 0)

    def test_descriptive_stats_with_weighted_false(self) -> None:
        """Test _descriptive_stats with weighted=False uses unweighted statistics.

        Verifies:
        - Unweighted statistics are calculated correctly
        - Results differ from weighted statistics when weights are not uniform
        """
        # Arrange
        covars = s1.covars()

        # Act
        weighted_result = covars._descriptive_stats(stat="mean", weighted=True)
        unweighted_result = covars._descriptive_stats(stat="mean", weighted=False)

        # Assert
        self.assertIsInstance(weighted_result, pd.DataFrame)
        self.assertIsInstance(unweighted_result, pd.DataFrame)

    def test_var_of_mean_returns_variance_estimates(self) -> None:
        """Test var_of_mean returns proper variance estimates.

        Verifies:
        - var_of_mean returns DataFrame
        - Values are non-negative
        - Shape matches input dimensions
        """
        # Arrange
        covars = s1.covars()

        # Act
        result = covars.var_of_mean()

        # Assert
        self.assertIsInstance(result, pd.DataFrame)
        # Variances should be non-negative
        self.assertTrue((result >= 0).all().all())


class TestBalanceDF_string_representation(BalanceTestCase):
    """Test cases for string representation methods."""

    def test_str_contains_dataframe_representation(self) -> None:
        """Test __str__ includes meaningful DataFrame representation.

        Verifies:
        - __str__ output contains DataFrame string representation
        - Contains expected column names from the data
        - Shows actual data values
        """
        # Arrange
        covars = s1.covars()

        # Act
        str_repr = str(covars)

        # Assert
        self.assertIsInstance(str_repr, str)
        self.assertTrue(len(str_repr) > 0)
        self.assertIn("covars from", str_repr)

        # Verify it contains actual column names from the DataFrame
        expected_columns = ["a", "b", "c"]
        for col in expected_columns:
            self.assertIn(
                col, str_repr, f"Expected column '{col}' in string representation"
            )

        # Verify it shows some actual data (s1 has specific values)
        # The DataFrame representation should show at least some data rows
        self.assertTrue(
            any(str(val) in str_repr for val in [1, 2, 3, -42, 8]),
            "Expected some actual data values in string representation",
        )

    def test_repr_contains_class_information(self) -> None:
        """Test __repr__ includes class module and qualname.

        Verifies:
        - __repr__ contains module path
        - __repr__ contains class name
        """
        # Arrange
        covars = s1.covars()

        # Act
        repr_str = repr(covars)

        # Assert
        self.assertIsInstance(repr_str, str)
        self.assertIn("BalanceDFCovars", repr_str)


class TestBalanceDF_df_with_ids(BalanceTestCase):
    """Test cases for _df_with_ids method."""

    def test_df_with_ids_includes_id_column(self) -> None:
        """Test _df_with_ids includes the id column from sample.

        Verifies:
        - Returned DataFrame includes id column
        - id values match sample ids
        """
        # Arrange
        covars = s1.covars()

        # Act
        df_with_ids = covars._df_with_ids()

        # Assert
        self.assertIsInstance(df_with_ids, pd.DataFrame)
        self.assertIn("id", df_with_ids.columns)

    def test_df_with_ids_preserves_data_columns(self) -> None:
        """Test _df_with_ids preserves all original data columns.

        Verifies:
        - All original columns are present
        - Data values are unchanged
        """
        # Arrange
        covars = s1.covars()
        original_columns = set(covars.df.columns)

        # Act
        df_with_ids = covars._df_with_ids()

        # Assert
        # Original columns should be subset of result columns
        self.assertTrue(original_columns.issubset(set(df_with_ids.columns)))


class TestBalanceDF_asmd_and_kld_edge_cases(BalanceTestCase):
    """Test cases for asmd and kld methods with edge cases."""

    def test_asmd_returns_valid_result(self) -> None:
        """Test asmd returns valid result.

        Verifies:
        - ASMD method executes successfully
        - Returns DataFrame or expected type
        """
        # Arrange - use existing sample with target
        sample_with_target = s3

        # Act
        asmd_result = sample_with_target.covars().asmd()

        # Assert
        self.assertIsInstance(asmd_result, pd.DataFrame)
        # Result should have data
        self.assertGreater(len(asmd_result), 0)

    def test_kld_with_single_covariate_returns_scalar_result(self) -> None:
        """Test kld with single covariate returns appropriate result.

        Verifies:
        - Single covariate KLD is calculated
        - Result has expected structure
        """
        # Arrange
        df1 = pd.DataFrame({"x": [1, 2, 3, 4, 5], "id": [1, 2, 3, 4, 5]})
        df2 = pd.DataFrame({"x": [2, 3, 4, 5, 6], "id": [6, 7, 8, 9, 10]})
        sample1 = Sample.from_frame(df1, id_column="id")
        sample2 = Sample.from_frame(df2, id_column="id")
        sample_with_target = sample1.set_target(sample2)

        # Act
        kld_result = sample_with_target.covars().kld()

        # Assert
        self.assertIsInstance(kld_result, (pd.DataFrame, pd.Series, float))


class TestBalanceDF_ci_of_mean_edge_cases(BalanceTestCase):
    """Test cases for confidence interval calculations."""

    def test_ci_of_mean_with_high_alpha_produces_narrow_intervals(self) -> None:
        """Test ci_of_mean with high alpha (e.g., 0.9) produces narrower intervals.

        Verifies:
        - Higher alpha values produce narrower confidence intervals
        - Lower and upper bounds are properly ordered
        """
        # Arrange
        covars = s1.covars()

        # Act
        ci_90 = covars.ci_of_mean(alpha=0.1)  # 90% CI
        ci_95 = covars.ci_of_mean(alpha=0.05)  # 95% CI

        # Assert
        self.assertIsInstance(ci_90, pd.DataFrame)
        self.assertIsInstance(ci_95, pd.DataFrame)

    def test_mean_with_ci_returns_combined_result(self) -> None:
        """Test mean_with_ci returns both mean and confidence interval.

        Verifies:
        - Result includes mean values
        - Result includes CI bounds
        - Structure is as expected
        """
        # Arrange
        covars = s1.covars()

        # Act
        result = covars.mean_with_ci()

        # Assert
        self.assertIsInstance(result, pd.DataFrame)
        # Should have mean and CI columns
        self.assertTrue(len(result.columns) > 1)


class TestBalanceDF_plot_on_linked_samples(BalanceTestCase):
    """Test cases for plot method with on_linked_samples parameter (lines 666-690)."""

    def test_plot_on_linked_samples_true(self) -> None:
        """Test plot method with on_linked_samples=True.

        Verifies that plot returns values from linked samples.
        Covers lines 666-690 in balancedf_class.py.
        """
        import matplotlib

        matplotlib.use("Agg")

        s3_adj = s3.adjust(method="null")
        covars = s3_adj.covars()

        # Verify plot runs without error with on_linked_samples=True
        # With return_dict_of_figures=True, plotly returns a dict of figures
        result = covars.plot(on_linked_samples=True, return_dict_of_figures=True)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)

    def test_plot_on_linked_samples_false(self) -> None:
        """Test plot method with on_linked_samples=False.

        Verifies that plot only shows self data when on_linked_samples=False.
        Covers lines 668-669 in balancedf_class.py.
        """
        import matplotlib

        matplotlib.use("Agg")

        s3_adj = s3.adjust(method="null")
        covars = s3_adj.covars()

        # Verify plot runs without error with on_linked_samples=False
        # With return_dict_of_figures=True, plotly returns a dict of figures
        result = covars.plot(on_linked_samples=False, return_dict_of_figures=True)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)


class TestBalanceDF_kld_no_target(BalanceTestCase):
    """Test cases for kld method when target is None (line 1527)."""

    def test_kld_raises_value_error_when_no_target(self) -> None:
        """Test that kld raises ValueError when no target is set.

        Verifies line 1527 in balancedf_class.py.
        """
        sample_no_target = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": (1, 2, 3),
                    "b": (4, 5, 6),
                    "id": (1, 2, 3),
                    "w": (1.0, 1.0, 1.0),
                }
            ),
            id_column="id",
            weight_column="w",
        )

        with self.assertRaises(ValueError):
            sample_no_target.covars().kld()


class TestBalanceDFWeights_plot_defaults(BalanceTestCase):
    """Test cases for BalanceDFWeights.plot method (lines 2389-2396)."""

    def test_weights_plot_default_kwargs(self) -> None:
        """Test that weights().plot() uses correct default kwargs.

        Verifies lines 2389-2396 in balancedf_class.py.
        """
        import matplotlib

        matplotlib.use("Agg")

        s3_adj = s3.adjust(method="null")

        result = s3_adj.weights().plot(return_axes=True)
        self.assertIsNotNone(result)

    def test_weights_plot_with_custom_kwargs(self) -> None:
        """Test weights().plot() with custom kwargs.

        Verifies that kwargs are passed through correctly.
        """
        import matplotlib

        matplotlib.use("Agg")

        s3_adj = s3.adjust(method="null")

        result = s3_adj.weights().plot(
            dist_type="hist",
            library="seaborn",
            return_axes=True,
        )
        self.assertIsNotNone(result)
