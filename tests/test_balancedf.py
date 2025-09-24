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
    BalanceCovarsDF,  # noqa
    BalanceDF,
    BalanceOutcomesDF,  # noqa
    BalanceWeightsDF,  # noqa
)
from balance.sample_class import Sample
from balance.testutil import BalanceTestCase


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
o: BalanceOutcomesDF = s1.outcomes()
s_o: Sample = TestDataFactory.create_multi_outcome_sample()
t_o: Sample = TestDataFactory.create_large_target_outcome_sample()
s_o2: Sample = s_o.set_target(t_o)
c: BalanceCovarsDF = s1.covars()
w: BalanceWeightsDF = s1.weights()
s1_bad_columns: Sample = TestDataFactory.create_sample_with_special_characters()


class TestBalanceOutcomesDF(BalanceTestCase):
    """Test cases for BalanceOutcomesDF class functionality."""

    def test_Sample_outcomes(self) -> None:
        """Test that Sample.outcomes() returns correct BalanceOutcomesDF instances.

        Verifies:
        - Returns BalanceOutcomesDF instance for samples with outcomes
        - Handles multicharacter column names correctly
        - Returns None for samples without outcomes
        """
        self.assertTrue(isinstance(s4.outcomes(), BalanceOutcomesDF))
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

    def test_BalanceOutcomesDF_df(self) -> None:
        """Test BalanceOutcomesDF.df property behavior and data integrity.

        Verifies:
        - df is implemented as a property decorator
        - Property cannot be called as a function
        - Data values are correctly converted to float type
        - Property accessor works correctly
        """
        # Verify that the @property decorator works properly.
        self.assertTrue(isinstance(BalanceOutcomesDF.df, property))
        # We can no longer call .df() as if it was a function:
        with self.assertRaisesRegex(TypeError, "'DataFrame' object is not callable"):
            o.df()  # pyre-ignore[29]: Testing property call error
        # Here is how we can call it as a function:
        self.assertEqual(BalanceOutcomesDF.df.fget(o), o.df)

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

    def test_BalanceOutcomesDF_names(self) -> None:
        """Test that BalanceOutcomesDF.names() returns correct outcome column names."""
        self.assertEqual(o.names(), ["o"])

    def test_BalanceOutcomesDF__sample(self) -> None:
        """Test that BalanceOutcomesDF._sample references the correct source sample."""
        self.assertTrue(o._sample is s1)

    def test_BalanceOutcomesDF_weights(self) -> None:
        """Test that BalanceOutcomesDF._weights returns correct weight Series."""
        pd.testing.assert_series_equal(o._weights, pd.Series((0.5, 2, 1, 1)))

    def test_BalanceOutcomesDF_relative_response_rates(self) -> None:
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

    def test_BalanceOutcomesDF_target_response_rates(self) -> None:
        """Test target_response_rates method for calculating target sample response rates."""
        self.assertEqual(
            s_o2.outcomes().target_response_rates(),
            pd.DataFrame({"o1": {"n": 8.0, "%": 100.0}, "o2": {"n": 6.0, "%": 75.0}}),
            lazy=True,
        )

    def test_BalanceOutcomesDF_summary(self) -> None:
        """Test BalanceOutcomesDF.summary() method output format and content.

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


class TestBalanceCovarsDF(BalanceTestCase):
    """Test cases for BalanceCovarsDF class functionality."""

    def test_BalanceCovarsDF_df(self) -> None:
        """Test BalanceCovarsDF.df property behavior and data type conversion.

        Verifies:
        - df is implemented as a property decorator
        - Property cannot be called as a function
        - Integer values are converted to floats in DataFrame
        - String values are preserved unchanged
        """
        # Verify that the @property decorator works properly.
        self.assertTrue(isinstance(BalanceCovarsDF.df, property))
        self.assertEqual(BalanceOutcomesDF.df.fget(c), c.df)
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

    def test_BalanceCovarsDF_names(self) -> None:
        """Test that BalanceCovarsDF.names() returns correct column names as list."""
        self.assertEqual(c.names(), ["a", "b", "c"])
        self.assertEqual(type(c.names()), list)

    def test_BalanceCovarsDF__sample(self) -> None:
        """Test that BalanceCovarsDF._sample references the correct source sample."""
        self.assertTrue(c._sample is s1)

    def test_BalanceCovarsDF_weights(self) -> None:
        """Test that BalanceCovarsDF._weights returns correct weight Series."""
        pd.testing.assert_series_equal(
            c._weights, pd.Series(np.array([0.5, 2.0, 1.0, 1.0]))
        )


class TestBalanceWeightsDF(BalanceTestCase):
    """Test cases for BalanceWeightsDF class functionality."""

    def test_BalanceWeightsDF_df(self) -> None:
        """Test BalanceWeightsDF.df property behavior and weight data access.

        Verifies:
        - df is implemented as a property decorator
        - Property cannot be called as a function
        - DataFrame contains correct weight values
        """
        # Verify that the @property decorator works properly.
        self.assertTrue(isinstance(BalanceWeightsDF.df, property))
        self.assertEqual(
            BalanceWeightsDF.df.fget(w),  # pyre-ignore[29]: Testing property getter
            w.df,
        )
        # We can no longer call .df() as if it was a function:
        with self.assertRaisesRegex(TypeError, "'DataFrame' object is not callable"):
            w.df()  # pyre-ignore[29]: Testing property call error
        # Check values are as expected:
        self.assertEqual(w.df, pd.DataFrame({"w": (0.5, 2, 1, 1)}))

    def test_BalanceWeightsDF_names(self) -> None:
        """Test that BalanceWeightsDF.names() returns correct weight column names."""
        self.assertEqual(w.names(), ["w"])

    def test_BalanceWeightsDF__sample(self) -> None:
        """Test that BalanceWeightsDF._sample references the correct source sample."""
        self.assertTrue(c._sample is s1)

    def test_BalanceWeightsDF_weights(self) -> None:
        """Test that BalanceWeightsDF._weights is None (weights don't have weights)."""
        self.assertTrue(w._weights is None)

    def test_BalanceWeightsDF_design_effect(self) -> None:
        s = Sample.from_frame(
            pd.DataFrame({"w": (1, 2, 4), "id": (1, 2, 3)}),
            id_column="id",
            weight_column="w",
        )
        self.assertTrue(s.weights().design_effect(), 7 / 3)

    def test_BalanceWeightsDF_trim(self) -> None:
        s = Sample.from_frame(
            pd.DataFrame({"w": np.random.uniform(0, 1, 10000), "id": range(0, 10000)}),
            id_column="id",
            weight_column="w",
        )
        s.weights().trim(percentile=(0, 0.11), keep_sum_of_weights=False)
        print(s.weights().df)
        print(max(s.weights().df.iloc[:, 0]))
        self.assertTrue(max(s.weights().df.iloc[:, 0]) < 0.9)

    def test_BalanceWeightsDF_summary(self) -> None:
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
        self.assertEqual(s1.covars().__class__, BalanceCovarsDF)

        # We get a different number of classes based on the number of linked items:
        the_dict = s1.covars()._BalanceDF_child_from_linked_samples()
        exp = [BalanceCovarsDF]
        self.assertEqual([v.__class__ for (k, v) in the_dict.items()], exp)

        the_dict = s3.covars()._BalanceDF_child_from_linked_samples()
        exp = 2 * [BalanceCovarsDF]
        self.assertEqual([v.__class__ for (k, v) in the_dict.items()], exp)

        the_dict = s3_null.covars()._BalanceDF_child_from_linked_samples()
        exp = 3 * [BalanceCovarsDF]
        self.assertEqual([v.__class__ for (k, v) in the_dict.items()], exp)

    def test__BalanceDF_child_from_linked_samples_weights_class(self) -> None:
        """Test that _BalanceDF_child_from_linked_samples works for BalanceWeightsDF."""
        # This also works for things other than BalanceCovarsDF:
        the_dict = s3_null.weights()._BalanceDF_child_from_linked_samples()
        exp = 3 * [BalanceWeightsDF]
        self.assertEqual([v.__class__ for (k, v) in the_dict.items()], exp)

    def test__BalanceDF_child_from_linked_samples_outcomes_with_none(self) -> None:
        """Test that _BalanceDF_child_from_linked_samples handles None outcomes correctly."""
        # Notice that with something like outcomes, we might get a None in return!
        the_dict = s3_null.outcomes()._BalanceDF_child_from_linked_samples()
        exp = [
            BalanceOutcomesDF,
            type(None),
            BalanceOutcomesDF,
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


class TestBalanceDF_to_download(BalanceTestCase):
    def test_BalanceDF_to_download(self) -> None:
        r = s1.covars().to_download()
        self.assertIsInstance(r, IPython.display.FileLink)


class TestBalanceDF_to_csv(BalanceTestCase):
    def test_BalanceDF_to_csv(self) -> None:
        with tempfile.NamedTemporaryFile() as tf:
            s1.weights().to_csv(path_or_buf=tf.name)
            r = tf.read()
            e = b"id,w\n1,0.5\n2,2.0\n3,1.0\n4,1.0\n"
            self.assertEqual(r, e)

    def test_BalanceDF_to_csv_first_default_argument_is_path(self) -> None:
        with tempfile.NamedTemporaryFile() as tf:
            s1.weights().to_csv(tf.name)
            r = tf.read()
            e = b"id,w\n1,0.5\n2,2.0\n3,1.0\n4,1.0\n"
            self.assertEqual(r, e)

    def test_BalanceDF_to_csv_output_with_no_path(self) -> None:
        with tempfile.NamedTemporaryFile():
            out = s1.weights().to_csv()
        self.assertEqual(out, "id,w\n1,0.5\n2,2.0\n3,1.0\n4,1.0\n")

    def test_BalanceDF_to_csv_output_with_path(self) -> None:
        with tempfile.NamedTemporaryFile() as tf:
            out = s1.weights().to_csv(path_or_buf=tf.name)
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

    def test_BalanceOutcomesDF___str__(self) -> None:
        # NOTE how the type is float even though the original input was integer.
        self.assertTrue(
            pd.DataFrame({"o": (7.0, 8.0, 9.0, 10.0)}).__str__() in o.__str__()
        )


class TestBalanceDF__repr__(BalanceTestCase):
    def test_BalanceWeightsDF___repr__(self) -> None:
        repr = w.__repr__()
        self.assertTrue("weights from" in repr)
        self.assertTrue(object.__repr__(s1) in repr)

    def test_BalanceCovarsDF___repr__(self) -> None:
        repr = c.__repr__()
        self.assertTrue("covars from" in repr)
        self.assertTrue(object.__repr__(s1) in repr)

    def test_BalanceOutcomesDF___repr__(self) -> None:
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
