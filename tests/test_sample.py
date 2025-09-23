# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for the Sample class in the balance library.

This module contains comprehensive tests for Sample class functionality including:
- Sample creation and validation
- Data manipulation and transformation
- Adjustment methods and model fitting
- Statistical computations and diagnostics
- Error handling and edge cases

The tests are organized into several test classes focusing on different aspects
of Sample functionality.
"""

# pyre-unsafe

from __future__ import absolute_import, division, print_function, unicode_literals

import tempfile

from copy import deepcopy

import balance.testutil
import IPython.display

import numpy as np
import pandas as pd

from balance.sample_class import Sample


# Test sample fixtures - shared across multiple test methods
# These represent common test scenarios for Sample functionality

# Sample with outcome column and mixed data types
s1 = Sample.from_frame(
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

# Sample without outcome columns for target testing
s2 = Sample.from_frame(
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

s3 = s1.set_target(s2)
s3_adjusted_null = s3.adjust(method="null")

# Sample with missing values and multiple outcome columns
s4 = Sample.from_frame(
    pd.DataFrame(
        {"a": (0, None, 2), "b": (0, None, 2), "c": ("a", "b", "c"), "id": (1, 2, 3)}
    ),
    outcome_columns=("b", "c"),
)


class TestSample(
    balance.testutil.BalanceTestCase,
):
    """Test class for basic Sample functionality and constructor behavior."""

    def test_constructor_not_implemented(self):
        """Test that Sample constructor raises NotImplementedError.

        The Sample class should not be instantiated directly without using
        the from_frame class method.
        """
        with self.assertRaises(NotImplementedError):
            s1 = Sample()
            print(s1)

    def test_Sample__str__(self):
        """Test string representation of Sample objects.

        Verifies that __str__ method correctly displays:
        - Number of observations and variables
        - Outcome columns information
        - Weight column information
        - Target setting status
        - Adjustment status
        """
        self.assertTrue("4 observations x 3 variables" in s1.__str__())
        self.assertTrue("outcome_columns: o" in s1.__str__())
        self.assertTrue("weight_column: w" in s1.__str__())

        self.assertTrue("outcome_columns: None" in s2.__str__())
        self.assertTrue("weight_column: w" in s2.__str__())

        self.assertTrue("Sample object with target set" in s3.__str__())
        self.assertTrue("target:" in s3.__str__())
        self.assertTrue("3 common variables" in s3.__str__())

        self.assertTrue(
            "Adjusted balance Sample object with target set using"
            in s3_adjusted_null.__str__()
        )

    def test_Sample__str__multiple_outcomes(self):
        """Test string representation with multiple outcome columns.

        Verifies that samples with multiple outcome columns display
        the outcome column names correctly in comma-separated format.
        """
        s1 = Sample.from_frame(
            pd.DataFrame(
                {"a": (1, 2, 3), "b": (4, 6, 8), "id": (1, 2, 3), "w": (0.5, 1, 2)}
            ),
            id_column="id",
            weight_column="w",
            outcome_columns=("a", "b"),
        )
        self.assertTrue("outcome_columns: a,b" in s1.__str__())

    def test_Sample_from_frame_id_column_detection(self):
        """Test automatic ID column detection and validation.

        Verifies that from_frame correctly:
        - Detects 'id' column automatically
        - Accepts explicit id_column parameter
        - Raises appropriate errors for missing or invalid columns
        - Handles null values in ID column appropriately
        """
        # Test automatic id column detection
        df = pd.DataFrame({"id": (1, 2), "a": (1, 2)})
        self.assertWarnsRegexp(
            "Guessed id column name id for the data", Sample.from_frame, df
        )
        # TODO: add tests for the two other warnings:
        # - self.assertWarnsRegexp("Casting id column to string", Sample.from_frame, df)
        # - self.assertWarnsRegexp("No weights passed, setting all weights to 1", Sample.from_frame, df)
        # Using the above would fail since the warnings are sent sequentially and using self.assertWarnsRegexp
        # only catches the first warning.
        self.assertEqual(
            Sample.from_frame(df).id_column, pd.Series((1, 2), name="id").astype(str)
        )

        # Test explicit id column specification
        df = pd.DataFrame({"b": (1, 2), "a": (1, 2)})
        self.assertEqual(
            Sample.from_frame(df, id_column="b").id_column,
            pd.Series((1, 2), name="b").astype(str),
        )

        # Test error when id column cannot be guessed
        with self.assertRaisesRegex(
            ValueError,
            "Cannot guess id column name for this DataFrame. Please provide a value in id_column",
        ):
            Sample.from_frame(df)

        # Test error when specified id column doesn't exist
        with self.assertRaisesRegex(
            ValueError,
            "Dataframe does not have column*",
        ):
            Sample.from_frame(df, id_column="c")

        # Test exception if values in id are null
        df = pd.DataFrame({"id": (1, None), "a": (1, 2)})
        with self.assertRaisesRegex(
            ValueError,
            "Null values are not allowed in the id_column",
        ):
            Sample.from_frame(df)

    def test_Sample_from_frame_id_uniqueness(self):
        """Test ID column uniqueness validation.

        Verifies that from_frame correctly validates ID uniqueness
        and respects the check_id_uniqueness parameter.
        """
        # Test default behavior - should require unique IDs
        df = pd.DataFrame({"id": (1, 2, 2)})
        with self.assertRaisesRegex(
            ValueError,
            "Values in the id_column must be unique",
        ):
            Sample.from_frame(df)

        # Test disabling uniqueness check
        df = pd.DataFrame({"id": (1, 2, 2)})
        self.assertEqual(
            Sample.from_frame(df, check_id_uniqueness=False).df.id,
            pd.Series(("1", "2", "2"), name="id"),
        )

    def test_Sample_from_frame_weight_column_handling(self):
        """Test weight column detection, validation, and default behavior.

        Verifies that from_frame correctly:
        - Detects weight columns automatically
        - Sets default weights when none provided
        - Validates weight values (numeric, non-negative, non-null)
        - Converts weight types appropriately
        """
        # Test automatic weight detection
        df = pd.DataFrame({"id": (1, 2), "weight": (1, 2)})
        self.assertWarnsRegexp("Guessing weight", Sample.from_frame, df)
        self.assertEqual(
            Sample.from_frame(df).weight_column, pd.Series((1.0, 2.0), name="weight")
        )

        # Test default weights when none provided
        df = pd.DataFrame({"id": (1, 2)})
        self.assertWarnsRegexp("No weights passed", Sample.from_frame, df)
        self.assertEqual(
            Sample.from_frame(df).weight_column, pd.Series((1.0, 1.0), name="weight")
        )

        # Test error for null weight values
        df = pd.DataFrame({"id": (1, 2, 3), "weight": (None, 3, 1.1)})
        with self.assertRaisesRegex(ValueError, "Null values are not allowed"):
            Sample.from_frame(df)

        df = pd.DataFrame({"id": (1, 2, 3), "weight": (None, None, None)})
        with self.assertRaisesRegex(ValueError, "Null values are not allowed"):
            Sample.from_frame(df)

    def test_Sample_from_frame_weight_validation(self):
        """Test weight value validation for numeric and non-negative constraints.

        Verifies that weights must be numeric and non-negative,
        with zero being acceptable.
        """
        # Test error for non-numeric weights
        df = pd.DataFrame({"id": (1, 2, 3), "weight": (1, "b", 2.1)})
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            Sample.from_frame(df)

        df = pd.DataFrame({"id": (1, 2, 3), "weight": (1, "5", 2.1)})
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            Sample.from_frame(df)

        df = pd.DataFrame({"id": (1, 2, 3), "weight": (1, 2.1, True)})
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            Sample.from_frame(df)

        # Test error for negative weight values
        df = pd.DataFrame({"id": (1, 2, 3), "weight": (1, -2, 2.1)})
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            Sample.from_frame(df)

        # Test that zero weights are acceptable
        df = pd.DataFrame({"id": (1, 2), "weight": (0, 2), "weight_column": "weight"})
        self.assertEqual(
            Sample.from_frame(df).weight_column, pd.Series((0.0, 2.0), name="weight")
        )

    def test_Sample_from_frame_type_conversion(self):
        """Test automatic type conversion for different numeric types.

        Verifies that from_frame correctly converts integer types
        to appropriate floating point types while preserving precision.
        """
        # Test int64 to float64 conversion
        df = pd.DataFrame({"id": (1, 2), "a": (1, 2)})
        self.assertEqual(df.a.dtype.type, np.int64)
        self.assertEqual(Sample.from_frame(df).df.a.dtype.type, np.float64)

        # Test int32 to float32 conversion
        df = pd.DataFrame({"id": (1, 2), "a": (1, 2)}, dtype=np.int32)
        self.assertEqual(df.a.dtype.type, np.int32)
        self.assertEqual(Sample.from_frame(df).df.a.dtype.type, np.float32)

        # Test int16 to float16 conversion
        df = pd.DataFrame({"id": (1, 2), "a": (1, 2)}, dtype=np.int16)
        self.assertEqual(df.a.dtype.type, np.int16)
        self.assertEqual(Sample.from_frame(df).df.a.dtype.type, np.float16)

        # Test int8 to float16 conversion (minimum float precision)
        df = pd.DataFrame({"id": (1, 2), "a": (1, 2)}, dtype=np.int8)
        self.assertEqual(df.a.dtype.type, np.int8)
        self.assertEqual(Sample.from_frame(df).df.a.dtype.type, np.float16)
        # TODO: add tests for other types of conversions

    def test_Sample_from_frame_deepcopy_behavior(self):
        """Test deepcopy parameter behavior.

        Verifies that use_deepcopy parameter controls whether
        the original DataFrame is modified during Sample creation.
        """
        # Test with use_deepcopy=False - original DataFrame should be modified
        df = pd.DataFrame({"id": (1, 2), "a": (1, 2)})
        self.assertEqual(df.id.dtype.type, np.int64)
        self.assertEqual(
            Sample.from_frame(df, use_deepcopy=False).df.id.dtype.type, np.object_
        )
        self.assertEqual(df.id.dtype.type, np.object_)

        # Test with use_deepcopy=True (default) - original DataFrame should be preserved
        df = pd.DataFrame({"id": (1, 2), "a": (1, 2)})
        self.assertEqual(df.id.dtype.type, np.int64)
        self.assertEqual(Sample.from_frame(df).df.id.dtype.type, np.object_)
        self.assertEqual(df.id.dtype.type, np.int64)

    def test_Sample_adjust(self):
        """Test Sample adjustment functionality.

        Verifies that adjust method correctly:
        - Accepts string method names
        - Accepts callable adjustment functions
        - Marks samples as adjusted after adjustment
        - Raises appropriate errors for invalid methods
        """
        from balance.weighting_methods.adjust_null import adjust_null

        # Test adjustment with string method name
        s3_adjusted_null = s1.set_target(s2).adjust(method="null")
        self.assertTrue(s3_adjusted_null.is_adjusted())

        # Test adjustment with callable function
        s3_adjusted_null = s1.set_target(s2).adjust(method=adjust_null)
        self.assertTrue(s3_adjusted_null.is_adjusted())

        # Test exception for invalid method
        with self.assertRaisesRegex(
            ValueError,
            "Method should be one of existing weighting methods",
        ):
            s1.set_target(s2).adjust(method=None)


class TestSample_base_and_adjust_methods(
    balance.testutil.BalanceTestCase,
):
    """Test class for Sample data access methods and property behavior."""

    def test_Sample_df(self):
        """Test DataFrame property access and type conversion behavior.

        Verifies that:
        - df property returns correctly formatted DataFrame
        - Integer values are converted to floats
        - ID values are converted to strings
        - Property decorator works correctly
        - Calling df as function raises appropriate error
        """
        # Test s1 DataFrame structure and type conversion
        e = pd.DataFrame(
            {
                "a": (1.0, 2.0, 3.0, 1.0),
                "b": (-42.0, 8.0, 2.0, -42.0),
                "o": (7.0, 8.0, 9.0, 10.0),
                "c": ("x", "y", "z", "v"),
                "id": ("1", "2", "3", "4"),
                "w": (0.5, 2, 1, 1),
            },
            columns=("id", "a", "b", "c", "o", "w"),
        )
        self.assertEqual(s1.df, e)

        # Test property decorator functionality
        self.assertTrue(isinstance(Sample.df, property))
        self.assertEqual(Sample.df.fget(s1), s1.df)

        # Test that df cannot be called as function
        with self.assertRaisesRegex(TypeError, "'DataFrame' object is not callable"):
            s1.df()

        # Test s2 DataFrame structure and type conversion
        e = pd.DataFrame(
            {
                "a": (1.0, 2.0, 3.0),
                "b": (4.0, 6.0, 8.0),
                "id": ("1", "2", "3"),
                "w": (0.5, 1, 2),
                "c": ("x", "y", "z"),
            },
            columns=("id", "a", "b", "c", "w"),
        )
        self.assertEqual(s2.df, e)

    # TODO: consider removing this test, since it's already tested in test_balancedf.py
    def test_Sample_outcomes(self):
        """Test outcome columns extraction functionality.

        Verifies that outcomes() method correctly extracts
        only the outcome columns with proper type conversion.
        """
        e = pd.DataFrame(
            {
                "o": (7.0, 8.0, 9.0, 10.0),
            },
            columns=["o"],
        )
        self.assertEqual(s1.outcomes().df, e)

    def test_Sample_weights(self):
        """Test weights extraction functionality.

        Verifies that weights() method correctly extracts
        only the weight column as a DataFrame.
        """
        e = pd.DataFrame(
            {
                "w": (0.5, 2, 1, 1),
            },
            columns=["w"],
        )
        self.assertEqual(s1.weights().df, e)

    # TODO: consider removing this test, since it's already tested in test_balancedf.py
    def test_Sample_covars(self):
        """Test covariate columns extraction functionality.

        Verifies that covars() method correctly extracts
        all non-special columns (excluding id, weight, outcome columns)
        with proper type conversion.
        """
        e = pd.DataFrame(
            {
                "a": (1.0, 2.0, 3.0, 1.0),
                "b": (-42.0, 8.0, 2.0, -42.0),
                "c": ("x", "y", "z", "v"),
            }
        )
        self.assertEqual(s1.covars().df, e)

    def _create_test_sample_with_target(self, source_size=1000, target_size=10000):
        """Helper method to create test samples for model testing.

        Args:
            source_size: Number of rows for source sample
            target_size: Number of rows for target sample

        Returns:
            Tuple of (source_sample, target_sample)
        """
        # Create source sample
        d = pd.DataFrame(np.random.rand(source_size, 10))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})
        source = Sample.from_frame(d)

        # Create target sample
        d = pd.DataFrame(np.random.rand(target_size, 10))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})
        target = Sample.from_frame(d)

        return source, target

    def test_Sample_model_null_adjustment(self):
        """Test model information for null adjustment method.

        Verifies that null adjustment correctly reports method name.
        """
        np.random.seed(112358)
        s, t = self._create_test_sample_with_target()

        a = s.adjust(t, max_de=None, method="null")
        m = a.model()
        self.assertEqual(m["method"], "null_adjustment")

    def test_Sample_model_ipw_adjustment(self):
        """Test model information for IPW adjustment method.

        Verifies that IPW adjustment correctly reports method name
        and includes expected model structure (perf, fit, coefs).
        """
        np.random.seed(112358)
        s, t = self._create_test_sample_with_target()

        a = s.adjust(t, max_de=None)
        m = a.model()
        self.assertEqual(m["method"], "ipw")

        # Test structure of IPW output
        self.assertTrue("perf" in m.keys())
        self.assertTrue("fit" in m.keys())
        self.assertTrue("coefs" in m["perf"].keys())

    def test_Sample_model_matrix(self):
        """Test model matrix generation for samples.

        Verifies that model_matrix method correctly:
        - Handles missing values by creating indicator variables
        - Converts categorical variables to dummy variables
        - Preserves numeric variables
        - Returns properly formatted model matrix

        Note: Main tests for model_matrix are in test_util.py
        """
        s = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": (0, 1, 2),
                    "b": (0, None, 2),
                    "c": ("a", "b", "a"),
                    "id": (1, 2, 3),
                }
            ),
            id_column="id",
        )
        e = pd.DataFrame(
            {
                "a": (0.0, 1.0, 2.0),
                "b": (0.0, 0.0, 2.0),
                "_is_na_b[T.True]": (0.0, 1.0, 0.0),
                "c[a]": (1.0, 0.0, 1.0),
                "c[b]": (0.0, 1.0, 0.0),
            }
        )
        r = s.model_matrix()
        self.assertEqual(r, e, lazy=True)

    def test_Sample_set_weights(self):
        s = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": (1, 2, 3, 1),
                    "id": (1, 2, 3, 4),
                    "w": (0.5, 2, 1, 1),
                }
            ),
            id_column="id",
            weight_column="w",
        )
        s.set_weights(pd.Series([1, 2, 3, 4]))
        self.assertEqual(s.weight_column, pd.Series([1.0, 2.0, 3.0, 4.0], name="w"))
        s.set_weights(pd.Series([1, 2, 3, 4], index=(1, 2, 5, 6)))
        self.assertEqual(
            s.weight_column, pd.Series([np.nan, 1.0, 2.0, np.nan], name="w")
        )
        # test warning
        self.assertWarnsRegexp(
            """Note that not all Sample units will be assigned weights""",
            Sample.set_weights,
            s,
            pd.Series([1, 2, 3, 4], index=(1, 2, 5, 6)),
        )
        # no warning
        self.assertNotWarns(
            Sample.set_weights,
            s,
            pd.Series([1, 2, 3, 4], index=(0, 1, 2, 3)),
        )

    def test_Sample_set_unadjusted(self):
        s5 = s1.set_unadjusted(s2)
        self.assertTrue(s5._links["unadjusted"] is s2)
        # test exceptions when there is no a second sample
        with self.assertRaisesRegex(
            TypeError,
            "set_unadjusted must be called with second_sample argument of type Sample",
        ):
            s1.set_unadjusted("Not a Sample object")

    def test_Sample_is_adjusted(self):
        self.assertFalse(s1.is_adjusted())
        self.assertFalse(s3.is_adjusted())
        self.assertTrue(s3_adjusted_null.is_adjusted())

    def test_Sample_set_target(self):
        s5 = s1.set_target(s2)
        self.assertTrue(s5._links["target"] is s2)
        # test exceptions when the provided object is not a second sample
        with self.assertRaisesRegex(
            ValueError,
            "A target, a Sample object, must be specified",
        ):
            s1.set_target("Not a Sample object")

    def test_Sample_has_target(self):
        self.assertFalse(s1.has_target())
        self.assertTrue(s1.set_target(s2).has_target())


class TestSample_metrics_methods(
    balance.testutil.BalanceTestCase,
):
    def _assert_dict_almost_equal(self, actual, expected, places=2):
        """Helper method to compare nested dictionaries with floating point tolerance.

        This addresses floating point precision differences introduced by Python 3.12's
        new summation algorithm.
        """
        self.assertEqual(
            set(actual.keys()), set(expected.keys()), "Dictionary keys don't match"
        )

        for key in expected.keys():
            if isinstance(expected[key], dict):
                self.assertIsInstance(
                    actual[key], dict, f"Value for key '{key}' should be a dict"
                )
                self._assert_dict_almost_equal(actual[key], expected[key], places)
            else:
                self.assertAlmostEqual(
                    actual[key],
                    expected[key],
                    places=places,
                    msg=f"Values for key '{key}' don't match within {places} decimal places",
                )

    def test_Sample_covar_means(self):
        s3_null = s1.adjust(s2, method="null")
        e = pd.DataFrame(
            {
                "a": [(0.5 * 1 + 2 * 2 + 3 * 1 + 1 * 1) / (0.5 + 2 + 1 + 1)],
                "b": [(-42 * 0.5 + 8 * 2 + 2 * 1 + -42 * 1) / (0.5 + 2 + 1 + 1)],
                "c[x]": [(1 * 0.5) / (0.5 + 2 + 1 + 1)],
                "c[y]": [(1 * 2) / (0.5 + 2 + 1 + 1)],
                "c[z]": [(1 * 1) / (0.5 + 2 + 1 + 1)],
                "c[v]": [(1 * 1) / (0.5 + 2 + 1 + 1)],
            }
        ).transpose()
        e = pd.concat((e,) * 2, axis=1, sort=True)
        e = pd.concat(
            (
                e,
                pd.DataFrame(
                    {
                        "a": [(1 * 0.5 + 2 * 1 + 3 * 2) / (0.5 + 1 + 2)],
                        "b": [(4 * 0.5 + 6 * 1 + 8 * 2) / (0.5 + 1 + 2)],
                        "c[x]": [(1 * 0.5) / (0.5 + 1 + 2)],
                        "c[y]": [(1 * 1) / (0.5 + 1 + 2)],
                        "c[z]": [(1 * 2) / (0.5 + 1 + 2)],
                        "c[v]": np.nan,
                    }
                ).transpose(),
            ),
            axis=1,
            sort=True,
        )
        e.columns = pd.Series(("unadjusted", "adjusted", "target"), name="source")
        self.assertEqual(s3_null.covar_means(), e)

        # test exceptions when there is no adjusted
        with self.assertRaisesRegex(
            ValueError,
            "This is not an adjusted Sample. Use sample.adjust to adjust the sample to target",
        ):
            s1.covar_means()

    def test_Sample_design_effect(self):
        self.assertEqual(s1.design_effect().round(3), 1.235)
        self.assertEqual(s4.design_effect(), 1.0)

    def test_Sample_design_effect_prop(self):
        s3_null = s1.adjust(s2, method="null")
        self.assertEqual(s3_null.design_effect_prop(), 0.0)

        # test exceptions when there is no adjusted
        with self.assertRaisesRegex(
            ValueError,
            "This is not an adjusted Sample. Use sample.adjust to adjust the sample to target",
        ):
            s1.design_effect_prop()

    def test_Sample_outcome_sd_prop(self):
        s3_null = s1.adjust(s2, method="null")
        self.assertEqual(s3_null.outcome_sd_prop(), pd.Series((0.0), index=["o"]))
        # test with two outcomes
        s1_two_outcomes = Sample.from_frame(
            pd.DataFrame(
                {
                    "o1": (7, 8, 9, 10),
                    "o2": (7, 8, 9, 11),
                    "c": ("x", "y", "z", "y"),
                    "id": (1, 2, 3, 4),
                    "w": (0.5, 2, 1, 1),
                },
            ),
            id_column="id",
            weight_column="w",
            outcome_columns=["o1", "o2"],
        )
        s3_null = s1_two_outcomes.adjust(s2, method="null")
        self.assertEqual(
            s3_null.outcome_sd_prop(), pd.Series((0.0, 0.0), index=["o1", "o2"])
        )

        # test exceptions when there is no adjusted
        with self.assertRaisesRegex(
            ValueError,
            "This Sample does not have outcome columns specified",
        ):
            s2.adjust(s2, method="null").outcome_sd_prop()

    def _create_samples_for_outcome_variance_tests(self):
        """Helper method to create samples for outcome variance ratio testing.

        Returns:
            Tuple of (target_sample, source_sample_data) for variance tests
        """
        np.random.seed(112358)

        # Create target sample
        d = pd.DataFrame(np.random.rand(1000, 10))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})
        t = Sample.from_frame(d)

        # Create source sample data
        d = pd.DataFrame(np.random.rand(1000, 11))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghijk"[i] for i in range(0, 11)})
        d["b"] = np.sqrt(d["b"])

        return t, d

    def test_outcome_variance_ratio_calculation(self):
        """Test outcome variance ratio calculation accuracy.

        Verifies that outcome_variance_ratio method correctly computes
        the ratio of adjusted to unadjusted outcome variances using
        weighted variance calculations.
        """
        from balance.stats_and_plots.weighted_stats import weighted_var

        t, d = self._create_samples_for_outcome_variance_tests()

        a_with_outcome = Sample.from_frame(d, outcome_columns=["k"])
        a_with_outcome_adjusted = a_with_outcome.adjust(t, max_de=1.5)

        # Test calculation matches manual weighted variance ratio calculation
        expected_ratio = (
            weighted_var(
                a_with_outcome_adjusted.outcomes().df,
                a_with_outcome_adjusted.weights().df["weight"],
            )
            / weighted_var(
                a_with_outcome_adjusted._links["unadjusted"].outcomes().df,
                a_with_outcome_adjusted._links["unadjusted"].weights().df["weight"],
            )
        ).iloc[0]

        actual_ratio = a_with_outcome_adjusted.outcome_variance_ratio().iloc[0]
        self.assertEqual(round(actual_ratio, 5), round(expected_ratio, 5))

    def test_outcome_variance_ratio_value(self):
        """Test expected outcome variance ratio value for test data.

        Verifies that the outcome variance ratio produces expected
        numerical results for the test dataset.
        """
        t, d = self._create_samples_for_outcome_variance_tests()

        a_with_outcome = Sample.from_frame(d, outcome_columns=["k"])
        a_with_outcome_adjusted = a_with_outcome.adjust(t, max_de=1.5)

        # Test expected variance ratio value
        self.assertEqual(
            round(a_with_outcome_adjusted.outcome_variance_ratio().iloc[0], 2), 0.98
        )

    def test_outcome_variance_ratio_null_adjustment(self):
        """Test outcome variance ratio with null adjustment.

        Verifies that null adjustment produces variance ratio of 1.0
        for all outcomes, since no actual adjustment is applied.
        """
        t, d = self._create_samples_for_outcome_variance_tests()

        # Test with multiple outcomes and null adjustment
        a_with_outcome = Sample.from_frame(d, outcome_columns=["j", "k"])
        a_with_outcome_adjusted = a_with_outcome.adjust(t, method="null")

        # Null adjustment should produce variance ratio of 1.0
        self.assertEqual(
            a_with_outcome_adjusted.outcome_variance_ratio(),
            pd.Series([1.0, 1.0], index=["j", "k"]),
        )

    def test_Sample_weights_summary(self):
        self.assertEqual(
            s1.weights().summary().round(2).to_dict(),
            {
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
            },
        )

    def test_Sample_summary(self):
        s1_summ = s1.summary()
        self.assertTrue("Model performance" not in s1_summ)
        self.assertTrue("Covar ASMD" not in s1_summ)

        s3_summ = s3.summary()
        self.assertTrue("Model performance" not in s1_summ)
        self.assertTrue("Covar ASMD (6 variables)" in s3_summ)
        self.assertTrue("design effect" not in s3_summ)

        s3_set_unadjusted = s3.set_unadjusted(s1)
        s3_summ = s3_set_unadjusted.summary()
        self.assertTrue("Covar ASMD reduction: 0.0%" in s3_summ)
        self.assertTrue("Covar ASMD (6 variables)" in s3_summ)
        self.assertTrue("->" in s3_summ)
        self.assertTrue("design effect" in s3_summ)

        s3_summ = s3_adjusted_null.summary()
        self.assertTrue("Covar ASMD reduction: 0.0%" in s3_summ)
        self.assertTrue("design effect" in s3_summ)

    def test_Sample_invalid_outcomes(self):
        with self.assertRaisesRegex(
            ValueError,
            r"outcome columns \['o'\] not in df columns \['a', 'id', 'weight'\]",
        ):
            Sample.from_frame(
                pd.DataFrame({"a": (1, 2, 3, 1), "id": (1, 2, 3, 4)}),
                outcome_columns="o",
            )

    def _create_samples_for_diagnostics_tests(self):
        """Helper method to create samples for diagnostics testing.

        Returns:
            Tuple of (source_sample, target_sample) for diagnostics tests
        """
        np.random.seed(112358)

        # Create source sample
        d = pd.DataFrame(np.random.rand(1000, 10))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})
        s = Sample.from_frame(d)

        # Create target sample
        d = pd.DataFrame(np.random.rand(1000, 10))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})
        t = Sample.from_frame(d)

        return s, t

    def test_Sample_diagnostics_ipw_method(self):
        """Test diagnostics output structure and content for IPW adjustment.

        Verifies that diagnostics method produces expected structure,
        columns, and metric counts for IPW (default) adjustment method.
        """
        s, t = self._create_samples_for_diagnostics_tests()

        a = s.adjust(t)
        a_diagnostics = a.diagnostics()

        # Test basic structure
        self.assertEqual(a_diagnostics.shape, (203, 3))
        self.assertEqual(a_diagnostics.columns.to_list(), ["metric", "val", "var"])

        # Test adjustment method is recorded correctly
        self.assertEqual(
            a_diagnostics[a_diagnostics["metric"] == "adjustment_method"]["var"].values,
            np.array(["ipw"]),
        )

        # Test metric counts for IPW method
        output = a_diagnostics.groupby("metric").size().to_dict()
        expected = {
            "adjustment_failure": 1,
            "adjustment_method": 1,
            "covar_asmd_adjusted": 11,
            "covar_asmd_improvement": 11,
            "covar_asmd_unadjusted": 11,
            "covar_main_asmd_adjusted": 11,
            "covar_main_asmd_improvement": 11,
            "covar_main_asmd_unadjusted": 11,
            "ipw_model_glance": 2,
            "ipw_multi_class": 1,
            "ipw_penalty": 1,
            "ipw_solver": 1,
            "model_coef": 92,
            "model_glance": 10,
            "size": 4,
            "weights_diagnostics": 24,
        }
        self.assertEqual(output, expected)

    def test_Sample_diagnostics_cbps_method(self):
        """Test diagnostics output structure and content for CBPS adjustment.

        Verifies that diagnostics method produces expected structure,
        columns, and metric counts for CBPS adjustment method.
        """
        s, t = self._create_samples_for_diagnostics_tests()

        b = s.adjust(t, method="cbps")
        b_diagnostics = b.diagnostics()

        # Test basic structure
        self.assertEqual(b_diagnostics.shape, (196, 3))
        self.assertEqual(b_diagnostics.columns.to_list(), ["metric", "val", "var"])

        # Test adjustment method is recorded correctly
        self.assertEqual(
            b_diagnostics[b_diagnostics["metric"] == "adjustment_method"]["var"].values,
            np.array(["cbps"]),
        )

        # Test metric counts for CBPS method
        output = b_diagnostics.groupby("metric").size().to_dict()
        expected = {
            "adjustment_failure": 1,
            "balance_optimize_result": 2,
            "gmm_optimize_result_bal_init": 2,
            "gmm_optimize_result_glm_init": 2,
            "rescale_initial_result": 2,
            "beta_optimal": 92,
            "covar_asmd_adjusted": 11,
            "covar_asmd_improvement": 11,
            "covar_asmd_unadjusted": 11,
            "covar_main_asmd_adjusted": 11,
            "covar_main_asmd_improvement": 11,
            "covar_main_asmd_unadjusted": 11,
            "adjustment_method": 1,
            "size": 4,
            "weights_diagnostics": 24,
        }
        self.assertEqual(output, expected)

    def test_Sample_diagnostics_null_method(self):
        """Test diagnostics output structure and content for null adjustment.

        Verifies that diagnostics method produces expected structure,
        columns, and method recording for null adjustment.
        """
        s, t = self._create_samples_for_diagnostics_tests()

        c = s.adjust(t, method="null")
        c_diagnostics = c.diagnostics()

        # Test basic structure
        self.assertEqual(c_diagnostics.shape, (96, 3))
        self.assertEqual(c_diagnostics.columns.to_list(), ["metric", "val", "var"])

        # Test adjustment method is recorded correctly
        self.assertEqual(
            c_diagnostics[c_diagnostics["metric"] == "adjustment_method"]["var"].values,
            np.array(["null_adjustment"]),
        )

    def _create_adjusted_sample_for_filtering_tests(self):
        """Helper method to create adjusted sample for filtering tests.

        Returns:
            Tuple of (adjusted_sample, target_sample) for use in filtering tests
        """
        np.random.seed(112358)

        # Create source sample with transformed 'b' column
        d = pd.DataFrame(np.random.rand(1000, 10))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})
        d["b"] = np.sqrt(d["b"])
        s = Sample.from_frame(d)

        # Create target sample
        d = pd.DataFrame(np.random.rand(1000, 10))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})
        t = Sample.from_frame(d)

        return s.adjust(t, max_de=1.5), t

    def test_Sample_keep_only_some_rows_columns_identity(self):
        """Test that keep_only_some_rows_columns returns same object when no filtering applied.

        Verifies that when both rows_to_keep and columns_to_keep are None,
        the method returns the same object reference.
        """
        a, _ = self._create_adjusted_sample_for_filtering_tests()

        # Should return the same object when no filtering is applied
        self.assertTrue(
            a is a.keep_only_some_rows_columns(rows_to_keep=None, columns_to_keep=None)
        )

    def test_Sample_keep_only_some_rows_columns_column_filtering(self):
        """Test column filtering functionality and its impact on ASMD calculations.

        Verifies that column filtering:
        - Correctly reduces the set of covariates
        - Updates ASMD calculations appropriately
        - Maintains proper diagnostics structure
        """
        a, _ = self._create_adjusted_sample_for_filtering_tests()

        # Filter to keep only columns 'b' and 'c'
        a2 = a.keep_only_some_rows_columns(
            rows_to_keep=None, columns_to_keep=["b", "c"]
        )

        # Test ASMD calculations before and after filtering
        output_orig = a.covars().asmd().round(2).to_dict()
        output_new = a2.covars().asmd().round(2).to_dict()

        expected_orig = {
            "j": {"self": 0.01, "unadjusted": 0.03, "unadjusted - self": 0.02},
            "i": {"self": 0.01, "unadjusted": 0.0, "unadjusted - self": -0.01},
            "h": {"self": 0.02, "unadjusted": 0.09, "unadjusted - self": 0.06},
            "g": {"self": 0.01, "unadjusted": 0.0, "unadjusted - self": -0.01},
            "f": {"self": 0.01, "unadjusted": 0.03, "unadjusted - self": 0.01},
            "e": {"self": 0.01, "unadjusted": 0.0, "unadjusted - self": -0.01},
            "d": {"self": 0.03, "unadjusted": 0.12, "unadjusted - self": 0.09},
            "c": {"self": 0.04, "unadjusted": 0.05, "unadjusted - self": 0.02},
            "b": {"self": 0.18, "unadjusted": 0.55, "unadjusted - self": 0.37},
            "a": {"self": 0.01, "unadjusted": 0.0, "unadjusted - self": -0.0},
            "mean(asmd)": {"self": 0.03, "unadjusted": 0.09, "unadjusted - self": 0.06},
        }
        expected_new = {
            "c": {"self": 0.04, "unadjusted": 0.05, "unadjusted - self": 0.02},
            "b": {"self": 0.18, "unadjusted": 0.55, "unadjusted - self": 0.37},
            "mean(asmd)": {"self": 0.11, "unadjusted": 0.3, "unadjusted - self": 0.20},
        }

        # Note: Using custom comparison to handle floating point precision differences in Python 3.12
        self._assert_dict_almost_equal(output_orig, expected_orig, places=1)
        self._assert_dict_almost_equal(output_new, expected_new, places=1)

    def test_Sample_keep_only_some_rows_columns_diagnostics_impact(self):
        """Test the impact of column filtering on diagnostics.

        Verifies that column filtering appropriately updates:
        - ASMD counts and values
        - Covariate counts
        - Weight diagnostics
        """
        a, _ = self._create_adjusted_sample_for_filtering_tests()
        a2 = a.keep_only_some_rows_columns(
            rows_to_keep=None, columns_to_keep=["b", "c"]
        )

        # Get diagnostics for comparison
        a_diag = a.diagnostics()
        a2_diag = a2.diagnostics()
        a_diag_tbl = a_diag.groupby("metric").size().to_dict()
        a2_diag_tbl = a2_diag.groupby("metric").size().to_dict()

        # Test weight normalization
        ss = a_diag.eval("(metric == 'weights_diagnostics') & (var == 'describe_mean')")
        self.assertEqual(round(float(a_diag[ss].val.iloc[0]), 4), 1.000)

        # Test ASMD count changes due to column filtering
        self.assertEqual(a_diag_tbl["covar_main_asmd_adjusted"], 11)
        self.assertEqual(a2_diag_tbl["covar_main_asmd_adjusted"], 3)

        # Test covariate count changes
        ss_condition = "(metric == 'size') & (var == 'sample_covars')"
        ss = a_diag.eval(ss_condition)
        ss2 = a2_diag.eval(ss_condition)
        self.assertEqual(int(a_diag[ss].val.iloc[0]), 10)
        self.assertEqual(int(a2_diag[ss2].val.iloc[0]), 2)

        # Test mean ASMD changes
        # Note: Using assertAlmostEqual to handle floating point precision differences in Python 3.12
        ss_condition = "(metric == 'covar_main_asmd_adjusted') & (var == 'mean(asmd)')"
        ss = a_diag.eval(ss_condition)
        ss2 = a2_diag.eval(ss_condition)
        self.assertAlmostEqual(
            round(float(a_diag[ss].val.iloc[0]), 4), 0.0329, places=3
        )
        self.assertAlmostEqual(
            round(float(a2_diag[ss2].val.iloc[0]), 3), 0.109, places=3
        )

    def test_Sample_keep_only_some_rows_columns_row_filtering(self):
        """Test row filtering functionality and its impact on sample sizes.

        Verifies that row filtering:
        - Correctly filters rows based on conditions
        - Maintains weight-data consistency
        - Updates sample size statistics appropriately
        """
        a, _ = self._create_adjusted_sample_for_filtering_tests()

        # Filter rows and columns
        a3 = a.keep_only_some_rows_columns(
            rows_to_keep="a>0.5", columns_to_keep=["b", "c"]
        )

        # Test weight-data consistency
        self.assertEqual(a3.df.shape[0], a3.weights().df.shape[0])

        # Test ASMD calculations with row filtering
        output_new = a3.covars().asmd().round(2).to_dict()
        expected_new = {
            "c": {"self": 0.05, "unadjusted": 0.07, "unadjusted - self": 0.02},
            "b": {"self": 0.25, "unadjusted": 0.61, "unadjusted - self": 0.36},
            "mean(asmd)": {"self": 0.15, "unadjusted": 0.34, "unadjusted - self": 0.19},
        }
        self.assertEqual(output_new, expected_new)

    def test_Sample_keep_only_some_rows_columns_sample_size_changes(self):
        """Test that row filtering correctly updates sample size statistics.

        Verifies that diagnostics correctly reflect changes in:
        - Sample observation counts
        - Target observation counts
        - Weight statistics
        """
        a, _ = self._create_adjusted_sample_for_filtering_tests()
        a2 = a.keep_only_some_rows_columns(
            rows_to_keep=None, columns_to_keep=["b", "c"]
        )
        a3 = a.keep_only_some_rows_columns(
            rows_to_keep="a>0.5", columns_to_keep=["b", "c"]
        )

        # Get diagnostics
        a_diag = a.diagnostics()
        a2_diag = a2.diagnostics()
        a3_diag = a3.diagnostics()
        a2_diag_tbl = a2_diag.groupby("metric").size().to_dict()
        a3_diag_tbl = a3_diag.groupby("metric").size().to_dict()

        # Test that column-only filtering preserves diagnostics structure
        self.assertEqual(a2_diag_tbl, a3_diag_tbl)

        # Test sample observation count changes
        ss_condition = "(metric == 'size') & (var == 'sample_obs')"
        self.assertEqual(int(a_diag[a_diag.eval(ss_condition)].val.iloc[0]), 1000)
        self.assertEqual(int(a2_diag[a2_diag.eval(ss_condition)].val.iloc[0]), 1000)
        self.assertEqual(int(a3_diag[a3_diag.eval(ss_condition)].val.iloc[0]), 508)

        # Test target observation count changes
        ss_condition = "(metric == 'size') & (var == 'target_obs')"
        self.assertEqual(int(a_diag[a_diag.eval(ss_condition)].val.iloc[0]), 1000)
        self.assertEqual(int(a2_diag[a2_diag.eval(ss_condition)].val.iloc[0]), 1000)
        self.assertEqual(int(a3_diag[a3_diag.eval(ss_condition)].val.iloc[0]), 516)

        # Test weight count changes
        ss = a_diag.eval(
            "(metric == 'weights_diagnostics') & (var == 'describe_count')"
        )
        self.assertEqual(int(a_diag[ss].val.iloc[0]), 1000)
        ss = a3_diag.eval(
            "(metric == 'weights_diagnostics') & (var == 'describe_count')"
        )
        self.assertEqual(int(a3_diag[ss].val.iloc[0]), 508)

        # Test design effect changes
        # Note: Using assertAlmostEqual to handle floating point precision differences in Python 3.12
        ss = a_diag.eval("(metric == 'weights_diagnostics') & (var == 'design_effect')")
        self.assertAlmostEqual(round(float(a_diag[ss].val.iloc[0]), 3), 1.468, places=2)
        ss = a3_diag.eval(
            "(metric == 'weights_diagnostics') & (var == 'design_effect')"
        )
        # Increased tolerance to handle Python 3.12's new summation algorithm
        self.assertAlmostEqual(
            round(float(a3_diag[ss].val.iloc[0]), 4), 1.4325, places=2
        )

    def test_Sample_keep_only_some_rows_columns_with_outcomes(self):
        """Test filtering functionality when outcome columns are present.

        Verifies that filtering works correctly with samples that have
        outcome columns and maintains proper filtering behavior.
        """
        _, t = self._create_adjusted_sample_for_filtering_tests()

        # Create sample with outcome columns
        np.random.seed(112358)
        d = pd.DataFrame(np.random.rand(1000, 11))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghijk"[i] for i in range(0, 11)})
        d["b"] = np.sqrt(d["b"])
        a_with_outcome = Sample.from_frame(d, outcome_columns=["k"])
        a_with_outcome_adjusted = a_with_outcome.adjust(t, max_de=1.5)

        # Test filtering using outcome variable - should affect sample but not target
        filtered_by_outcome = a_with_outcome_adjusted.keep_only_some_rows_columns(
            rows_to_keep="k>0.5"
        )
        self.assertEqual(filtered_by_outcome.df.shape, (481, 13))

        # Test combined row and column filtering with outcomes
        filtered_combined = a_with_outcome_adjusted.keep_only_some_rows_columns(
            rows_to_keep="b>0.5", columns_to_keep=["b", "c"]
        )

        # Test that outcome calculations work correctly after filtering
        self.assertEqual(
            filtered_combined.outcomes().mean().round(3).to_dict(),
            {"k": {"self": 0.492, "unadjusted": 0.494}},
        )

    def test_Sample_keep_only_some_rows_columns_column_warnings(self):
        """Test warning behavior when requested columns don't exist.

        Verifies that appropriate warnings are issued when some
        requested columns are not present in the sample.
        """
        # Test warning when some columns don't exist
        self.assertWarnsRegexp(
            "Note that not all columns_to_keep are in Sample",
            s1.keep_only_some_rows_columns,
            columns_to_keep=["g", "a"],
        )

        # Test that existing columns are still kept
        filtered_sample = s1.keep_only_some_rows_columns(columns_to_keep=["g", "a"])
        self.assertEqual(filtered_sample._df.columns.tolist(), ["a"])


class TestSample_to_download(balance.testutil.BalanceTestCase):
    def test_Sample_to_download(self):
        r = s1.to_download()
        self.assertIsInstance(r, IPython.display.FileLink)

    def test_Sample_to_csv(self):
        with tempfile.NamedTemporaryFile() as tf:
            s1.to_csv(path_or_buf=tf.name)
            r = tf.read()
            e = (
                b"id,a,b,c,o,w\n1,1,-42,x,7,0.5\n"
                b"2,2,8,y,8,2.0\n3,3,2,z,9,1.0\n4,1,-42,v,10,1.0\n"
            )
            self.assertTrue(r, e)


class TestSamplePrivateAPI(balance.testutil.BalanceTestCase):
    def test__links(self):
        self.assertTrue(len(s1._links.keys()) == 0)

        self.assertTrue(s3._links["target"] is s2)
        self.assertTrue(s3.has_target())

        self.assertTrue(s3_adjusted_null._links["target"] is s2)
        self.assertTrue(s3_adjusted_null._links["unadjusted"] is s3)
        self.assertTrue(s3_adjusted_null.has_target())

    def test__special_columns_names(self):
        self.assertEqual(
            sorted(s4._special_columns_names()), ["b", "c", "id", "weight"]
        )

    # NOTE how integers were changed into floats.
    def test__special_columns(self):
        # NOTE how integers in weight were changed into floats.
        self.assertEqual(
            s4._special_columns(),
            pd.DataFrame(
                {
                    "id": ("1", "2", "3"),
                    # Weights were filled automatically to be floats of 1.0:
                    "weight": (1.0, 1.0, 1.0),
                    "b": (0.0, None, 2.0),
                    "c": ("a", "b", "c"),
                }
            ),
        )

    def test__covar_columns_names(self):
        self.assertEqual(sorted(s1._covar_columns_names()), ["a", "b", "c"])

    def test__covar_columns(self):
        # NOTE how integers were changed into floats.
        self.assertEqual(
            s1._covar_columns(),
            pd.DataFrame(
                {
                    "a": (1.0, 2.0, 3.0, 1.0),
                    "b": (-42.0, 8.0, 2.0, -42.0),
                    "c": ("x", "y", "z", "v"),
                }
            ),
        )

    def test_Sample__check_if_adjusted(self):
        with self.assertRaisesRegex(
            ValueError,
            "This is not an adjusted Sample. Use sample.adjust to adjust the sample to target",
        ):
            s1._check_if_adjusted()
        with self.assertRaisesRegex(
            ValueError,
            "This is not an adjusted Sample. Use sample.adjust to adjust the sample to target",
        ):
            s3._check_if_adjusted()
        self.assertTrue(
            s3_adjusted_null._check_if_adjusted() is None
        )  # Does not raise an error

    def test_Sample__no_target_error(self):
        # test exception when the is no target
        with self.assertRaisesRegex(
            ValueError,
            "This Sample does not have a target set. Use sample.set_target to add target",
        ):
            s1._no_target_error()
        s3._no_target_error()  # Should not raise an error

    def test_Sample__check_outcomes_exists(self):
        with self.assertRaisesRegex(
            ValueError,
            "This Sample does not have outcome columns specified",
        ):
            s2._check_outcomes_exists()
        self.assertTrue(s1._check_outcomes_exists() is None)  # Does not raise an error


class TestSample_NA_behavior(balance.testutil.BalanceTestCase):
    def test_can_handle_various_NAs(self):
        # Testing if we can handle NA values from pandas

        def get_sample_to_adjust(df, standardize_types=True):
            s1 = Sample.from_frame(df, standardize_types=standardize_types)
            s2 = deepcopy(s1)
            s2.set_weights(np.ones(100))
            return s1.set_target(s2)

        np.random.seed(123)
        df = pd.DataFrame(
            {
                "a": np.random.uniform(size=100),
                "c": np.random.choice(
                    ["a", "b", "c", "d"],
                    size=100,
                    replace=True,
                    p=[0.01, 0.04, 0.5, 0.45],
                ),
                "id": range(100),
                "weight": np.random.uniform(size=100) + 0.5,
            }
        )

        # This works fine
        smpl_to_adj = get_sample_to_adjust(df)
        self.assertIsInstance(smpl_to_adj.adjust(method="ipw"), Sample)

        # This should raise a TypeError:
        with self.assertRaisesRegex(
            TypeError,
            "boolean value of NA is ambiguous",
        ):
            smpl_to_adj = get_sample_to_adjust(df)
            # smpl_to_adj._df.iloc[0, 0] = pd.NA
            smpl_to_adj._df.iloc[0, 1] = pd.NA
            # This will raise the error:
            smpl_to_adj.adjust(method="ipw")

        # This works fine
        df.iloc[0, 0] = np.nan
        df.iloc[0, 1] = np.nan
        smpl_to_adj = get_sample_to_adjust(df)
        self.assertIsInstance(smpl_to_adj.adjust(method="ipw"), Sample)

        # This also works fine (thanks to standardize_types=True)
        df.iloc[0, 0] = pd.NA
        df.iloc[0, 1] = pd.NA
        smpl_to_adj = get_sample_to_adjust(df)
        self.assertIsInstance(smpl_to_adj.adjust(method="ipw"), Sample)

        # Turning standardize_types to False should raise a TypeError (since we have pd.NA):
        with self.assertRaisesRegex(
            TypeError,
            "boolean value of NA is ambiguous",
        ):
            # df.iloc[0, 0] = pd.NA
            df.iloc[0, 1] = pd.NA
            smpl_to_adj = get_sample_to_adjust(df, standardize_types=False)
            smpl_to_adj.adjust(method="ipw")
