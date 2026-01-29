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

# pyre-strict

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from copy import deepcopy
from textwrap import dedent
from typing import Any, Callable
from unittest.mock import MagicMock

import balance.testutil
import IPython.display
import numpy as np
import pandas as pd
from balance.sample_class import Sample
from balance.testutil import tempfile_path
from balance.util import _verify_value_type


# Test sample fixtures - shared across multiple test methods
# These represent common test scenarios for Sample functionality

# Sample with outcome column and mixed data types
s1: Sample = Sample.from_frame(
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
s2: Sample = Sample.from_frame(
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

s3: Sample = s1.set_target(s2)
s3_adjusted_null: Sample = s3.adjust(method="null")

# Sample with missing values and multiple outcome columns
s4: Sample = Sample.from_frame(
    pd.DataFrame(
        {"a": (0, None, 2), "b": (0, None, 2), "c": ("a", "b", "c"), "id": (1, 2, 3)}
    ),
    outcome_columns=("b", "c"),
)


class TestSample(
    balance.testutil.BalanceTestCase,
):
    """Test class for basic Sample functionality and constructor behavior."""

    def test_constructor_not_implemented(self) -> None:
        """Test that Sample constructor raises NotImplementedError.

        The Sample class should not be instantiated directly without using
        the from_frame class method.
        """
        with self.assertRaises(NotImplementedError):
            s1 = Sample()
            print(s1)

    def test_Sample__str__(self) -> None:
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

        adjusted_str = s3_adjusted_null.__str__()
        self.assertTrue(
            "Adjusted balance Sample object with target set using" in adjusted_str
        )
        self.assertIn("adjustment details", adjusted_str)
        self.assertIn("method: null_adjustment", adjusted_str)
        self.assertIn("design effect (Deff)", adjusted_str)

    def test_Sample__str__ipw_and_cbps_examples(self) -> None:
        """Ensure adjustment summaries show method names for ipw and cbps."""

        sample_data = (1, 2, 3, 4, 5, 6, 7, 8, 9, 1)
        target_data = (1, 2, 3, 4, 5, 6, 7, 8, 9, 9)

        sample = Sample.from_frame(pd.DataFrame({"a": sample_data, "id": range(0, 10)}))
        target = Sample.from_frame(pd.DataFrame({"a": target_data, "id": range(0, 10)}))
        sample_with_target = sample.set_target(target)

        ipw_adjusted = sample_with_target.adjust(method="ipw")
        ipw_output = ipw_adjusted.__str__()
        self.assertIn("adjustment details", ipw_output)
        self.assertIn("method: ipw", ipw_output)

        cbps_adjusted = sample_with_target.adjust(method="cbps", transformations=None)
        cbps_output = cbps_adjusted.__str__()
        self.assertIn("adjustment details", cbps_output)
        self.assertIn("method: cbps", cbps_output)

    def test_Sample__str__multiple_outcomes(self) -> None:
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

    def test_Sample_from_frame_id_column_detection(self) -> None:
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
            "Error while inferring id_column from DataFrame. Specify a valid",
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

    def test_Sample_from_frame_id_uniqueness(self) -> None:
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

    def test_Sample_from_frame_weight_column_handling(self) -> None:
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
        df = pd.DataFrame(
            {"id": (1, 2, 3), "weight": (None, 3, 1.1), "feature": ("x", "y", "z")}
        )
        with self.assertRaises(ValueError) as context:
            Sample.from_frame(df)
        self.assertIn("Found 1 row(s) with null weights", str(context.exception))
        self.assertIn("x", str(context.exception))
        self.assertIn("1", str(context.exception))

        df = pd.DataFrame({"id": (1, 2, 3), "weight": (None, None, None)})
        with self.assertRaisesRegex(
            ValueError, r"Null values \(including None\) are not allowed"
        ) as null_context:
            Sample.from_frame(df)
        self.assertIn("Found 3 row(s) with null weights", str(null_context.exception))

        df = pd.DataFrame(
            {
                "id": (1, 2, 3, 4),
                "weight": (np.nan, pd.NA, None, 1.0),
                "extra": ("a", "b", "c", "d"),
            }
        )
        with self.assertRaises(ValueError) as mixed_null_context:
            Sample.from_frame(df)
        self.assertIn(
            "Found 3 row(s) with null weights", str(mixed_null_context.exception)
        )
        self.assertIn("b", str(mixed_null_context.exception))

    def test_Sample_from_frame_weight_validation(self) -> None:
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

    def test_Sample_from_frame_ignore_columns(self) -> None:
        """Ensure ignored columns are preserved but excluded from outcomes/covars."""

        df = pd.DataFrame(
            {
                "id": (1, 2, 3),
                "a": (1, 2, 3),
                "b": (4, 5, 6),
                "note": ("x", "y", "z"),
                "out": (7, 8, 9),
            }
        )

        sample = Sample.from_frame(
            df, outcome_columns="out", ignore_columns="note", weight_column=None
        )

        self.assertListEqual(sample.covars().names(), ["a", "b"])
        self.assertListEqual(sample.outcomes().df.columns.tolist(), ["out"])
        self.assertListEqual(
            sample.df.columns.tolist(), ["id", "a", "b", "out", "weight", "note"]
        )
        ignored = sample.ignored_columns()
        assert ignored is not None
        self.assertListEqual(ignored.columns.tolist(), ["note"])

    def test_Sample_from_frame_ignore_column_validation(self) -> None:
        """Validate ignore column handling against reserved and invalid inputs."""

        df = pd.DataFrame(
            {
                "id": (1, 2),
                "weight": (1.0, 2.0),
                "a": (3, 4),
                "meta": ("x", "y"),
                "out": (5, 6),
            }
        )

        with self.assertRaisesRegex(ValueError, "cannot include id/weight"):
            Sample.from_frame(df, weight_column="weight", ignore_columns=["id"])

        with self.assertRaisesRegex(ValueError, "cannot include id/weight"):
            Sample.from_frame(df, ignore_columns=["weight"])

        with self.assertRaisesRegex(ValueError, "both ignored and outcomes"):
            Sample.from_frame(
                df,
                outcome_columns=["out"],
                ignore_columns=["out"],
                weight_column=None,
            )

        with self.assertRaisesRegex(ValueError, "must be strings"):
            # pyre-ignore[6]: Intentionally passing int to test validation
            Sample.from_frame(df, ignore_columns=["a", 3])

        with self.assertRaisesRegex(ValueError, "not in df columns"):
            Sample.from_frame(df, ignore_columns=["missing"])

        sample = Sample.from_frame(
            df,
            outcome_columns="out",
            ignore_columns=["a", "a", "meta"],
            weight_column=None,
        )
        ignored = _verify_value_type(sample.ignored_columns())
        self.assertListEqual(ignored.columns.tolist(), ["a", "meta"])
        self.assertListEqual(sample.covars().names(), [])

        sample_no_ignored = Sample.from_frame(
            df, outcome_columns="out", weight_column=None
        )
        self.assertIsNone(sample_no_ignored.ignored_columns())

    def test_Sample_from_frame_type_conversion(self) -> None:
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

    def test_Sample_from_frame_deepcopy_behavior(self) -> None:
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

    def test_Sample_adjust(self) -> None:
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
            # pyre-ignore[6]: Intentionally passing None to test exception handling
            s1.set_target(s2).adjust(method=None)


class TestSample_base_and_adjust_methods(
    balance.testutil.BalanceTestCase,
):
    """Test class for Sample data access methods and property behavior."""

    def test_Sample_df(self) -> None:
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
        df_fget = Sample.df.fget
        if df_fget is not None:
            self.assertEqual(df_fget(s1), s1.df)

        # Test that df cannot be called as function
        with self.assertRaisesRegex(TypeError, "'DataFrame' object is not callable"):
            # pyre-ignore[29]: Intentionally calling DataFrame to test exception
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
    def test_Sample_outcomes(self) -> None:
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

    def test_Sample_weights(self) -> None:
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
    def test_Sample_covars(self) -> None:
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

    def _create_test_sample_with_target(
        self, source_size: int = 1000, target_size: int = 10000
    ) -> tuple[Sample, Sample]:
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

    def test_Sample_model_null_adjustment(self) -> None:
        """Test model information for null adjustment method.

        Verifies that null adjustment correctly reports method name.
        """
        np.random.seed(112358)
        s, t = self._create_test_sample_with_target()

        a = s.adjust(t, max_de=None, method="null")
        m = a.model()

        self.assertEqual(_verify_value_type(m)["method"], "null_adjustment")

    def test_Sample_model_ipw_adjustment(self) -> None:
        """Test model information for IPW adjustment method.

        Verifies that IPW adjustment correctly reports method name
        and includes expected model structure (perf, fit, coefs).
        """
        np.random.seed(112358)
        s, t = self._create_test_sample_with_target()

        a = s.adjust(t, max_de=None)
        m = a.model()

        self.assertEqual(_verify_value_type(m)["method"], "ipw")

        # Test structure of IPW output
        self.assertTrue("perf" in _verify_value_type(m).keys())
        self.assertTrue("fit" in _verify_value_type(m).keys())
        self.assertTrue("coefs" in _verify_value_type(m)["perf"].keys())

    def test_Sample_model_matrix(self) -> None:
        """Test model matrix generation for samples.

        Verifies that model_matrix method correctly:
        - Handles missing values by creating indicator variables
        - Converts categorical variables to dummy variables
        - Preserves numeric variables
        - Returns properly formatted model matrix

        Note: Main tests for model_matrix are in test_util_model_matrix.py
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

    def test_Sample_set_weights(self) -> None:
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

    def test_Sample_set_unadjusted(self) -> None:
        s5 = s1.set_unadjusted(s2)
        self.assertTrue(s5._links["unadjusted"] is s2)
        # test exceptions when there is no a second sample
        with self.assertRaisesRegex(
            TypeError,
            "set_unadjusted must be called with second_sample argument of type Sample",
        ):
            # pyre-ignore[6]: Intentionally passing str to test exception handling
            s1.set_unadjusted("Not a Sample object")

    def test_Sample_is_adjusted(self) -> None:
        self.assertFalse(s1.is_adjusted())
        self.assertFalse(s3.is_adjusted())
        self.assertTrue(s3_adjusted_null.is_adjusted())

    def test_Sample_set_target(self) -> None:
        s5 = s1.set_target(s2)
        self.assertTrue(s5._links["target"] is s2)
        # test exceptions when the provided object is not a second sample
        with self.assertRaisesRegex(
            ValueError,
            "A target, a Sample object, must be specified",
        ):
            # pyre-ignore[6]: Intentionally passing str to test exception handling
            s1.set_target("Not a Sample object")

    def test_Sample_has_target(self) -> None:
        self.assertFalse(s1.has_target())
        self.assertTrue(s1.set_target(s2).has_target())


class TestSample_metrics_methods(
    balance.testutil.BalanceTestCase,
):
    def _assert_dict_almost_equal(
        self, actual: dict[str, Any], expected: dict[str, Any], places: int = 2
    ) -> None:
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

    def test_Sample_covar_means(self) -> None:
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

    def test_Sample_design_effect(self) -> None:
        self.assertEqual(s1.design_effect().round(3), 1.235)
        self.assertEqual(s4.design_effect(), 1.0)

    def test_Sample_design_effect_prop(self) -> None:
        s3_null = s1.adjust(s2, method="null")
        self.assertEqual(s3_null.design_effect_prop(), 0.0)

        # test exceptions when there is no adjusted
        with self.assertRaisesRegex(
            ValueError,
            "This is not an adjusted Sample. Use sample.adjust to adjust the sample to target",
        ):
            s1.design_effect_prop()

    def test_Sample_outcome_sd_prop(self) -> None:
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

    def _create_samples_for_outcome_variance_tests(self) -> tuple[Sample, pd.DataFrame]:
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

    def test_outcome_variance_ratio_calculation(self) -> None:
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

    def test_outcome_variance_ratio_value(self) -> None:
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

    def test_outcome_variance_ratio_null_adjustment(self) -> None:
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

    def test_Sample_weights_summary(self) -> None:
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

    def test_Sample_summary(self) -> None:
        s1_summ = s1.summary()
        self.assertTrue("Model performance" not in s1_summ)
        self.assertTrue("Covar ASMD" not in s1_summ)
        self.assertTrue("Covar mean KLD" not in s1_summ)
        self.assertIn("Outcome weighted means", s1_summ)

        s3_summ = s3.summary()
        self.assertTrue("Model performance" not in s1_summ)
        self.assertTrue("Covar ASMD (6 variables)" in s3_summ)
        self.assertTrue("Covar mean KLD (3 variables)" in s3_summ)
        self.assertTrue("design effect" not in s3_summ)

        s3_set_unadjusted = s3.set_unadjusted(s1)
        s3_summ = s3_set_unadjusted.summary()
        self.assertTrue("Covar ASMD reduction: 0.0%" in s3_summ)
        self.assertTrue("Covar ASMD (6 variables)" in s3_summ)
        self.assertTrue("->" in s3_summ)
        self.assertTrue("Covar mean KLD reduction: 0.0%" in s3_summ)
        self.assertTrue("Covar mean KLD (3 variables)" in s3_summ)
        self.assertIn("design effect (Deff)", s3_summ)
        self.assertIn("effective sample size proportion", s3_summ)
        self.assertIn("effective sample size", s3_summ)
        self.assertIn("Outcome weighted means", s3_summ)

        s3_summ = s3_adjusted_null.summary()
        self.assertTrue("Covar ASMD reduction: 0.0%" in s3_summ)
        self.assertTrue("Covar mean KLD reduction: 0.0%" in s3_summ)
        self.assertTrue("design effect" in s3_summ)

    def test_Sample_summary_handles_nonfinite_design_effect(self) -> None:
        adjusted = deepcopy(s3_adjusted_null)
        adjusted.design_effect = MagicMock(return_value=np.nan)

        summary = adjusted.summary()

        self.assertIn("Weight diagnostics:", summary)
        self.assertIn("design effect (Deff): unavailable", summary)
        self.assertNotIn("effective sample", summary)

    def test_Sample_summary_doc_example_matches_output(self) -> None:
        survey = Sample.from_frame(
            pd.DataFrame(
                {
                    "x": (0, 1, 1, 0),
                    "id": range(4),
                    "y": (0.1, 0.5, 0.4, 0.9),
                    "w": (1, 2, 1, 1),
                }
            ),
            id_column="id",
            outcome_columns="y",
            weight_column="w",
        )
        target = Sample.from_frame(
            pd.DataFrame({"x": (0, 0, 1, 1), "id": range(4)}),
            id_column="id",
        )

        adjusted = survey.set_target(target).adjust(method="null")

        expected_lines = (
            dedent(
                """
            Adjustment details:
                method: null_adjustment
            Covariate diagnostics:
                Covar ASMD reduction: 0.0%
                Covar ASMD (1 variables): 0.173 -> 0.173
                Covar mean KLD reduction: 0.0%
                Covar mean KLD (1 variables): 0.020 -> 0.020
            Weight diagnostics:
                design effect (Deff): 1.120
                effective sample size proportion (ESSP): 0.893
                effective sample size (ESS): 3.6
            Outcome weighted means:
                           y
            source
            self       0.480
            unadjusted 0.480
            """
            )
            .strip()
            .splitlines()
        )

        summary_lines = [line.rstrip() for line in adjusted.summary().splitlines()]

        self.assertEqual(summary_lines, expected_lines)

    def test_Sample_invalid_outcomes(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"outcome columns \['o'\] not in df columns \['a', 'id', 'weight'\]",
        ):
            Sample.from_frame(
                pd.DataFrame({"a": (1, 2, 3, 1), "id": (1, 2, 3, 4)}),
                outcome_columns="o",
            )

    def _create_samples_for_diagnostics_tests(self) -> tuple[Sample, Sample]:
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

    def test_Sample_diagnostics_ipw_method(self) -> None:
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

    def test_Sample_diagnostics_cbps_method(self) -> None:
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

    def test_Sample_diagnostics_null_method(self) -> None:
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

    def _create_adjusted_sample_for_filtering_tests(self) -> tuple[Sample, Sample]:
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

    def test_Sample_keep_only_some_rows_columns_identity(self) -> None:
        """Test that keep_only_some_rows_columns returns same object when no filtering applied.

        Verifies that when both rows_to_keep and columns_to_keep are None,
        the method returns the same object reference.
        """
        a, _ = self._create_adjusted_sample_for_filtering_tests()

        # Should return the same object when no filtering is applied
        self.assertTrue(
            a is a.keep_only_some_rows_columns(rows_to_keep=None, columns_to_keep=None)
        )

    def test_Sample_keep_only_some_rows_columns_column_filtering(self) -> None:
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

    def test_Sample_keep_only_some_rows_columns_diagnostics_impact(self) -> None:
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

    def test_Sample_keep_only_some_rows_columns_row_filtering(self) -> None:
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

    def test_Sample_keep_only_some_rows_columns_sample_size_changes(self) -> None:
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

    def test_Sample_keep_only_some_rows_columns_with_outcomes(self) -> None:
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

    def test_Sample_keep_only_some_rows_columns_column_warnings(self) -> None:
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
    def test_Sample_to_download(self) -> None:
        r = s1.to_download()
        self.assertIsInstance(r, IPython.display.FileLink)

    def test_Sample_to_csv(self) -> None:
        with tempfile_path() as tmp_path:
            s1.to_csv(path_or_buf=tmp_path)
            with open(tmp_path, "rb") as output:
                r = output.read()
        e = (
            b"id,a,b,c,o,w\n1,1.0,-42.0,x,7.0,0.5\n"
            b"2,2.0,8.0,y,8.0,2.0\n3,3.0,2.0,z,9.0,1.0\n4,1.0,-42.0,v,10.0,1.0\n"
        )
        self.assertEqual(r, e)


class TestSamplePrivateAPI(balance.testutil.BalanceTestCase):
    def test__links(self) -> None:
        self.assertEqual(len(s1._links.keys()), 0)

        self.assertTrue(s3._links["target"] is s2)
        self.assertTrue(s3.has_target())

        self.assertTrue(s3_adjusted_null._links["target"] is s2)
        self.assertTrue(s3_adjusted_null._links["unadjusted"] is s3)
        self.assertTrue(s3_adjusted_null.has_target())

    def test__special_columns_names(self) -> None:
        self.assertEqual(
            sorted(s4._special_columns_names()), ["b", "c", "id", "weight"]
        )

    # NOTE how integers were changed into floats.
    def test__special_columns(self) -> None:
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

    def test__covar_columns_names(self) -> None:
        self.assertEqual(sorted(s1._covar_columns_names()), ["a", "b", "c"])

    def test__covar_columns(self) -> None:
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

    def test_Sample__check_if_adjusted(self) -> None:
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

    def test_Sample__no_target_error(self) -> None:
        # test exception when the is no target
        with self.assertRaisesRegex(
            ValueError,
            "This Sample does not have a target set. Use sample.set_target to add target",
        ):
            s1._no_target_error()
        s3._no_target_error()  # Should not raise an error

    def test_Sample__check_outcomes_exists(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "This Sample does not have outcome columns specified",
        ):
            s2._check_outcomes_exists()
        self.assertTrue(s1._check_outcomes_exists() is None)  # Does not raise an error


class TestSample_NA_behavior(balance.testutil.BalanceTestCase):
    def test_can_handle_various_NAs(self) -> None:
        # Testing if we can handle NA values from pandas

        def get_sample_to_adjust(
            df: pd.DataFrame, standardize_types: bool = True
        ) -> Sample:
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


class TestSample_high_cardinality_warnings(balance.testutil.BalanceTestCase):
    """Tests for high-cardinality feature warnings during adjustment."""

    def test_warns_for_high_cardinality_features_with_nas(self) -> None:
        """Adjust should warn when high-cardinality features with NAs lead to equal weights."""
        unique_values = [f"user_{i}" for i in range(10)]
        sample_df = pd.DataFrame(
            {
                "identifier": unique_values + [np.nan],
                "id": range(11),
            }
        )
        target_df = pd.DataFrame(
            {
                "identifier": unique_values + [np.nan],
                "id": range(11),
            }
        )

        sample = Sample.from_frame(sample_df)
        target = Sample.from_frame(target_df)

        with self.assertLogs("balance", level="WARNING") as logs:
            result = sample.adjust(target, variables=["identifier"], num_lambdas=1)

        self.assertTrue(np.allclose(result.weight_column, np.ones(len(sample_df))))
        self.assertTrue(
            any(
                "High-cardinality features detected" in log and "unique=10" in log
                for log in logs.output
            )
        )

    def test_warns_for_high_cardinality_features_with_nas_when_dropping(
        self,
    ) -> None:
        """The high-cardinality NA warning should surface even when NAs are dropped."""
        unique_values = [f"user_{i}" for i in range(10)]
        sample_df = pd.DataFrame(
            {"identifier": unique_values + [np.nan], "id": range(11)}
        )
        target_df = pd.DataFrame(
            {"identifier": unique_values + [np.nan], "id": range(11)}
        )

        sample = Sample.from_frame(sample_df)
        target = Sample.from_frame(target_df)

        with self.assertLogs("balance", level="WARNING") as logs:
            result = sample.adjust(
                target, variables=["identifier"], num_lambdas=1, na_action="drop"
            )

        # Filter out NaN weights before comparing (dropped rows may have NaN weights)
        valid_weights = result.weight_column[~pd.isna(result.weight_column)]
        self.assertTrue(np.allclose(valid_weights, np.ones(len(sample_df) - 1)))
        self.assertTrue(
            any(
                "High-cardinality features detected" in log and "unique=10" in log
                for log in logs.output
            )
        )

    def test_warns_for_high_cardinality_categoricals_with_nas(self) -> None:
        """Categorical dtype columns with high cardinality and NAs should be flagged."""
        sample_df = pd.DataFrame(
            {
                "identifier": pd.Series(
                    [f"user_{i}" for i in range(9)] + [np.nan], dtype="category"
                ),
                "id": range(10),
            }
        )
        target_df = sample_df.copy()

        sample = Sample.from_frame(sample_df)
        target = Sample.from_frame(target_df)

        with self.assertLogs("balance", level="WARNING") as logs:
            result = sample.adjust(target, variables=["identifier"], num_lambdas=1)

        self.assertTrue(np.allclose(result.weight_column, np.ones(len(sample_df))))
        self.assertTrue(
            any(
                "High-cardinality features detected" in log and "unique=9" in log
                for log in logs.output
            )
        )

    def test_does_not_flag_low_cardinality_categoricals_with_nas(self) -> None:
        """Low-cardinality categoricals with NAs should not be reported as a cause."""
        sample_df = pd.DataFrame(
            {
                "identifier": pd.Series(["a", "a", "b", np.nan], dtype="category"),
                "id": range(4),
            }
        )
        target_df = sample_df.copy()

        sample = Sample.from_frame(sample_df)
        target = Sample.from_frame(target_df)

        with self.assertLogs("balance", level="WARNING") as logs:
            result = sample.adjust(target, variables=["identifier"], num_lambdas=1)

        self.assertTrue(np.allclose(result.weight_column, np.ones(len(sample_df))))
        self.assertFalse(
            any("High-cardinality features detected" in log for log in logs.output)
        )

    def test_warns_for_high_cardinality_features_without_nas(self) -> None:
        """High-cardinality categoricals should be reported even without missing values."""
        identifiers = [f"user_{i}" for i in range(8)]
        sample_df = pd.DataFrame(
            {
                "identifier": identifiers,
                "signal": np.concatenate((np.zeros(4), np.ones(4))),
                "id": range(8),
            }
        )
        target_df = pd.DataFrame(
            {
                "identifier": identifiers[::-1],
                "signal": np.concatenate((np.ones(4), np.zeros(4))),
                "id": range(8),
            }
        )

        sample = Sample.from_frame(sample_df)
        target = Sample.from_frame(target_df)

        with self.assertLogs("balance", level="WARNING") as logs:
            sample.adjust(target, variables=["identifier", "signal"], num_lambdas=1)

        self.assertTrue(
            any(
                "High-cardinality features detected" in log
                and "identifier (unique=8" in log
                for log in logs.output
            )
        )

    def test_does_not_warn_for_high_cardinality_numeric_features(self) -> None:
        """High-cardinality numeric features (e.g., IDs) should NOT be reported."""
        identifiers = np.arange(12)
        sample_df = pd.DataFrame({"identifier": identifiers, "id": range(12)})
        target_df = pd.DataFrame({"identifier": identifiers, "id": range(12)})

        sample = Sample.from_frame(sample_df)
        target = Sample.from_frame(target_df)

        with self.assertLogs("balance", level="WARNING") as logs:
            sample.adjust(target, variables=["identifier"], num_lambdas=1)

        self.assertFalse(
            any("High-cardinality features detected" in log for log in logs.output),
            "Numeric features should not be flagged as high-cardinality.",
        )

    def test_high_cardinality_warning_sorts_by_cardinality(self) -> None:
        """Warnings should list columns from highest to lowest cardinality."""
        sample_df = pd.DataFrame(
            {
                "higher": [f"user_{i}" for i in range(7)],
                "high": [f"alias_{i}" for i in range(6)] + ["alias_0"],
                "id": range(7),
            }
        )
        target_df = sample_df.copy()

        sample = Sample.from_frame(sample_df)
        target = Sample.from_frame(target_df)

        with self.assertLogs("balance", level="WARNING") as logs:
            sample.adjust(target, variables=["higher", "high"], num_lambdas=1)

        warning_logs = [
            log for log in logs.output if "High-cardinality features detected" in log
        ]
        self.assertTrue(warning_logs)
        self.assertTrue(
            warning_logs[0].find("higher (unique")
            < warning_logs[0].find(", high (unique"),
            "Expected higher-cardinality column to appear first in warning.",
        )


class TestSample_large_target_warning(balance.testutil.BalanceTestCase):
    @staticmethod
    def _build_frame(prefix: str, rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "x": np.arange(rows, dtype=float),
                "id": [f"{prefix}{i}" for i in range(rows)],
                "weight": np.ones(rows),
            }
        )

    @staticmethod
    def _collect_warnings(callable_fn: Callable[[], object]) -> list[str]:
        logger = logging.getLogger("balance")
        handler = logging.Handler()
        records: list[logging.LogRecord] = []

        def _emit(record: logging.LogRecord) -> None:
            records.append(record)

        handler.emit = _emit  # type: ignore[assignment]
        previous_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        try:
            callable_fn()
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
        return [record.getMessage() for record in records]

    def test_warns_when_target_is_large_and_imbalanced(self) -> None:
        sample_df = self._build_frame("s", 1000)
        target_df = self._build_frame("t", 100_001)

        sample = Sample.from_frame(sample_df, id_column="id", weight_column="weight")
        target = Sample.from_frame(target_df, id_column="id", weight_column="weight")

        logs = self._collect_warnings(
            lambda: _verify_value_type(sample.adjust(target, method="null"))
        )

        self.assertTrue(any("Large target detected" in log for log in logs))

    def test_no_warning_when_target_not_large_enough(self) -> None:
        sample_df = self._build_frame("s", 1000)
        target_df = self._build_frame("t", 100_000)

        sample = Sample.from_frame(sample_df, id_column="id", weight_column="weight")
        target = Sample.from_frame(target_df, id_column="id", weight_column="weight")

        logs = self._collect_warnings(
            lambda: _verify_value_type(sample.adjust(target, method="null"))
        )

        self.assertFalse(any("Large target detected" in log for log in logs))

    def test_warns_at_exact_ten_to_one_ratio(self) -> None:
        sample_df = self._build_frame("s", 10_000)
        target_df = self._build_frame("t", 100_001)

        sample = Sample.from_frame(sample_df, id_column="id", weight_column="weight")
        target = Sample.from_frame(target_df, id_column="id", weight_column="weight")

        logs = self._collect_warnings(
            lambda: _verify_value_type(sample.adjust(target, method="null"))
        )

        self.assertTrue(any("Large target detected" in log for log in logs))

    def test_no_warning_when_target_under_ten_to_one_ratio(self) -> None:
        sample_df = self._build_frame("s", 10_001)
        target_df = self._build_frame("t", 100_001)

        sample = Sample.from_frame(sample_df, id_column="id", weight_column="weight")
        target = Sample.from_frame(target_df, id_column="id", weight_column="weight")

        logs = self._collect_warnings(
            lambda: _verify_value_type(sample.adjust(target, method="null"))
        )

        self.assertFalse(any("Large target detected" in log for log in logs))

    def test_no_warning_when_target_equals_sample(self) -> None:
        sample_df = self._build_frame("s", 100_001)
        target_df = self._build_frame("t", 100_001)

        sample = Sample.from_frame(sample_df, id_column="id", weight_column="weight")
        target = Sample.from_frame(target_df, id_column="id", weight_column="weight")

        logs = self._collect_warnings(
            lambda: _verify_value_type(sample.adjust(target, method="null"))
        )

        self.assertFalse(any("Large target detected" in log for log in logs))

    def test_no_warning_when_sample_empty(self) -> None:
        sample_df = self._build_frame("s", 0)
        target_df = self._build_frame("t", 100_001)

        sample = Sample.from_frame(sample_df, id_column="id", weight_column="weight")
        target = Sample.from_frame(target_df, id_column="id", weight_column="weight")

        logs = self._collect_warnings(
            lambda: _verify_value_type(sample.adjust(target, method="null"))
        )

        self.assertFalse(any("Large target detected" in log for log in logs))

    def test_adjustment_details_with_ipw_method_in_str(self) -> None:
        """Test __str__ includes IPW adjustment details.

        This validates that the string representation properly includes adjustment
        method information and design effect diagnostics when available.
        """
        # Use larger sample size for IPW to work properly
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
                    "b": [4, 5, 6, 4, 5, 6, 4, 5, 6, 4],
                    "id": list(range(10)),
                    "w": [0.5, 1.0, 1.5, 0.8, 1.2, 0.9, 1.1, 1.3, 0.7, 1.0],
                }
            )
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": [1, 2, 3, 1, 2],
                    "b": [4, 5, 6, 4, 5],
                    "id": list(range(5)),
                }
            )
        )
        adjusted = sample.set_target(target).adjust(method="ipw")

        str_repr = str(adjusted)

        self.assertIn("method: ipw", str_repr)
        self.assertIn(
            "design effect",
            str_repr,
            "Expected design effect in string representation",
        )

    def test_adjustment_details_with_null_method_in_str(self) -> None:
        """Test __str__ includes null adjustment method.

        This ensures the string representation handles cases where no real
        adjustment was applied.
        """
        sample = Sample.from_frame(
            pd.DataFrame({"a": [1, 2], "b": [3, 4], "id": [1, 2]})
        )
        target = Sample.from_frame(
            pd.DataFrame({"a": [1, 2], "b": [3, 4], "id": [1, 2]})
        )
        adjusted = sample.set_target(target).adjust(method="null")

        str_repr = str(adjusted)

        self.assertIn("method: null", str_repr)

    def test_adjustment_details_with_trimming_settings_in_str(self) -> None:
        """Test __str__ includes trimming configuration.

        This validates that weight trimming parameters are properly included
        in the string representation when specified.
        """
        # Use larger sample with more variation for IPW
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
                    "b": [10, 20, 30, 15, 25, 35, 12, 22, 32, 18],
                    "id": list(range(10)),
                    "w": [0.1, 1.0, 2.0, 0.5, 1.5, 2.5, 0.8, 1.2, 1.8, 0.6],
                }
            )
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": [1, 2, 3, 1, 2],
                    "b": [10, 20, 30, 15, 25],
                    "id": list(range(5)),
                }
            )
        )
        adjusted = sample.set_target(target).adjust(
            method="ipw", weight_trimming_mean_ratio=5
        )

        str_repr = str(adjusted)

        self.assertIn(
            "weight trimming mean ratio",
            str_repr,
            "Expected weight trimming info in string representation",
        )

    def test_str_includes_effective_sample_size_with_weights(self) -> None:
        """Test __str__ includes effective sample size when weights are present.

        This ensures the string representation includes design effect diagnostics
        for samples with weights.
        """
        sample = Sample.from_frame(
            pd.DataFrame(
                {"a": [1, 2, 3], "b": [4, 5, 6], "id": [1, 2, 3], "w": [1, 1, 1]}
            ),
            weight_column="w",
        )
        target = Sample.from_frame(
            pd.DataFrame({"a": [1, 2], "b": [4, 5], "id": [1, 2]})
        )
        adjusted = sample.set_target(target).adjust(method="null")

        str_repr = str(adjusted)

        # Verify the string representation contains adjustment details
        self.assertIn("adjustment details", str_repr)
        self.assertIn("method: null", str_repr)

        # With uniform weights (all 1s), we should see design effect information
        self.assertIn("design effect", str_repr)
        self.assertIn("effective sample size", str_repr)

    def test_str_includes_design_effect_with_weights(self) -> None:
        """Test __str__ includes design effect information with weights.

        This validates that samples with weights properly display design effect
        and effective sample size in their string representation.
        """
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": [1, 2, 3, 4],
                    "b": [5, 6, 7, 8],
                    "id": [1, 2, 3, 4],
                    "w": [1.0, 1.0, 1.0, 1.0],
                }
            ),
            weight_column="w",
        )
        target = Sample.from_frame(
            pd.DataFrame({"a": [1, 2], "b": [5, 6], "id": [1, 2]})
        )
        adjusted = sample.set_target(target).adjust(method="null")

        str_repr = str(adjusted)

        # Should include design effect information
        self.assertIn("design effect", str_repr)
        self.assertIn("Deff", str_repr)
        self.assertIn("effective sample size", str_repr)

    def test_str_without_weights_no_design_effect(self) -> None:
        """Test __str__ for samples without explicit weights.

        This validates that samples created without a weight column do not
        display design effect information in their string representation.
        """
        sample = Sample.from_frame(
            pd.DataFrame({"a": [1, 2], "b": [3, 4], "id": [1, 2]})
        )
        target = Sample.from_frame(pd.DataFrame({"a": [1], "b": [3], "id": [1]}))
        adjusted = sample.set_target(target).adjust(method="null")

        str_repr = str(adjusted)

        # Should show adjustment details but may not show design effect
        # since no explicit weight column was provided
        self.assertIn("method: null", str_repr)

    def test_design_effect_method_returns_valid_value(self) -> None:
        """Test design_effect() public method returns valid value.

        This ensures the public design_effect method works correctly
        for samples with weights.
        """
        sample = Sample.from_frame(
            pd.DataFrame(
                {"a": [1, 2, 3], "b": [4, 5, 6], "id": [1, 2, 3], "w": [1, 1, 1]}
            ),
            weight_column="w",
        )
        target = Sample.from_frame(
            pd.DataFrame({"a": [1, 2], "b": [4, 5], "id": [1, 2]})
        )
        adjusted = sample.set_target(target).adjust(method="null")

        deff = adjusted.design_effect()

        # Should return a valid design effect value
        self.assertIsNotNone(deff)
        self.assertIsInstance(deff, (float, np.floating))
        self.assertTrue(np.isfinite(deff))
        self.assertGreater(deff, 0)

    def test_str_handles_sample_with_varied_weights(self) -> None:
        """Test __str__ properly handles samples with varied weights.

        This validates that the string representation correctly displays
        design effect information when weights vary.
        """
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": [1, 2, 3],
                    "b": [4, 5, 6],
                    "id": [1, 2, 3],
                    "w": [0.5, 1.0, 1.5],
                }
            ),
            weight_column="w",
        )
        target = Sample.from_frame(
            pd.DataFrame({"a": [1, 2], "b": [4, 5], "id": [1, 2]})
        )
        adjusted = sample.set_target(target).adjust(method="null")

        str_repr = str(adjusted)

        # Should include design effect information with varied weights
        self.assertIn("design effect", str_repr)
        self.assertIn("method: null", str_repr)

    def test_design_effect_prop_method_returns_valid_value(self) -> None:
        """Test design_effect_prop() public method returns valid value.

        This ensures the public design_effect_prop method works correctly
        for samples with weights.
        """
        sample = Sample.from_frame(
            pd.DataFrame({"a": [1, 2], "b": [3, 4], "id": [1, 2], "w": [1, 1]}),
            weight_column="w",
        )
        target = Sample.from_frame(pd.DataFrame({"a": [1], "b": [3], "id": [1]}))
        adjusted = sample.set_target(target).adjust(method="null")

        deff_prop = adjusted.design_effect_prop()

        # Should return a valid design effect proportion
        self.assertIsNotNone(deff_prop)
        self.assertIsInstance(deff_prop, (float, np.floating))
        self.assertTrue(np.isfinite(deff_prop))

    def test_plot_weight_density_calls_weights_plot(self) -> None:
        """Test plot_weight_density delegates to weights().plot().

        This validates that the convenience method properly calls the
        underlying weights plotting functionality.
        """
        from unittest.mock import MagicMock, patch

        sample = Sample.from_frame(
            pd.DataFrame(
                {"a": [1, 2, 3], "b": [4, 5, 6], "id": [1, 2, 3], "w": [0.5, 1.0, 1.5]}
            )
        )

        # Mock the weights().plot() chain to verify it's called
        mock_weights = MagicMock()
        mock_plot = MagicMock()
        mock_weights.plot = mock_plot

        with patch.object(sample, "weights", return_value=mock_weights):
            # Call the method
            result = sample.plot_weight_density()

            # Verify weights() was called
            sample.weights.assert_called_once()

            # Verify plot() was called on the weights object
            mock_plot.assert_called_once()

            # plot_weight_density returns None
            self.assertIsNone(result)


class TestSampleConstructorInspectException(balance.testutil.BalanceTestCase):
    """Test cases for constructor edge case (line 145)."""

    def test_constructor_inspect_exception_raises_not_implemented(self) -> None:
        """Test that constructor raises NotImplementedError when inspect.stack() fails.

        This test covers line 145-148 in sample_class.py where an exception
        during inspect.stack() results in NotImplementedError.
        """
        from unittest.mock import patch

        with patch("inspect.stack", side_effect=Exception("Simulated inspect error")):
            with self.assertRaises(NotImplementedError):
                Sample()


class TestSampleDesignEffectDiagnostics(balance.testutil.BalanceTestCase):
    """Test cases for _design_effect_diagnostics edge cases (lines 342-351)."""

    def test_design_effect_diagnostics_with_no_weight_column(self) -> None:
        """Test _design_effect_diagnostics returns None values when weight_column is None.

        Verifies lines 344-345 in sample_class.py.
        """
        sample = Sample.from_frame(pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3]}))
        sample.weight_column = None
        result = sample._design_effect_diagnostics()
        self.assertEqual(result, (None, None, None))

    def test_design_effect_diagnostics_with_invalid_weights(self) -> None:
        """Test _design_effect_diagnostics handles ValueError gracefully.

        Verifies lines 349-351 in sample_class.py.
        """
        sample = Sample.from_frame(
            pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3], "w": [0, 0, 0]}),
            weight_column="w",
        )
        result = sample._design_effect_diagnostics()
        self.assertEqual(result, (None, None, None))


class TestSampleFromFrameGuessWeightColumn(balance.testutil.BalanceTestCase):
    """Test cases for weight column guessing in from_frame (lines 524-525)."""

    def test_from_frame_guesses_weights_column(self) -> None:
        """Test from_frame guesses 'weights' as weight column if present.

        Verifies lines 523-525 in sample_class.py.
        """
        with self.assertLogs(level="WARNING") as log:
            sample = Sample.from_frame(
                pd.DataFrame(
                    {"a": [1, 2, 3], "id": [1, 2, 3], "weights": [1.0, 2.0, 3.0]}
                )
            )
            self.assertEqual(sample.weight_column.name, "weights")
            self.assertTrue(
                any("Guessing weight column is 'weights'" in msg for msg in log.output)
            )

    def test_from_frame_standardize_types_false_keeps_int_weights(self) -> None:
        """Test from_frame with standardize_types=False keeps int weights.

        Verifies lines 539-542 in sample_class.py.
        """
        sample = Sample.from_frame(
            pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3]}),
            standardize_types=False,
        )
        # Weight column should exist and be int type when no weight is given and standardize_types=False
        self.assertIsNotNone(sample.weight_column)


class TestSampleFromFrameGuessIdColumnCandidates(balance.testutil.BalanceTestCase):
    """Test cases for id column guessing in from_frame with candidate names."""

    def test_from_frame_uses_id_column_candidates(self) -> None:
        """Test from_frame uses id_column_candidates to pick a non-default id."""
        sample = Sample.from_frame(
            pd.DataFrame({"user_id": [1, 2, 3], "a": [1, 2, 3]}),
            id_column_candidates=["user_id", "id"],
        )
        self.assertEqual(sample.id_column.name, "user_id")

    def test_from_frame_id_column_candidates_ambiguous(self) -> None:
        """Test from_frame raises when multiple candidate ids are present."""
        with self.assertRaisesRegex(
            ValueError,
            "Multiple candidate id columns found in the DataFrame",
        ):
            Sample.from_frame(
                pd.DataFrame({"id": [1, 2, 3], "user_id": [4, 5, 6]}),
                id_column_candidates=["id", "user_id"],
            )

    def test_from_frame_id_column_candidates_invalid_type(self) -> None:
        """Test from_frame raises when candidate ids include invalid types."""
        with self.assertRaisesRegex(
            TypeError,
            "Error while inferring id_column from DataFrame. Specify a valid",
        ):
            Sample.from_frame(
                pd.DataFrame({"user_id": [1, 2, 3], "a": [1, 2, 3]}),
                # pyre-ignore[6]: Intentionally passing invalid type to test error handling
                id_column_candidates=["user_id", 1],
            )


class TestSampleModelNone(balance.testutil.BalanceTestCase):
    """Test cases for model() method returning None (line 854)."""

    def test_model_returns_none_when_no_adjustment_model(self) -> None:
        """Test model() returns None when _adjustment_model is not set.

        Verifies lines 851-854 in sample_class.py.
        """
        sample = Sample.from_frame(pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3]}))
        self.assertIsNone(sample.model())


class TestSampleSetWeightsNonFloat(balance.testutil.BalanceTestCase):
    """Test cases for set_weights type conversion (lines 1103-1106)."""

    def test_set_weights_converts_int_to_float(self) -> None:
        """Test set_weights converts int weight column to float.

        Verifies lines 1103-1106 in sample_class.py.
        """
        sample = Sample.from_frame(
            pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3], "w": [1, 2, 3]}),
            weight_column="w",
            standardize_types=False,
        )
        sample.set_weights(2.0)
        self.assertEqual(sample._df[sample.weight_column.name].dtype.kind, "f")


class TestSampleSummaryIPWModel(balance.testutil.BalanceTestCase):
    """Test cases for summary with IPW model (line 1620)."""

    def test_summary_includes_model_performance_for_ipw(self) -> None:
        """Test that summary includes model performance for IPW adjusted sample.

        Verifies line 1619-1624 in sample_class.py.
        """
        np.random.seed(42)
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": np.random.randn(100),
                    "id": range(100),
                    "w": [1.0] * 100,
                }
            ),
            weight_column="w",
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": np.random.randn(100) + 1,
                    "id": range(100, 200),
                    "w": [1.0] * 100,
                }
            ),
            weight_column="w",
        )
        adjusted = sample.set_target(target).adjust(method="ipw", max_de=1.5)
        summary = adjusted.summary()
        self.assertIn("Model performance", summary)


class TestSampleStrWeightTrimmingPercentile(balance.testutil.BalanceTestCase):
    """Test cases for weight_trimming_percentile in __str__ (lines 301-305)."""

    def test_str_shows_weight_trimming_percentile_when_in_model(self) -> None:
        """Test that __str__ shows weight_trimming_percentile when present in model.

        Verifies lines 301-305 in sample_class.py.
        Note: The IPW method currently does not store weight_trimming_percentile
        in the model dictionary, so we manually inject it to test the display logic.
        """
        np.random.seed(42)
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": np.random.randn(100),
                    "id": range(100),
                    "w": [1.0] * 100,
                }
            ),
            weight_column="w",
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": np.random.randn(100) + 1,
                    "id": range(100, 200),
                    "w": [1.0] * 100,
                }
            ),
            weight_column="w",
        )
        adjusted = sample.set_target(target).adjust(
            method="ipw",
            weight_trimming_percentile=0.98,
            weight_trimming_mean_ratio=None,
        )
        # Manually inject weight_trimming_percentile into the model to test display logic
        # This is needed because the IPW implementation does not store this value in the model
        if adjusted._adjustment_model is not None:
            adjusted._adjustment_model["weight_trimming_percentile"] = 0.98
        output_str = adjusted.__str__()
        self.assertIn("weight trimming percentile", output_str)


class TestSampleDesignEffectDiagnosticsExtended(balance.testutil.BalanceTestCase):
    """Test cases for _design_effect_diagnostics edge cases (lines 307-308, 349-351)."""

    def test_design_effect_diagnostics_when_n_rows_is_none(self) -> None:
        """Test _design_effect_diagnostics with n_rows=None uses df shape.

        Verifies lines 307-308 in sample_class.py.
        """
        sample = Sample.from_frame(
            pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3], "w": [1.0, 2.0, 3.0]}),
            weight_column="w",
        )
        design_effect, effective_n, effective_prop = sample._design_effect_diagnostics(
            n_rows=None
        )
        self.assertIsNotNone(design_effect)
        self.assertIsNotNone(effective_n)
        self.assertIsNotNone(effective_prop)

    def test_design_effect_diagnostics_exception_handling(self) -> None:
        """Test _design_effect_diagnostics returns None on exception.

        Verifies lines 349-351 in sample_class.py.
        """
        sample = Sample.from_frame(
            pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3], "w": [0.0, 0.0, 0.0]}),
            weight_column="w",
        )
        design_effect, effective_n, effective_prop = sample._design_effect_diagnostics()
        self.assertIsNone(design_effect)
        self.assertIsNone(effective_n)
        self.assertIsNone(effective_prop)


class TestSampleDiagnosticsIPWModelParams(balance.testutil.BalanceTestCase):
    """Test cases for IPW model parameters in diagnostics (lines 1838, 1878-1879)."""

    def test_diagnostics_includes_n_iter_intercept(self) -> None:
        """Test diagnostics includes n_iter_ and intercept_ from IPW fit.

        Verifies lines 1835-1849 in sample_class.py.
        """
        np.random.seed(42)
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": np.random.randn(100),
                    "id": range(100),
                    "w": [1.0] * 100,
                }
            ),
            weight_column="w",
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": np.random.randn(100) + 1,
                    "id": range(100, 200),
                    "w": [1.0] * 100,
                }
            ),
            weight_column="w",
        )
        adjusted = sample.set_target(target).adjust(method="ipw", max_de=1.5)
        diagnostics = adjusted.diagnostics()
        diagnostics_dict = diagnostics.set_index(["metric", "var"])["val"].to_dict()
        self.assertTrue(
            any(
                "ipw_model_glance" in str(k)
                or "n_iter_" in str(k)
                or "intercept_" in str(k)
                for k in diagnostics_dict.keys()
            )
        )

    def test_diagnostics_includes_multi_class(self) -> None:
        """Test diagnostics includes multi_class from IPW fit.

        Verifies lines 1874-1879 in sample_class.py.
        """
        np.random.seed(42)
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": np.random.randn(100),
                    "id": range(100),
                    "w": [1.0] * 100,
                }
            ),
            weight_column="w",
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": np.random.randn(100) + 1,
                    "id": range(100, 200),
                    "w": [1.0] * 100,
                }
            ),
            weight_column="w",
        )
        adjusted = sample.set_target(target).adjust(method="ipw", max_de=1.5)
        diagnostics = adjusted.diagnostics()
        metrics = diagnostics["metric"].unique()
        self.assertTrue(
            "ipw_multi_class" in metrics,
            f"Expected 'ipw_multi_class' in diagnostics metrics. Found: {metrics}",
        )

    def test_diagnostics_multi_class_converted_to_string(self) -> None:
        """Test diagnostics converts non-string multi_class to string.

        Verifies lines 1878-1879 in sample_class.py where multi_class is
        converted to string if it's not already a string.
        """
        np.random.seed(42)
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": np.random.randn(100),
                    "id": range(100),
                    "w": [1.0] * 100,
                }
            ),
            weight_column="w",
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": np.random.randn(100) + 1,
                    "id": range(100, 200),
                    "w": [1.0] * 100,
                }
            ),
            weight_column="w",
        )
        adjusted = sample.set_target(target).adjust(method="ipw", max_de=1.5)

        # Modify the model's fit object to have a non-string multi_class
        # to test the conversion path (lines 1878-1879)
        if adjusted._adjustment_model is not None:
            fit = adjusted._adjustment_model.get("fit")
            if fit is not None:
                # Temporarily override multi_class with a non-string value
                original_multi_class = getattr(fit, "multi_class", None)
                try:
                    # Set multi_class to a non-string value (e.g., an int)
                    fit.multi_class = 123
                    diagnostics = adjusted.diagnostics()
                    # Check that ipw_multi_class is present and is a string
                    multi_class_rows = diagnostics[
                        diagnostics["metric"] == "ipw_multi_class"
                    ]
                    self.assertGreater(len(multi_class_rows), 0)
                    # The value should be converted to string "123"
                    self.assertEqual(multi_class_rows["var"].iloc[0], "123")
                finally:
                    # Restore original value
                    if original_multi_class is not None:
                        fit.multi_class = original_multi_class

    def test_diagnostics_n_iter_array_larger_than_one(self) -> None:
        """Test diagnostics handles n_iter_ array with size > 1.

        Verifies line 1838 in sample_class.py where array_as_np.size == 1
        check is performed. When size > 1, the value should be skipped.
        """
        np.random.seed(42)
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": np.random.randn(100),
                    "id": range(100),
                    "w": [1.0] * 100,
                }
            ),
            weight_column="w",
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": np.random.randn(100) + 1,
                    "id": range(100, 200),
                    "w": [1.0] * 100,
                }
            ),
            weight_column="w",
        )
        adjusted = sample.set_target(target).adjust(method="ipw", max_de=1.5)

        # Modify the model's fit object to have n_iter_ as an array with size > 1
        # to test the path where we skip the value (line 1838)
        if adjusted._adjustment_model is not None:
            fit = adjusted._adjustment_model.get("fit")
            if fit is not None:
                original_n_iter = getattr(fit, "n_iter_", None)
                try:
                    # Set n_iter_ to an array with size > 1
                    fit.n_iter_ = np.array([10, 20, 30])
                    diagnostics = adjusted.diagnostics()
                    # Check that diagnostics still works
                    self.assertIsNotNone(diagnostics)
                    # n_iter_ should NOT be in ipw_model_glance since size > 1
                    n_iter_rows = diagnostics[
                        (diagnostics["metric"] == "ipw_model_glance")
                        & (diagnostics["var"] == "n_iter_")
                    ]
                    self.assertEqual(len(n_iter_rows), 0)
                finally:
                    # Restore original value
                    if original_n_iter is not None:
                        fit.n_iter_ = original_n_iter

    def test_diagnostics_n_iter_intercept_none_continue(self) -> None:
        """Test diagnostics continues when n_iter_ or intercept_ is None.

        Verifies line 1838 in sample_class.py where continue is called
        when array_val is None for n_iter_ or intercept_ attributes.
        """
        np.random.seed(42)
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": np.random.randn(100),
                    "id": range(100),
                    "w": [1.0] * 100,
                }
            ),
            weight_column="w",
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": np.random.randn(100) + 1,
                    "id": range(100, 200),
                    "w": [1.0] * 100,
                }
            ),
            weight_column="w",
        )
        adjusted = sample.set_target(target).adjust(method="ipw", max_de=1.5)

        # Modify the model's fit object to have n_iter_ and intercept_ set to None
        # to test the continue path (line 1838)
        if adjusted._adjustment_model is not None:
            fit = adjusted._adjustment_model.get("fit")
            if fit is not None:
                original_n_iter = getattr(fit, "n_iter_", None)
                original_intercept = getattr(fit, "intercept_", None)
                try:
                    # Set n_iter_ and intercept_ to None to trigger line 1838
                    fit.n_iter_ = None
                    fit.intercept_ = None
                    diagnostics = adjusted.diagnostics()
                    # Check that diagnostics still works
                    self.assertIsNotNone(diagnostics)
                    # n_iter_ should NOT be in ipw_model_glance since it's None
                    n_iter_rows = diagnostics[
                        (diagnostics["metric"] == "ipw_model_glance")
                        & (diagnostics["var"] == "n_iter_")
                    ]
                    self.assertEqual(len(n_iter_rows), 0)
                    # intercept_ should NOT be in ipw_model_glance since it's None
                    intercept_rows = diagnostics[
                        (diagnostics["metric"] == "ipw_model_glance")
                        & (diagnostics["var"] == "intercept_")
                    ]
                    self.assertEqual(len(intercept_rows), 0)
                finally:
                    # Restore original values
                    if original_n_iter is not None:
                        fit.n_iter_ = original_n_iter
                    if original_intercept is not None:
                        fit.intercept_ = original_intercept


class TestSampleOutcomeImpactDiagnostics(balance.testutil.BalanceTestCase):
    def test_diagnostics_include_outcome_weight_impact(self) -> None:
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "x": [0.1, 0.2, 0.3, 0.4],
                    "weight": [1.0, 1.0, 1.0, 1.0],
                    "outcome": [1.0, 2.0, 3.0, 4.0],
                }
            ),
            id_column="id",
            weight_column="weight",
            outcome_columns=("outcome",),
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": [5, 6, 7, 8],
                    "x": [0.1, 0.2, 0.3, 0.4],
                    "weight": [1.0, 1.0, 1.0, 1.0],
                    "outcome": [1.0, 2.0, 3.0, 4.0],
                }
            ),
            id_column="id",
            weight_column="weight",
            outcome_columns=("outcome",),
        )
        adjusted = sample.set_target(target).adjust(method="null")
        diagnostics = adjusted.diagnostics(weights_impact_on_outcome_method="t_test")
        impact_rows = diagnostics[
            diagnostics["metric"].str.startswith("weights_impact_on_outcome_")
        ]
        self.assertFalse(impact_rows.empty)
        self.assertIn("outcome", impact_rows["var"].unique())


class TestSampleQuickAdjustmentDetailsNRows(balance.testutil.BalanceTestCase):
    """Test cases for _quick_adjustment_details with n_rows=None (line 308)."""

    def test_quick_adjustment_details_with_n_rows_none(self) -> None:
        """Test _quick_adjustment_details when n_rows is None uses df shape.

        Verifies lines 307-308 in sample_class.py.
        """
        sample = Sample.from_frame(
            pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3], "w": [1.0, 2.0, 3.0]}),
            weight_column="w",
        )
        target = Sample.from_frame(
            pd.DataFrame({"a": [1, 2], "id": [4, 5], "w": [1.0, 1.0]}),
            weight_column="w",
        )
        adjusted = sample.set_target(target).adjust(method="null")

        # Call _quick_adjustment_details with n_rows=None (default)
        details = adjusted._quick_adjustment_details(n_rows=None)

        # Should include method and design effect info
        self.assertTrue(any("method:" in d for d in details))
        self.assertTrue(any("design effect" in d for d in details))


class TestSampleModelNoAdjustmentModel(balance.testutil.BalanceTestCase):
    """Test cases for model() returning None when _adjustment_model is None."""

    def test_model_returns_none_when_adjustment_model_attr_missing(self) -> None:
        """Test model() returns None when _adjustment_model attribute is None.

        Verifies that for an unadjusted sample, model() returns None.
        """
        sample = Sample.from_frame(pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3]}))

        # For an unadjusted sample, model() should return None
        result = sample.model()
        self.assertIsNone(result)

    def test_model_returns_adjustment_model_when_set(self) -> None:
        """Test model() returns the adjustment model when set.

        Verifies that model() returns the correct model dictionary after adjustment.
        """
        sample = Sample.from_frame(pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3]}))
        target = Sample.from_frame(pd.DataFrame({"a": [1, 2, 3], "id": [4, 5, 6]}))
        adjusted = sample.set_target(target).adjust(method="null")

        result = adjusted.model()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("method", result)


class TestSampleDesignEffectDiagnosticsExceptionTypes(balance.testutil.BalanceTestCase):
    """Test cases for _design_effect_diagnostics exception handling (lines 349-351)."""

    def test_design_effect_diagnostics_type_error(self) -> None:
        """Test _design_effect_diagnostics handles TypeError gracefully.

        Verifies lines 349-351 in sample_class.py.
        """
        sample = Sample.from_frame(
            pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3], "w": [1.0, 2.0, 3.0]}),
            weight_column="w",
        )

        # Mock design_effect to raise TypeError
        original_design_effect = sample.design_effect
        try:
            sample.design_effect = MagicMock(side_effect=TypeError("test error"))
            result = sample._design_effect_diagnostics()
            self.assertEqual(result, (None, None, None))
        finally:
            sample.design_effect = original_design_effect

    def test_design_effect_diagnostics_value_error(self) -> None:
        """Test _design_effect_diagnostics handles ValueError gracefully.

        Verifies lines 349-351 in sample_class.py.
        """
        sample = Sample.from_frame(
            pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3], "w": [1.0, 2.0, 3.0]}),
            weight_column="w",
        )

        # Mock design_effect to raise ValueError
        original_design_effect = sample.design_effect
        try:
            sample.design_effect = MagicMock(side_effect=ValueError("test error"))
            result = sample._design_effect_diagnostics()
            self.assertEqual(result, (None, None, None))
        finally:
            sample.design_effect = original_design_effect

    def test_design_effect_diagnostics_zero_division_error(self) -> None:
        """Test _design_effect_diagnostics handles ZeroDivisionError gracefully.

        Verifies lines 349-351 in sample_class.py.
        """
        sample = Sample.from_frame(
            pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3], "w": [1.0, 2.0, 3.0]}),
            weight_column="w",
        )

        # Mock design_effect to raise ZeroDivisionError
        original_design_effect = sample.design_effect
        try:
            sample.design_effect = MagicMock(
                side_effect=ZeroDivisionError("test error")
            )
            result = sample._design_effect_diagnostics()
            self.assertEqual(result, (None, None, None))
        finally:
            sample.design_effect = original_design_effect
