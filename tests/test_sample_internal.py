# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for Sample class internal/private API methods.

These tests exercise underscore-prefixed methods and internal implementation
details of the Sample class. They are separated from public API tests to make
it clear which tests can be removed when Sample's internals are replaced by
SampleFrame/BalanceFrame.
"""

# pyre-strict

from __future__ import absolute_import, division, print_function, unicode_literals

from unittest.mock import MagicMock

import balance.testutil
import pandas as pd
from balance.sample_class import Sample


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
