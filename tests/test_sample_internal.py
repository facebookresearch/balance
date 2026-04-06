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

from copy import deepcopy
from typing import Any

import balance.testutil
import pandas as pd
from balance.sample_class import Sample
from balance.sample_frame import SampleFrame


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

    def test_Sample__require_adjusted(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"is not adjusted\. Use \.adjust\(\)",
        ):
            s1._require_adjusted()
        with self.assertRaisesRegex(
            ValueError,
            r"is not adjusted\. Use \.adjust\(\)",
        ):
            s3._require_adjusted()
        self.assertTrue(
            s3_adjusted_null._require_adjusted() is None
        )  # Does not raise an error

    def test_Sample__require_target(self) -> None:
        # test exception when there is no target
        with self.assertRaisesRegex(
            ValueError,
            r"does not have a target set\. Use \.set_target\(\)",
        ):
            s1._require_target()
        s3._require_target()  # Should not raise an error

    def test_Sample__require_outcomes(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"does not have outcome columns specified",
        ):
            s2._require_outcomes()
        self.assertTrue(s1._require_outcomes() is None)  # Does not raise an error


class TestSampleDesignEffectDiagnostics(balance.testutil.BalanceTestCase):
    """Test cases for _design_effect_diagnostics edge cases (lines 342-351)."""

    def test_design_effect_diagnostics_with_uniform_weights(self) -> None:
        """Test _design_effect_diagnostics with uniform weights returns Deff=1.

        When all weights are equal, the design effect should be 1.0.
        """
        sample = Sample.from_frame(pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3]}))
        result = sample._design_effect_diagnostics()
        design_effect: float | None = result[0]
        assert design_effect is not None
        self.assertAlmostEqual(design_effect, 1.0)

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
        result = sample.model
        self.assertIsNone(result)

    def test_model_returns_adjustment_model_when_set(self) -> None:
        """Test model() returns the adjustment model when set.

        Verifies that model() returns the correct model dictionary after adjustment.
        """
        sample = Sample.from_frame(pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3]}))
        target = Sample.from_frame(pd.DataFrame({"a": [1, 2, 3], "id": [4, 5, 6]}))
        adjusted = sample.set_target(target).adjust(method="null")

        result = adjusted.model
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("method", result)


class TestSampleDesignEffectDiagnosticsExceptionTypes(balance.testutil.BalanceTestCase):
    """Test cases for _design_effect_diagnostics exception handling."""

    def test_design_effect_diagnostics_type_error(self) -> None:
        """Test _design_effect_diagnostics handles TypeError gracefully."""
        from unittest.mock import patch

        sample = Sample.from_frame(
            pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3], "w": [1.0, 2.0, 3.0]}),
            weight_column="w",
        )
        with patch(
            "balance.stats_and_plots.weights_stats.design_effect",
            side_effect=TypeError("test error"),
        ):
            result = sample._design_effect_diagnostics()
            self.assertEqual(result, (None, None, None))

    def test_design_effect_diagnostics_value_error(self) -> None:
        """Test _design_effect_diagnostics handles ValueError gracefully."""
        from unittest.mock import patch

        sample = Sample.from_frame(
            pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3], "w": [1.0, 2.0, 3.0]}),
            weight_column="w",
        )
        with patch(
            "balance.stats_and_plots.weights_stats.design_effect",
            side_effect=ValueError("test error"),
        ):
            result = sample._design_effect_diagnostics()
            self.assertEqual(result, (None, None, None))

    def test_design_effect_diagnostics_zero_division_error(self) -> None:
        """Test _design_effect_diagnostics handles ZeroDivisionError gracefully."""
        from unittest.mock import patch

        sample = Sample.from_frame(
            pd.DataFrame({"a": [1, 2, 3], "id": [1, 2, 3], "w": [1.0, 2.0, 3.0]}),
            weight_column="w",
        )
        with patch(
            "balance.stats_and_plots.weights_stats.design_effect",
            side_effect=ZeroDivisionError("test error"),
        ):
            result = sample._design_effect_diagnostics()
            self.assertEqual(result, (None, None, None))


class TestSampleInternalSampleFrame(
    balance.testutil.BalanceTestCase,
):
    """Tests for the internal SampleFrame backing of Sample."""

    def _make_sample(
        self,
        outcome: bool = True,
        ignore: bool = False,
    ) -> Sample:
        data: dict[str, Any] = {
            "id": ["1", "2", "3"],
            "x": [10.0, 20.0, 30.0],
            "y": [1.0, 2.0, 3.0],
            "weight": [1.0, 1.0, 1.0],
        }
        if ignore:
            data["misc_col"] = ["a", "b", "c"]
        return Sample.from_frame(
            pd.DataFrame(data),
            id_column="id",
            weight_column="weight",
            outcome_columns="y" if outcome else None,
            ignored_columns="misc_col" if ignore else None,
            standardize_types=False,
        )

    def test_sample_has_sample_frame(self) -> None:
        s = self._make_sample()
        self.assertIsNotNone(s._sample_frame)
        self.assertIsInstance(s._sample_frame, SampleFrame)

    def test_df_property_returns_sample_frame_df(self) -> None:
        s = self._make_sample()
        # _df should be the same object as _sample_frame._df
        self.assertIs(s._df, s._sample_frame._df)

    def test_df_setter_updates_sample_frame(self) -> None:
        s = self._make_sample()
        new_df = s._df.copy()
        new_df["x"] = [100.0, 200.0, 300.0]
        s._df = new_df
        pd.testing.assert_frame_equal(s._sample_frame._df, new_df)

    def test_outcome_columns_property(self) -> None:
        s = self._make_sample(outcome=True)
        outcome_cols = s._outcome_columns
        self.assertIsNotNone(outcome_cols)
        assert outcome_cols is not None
        self.assertEqual(outcome_cols.columns.tolist(), ["y"])
        pd.testing.assert_series_equal(
            outcome_cols["y"],
            pd.Series([1.0, 2.0, 3.0], name="y"),
            check_names=False,
        )

    def test_outcome_columns_none_when_no_outcomes(self) -> None:
        s = self._make_sample(outcome=False)
        self.assertIsNone(s._outcome_columns)

    def test_outcome_columns_setter(self) -> None:
        s = self._make_sample(outcome=True)
        # Setting to None clears outcomes
        s._outcome_columns = None
        self.assertIsNone(s._outcome_columns)
        self.assertEqual(s._sample_frame._column_roles["outcomes"], [])

    def test_ignored_column_names_property(self) -> None:
        s = self._make_sample(ignore=True)
        self.assertEqual(s._ignored_column_names, ["misc_col"])

    def test_ignored_column_names_empty_by_default(self) -> None:
        s = self._make_sample(ignore=False)
        self.assertEqual(s._ignored_column_names, [])

    def test_ignored_column_names_setter(self) -> None:
        s = self._make_sample(ignore=True)
        s._ignored_column_names = ["x"]
        self.assertEqual(s._sample_frame._column_roles["ignored"], ["x"])

    def test_covar_columns_inferred_correctly(self) -> None:
        s = self._make_sample(outcome=True, ignore=True)
        # covars = all columns minus id, weight, outcome, ignored
        self.assertEqual(s._covar_columns_names(), ["x"])

    def test_from_frame_builds_sample_frame_with_correct_roles(self) -> None:
        s = self._make_sample(outcome=True, ignore=True)
        sf = s._sample_frame
        self.assertEqual(sf._id_column_name, "id")
        self.assertEqual(sf._column_roles["weights"], ["weight"])
        self.assertEqual(sf._column_roles["outcomes"], ["y"])
        self.assertEqual(sf._column_roles["ignored"], ["misc_col"])
        self.assertEqual(sf._column_roles["covars"], ["x"])

    def test_set_target_preserves_sample_frame(self) -> None:
        s = self._make_sample()
        t = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": ["4", "5", "6"],
                    "x": [40.0, 50.0, 60.0],
                    "y": [4.0, 5.0, 6.0],
                    "weight": [1.0, 1.0, 1.0],
                }
            ),
            id_column="id",
            weight_column="weight",
            outcome_columns="y",
            standardize_types=False,
        )
        s_with_target = s.set_target(t)
        self.assertIsNotNone(s_with_target._sample_frame)
        # Target should also have its own SampleFrame
        target = s_with_target._links["target"]
        self.assertIsNotNone(target._sample_frame)

    def test_unadjusted_has_no_weight_metadata(self) -> None:
        s = self._make_sample()
        self.assertEqual(s._sample_frame.weight_metadata(), {})

    def test_adjust_records_weight_metadata(self) -> None:
        s = self._make_sample()
        t = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": ["4", "5", "6"],
                    "x": [40.0, 50.0, 60.0],
                    "weight": [1.0, 1.0, 1.0],
                }
            ),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        s_with_target = s.set_target(t)
        adjusted = s_with_target.adjust(method="null")
        meta = adjusted._sample_frame.weight_metadata()
        self.assertEqual(meta["method"], "null")
        self.assertTrue(meta["adjusted"])

    def test_adjust_ipw_records_weight_metadata(self) -> None:
        s = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": [str(i) for i in range(20)],
                    "x": [float(i) for i in range(20)],
                    "weight": [1.0] * 20,
                }
            ),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        t = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": [str(i) for i in range(20, 40)],
                    "x": [float(i) for i in range(20, 40)],
                    "weight": [1.0] * 20,
                }
            ),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        s_with_target = s.set_target(t)
        adjusted = s_with_target.adjust(method="ipw")
        meta = adjusted._sample_frame.weight_metadata()
        self.assertEqual(meta["method"], "ipw")
        self.assertTrue(meta["adjusted"])

    def test_adjust_callable_records_weight_metadata(self) -> None:
        def my_custom_method(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "weight": pd.Series([1.0, 1.0, 1.0]),
                "model": {"method": "custom"},
            }

        s = self._make_sample()
        t = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": ["4", "5", "6"],
                    "x": [40.0, 50.0, 60.0],
                    "weight": [1.0, 1.0, 1.0],
                }
            ),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        s_with_target = s.set_target(t)
        adjusted = s_with_target.adjust(method=my_custom_method)
        meta = adjusted._sample_frame.weight_metadata()
        self.assertEqual(meta["method"], "my_custom_method")
        self.assertTrue(meta["adjusted"])

    def test_set_weights_syncs_sample_frame(self) -> None:
        s = self._make_sample()
        new_weights = pd.Series([2.0, 3.0, 4.0], index=s._df.index)
        s.set_weights(new_weights)
        # Both the plain weight_column attr and the SampleFrame's _df
        # should reflect the new weights.
        pd.testing.assert_series_equal(
            s.weight_series,
            s._sample_frame._df["weight"],
        )
        self.assertEqual(s._sample_frame._df["weight"].tolist(), [2.0, 3.0, 4.0])

    def test_keep_only_some_rows_columns_preserves_outcomes(self) -> None:
        s = self._make_sample(outcome=True)
        # Keep only column "x" — outcome column "y" should be preserved
        filtered = s.keep_only_some_rows_columns(columns_to_keep=["id", "x", "weight"])
        self.assertIn("y", filtered._df.columns.tolist())

    def test_deepcopy_preserves_sample_frame(self) -> None:
        s = self._make_sample()
        s2 = deepcopy(s)
        self.assertIsNotNone(s2._sample_frame)
        # Modifying the copy should not affect the original
        s2._df["x"] = [100.0, 200.0, 300.0]
        self.assertEqual(s._df["x"].tolist(), [10.0, 20.0, 30.0])
