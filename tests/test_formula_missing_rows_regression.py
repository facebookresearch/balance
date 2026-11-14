# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from balance.util import _assert_type
from balance.utils.model_matrix import (
    build_design_matrix,
    build_model_matrix,
    model_matrix,
)
from patsy import PatsyError
from scipy.sparse import csc_matrix
from sklearn.preprocessing import StandardScaler


class FormulaMissingRowsRegressionTest(unittest.TestCase):
    def test_formula_drops_rows_with_debug_warning_without_mixing_populations(
        self,
    ) -> None:
        sample = pd.DataFrame({"x": [-1.0, 4.0]}, index=pd.Index(["bad", "sample"]))
        target = pd.DataFrame(
            {"x": [9.0, 16.0]}, index=pd.Index(["target1", "target2"])
        )
        with self.assertLogs("balance.utils", level="WARNING") as logs:
            result = model_matrix(sample, target, formula="I(x ** (1/2))")
        np.testing.assert_allclose(result["sample"], [[2.0]])
        np.testing.assert_allclose(result["target"], [[3.0], [4.0]])
        np.testing.assert_array_equal(result["sample_rows"], [1])
        np.testing.assert_array_equal(result["target_rows"], [0, 1])
        warning = " ".join(logs.output)
        for detail in (
            "1 of 4 rows",
            "I(x ** (1/2))",
            "bad",
            "position",
            "x",
            "-1.0",
            "evaluated_values",
        ):
            self.assertIn(detail, warning)

    def test_multiple_formulas_use_common_rows_in_every_output_type(self) -> None:
        sample = pd.DataFrame({"x": [-1.0, 4.0]}, index=pd.Index([7, 7]))
        target = pd.DataFrame({"x": [9.0, 16.0]}, index=pd.Index([7, 7]))
        for output_type in ("dataframe", "matrix", "sparse"):
            with self.subTest(output_type=output_type):
                with self.assertLogs("balance.utils", level="WARNING"):
                    result = model_matrix(
                        sample,
                        target,
                        formula=["I(x ** (1/2))", "I((10-x) ** (1/2))"],
                        return_type="one",
                        return_var_type=output_type,
                    )
                actual = result["model_matrix"]
                if output_type == "sparse":
                    actual = _assert_type(actual, csc_matrix).toarray()
                np.testing.assert_allclose(actual, [[2.0, np.sqrt(6)], [3.0, 1.0]])
                np.testing.assert_array_equal(result["sample_rows"], [1])
                np.testing.assert_array_equal(result["target_rows"], [0])
                self.assertEqual(
                    _assert_type(result["penalty_factor"], np.ndarray).tolist(), [1, 1]
                )

    def test_duplicate_indexes_and_formula_index_values_are_preserved(self) -> None:
        frame = pd.DataFrame(
            {"x": [-1.0, 4.0, -9.0, 16.0]}, index=pd.Index([7, 7, 2, 7])
        )
        original = frame.copy(deep=True)
        with self.assertLogs("balance.utils", level="WARNING"):
            result = build_model_matrix(
                frame, formula="I(x ** (1/2) + x.index.to_numpy())"
            )
        actual = result["model_matrix"]
        np.testing.assert_allclose(actual, [[9.0], [11.0]])
        pd.testing.assert_index_equal(actual.index, pd.Index([7, 7]))
        np.testing.assert_array_equal(result["row_positions"], [1, 3])
        pd.testing.assert_frame_equal(frame, original)

    def test_raw_and_formula_drops_compose_original_row_positions(self) -> None:
        sample = pd.DataFrame({"x": [np.nan, -1.0, 4.0, 9.0]})
        target = pd.DataFrame({"x": [np.nan, 4.0, -1.0, 9.0]})
        with self.assertLogs("balance.utils", level="WARNING") as logs:
            result = model_matrix(sample, target, formula="I(x ** (1/2))", add_na=False)
        np.testing.assert_allclose(result["sample"], [[2.0], [3.0]])
        np.testing.assert_allclose(result["target"], [[2.0], [3.0]])
        np.testing.assert_array_equal(result["sample_rows"], [2, 3])
        np.testing.assert_array_equal(result["target_rows"], [1, 3])
        warning = " ".join(logs.output)
        self.assertIn("'population': 'sample', 'original_position': 1", warning)
        self.assertIn("'population': 'target', 'original_position': 2", warning)

    def test_all_missing_formula_rows_drop_with_bounded_debug_examples(self) -> None:
        frame = pd.DataFrame(
            {"x": [-1.0] * 100}, index=pd.Index([f"bad-{i}" for i in range(100)])
        )
        with self.assertLogs("balance.utils", level="WARNING") as logs:
            result = build_model_matrix(frame, formula="I(x ** (1/2))")
        self.assertEqual(result["model_matrix"].shape, (0, 1))
        np.testing.assert_array_equal(result["row_positions"], [])
        warning = " ".join(logs.output)
        for detail in (
            "100 of 100 rows",
            "first 5",
            "bad-0",
            "bad-4",
            "-1.0",
            "I(x ** (1/2))",
        ):
            self.assertIn(detail, warning)
        self.assertNotIn("bad-5", warning)
        self.assertLess(len(warning), 5000)

    def test_missing_categorical_formula_rows_drop_without_becoming_a_level(
        self,
    ) -> None:
        frame = pd.DataFrame({"group": ["a", "b", "a"], "x": [-1, 1, 1]})
        with self.assertLogs("balance.utils", level="WARNING"):
            result = build_model_matrix(frame, formula="I(group.where(x > 0))")
        np.testing.assert_array_equal(result["row_positions"], [1, 2])
        np.testing.assert_allclose(result["model_matrix"], [[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(
            result["model_matrix_columns"],
            ["I(group.where(x > 0))[a]", "I(group.where(x > 0))[b]"],
        )

    def test_design_matrix_scaler_uses_the_retained_weights(self) -> None:
        sample = pd.DataFrame({"x": [-1.0, 4.0, 9.0]})
        target = pd.DataFrame({"x": [4.0, -1.0, 9.0]})
        with self.assertLogs("balance.utils", level="WARNING"):
            result = build_design_matrix(
                sample,
                target,
                formula="I(x ** (1/2))",
                scaler_weights=np.array([200.0, 3.0, 5.0, 7.0, 2000.0, 11.0]),
            )
        reference = StandardScaler(with_mean=False).fit(
            np.array([[2.0], [3.0], [2.0], [3.0]]),
            sample_weight=np.array([3.0, 5.0, 7.0, 11.0]),
        )
        np.testing.assert_allclose(result["fit_scaler"].mean_, reference.mean_)
        np.testing.assert_allclose(result["fit_scaler"].scale_, reference.scale_)
        self.assertEqual(result["sample_n"], 2)
        np.testing.assert_array_equal(result["sample_rows"], [1, 2])
        np.testing.assert_array_equal(result["target_rows"], [0, 2])

    def test_post_drop_scaler_weight_mass_loss_is_an_actionable_fit_error(self) -> None:
        sample = pd.DataFrame({"x": [-1.0, 4.0]})
        target = pd.DataFrame({"x": [-1.0, 9.0]})
        with self.assertLogs("balance.utils", level="WARNING"):
            with self.assertRaisesRegex(
                ValueError, "no finite positive scaler weight mass"
            ):
                build_design_matrix(
                    sample,
                    target,
                    formula="I(x ** (1/2))",
                    scaler_weights=np.array([100.0, 0.0, 100.0, 0.0]),
                )

    def test_valid_formulas_continue_and_unrelated_errors_still_raise(self) -> None:
        frame = pd.DataFrame({"x": [1.0, 4.0]})
        with patch("balance.utils.model_matrix.logger.warning") as warning:
            result = build_model_matrix(frame, formula="I(x ** (1/2))")
        np.testing.assert_allclose(result["model_matrix"], [[1.0], [2.0]])
        warning.assert_not_called()
        with self.assertRaises(PatsyError):
            build_model_matrix(frame, formula="unknown(x)")
