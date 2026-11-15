# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from balance import Sample
from balance.utils.data_transformation import add_na_indicator
from balance.utils.pandas_utils import _safe_fillna_and_infer


class CategoricalNAActionRegressionTest(unittest.TestCase):
    def test_public_indicator_keeps_complete_categorical_columns(self) -> None:
        frame = pd.DataFrame({"a": pd.Categorical(["a", "b"])})
        result = add_na_indicator(frame, replace_val_obj="__NaN__")
        pd.testing.assert_frame_equal(result, frame)

    def test_public_indicator_adds_missing_category_without_mutating_input(
        self,
    ) -> None:
        frame = pd.DataFrame({"a": pd.Categorical(["a", None])})
        original = frame.copy(deep=True)
        result = add_na_indicator(frame, replace_val_obj="__NaN__")
        self.assertEqual(result["a"].tolist(), ["a", "__NaN__"])
        self.assertEqual(result["_is_na_a"].tolist(), [False, True])
        pd.testing.assert_frame_equal(frame, original)

    def test_public_indicator_numeric_fill_preserves_nullable_dtype(self) -> None:
        for dtype in ("Int64", "Float64"):
            with self.subTest(dtype=dtype):
                frame = pd.DataFrame({"a": pd.Series([1, pd.NA], dtype=dtype)})
                original = frame.copy(deep=True)
                result = add_na_indicator(frame)
                self.assertEqual(result["a"].tolist(), [1, 0])
                self.assertEqual(str(result["a"].dtype), dtype)
                self.assertEqual(result["_is_na_a"].tolist(), [False, True])
                pd.testing.assert_frame_equal(frame, original)

    def test_nullable_numeric_fill_uses_object_only_for_string_sentinel(self) -> None:
        for dtype in ("Int64", "Float64"):
            with self.subTest(dtype=dtype):
                series = pd.Series([1, pd.NA], dtype=dtype)
                original = series.copy(deep=True)
                sentinel = _safe_fillna_and_infer(series, "__NaN__")
                self.assertEqual(sentinel.tolist(), [1, "__NaN__"])
                self.assertEqual(sentinel.dtype, object)
                numeric = _safe_fillna_and_infer(series, 0)
                self.assertEqual(numeric.tolist(), [1, 0])
                self.assertEqual(str(numeric.dtype), dtype)
                pd.testing.assert_series_equal(series, original)

    def test_public_nullable_numeric_fit_and_weight_replay(self) -> None:
        for method in ("rake", "poststratify"):
            for dtype in ("Int64", "Float64"):
                with self.subTest(method=method, dtype=dtype):
                    frame = pd.DataFrame(
                        {
                            "id": ["1", "2", "3", "4"],
                            "a": pd.Series([1, pd.NA, 2, 1], dtype=dtype),
                            "b": ["x", "y", "x", "y"],
                            "w": [1.0] * 4,
                        }
                    )
                    sample = Sample.from_frame(
                        frame,
                        id_column="id",
                        weight_column="w",
                        standardize_types=False,
                    )
                    target = Sample.from_frame(
                        frame.assign(id=["5", "6", "7", "8"]),
                        id_column="id",
                        weight_column="w",
                        standardize_types=False,
                    )
                    sample = sample.set_target(target)
                    adjusted = sample.adjust(
                        method=method,
                        transformations=None,
                        na_action="add_indicator",
                        store_fit_metadata=True,
                    )
                    predicted = adjusted.predict_weights(data=sample)
                    np.testing.assert_allclose(predicted.to_numpy(), np.ones(4))

    def test_public_categorical_fit_and_weight_replay(self) -> None:
        for method in ("rake", "poststratify"):
            for values in (["a", "b"], ["a", None]):
                with self.subTest(method=method, values=values):
                    frame = pd.DataFrame(
                        {"id": ["1", "2"], "a": pd.Categorical(values), "w": [1.0, 1.0]}
                    )
                    sample = Sample.from_frame(
                        frame,
                        id_column="id",
                        weight_column="w",
                        standardize_types=False,
                    )
                    target = Sample.from_frame(
                        frame.assign(id=["3", "4"]),
                        id_column="id",
                        weight_column="w",
                        standardize_types=False,
                    )
                    sample = sample.set_target(target)
                    adjusted = sample.adjust(
                        method=method,
                        transformations=None,
                        na_action="add_indicator",
                        store_fit_metadata=True,
                    )
                    predicted = adjusted.predict_weights(data=sample)
                    np.testing.assert_allclose(predicted.to_numpy(), [1.0, 1.0])
