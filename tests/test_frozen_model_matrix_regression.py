# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pickle
import unittest

import numpy as np
import pandas as pd
from balance import Sample
from balance.outcome_models import fit_outcome_model, predict_outcome
from balance.util import _assert_type
from balance.utils.model_matrix import build_design_matrix, model_matrix
from sklearn.linear_model import LinearRegression


class FrozenModelMatrixRegressionTest(unittest.TestCase):
    def test_drop_path_keeps_numeric_categorical_levels_for_fit_and_replay(
        self,
    ) -> None:
        training = pd.DataFrame(
            {
                "g": pd.Categorical([1, 2, 3], categories=[1, 2, 3]),
                "x": [10.0, 20.0, None],
            },
            index=pd.Index([10, 20, 30]),
        )
        fitted = model_matrix(training, formula="g", add_na=False)
        fitted_sample = _assert_type(fitted["sample"], pd.DataFrame)
        self.assertEqual(fitted_sample.index.tolist(), [10, 20])
        self.assertEqual(
            fitted_sample.values.tolist(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        scoring = pd.DataFrame(
            {"g": pd.Categorical([3, 1], categories=[1, 3]), "x": [30.0, 10.0]},
            index=pd.Index([30, 10]),
        )
        replay = model_matrix(
            scoring,
            add_na=False,
            preprocessing_state=_assert_type(fitted["preprocessing_state"], dict),
        )
        replay_sample = _assert_type(replay["sample"], pd.DataFrame)
        self.assertEqual(replay_sample.index.tolist(), [30, 10])
        self.assertEqual(
            replay["model_matrix_columns_names"], fitted["model_matrix_columns_names"]
        )
        self.assertEqual(
            replay_sample.values.tolist(), [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        )

    def test_numeric_object_comparisons_keep_their_value_types(self) -> None:
        frame = pd.DataFrame({"grp": pd.Series([1, 2, 3], dtype=object)})
        formula = "grp + I(grp == 1)"
        train = model_matrix(frame, formula=formula)
        replay = model_matrix(
            frame.iloc[:2],
            formula=formula,
            preprocessing_state=_assert_type(train["preprocessing_state"], dict),
        )
        np.testing.assert_array_equal(
            _assert_type(replay["sample"], pd.DataFrame).values,
            _assert_type(train["sample"], pd.DataFrame).values[:2],
        )

    def test_raw_replay_maps_novel_categories_to_missing(self) -> None:
        training = pd.DataFrame({"grp": pd.Categorical(["a", "b"])})
        fit = build_design_matrix(training, training, use_model_matrix=False)
        for novel in (None, "novel"):
            with self.subTest(novel=novel):
                holdout = pd.DataFrame({"grp": pd.Categorical(["a", novel])})
                replay = build_design_matrix(
                    holdout,
                    training,
                    use_model_matrix=False,
                    project_to_columns=fit["columns"],
                    preprocessing_state=fit["preprocessing_state"],
                )
                self.assertEqual(replay["combined_matrix"]["grp"].iloc[0], "a")
                self.assertTrue(pd.isna(replay["combined_matrix"]["grp"].iloc[1]))

    def test_missingness_and_ordered_levels_survive_replay(self) -> None:
        frame = pd.DataFrame(
            {
                "a": pd.Categorical([2, 10, 1], categories=[2, 10, 1], ordered=True),
                "x": [1.0, 2.0, None],
            },
            index=pd.Index([10, 20, 30]),
        )
        train = build_design_matrix(frame, frame, one_hot_encoding=True)
        restored = pickle.loads(pickle.dumps(train))
        replay = build_design_matrix(
            frame.iloc[:2],
            frame.iloc[:2],
            one_hot_encoding=True,
            preprocessing_state=restored["preprocessing_state"],
            project_to_columns=train["columns"],
        )
        np.testing.assert_array_equal(
            replay["combined_matrix"].toarray()[:2],
            train["combined_matrix"].toarray()[:2],
        )

    def test_replay_validates_formula_and_version(self) -> None:
        frame = pd.DataFrame({"x": [1.0, 2.0]})
        train = model_matrix(frame)
        state = _assert_type(train["preprocessing_state"], dict)
        with self.assertRaisesRegex(ValueError, "formula must match"):
            model_matrix(frame, formula="I(x * 2)", preprocessing_state=state)
        with self.assertRaisesRegex(ValueError, "version"):
            model_matrix(frame, preprocessing_state={**state, "version": 999})

    def test_replays_contrasts_after_pickle_on_changed_level_counts(self) -> None:
        frame = pd.DataFrame({"a": ["a", "b", "c"]}, index=pd.Index([10, 20, 30]))
        train = build_design_matrix(frame, frame, one_hot_encoding=True)
        state = pickle.loads(pickle.dumps(train["preprocessing_state"]))
        holdout = frame.iloc[:2]
        replay = build_design_matrix(
            holdout,
            holdout,
            one_hot_encoding=True,
            formula=train["resolved_formula"],
            project_to_columns=train["columns"],
            preprocessing_state=state,
        )
        np.testing.assert_array_equal(
            replay["combined_matrix"].toarray()[:2],
            train["combined_matrix"].toarray()[:2],
        )
        self.assertEqual(replay["sample_n"], 2)

    def test_model_matrix_exports_reusable_state(self) -> None:
        train = model_matrix(
            pd.DataFrame({"a": ["a", "b", "c"]}), one_hot_encoding=True
        )
        state = _assert_type(
            pickle.loads(pickle.dumps(train["preprocessing_state"])), dict
        )
        replay = model_matrix(
            pd.DataFrame({"a": ["a", "b"]}),
            one_hot_encoding=True,
            preprocessing_state=state,
        )
        np.testing.assert_array_equal(
            _assert_type(replay["sample"], pd.DataFrame).values,
            _assert_type(train["sample"], pd.DataFrame).values[:2],
        )

    def test_rejects_unseen_categories(self) -> None:
        frame = pd.DataFrame({"a": ["a", "b", "c"]})
        train = build_design_matrix(frame, frame, one_hot_encoding=True)
        with self.assertRaisesRegex(ValueError, "unseen categor"):
            build_design_matrix(
                pd.DataFrame({"a": ["d"]}),
                frame,
                one_hot_encoding=True,
                project_to_columns=train["columns"],
                preprocessing_state=train["preprocessing_state"],
            )

    def test_stateful_training_is_supported_but_replay_is_rejected(self) -> None:
        sample = pd.DataFrame({"x": [1.0, 2.0]})
        target = pd.DataFrame({"x": [3.0, 4.0]})
        train = build_design_matrix(sample, target, formula="center(x)")
        np.testing.assert_array_equal(
            train["combined_matrix"].toarray().ravel(), [-1.5, -0.5, 0.5, 1.5]
        )
        state = pickle.loads(pickle.dumps(train["preprocessing_state"]))
        with self.assertRaisesRegex(ValueError, "stateful"):
            build_design_matrix(
                sample,
                target,
                formula="center(x)",
                project_to_columns=train["columns"],
                preprocessing_state=state,
            )

    def test_legacy_categorical_replay_requires_frozen_state(self) -> None:
        frame = pd.DataFrame({"a": ["a", "b", "c"]})
        train = build_design_matrix(frame, frame, one_hot_encoding=True)
        with self.assertRaisesRegex(ValueError, "preprocessing_state"):
            build_design_matrix(
                frame.iloc[:2],
                frame.iloc[:2],
                one_hot_encoding=True,
                project_to_columns=train["columns"],
                formula=train["resolved_formula"],
            )

    def test_bool_arithmetic_keeps_values_during_fit_and_replay(self) -> None:
        frame = pd.DataFrame(
            {"b": [True, False, True, False], "x": [1.0, 2.0, 3.0, 4.0]}
        )
        for one_hot in (False, True):
            with self.subTest(one_hot=one_hot):
                fit = build_design_matrix(
                    frame, frame.iloc[:0], formula="I(b * x)", one_hot_encoding=one_hot
                )
                self.assertEqual(fit["columns"], ["I(b * x)"])
                np.testing.assert_array_equal(
                    fit["combined_matrix"].toarray(), [[1.0], [0.0], [3.0], [0.0]]
                )
                for rows, expected in (
                    ([0, 2], [[1.0], [3.0]]),
                    ([1, 3], [[0.0], [0.0]]),
                ):
                    replay = build_design_matrix(
                        frame.iloc[rows],
                        frame.iloc[:0],
                        formula=fit["resolved_formula"],
                        one_hot_encoding=one_hot,
                        project_to_columns=fit["columns"],
                        preprocessing_state=pickle.loads(
                            pickle.dumps(fit["preprocessing_state"])
                        ),
                    )
                    np.testing.assert_array_equal(
                        replay["combined_matrix"].toarray(), expected
                    )

    def test_bool_inversion_keeps_both_factor_levels_on_replay(self) -> None:
        frame = pd.DataFrame({"b": [True, False, True, False]})
        fit = model_matrix(frame, formula="I(~b)")
        np.testing.assert_array_equal(
            _assert_type(fit["sample"], pd.DataFrame).values,
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        )
        for rows, expected in (
            ([0, 2], [[1.0, 0.0], [1.0, 0.0]]),
            ([1, 3], [[0.0, 1.0], [0.0, 1.0]]),
        ):
            replay = model_matrix(
                frame.iloc[rows],
                preprocessing_state=pickle.loads(
                    pickle.dumps(fit["preprocessing_state"])
                ),
            )
            self.assertEqual(
                fit["model_matrix_columns_names"], replay["model_matrix_columns_names"]
            )
            np.testing.assert_array_equal(
                _assert_type(replay["sample"], pd.DataFrame).values, expected
            )

    def test_object_numeric_ordering_and_multiplication_keep_values_on_replay(
        self,
    ) -> None:
        cases = [
            ([1, 3, 1, 3], "I(g > 1)"),
            ([1, 3, 1, 3], "I(g * 2)"),
            ([1.5, 3.5, 1.5, 3.5], "I(g > 2)"),
            ([1.5, 3.5, 1.5, 3.5], "I(g * 2)"),
        ]
        for values, formula in cases:
            with self.subTest(formula=formula, values=values):
                frame = pd.DataFrame({"g": pd.Series(values, dtype=object)})
                fit = model_matrix(frame, formula=formula)
                np.testing.assert_array_equal(
                    _assert_type(fit["sample"], pd.DataFrame).values,
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                )
                for rows, expected in (
                    ([0, 2], [[1.0, 0.0], [1.0, 0.0]]),
                    ([1, 3], [[0.0, 1.0], [0.0, 1.0]]),
                ):
                    replay = model_matrix(
                        frame.iloc[rows],
                        preprocessing_state=pickle.loads(
                            pickle.dumps(fit["preprocessing_state"])
                        ),
                    )
                    self.assertEqual(
                        fit["model_matrix_columns_names"],
                        replay["model_matrix_columns_names"],
                    )
                    np.testing.assert_array_equal(
                        _assert_type(replay["sample"], pd.DataFrame).values, expected
                    )

    def test_sum_contrasts_with_object_ordering_keep_prediction_values(self) -> None:
        covars = pd.DataFrame({"g": pd.Series([1, 2, 3], dtype=object)})
        fitted = fit_outcome_model(
            covars,
            pd.DataFrame({"y": [2.0, 4.0, 8.0]}),
            model=LinearRegression(),
            transformations=None,
            use_model_matrix=True,
            formula="C(g, Sum) + I(g > 1)",
        )
        restored = pickle.loads(pickle.dumps(fitted))
        np.testing.assert_allclose(
            predict_outcome(restored, covars.iloc[[2, 0]])["y"], [8.0, 2.0], atol=1e-12
        )
        np.testing.assert_allclose(
            predict_outcome(restored, covars.iloc[[1]])["y"], [4.0], atol=1e-12
        )

    def test_public_ipw_accepts_bool_arithmetic_formula(self) -> None:
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": ["1", "2", "3", "4", "5", "6"],
                    "b": [True, False, True, False, True, True],
                    "x": [1.0, 2.0, 3.0, 4.0, 2.0, 1.0],
                }
            )
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": [str(i) for i in range(10, 18)],
                    "b": [False, True, True, False, False, True, False, True],
                    "x": [2.0, 2.0, 5.0, 1.0, 3.0, 4.0, 1.0, 2.0],
                }
            )
        )
        adjusted = sample.set_target(target).adjust(
            method="ipw",
            formula="x + I(b * x)",
            transformations=None,
            num_lambdas=3,
            lambda_min=0.1,
            lambda_max=10.0,
            weight_trimming_mean_ratio=None,
            random_seed=2020,
        )
        reference_sample = sample.df.copy()
        reference_target = target.df.copy()
        for frame in (reference_sample, reference_target):
            frame["b_times_x"] = frame["b"] * frame["x"]
        expected = (
            Sample.from_frame(reference_sample)
            .set_target(Sample.from_frame(reference_target))
            .adjust(
                method="ipw",
                formula="x + b_times_x",
                transformations=None,
                num_lambdas=3,
                lambda_min=0.1,
                lambda_max=10.0,
                weight_trimming_mean_ratio=None,
                random_seed=2020,
            )
        )
        np.testing.assert_allclose(
            adjusted.weight_series, expected.weight_series, rtol=1e-10, atol=1e-12
        )

    def test_numpy_categories_keep_fitted_prediction_values(self) -> None:
        cases = [
            ([np.int64(1), np.int64(2), np.int64(3)], [3, 1], [8.0, 2.0]),
            (
                [np.float64(1.5), np.float64(2.5), np.float64(3.5)],
                [3.5, 1.5],
                [8.0, 2.0],
            ),
            ([np.bool_(False), np.bool_(True)], [True, False], [4.0, 2.0]),
        ]
        for levels, scoring, expected in cases:
            with self.subTest(levels=levels):
                covars = pd.DataFrame({"grp": pd.Series(levels, dtype=object)})
                outcomes = pd.DataFrame({"y": [2.0, 4.0, 8.0][: len(levels)]})
                fitted = fit_outcome_model(
                    covars,
                    outcomes,
                    model=LinearRegression(),
                    transformations=None,
                    use_model_matrix=True,
                    formula="grp",
                )
                restored = pickle.loads(pickle.dumps(fitted))
                actual = predict_outcome(
                    restored,
                    pd.DataFrame({"grp": pd.Series(scoring, dtype=object)}),
                )["y"]
                np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_obsolete_frozen_columns_require_refitting_not_zero_filling(self) -> None:
        covars = pd.DataFrame({"grp": pd.Series([1, 2, 3], dtype=object)})
        outcomes = pd.DataFrame({"y": [2.0, 4.0, 8.0]})
        fitted = fit_outcome_model(
            covars,
            outcomes,
            model=LinearRegression(),
            transformations=None,
            use_model_matrix=True,
            formula="grp",
        )
        fitted["X_matrix_columns"] = [
            "grp[np.int64(1)]",
            "grp[np.int64(2)]",
            "grp[np.int64(3)]",
        ]
        with self.assertRaisesRegex(ValueError, "Fit-time columns.*Refit"):
            predict_outcome(fitted, covars)
