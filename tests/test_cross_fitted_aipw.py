# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-fitted target means on independently chosen synthetic fixtures.

The linear oracle uses NumPy least squares, not balance or sklearn predictions.
These are target-population means, not causal treatment-effect contrasts.
"""

from __future__ import annotations

import logging
import pickle
import unittest
import warnings
from contextlib import contextmanager
from typing import Any, Iterator
from unittest import mock

import balance.testutil
import numpy as np
import pandas as pd
import pytest
from balance import outcome_models
from balance.outcome_models import aipw as aipw_module
from balance.sample_class import Sample
from balance.testutil import _SKLEARN_1_4_AVAILABLE
from balance.utils.input_validation import _assert_type
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    BaggingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = pd.DataFrame(
        {
            "id": [str(i) for i in range(8)],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "y": [1.2, 3.4, 2.1, 7.8, 4.3, 8.9, 6.2, 12.7],
            "weight": [1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0],
        }
    )
    target = pd.DataFrame(
        {
            "id": ["10", "11", "12", "13"],
            "x": [0.5, 2.5, 4.5, 7.5],
            "weight": [2.0, 3.0, 4.0, 3.0],
        }
    )
    return sample, target


def _keep_weights(sample_df, sample_weights, target_df, target_weights):
    del sample_df, target_df
    return {
        "weight": sample_weights * target_weights.sum() / sample_weights.sum(),
        "model": {"method": "test_calibration"},
    }


def _frame(sample: pd.DataFrame, target: pd.DataFrame, **fit_kwargs) -> Sample:
    frame = Sample.from_frame(
        sample, id_column="id", weight_column="weight", outcome_columns=["y"]
    ).set_target(Sample.from_frame(target, id_column="id", weight_column="weight"))
    return frame.adjust(method=_keep_weights).fit_outcome_model(**fit_kwargs)


def _pure(sample, target, *, fit_kwargs=None, **kwargs):
    return aipw_module.cross_fitted_aipw_point_estimate(
        sample[["x"]],
        sample[["y"]],
        sample["weight"],
        target[["x"]],
        target["weight"],
        fit_kwargs={"model": LinearRegression()} if fit_kwargs is None else fit_kwargs,
        **kwargs,
    )


def _linear_oracle(sample, target, *, n_folds=2, random_seed=2020, fit_weight=None):
    """OLS per fold; omit missing Y and formula-dropped X by row position."""
    x = np.column_stack([np.ones(len(sample)), sample["x"].to_numpy()])
    xt = np.column_stack([np.ones(len(target)), target["x"].to_numpy()])
    y = sample["y"].to_numpy(dtype=float)
    observed = np.flatnonzero(~np.isnan(y))
    folds = np.array_split(
        np.random.RandomState(random_seed).permutation(observed), n_folds
    )
    oof = np.full(len(sample), np.nan)
    target_predictions = []
    for held_out in folds:
        train = np.setdiff1d(observed, held_out)
        train = train[np.isfinite(x[train]).all(axis=1)]
        root_weight = (
            np.ones(len(train))
            if fit_weight is None
            else np.sqrt(np.asarray(fit_weight)[train])
        )
        beta = np.linalg.lstsq(
            x[train] * root_weight[:, None], y[train] * root_weight, rcond=None
        )[0]
        oof[held_out] = x[held_out] @ beta
        target_predictions.append(xt @ beta)
    target_prediction = np.mean(target_predictions, axis=0)
    valid_target = np.isfinite(target_prediction)
    target_mean = np.average(
        target_prediction[valid_target], weights=target["weight"][valid_target]
    )
    valid_residual = np.isfinite(y - oof)
    residual_mean = np.average(
        (y - oof)[valid_residual],
        weights=sample["weight"].to_numpy()[valid_residual],
    )
    return float(target_mean + residual_mean)


@contextmanager
def _capture_fold_models() -> Iterator[list[dict[str, Any]]]:
    """Observe real fold fits without replacing their training or predictions."""
    models: list[dict[str, Any]] = []
    original_fit = aipw_module.fit_outcome_model

    def record_fit(*args, **kwargs) -> dict[str, Any]:
        fitted = original_fit(*args, **kwargs)
        models.append(fitted)
        return fitted

    with mock.patch.object(aipw_module, "fit_outcome_model", side_effect=record_fit):
        yield models


class CrossFittedAipwTest(balance.testutil.BalanceTestCase):
    def test_frame_opt_in_matches_independent_linear_oracle(self) -> None:
        sample, target = _fixture()
        frame = _frame(sample, target, model=LinearRegression())

        result = frame.aipw(n_folds=2, random_seed=2020)

        self.assertAlmostEqual(
            float(result["y"]), _linear_oracle(sample, target), places=10
        )
        self.assertEqual(result.index.tolist(), ["y"])

    def test_explicit_none_is_exactly_the_legacy_estimate(self) -> None:
        sample, target = _fixture()
        frame = _frame(sample, target, model=LinearRegression())
        legacy = frame.aipw()

        explicit_none = frame.aipw(n_folds=None)

        pd.testing.assert_series_equal(explicit_none, legacy, check_exact=True)

    def test_pure_linear_oracle_with_independent_fit_weights(self) -> None:
        sample, target = _fixture()
        for fit_weight in (None, np.array([3, 1, 2, 1, 4, 1, 2, 1], dtype=float)):
            with self.subTest(weighted=fit_weight is not None):
                result = _pure(sample, target, n_folds=2, fit_sample_weight=fit_weight)
                expected = _linear_oracle(sample, target, fit_weight=fit_weight)
                self.assertAlmostEqual(result["y"], expected, places=10)
        self.assertIs(
            outcome_models.cross_fitted_aipw_point_estimate,
            aipw_module.cross_fitted_aipw_point_estimate,
        )

    def test_overfit_nearest_neighbor_uses_held_out_residuals(self) -> None:
        sample, target = _fixture()
        sample = sample.iloc[:6].copy()
        sample["x"] = [0.0, 1.0, 3.0, 7.0, 12.0, 20.0]
        target["x"] = [0.2, 2.3, 4.1, 21.0]
        target["weight"] = [2.0, 2.0, 3.0, 3.0]
        frame = _frame(sample, target, model=KNeighborsRegressor(n_neighbors=1))
        # Six leave-one-out fits: only the closest responder's omitted fold
        # changes a target prediction. No distances are tied in this fixture.
        target_fold_mean = (
            np.array([5 * 1.2 + 3.4, 5 * 2.1 + 3.4, 5 * 2.1 + 7.8, 5 * 8.9 + 4.3]) / 6
        )
        held_out_prediction = np.array([3.4, 1.2, 3.4, 2.1, 7.8, 4.3])
        expected = np.average(target_fold_mean, weights=target["weight"]) + np.average(
            sample["y"] - held_out_prediction, weights=sample["weight"]
        )

        result = frame.aipw(n_folds=6)

        self.assertAlmostEqual(float(result["y"]), float(expected), places=10)
        self.assertNotAlmostEqual(
            float(result["y"]), float(frame.aipw()["y"]), places=5
        )

    def test_seed_is_repeatable_and_changes_the_partition(self) -> None:
        sample, target = _fixture()
        first = _pure(sample, target, n_folds=2, random_seed=17)
        repeated = _pure(sample, target, n_folds=2, random_seed=17)
        different = _pure(sample, target, n_folds=2, random_seed=18)

        self.assertEqual(repeated, first)
        self.assertAlmostEqual(
            different["y"], _linear_oracle(sample, target, random_seed=18), places=10
        )
        self.assertNotAlmostEqual(different["y"], first["y"], places=6)
        self.assertEqual(
            _pure(sample, target), _pure(sample, target, n_folds=5, random_seed=2020)
        )

    def test_duplicate_and_nonmatching_labels_are_positional(self) -> None:
        sample, target = _fixture()
        expected = _pure(sample, target, n_folds=2)
        inputs = [
            sample[["x"]],
            sample[["y"]],
            sample["weight"],
            target[["x"]],
            target["weight"],
        ]
        labels = [[9] * 8, list(range(30, 38)), ["w"] * 8, [4, 4, 2, 2], ["t"] * 4]
        relabeled = [value.set_axis(index) for value, index in zip(inputs, labels)]
        before = pickle.dumps(relabeled)

        result = aipw_module.cross_fitted_aipw_point_estimate(
            *relabeled, fit_kwargs={"model": LinearRegression()}, n_folds=2
        )

        self.assertAlmostEqual(result["y"], expected["y"], places=12)
        self.assertEqual(pickle.dumps(relabeled), before)

    def test_fold_and_seed_validation(self) -> None:
        sample, target = _fixture()
        invalid = [
            {"n_folds": value} for value in (None, True, 1, 0, -2, 2.5, "2", 9)
        ] + [{"random_seed": value} for value in (True, -1, 2**32, 1.5, None, "0")]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    _pure(sample, target, **kwargs)
        self.assertTrue(
            np.isfinite(_pure(sample, target, n_folds=8, random_seed=2**32 - 1)["y"])
        )

    def test_residual_and_fit_weight_validation_are_separate(self) -> None:
        sample, target = _fixture()
        zero_residual = sample.copy()
        zero_residual["weight"] = [0.0, 3.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0]
        result = _pure(zero_residual, target, n_folds=2)
        self.assertAlmostEqual(
            result["y"], _linear_oracle(zero_residual, target), places=10
        )
        for value in (-1.0, np.nan, np.inf, 0.0):
            fit_weight = np.ones(8)
            fit_weight[0] = value
            with self.subTest(fit_weight=value):
                with self.assertRaisesRegex(ValueError, "weight"):
                    _pure(sample, target, n_folds=2, fit_sample_weight=fit_weight)
        for weights in ([1.0] * 7, [0.0] * 8, [-1.0] + [2.0] * 7, [np.nan] + [1.0] * 7):
            invalid = sample.copy()
            if len(weights) == 8:
                invalid["weight"] = weights
                with self.subTest(residual=weights):
                    with self.assertRaisesRegex(ValueError, "weight|scale"):
                        _pure(invalid, target, n_folds=2)
            else:
                with self.assertRaisesRegex(ValueError, "weight|length|row"):
                    _pure(
                        sample, target, n_folds=2, fit_sample_weight=np.array(weights)
                    )
        with self.assertRaisesRegex(ValueError, "scale"):
            _pure(sample, target.assign(weight=target["weight"] * 2), n_folds=2)

    def test_missing_outcomes_are_omitted_separately_per_column(self) -> None:
        sample, target = _fixture()
        outcomes = sample[["y"]].copy()
        outcomes.loc[[0, 5], "y"] = np.nan
        outcomes["z"] = [np.nan, 4.2, 1.1, 8.3, 5.6, 2.4, 9.7, np.nan]
        fit_weight = np.array([3, 1, 2, 1, 4, 1, 2, 1], dtype=float)

        result = aipw_module.cross_fitted_aipw_point_estimate(
            sample[["x"]],
            outcomes,
            sample["weight"],
            target[["x"]],
            target["weight"],
            fit_kwargs={"model": LinearRegression()},
            fit_sample_weight=fit_weight,
            n_folds=2,
        )

        for column in ("y", "z"):
            with self.subTest(column=column):
                expected = _linear_oracle(
                    sample.assign(y=outcomes[column]), target, fit_weight=fit_weight
                )
                self.assertAlmostEqual(result[column], expected, places=10)
        self.assertEqual(list(result), ["y", "z"])
        incomplete = sample.assign(y=outcomes["y"])
        frame = _frame(incomplete, target, model=LinearRegression())
        self.assertAlmostEqual(
            float(frame.aipw(n_folds=2)["y"]),
            _linear_oracle(incomplete, target),
            places=10,
        )
        weighted_frame = _frame(
            incomplete, target, model=LinearRegression(), weighted=True
        )
        with self.assertRaisesRegex(
            ValueError, "original fitting weights|fit_sample_weight"
        ):
            weighted_frame.aipw(n_folds=2)

    def test_nullable_binary_and_rare_classes_do_not_break_training(self) -> None:
        sample, target = _fixture()
        sample.loc[8] = ["8", 8.0, np.nan, 1.0]
        sample["weight"] = 1.0
        target["weight"] = [2.0, 2.0, 2.0, 3.0]
        results = []
        for dtype in ("Int64", "boolean", "float64"):
            missing = np.nan if dtype == "float64" else pd.NA
            sample["y"] = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, missing], dtype=dtype)
            with self.subTest(dtype=dtype), warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                result = _pure(
                    sample,
                    target,
                    n_folds=4,
                    fit_kwargs={"model": LogisticRegression()},
                )
                self.assertTrue(np.isfinite(result["y"]))
                self.assertFalse(
                    any("least populated class" in str(item.message) for item in ws)
                )
                results.append(result["y"])
        self.assertAlmostEqual(results[0], results[1], places=12)
        self.assertAlmostEqual(results[0], results[2], places=12)

    def test_binary_fold_labels_use_native_integers_for_older_sklearn(self) -> None:
        sample, target = _fixture()
        sample["y"] = pd.Series([0, 0, 0, 0, 0, 0, 1, 1], dtype="Int64")
        labels = []
        original_fit = LogisticRegression.fit

        def capture_fit(estimator, features, outcome, **kwargs):
            labels.append(outcome.copy())
            return original_fit(estimator, features, outcome, **kwargs)

        before = pickle.dumps(sample)
        with mock.patch.object(LogisticRegression, "fit", new=capture_fit):
            result = _pure(
                sample, target, n_folds=4, fit_kwargs={"model": LogisticRegression()}
            )

        self.assertTrue(np.isfinite(result["y"]))
        self.assertEqual(len(labels), 4)
        for outcome in labels:
            self.assertEqual(outcome.dtype, np.dtype("int64"))
            self.assertEqual(set(outcome.tolist()), {0, 1})
        self.assertEqual(pickle.dumps(sample), before)

    def test_insufficient_observations_and_binary_classes_are_actionable(self) -> None:
        sample, target = _fixture()
        cases = (
            (pd.Series([np.nan] * 8), LinearRegression()),
            (pd.Series([1.2] + [np.nan] * 7), LinearRegression()),
            (pd.Series([0] + [1] * 7, dtype="Int64"), LogisticRegression()),
            (pd.Series([0] * 8, dtype="Int64"), LogisticRegression()),
            (pd.Series(pd.Categorical([0, 2] * 4)), LogisticRegression()),
            (pd.Series(["0", "1"] * 4, dtype="string"), LogisticRegression()),
            (pd.Series(["0.0", "1.0"] * 4), LogisticRegression()),
        )
        for y, learner in cases:
            with self.subTest(y=y.tolist()):
                with self.assertRaisesRegex(ValueError, "observ|class|binary|0/1|fold"):
                    _pure(
                        sample.assign(y=y),
                        target,
                        n_folds=2,
                        fit_kwargs={"model": learner},
                    )

    def test_rare_category_schema_is_frozen_but_scalers_are_training_only(self) -> None:
        sample, target = _fixture()
        covars = sample[["x"]].assign(group=["usual"] * 7 + ["rare"])
        covars.loc[7, "x"] = 1000.0
        target_covars = target[["x"]].assign(group=["rare", "usual", "usual", "rare"])
        fold_models = []
        original_fit = aipw_module.fit_outcome_model

        def record_fit(*args, **kwargs):
            fitted = original_fit(*args, **kwargs)
            fold_models.append(fitted)
            return fitted

        with mock.patch.object(
            aipw_module, "fit_outcome_model", side_effect=record_fit
        ):
            result = aipw_module.cross_fitted_aipw_point_estimate(
                covars,
                sample[["y"]],
                sample["weight"],
                target_covars,
                target["weight"],
                fit_kwargs={"model": DummyRegressor(), "use_model_matrix": True},
                fit_sample_weight=np.ones(8),
                n_folds=2,
            )

        self.assertTrue(np.isfinite(result["y"]))
        self.assertEqual(len(fold_models), 2)
        self.assertEqual(
            fold_models[0]["X_matrix_columns"], fold_models[1]["X_matrix_columns"]
        )
        held_out_folds = np.array_split(np.random.RandomState(2020).permutation(8), 2)
        for fitted, held_out in zip(fold_models, held_out_folds):
            self.assertNotIn("training_inputs", fitted)
            train = np.setdiff1d(np.arange(8), held_out)
            x_column = fitted["X_matrix_columns"].index("x")
            scaler = fitted["fit_scaler"]
            self.assertAlmostEqual(
                float(scaler.mean_[x_column]),
                float(covars["x"].iloc[train].mean()),
                places=10,
            )
            self.assertAlmostEqual(
                float(scaler.var_[x_column]),
                float(covars["x"].iloc[train].var(ddof=0)),
                places=10,
            )

    def test_original_fit_weights_survive_later_adjustment(self) -> None:
        sample, target = _fixture()
        active_weights = np.array([3, 1, 1, 1, 2, 1, 2, 1], dtype=float)

        def change_weights(sample_df, sample_weights, target_df, target_weights):
            del sample_df, target_df, target_weights
            return {
                "weight": pd.Series(active_weights, index=sample_weights.index),
                "model": {"method": "test_reweight"},
            }

        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                frame = Sample.from_frame(sample, outcome_columns=["y"]).set_target(
                    Sample.from_frame(target)
                )
                frame.fit_outcome_model(model=LinearRegression(), weighted=weighted)
                frame = frame.adjust(method=change_weights)
                expected = _linear_oracle(
                    sample.assign(weight=active_weights),
                    target,
                    fit_weight=sample["weight"] if weighted else None,
                )

                result = frame.aipw(n_folds=2)

                self.assertAlmostEqual(float(result["y"]), expected, places=10)
                wrong = _linear_oracle(
                    sample.assign(weight=active_weights),
                    target,
                    fit_weight=active_weights,
                )
                self.assertNotAlmostEqual(float(result["y"]), wrong, places=6)

    def test_unselected_covariates_and_outcomes_are_ignored(self) -> None:
        sample, target = _fixture()
        sample["irrelevant"] = np.arange(8) ** 3
        sample["z"] = sample["y"] * 2
        target["irrelevant"] = [-1, -2, -3, -4]
        frame = Sample.from_frame(sample, outcome_columns=["y", "z"]).set_target(
            Sample.from_frame(target)
        )
        frame = frame.adjust(method=_keep_weights).fit_outcome_model(
            model=LinearRegression(), variables=["x"], outcome_columns=["y"]
        )
        frame._sf_sample._df.loc[:, "irrelevant"] = 999
        frame._sf_sample._df.loc[:, "z"] = -999
        _assert_type(frame._sf_target)._df.loc[:, "irrelevant"] = 888

        result = frame.aipw(n_folds=2)

        self.assertEqual(result.index.tolist(), ["y"])
        self.assertAlmostEqual(
            float(result["y"]), _linear_oracle(sample, target), places=10
        )

    def test_success_and_failed_fold_fit_do_not_mutate_frame(self) -> None:
        sample, target = _fixture()
        binary_sample = pd.DataFrame(
            {
                "id": [str(i) for i in range(16)],
                "x": np.arange(16, dtype=float),
                "y": pd.Series([0, 1] * 8, dtype="Int64"),
                "weight": np.ones(16),
            }
        )
        for calibrated in (False, True):
            with self.subTest(calibrated=calibrated):
                data = binary_sample if calibrated else sample
                learner = LogisticRegression() if calibrated else LinearRegression()
                frame = _frame(data, target, model=learner, calibrate=calibrated)
                frame.predict_outcomes(on="both")
                before = pickle.dumps(frame)
                model = frame.outcome_model
                logger_level = logging.getLogger("balance").level
                if calibrated:
                    # The full fit has eight rows per class; two folds leave
                    # four per class, insufficient for sklearn's five-way calibration.
                    with self.assertRaisesRegex(ValueError, "fold|n_splits|class"):
                        frame.aipw(n_folds=2)
                else:
                    self.assertTrue(np.isfinite(frame.aipw(n_folds=2)["y"]))
                self.assertIs(frame.outcome_model, model)
                self.assertEqual(pickle.dumps(frame), before)
                self.assertEqual(logging.getLogger("balance").level, logger_level)

    def test_legacy_or_changed_training_data_require_refitting(self) -> None:
        sample, target = _fixture()
        for changed in ("legacy", "covariate", "outcome", "rows", "borrowed"):
            with self.subTest(changed=changed):
                frame = _frame(sample, target, model=LinearRegression())
                if changed == "legacy":
                    _assert_type(frame.outcome_model).pop("training_inputs")
                    self.assertTrue(np.isfinite(frame.aipw()["y"]))
                elif changed == "covariate":
                    frame._sf_sample._df.loc[0, "x"] = 100.0
                elif changed == "outcome":
                    frame._sf_sample._df.loc[0, "y"] = 100.0
                elif changed == "rows":
                    frame._sf_sample._df = frame._sf_sample._df.iloc[::-1].copy()
                else:
                    other = _frame(
                        sample.assign(x=sample["x"] + 1),
                        target,
                        model=LinearRegression(),
                    )
                    frame._sf_sample._outcome_model = other.outcome_model
                before = pickle.dumps(frame)

                with self.assertRaisesRegex(ValueError, "[Rr]e.?fit|fit_outcome_model"):
                    frame.aipw(n_folds=2)

                self.assertEqual(pickle.dumps(frame), before)

    def test_formula_dropped_rows_stay_aligned_and_leave_weighted_means(self) -> None:
        sample, target = _fixture()
        frame = _frame(
            sample,
            target,
            model=LinearRegression(),
            formula="np.sqrt(x.where(x >= 2) - 2)",
        )
        # x < 2 has no formula value: responders 0/1 and target 0 are omitted,
        # not scored with a substituted feature or matched by index label.
        with np.errstate(invalid="ignore"):
            transformed_sample = sample.assign(x=np.sqrt(sample["x"] - 2))
            transformed_target = target.assign(x=np.sqrt(target["x"] - 2))
        expected = _linear_oracle(transformed_sample, transformed_target)

        result = frame.aipw(n_folds=2)

        self.assertAlmostEqual(float(result["y"]), expected, places=10)

    def test_constant_weights_warn_only_on_the_legacy_path(self) -> None:
        sample, target = _fixture()
        sample["weight"] = 1.0
        frame = _frame(sample, target, model=LinearRegression())
        with self.assertLogs("balance", level="WARNING") as legacy_logs:
            frame.aipw()
        with self.assertLogs("balance", level="INFO") as crossfit_logs:
            result = frame.aipw(n_folds=2)

        self.assertTrue(any("reduces" in message for message in legacy_logs.output))
        self.assertFalse(any("reduces" in message for message in crossfit_logs.output))
        calibrated_sample = sample.assign(
            weight=_assert_type(frame.weight_series).to_numpy()
        )
        self.assertAlmostEqual(
            float(result["y"]), _linear_oracle(calibrated_sample, target), places=10
        )

    def test_calibrated_mixed_learners_replay_the_fit_recipe(self) -> None:
        sample, target = _fixture()
        sample = pd.DataFrame(
            {
                "id": [str(i) for i in range(24)],
                "x": np.arange(24, dtype=float),
                "y": 1.2 + 2 * np.arange(24),
                "flag": pd.Series([0, 1] * 12, dtype="Int64"),
                "weight": np.ones(24),
            }
        )
        target["weight"] = 6.0
        learners = {"y": LinearRegression(), "flag": LogisticRegression(max_iter=1000)}
        frame = Sample.from_frame(sample, outcome_columns=["y", "flag"]).set_target(
            Sample.from_frame(target)
        )
        frame = frame.adjust(method=_keep_weights).fit_outcome_model(
            model=learners, calibrate=True
        )
        before = pickle.dumps(frame.outcome_model)

        result = frame.aipw(n_folds=2)
        pure = aipw_module.cross_fitted_aipw_point_estimate(
            sample[["x"]],
            sample[["y", "flag"]],
            sample["weight"],
            target[["x"]],
            _assert_type(target["weight"], pd.Series),
            fit_kwargs={"model": learners, "calibrate": True, "use_model_matrix": True},
            n_folds=2,
        )

        self.assertEqual(result.index.tolist(), ["y", "flag"])
        self.assertAlmostEqual(float(result["y"]), 8.7, places=10)
        self.assertAlmostEqual(float(result["flag"]), pure["flag"], places=12)
        self.assertTrue(_assert_type(frame.outcome_model)["calibrated"])
        self.assertEqual(pickle.dumps(frame.outcome_model), before)

    @pytest.mark.requires_sklearn_1_4
    @unittest.skipUnless(_SKLEARN_1_4_AVAILABLE, "requires sklearn >= 1.4")
    def test_native_categorical_recipe_matches_the_pure_function(self) -> None:
        sample, target = _fixture()
        sample["group"] = pd.Categorical(["a", "b"] * 4)
        target["group"] = pd.Categorical(["a", "b"] * 2)
        learner = HistGradientBoostingRegressor(
            min_samples_leaf=2, max_iter=3, random_state=0
        )
        frame = _frame(sample, target, model=learner, use_model_matrix=False)

        result = frame.aipw(n_folds=2)
        pure = aipw_module.cross_fitted_aipw_point_estimate(
            sample[["x", "group"]],
            sample[["y"]],
            sample["weight"],
            target[["x", "group"]],
            target["weight"],
            fit_kwargs={"model": learner, "use_model_matrix": False},
            n_folds=2,
        )

        self.assertAlmostEqual(float(result["y"]), pure["y"], places=12)
        self.assertFalse(_assert_type(frame.outcome_model)["use_model_matrix"])

    def test_full_outcome_kind_survives_binary_or_constant_training_folds(self) -> None:
        _, target = _fixture()
        sample = pd.DataFrame(
            {
                "id": [str(i) for i in range(20)],
                "x": np.arange(20, dtype=float),
                "y": [0, 1] * 9 + [2, 2],
                "weight": np.ones(20),
            }
        )
        target["weight"] = 5.0
        frame = _frame(sample, target, model=LinearRegression())

        result = frame.aipw(n_folds=5, random_seed=0)

        expected = _linear_oracle(sample, target, n_folds=5, random_seed=0)
        self.assertAlmostEqual(float(result["y"]), expected, places=10)
        with _capture_fold_models() as folds:
            auto = _pure(
                sample, target, fit_kwargs={"model": "auto"}, n_folds=5, random_seed=0
            )
        self.assertTrue(np.isfinite(auto["y"]))
        self.assertEqual(
            [fold["prediction_kind"]["y"] for fold in folds], ["regression"] * 5
        )
        count_sample = sample.iloc[:10].assign(y=[0] * 8 + [2] * 2)
        count_target = target.assign(weight=[2.0, 3.0, 3.0, 2.0])
        count_result = _pure(count_sample, count_target, n_folds=2, random_seed=0)
        self.assertAlmostEqual(
            count_result["y"],
            _linear_oracle(count_sample, count_target, random_seed=0),
            places=10,
        )

    def test_cross_fitting_preserves_random_forest_and_bagging_families(self) -> None:
        sample, target = _fixture()
        forest = RandomForestRegressor(n_estimators=7, max_depth=2, random_state=31)
        bagging = BaggingRegressor(n_estimators=7, random_state=31)
        parameter = (
            "estimator" if "estimator" in bagging.get_params() else "base_estimator"
        )
        bagging.set_params(**{parameter: DecisionTreeRegressor(max_depth=2)})
        for blueprint in (forest, bagging):
            with self.subTest(family=type(blueprint).__name__):
                frame = _frame(sample, target, model=blueprint)
                before = pickle.dumps(frame.outcome_model)
                with _capture_fold_models() as folds:
                    result = frame.aipw(n_folds=2)
                pure = _pure(sample, target, fit_kwargs={"model": blueprint}, n_folds=2)
                self.assertAlmostEqual(float(result["y"]), pure["y"], places=12)
                self.assertEqual(len(folds), 2)
                for fold in folds:
                    learner = fold["fit"]["y"]
                    self.assertIsInstance(learner, type(blueprint))
                    self.assertEqual(learner.get_params()["n_estimators"], 7)
                    self.assertEqual(learner.get_params()["random_state"], 31)
                self.assertEqual(pickle.dumps(frame.outcome_model), before)

    def test_explicit_calibration_wrapper_preserves_cv_and_method(self) -> None:
        _, target = _fixture()
        sample = pd.DataFrame(
            {
                "id": [str(i) for i in range(16)],
                "x": np.arange(16, dtype=float),
                "y": pd.Series([0, 1] * 8, dtype="Int64"),
                "weight": np.ones(16),
            }
        )
        target["weight"] = 4.0
        blueprint = CalibratedClassifierCV(
            LogisticRegression(max_iter=1000), cv=2, method="isotonic"
        )
        frame = _frame(sample, target, model=blueprint, calibrate=False)
        before = pickle.dumps(frame.outcome_model)

        with _capture_fold_models() as folds:
            result = frame.aipw(n_folds=2)
        pure = _pure(
            sample,
            target,
            fit_kwargs={"model": blueprint, "calibrate": False},
            n_folds=2,
        )

        self.assertAlmostEqual(float(result["y"]), pure["y"], places=12)
        self.assertEqual(len(folds), 2)
        for fold in folds:
            learner = _assert_type(fold["fit"]["y"], CalibratedClassifierCV)
            self.assertEqual(learner.cv, 2)
            self.assertEqual(learner.method, "isotonic")
            self.assertEqual(fold["prediction_kind"]["y"], "proba")
        self.assertFalse(_assert_type(frame.outcome_model)["calibrated"])
        self.assertEqual(pickle.dumps(frame.outcome_model), before)
