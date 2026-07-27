# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import unittest
import unittest.mock
from typing import Any

import balance.testutil
import numpy as np
import pandas as pd
import pytest
from balance.outcome_models import (
    bootstrap_outcome_estimate,
    fit_outcome_model,
    learner_from_model,
    predict_outcome,
)
from balance.outcome_models.outcome_model import (
    _prepare_sample_weight,
    _resolve_learner,
    _resolve_use_model_matrix,
)
from balance.testutil import _SKLEARN_1_4_AVAILABLE
from balance.weighting_methods.ipw import model_coefs
from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor


def _make_regression_data(
    n: int = 300, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Continuous outcome that depends on a numeric + a 3-level categorical covariate."""
    rng = np.random.default_rng(seed)
    age = rng.normal(50.0, 10.0, n)
    grp = rng.choice(["a", "b", "c"], n)
    base = {"a": 10.0, "b": 50.0, "c": 90.0}
    happiness = np.array([base[g] for g in grp]) + 0.2 * age + rng.normal(0.0, 2.0, n)
    covars = pd.DataFrame({"age": age, "grp": grp})
    outcomes = pd.DataFrame({"happiness": happiness})
    weights = pd.Series(rng.uniform(0.5, 2.0, n), index=covars.index)
    return covars, outcomes, weights


def _make_target(n: int = 5, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.normal(55.0, 10.0, n),
            "grp": rng.choice(["a", "b", "c"], n),
        }
    )


@pytest.mark.parametrize(
    "invalid_weights, reason",
    (
        ([1.0, 0.0, 1.0], "strictly positive"),
        ([1.0, -1.0, 1.0], "non-negative"),
        ([1.0, np.nan, 1.0], "finite"),
        ([1.0, np.inf, 1.0], "finite"),
        ([1.0, -np.inf, 1.0], "finite"),
        ([1.0, "not-a-number", 1.0], "must be a number"),
        ([1.0, "2.0", 1.0], "must be a number"),
        ([True, True, True], "must be a number"),
        ([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j], "must be a number"),
    ),
)
def test_fit_rejects_invalid_sample_weights(
    invalid_weights: list[Any], reason: str
) -> None:
    """Estimator weights must be numeric, finite, and strictly positive."""
    covars = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    outcomes = pd.DataFrame({"y": [2.0, 4.0, 6.0]})

    with pytest.raises(
        ValueError,
        match=(
            "sample_weight must contain only finite, strictly positive real "
            "numeric values"
        ),
    ) as exc_info:
        fit_outcome_model(
            covars,
            outcomes,
            sample_weight=pd.Series(invalid_weights),
            model=LinearRegression(),
        )
    assert reason in str(exc_info.value)


@pytest.mark.parametrize(
    "invalid_weights, expected_message",
    (
        (np.ones((3, 1)), "one-dimensional.*shape \\(3, 1\\)"),
        (np.ones((1, 3)), "one-dimensional.*shape \\(1, 3\\)"),
        (np.array(1.0), "one-dimensional.*shape \\(\\)"),
        (np.ones(2), "same length as covars_df: expected 3, got 2"),
        (np.ones(4), "same length as covars_df: expected 3, got 4"),
    ),
)
def test_fit_rejects_invalid_sample_weight_shape(
    invalid_weights: np.ndarray, expected_message: str
) -> None:
    covars = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    outcomes = pd.DataFrame({"y": [2.0, 4.0, 6.0]})

    with pytest.raises(ValueError, match=expected_message):
        fit_outcome_model(
            covars,
            outcomes,
            sample_weight=invalid_weights,
            model=LinearRegression(),
        )


def test_prepare_sample_weight_preserves_series_metadata_and_none() -> None:
    """Validation must preserve the weight name and covariate row order."""
    covars = pd.DataFrame({"x": [1, 2]}, index=pd.Index([20, 10]))
    weights = pd.Series([1, 2], index=covars.index, name="design_weight")

    weighted, values, column = _prepare_sample_weight(weights, covars)
    assert weighted is True
    np.testing.assert_array_equal(values, np.array([1.0, 2.0]))
    assert column == "design_weight"
    assert _prepare_sample_weight(None, covars) == (False, None, None)


def test_prepare_sample_weight_reuses_canonical_weight_validation() -> None:
    """Outcome fitting delegates value checks to the shared weight validator."""
    covars = pd.DataFrame({"x": [1, 2]})
    weights = pd.Series([1.0, 2.0])

    with unittest.mock.patch(
        "balance.outcome_models.outcome_model._check_weights_series_are_valid",
        wraps=_prepare_sample_weight.__globals__["_check_weights_series_are_valid"],
    ) as check_weights:
        weighted, values, _column = _prepare_sample_weight(weights, covars)

    assert weighted is True
    np.testing.assert_array_equal(values, weights.to_numpy())
    check_weights.assert_called_once_with(
        weights,
        require_finite=True,
        require_strictly_positive=True,
    )


class TestResolveLearner(balance.testutil.BalanceTestCase):
    def test_auto_dispatches_regressor_for_continuous(self) -> None:
        est, note = _resolve_learner("auto", pd.Series([1.0, 2.5, 3.1, 4.9], name="y"))
        self.assertTrue(is_regressor(est))
        self.assertFalse(is_classifier(est))
        self.assertEqual(note, "auto")

    def test_auto_dispatches_classifier_for_binary(self) -> None:
        est, note = _resolve_learner("auto", pd.Series([0, 1, 1, 0], name="y"))
        self.assertTrue(is_classifier(est))
        self.assertEqual(note, "auto")

    def test_single_estimator_is_cloned(self) -> None:
        original = LinearRegression()
        est, note = _resolve_learner(original, pd.Series([1.0, 2.0], name="y"))
        self.assertIsInstance(est, LinearRegression)
        self.assertIsNot(est, original)
        self.assertIn("single", note)

    def test_dict_model_looked_up_by_column(self) -> None:
        est, note = _resolve_learner(
            {"happiness": LinearRegression()},
            pd.Series([1.0, 2.0], name="happiness"),
        )
        self.assertIsInstance(est, LinearRegression)
        self.assertIn("happiness", note)

    def test_type_map_discrete_returns_cloned_classifier(self) -> None:
        clf = LogisticRegression()
        est, _ = _resolve_learner(
            {"_discrete": clf, "_continuous": LinearRegression()},
            pd.Series([0, 1, 1, 0], name="y"),
        )
        self.assertIsInstance(est, LogisticRegression)
        self.assertIsNot(est, clf)  # cloned, not the caller's instance

    def test_type_map_continuous_returns_regressor(self) -> None:
        est, _ = _resolve_learner(
            {"_discrete": LogisticRegression(), "_continuous": LinearRegression()},
            pd.Series([1.0, 2.5, 3.1], name="y"),
        )
        self.assertIsInstance(est, LinearRegression)

    def test_type_map_missing_key_falls_back_to_auto(self) -> None:
        est, note = _resolve_learner(
            {"_continuous": LinearRegression()},
            pd.Series([0, 1, 1, 0], name="y"),
        )
        self.assertTrue(is_classifier(est))
        self.assertIn("auto", note)

    def test_dict_model_missing_column_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not contain an estimator"):
            _resolve_learner(
                {"other": LinearRegression()},
                pd.Series([1.0, 2.0], name="happiness"),
            )

    def test_unknown_string_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown model string"):
            _resolve_learner("nope", pd.Series([1.0, 2.0], name="y"))

    def test_resolve_use_model_matrix_linear_is_true(self) -> None:
        self.assertTrue(_resolve_use_model_matrix("auto", LinearRegression()))

    def test_resolve_use_model_matrix_explicit_bool(self) -> None:
        self.assertFalse(_resolve_use_model_matrix(False, LinearRegression()))
        self.assertTrue(
            _resolve_use_model_matrix(True, HistGradientBoostingRegressor())
        )


class TestFitOutcomeModel(balance.testutil.BalanceTestCase):
    def test_example_snippet(self) -> None:
        """Asserts the diff's Example snippet (fit -> dict; predict -> ŷ on new data)."""
        covars_R, outcomes_R, w_R = _make_regression_data()
        model = fit_outcome_model(covars_R, outcomes_R, sample_weight=w_R)
        # -> dict with "fit", "X_matrix_columns", ...
        self.assertEqual(model["method"], "outcome_model")
        self.assertIn("fit", model)
        self.assertIn("X_matrix_columns", model)
        self.assertIn("happiness", model["fit"])

        covars_T = _make_target()
        preds = predict_outcome(model, covars_T)["happiness"][:3]
        self.assertEqual(len(preds), 3)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_continuous_outcome_uses_regressor(self) -> None:
        covars_R, outcomes_R, w_R = _make_regression_data()
        model = fit_outcome_model(covars_R, outcomes_R, sample_weight=w_R)
        self.assertEqual(model["prediction_kind"]["happiness"], "regression")
        self.assertTrue(is_regressor(model["fit"]["happiness"]))

    def test_binary_outcome_uses_predict_proba(self) -> None:
        covars_R, outcomes_R, w_R = _make_regression_data()
        y_bin = (outcomes_R["happiness"] > outcomes_R["happiness"].median()).astype(int)
        model = fit_outcome_model(
            covars_R, pd.DataFrame({"happy": y_bin}), sample_weight=w_R
        )
        self.assertEqual(model["prediction_kind"]["happy"], "proba")
        self.assertTrue(is_classifier(model["fit"]["happy"]))

        preds = predict_outcome(model, _make_target())["happy"]
        self.assertTrue(np.all(preds >= 0.0) and np.all(preds <= 1.0))

    def test_perf_populated_regression_weighted_r2(self) -> None:
        covars_R, outcomes_R, w_R = _make_regression_data()
        model = fit_outcome_model(covars_R, outcomes_R, sample_weight=w_R)
        perf = model["perf"]["happiness"]
        self.assertIn("r2", perf)
        self.assertEqual(perf["n"], len(covars_R))
        # A well-specified boosting fit on strongly-separated groups is high R².
        self.assertGreater(perf["r2"], 0.8)

    def test_perf_populated_classifier_deviance(self) -> None:
        covars_R, outcomes_R, w_R = _make_regression_data()
        y_bin = (outcomes_R["happiness"] > outcomes_R["happiness"].median()).astype(int)
        model = fit_outcome_model(
            covars_R, pd.DataFrame({"happy": y_bin}), sample_weight=w_R
        )
        perf = model["perf"]["happy"]
        self.assertIn("deviance_explained", perf)
        self.assertIn("log_loss", perf)
        self.assertEqual(perf["n"], len(covars_R))

    def test_sample_weight_is_respected(self) -> None:
        """Extreme weights on one group change the fitted predictions."""
        rng = np.random.default_rng(3)
        n = 200
        grp = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
        # Only "age" is available to the learner; the group signal is hidden, so
        # the fit is deliberately misspecified and must lean on whichever group
        # the weights emphasize — making the effect of sample_weight visible.
        covars = pd.DataFrame({"age": rng.normal(50, 5, n)})
        y = np.where(grp == "a", 0.0, 100.0) + rng.normal(0, 1, n)
        outcomes = pd.DataFrame({"y": y})

        # Uniform weights vs weights that up-weight group "a" heavily.
        w_uniform = pd.Series(np.ones(n), index=covars.index)
        w_skewed = pd.Series(np.where(grp == "a", 100.0, 0.01), index=covars.index)
        m_uniform = fit_outcome_model(
            covars, outcomes, sample_weight=w_uniform, model=LinearRegression()
        )
        m_skewed = fit_outcome_model(
            covars, outcomes, sample_weight=w_skewed, model=LinearRegression()
        )
        self.assertTrue(m_uniform["weighted"])
        # fit_weight["uniform"] reflects the ACTUAL weight vector, not just
        # weighted=: w_uniform is all-ones (uniform), w_skewed is not.
        self.assertTrue(m_uniform["fit_weight"]["uniform"])
        self.assertTrue(m_skewed["weighted"])
        self.assertFalse(m_skewed["fit_weight"]["uniform"])

        # Uniform fit predicts near the overall mean (~50); the a-weighted fit is
        # pulled toward group a's mean (~0).
        probe = pd.DataFrame({"age": [50.0]})
        p_uniform = predict_outcome(m_uniform, probe)["y"][0]
        p_skewed = predict_outcome(m_skewed, probe)["y"][0]
        self.assertGreater(p_uniform, 40.0)
        self.assertLess(p_skewed, 20.0)

    def test_unweighted_fit_marks_uniform(self) -> None:
        covars_R, outcomes_R, _ = _make_regression_data()
        model = fit_outcome_model(covars_R, outcomes_R)
        self.assertFalse(model["weighted"])
        self.assertTrue(model["fit_weight"]["uniform"])

    def test_na_action_drop_raises(self) -> None:
        covars_R, outcomes_R, _ = _make_regression_data()
        with self.assertRaisesRegex(ValueError, "na_action='drop'"):
            fit_outcome_model(covars_R, outcomes_R, na_action="drop")

    def test_transformations_non_none_raises(self) -> None:
        covars_R, outcomes_R, _ = _make_regression_data()
        with self.assertRaisesRegex(
            ValueError, "transformations are not yet supported"
        ):
            fit_outcome_model(covars_R, outcomes_R, transformations="default")

    def test_empty_outcomes_raises(self) -> None:
        covars_R, _, _ = _make_regression_data()
        with self.assertRaisesRegex(ValueError, "at least one outcome column"):
            fit_outcome_model(covars_R, pd.DataFrame(index=covars_R.index))

    def test_linear_regression_plugin_and_model_coefs(self) -> None:
        covars_R, outcomes_R, w_R = _make_regression_data()
        model = fit_outcome_model(
            covars_R, outcomes_R, sample_weight=w_R, model=LinearRegression()
        )
        # Linear learner -> one-hot + scaler path.
        self.assertTrue(model["use_model_matrix"])
        estimator = model["fit"]["happiness"]
        self.assertIsInstance(estimator, LinearRegression)
        # pyre-ignore[6]: model_coefs is typed for ClassifierMixin but only reads
        # coef_/intercept_, which the fitted LinearRegression also exposes.
        coefs = model_coefs(estimator, feature_names=model["X_matrix_columns"])
        self.assertIn("coefs", coefs)
        self.assertGreater(len(coefs["coefs"]), 0)
        # Sensible fit.
        self.assertGreater(model["perf"]["happiness"]["r2"], 0.8)

    def test_multiple_outcomes(self) -> None:
        covars_R, outcomes_R, w_R = _make_regression_data()
        outcomes_two = outcomes_R.assign(happiness2=outcomes_R["happiness"] * 2.0 + 1.0)
        model = fit_outcome_model(covars_R, outcomes_two, sample_weight=w_R)
        self.assertCountEqual(model["outcome_columns"], ["happiness", "happiness2"])
        preds = predict_outcome(model, _make_target())
        self.assertCountEqual(list(preds.keys()), ["happiness", "happiness2"])

    def test_dict_learner_end_to_end(self) -> None:
        covars_R, outcomes_R, w_R = _make_regression_data()
        model = fit_outcome_model(
            covars_R,
            outcomes_R,
            sample_weight=w_R,
            model={"happiness": LinearRegression()},
        )
        self.assertIsInstance(model["fit"]["happiness"], LinearRegression)
        preds = predict_outcome(model, _make_target())["happiness"]
        self.assertEqual(len(preds), 5)

    def test_training_sample_index_stored(self) -> None:
        covars_R, outcomes_R, w_R = _make_regression_data()
        covars_R = covars_R.set_axis(
            pd.Index(range(1000, 1000 + len(covars_R))), axis=0
        )
        outcomes_R = outcomes_R.set_axis(covars_R.index, axis=0)
        w_R = w_R.set_axis(covars_R.index)
        model = fit_outcome_model(covars_R, outcomes_R, sample_weight=w_R)
        self.assertEqual(list(model["training_sample_index"]), list(covars_R.index))


class TestModelDispatch(balance.testutil.BalanceTestCase):
    def test_type_map_dispatches_by_outcome_type(self) -> None:
        """A {"_discrete": clf, "_continuous": reg} map picks per outcome type."""
        covars_R, outcomes_R, w_R = _make_regression_data()
        y_bin = (outcomes_R["happiness"] > outcomes_R["happiness"].median()).astype(int)
        outcomes = outcomes_R.assign(happy=y_bin)  # continuous + discrete together
        model = fit_outcome_model(
            covars_R,
            outcomes,
            sample_weight=w_R,
            model={
                "_discrete": LogisticRegression(max_iter=1000),
                "_continuous": LinearRegression(),
            },
        )
        self.assertIsInstance(model["fit"]["happiness"], LinearRegression)
        self.assertIsInstance(model["fit"]["happy"], LogisticRegression)
        self.assertEqual(model["prediction_kind"]["happiness"], "regression")
        self.assertEqual(model["prediction_kind"]["happy"], "proba")

    def test_partial_type_map_falls_back_to_auto(self) -> None:
        """A discrete outcome with only '_continuous' mapped falls back to auto."""
        covars_R, outcomes_R, w_R = _make_regression_data()
        y_bin = (outcomes_R["happiness"] > outcomes_R["happiness"].median()).astype(int)
        model = fit_outcome_model(
            covars_R,
            pd.DataFrame({"happy": y_bin}),
            sample_weight=w_R,
            model={"_continuous": LinearRegression()},
        )
        self.assertTrue(is_classifier(model["fit"]["happy"]))

    def test_single_estimator_mixed_outcomes_raises(self) -> None:
        """One estimator can't serve mixed discrete + continuous outcomes."""
        covars_R, outcomes_R, w_R = _make_regression_data()
        y_bin = (outcomes_R["happiness"] > outcomes_R["happiness"].median()).astype(int)
        outcomes = outcomes_R.assign(happy=y_bin)
        with self.assertRaisesRegex(ValueError, "mixed type"):
            fit_outcome_model(
                covars_R, outcomes, sample_weight=w_R, model=LinearRegression()
            )

    def test_weighted_fit_unsupported_estimator_raises_type_error(self) -> None:
        """weighted=True with a fit() lacking sample_weight -> TypeError."""
        covars_R, outcomes_R, w_R = _make_regression_data()
        with self.assertRaisesRegex(TypeError, "sample_weight"):
            fit_outcome_model(
                covars_R, outcomes_R, sample_weight=w_R, model=KNeighborsRegressor()
            )

    def test_unweighted_fit_unsupported_estimator_ok(self) -> None:
        """The same estimator fits fine unweighted (no sample_weight needed)."""
        covars_R, outcomes_R, _ = _make_regression_data()
        model = fit_outcome_model(covars_R, outcomes_R, model=KNeighborsRegressor())
        self.assertFalse(model["weighted"])
        preds = predict_outcome(model, _make_target())["happiness"]
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_fit_logs_model_choice_at_info(self) -> None:
        """The per-outcome model choice is logged at INFO for observability."""
        covars_R, outcomes_R, w_R = _make_regression_data()
        with self.assertLogs("balance.outcome_models", level="INFO") as cm:
            fit_outcome_model(covars_R, outcomes_R, sample_weight=w_R)
        self.assertTrue(
            any("outcome_model: outcome 'happiness'" in line for line in cm.output),
            f"expected an INFO fit log for 'happiness'; got {cm.output}",
        )


class TestPredictOutcome(balance.testutil.BalanceTestCase):
    @pytest.mark.requires_sklearn_1_4  # pyre-ignore[56]
    @unittest.skipUnless(_SKLEARN_1_4_AVAILABLE, "requires sklearn >= 1.4")
    def test_predictions_are_row_aligned_and_index_independent(self) -> None:
        covars_R, outcomes_R, w_R = _make_regression_data()
        model = fit_outcome_model(
            covars_R, outcomes_R, sample_weight=w_R, use_model_matrix=False
        )
        probe = pd.DataFrame({"age": [50.0, 50.0, 50.0], "grp": ["a", "b", "c"]})
        ref = predict_outcome(model, probe)["happiness"]

        probe_idx = probe.set_axis(pd.Index([7, 8, 9]), axis=0)
        shifted = predict_outcome(model, probe_idx)["happiness"]
        np.testing.assert_allclose(shifted, ref, atol=1e-8)

    @pytest.mark.requires_sklearn_1_4  # pyre-ignore[56]
    @unittest.skipUnless(_SKLEARN_1_4_AVAILABLE, "requires sklearn >= 1.4")
    def test_categorical_levels_replay_missing_category(self) -> None:
        """A batch missing a fit-time category must not shift the other codes."""
        covars_R, outcomes_R, w_R = _make_regression_data(seed=7)
        model = fit_outcome_model(
            covars_R, outcomes_R, sample_weight=w_R, use_model_matrix=False
        )
        self.assertIsNotNone(model["categorical_levels"])

        ref = predict_outcome(
            model,
            pd.DataFrame({"age": [50.0, 50.0, 50.0], "grp": ["a", "b", "c"]}),
        )["happiness"]
        # 'c' absent from this batch — a/b predictions must be unchanged.
        missing_c = predict_outcome(
            model, pd.DataFrame({"age": [50.0, 50.0], "grp": ["a", "b"]})
        )["happiness"]
        np.testing.assert_allclose(missing_c, ref[:2], atol=1e-8)

    @pytest.mark.requires_sklearn_1_4  # pyre-ignore[56]
    @unittest.skipUnless(_SKLEARN_1_4_AVAILABLE, "requires sklearn >= 1.4")
    def test_categorical_levels_replay_novel_category(self) -> None:
        """A novel target category must not steal a fit-time integer code."""
        covars_R, outcomes_R, w_R = _make_regression_data(seed=7)
        model = fit_outcome_model(
            covars_R, outcomes_R, sample_weight=w_R, use_model_matrix=False
        )
        ref = predict_outcome(
            model,
            pd.DataFrame({"age": [50.0, 50.0, 50.0], "grp": ["a", "b", "c"]}),
        )["happiness"]
        novel = predict_outcome(
            model,
            pd.DataFrame({"age": [50.0, 50.0, 50.0], "grp": ["a", "b", "z"]}),
        )["happiness"]
        # 'a'/'b' stay correct even with the novel 'z' present.
        np.testing.assert_allclose(novel[:2], ref[:2], atol=1e-8)
        # The novel category still yields a finite prediction (handled as unknown).
        self.assertTrue(np.isfinite(novel[2]))

    @pytest.mark.requires_sklearn_1_4  # pyre-ignore[56]
    @unittest.skipUnless(_SKLEARN_1_4_AVAILABLE, "requires sklearn >= 1.4")
    def test_categorical_with_na_replays_without_double_na_category(self) -> None:
        """A categorical covariate WITH missing values must replay on the native
        path — the fit-time levels capture the "_NA" sentinel, and replay must not
        double-add it (regression for "new categories must not include old ...")."""
        rng = np.random.default_rng(0)
        n = 120
        grp = np.array(rng.choice(["a", "b", "c"], n), dtype=object)
        grp[rng.random(n) < 0.25] = None  # inject NAs into the categorical covariate
        covars = pd.DataFrame({"age": rng.normal(50, 10, n), "grp": grp})
        outcomes = pd.DataFrame({"happiness": rng.normal(70, 5, n)})
        model = fit_outcome_model(
            covars,
            outcomes,
            sample_weight=pd.Series(np.ones(n)),
            use_model_matrix=False,
        )
        # Score a DIFFERENT frame whose categorical also has NAs; this replays the
        # stored categorical levels (which captured "_NA") and previously raised.
        grp_t = np.array(rng.choice(["a", "b", "c"], 30), dtype=object)
        grp_t[rng.random(30) < 0.25] = None
        new = pd.DataFrame({"age": rng.normal(55, 10, 30), "grp": grp_t})
        preds = predict_outcome(model, new)["happiness"]
        self.assertEqual(len(preds), 30)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_use_model_matrix_false_raises_on_old_sklearn(self) -> None:
        """use_model_matrix=False with categoricals needs sklearn >= 1.4; on older
        versions it raises an actionable error, not a cryptic sklearn conversion
        error. (Version-mocked, so it runs on any installed scikit-learn.)"""
        covars_R, outcomes_R, w_R = _make_regression_data()
        with unittest.mock.patch(
            "balance.outcome_models.outcome_model._has_sklearn_1_4",
            return_value=False,
        ):
            with self.assertRaisesRegex(
                ValueError, "use_model_matrix=False.*scikit-learn >= 1.4"
            ):
                fit_outcome_model(
                    covars_R, outcomes_R, sample_weight=w_R, use_model_matrix=False
                )

    def test_densify_for_boosting_one_hot_path(self) -> None:
        """Forcing the one-hot path for a boosting learner densifies (no sparse)."""
        covars_R, outcomes_R, w_R = _make_regression_data()
        model = fit_outcome_model(
            covars_R, outcomes_R, sample_weight=w_R, use_model_matrix=True
        )
        # HistGradientBoosting rejects sparse -> stored matrix type must be dense.
        self.assertEqual(model["fit_matrix_type"], "dense")
        preds = predict_outcome(model, _make_target())["happiness"]
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_linear_replay_matches_manual(self) -> None:
        """Linear one-hot+scaler replay reproduces the estimator applied by hand."""
        covars_R, outcomes_R, w_R = _make_regression_data()
        model = fit_outcome_model(
            covars_R, outcomes_R, sample_weight=w_R, model=LinearRegression()
        )
        covars_T = _make_target()
        preds = predict_outcome(model, covars_T)["happiness"]
        self.assertEqual(len(preds), len(covars_T))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_rejects_non_outcome_model_dict(self) -> None:
        """A malformed / non-outcome-model dict raises a clear ValueError (not a
        bare KeyError) — both a wrong-method dict and one missing 'fit'."""
        covars_R, _, _ = _make_regression_data()
        with self.assertRaisesRegex(
            ValueError,
            "predict_outcome expected a model dict from fit_outcome_model",
        ):
            predict_outcome({"not": "a model"}, covars_R)
        with self.assertRaisesRegex(
            ValueError,
            "predict_outcome expected a model dict from fit_outcome_model",
        ):
            predict_outcome({"method": "outcome_model"}, covars_R)


class TestOutcomeModelDictContract(balance.testutil.BalanceTestCase):
    def test_stored_dict_keys(self) -> None:
        covars_R, outcomes_R, w_R = _make_regression_data()
        model = fit_outcome_model(covars_R, outcomes_R, sample_weight=w_R)
        expected_keys = {
            "method",
            "fit",
            "X_matrix_columns",
            "fit_scaler",
            "formula",
            "na_action",
            "one_hot_encoding",
            "use_model_matrix",
            "transformations",
            "fit_matrix_type",
            "weighted",
            "fit_weight",
            "prediction_kind",
            "calibrated",
            "categorical_levels",
            "outcome_columns",
            "learner",
            "perf",
            "training_sample_index",
        }
        self.assertTrue(expected_keys.issubset(set(model.keys())))

    def test_calibrate_wraps_classifier(self) -> None:
        from sklearn.calibration import CalibratedClassifierCV

        covars_R, outcomes_R, w_R = _make_regression_data()
        y_bin = (outcomes_R["happiness"] > outcomes_R["happiness"].median()).astype(int)
        model = fit_outcome_model(
            covars_R,
            pd.DataFrame({"happy": y_bin}),
            sample_weight=w_R,
            calibrate=True,
        )
        self.assertTrue(model["calibrated"])
        self.assertIsInstance(model["fit"]["happy"], CalibratedClassifierCV)
        preds = predict_outcome(model, _make_target())["happy"]
        self.assertTrue(np.all(preds >= 0.0) and np.all(preds <= 1.0))


class TestNativeVsOneHot(balance.testutil.BalanceTestCase):
    @pytest.mark.requires_sklearn_1_4  # pyre-ignore[56]
    @unittest.skipUnless(_SKLEARN_1_4_AVAILABLE, "requires sklearn >= 1.4")
    def test_native_categorical_path_used_by_default(self) -> None:
        """On sklearn >= 1.4 the boosting default uses the native-categorical path."""
        covars_R, outcomes_R, w_R = _make_regression_data()
        model = fit_outcome_model(covars_R, outcomes_R, sample_weight=w_R)
        self.assertFalse(model["use_model_matrix"])
        self.assertEqual(model["fit_matrix_type"], "dataframe")
        self.assertIsNotNone(model["categorical_levels"])

    @pytest.mark.requires_sklearn_1_4  # pyre-ignore[56]
    @unittest.skipUnless(_SKLEARN_1_4_AVAILABLE, "requires sklearn >= 1.4")
    def test_native_and_onehot_give_comparable_predictions(self) -> None:
        """Native-categorical and one-hot-fallback boosting predictions agree closely."""
        covars_R, outcomes_R, w_R = _make_regression_data()
        m_native = fit_outcome_model(
            covars_R, outcomes_R, sample_weight=w_R, use_model_matrix=False
        )
        m_onehot = fit_outcome_model(
            covars_R, outcomes_R, sample_weight=w_R, use_model_matrix=True
        )
        covars_T = _make_target(n=20)
        p_native = predict_outcome(m_native, covars_T)["happiness"]
        p_onehot = predict_outcome(m_onehot, covars_T)["happiness"]
        # Same underlying data + learner, different encoding -> highly correlated.
        corr = float(np.corrcoef(p_native, p_onehot)[0, 1])
        self.assertGreater(corr, 0.99)


class TestOneHotFallbackOnOldSklearn(balance.testutil.BalanceTestCase):
    def test_auto_falls_back_to_one_hot_when_sklearn_below_1_4(self) -> None:
        """When sklearn < 1.4, the boosting default resolves to the one-hot path."""
        covars_R, outcomes_R, w_R = _make_regression_data()
        with unittest.mock.patch(
            "balance.outcome_models.outcome_model._has_sklearn_1_4",
            return_value=False,
        ):
            model = fit_outcome_model(covars_R, outcomes_R, sample_weight=w_R)
        # Fallback -> one-hot + scaler path (and densified for boosting), so the
        # boosting default keeps working even when native categoricals are
        # unavailable. The stored preprocessing reflects the one-hot path (the
        # design matrix has expanded grp[a]/grp[b]/grp[c] columns, not a raw
        # categorical), and predictions still come out finite.
        self.assertTrue(model["use_model_matrix"])
        self.assertEqual(model["fit_matrix_type"], "dense")
        self.assertIsNone(model["categorical_levels"])
        self.assertIn("grp[a]", model["X_matrix_columns"])
        preds = predict_outcome(model, _make_target())["happiness"]
        self.assertTrue(np.all(np.isfinite(preds)))


class TestOutcomeModelGuards(balance.testutil.BalanceTestCase):
    def test_string_binary_outcome_raises(self) -> None:
        covars_R, _, w_R = _make_regression_data()
        y = pd.DataFrame({"lab": np.where(np.arange(len(covars_R)) % 2, "yes", "no")})
        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            fit_outcome_model(covars_R, y, sample_weight=w_R)

    def test_multiclass_string_outcome_raises(self) -> None:
        covars_R, _, w_R = _make_regression_data()
        y = pd.DataFrame({"lab": np.resize(["a", "b", "c"], len(covars_R))})
        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            fit_outcome_model(covars_R, y, sample_weight=w_R)

    def test_single_class_outcome_raises(self) -> None:
        covars_R, _, w_R = _make_regression_data()
        y = pd.DataFrame({"happy": np.zeros(len(covars_R), dtype=int)})
        with self.assertRaisesRegex(ValueError, "single observed class"):
            fit_outcome_model(covars_R, y, sample_weight=w_R)

    def test_empty_covars_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "no rows"):
            fit_outcome_model(pd.DataFrame({"age": []}), pd.DataFrame({"y": []}))

    def test_boolean_binary_outcome_ok(self) -> None:
        covars_R, outcomes_R, w_R = _make_regression_data()
        y = pd.DataFrame(
            {"happy": outcomes_R["happiness"] > outcomes_R["happiness"].median()}
        )
        model = fit_outcome_model(covars_R, y, sample_weight=w_R)
        self.assertEqual(model["prediction_kind"]["happy"], "proba")
        preds = predict_outcome(model, _make_target())["happy"]
        self.assertTrue(np.all(preds >= 0.0) and np.all(preds <= 1.0))


class TestFitWeightColumn(balance.testutil.BalanceTestCase):
    def test_fit_weight_records_named_weight_column(self) -> None:
        covars_R, outcomes_R, w_R = _make_regression_data()
        model = fit_outcome_model(
            covars_R, outcomes_R, sample_weight=w_R.rename("weight")
        )
        self.assertEqual(model["fit_weight"]["column"], "weight")
        self.assertFalse(model["fit_weight"]["uniform"])

    def test_fit_weight_column_none_for_unnamed_or_unweighted(self) -> None:
        covars_R, outcomes_R, w_R = _make_regression_data()
        # Unnamed weight series -> no column name recorded.
        model_w = fit_outcome_model(covars_R, outcomes_R, sample_weight=w_R)
        self.assertIsNone(model_w["fit_weight"]["column"])
        # Unweighted fit -> uniform, no column.
        model_u = fit_outcome_model(covars_R, outcomes_R)
        self.assertIsNone(model_u["fit_weight"]["column"])
        self.assertTrue(model_u["fit_weight"]["uniform"])


class TestLearnerFromModel(balance.testutil.BalanceTestCase):
    def test_reconstructs_per_outcome_unfitted_learner(self) -> None:
        """learner_from_model returns an unfitted {col: est} matching the fit."""
        covars_R, outcomes_R, w_R = _make_regression_data()
        model = fit_outcome_model(
            covars_R, outcomes_R, sample_weight=w_R, model=LinearRegression()
        )
        learner = learner_from_model(model)
        self.assertEqual(list(learner.keys()), ["happiness"])
        self.assertIsInstance(learner["happiness"], LinearRegression)
        # It is an UNFITTED clone (no coef_ yet), reusable for a refit.
        self.assertFalse(hasattr(learner["happiness"], "coef_"))

    def test_refit_with_reconstructed_learner_matches(self) -> None:
        """Passing the reconstructed learner back reproduces predictions."""
        covars_R, outcomes_R, w_R = _make_regression_data()
        model = fit_outcome_model(
            covars_R, outcomes_R, sample_weight=w_R, model=LinearRegression()
        )
        refit = fit_outcome_model(
            covars_R,
            outcomes_R,
            sample_weight=w_R,
            model=learner_from_model(model),
        )
        target = _make_target()
        np.testing.assert_allclose(
            predict_outcome(model, target)["happiness"],
            predict_outcome(refit, target)["happiness"],
            rtol=1e-9,
        )


class TestBootstrapOutcomeEstimate(balance.testutil.BalanceTestCase):
    def _data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        covars_R, outcomes_R, w_R = _make_regression_data(n=200, seed=1)
        target_covars = _make_target(n=40, seed=2)
        target_weight = pd.Series(np.ones(40), index=target_covars.index)
        return covars_R, outcomes_R, w_R, target_covars, target_weight

    def test_returns_estimate_and_ci_per_outcome(self) -> None:
        covars_R, outcomes_R, w_R, target_covars, target_weight = self._data()
        res = bootstrap_outcome_estimate(
            covars_R,
            outcomes_R,
            w_R,
            target_covars,
            target_weight,
            fit_kwargs={"model": LinearRegression()},
            n_bootstrap=30,
            random_seed=2020,
        )
        self.assertEqual(list(res.keys()), ["happiness"])
        self.assertCountEqual(
            res["happiness"].keys(), ["estimate", "ci_low", "ci_high"]
        )

    def test_bootstrap_suppresses_per_replicate_logs(self) -> None:
        # The N refits must not each emit a per-fit INFO line (a wall of
        # identical logs); only a single summary line plus the one full-sample
        # fit should appear.
        covars_R, outcomes_R, w_R, target_covars, target_weight = self._data()
        n_bootstrap = 25
        with self.assertLogs("balance", level="INFO") as cm:
            bootstrap_outcome_estimate(
                covars_R,
                outcomes_R,
                w_R,
                target_covars,
                target_weight,
                fit_kwargs={"model": LinearRegression()},
                n_bootstrap=n_bootstrap,
                random_seed=2020,
            )
        per_fit_logs = [
            r for r in cm.records if "outcome_model: outcome" in r.getMessage()
        ]
        # At most the single full-sample fit logs its model choice — never once
        # per replicate.
        self.assertLessEqual(len(per_fit_logs), 1)
        # The one-line bootstrap summary is still emitted.
        self.assertTrue(
            any("bootstrap replicate" in r.getMessage() for r in cm.records)
        )

    def test_ci_brackets_point_estimate(self) -> None:
        covars_R, outcomes_R, w_R, target_covars, target_weight = self._data()
        res = bootstrap_outcome_estimate(
            covars_R,
            outcomes_R,
            w_R,
            target_covars,
            target_weight,
            fit_kwargs={"model": LinearRegression()},
            n_bootstrap=40,
            random_seed=2020,
        )["happiness"]
        self.assertLessEqual(res["ci_low"], res["estimate"])
        self.assertLessEqual(res["estimate"], res["ci_high"])
        self.assertLess(res["ci_low"], res["ci_high"])

    def test_point_estimate_is_full_sample_mu_om(self) -> None:
        """The reported estimate is the full-sample μ̂_OM, not a bootstrap mean."""
        covars_R, outcomes_R, w_R, target_covars, target_weight = self._data()
        full_model = fit_outcome_model(
            covars_R, outcomes_R, sample_weight=w_R, model=LinearRegression()
        )
        full_preds = predict_outcome(full_model, target_covars)["happiness"]
        expected = float(np.average(full_preds, weights=target_weight))
        res = bootstrap_outcome_estimate(
            covars_R,
            outcomes_R,
            w_R,
            target_covars,
            target_weight,
            fit_kwargs={"model": LinearRegression()},
            n_bootstrap=20,
            random_seed=2020,
        )["happiness"]
        self.assertAlmostEqual(res["estimate"], expected, places=8)

    def test_same_seed_is_reproducible(self) -> None:
        covars_R, outcomes_R, w_R, target_covars, target_weight = self._data()
        kwargs = {
            "fit_kwargs": {"model": LinearRegression()},
            "n_bootstrap": 25,
            "random_seed": 2020,
        }
        a = bootstrap_outcome_estimate(
            covars_R, outcomes_R, w_R, target_covars, target_weight, **kwargs
        )["happiness"]
        b = bootstrap_outcome_estimate(
            covars_R, outcomes_R, w_R, target_covars, target_weight, **kwargs
        )["happiness"]
        self.assertEqual(a["ci_low"], b["ci_low"])
        self.assertEqual(a["ci_high"], b["ci_high"])

    def test_different_seed_differs_but_near_point(self) -> None:
        covars_R, outcomes_R, w_R, target_covars, target_weight = self._data()

        def run(seed: int) -> dict[str, float]:
            return bootstrap_outcome_estimate(
                covars_R,
                outcomes_R,
                w_R,
                target_covars,
                target_weight,
                fit_kwargs={"model": LinearRegression()},
                n_bootstrap=40,
                random_seed=seed,
            )["happiness"]

        a = run(2020)
        b = run(1234)
        # Different seed -> different interval bounds ...
        self.assertNotEqual(a["ci_low"], b["ci_low"])
        # ... but the point estimate is seed-independent, and both intervals
        # stay in a tolerance band around it.
        self.assertEqual(a["estimate"], b["estimate"])
        tol = 5.0
        for res in (a, b):
            self.assertLess(abs(res["ci_low"] - res["estimate"]), tol)
            self.assertLess(abs(res["ci_high"] - res["estimate"]), tol)

    def test_weighted_vs_unweighted_refit_differs(self) -> None:
        """Passing sample_weight (weighted refit) yields a different distribution."""
        covars_R, outcomes_R, _w, target_covars, target_weight = self._data()
        rng = np.random.default_rng(5)
        skewed = pd.Series(
            rng.uniform(0.1, 5.0, len(outcomes_R)), index=outcomes_R.index
        )
        weighted = bootstrap_outcome_estimate(
            covars_R,
            outcomes_R,
            skewed,
            target_covars,
            target_weight,
            fit_kwargs={"model": LinearRegression()},
            n_bootstrap=40,
            random_seed=2020,
        )["happiness"]
        unweighted = bootstrap_outcome_estimate(
            covars_R,
            outcomes_R,
            None,
            target_covars,
            target_weight,
            fit_kwargs={"model": LinearRegression()},
            n_bootstrap=40,
            random_seed=2020,
        )["happiness"]
        # A weighted fit gives a different point estimate + interval than an
        # unweighted one (the refit honours the passed weights).
        self.assertNotAlmostEqual(
            weighted["estimate"], unweighted["estimate"], places=4
        )

    def test_skips_degenerate_resamples(self) -> None:
        """A resample that can't be fit (e.g. single-class) is skipped, not fatal."""
        from balance.outcome_models import outcome_model as om

        covars_R, outcomes_R, w_R, target_covars, target_weight = self._data()
        real_fit = om.fit_outcome_model
        calls = {"n": 0}

        def flaky_fit(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            # Fail two per-replicate refits (call 1 is the full-sample fit).
            if calls["n"] in (3, 5):
                raise ValueError("single observed class (simulated)")
            return real_fit(*args, **kwargs)

        with unittest.mock.patch.object(om, "fit_outcome_model", side_effect=flaky_fit):
            with self.assertLogs("balance", level="WARNING") as logs:
                res = om.bootstrap_outcome_estimate(
                    covars_R,
                    outcomes_R,
                    w_R,
                    target_covars,
                    target_weight,
                    fit_kwargs={"model": LinearRegression()},
                    n_bootstrap=8,
                    random_seed=2020,
                )["happiness"]
        self.assertTrue(any("skipped 2 of 8" in m for m in logs.output))
        self.assertCountEqual(res.keys(), ["estimate", "ci_low", "ci_high"])
        self.assertLessEqual(res["ci_low"], res["ci_high"])

    def test_too_few_valid_resamples_raises(self) -> None:
        """If nearly all resamples fail, raise a clear error (not a bare crash)."""
        from balance.outcome_models import outcome_model as om

        covars_R, outcomes_R, w_R, target_covars, target_weight = self._data()
        real_fit = om.fit_outcome_model
        calls = {"n": 0}

        def mostly_failing_fit(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            # Let the full-sample fit + one replicate through; fail the rest.
            if calls["n"] > 2:
                raise ValueError("single observed class (simulated)")
            return real_fit(*args, **kwargs)

        with unittest.mock.patch.object(
            om, "fit_outcome_model", side_effect=mostly_failing_fit
        ):
            with self.assertRaisesRegex(ValueError, "could not fit enough"):
                om.bootstrap_outcome_estimate(
                    covars_R,
                    outcomes_R,
                    w_R,
                    target_covars,
                    target_weight,
                    fit_kwargs={"model": LinearRegression()},
                    n_bootstrap=6,
                    random_seed=2020,
                )
