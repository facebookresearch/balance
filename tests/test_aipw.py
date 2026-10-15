# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""Tests for the doubly-robust (AIPW) estimate ``μ̂_DR`` (``bf.aipw()``).

Coverage:
  * an **independent numpy oracle** — recomputes ``μ̂_DR`` from hand-rolled OLS
    (not balance's predict path); exact for a linear ``ĝ``;
  * the algebraic identity ``μ̂_DR = μ̂_IPW + (μ̂_OM_target − μ̂_OM_sample,w)``;
  * pure-function vs. frame-method agreement;
  * the uniform-weights reduction to ``μ̂_OM`` (+ warning);
  * error paths (no outcome model / no target).

A separate cross-language check against R lives in ``test_aipw_vs_r.py``.

The fixtures deliberately combine an **unweighted** linear ``ĝ`` with
**non-uniform** balance weights so the augmentation term is non-zero (for a
linear ``ĝ`` fit with the same weights, ``μ̂_DR`` collapses to ``μ̂_OM``).
"""

from __future__ import annotations

import warnings
from unittest import mock

import balance.testutil
import numpy as np
import pandas as pd
from balance.outcome_models import aipw_point_estimate
from balance.outcome_models.aipw import (
    _AIPW_WEIGHT_SUM_RTOL,
    _validate_aipw_weight_scale,
)
from balance.sample_class import Sample
from sklearn.linear_model import LinearRegression, LogisticRegression


def _make_aipw_fixture(
    seed: int = 7, n_sample: int = 300, n_target: int = 200
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Numeric-only responders (with a non-uniform weight) + a target."""
    rng = np.random.default_rng(seed)
    sample_df = pd.DataFrame(
        {
            "id": np.arange(n_sample),
            "x1": rng.normal(50.0, 10.0, n_sample),
            "x2": rng.normal(0.0, 1.0, n_sample),
            "weight": rng.uniform(0.5, 3.0, n_sample),
        }
    )
    sample_df["y"] = (
        3.0
        + 1.5 * sample_df["x1"]
        - 2.0 * sample_df["x2"]
        + rng.normal(0.0, 2.0, n_sample)
    )
    target_df = pd.DataFrame(
        {
            "id": np.arange(n_target),
            "x1": rng.normal(55.0, 12.0, n_target),
            "x2": rng.normal(0.3, 1.0, n_target),
            "weight": rng.uniform(0.5, 3.0, n_target),
        }
    )
    return sample_df, target_df


def _fitted_frame(sample_df: pd.DataFrame, target_df: pd.DataFrame) -> Sample:
    s = Sample.from_frame(
        sample_df, id_column="id", weight_column="weight", outcome_columns=["y"]
    )
    t = Sample.from_frame(target_df, id_column="id", weight_column="weight")
    st = s.set_target(t).adjust(method=_calibrate_existing_weights)
    return st.fit_outcome_model(model=LinearRegression())


def _calibrate_existing_weights(
    sample_df: pd.DataFrame,
    sample_weights: pd.Series,
    target_df: pd.DataFrame,
    target_weights: pd.Series,
) -> dict[str, object]:
    """Preserve relative fixture weights while calibrating to target total."""
    del sample_df, target_df
    calibrated = sample_weights * target_weights.sum() / sample_weights.sum()
    return {"weight": calibrated, "model": {"method": "test_calibration"}}


def _numpy_mu_dr(sample_df: pd.DataFrame, target_df: pd.DataFrame) -> float:
    """Independent μ̂_DR: unweighted OLS ĝ, then the AIPW formula."""
    xs = sample_df[["x1", "x2"]].to_numpy()
    ys = sample_df["y"].to_numpy()
    ws = sample_df["weight"].to_numpy()
    xt = target_df[["x1", "x2"]].to_numpy()
    wt = target_df["weight"].to_numpy()

    design_s = np.column_stack([np.ones(len(xs)), xs])
    beta, *_ = np.linalg.lstsq(design_s, ys, rcond=None)
    ghat_s = design_s @ beta
    ghat_t = np.column_stack([np.ones(len(xt)), xt]) @ beta

    mu_om_target = float(np.average(ghat_t, weights=wt))
    augmentation = float(np.average(ys - ghat_s, weights=ws))
    return mu_om_target + augmentation


class AipwTest(balance.testutil.BalanceTestCase):
    def test_weight_scale_validation_accepts_valid_edge_cases(self) -> None:
        _validate_aipw_weight_scale(np.array([0.0, 1.0]), np.array([0.25, 0.75]))
        _validate_aipw_weight_scale(
            np.array([1.0]),
            np.array([1.0 + _AIPW_WEIGHT_SUM_RTOL / 2]),
        )

    def test_weight_scale_validation_rejects_invalid_vectors(self) -> None:
        invalid_cases = (
            (np.array([]), np.array([1.0]), "non-empty"),
            (np.array([[1.0]]), np.array([1.0]), "one-dimensional"),
            (np.array(["bad"]), np.array([1.0]), "numeric responder"),
            (np.array([np.nan]), np.array([1.0]), "finite responder"),
            (np.array([1.0]), np.array([np.inf]), "finite target"),
            (
                np.array([np.finfo(float).max, np.finfo(float).max]),
                np.array([1.0]),
                "finite responder and target weight totals",
            ),
            (np.array([-1.0, 2.0]), np.array([1.0]), "non-negative responder"),
            (np.array([1.0]), np.array([-1.0, 2.0]), "non-negative target"),
            (np.array([0.0]), np.array([1.0]), "positive responder and target"),
            (np.array([1.0]), np.array([0.0]), "positive responder and target"),
        )
        for sample_weight, target_weight, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _validate_aipw_weight_scale(sample_weight, target_weight)

    def test_weight_scale_validation_rejects_tolerance_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "relative weight-total difference"):
            _validate_aipw_weight_scale(
                np.array([1_000_001.0]), np.array([1_000_000.0])
            )

    def test_weight_scale_validation_rejects_nonfinite_computed_totals(self) -> None:
        with mock.patch(
            "balance.outcome_models.aipw.math.fsum", return_value=float("inf")
        ):
            with self.assertRaisesRegex(
                ValueError, "finite responder and target weight totals"
            ):
                _validate_aipw_weight_scale(np.array([1.0]), np.array([1.0]))

    def test_aipw_matches_independent_numpy_oracle(self) -> None:
        sample_df, target_df = _make_aipw_fixture()
        dr = _fitted_frame(sample_df, target_df).aipw()
        self.assertAlmostEqual(
            float(dr["y"]), _numpy_mu_dr(sample_df, target_df), places=8
        )

    def test_aipw_algebraic_identity(self) -> None:
        """μ̂_DR == μ̂_IPW + (μ̂_OM_target − μ̂_OM_sample,w)."""
        sample_df, target_df = _make_aipw_fixture()
        st = _fitted_frame(sample_df, target_df)

        xs = sample_df[["x1", "x2"]].to_numpy()
        ys = sample_df["y"].to_numpy()
        ws = sample_df["weight"].to_numpy()
        beta, *_ = np.linalg.lstsq(
            np.column_stack([np.ones(len(xs)), xs]), ys, rcond=None
        )
        ghat_s = np.column_stack([np.ones(len(xs)), xs]) @ beta
        mu_ipw = float(np.average(ys, weights=ws))
        mu_om_sample_w = float(np.average(ghat_s, weights=ws))
        mu_om_target = _numpy_mu_dr(sample_df, target_df) - float(
            np.average(ys - ghat_s, weights=ws)
        )

        self.assertAlmostEqual(
            float(st.aipw()["y"]),
            mu_ipw + (mu_om_target - mu_om_sample_w),
            places=8,
        )

    def test_pure_function_matches_frame_method(self) -> None:
        sample_df, target_df = _make_aipw_fixture()
        st = _fitted_frame(sample_df, target_df)
        model = st.outcome_model
        assert model is not None
        pure = aipw_point_estimate(
            sample_df[["x1", "x2"]],
            sample_df[["y"]],
            sample_df["weight"],
            target_df[["x1", "x2"]],
            target_df["weight"],
            model,
        )
        self.assertAlmostEqual(float(st.aipw()["y"]), pure["y"], places=10)

    def test_cross_fitted_aipw_is_deterministic(self) -> None:
        sample_df, target_df = _make_aipw_fixture()
        st = _fitted_frame(sample_df, target_df)

        first = st.aipw(n_folds=5, random_seed=123)
        second = st.aipw(n_folds=5, random_seed=123)

        self.assertAlmostEqual(float(first["y"]), float(second["y"]), places=12)
        self.assertTrue(np.isfinite(first["y"]))

    def test_cross_fitted_aipw_validates_fold_count(self) -> None:
        sample_df, target_df = _make_aipw_fixture(n_sample=10)
        st = _fitted_frame(sample_df, target_df)

        for invalid in (True, 1, 11):
            with self.subTest(n_folds=invalid):
                with self.assertRaisesRegex(ValueError, "n_folds"):
                    st.aipw(n_folds=invalid)

        for invalid_seed in (True, -1, 1.5, 2**32):
            with self.subTest(random_seed=invalid_seed):
                with self.assertRaisesRegex(ValueError, "random_seed"):
                    st.aipw(n_folds=2, random_seed=invalid_seed)

    def test_cross_fitted_aipw_validates_aligned_inputs(self) -> None:
        sample_df, target_df = _make_aipw_fixture(n_sample=10, n_target=5)
        st = _fitted_frame(sample_df, target_df)
        model = st.outcome_model
        assert model is not None
        from balance.outcome_models.outcome_model import learner_from_model

        common = {
            "sample_covars": st._sf_sample.df_covars,
            "outcomes": st._outcome_columns,
            "sample_weight": st.weight_series,
            "target_covars": st._sf_target.df_covars,
            "target_weight": st._sf_target.weight_series,
            "fit_kwargs": {"model": learner_from_model(model)},
            "n_folds": 2,
        }
        invalid_overrides = (
            ({"outcomes": st._outcome_columns.iloc[:-1]}, "same number of rows"),
            ({"sample_weight": np.ones(9)}, "sample_weight"),
            ({"sample_weight": np.array(["bad"] * 10)}, "numeric"),
            ({"target_weight": np.ones((5, 1))}, "target_weight"),
            ({"fit_sample_weight": np.ones(9)}, "fit_sample_weight"),
            ({"fit_sample_weight": np.zeros(10)}, "strictly positive"),
            ({"fit_kwargs": {"sample_weight": np.ones(10)}}, "must not contain"),
            (
                {
                    "outcomes": st._outcome_columns.assign(y=np.inf),
                },
                "finite observed",
            ),
        )
        from balance.outcome_models import cross_fitted_aipw_point_estimate

        for override, message in invalid_overrides:
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, message):
                    cross_fitted_aipw_point_estimate(**(common | override))

        outcomes_with_missing = st._outcome_columns.copy()
        outcomes_with_missing.loc[:4, "y"] = np.nan
        zero_observed_weight = np.r_[np.ones(5), np.zeros(5)]
        target_weight = np.full(5, zero_observed_weight.sum() / 5)
        with self.assertRaisesRegex(ValueError, "positive total AIPW residual"):
            cross_fitted_aipw_point_estimate(
                **(
                    common
                    | {
                        "outcomes": outcomes_with_missing,
                        "sample_weight": zero_observed_weight,
                        "target_weight": target_weight,
                    }
                )
            )

    def test_cross_fitted_aipw_supports_missing_outcomes(self) -> None:
        sample_df, target_df = _make_aipw_fixture()
        sample_df.loc[[1, 17, 42], "y"] = np.nan
        st = _fitted_frame(sample_df.dropna(), target_df)
        # Restore the incomplete responder outcome after fitting so this tests
        # the per-outcome observed-row folds used by the AIPW augmentation.
        st._sf_sample._df.loc[st._sf_sample._df.index[:3], "y"] = np.nan

        estimate = st.aipw(n_folds=3, random_seed=1)

        self.assertTrue(np.isfinite(estimate["y"]))

    def test_cross_fitted_aipw_handles_each_outcome_missingness_independently(
        self,
    ) -> None:
        sample_df, target_df = _make_aipw_fixture()
        sample_df["z"] = 2.0 * sample_df["y"] + 1.0
        sample = Sample.from_frame(
            sample_df,
            id_column="id",
            weight_column="weight",
            outcome_columns=["y", "z"],
        )
        target = Sample.from_frame(target_df, id_column="id", weight_column="weight")
        st = (
            sample.set_target(target)
            .adjust(method=_calibrate_existing_weights)
            .fit_outcome_model(model=LinearRegression())
        )
        st._sf_sample._df.loc[st._sf_sample._df.index[:3], "y"] = np.nan
        st._sf_sample._df.loc[st._sf_sample._df.index[3:7], "z"] = np.nan

        estimate = st.aipw(n_folds=5, random_seed=11)

        self.assertEqual(set(estimate.index), {"y", "z"})
        self.assertTrue(np.isfinite(estimate.to_numpy()).all())

    def test_cross_fitted_binary_outcome_with_rare_class_is_warning_free(self) -> None:
        sample_df, target_df = _make_aipw_fixture(n_sample=30, n_target=20)
        sample_df["y"] = 0
        sample_df.loc[[0, 1], "y"] = 1
        sample = Sample.from_frame(
            sample_df,
            id_column="id",
            weight_column="weight",
            outcome_columns=["y"],
        )
        target = Sample.from_frame(target_df, id_column="id", weight_column="weight")
        st = (
            sample.set_target(target)
            .adjust(method=_calibrate_existing_weights)
            .fit_outcome_model(model=LogisticRegression(max_iter=500))
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            estimate = st.aipw(n_folds=5, random_seed=7)

        self.assertTrue(np.isfinite(estimate["y"]))

    def test_pure_function_is_index_independent(self) -> None:
        """μ̂_DR is unchanged when the (already row-aligned) sample covariates,
        outcomes, and weights carry a shuffled, non-default index — the residual
        must not silently misalign Y against ĝ(X_S)."""
        sample_df, target_df = _make_aipw_fixture()
        model = _fitted_frame(sample_df, target_df).outcome_model
        assert model is not None

        sample_covars = sample_df[["x1", "x2"]]
        outcomes = sample_df[["y"]]
        sample_weight = sample_df["weight"]
        target_covars = target_df[["x1", "x2"]]
        target_weight = target_df["weight"]

        aligned = aipw_point_estimate(
            sample_covars,
            outcomes,
            sample_weight,
            target_covars,
            target_weight,
            model,
        )

        # Relabel the sample inputs with a shuffled, non-default index WITHOUT
        # reordering rows; a caller whose frames merely carry odd index labels
        # must get the identical estimate.
        shuffled_index = pd.Index(np.random.default_rng(0).permutation(len(sample_df)))
        shuffled = aipw_point_estimate(
            sample_covars.set_axis(shuffled_index, axis=0),
            outcomes.set_axis(shuffled_index, axis=0),
            sample_weight.set_axis(shuffled_index),
            target_covars,
            target_weight,
            model,
        )
        self.assertAlmostEqual(aligned["y"], shuffled["y"], places=12)

    def test_all_nan_outcome_column_returns_nan_and_warns(self) -> None:
        """A column with no observed responder outcome -> NaN μ̂_DR + a WARNING."""
        sample_df, target_df = _make_aipw_fixture()
        model = _fitted_frame(sample_df, target_df).outcome_model
        assert model is not None

        outcomes = sample_df[["y"]].copy()
        outcomes["y"] = np.nan  # no observed (non-NaN) responder outcome
        with self.assertLogs("balance", level="WARNING") as cm:
            result = aipw_point_estimate(
                sample_df[["x1", "x2"]],
                outcomes,
                sample_df["weight"],
                target_df[["x1", "x2"]],
                target_df["weight"],
                model,
            )
        self.assertTrue(np.isnan(result["y"]))
        self.assertTrue(any("no observed" in m for m in cm.output))

    def test_uniform_weights_reduces_to_om_and_warns(self) -> None:
        sample_df, target_df = _make_aipw_fixture()
        sample_df["weight"] = 1.0  # uniform -> no weighting fitted
        st = _fitted_frame(sample_df, target_df)

        # For a linear ĝ with intercept and uniform responder weights the
        # augmentation is exactly 0, so μ̂_DR collapses to the target μ̂_OM
        # (here _numpy_mu_dr returns that target term, its augmentation being 0).
        mu_om_target = _numpy_mu_dr(sample_df, target_df)
        with self.assertLogs("balance", level="WARNING") as cm:
            dr = st.aipw()
        self.assertTrue(any("reduces to" in m for m in cm.output))
        self.assertAlmostEqual(float(dr["y"]), mu_om_target, places=8)

    def test_aipw_requires_outcome_model(self) -> None:
        sample_df, target_df = _make_aipw_fixture()
        s = Sample.from_frame(
            sample_df, id_column="id", weight_column="weight", outcome_columns=["y"]
        )
        t = Sample.from_frame(target_df, id_column="id", weight_column="weight")
        st = s.set_target(t)
        with self.assertRaisesRegex(ValueError, "requires a fitted outcome model"):
            st.aipw()

    def test_aipw_requires_target(self) -> None:
        sample_df, _target_df = _make_aipw_fixture()
        s = Sample.from_frame(
            sample_df, id_column="id", weight_column="weight", outcome_columns=["y"]
        )
        s = s.fit_outcome_model(model=LinearRegression())
        with self.assertRaisesRegex(ValueError, "requires a target"):
            s.aipw()

    def test_aipw_requires_adjusted_weights(self) -> None:
        sample_df, target_df = _make_aipw_fixture()
        s = Sample.from_frame(
            sample_df, id_column="id", weight_column="weight", outcome_columns=["y"]
        )
        t = Sample.from_frame(target_df, id_column="id", weight_column="weight")
        st = s.set_target(t).fit_outcome_model(model=LinearRegression())
        with self.assertRaisesRegex(ValueError, "adjust\\(\\)-calibrated"):
            st.aipw()

    def test_aipw_rejects_adjusted_weights_on_different_scale(self) -> None:
        sample_df, target_df = _make_aipw_fixture()
        st = _fitted_frame(sample_df, target_df)
        st._sf_sample._df.loc[:, st.weight_column] *= 2.0
        with self.assertRaisesRegex(ValueError, "same population scale"):
            st.aipw()

    def test_summary_shows_ipw_om_dr_when_model_fit(self) -> None:
        sample_df, target_df = _make_aipw_fixture()
        summary = _fitted_frame(sample_df, target_df).summary()
        self.assertIn("Outcome estimates:", summary)
        self.assertIn("mu_IPW", summary)
        self.assertIn("mu_OM", summary)
        self.assertIn("mu_DR", summary)
        # the rich section replaces the plain "Outcome weighted means" block
        self.assertNotIn("Outcome weighted means", summary)

    def test_summary_unchanged_without_outcome_model(self) -> None:
        sample_df, target_df = _make_aipw_fixture()
        s = Sample.from_frame(
            sample_df, id_column="id", weight_column="weight", outcome_columns=["y"]
        )
        t = Sample.from_frame(target_df, id_column="id", weight_column="weight")
        st = s.set_target(t)  # target set, but NO outcome model fit
        summary = st.summary()
        self.assertIn("Outcome weighted means", summary)
        self.assertNotIn("Outcome estimates:", summary)
        self.assertNotIn("mu_DR", summary)
