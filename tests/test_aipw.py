# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

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

import balance.testutil
import numpy as np
import pandas as pd
from balance.outcome_models import aipw_point_estimate
from balance.sample_class import Sample
from sklearn.linear_model import LinearRegression


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
    st = s.set_target(t)
    return st.fit_outcome_model(model=LinearRegression())


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
