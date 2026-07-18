# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Independent-oracle validation for the outcome-model estimate ``μ̂_OM``.

The existing ``test_outcome_model.py`` checks ``μ̂_OM`` mostly against
``np.average(predict_outcome(...), weights=w_T)`` — i.e. against the *same*
prediction path it is testing (self-referential), plus small hand arithmetic.
This module adds an **independent** oracle: it recomputes the g-computation /
regression-adjustment transport estimator

    μ̂_OM = weighted_mean_over_target( ĝ(X_T) ),   ĝ(X) = E[Y|X] fit on responders

with a hand-rolled OLS / WLS solved via the normal equations (numpy only), and
asserts balance's ``fit_outcome_model(LinearRegression()) -> predict_outcome ->
weighted_mean`` matches it to machine precision.

Why this is exact for a linear learner: balance's linear path fits OLS on a
one-hot + ``StandardScaler`` design. Standardizing the columns is an invertible
linear reparametrization that is absorbed into the coefficients, so (with the
default intercept) the *fitted linear function* — and therefore every prediction
on the target — is identical to plain OLS on the raw covariates. The data below
is numeric with no missing values, so the design is unambiguous and the match is
exact (``atol=1e-8``), not merely approximate.

This is the Python-only counterpart to the external R cross-check (see
``r_oracles/generate_outcome_model_fixture.R``): ``stdReg2::standardize_glm`` /
``marginaleffects::avg_predictions`` compute the same estimand and agree with
this oracle; keeping a pure-Python version means the guarantee is enforced in CI
without an R dependency.
"""

from __future__ import annotations

import balance.testutil
import numpy as np
import pandas as pd
from balance.outcome_models import fit_outcome_model, predict_outcome
from balance.stats_and_plots.weighted_stats import weighted_mean
from sklearn.linear_model import LinearRegression


def _make_linear_fixture(
    seed: int = 42,
    n_responders: int = 400,
    n_target: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Numeric-only responders + a differently-distributed target.

    Returns ``(covars_R, outcomes_R, w_R, covars_T, w_T)``. The outcome is a
    linear function of the covariates plus noise; the target covariates are drawn
    from shifted distributions so ``μ̂_OM`` genuinely extrapolates the fit.
    """
    rng = np.random.default_rng(seed)
    covars_r = pd.DataFrame(
        {
            "x1": rng.normal(50.0, 10.0, n_responders),
            "x2": rng.normal(0.0, 1.0, n_responders),
            "x3": rng.uniform(-2.0, 2.0, n_responders),
        }
    )
    y = (
        3.0
        + 1.5 * covars_r["x1"]
        - 2.0 * covars_r["x2"]
        + 0.7 * covars_r["x3"]
        + rng.normal(0.0, 2.0, n_responders)
    )
    outcomes_r = pd.DataFrame({"y": y.to_numpy()})
    w_r = pd.Series(rng.uniform(0.5, 2.0, n_responders), index=covars_r.index)

    covars_t = pd.DataFrame(
        {
            "x1": rng.normal(55.0, 12.0, n_target),
            "x2": rng.normal(0.3, 1.0, n_target),
            "x3": rng.uniform(-2.0, 2.0, n_target),
        }
    )
    w_t = pd.Series(rng.uniform(0.5, 2.0, n_target), index=covars_t.index)
    return covars_r, outcomes_r, w_r, covars_t, w_t


def _ols_beta(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Independent OLS/WLS coefficients via the (weighted) normal equations."""
    design = np.column_stack([np.ones(len(x)), x])
    if w is None:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        return beta
    wd = design * w[:, None]
    return np.linalg.solve(design.T @ wd, wd.T @ y)


class OutcomeModelOracleTest(balance.testutil.BalanceTestCase):
    def test_muom_matches_independent_ols_unweighted(self) -> None:
        covars_r, outcomes_r, _w_r, covars_t, w_t = _make_linear_fixture()

        model = fit_outcome_model(covars_r, outcomes_r, model=LinearRegression())
        preds = predict_outcome(model, covars_t)["y"]
        mu_balance = float(weighted_mean(pd.Series(preds), w_t).iloc[0])

        beta = _ols_beta(covars_r.to_numpy(), outcomes_r["y"].to_numpy())
        preds_oracle = (
            np.column_stack([np.ones(len(covars_t)), covars_t.to_numpy()]) @ beta
        )
        mu_oracle = float(np.average(preds_oracle, weights=w_t.to_numpy()))

        self.assertTrue(np.allclose(preds, preds_oracle, atol=1e-8))
        self.assertAlmostEqual(mu_balance, mu_oracle, places=8)

    def test_muom_matches_independent_wls_weighted(self) -> None:
        covars_r, outcomes_r, w_r, covars_t, w_t = _make_linear_fixture()

        model = fit_outcome_model(
            covars_r, outcomes_r, sample_weight=w_r, model=LinearRegression()
        )
        preds = predict_outcome(model, covars_t)["y"]
        mu_balance = float(weighted_mean(pd.Series(preds), w_t).iloc[0])

        beta = _ols_beta(
            covars_r.to_numpy(), outcomes_r["y"].to_numpy(), w_r.to_numpy()
        )
        preds_oracle = (
            np.column_stack([np.ones(len(covars_t)), covars_t.to_numpy()]) @ beta
        )
        mu_oracle = float(np.average(preds_oracle, weights=w_t.to_numpy()))

        self.assertTrue(np.allclose(preds, preds_oracle, atol=1e-8))
        self.assertAlmostEqual(mu_balance, mu_oracle, places=8)
