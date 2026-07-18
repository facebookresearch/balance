# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Cross-language validation of ``μ̂_OM`` against an external R oracle.

Companion to ``test_outcome_model_oracle.py`` (the pure-Python independent
oracle). This test consumes a fixture generated in R by
``r_oracles/generate_outcome_model_fixture.R`` — data plus the R-computed
``μ̂_OM`` (via ``lm -> predict(newdata=target) -> weighted.mean``, the regression-
standardization / transport estimator, cross-checked with
``marginaleffects::avg_predictions``). It asserts balance reproduces the R value.

Mirrors the ``test_cbps_in_balance_vs_r`` precedent, but because the estimator is
exactly reproducible for a linear learner the check is an exact ``np.allclose``
(``atol=1e-6``), not a correlation threshold.

The fixture CSVs are checked into ``datasets/``; the test skips only if they are
absent. To regenerate them::

    Rscript parent_balance/tests/r_oracles/generate_outcome_model_fixture.R
    mv sim_data_outcome_model*.csv core_stats/balance/datasets/
"""

from __future__ import annotations

import pathlib
import unittest

import balance.testutil
import numpy as np
import pandas as pd
from balance.outcome_models import fit_outcome_model, predict_outcome
from balance.stats_and_plots.weighted_stats import weighted_mean
from sklearn.linear_model import LinearRegression

_DATASETS_DIR: pathlib.Path = pathlib.Path(balance.__file__).parent.joinpath("datasets")
_DATA_CSV: pathlib.Path = _DATASETS_DIR.joinpath("sim_data_outcome_model.csv")
_EXPECTED_CSV: pathlib.Path = _DATASETS_DIR.joinpath(
    "sim_data_outcome_model_expected.csv"
)
_FIXTURE_AVAILABLE: bool = _DATA_CSV.exists() and _EXPECTED_CSV.exists()


@unittest.skipUnless(
    _FIXTURE_AVAILABLE,
    "R outcome-model fixture not generated; run r_oracles/"
    "generate_outcome_model_fixture.R and move the CSVs into datasets/.",
)
class OutcomeModelVsRTest(balance.testutil.BalanceTestCase):
    def _load(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, float]]:
        df = pd.read_csv(_DATA_CSV)
        sample = df[df.is_target == 0]
        target = df[df.is_target == 1]
        covars_cols = ["x1", "x2", "x3"]
        expected_df = pd.read_csv(_EXPECTED_CSV)
        expected = dict(zip(expected_df.estimator, expected_df.mu_OM))
        return (
            sample[covars_cols].reset_index(drop=True),
            target[covars_cols].reset_index(drop=True),
            target["w"].reset_index(drop=True),
            expected,
        )

    def _mu_om(
        self,
        covars_r: pd.DataFrame,
        outcomes_r: pd.DataFrame,
        covars_t: pd.DataFrame,
        w_t: pd.Series,
        sample_weight: pd.Series | None,
    ) -> float:
        model = fit_outcome_model(
            covars_r, outcomes_r, sample_weight=sample_weight, model=LinearRegression()
        )
        preds = predict_outcome(model, covars_t)["y"]
        return float(weighted_mean(pd.Series(preds), w_t).iloc[0])

    def test_muom_unweighted_matches_r(self) -> None:
        df = pd.read_csv(_DATA_CSV)
        sample = df[df.is_target == 0].reset_index(drop=True)
        covars_r, covars_t, w_t, expected = self._load()
        outcomes_r = pd.DataFrame({"y": sample["y"].to_numpy()})

        mu = self._mu_om(covars_r, outcomes_r, covars_t, w_t, sample_weight=None)
        self.assertTrue(
            np.allclose(mu, expected["mu_OM_unweighted_fit"], atol=1e-6),
            msg=f"balance μ̂_OM {mu} != R {expected['mu_OM_unweighted_fit']}",
        )

    def test_muom_weighted_matches_r(self) -> None:
        df = pd.read_csv(_DATA_CSV)
        sample = df[df.is_target == 0].reset_index(drop=True)
        covars_r, covars_t, w_t, expected = self._load()
        outcomes_r = pd.DataFrame({"y": sample["y"].to_numpy()})
        w_r = sample["w"].reset_index(drop=True)

        mu = self._mu_om(covars_r, outcomes_r, covars_t, w_t, sample_weight=w_r)
        self.assertTrue(
            np.allclose(mu, expected["mu_OM_weighted_fit"], atol=1e-6),
            msg=f"balance μ̂_OM {mu} != R {expected['mu_OM_weighted_fit']}",
        )
