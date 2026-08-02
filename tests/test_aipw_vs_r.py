# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Cross-language validation of ``μ̂_DR`` (``bf.aipw()``) against an R oracle.

Consumes a fixture generated in R by
``r_oracles/generate_aipw_fixture.R`` — data (with given balance weights) plus
the R-computed doubly-robust estimate ``μ̂_DR`` (base-R ``lm`` for ``ĝ`` then the
augmentation identity). Asserts balance's ``aipw()`` reproduces the R value
exactly (linear ``ĝ`` ⇒ ``atol=1e-6``, cross-language).

Skips if the fixture CSVs are absent from ``datasets/``.
"""

from __future__ import annotations

import pathlib
import unittest

import balance.testutil
import numpy as np
import pandas as pd
from balance.sample_class import Sample
from sklearn.linear_model import LinearRegression

_DATASETS_DIR: pathlib.Path = pathlib.Path(balance.__file__).parent.joinpath("datasets")
_DATA_CSV: pathlib.Path = _DATASETS_DIR.joinpath("sim_data_aipw.csv")
_EXPECTED_CSV: pathlib.Path = _DATASETS_DIR.joinpath("sim_data_aipw_expected.csv")
_FIXTURE_AVAILABLE: bool = _DATA_CSV.exists() and _EXPECTED_CSV.exists()


@unittest.skipUnless(
    _FIXTURE_AVAILABLE,
    "R AIPW fixture not generated; run r_oracles/generate_aipw_fixture.R and "
    "move the CSVs into datasets/.",
)
class AipwVsRTest(balance.testutil.BalanceTestCase):
    def test_mu_dr_matches_r(self) -> None:
        df = pd.read_csv(_DATA_CSV)
        sample = df[df.is_target == 0].reset_index(drop=True)
        target = df[df.is_target == 1].reset_index(drop=True)
        expected_df = pd.read_csv(_EXPECTED_CSV)
        expected = dict(zip(expected_df.estimator, expected_df.value))

        sample = sample.assign(id=np.arange(len(sample)))
        target = target.assign(id=np.arange(len(target)))
        s = Sample.from_frame(
            sample[["id", "x1", "x2", "y", "w"]],
            id_column="id",
            weight_column="w",
            outcome_columns=["y"],
        )
        t = Sample.from_frame(
            target[["id", "x1", "x2", "w"]], id_column="id", weight_column="w"
        )

        def calibrate(
            sample_df: pd.DataFrame,
            sample_weights: pd.Series,
            target_df: pd.DataFrame,
            target_weights: pd.Series,
        ) -> dict[str, object]:
            del sample_df, target_df
            return {
                "weight": sample_weights * target_weights.sum() / sample_weights.sum(),
                "model": {"method": "test_calibration"},
            }

        st = s.set_target(t).adjust(method=calibrate)
        # UNWEIGHTED ĝ, matching the R oracle's lm().
        st = st.fit_outcome_model(model=LinearRegression())

        mu_dr = float(st.aipw()["y"])
        self.assertTrue(
            np.allclose(mu_dr, expected["mu_DR"], atol=1e-6),
            msg=f"balance μ̂_DR {mu_dr} != R {expected['mu_DR']}",
        )
