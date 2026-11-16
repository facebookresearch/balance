# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compare weights with executed survey 4.1-1 results on its exact fixtures.

Inputs: third-party/R/survey/tests/kalton.R and tests/badcal.R.
Expected values: installed survey::rake/postStratify/calibrate/svytable;
Kalton's rounded ratios also reproduce tests/kalton.Rout.save.
Source license: GPL-2 | GPL-3; copyright 2002-2014 Thomas Lumley, except
as stated in inst/COPYING. Kalton & Flores-Cervantes (2003), Weighting methods,
J Official Stat 19(2), 81-97; badcal.R credits Takahiro Tsuchiya's bug report.
Exact source hashes, runtime, notices and license texts: fixtures/weighting/.
The Python harness is newly written; the fixture values are reproduced from R.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from balance.weighting_methods.poststratify import poststratify
from balance.weighting_methods.rake import rake


class SurveyInspiredWeightingTest(unittest.TestCase):
    def test_joint_cells_and_margins_have_different_expected_weights(self) -> None:
        fixture = pd.read_csv(
            Path(__file__).parent / "fixtures" / "weighting" / "survey_kalton.csv",
            dtype={"a": str, "b": str},
        )
        sample_rows = np.repeat(np.arange(12), fixture.sample_count)
        target_rows = np.repeat(np.arange(12), fixture.target_count)
        sample = fixture.iloc[sample_rows][["a", "b"]].reset_index(drop=True)
        target = fixture.iloc[target_rows][["a", "b"]].reset_index(drop=True)
        design = pd.Series(1.0, index=sample.index)
        population = pd.Series(1.0, index=target.index)
        joint = poststratify(sample, design, target, population, transformations=None)
        marginal = rake(
            sample,
            design,
            target,
            population,
            transformations=None,
            convergence_rate=1e-10,
            rate_tolerance=1e-12,
        )
        expected_joint = (fixture.poststratify_total / fixture.sample_count).to_numpy()[
            sample_rows
        ]
        expected_rake = fixture.rake_multiplier.to_numpy()[sample_rows]
        np.testing.assert_allclose(joint["weight"], expected_joint, atol=1e-8, rtol=0)
        np.testing.assert_allclose(marginal["weight"], expected_rake, atol=1e-9, rtol=0)
        np.testing.assert_array_equal(
            np.round(marginal["weight"].to_numpy(), 2),
            fixture.recorded_rake_multiplier.to_numpy()[sample_rows],
        )
        self.assertEqual(marginal["model"]["converged"], 1)
        self.assertFalse(np.allclose(marginal["weight"], joint["weight"]))
        pd.testing.assert_index_equal(joint["weight"].index, sample.index)
        pd.testing.assert_index_equal(marginal["weight"].index, sample.index)
        self.assertEqual(len(sample), 1000)
        self.assertEqual(len(target), 1500)
        self.assertAlmostEqual(float(marginal["weight"].sum()), 1500.0)

    def test_weighting_is_independent_of_initial_weight_scale(self) -> None:
        fixture = pd.read_csv(
            Path(__file__).parent / "fixtures" / "weighting" / "survey_badcal.csv",
            dtype={"x": str},
        )
        sample = fixture[["x"]]
        target = pd.DataFrame({"x": ["1", "2"]})
        # Original calibrate population=(10000,5000): intercept and x=2 total.
        totals = pd.Series([5000.0, 5000.0])
        for method in (poststratify, rake):
            for scale in (1e-6, 1.0, 1e6):
                with self.subTest(method=method.__name__, scale=scale):
                    result = method(
                        sample,
                        fixture.design_weight * scale,
                        target,
                        pd.Series(totals.to_numpy()),
                        transformations=None,
                    )
                    np.testing.assert_allclose(
                        result["weight"], fixture.expected_weight, atol=1e-7, rtol=0
                    )
                    self.assertAlmostEqual(float(result["weight"].sum()), 10000.0)

    def test_target_scale_changes_weights_but_not_cell_proportions(self) -> None:
        fixture = pd.read_csv(
            Path(__file__).parent / "fixtures" / "weighting" / "survey_badcal.csv",
            dtype={"x": str},
        )
        target = pd.DataFrame({"x": ["1", "2"]})
        # Original calibrate population=(10000,5000): intercept and x=2 total.
        totals = pd.Series([5000.0, 5000.0])
        for method in (poststratify, rake):
            with self.subTest(method=method.__name__):
                result = method(
                    fixture[["x"]],
                    fixture.design_weight,
                    target,
                    pd.Series(totals.to_numpy()) * 7,
                    transformations=None,
                )
                np.testing.assert_allclose(
                    result["weight"], fixture.expected_weight * 7, atol=1e-6, rtol=0
                )
                np.testing.assert_allclose(
                    result["weight"].groupby(fixture.x).sum() / result["weight"].sum(),
                    (totals / totals.sum()).to_numpy(),
                )
