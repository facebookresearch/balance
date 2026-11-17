# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""R paired t-test oracle on actual LaLonde outcomes and MatchIt weights.

Source MatchIt 4.4.0 data/lalonde.tab, GPL (>=2), data references/notices and
hashes in fixtures/weight_impact/. R stats::t.test(paired=TRUE) supplies every result field;
only mean-normalization of the supplied weights matches the documented balance
estimator. No R design-based causal uncertainty is claimed equivalent.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from balance import Sample
from balance.stats_and_plots.impact_of_weights_on_outcome import (
    compare_adjusted_weighted_outcome_ss,
    weights_impact_on_outcome_ss,
)

# Scientific-stack stubs omit this supported runtime parser literal.
_ROUND_TRIP: Any = "round_trip"


class RWeightImpactOracleTest(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).parent / "fixtures" / "weight_impact"
        self.inputs = pd.read_csv(
            root / "matchit_lalonde_weight_impact_inputs.csv",
            float_precision=_ROUND_TRIP,
        )
        self.expected = pd.read_csv(
            root / "matchit_lalonde_weight_impact_expected.csv",
            float_precision=_ROUND_TRIP,
        ).iloc[0]

    def test_mean_interval_t_statistic_and_p_value_match_r_paired_test(self) -> None:
        result = weights_impact_on_outcome_ss(
            self.inputs.y, self.inputs.w0, self.inputs.w1
        )
        self.assertEqual(result.index.tolist(), self.expected.index.tolist())
        np.testing.assert_allclose(result, self.expected, atol=1e-8, rtol=1e-12)
        self.assertEqual(int(result["n"]), 429)

    def test_adjusted_sample_comparison_matches_the_same_unrounded_r_result(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "id": self.inputs.row_id.astype(str),
                "position": self.inputs.row_id.astype(float),
                "y": self.inputs.y,
                "weight": self.inputs.w0,
            }
        )
        sample = Sample.from_frame(
            frame, id_column="id", weight_column="weight", outcome_columns=["y"]
        )
        target = Sample.from_frame(
            frame.drop(columns="y"), id_column="id", weight_column="weight"
        )
        linked = sample.set_target(target)
        before = linked.adjust(method="null")
        after = linked.adjust(method="null")
        after.set_weights(pd.Series(self.inputs.w1.to_numpy(), index=after.df.index))
        result = compare_adjusted_weighted_outcome_ss(before, after, round_ndigits=None)
        self.assertEqual(result.index.tolist(), ["y"])
        self.assertEqual(result.columns.tolist(), self.expected.index.tolist())
        np.testing.assert_allclose(
            result.loc["y"], self.expected, atol=1e-8, rtol=1e-12
        )
