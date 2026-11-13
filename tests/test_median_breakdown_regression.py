# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import numpy as np
from balance.stats_and_plots.weights_stats import weighted_median_breakdown_point


class MedianBreakdownRegressionTest(unittest.TestCase):
    def test_counts_crossing_observation(self) -> None:
        self.assertAlmostEqual(weighted_median_breakdown_point([4, 3, 3]), 2 / 3)
        self.assertAlmostEqual(weighted_median_breakdown_point([1, 1, 1]), 2 / 3)
        self.assertEqual(weighted_median_breakdown_point([1, 1, 1, 1]), 0.5)
        self.assertAlmostEqual(weighted_median_breakdown_point([6, 2, 2]), 1 / 3)

    def test_uniform_weights_need_half_the_rows(self) -> None:
        """Equal weights reach 50% after exactly n/2 rows, for any n and scale
        (normalized running sums such as 6 * (1/12) round just below 0.5)."""
        for n in (12, 14, 20, 38):
            for scale in (1.0, 0.1, 3.0, 1e-3):
                with self.subTest(n=n, scale=scale):
                    self.assertEqual(
                        weighted_median_breakdown_point(np.full(n, scale)), 0.5
                    )

    def test_result_is_invariant_to_rescaling(self) -> None:
        """Multiplying every weight by a constant does not change the result."""
        for scale in (1.0, 0.1, 7.0, 1e-6):
            with self.subTest(scale=scale):
                self.assertAlmostEqual(
                    weighted_median_breakdown_point(np.array([4.0, 3.0, 3.0]) * scale),
                    2 / 3,
                )

    def test_exact_half_counts_and_just_below_half_needs_next_row(self) -> None:
        """A row reaching exactly half the total suffices; one falling short of
        half by a tiny amount needs the next row."""
        self.assertAlmostEqual(weighted_median_breakdown_point([2.0, 1.0, 1.0]), 1 / 3)
        self.assertAlmostEqual(
            weighted_median_breakdown_point([2.0 - 1e-12, 1.0, 1.0]), 2 / 3
        )
