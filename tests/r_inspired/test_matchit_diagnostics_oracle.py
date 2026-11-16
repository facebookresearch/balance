# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""MatchIt 4.4.0 summary-table oracles on its actual 614-row LaLonde data.

Source: third-party/R/MatchIt/data/lalonde.tab and assessing-balance vignette.
GPL (>=2); package authors/source notices and original data references are in
fixtures/diagnostics/README.md. Independently executed R tables reproduce the maintainer
HTML. Only absolute sign is applied to continuous SMD; binary SMD conversion
and eCDF Mean are deliberately not asserted as raw balance equivalents.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from balance import Sample
from balance.stats_and_plots.weighted_comparisons_stats import (
    asmd,
    ks,
    outcome_variance_ratio,
)
from balance.stats_and_plots.weighted_stats import weighted_mean


# Scientific-stack stubs omit this supported runtime parser literal.
_ROUND_TRIP: Any = "round_trip"


class MatchItDiagnosticsOracleTest(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).parent / "fixtures" / "diagnostics"
        self.data = pd.read_csv(
            root / "lalonde_matchit_weights.csv",
            dtype={"id": str},
            float_precision=_ROUND_TRIP,
        )
        self.expected = pd.read_csv(
            root / "lalonde_matchit_expected.csv", float_precision=_ROUND_TRIP
        )
        self.within = pd.read_csv(
            root / "lalonde_subclass_within_ks_expected.csv",
            float_precision=_ROUND_TRIP,
        )

    def _frames(
        self, method: str, continuous: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        reference = self.expected.loc[self.expected.method == method]
        if continuous:
            reference = reference.loc[~reference.is_binary]
        columns = reference.column.tolist()
        control = self.data.loc[self.data.treat == 0]
        treated = self.data.loc[self.data.treat == 1]
        weight = {
            "unadjusted": None,
            "nearest_replace": "w_nn_replace",
            "subclass4": "w_subclass4",
        }[method]
        wc = pd.Series(1.0, index=control.index) if weight is None else control[weight]
        wt = pd.Series(1.0, index=treated.index) if weight is None else treated[weight]
        return control[columns], treated[columns], wc, wt, reference

    def test_continuous_means_smd_and_variance_ratios_match_r_tables(self) -> None:
        for method in ("unadjusted", "nearest_replace", "subclass4"):
            with self.subTest(method=method):
                control, treated, wc, wt, reference = self._frames(
                    method, continuous=True
                )
                np.testing.assert_allclose(
                    weighted_mean(control, wc),
                    reference.means_control,
                    atol=1e-8,
                    rtol=1e-12,
                )
                np.testing.assert_allclose(
                    weighted_mean(treated, wt),
                    reference.means_treated,
                    atol=1e-8,
                    rtol=1e-12,
                )
                actual = asmd(
                    control,
                    treated,
                    wc,
                    wt,
                    std_type="target",
                    aggregate_by_main_covar=False,
                )
                np.testing.assert_allclose(
                    actual.loc[reference.column],
                    reference.matchit_std_mean_diff.abs(),
                    atol=1e-12,
                    rtol=1e-12,
                )
                ratio = outcome_variance_ratio(treated, control, wt, wc)
                np.testing.assert_allclose(
                    ratio, reference.matchit_var_ratio, atol=1e-12, rtol=1e-12
                )

    def test_weighted_ecdf_max_matches_matchit_for_all_variables(self) -> None:
        for method in ("unadjusted", "nearest_replace", "subclass4"):
            with self.subTest(method=method):
                control, treated, wc, wt, reference = self._frames(method)
                actual = ks(control, treated, wc, wt, aggregate_by_main_covar=False)
                np.testing.assert_allclose(
                    actual.loc[reference.column],
                    reference.matchit_ecdf_max,
                    atol=1e-12,
                    rtol=1e-12,
                )

    def test_within_subclass_ecdf_max_matches_all_four_r_tables(self) -> None:
        for subclass in (1, 2, 3, 4):
            with self.subTest(subclass=subclass):
                reference = self.within.loc[self.within.subclass == subclass]
                rows = self.data.loc[self.data.subclass4 == subclass]
                columns = reference.column.tolist()
                actual = ks(
                    rows.loc[rows.treat == 0, columns],
                    rows.loc[rows.treat == 1, columns],
                    aggregate_by_main_covar=False,
                )
                np.testing.assert_allclose(
                    actual.loc[columns],
                    reference.matchit_ecdf_max,
                    atol=1e-12,
                    rtol=1e-12,
                )

    def test_sample_covariate_views_match_the_same_r_subclass_summary(self) -> None:
        reference = self.expected.loc[
            (self.expected.method == "subclass4") & ~self.expected.is_binary
        ]
        columns = reference.column.tolist()
        sample = Sample.from_frame(
            self.data.loc[self.data.treat == 0, ["id"] + columns + ["w_subclass4"]],
            id_column="id",
            weight_column="w_subclass4",
        )
        target = Sample.from_frame(
            self.data.loc[self.data.treat == 1, ["id"] + columns + ["w_subclass4"]],
            id_column="id",
            weight_column="w_subclass4",
        )
        covars = sample.set_target(target).covars()
        np.testing.assert_allclose(
            covars.asmd().loc["self", columns],
            reference.matchit_std_mean_diff.abs(),
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            covars.ks().loc["self", columns],
            reference.matchit_ecdf_max,
            atol=1e-12,
            rtol=1e-12,
        )
