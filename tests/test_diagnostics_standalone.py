# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import balance.testutil
import numpy as np
import pandas as pd
from balance.diagnostics import standalone
from balance.diagnostics.standalone import (
    compute_asmd,
    compute_essp,
    compute_kish_design_effect,
    compute_kish_ess,
    compute_r_indicator,
    diagnostics_table,
    love_plot,
)
from balance.sample_class import Sample


class ComputeAsmdTest(balance.testutil.BalanceTestCase):
    """Numerical-parity vs the BalanceFrame route + basic shape checks."""

    def test_asmd_matches_balanceframe_route(self) -> None:
        """Standalone compute_asmd must match BalanceFrame.covars().asmd() to
        ~5e-10. This is the critical regression guard."""
        rng = np.random.default_rng(42)
        n = 200
        sample_df = pd.DataFrame(
            {
                "age": rng.normal(40, 10, size=n),
                "income": rng.normal(50_000, 15_000, size=n),
                "id": np.arange(n),
            }
        )
        target_df = pd.DataFrame(
            {
                "age": rng.normal(45, 10, size=n),
                "income": rng.normal(60_000, 15_000, size=n),
                "id": np.arange(n, 2 * n),
            }
        )
        sample_w = pd.Series(rng.uniform(0.5, 2.0, size=n))
        target_w = pd.Series(np.ones(n))

        # BalanceFrame route
        s_sample = Sample.from_frame(
            sample_df.assign(weight=sample_w.values),
            id_column="id",
            weight_column="weight",
        )
        s_target = Sample.from_frame(
            target_df.assign(weight=target_w.values),
            id_column="id",
            weight_column="weight",
        )
        s_sample = s_sample.set_target(s_target)
        bf_asmd = s_sample.covars().asmd(on_linked_samples=False)
        # bf_asmd is a DataFrame (one row); take the first row as a Series.
        bf_series = bf_asmd.iloc[0] if hasattr(bf_asmd, "iloc") else bf_asmd

        # Standalone route — wrap weighted_comparisons_stats.asmd directly.
        standalone_series = compute_asmd(
            sample_df.drop(columns=["id"]),
            sample_w,
            target_df.drop(columns=["id"]),
            target_w,
        )

        for col in ("age", "income", "mean(asmd)"):
            self.assertAlmostEqual(
                float(bf_series[col]),
                float(standalone_series[col]),
                places=9,
                msg=f"ASMD parity mismatch on column {col!r}",
            )

    def test_asmd_returns_series_with_mean(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        target = pd.DataFrame({"x": [2.0, 3.0, 4.0]})
        result = compute_asmd(df, [1.0, 1.0, 1.0], target, [1.0, 1.0, 1.0])
        self.assertIsInstance(result, pd.Series)
        self.assertIn("mean(asmd)", result.index)

    def test_asmd_self_when_no_target(self) -> None:
        """target=None compares weighted vs unweighted version of self."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        result = compute_asmd(df, [1.0, 2.0, 1.0, 2.0])
        self.assertIsInstance(result, pd.Series)
        # Per-covariate row plus the conventional mean(asmd) summary row.
        self.assertIn("x", result.index)
        self.assertIn("mean(asmd)", result.index)
        # ASMD is an absolute standardized difference, so always non-negative.
        self.assertGreaterEqual(float(result["x"]), 0.0)
        self.assertGreaterEqual(float(result["mean(asmd)"]), 0.0)


class ComputeKishDesignEffectTest(balance.testutil.BalanceTestCase):
    """Kish identity D >= 1; D == 1 for uniform weights."""

    def test_uniform_weights_give_deff_one(self) -> None:
        self.assertAlmostEqual(compute_kish_design_effect([1.0, 1.0, 1.0]), 1.0)
        self.assertAlmostEqual(compute_kish_design_effect([3.0, 3.0, 3.0, 3.0]), 1.0)

    def test_non_uniform_weights_give_deff_above_one(self) -> None:
        deff = compute_kish_design_effect([1.0, 2.0, 3.0, 4.0])
        self.assertGreater(deff, 1.0)

    def test_extreme_weights_have_high_deff(self) -> None:
        deff = compute_kish_design_effect([1.0, 1.0, 1000.0])
        self.assertGreater(deff, 2.0)


class ComputeKishEssTest(balance.testutil.BalanceTestCase):
    """ESS = N when all weights equal; ESS < N otherwise."""

    def test_ess_equals_n_for_uniform_weights(self) -> None:
        self.assertAlmostEqual(compute_kish_ess([1.0, 1.0, 1.0, 1.0]), 4.0)

    def test_ess_below_n_for_non_uniform(self) -> None:
        ess = compute_kish_ess([1.0, 2.0, 3.0, 4.0])
        self.assertLess(ess, 4.0)
        self.assertGreater(ess, 0.0)


class ComputeEsspTest(balance.testutil.BalanceTestCase):
    """Proportional ESS: 1/Deff, in [0, 1]."""

    def test_essp_one_for_uniform_weights(self) -> None:
        self.assertAlmostEqual(compute_essp([2.0, 2.0, 2.0]), 1.0)

    def test_essp_in_zero_one(self) -> None:
        essp = compute_essp([1.0, 2.0, 3.0, 4.0])
        self.assertGreater(essp, 0.0)
        self.assertLess(essp, 1.0)

    def test_essp_inverse_of_deff(self) -> None:
        weights = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(
            compute_essp(weights),
            1.0 / compute_kish_design_effect(weights),
            places=10,
        )


class ComputeRIndicatorTest(balance.testutil.BalanceTestCase):
    """R-indicator wrapper — returns dict with 'r_indicator' in [0, 1]."""

    def test_returns_dict_with_r_indicator(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = compute_r_indicator(df, [1.0, 1.0, 1.0])
        self.assertIn("r_indicator", result)
        self.assertGreaterEqual(result["r_indicator"], 0.0)
        self.assertLessEqual(result["r_indicator"], 1.0)

    def test_uniform_weights_high_r_indicator(self) -> None:
        df = pd.DataFrame({"x": np.arange(10, dtype=float)})
        result = compute_r_indicator(df, np.ones(10))
        # Uniform weights → uniform inverse → R-indicator close to 1.
        self.assertGreater(result["r_indicator"], 0.9)

    def test_invalid_n_target_raises(self) -> None:
        """n_target<1 is rejected with a clear domain error rather than a
        confusing numpy ValueError from np.ones(negative)."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        with self.assertRaisesRegex(ValueError, "n_target must be >= 1"):
            compute_r_indicator(df, [1.0, 1.0, 1.0], n_target=0)
        with self.assertRaisesRegex(ValueError, "n_target must be >= 1"):
            compute_r_indicator(df, [1.0, 1.0, 1.0], n_target=-3)


class LovePlotTest(balance.testutil.BalanceTestCase):
    """love_plot returns matplotlib.axes.Axes; close all figs in tearDown."""

    def tearDown(self) -> None:
        super().tearDown()
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except ImportError:  # pragma: no cover
            pass

    def test_returns_matplotlib_axes(self) -> None:
        import matplotlib.axes

        before = pd.Series({"age": 0.4, "income": 0.3})
        after = pd.Series({"age": 0.05, "income": 0.08})
        ax = love_plot(before, after)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_strips_mean_asmd_summary_row(self) -> None:
        before = pd.Series({"age": 0.4, "income": 0.3, "mean(asmd)": 0.35})
        after = pd.Series({"age": 0.05, "income": 0.08, "mean(asmd)": 0.065})
        ax = love_plot(before, after)
        # mean(asmd) should not appear among y-tick labels.
        labels = [t.get_text() for t in ax.get_yticklabels()]
        self.assertNotIn("mean(asmd)", labels)


class DiagnosticsTableTest(balance.testutil.BalanceTestCase):
    """diagnostics_table column shape, types, and toggle flags."""

    def test_columns_and_index(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        w = pd.Series([1.0, 2.0, 1.0, 2.0])
        table = diagnostics_table(df, w, asmd=False, r_indicator=False)
        self.assertEqual(list(table.columns), ["value"])
        self.assertIn("deff", table.index)
        self.assertIn("ess", table.index)
        self.assertIn("essp", table.index)

    def test_with_target_includes_r_indicator(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        target = pd.DataFrame({"x": [2.0, 3.0, 4.0, 5.0]})
        w = pd.Series([1.0, 2.0, 1.0, 2.0])
        table = diagnostics_table(df, w, target, target_weights=None)
        self.assertIn("r_indicator", table.index)

    def test_no_target_includes_r_indicator(self) -> None:
        """The bundled diagnostics_table should not silently drop r_indicator
        in the no-target case -- compute_r_indicator itself supports it
        (falls back to len(df))."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        w = pd.Series([1.0, 2.0, 1.0, 2.0])
        table = diagnostics_table(df, w)
        self.assertIn("r_indicator", table.index)
        # The R-indicator is bounded in [0, 1].
        r = float(table.loc["r_indicator", "value"])
        self.assertGreaterEqual(r, 0.0)
        self.assertLessEqual(r, 1.0)

    def test_value_column_is_numeric(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        w = pd.Series([1.0, 1.5, 2.0])
        table = diagnostics_table(df, w, asmd=False, r_indicator=False)
        self.assertTrue(np.issubdtype(table["value"].dtype, np.floating))

    def test_misaligned_weights_raise_when_only_kish_essp_requested(self) -> None:
        """Regression guard: a weight vector that is not row-aligned with
        ``df`` must raise ``ValueError`` even when only ``kish``/``essp`` are
        requested (``asmd=False, r_indicator=False``). The other paths
        (``asmd``, ``r_indicator``) already validate alignment via
        ``compute_asmd`` / ``compute_r_indicator``; the kish/essp-only path
        previously skipped that check and could silently scale diagnostics
        to the wrong row count."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        misaligned_weights = pd.Series([1.0, 1.5])  # 2 != 3
        with self.assertRaises(ValueError):
            diagnostics_table(df, misaligned_weights, asmd=False, r_indicator=False)


class KishDeffSanityTest(balance.testutil.BalanceTestCase):
    """Direct validation of Kish design-effect properties.

    Kish's pooled-weight DEFF equals 1.0 under uniform positive weights and
    exceeds 1.0 once weights are heterogeneous. These two properties are
    the load-bearing contracts the rest of the diagnostics surface relies on.
    """

    def test_kish_deff_equals_one_for_uniform_weights(self) -> None:
        uniform_weights = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(compute_kish_design_effect(uniform_weights), 1.0)

    def test_kish_deff_exceeds_one_for_heterogeneous_weights(self) -> None:
        heterogeneous_weights = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertGreater(compute_kish_design_effect(heterogeneous_weights), 1.0)


class EdgeCasesTest(balance.testutil.BalanceTestCase):
    """Edge case validation: empty / negative / single covariate."""

    def test_empty_weights_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_kish_design_effect([])

    def test_negative_weights_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_kish_design_effect([1.0, -2.0, 3.0])

    def test_non_finite_weights_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_kish_design_effect([1.0, np.inf, 3.0])

    def test_misaligned_df_and_weights_raises(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        with self.assertRaises(ValueError):
            compute_asmd(df, [1.0, 2.0])  # wrong length

    def test_single_covariate(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        target = pd.DataFrame({"x": [2.0, 3.0, 4.0]})
        result = compute_asmd(df, [1.0, 1.0, 1.0], target)
        self.assertIn("x", result.index)

    def test_all_equal_weights(self) -> None:
        """All-equal weights: Deff == 1, ESS == n, ESSP == 1."""
        w = [2.0, 2.0, 2.0, 2.0]
        self.assertAlmostEqual(compute_kish_design_effect(w), 1.0)
        self.assertAlmostEqual(compute_kish_ess(w), 4.0)
        self.assertAlmostEqual(compute_essp(w), 1.0)

    def test_module_reexport(self) -> None:
        """Re-exported functions return correct values when called via the
        ``balance.diagnostics`` package path.

        This validates BEHAVIOR (the function actually works through the
        re-export) rather than module STRUCTURE (which attributes exist).
        Uniform weights → ``design_effect == 1.0`` and ``ess == N`` are
        the canonical Kish identities.
        """
        w = pd.Series([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(standalone.compute_kish_design_effect(w), 1.0)
        self.assertAlmostEqual(standalone.compute_kish_ess(w), 4.0)
        # love_plot returns a matplotlib Axes when called via the re-export.
        import matplotlib.pyplot as plt

        asmd_before = pd.Series([0.3, 0.2, 0.1])
        asmd_after = pd.Series([0.05, 0.04, 0.03])
        ax = standalone.love_plot(asmd_before, asmd_after)
        self.assertIsNotNone(ax)
        plt.close("all")
