# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import pandas as pd
from balance.balance_frame import BalanceFrame
from balance.datasets import load_data
from balance.sample_frame import SampleFrame
from balance.testutil import BalanceTestCase


class TestBalanceFrameConstruction(BalanceTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.resp_df = pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "x1": [10.0, 20.0, 30.0],
                "x2": [1.0, 2.0, 3.0],
                "weight": [1.0, 1.0, 1.0],
            }
        )
        self.tgt_df = pd.DataFrame(
            {
                "id": ["4", "5", "6"],
                "x1": [15.0, 25.0, 35.0],
                "x2": [1.5, 2.5, 3.5],
                "weight": [1.0, 1.0, 1.0],
            }
        )
        self.resp_sf = SampleFrame.from_frame(self.resp_df)
        self.tgt_sf = SampleFrame.from_frame(self.tgt_df)

    def test_basic_construction(self) -> None:
        bf = BalanceFrame(responders=self.resp_sf, target=self.tgt_sf)
        self.assertIsInstance(bf, BalanceFrame)
        self.assertIs(bf.responders, self.resp_sf)
        self.assertIs(bf.target, self.tgt_sf)

    def test_is_adjusted_false_on_creation(self) -> None:
        bf = BalanceFrame(responders=self.resp_sf, target=self.tgt_sf)
        self.assertFalse(bf.is_adjusted)

    def test_unadjusted_none_on_creation(self) -> None:
        bf = BalanceFrame(responders=self.resp_sf, target=self.tgt_sf)
        self.assertIsNone(bf.unadjusted)

    def test_model_none_on_creation(self) -> None:
        bf = BalanceFrame(responders=self.resp_sf, target=self.tgt_sf)
        self.assertIsNone(bf.model())

    def test_missing_responders_raises(self) -> None:
        with self.assertRaises(TypeError):
            BalanceFrame(target=self.tgt_sf)

    def test_no_target_construction(self) -> None:
        bf = BalanceFrame(responders=self.resp_sf)
        self.assertIsInstance(bf, BalanceFrame)
        self.assertIs(bf.responders, self.resp_sf)
        self.assertIsNone(bf.target)
        self.assertFalse(bf.has_target())

    def test_no_args_returns_bare_instance(self) -> None:
        # BalanceFrame() with no args returns a bare instance to support
        # copy.deepcopy(), which calls __new__ without arguments.
        bf = BalanceFrame()
        self.assertIsInstance(bf, BalanceFrame)

    def test_non_sampleframe_responders_raises(self) -> None:
        with self.assertRaises(TypeError):
            BalanceFrame._create(
                responders="not a SampleFrame",  # pyre-ignore[6]
                target=self.tgt_sf,
            )

    def test_non_sampleframe_target_raises(self) -> None:
        with self.assertRaises(TypeError):
            BalanceFrame._create(
                responders=self.resp_sf,
                target=pd.DataFrame(),  # pyre-ignore[6]
            )


class TestBalanceFrameCovarOverlap(BalanceTestCase):
    def test_zero_overlap_raises(self) -> None:
        resp_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [1], "a": [10.0], "w": [1.0]}),
            id_column="id",
            covars_columns=["a"],
            weight_columns=["w"],
        )
        tgt_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [2], "b": [20.0], "w": [1.0]}),
            id_column="id",
            covars_columns=["b"],
            weight_columns=["w"],
        )
        with self.assertRaises(ValueError) as ctx:
            BalanceFrame(responders=resp_sf, target=tgt_sf)
        self.assertIn("no covariate columns", str(ctx.exception))

    def test_partial_overlap_warns(self) -> None:
        resp_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [1], "x": [10.0], "a": [1.0], "w": [1.0]}),
            id_column="id",
            covars_columns=["x", "a"],
            weight_columns=["w"],
        )
        tgt_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [2], "x": [20.0], "b": [2.0], "w": [1.0]}),
            id_column="id",
            covars_columns=["x", "b"],
            weight_columns=["w"],
        )
        with self.assertLogs("balance", level="WARNING") as cm:
            bf = BalanceFrame(responders=resp_sf, target=tgt_sf)
        self.assertTrue(any("different covariate columns" in msg for msg in cm.output))
        self.assertIsInstance(bf, BalanceFrame)

    def test_full_overlap_no_warning(self) -> None:
        resp_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [1], "x": [10.0], "w": [1.0]}),
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
        )
        tgt_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [2], "x": [20.0], "w": [1.0]}),
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
        )
        # Capture any balance logger warnings — there should be none
        with self.assertLogs("balance", level="INFO") as cm:
            logging.getLogger("balance").info("sentinel")
            bf = BalanceFrame(responders=resp_sf, target=tgt_sf)
        warning_msgs = [m for m in cm.output if "WARNING" in m]
        self.assertEqual(len(warning_msgs), 0)
        self.assertIsInstance(bf, BalanceFrame)


class TestBalanceFrameDeepCopy(BalanceTestCase):
    def test_deepcopy_unadjusted(self) -> None:
        resp_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "w": [1.0, 1.0]}),
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
        )
        tgt_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "w": [1.0, 1.0]}),
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
        )
        bf = BalanceFrame(responders=resp_sf, target=tgt_sf)
        bf_copy = copy.deepcopy(bf)
        self.assertIsInstance(bf_copy, BalanceFrame)
        self.assertFalse(bf_copy.is_adjusted)
        self.assertIsNot(bf_copy.responders, bf.responders)
        self.assertEqual(len(bf_copy.responders), len(bf.responders))


class TestBalanceFrameRepr(BalanceTestCase):
    def test_repr_unadjusted(self) -> None:
        resp_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "w": [1.0, 1.0]}),
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
        )
        tgt_sf = SampleFrame._create(
            df=pd.DataFrame(
                {"id": [3, 4, 5], "x": [15.0, 25.0, 35.0], "w": [1.0, 1.0, 1.0]}
            ),
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
        )
        bf = BalanceFrame(responders=resp_sf, target=tgt_sf)
        r = repr(bf)
        self.assertIn("unadjusted", r)
        self.assertIn("2 responders", r)
        self.assertIn("3 target observations", r)
        self.assertIn("1 covariates", r)
        self.assertIn("x", r)

    def test_str_equals_repr(self) -> None:
        resp_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [1], "x": [10.0], "w": [1.0]}),
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
        )
        tgt_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [2], "x": [20.0], "w": [1.0]}),
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
        )
        bf = BalanceFrame(responders=resp_sf, target=tgt_sf)
        self.assertEqual(str(bf), repr(bf))


class TestBalanceFrameCreateDirect(BalanceTestCase):
    def test_create_direct(self) -> None:
        """Test that _create() works as an alternative to __new__."""
        resp_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [1], "x": [10.0], "w": [1.0]}),
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
        )
        tgt_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [2], "x": [20.0], "w": [1.0]}),
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
        )
        bf = BalanceFrame._create(responders=resp_sf, target=tgt_sf)
        self.assertIsInstance(bf, BalanceFrame)
        self.assertFalse(bf.is_adjusted)

    def test_properties_accessible(self) -> None:
        """Verify all public properties are accessible after construction."""
        resp_sf = SampleFrame._create(
            df=pd.DataFrame(
                {"id": [1, 2], "x": [10.0, 20.0], "y": [1.0, 0.0], "w": [1.0, 2.0]}
            ),
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
            outcome_columns=["y"],
        )
        tgt_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "w": [1.0, 1.0]}),
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
        )
        bf = BalanceFrame(responders=resp_sf, target=tgt_sf)
        # Access all properties
        self.assertIsNotNone(bf.responders)
        self.assertIsNotNone(bf.target)
        self.assertIsNone(bf.unadjusted)
        self.assertFalse(bf.is_adjusted)
        self.assertIsNone(bf.model())
        # Responder data is correct
        self.assertEqual(len(bf.responders.df_covars), 2)
        self.assertListEqual(list(bf.responders.df_weights.columns), ["w"])


class TestBalanceFrameAdjust(BalanceTestCase):
    def setUp(self) -> None:
        super().setUp()
        target_df, sample_df = load_data()
        assert target_df is not None and sample_df is not None
        self.resp_sf = SampleFrame.from_frame(sample_df, outcome_columns=["happiness"])
        self.tgt_sf = SampleFrame.from_frame(target_df, outcome_columns=["happiness"])
        self.bf = BalanceFrame(responders=self.resp_sf, target=self.tgt_sf)

    def test_adjust_ipw(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        self.assertTrue(adjusted.is_adjusted)
        model = adjusted.model()
        self.assertIsNotNone(model)
        assert model is not None  # for Pyre narrowing
        self.assertIn("method", model)
        self.assertEqual(model["method"], "ipw")

    def test_adjust_preserves_original_weights(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        # Original weight column still present alongside adjusted
        all_w_cols = adjusted.responders.weight_columns
        self.assertIn("weight", all_w_cols)
        self.assertIn("weight_adjusted", all_w_cols)
        # Active weight is the adjusted one
        self.assertListEqual(
            list(adjusted.responders.df_weights.columns), ["weight_adjusted"]
        )

    def test_adjust_immutability(self) -> None:
        original_weight_vals = self.bf.responders.df_weights.iloc[:, 0].tolist()
        adjusted = self.bf.adjust(method="ipw")
        # Original BalanceFrame is unchanged
        self.assertFalse(self.bf.is_adjusted)
        self.assertIsNone(self.bf.model())
        self.assertIsNone(self.bf.unadjusted)
        # Original weights unchanged
        after_weight_vals = self.bf.responders.df_weights.iloc[:, 0].tolist()
        self.assertEqual(original_weight_vals, after_weight_vals)
        # Adjusted BalanceFrame has different weights
        adj_weight_vals = adjusted.responders.df_weights.iloc[:, 0].tolist()
        self.assertNotEqual(original_weight_vals, adj_weight_vals)

    def test_adjust_custom_callable(self) -> None:
        def dummy_method(
            sample_df: pd.DataFrame,
            sample_weights: pd.Series,
            target_df: pd.DataFrame,
            target_weights: pd.Series,
        ) -> dict[str, Any]:
            return {
                "weight": pd.Series(
                    np.ones(len(sample_df)) * 42.0, index=sample_df.index
                ),
                "model": {"method": "dummy"},
            }

        adjusted = self.bf.adjust(method=dummy_method)
        self.assertTrue(adjusted.is_adjusted)
        # All adjusted weights should be 42.0
        for w in adjusted.responders.df_weights.iloc[:, 0].tolist():
            self.assertAlmostEqual(w, 42.0, places=5)
        model = adjusted.model()
        assert model is not None
        self.assertEqual(model["method"], "dummy")

    def test_adjust_unadjusted_stored(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        unadj = adjusted.unadjusted
        self.assertIsNotNone(unadj)
        assert unadj is not None  # for Pyre narrowing
        # unadjusted should have original weight only
        self.assertListEqual(list(unadj.df_weights.columns), ["weight"])

    def test_adjust_repr_shows_adjusted(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        self.assertIn("adjusted", repr(adjusted))
        self.assertNotIn("unadjusted", repr(adjusted))

    def test_adjust_weight_metadata(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        meta = adjusted.responders.weight_metadata("weight_adjusted")
        self.assertEqual(meta["method"], "ipw")
        self.assertIn("model", meta)

    def test_adjust_already_adjusted_raises(self) -> None:
        """Calling adjust() on an already-adjusted BalanceFrame raises ValueError."""
        adjusted = self.bf.adjust(method="ipw")
        with self.assertRaises(ValueError) as ctx:
            adjusted.adjust(method="ipw")
        self.assertIn("already-adjusted", str(ctx.exception))

    def test_adjust_stores_method_name_in_model(self) -> None:
        """adjust() stores the method name in the adjustment model dict."""
        adjusted = self.bf.adjust(method="ipw")
        model = adjusted.model()
        self.assertIsInstance(model, dict)
        assert model is not None
        self.assertEqual(model["method"], "ipw")

    def test_adjust_stores_custom_method_name_in_model(self) -> None:
        """adjust() stores 'custom' when a callable method is used."""

        def custom_method(
            sample_df: pd.DataFrame,
            sample_weights: pd.Series,
            target_df: pd.DataFrame,
            target_weights: pd.Series,
        ) -> dict[str, Any]:
            return {"weight": sample_weights, "model": {"info": "test"}}

        adjusted = self.bf.adjust(method=custom_method)
        model = adjusted.model()
        self.assertIsInstance(model, dict)
        assert model is not None
        self.assertEqual(model["method"], "custom")
        self.assertEqual(model["info"], "test")

    def test_adjust_invalid_method_raises(self) -> None:
        """adjust() with an invalid method string raises ValueError."""
        with self.assertRaises(ValueError):
            self.bf.adjust(method="nonexistent_method")

    def test_adjust_invalid_method_type_raises(self) -> None:
        """adjust() with a non-string, non-callable method raises ValueError."""
        with self.assertRaises(ValueError):
            self.bf.adjust(method=42)  # pyre-ignore[6]

    def test_adjust_deepcopy_adjusted(self) -> None:
        """Deepcopy of an adjusted BalanceFrame preserves adjustment state."""
        adjusted = self.bf.adjust(method="ipw")
        adjusted_copy = copy.deepcopy(adjusted)
        self.assertTrue(adjusted_copy.is_adjusted)
        self.assertIsNotNone(adjusted_copy.unadjusted)
        self.assertIsNotNone(adjusted_copy.model())
        # The copy's weights should match
        pd.testing.assert_series_equal(
            adjusted_copy.responders.df_weights.iloc[:, 0],
            adjusted.responders.df_weights.iloc[:, 0],
        )


class TestBalanceFrameSetTarget(BalanceTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.resp_df = pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "x1": [10.0, 20.0, 30.0],
                "x2": [1.0, 2.0, 3.0],
                "weight": [1.0, 1.0, 1.0],
            }
        )
        self.tgt_df = pd.DataFrame(
            {
                "id": ["4", "5", "6"],
                "x1": [15.0, 25.0, 35.0],
                "x2": [1.5, 2.5, 3.5],
                "weight": [1.0, 1.0, 1.0],
            }
        )
        self.resp_sf = SampleFrame.from_frame(self.resp_df)
        self.tgt_sf = SampleFrame.from_frame(self.tgt_df)

    def test_has_target_false_without_target(self) -> None:
        bf = BalanceFrame(responders=self.resp_sf)
        self.assertFalse(bf.has_target())

    def test_has_target_true_with_target(self) -> None:
        bf = BalanceFrame(responders=self.resp_sf, target=self.tgt_sf)
        self.assertTrue(bf.has_target())

    def test_set_target_in_place(self) -> None:
        bf = BalanceFrame(responders=self.resp_sf)
        result = bf.set_target(self.tgt_sf)
        self.assertIs(result, bf)
        self.assertTrue(bf.has_target())
        self.assertIs(bf.target, self.tgt_sf)

    def test_set_target_copy(self) -> None:
        bf = BalanceFrame(responders=self.resp_sf)
        new_bf = bf.set_target(self.tgt_sf, in_place=False)
        self.assertIsNot(new_bf, bf)
        self.assertTrue(new_bf.has_target())
        self.assertFalse(bf.has_target())

    def test_set_target_replaces_existing(self) -> None:
        bf = BalanceFrame(responders=self.resp_sf, target=self.tgt_sf)
        tgt2_df = pd.DataFrame(
            {
                "id": ["7", "8"],
                "x1": [50.0, 60.0],
                "x2": [5.0, 6.0],
                "weight": [1.0, 1.0],
            }
        )
        tgt2_sf = SampleFrame.from_frame(tgt2_df)
        bf.set_target(tgt2_sf)
        self.assertIs(bf.target, tgt2_sf)

    def test_set_target_resets_adjustment(self) -> None:
        bf = BalanceFrame(responders=self.resp_sf, target=self.tgt_sf)
        adjusted = bf.adjust(method="null")
        self.assertTrue(adjusted.is_adjusted)
        # Replace target on the adjusted frame — should reset adjustment state
        tgt2_df = pd.DataFrame(
            {
                "id": ["7", "8"],
                "x1": [50.0, 60.0],
                "x2": [5.0, 6.0],
                "weight": [1.0, 1.0],
            }
        )
        tgt2_sf = SampleFrame.from_frame(tgt2_df)
        adjusted.set_target(tgt2_sf)
        self.assertFalse(adjusted.is_adjusted)
        self.assertIsNone(adjusted.model())

    def test_set_target_non_sampleframe_raises(self) -> None:
        bf = BalanceFrame(responders=self.resp_sf)
        with self.assertRaises(TypeError):
            bf.set_target("not a SampleFrame")  # pyre-ignore[6]

    def test_set_target_no_overlap_raises(self) -> None:
        bf = BalanceFrame(responders=self.resp_sf)
        bad_tgt = SampleFrame._create(
            df=pd.DataFrame({"id": [1], "z": [10.0], "w": [1.0]}),
            id_column="id",
            covars_columns=["z"],
            weight_columns=["w"],
        )
        with self.assertRaises(ValueError) as ctx:
            bf.set_target(bad_tgt)
        self.assertIn("no covariate columns", str(ctx.exception))

    def test_adjust_without_target_raises(self) -> None:
        bf = BalanceFrame(responders=self.resp_sf)
        with self.assertRaises(ValueError) as ctx:
            bf.adjust(method="ipw")
        self.assertIn("without a target", str(ctx.exception))

    def test_no_target_repr(self) -> None:
        bf = BalanceFrame(responders=self.resp_sf)
        r = repr(bf)
        self.assertIn("no target", r)
        self.assertIn("3 responders", r)


class TestBalanceFrameCovarsWeightsOutcomes(BalanceTestCase):
    """Tests for BalanceFrame.covars(), .weights(), .outcomes() integration."""

    def setUp(self) -> None:
        super().setUp()
        target_df, sample_df = load_data()
        assert target_df is not None and sample_df is not None
        self.resp_sf = SampleFrame.from_frame(sample_df, outcome_columns=["happiness"])
        self.tgt_sf = SampleFrame.from_frame(target_df, outcome_columns=["happiness"])
        self.bf = BalanceFrame(responders=self.resp_sf, target=self.tgt_sf)

    def test_covars_returns_balancedf_covars(self) -> None:
        from balance.balancedf_class import BalanceDFCovars

        result = self.bf.covars()
        self.assertIsInstance(result, BalanceDFCovars)

    def test_covars_df(self) -> None:
        covars_df = self.bf.covars().df
        self.assertIsInstance(covars_df, pd.DataFrame)
        self.assertTrue(len(covars_df) > 0)

    def test_covars_names(self) -> None:
        names = self.bf.covars().names()
        self.assertIsInstance(names, list)
        self.assertTrue(len(names) > 0)

    def test_covars_mean(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        mean_df = adjusted.covars().mean()
        self.assertIsInstance(mean_df, pd.DataFrame)
        self.assertEqual(mean_df.index.name, "source")
        self.assertIn("self", mean_df.index)
        self.assertIn("target", mean_df.index)
        self.assertIn("unadjusted", mean_df.index)

    def test_covars_std(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        std_df = adjusted.covars().std()
        self.assertIsInstance(std_df, pd.DataFrame)
        self.assertIn("self", std_df.index)

    def test_covars_var_of_mean(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        vom = adjusted.covars().var_of_mean()
        self.assertIsInstance(vom, pd.DataFrame)

    def test_covars_ci_of_mean(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        ci = adjusted.covars().ci_of_mean()
        self.assertIsInstance(ci, pd.DataFrame)

    def test_covars_mean_with_ci(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        mwci = adjusted.covars().mean_with_ci()
        self.assertIsInstance(mwci, pd.DataFrame)

    def test_covars_summary(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        summary = adjusted.covars().summary()
        self.assertIsInstance(summary, pd.DataFrame)

    def test_covars_model_matrix(self) -> None:
        mm = self.bf.covars().model_matrix()
        self.assertIsInstance(mm, pd.DataFrame)
        self.assertTrue(len(mm) > 0)

    def test_covars_asmd(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        asmd_df = adjusted.covars().asmd()
        self.assertIsInstance(asmd_df, pd.DataFrame)

    def test_covars_asmd_improvement(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        improvement = adjusted.covars().asmd_improvement()
        self.assertIsInstance(improvement, (float, np.floating))

    def test_covars_links_unadjusted(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        linked = adjusted.covars()._BalanceDF_child_from_linked_samples()
        self.assertIn("self", linked)
        self.assertIn("target", linked)
        self.assertIn("unadjusted", linked)
        self.assertEqual(len(linked), 3)

    def test_covars_links_no_unadjusted(self) -> None:
        linked = self.bf.covars()._BalanceDF_child_from_linked_samples()
        self.assertIn("self", linked)
        self.assertIn("target", linked)
        self.assertNotIn("unadjusted", linked)
        self.assertEqual(len(linked), 2)

    def test_covars_to_csv(self) -> None:
        csv_str = self.bf.covars().to_csv()
        self.assertIsInstance(csv_str, str)
        self.assertTrue(len(csv_str) > 0)

    def test_weights_returns_balancedf_weights(self) -> None:
        from balance.balancedf_class import BalanceDFWeights

        result = self.bf.weights()
        self.assertIsInstance(result, BalanceDFWeights)

    def test_weights_df(self) -> None:
        w_df = self.bf.weights().df
        self.assertIsInstance(w_df, pd.DataFrame)
        self.assertTrue(len(w_df) > 0)

    def test_weights_summary(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        summary = adjusted.weights().summary()
        self.assertIsInstance(summary, pd.DataFrame)

    def test_weights_design_effect(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        deff = adjusted.weights().design_effect()
        self.assertIsInstance(deff, float)
        self.assertTrue(deff >= 1.0)

    def test_weights_trim(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        weights_obj = adjusted.weights()
        # trim() mutates in place and returns None
        result = weights_obj.trim()
        self.assertIsNone(result)

    def test_outcomes_returns_balancedf_outcomes(self) -> None:
        from balance.balancedf_class import BalanceDFOutcomes

        result = self.bf.outcomes()
        self.assertIsInstance(result, BalanceDFOutcomes)

    def test_outcomes_none_when_no_outcomes(self) -> None:
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {"id": [1, 2, 3], "x": [10.0, 20.0, 30.0], "weight": [1.0, 1.0, 1.0]}
            )
        )
        tgt_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {"id": [4, 5, 6], "x": [15.0, 25.0, 35.0], "weight": [1.0, 1.0, 1.0]}
            )
        )
        bf = BalanceFrame(responders=resp_sf, target=tgt_sf)
        self.assertIsNone(bf.outcomes())

    def test_outcomes_df(self) -> None:
        result = self.bf.outcomes()
        self.assertIsNotNone(result)
        assert result is not None
        o_df = result.df
        self.assertIsInstance(o_df, pd.DataFrame)
        self.assertIn("happiness", o_df.columns)

    def test_outcomes_summary(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        result = adjusted.outcomes()
        self.assertIsNotNone(result)
        assert result is not None
        summary = result.summary()
        self.assertIsInstance(summary, str)

    def test_outcomes_relative_response_rates(self) -> None:
        result = self.bf.outcomes()
        self.assertIsNotNone(result)
        assert result is not None
        rrr = result.relative_response_rates()
        self.assertIsInstance(rrr, pd.DataFrame)

    def test_outcomes_target_response_rates(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        result = adjusted.outcomes()
        self.assertIsNotNone(result)
        assert result is not None
        trr = result.target_response_rates()
        self.assertIsInstance(trr, pd.DataFrame)

    def test_numerical_equivalence_with_sample(self) -> None:
        """CRITICAL: Verify BalanceFrame produces same results as Sample API."""
        from balance.sample_class import Sample

        target_df, sample_df = load_data()
        assert target_df is not None and sample_df is not None

        # --- Old Sample API ---
        old_sample = Sample.from_frame(sample_df, outcome_columns=["happiness"])
        old_target = Sample.from_frame(target_df, outcome_columns=["happiness"])
        old_sample_with_target = old_sample.set_target(old_target)
        old_adjusted = old_sample_with_target.adjust(method="ipw")
        old_covars_mean = old_adjusted.covars().mean()
        old_covars_asmd = old_adjusted.covars().asmd()

        # --- New BalanceFrame API ---
        new_resp = SampleFrame.from_frame(sample_df, outcome_columns=["happiness"])
        new_tgt = SampleFrame.from_frame(target_df, outcome_columns=["happiness"])
        new_bf = BalanceFrame(responders=new_resp, target=new_tgt)
        new_adjusted = new_bf.adjust(method="ipw")
        new_covars_mean = new_adjusted.covars().mean()
        new_covars_asmd = new_adjusted.covars().asmd()

        # Compare covars().mean()
        self.assertIsInstance(old_covars_mean, pd.DataFrame)
        self.assertIsInstance(new_covars_mean, pd.DataFrame)
        old_mean_numeric = old_covars_mean.select_dtypes(include=[np.number])
        new_mean_numeric = new_covars_mean.select_dtypes(include=[np.number])
        self.assertEqual(
            sorted(old_mean_numeric.columns.tolist()),
            sorted(new_mean_numeric.columns.tolist()),
        )

        # Compare ASMD values
        self.assertIsInstance(old_covars_asmd, pd.DataFrame)
        self.assertIsInstance(new_covars_asmd, pd.DataFrame)
        old_asmd_numeric = old_covars_asmd.select_dtypes(include=[np.number])
        new_asmd_numeric = new_covars_asmd.select_dtypes(include=[np.number])
        for col in old_asmd_numeric.columns:
            if col in new_asmd_numeric.columns:
                for old_val, new_val in zip(
                    old_asmd_numeric[col].dropna(), new_asmd_numeric[col].dropna()
                ):
                    self.assertAlmostEqual(old_val, new_val, places=5)

    def test_weights_equivalence_with_sample(self) -> None:
        """Verify BalanceFrame weights analysis matches Sample API."""
        from balance.sample_class import Sample

        target_df, sample_df = load_data()
        assert target_df is not None and sample_df is not None

        # Old API
        old_sample = Sample.from_frame(sample_df, outcome_columns=["happiness"])
        old_target = Sample.from_frame(target_df, outcome_columns=["happiness"])
        old_adjusted = old_sample.set_target(old_target).adjust(method="ipw")
        old_deff = old_adjusted.weights().design_effect()

        # New API
        new_resp = SampleFrame.from_frame(sample_df, outcome_columns=["happiness"])
        new_tgt = SampleFrame.from_frame(target_df, outcome_columns=["happiness"])
        new_adjusted = BalanceFrame(responders=new_resp, target=new_tgt).adjust(
            method="ipw"
        )
        new_deff = new_adjusted.weights().design_effect()

        self.assertAlmostEqual(old_deff, new_deff, places=5)
