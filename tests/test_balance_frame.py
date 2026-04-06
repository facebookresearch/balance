# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import copy
import io
import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from balance.balance_frame import BalanceFrame
from balance.datasets import load_data
from balance.sample_class import Sample
from balance.sample_frame import SampleFrame
from balance.testutil import BalanceTestCase
from balance.util import _assert_type


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
        bf = BalanceFrame(sample=self.resp_sf, sf_target=self.tgt_sf)
        self.assertIsInstance(bf, BalanceFrame)
        self.assertIs(bf.responders, self.resp_sf)
        self.assertIs(bf.target, self.tgt_sf)

    def test_is_adjusted_false_on_creation(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf, sf_target=self.tgt_sf)
        self.assertFalse(bf.is_adjusted)

    def test_unadjusted_none_on_creation(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf, sf_target=self.tgt_sf)
        self.assertIsNone(bf.unadjusted)

    def test_model_none_on_creation(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf, sf_target=self.tgt_sf)
        self.assertIsNone(bf.model)

    def test_missing_responders_raises(self) -> None:
        with self.assertRaises(TypeError):
            BalanceFrame(sf_target=self.tgt_sf)

    def test_no_target_construction(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf)
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
                sample="not a SampleFrame",  # pyre-ignore[6]
                sf_target=self.tgt_sf,
            )

    def test_non_sampleframe_target_raises(self) -> None:
        with self.assertRaises(TypeError):
            BalanceFrame._create(
                sample=self.resp_sf,
                sf_target=pd.DataFrame(),  # pyre-ignore[6]
            )


class TestBalanceFrameCovarOverlap(BalanceTestCase):
    def test_zero_overlap_raises(self) -> None:
        resp_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [1], "a": [10.0], "w": [1.0]}),
            id_column="id",
            covar_columns=["a"],
            weight_columns=["w"],
        )
        tgt_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [2], "b": [20.0], "w": [1.0]}),
            id_column="id",
            covar_columns=["b"],
            weight_columns=["w"],
        )
        with self.assertRaises(ValueError) as ctx:
            BalanceFrame(sample=resp_sf, sf_target=tgt_sf)
        self.assertIn("no covariate columns", str(ctx.exception))

    def test_partial_overlap_warns(self) -> None:
        resp_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [1], "x": [10.0], "a": [1.0], "w": [1.0]}),
            id_column="id",
            covar_columns=["x", "a"],
            weight_columns=["w"],
        )
        tgt_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [2], "x": [20.0], "b": [2.0], "w": [1.0]}),
            id_column="id",
            covar_columns=["x", "b"],
            weight_columns=["w"],
        )
        with self.assertLogs("balance", level="WARNING") as cm:
            bf = BalanceFrame(sample=resp_sf, sf_target=tgt_sf)
        self.assertTrue(any("different covariate columns" in msg for msg in cm.output))
        self.assertIsInstance(bf, BalanceFrame)

    def test_full_overlap_no_warning(self) -> None:
        resp_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [1], "x": [10.0], "w": [1.0]}),
            id_column="id",
            covar_columns=["x"],
            weight_columns=["w"],
        )
        tgt_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [2], "x": [20.0], "w": [1.0]}),
            id_column="id",
            covar_columns=["x"],
            weight_columns=["w"],
        )
        # Capture any balance logger warnings — there should be none
        with self.assertLogs("balance", level="INFO") as cm:
            logging.getLogger("balance").info("sentinel")
            bf = BalanceFrame(sample=resp_sf, sf_target=tgt_sf)
        warning_msgs = [m for m in cm.output if "WARNING" in m]
        self.assertEqual(len(warning_msgs), 0)
        self.assertIsInstance(bf, BalanceFrame)


class TestBalanceFrameDeepCopy(BalanceTestCase):
    def test_deepcopy_unadjusted(self) -> None:
        resp_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "w": [1.0, 1.0]}),
            id_column="id",
            covar_columns=["x"],
            weight_columns=["w"],
        )
        tgt_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "w": [1.0, 1.0]}),
            id_column="id",
            covar_columns=["x"],
            weight_columns=["w"],
        )
        bf = BalanceFrame(sample=resp_sf, sf_target=tgt_sf)
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
            covar_columns=["x"],
            weight_columns=["w"],
        )
        tgt_sf = SampleFrame._create(
            df=pd.DataFrame(
                {"id": [3, 4, 5], "x": [15.0, 25.0, 35.0], "w": [1.0, 1.0, 1.0]}
            ),
            id_column="id",
            covar_columns=["x"],
            weight_columns=["w"],
        )
        bf = BalanceFrame(sample=resp_sf, sf_target=tgt_sf)
        r = repr(bf)
        # New format uses Sample-style display
        self.assertIn("with target set", r)
        self.assertIn("2 observations", r)
        self.assertIn("1 variables", r)
        self.assertIn("x", r)

    def test_str_in_repr(self) -> None:
        resp_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [1], "x": [10.0], "w": [1.0]}),
            id_column="id",
            covar_columns=["x"],
            weight_columns=["w"],
        )
        tgt_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [2], "x": [20.0], "w": [1.0]}),
            id_column="id",
            covar_columns=["x"],
            weight_columns=["w"],
        )
        bf = BalanceFrame(sample=resp_sf, sf_target=tgt_sf)
        # __repr__ includes __str__ output
        self.assertIn(str(bf), repr(bf))


class TestBalanceFrameCreateDirect(BalanceTestCase):
    def test_create_direct(self) -> None:
        """Test that _create() works as an alternative to __new__."""
        resp_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [1], "x": [10.0], "w": [1.0]}),
            id_column="id",
            covar_columns=["x"],
            weight_columns=["w"],
        )
        tgt_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [2], "x": [20.0], "w": [1.0]}),
            id_column="id",
            covar_columns=["x"],
            weight_columns=["w"],
        )
        bf = BalanceFrame._create(sample=resp_sf, sf_target=tgt_sf)
        self.assertIsInstance(bf, BalanceFrame)
        self.assertFalse(bf.is_adjusted)

    def test_properties_accessible(self) -> None:
        """Verify all public properties are accessible after construction."""
        resp_sf = SampleFrame._create(
            df=pd.DataFrame(
                {"id": [1, 2], "x": [10.0, 20.0], "y": [1.0, 0.0], "w": [1.0, 2.0]}
            ),
            id_column="id",
            covar_columns=["x"],
            weight_columns=["w"],
            outcome_columns=["y"],
        )
        tgt_sf = SampleFrame._create(
            df=pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "w": [1.0, 1.0]}),
            id_column="id",
            covar_columns=["x"],
            weight_columns=["w"],
        )
        bf = BalanceFrame(sample=resp_sf, sf_target=tgt_sf)
        # Access all properties
        self.assertIsNotNone(bf.responders)
        self.assertIsNotNone(bf.target)
        self.assertIsNone(bf.unadjusted)
        self.assertFalse(bf.is_adjusted)
        self.assertIsNone(bf.model)
        # Responder data is correct
        self.assertEqual(len(bf.responders.df_covars), 2)
        self.assertListEqual(list(bf.responders.df_weights.columns), ["w"])


class TestBalanceFrameAdjust(BalanceTestCase):
    def setUp(self) -> None:
        super().setUp()
        resp_df = pd.DataFrame(
            {
                "id": [str(i) for i in range(1, 6)],
                "x": [0, 1, 1, 0, 1],
                "happiness": [0.1, 0.5, 0.4, 0.9, 0.2],
                "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        tgt_df = pd.DataFrame(
            {
                "id": [str(i) for i in range(6, 11)],
                "x": [0, 0, 1, 1, 0],
                "happiness": [0.3, 0.6, 0.7, 0.2, 0.5],
                "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        self.resp_sf = SampleFrame.from_frame(resp_df, outcome_columns=["happiness"])
        self.tgt_sf = SampleFrame.from_frame(tgt_df, outcome_columns=["happiness"])
        self.bf = BalanceFrame(sample=self.resp_sf, sf_target=self.tgt_sf)

    def test_adjust_ipw(self) -> None:
        adjusted = self.bf.adjust(method="ipw")
        self.assertTrue(adjusted.is_adjusted)
        model = adjusted.model
        self.assertIsNotNone(model)
        assert model is not None  # for Pyre narrowing
        self.assertIn("method", model)
        self.assertEqual(model["method"], "ipw")

    def test_adjust_preserves_original_weights(self) -> None:
        adjusted = self.bf.adjust(method="null")
        # Weight history columns present
        all_w_cols = adjusted.responders.weight_columns_all
        self.assertIn("weight", all_w_cols)
        self.assertIn("weight_pre_adjust", all_w_cols)
        self.assertIn("weight_adjusted_1", all_w_cols)
        # Active weight keeps its original name
        self.assertEqual(_assert_type(adjusted.weight_series).name, "weight")
        self.assertListEqual(list(adjusted.responders.df_weights.columns), ["weight"])

    def test_adjust_immutability(self) -> None:
        original_weight_vals = self.bf.responders.df_weights.iloc[:, 0].tolist()
        adjusted = self.bf.adjust(method="ipw")
        # Original BalanceFrame is unchanged
        self.assertFalse(self.bf.is_adjusted)
        self.assertIsNone(self.bf.model)
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
        model = adjusted.model
        assert model is not None
        self.assertEqual(model["method"], "dummy")

    def test_adjust_unadjusted_stored(self) -> None:
        adjusted = self.bf.adjust(method="null")
        unadj = adjusted.unadjusted
        self.assertIsNotNone(unadj)
        assert unadj is not None  # for Pyre narrowing
        # unadjusted should have original weight only
        self.assertListEqual(list(unadj.df_weights.columns), ["weight"])

    def test_adjust_repr_shows_adjusted(self) -> None:
        adjusted = self.bf.adjust(method="null")
        self.assertIn("adjusted", repr(adjusted))
        self.assertNotIn("unadjusted", repr(adjusted))

    def test_adjust_weight_metadata(self) -> None:
        adjusted = self.bf.adjust(method="null")
        meta = adjusted.responders.weight_metadata("weight_adjusted_1")
        self.assertEqual(meta["method"], "null")
        self.assertIn("model", meta)

    def test_adjust_compound_null_then_null(self) -> None:
        """Adjusting twice with null method produces a valid adjusted BF."""
        adj1 = self.bf.adjust(method="null")
        adj2 = adj1.adjust(method="null")
        self.assertTrue(adj2.is_adjusted)
        self.assertIsNotNone(adj2.model)

    def test_adjust_compound_preserves_original_pre_adjust(self) -> None:
        """_sf_sample_pre_adjust always points to the original baseline."""
        adj1 = self.bf.adjust(method="null")
        adj2 = adj1.adjust(method="null")
        adj3 = adj2.adjust(method="null")
        # All should share the same original pre-adjust SampleFrame
        self.assertIs(adj1._sf_sample_pre_adjust, self.resp_sf)
        self.assertIs(adj2._sf_sample_pre_adjust, self.resp_sf)
        self.assertIs(adj3._sf_sample_pre_adjust, self.resp_sf)

    def test_adjust_compound_unadjusted_link_is_original(self) -> None:
        """_links['unadjusted'] always points to the original BF."""
        adj1 = self.bf.adjust(method="null")
        adj2 = adj1.adjust(method="null")
        adj3 = adj2.adjust(method="null")
        # The unadjusted link should always be the original (self.bf)
        self.assertIs(adj1._links["unadjusted"], self.bf)
        self.assertIs(adj2._links["unadjusted"], self.bf)
        self.assertIs(adj3._links["unadjusted"], self.bf)
        # _build_links_dict should also return the original pre-adjust
        links2 = adj2._build_links_dict()
        self.assertIs(links2["unadjusted"], self.resp_sf)
        links3 = adj3._build_links_dict()
        self.assertIs(links3["unadjusted"], self.resp_sf)

    def test_adjust_compound_uses_previous_weights_as_design(self) -> None:
        """Second adjust uses the first adjustment's weights as design weights."""
        # First adjust with null: weights pass through unchanged
        adj1 = self.bf.adjust(method="null")
        w1 = adj1.responders.df_weights.iloc[:, 0].copy()

        # Manually set different weights, then adjust again
        adj1_modified = copy.deepcopy(adj1)
        adj1_modified.set_weights(pd.Series([2.0, 3.0, 1.0, 4.0, 0.5]))
        adj2 = adj1_modified.adjust(method="null")
        w2 = adj2.responders.df_weights.iloc[:, 0]
        # Null passes through design weights, so second adjust should
        # use the modified weights as input
        self.assertFalse(w1.equals(w2))

    def test_adjust_compound_model_is_latest(self) -> None:
        """model stores only the latest adjustment's info."""
        adj1 = self.bf.adjust(method="null")
        adj2 = adj1.adjust(method="ipw")
        model = adj2.model
        self.assertIsNotNone(model)
        assert model is not None
        # Should reflect IPW, not null
        self.assertIn("ipw", model["method"])

    def test_adjust_compound_three_steps(self) -> None:
        """Three sequential adjustments maintain all invariants."""
        adj1 = self.bf.adjust(method="null")
        adj2 = adj1.adjust(method="null")
        adj3 = adj2.adjust(method="null")
        self.assertTrue(adj3.is_adjusted)
        self.assertTrue(adj3.has_target)
        # Original BF is unchanged
        self.assertFalse(self.bf.is_adjusted)
        # Pre-adjust is original
        self.assertIs(adj3._sf_sample_pre_adjust, self.resp_sf)
        # Summary should work without error
        adj3.summary()

    def test_adjust_compound_weight_history(self) -> None:
        """Weight history columns accumulate across compound adjustments."""
        adj1 = self.bf.adjust(method="null")
        # After 1st adjust: weight, weight_pre_adjust, weight_adjusted_1
        w1_cols = adj1.responders.weight_columns_all
        self.assertIn("weight", w1_cols)
        self.assertIn("weight_pre_adjust", w1_cols)
        self.assertIn("weight_adjusted_1", w1_cols)
        self.assertEqual(_assert_type(adj1.weight_series).name, "weight")

        adj2 = adj1.adjust(method="null")
        # After 2nd adjust: adds weight_adjusted_2
        w2_cols = adj2.responders.weight_columns_all
        self.assertIn("weight_adjusted_1", w2_cols)
        self.assertIn("weight_adjusted_2", w2_cols)
        self.assertEqual(_assert_type(adj2.weight_series).name, "weight")

        adj3 = adj2.adjust(method="null")
        # After 3rd adjust: adds weight_adjusted_3
        w3_cols = adj3.responders.weight_columns_all
        self.assertIn("weight_adjusted_1", w3_cols)
        self.assertIn("weight_adjusted_2", w3_cols)
        self.assertIn("weight_adjusted_3", w3_cols)
        self.assertEqual(_assert_type(adj3.weight_series).name, "weight")

        # Active weight values should equal the latest weight_adjusted_N
        pd.testing.assert_series_equal(
            adj3.responders._df["weight"],
            adj3.responders._df["weight_adjusted_3"],
            check_names=False,
        )

    def test_adjust_compound_asmd_improvement_vs_original(self) -> None:
        """asmd_improvement() computes total improvement vs original baseline."""
        adj1 = self.bf.adjust(method="ipw")
        adj2 = adj1.adjust(method="null")
        # Since null doesn't change weights, improvement should be the same
        imp1 = adj1.covars().asmd_improvement()
        imp2 = adj2.covars().asmd_improvement()
        self.assertAlmostEqual(imp1, imp2, places=5)

    def test_adjust_stores_method_name_in_model(self) -> None:
        """adjust() stores the method name in the adjustment model dict."""
        adjusted = self.bf.adjust(method="null")
        model = adjusted.model
        self.assertIsInstance(model, dict)
        assert model is not None
        self.assertEqual(model["method"], "null_adjustment")

    def test_adjust_stores_custom_method_name_in_model(self) -> None:
        """adjust() stores the callable's __name__ when a callable method is used."""

        def custom_method(
            sample_df: pd.DataFrame,
            sample_weights: pd.Series,
            target_df: pd.DataFrame,
            target_weights: pd.Series,
        ) -> dict[str, Any]:
            return {"weight": sample_weights, "model": {"info": "test"}}

        adjusted = self.bf.adjust(method=custom_method)
        model = adjusted.model
        self.assertIsInstance(model, dict)
        assert model is not None
        self.assertEqual(model["method"], "custom_method")
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
        adjusted = self.bf.adjust(method="null")
        adjusted_copy = copy.deepcopy(adjusted)
        self.assertTrue(adjusted_copy.is_adjusted)
        self.assertIsNotNone(adjusted_copy.unadjusted)
        self.assertIsNotNone(adjusted_copy.model)
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
        bf = BalanceFrame(sample=self.resp_sf)
        self.assertFalse(bf.has_target())

    def test_has_target_true_with_target(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf, sf_target=self.tgt_sf)
        self.assertTrue(bf.has_target())

    def test_set_target_in_place(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf)
        result = bf.set_target(self.tgt_sf)
        self.assertIs(result, bf)
        self.assertTrue(bf.has_target())
        self.assertIs(bf.target, self.tgt_sf)

    def test_set_target_copy(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf)
        new_bf = bf.set_target(self.tgt_sf, in_place=False)
        self.assertIsNot(new_bf, bf)
        self.assertTrue(new_bf.has_target())
        self.assertFalse(bf.has_target())

    def test_set_target_replaces_existing(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf, sf_target=self.tgt_sf)
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
        bf = BalanceFrame(sample=self.resp_sf, sf_target=self.tgt_sf)
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
        # set_target returns a NEW BalanceFrame; the original stays adjusted.
        retargeted = adjusted.set_target(tgt2_sf)
        self.assertFalse(retargeted.is_adjusted)
        self.assertIsNone(retargeted.model)

    def test_set_target_non_sampleframe_raises(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf)
        with self.assertRaises(TypeError):
            bf.set_target("not a SampleFrame")  # pyre-ignore[6]

    def test_set_target_no_overlap_raises(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf)
        bad_tgt = SampleFrame._create(
            df=pd.DataFrame({"id": [1], "z": [10.0], "w": [1.0]}),
            id_column="id",
            covar_columns=["z"],
            weight_columns=["w"],
        )
        with self.assertRaises(ValueError) as ctx:
            bf.set_target(bad_tgt)
        self.assertIn("no covariate columns", str(ctx.exception))

    def test_adjust_without_target_raises(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf)
        with self.assertRaises(ValueError) as ctx:
            bf.adjust(method="ipw")
        self.assertIn("does not have a target set", str(ctx.exception))

    def test_no_target_repr(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf)
        r = repr(bf)
        # The new __str__ uses Sample-style format: no "with target set" text
        self.assertNotIn("with target set", r)
        self.assertIn("3 observations", r)


class TestBalanceFrameCovarsWeightsOutcomes(BalanceTestCase):
    """Tests for BalanceFrame.covars(), .weights(), .outcomes() integration."""

    def setUp(self) -> None:
        super().setUp()
        resp_df = pd.DataFrame(
            {
                "id": [str(i) for i in range(1, 6)],
                "x": [0, 1, 1, 0, 1],
                "happiness": [0.1, 0.5, 0.4, 0.9, 0.2],
                "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        tgt_df = pd.DataFrame(
            {
                "id": [str(i) for i in range(6, 11)],
                "x": [0, 0, 1, 1, 0],
                "happiness": [0.3, 0.6, 0.7, 0.2, 0.5],
                "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        self.resp_sf = SampleFrame.from_frame(resp_df, outcome_columns=["happiness"])
        self.tgt_sf = SampleFrame.from_frame(tgt_df, outcome_columns=["happiness"])
        self.bf = BalanceFrame(sample=self.resp_sf, sf_target=self.tgt_sf)

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
        adjusted = self.bf.adjust(method="null")
        mean_df = adjusted.covars().mean()
        self.assertIsInstance(mean_df, pd.DataFrame)
        self.assertEqual(mean_df.index.name, "source")
        self.assertIn("self", mean_df.index)
        self.assertIn("target", mean_df.index)
        self.assertIn("unadjusted", mean_df.index)

    def test_covars_std(self) -> None:
        adjusted = self.bf.adjust(method="null")
        std_df = adjusted.covars().std()
        self.assertIsInstance(std_df, pd.DataFrame)
        self.assertIn("self", std_df.index)

    def test_covars_var_of_mean(self) -> None:
        adjusted = self.bf.adjust(method="null")
        vom = adjusted.covars().var_of_mean()
        self.assertIsInstance(vom, pd.DataFrame)

    def test_covars_ci_of_mean(self) -> None:
        adjusted = self.bf.adjust(method="null")
        ci = adjusted.covars().ci_of_mean()
        self.assertIsInstance(ci, pd.DataFrame)

    def test_covars_mean_with_ci(self) -> None:
        adjusted = self.bf.adjust(method="null")
        mwci = adjusted.covars().mean_with_ci()
        self.assertIsInstance(mwci, pd.DataFrame)

    def test_covars_summary(self) -> None:
        adjusted = self.bf.adjust(method="null")
        summary = adjusted.covars().summary()
        self.assertIsInstance(summary, pd.DataFrame)

    def test_covars_model_matrix(self) -> None:
        mm = self.bf.covars().model_matrix()
        self.assertIsInstance(mm, pd.DataFrame)
        self.assertTrue(len(mm) > 0)

    def test_covars_asmd(self) -> None:
        adjusted = self.bf.adjust(method="null")
        asmd_df = adjusted.covars().asmd()
        self.assertIsInstance(asmd_df, pd.DataFrame)

    def test_covars_asmd_improvement(self) -> None:
        adjusted = self.bf.adjust(method="null")
        improvement = adjusted.covars().asmd_improvement()
        self.assertIsInstance(improvement, (float, np.floating))

    def test_covars_links_unadjusted(self) -> None:
        adjusted = self.bf.adjust(method="null")
        linked = adjusted.covars()._balancedf_child_from_linked_samples()
        self.assertIn("self", linked)
        self.assertIn("target", linked)
        self.assertIn("unadjusted", linked)
        self.assertEqual(len(linked), 3)

    def test_covars_links_no_unadjusted(self) -> None:
        linked = self.bf.covars()._balancedf_child_from_linked_samples()
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
        adjusted = self.bf.adjust(method="null")
        summary = adjusted.weights().summary()
        self.assertIsInstance(summary, pd.DataFrame)

    def test_weights_design_effect(self) -> None:
        adjusted = self.bf.adjust(method="null")
        deff = adjusted.weights().design_effect()
        self.assertIsInstance(deff, float)
        self.assertTrue(deff >= 1.0)

    def test_weights_trim(self) -> None:
        adjusted = self.bf.adjust(method="null")
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
        bf = BalanceFrame(sample=resp_sf, sf_target=tgt_sf)
        self.assertIsNone(bf.outcomes())

    def test_outcomes_df(self) -> None:
        result = self.bf.outcomes()
        self.assertIsNotNone(result)
        assert result is not None
        o_df = result.df
        self.assertIsInstance(o_df, pd.DataFrame)
        self.assertIn("happiness", o_df.columns)

    def test_outcomes_summary(self) -> None:
        adjusted = self.bf.adjust(method="null")
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
        adjusted = self.bf.adjust(method="null")
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
        sample_df = sample_df.head(50)
        target_df = target_df.head(100)

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
        new_bf = BalanceFrame(sample=new_resp, sf_target=new_tgt)
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
        sample_df = sample_df.head(50)
        target_df = target_df.head(100)

        # Old API
        old_sample = Sample.from_frame(sample_df, outcome_columns=["happiness"])
        old_target = Sample.from_frame(target_df, outcome_columns=["happiness"])
        old_adjusted = old_sample.set_target(old_target).adjust(method="ipw")
        old_deff = old_adjusted.weights().design_effect()

        # New API
        new_resp = SampleFrame.from_frame(sample_df, outcome_columns=["happiness"])
        new_tgt = SampleFrame.from_frame(target_df, outcome_columns=["happiness"])
        new_adjusted = BalanceFrame(sample=new_resp, sf_target=new_tgt).adjust(
            method="ipw"
        )
        new_deff = new_adjusted.weights().design_effect()

        self.assertAlmostEqual(old_deff, new_deff, places=5)


class TestBalanceFrameSummaryDiagnostics(BalanceTestCase):
    """Tests for BalanceFrame.summary(), diagnostics(), design_effect_prop(),
    _design_effect_diagnostics(), and _quick_adjustment_details().
    """

    def setUp(self) -> None:
        super().setUp()
        self.resp_df = pd.DataFrame(
            {
                "id": [str(i) for i in range(1, 5)],
                "x": [0, 1, 1, 0],
                "y": [0.1, 0.5, 0.4, 0.9],
                "weight": [1.0, 2.0, 1.0, 1.0],
            }
        )
        self.tgt_df = pd.DataFrame(
            {
                "id": [str(i) for i in range(5, 9)],
                "x": [0, 0, 1, 1],
                "weight": [1.0, 1.0, 1.0, 1.0],
            }
        )
        resp_sf = SampleFrame.from_frame(
            self.resp_df,
            outcome_columns=["y"],
        )
        tgt_sf = SampleFrame.from_frame(self.tgt_df)
        self.bf = BalanceFrame(sample=resp_sf, sf_target=tgt_sf)
        self.bf_adjusted = self.bf.adjust(method="null")

    # --- summary() tests ---

    def test_summary_adjusted_contains_sections(self) -> None:
        result = self.bf_adjusted.summary()
        self.assertIn("Adjustment details:", result)
        self.assertIn("Covariate diagnostics:", result)
        self.assertIn("Weight diagnostics:", result)

    def test_summary_unadjusted_has_covar_diagnostics(self) -> None:
        result = self.bf.summary()
        self.assertIn("Covariate diagnostics:", result)
        self.assertNotIn("Weight diagnostics:", result)
        self.assertNotIn("Adjustment details:", result)

    def test_summary_adjusted_with_outcomes(self) -> None:
        result = self.bf_adjusted.summary()
        self.assertIn("Outcome weighted means:", result)
        self.assertIn("y", result)

    def test_summary_matches_sample(self) -> None:
        """Cross-validate summary() output with the old Sample API."""
        old_sample = Sample.from_frame(
            self.resp_df,
            id_column="id",
            weight_column="weight",
            outcome_columns="y",
        )
        old_target = Sample.from_frame(
            self.tgt_df,
            id_column="id",
            weight_column="weight",
        )
        old_adjusted = old_sample.set_target(old_target).adjust(method="null")

        old_summary = old_adjusted.summary()
        new_summary = self.bf_adjusted.summary()
        # The old Sample API normalises method to "null" while BF preserves
        # the raw name "null_adjustment".  Normalise both before comparing.
        self.assertEqual(
            old_summary.replace("null_adjustment", "null"),
            new_summary.replace("null_adjustment", "null"),
        )

    # --- diagnostics() tests ---

    def test_diagnostics_adjusted_columns(self) -> None:
        result = self.bf_adjusted.diagnostics()
        self.assertEqual(result.columns.tolist(), ["metric", "val", "var"])

    def test_diagnostics_unadjusted_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.bf.diagnostics()

    def test_diagnostics_has_expected_metrics(self) -> None:
        result = self.bf_adjusted.diagnostics()
        metrics = result["metric"].unique().tolist()
        self.assertIn("size", metrics)
        self.assertIn("weights_diagnostics", metrics)
        self.assertIn("adjustment_method", metrics)

    def test_diagnostics_matches_sample(self) -> None:
        """Cross-validate diagnostics() output with the old Sample API."""
        old_sample = Sample.from_frame(
            self.resp_df,
            id_column="id",
            weight_column="weight",
            outcome_columns="y",
        )
        old_target = Sample.from_frame(
            self.tgt_df,
            id_column="id",
            weight_column="weight",
        )
        old_adjusted = old_sample.set_target(old_target).adjust(method="null")

        old_diag = old_adjusted.diagnostics()
        new_diag = self.bf_adjusted.diagnostics()

        self.assertEqual(old_diag.shape, new_diag.shape)
        self.assertEqual(old_diag["metric"].tolist(), new_diag["metric"].tolist())

        # The old Sample API normalises the method name to "null" while BF
        # preserves the raw name "null_adjustment".  Normalise both sides.
        def _normalise(v: object) -> object:
            return v.replace("null_adjustment", "null") if isinstance(v, str) else v

        old_vars = [_normalise(v) for v in old_diag["var"].tolist()]
        new_vars = [_normalise(v) for v in new_diag["var"].tolist()]
        self.assertEqual(old_vars, new_vars)

        # Compare numeric vals with tolerance
        for i in range(len(old_diag)):
            old_val = old_diag.iloc[i]["val"]
            new_val = new_diag.iloc[i]["val"]
            if isinstance(old_val, (int, float, np.floating)):
                self.assertAlmostEqual(float(old_val), float(new_val), places=5)
            else:
                old_str = str(old_val).replace("null_adjustment", "null")
                self.assertEqual(old_str, str(new_val))

    def test_diagnostics_load_data_equivalence(self) -> None:
        """Numerical equivalence using load_data()."""
        target_df, sample_df = load_data()
        assert sample_df is not None
        assert target_df is not None
        sample_df = sample_df.head(50)

        target_head = target_df.head(100).drop(columns=["happiness"], errors="ignore")

        old_sample = Sample.from_frame(
            sample_df,
            id_column="id",
            outcome_columns=["happiness"],
        )
        old_target = Sample.from_frame(
            target_head,
            id_column="id",
        )
        old_adjusted = old_sample.set_target(old_target).adjust(method="ipw")

        resp_sf = SampleFrame.from_frame(sample_df, outcome_columns=["happiness"])
        tgt_sf = SampleFrame.from_frame(target_head)
        bf = BalanceFrame(sample=resp_sf, sf_target=tgt_sf)
        bf_adjusted = bf.adjust(method="ipw")

        old_diag = old_adjusted.diagnostics()
        new_diag = bf_adjusted.diagnostics()

        self.assertEqual(old_diag.shape, new_diag.shape)
        self.assertEqual(old_diag["metric"].tolist(), new_diag["metric"].tolist())

    def test_summary_load_data_equivalence(self) -> None:
        """Numerical equivalence of summary() using load_data()."""
        target_df, sample_df = load_data()
        assert sample_df is not None
        assert target_df is not None
        sample_df = sample_df.head(50)

        target_head = target_df.head(100).drop(columns=["happiness"], errors="ignore")

        old_sample = Sample.from_frame(
            sample_df,
            id_column="id",
            outcome_columns=["happiness"],
        )
        old_target = Sample.from_frame(
            target_head,
            id_column="id",
        )
        old_adjusted = old_sample.set_target(old_target).adjust(method="ipw")

        resp_sf = SampleFrame.from_frame(sample_df, outcome_columns=["happiness"])
        tgt_sf = SampleFrame.from_frame(target_head)
        bf = BalanceFrame(sample=resp_sf, sf_target=tgt_sf)
        bf_adjusted = bf.adjust(method="ipw")

        self.assertEqual(old_adjusted.summary(), bf_adjusted.summary())

    # --- _design_effect_diagnostics() tests ---

    def test_design_effect_diagnostics_returns_tuple(self) -> None:
        de, ess, essp = self.bf._design_effect_diagnostics()
        self.assertIsNotNone(de)
        self.assertIsNotNone(ess)
        self.assertIsNotNone(essp)
        self.assertIsInstance(de, float)

    # --- _quick_adjustment_details() tests ---

    def test_quick_adjustment_details_adjusted(self) -> None:
        details = self.bf_adjusted._quick_adjustment_details()
        self.assertIsInstance(details, list)
        method_lines = [d for d in details if d.startswith("method:")]
        self.assertEqual(len(method_lines), 1)
        self.assertIn("null", method_lines[0])

    def test_quick_adjustment_details_with_precomputed(self) -> None:
        """Pre-computed de/ess/essp are used instead of recomputing."""
        details = self.bf_adjusted._quick_adjustment_details(de=2.0, ess=5.0, essp=0.5)
        de_lines = [d for d in details if "design effect" in d]
        self.assertEqual(len(de_lines), 1)
        self.assertIn("2.000", de_lines[0])


class TestBalanceFrameParityHelpers(BalanceTestCase):
    """Tests for df_ignored access and id_column parity methods."""

    def setUp(self) -> None:
        super().setUp()
        self.resp_df = pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "x1": [10.0, 20.0, 30.0],
                "weight": [1.0, 1.0, 1.0],
                "region": ["US", "UK", "CA"],
            }
        )
        self.tgt_df = pd.DataFrame(
            {
                "id": ["4", "5", "6"],
                "x1": [15.0, 25.0, 35.0],
                "weight": [1.0, 1.0, 1.0],
            }
        )
        self.resp_sf = SampleFrame.from_frame(self.resp_df, ignored_columns=["region"])
        self.tgt_sf = SampleFrame.from_frame(self.tgt_df)
        self.bf = BalanceFrame(sample=self.resp_sf, sf_target=self.tgt_sf)

    def test_df_ignored_present(self) -> None:
        result = self.bf.responders.df_ignored
        self.assertIsNotNone(result)
        self.assertListEqual(list(result.columns), ["region"])
        self.assertListEqual(list(result["region"]), ["US", "UK", "CA"])

    def test_df_ignored_none(self) -> None:
        resp = SampleFrame.from_frame(
            pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 1.0]})
        )
        tgt = SampleFrame.from_frame(
            pd.DataFrame({"id": [3, 4], "x": [15.0, 25.0], "weight": [1.0, 1.0]})
        )
        bf = BalanceFrame(sample=resp, sf_target=tgt)
        self.assertIsNone(bf.responders.df_ignored)

    def test_id_column_returns_series(self) -> None:
        result = self.bf.id_column
        self.assertIsInstance(result, pd.Series)
        assert result is not None
        self.assertEqual(result.tolist(), ["1", "2", "3"])

    def test_id_column_delegates_to_responders(self) -> None:
        pd.testing.assert_series_equal(self.bf.id_column, self.bf.responders.id_column)


class TestBalanceFrameDfExportFilter(BalanceTestCase):
    """Tests for df property, keep_only_some_rows_columns, to_csv, to_download."""

    def setUp(self) -> None:
        super().setUp()
        self.resp_df = pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "x1": [10.0, 20.0, 30.0],
                "x2": [1.0, 2.0, 3.0],
                "weight": [1.0, 1.0, 1.0],
                "y": [100.0, 200.0, 300.0],
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
        self.resp_sf = SampleFrame.from_frame(self.resp_df, outcome_columns=["y"])
        self.tgt_sf = SampleFrame.from_frame(self.tgt_df)
        self.bf = BalanceFrame(sample=self.resp_sf, sf_target=self.tgt_sf)
        self.bf_adjusted = self.bf.adjust(method="null")

    # --- df_all property (combined view) ---

    def test_df_all_unadjusted_has_source_column(self) -> None:
        result = self.bf.df_all
        self.assertIn("source", result.columns)
        self.assertCountEqual(result["source"].unique().tolist(), ["self", "target"])

    def test_df_all_unadjusted_row_count(self) -> None:
        result = self.bf.df_all
        self.assertEqual(len(result), 6)  # 3 resp + 3 target

    def test_df_all_adjusted_has_unadjusted_source(self) -> None:
        result = self.bf_adjusted.df_all
        self.assertCountEqual(
            result["source"].unique().tolist(),
            ["self", "target", "unadjusted"],
        )

    def test_df_all_adjusted_row_count(self) -> None:
        result = self.bf_adjusted.df_all
        self.assertEqual(len(result), 9)  # 3 resp + 3 target + 3 unadjusted

    def test_df_all_contains_all_columns(self) -> None:
        result = self.bf.df_all
        for col in ["id", "x1", "x2", "weight", "source"]:
            self.assertIn(col, result.columns)

    # --- df property (flat, user-facing) ---

    def test_df_returns_responders_only(self) -> None:
        result = self.bf.df
        self.assertNotIn("source", result.columns)
        self.assertEqual(len(result), 3)  # responders only

    def test_df_contains_data_columns(self) -> None:
        result = self.bf.df
        for col in ["id", "x1", "x2", "weight"]:
            self.assertIn(col, result.columns)

    # --- keep_only_some_rows_columns ---

    def test_keep_rows_filter(self) -> None:
        filtered = self.bf.keep_only_some_rows_columns(rows_to_keep="x1 > 15")
        self.assertEqual(len(filtered.responders._df), 2)  # 20, 30

    def test_keep_rows_immutable(self) -> None:
        self.bf.keep_only_some_rows_columns(rows_to_keep="x1 > 15")
        self.assertEqual(len(self.bf.responders._df), 3)  # original unchanged

    def test_keep_columns_filter(self) -> None:
        filtered = self.bf.keep_only_some_rows_columns(columns_to_keep=["x1"])
        self.assertEqual(sorted(filtered.responders._column_roles["covars"]), ["x1"])

    def test_keep_columns_preserves_special(self) -> None:
        filtered = self.bf.keep_only_some_rows_columns(columns_to_keep=["x1"])
        self.assertIn("id", filtered.responders._df.columns)
        self.assertIn("weight", filtered.responders._df.columns)

    def test_keep_none_returns_self(self) -> None:
        result = self.bf.keep_only_some_rows_columns()
        self.assertIs(result, self.bf)

    def test_keep_rows_on_adjusted(self) -> None:
        filtered = self.bf_adjusted.keep_only_some_rows_columns(rows_to_keep="x1 > 15")
        self.assertEqual(len(filtered.responders._df), 2)
        self.assertTrue(filtered.is_adjusted)
        self.assertEqual(len(filtered.unadjusted._df), 2)

    def test_keep_rows_target_undefined_variable(self) -> None:
        """Filter by outcome column that target doesn't have — should warn, not fail."""
        filtered = self.bf.keep_only_some_rows_columns(rows_to_keep="y > 150")
        self.assertEqual(len(filtered.responders._df), 2)
        assert filtered.target is not None
        self.assertEqual(len(filtered.target._df), 3)

    def test_keep_columns_with_no_active_weight(self) -> None:
        """Column filtering works when SampleFrame has no active weight column."""
        resp_sf = SampleFrame._create(
            df=pd.DataFrame({"id": ["1", "2"], "x1": [10.0, 20.0]}),
            id_column="id",
            covar_columns=["x1"],
            weight_columns=[],
        )
        tgt_sf = SampleFrame._create(
            df=pd.DataFrame({"id": ["3", "4"], "x1": [15.0, 25.0]}),
            id_column="id",
            covar_columns=["x1"],
            weight_columns=[],
        )
        bf = BalanceFrame(sample=resp_sf, sf_target=tgt_sf)
        filtered = bf.keep_only_some_rows_columns(columns_to_keep=["x1"])
        self.assertIn("x1", filtered.responders._df.columns)
        self.assertIn("id", filtered.responders._df.columns)
        self.assertIsNone(filtered.responders._weight_column_name)

    # --- to_csv ---

    def test_to_csv_returns_string(self) -> None:
        result = self.bf.to_csv()
        self.assertIsInstance(result, str)
        assert result is not None
        # df (flat) does not include "source" column
        self.assertIn("id", result)
        self.assertIn("weight", result)

    def test_to_csv_roundtrip(self) -> None:
        csv_text = self.bf.to_csv()
        roundtrip_df = pd.read_csv(io.StringIO(csv_text))
        self.assertEqual(len(roundtrip_df), 3)  # responders only
        self.assertNotIn("source", roundtrip_df.columns)

    def test_to_csv_to_file(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            self.bf.to_csv(f.name)
            roundtrip = pd.read_csv(f.name)
            self.assertEqual(len(roundtrip), 3)  # responders only

    # --- to_download ---

    def test_to_download_returns_filelink(self) -> None:
        import tempfile

        from IPython.lib.display import FileLink

        result = self.bf.to_download(tempdir=tempfile.gettempdir())
        self.assertIsInstance(result, FileLink)


class TestBalanceFrameMissingIntegration(BalanceTestCase):
    """Integration tests covering scenarios not addressed by the focused unit tests."""

    def setUp(self) -> None:
        super().setUp()
        self.resp_df = pd.DataFrame(
            {
                "id": [str(i) for i in range(1, 5)],
                "x": [0.0, 1.0, 1.0, 0.0],
                "y": [0.1, 0.5, 0.4, 0.9],
                "weight": [1.0, 2.0, 1.0, 1.0],
            }
        )
        self.tgt_df = pd.DataFrame(
            {
                "id": [str(i) for i in range(5, 9)],
                "x": [0.0, 0.0, 1.0, 1.0],
                "weight": [1.0, 1.0, 1.0, 1.0],
            }
        )
        self.resp_sf = SampleFrame.from_frame(self.resp_df, outcome_columns=["y"])
        self.tgt_sf = SampleFrame.from_frame(self.tgt_df)
        self.bf = BalanceFrame(sample=self.resp_sf, sf_target=self.tgt_sf)

    def test_full_pipeline_adjust_summary_diagnostics_to_csv(self) -> None:
        """Full pipeline: adjust -> summary -> diagnostics -> to_csv all succeed."""
        adjusted = self.bf.adjust(method="null")

        summary_str = adjusted.summary()
        self.assertIsInstance(summary_str, str)
        self.assertIn("Covariate diagnostics:", summary_str)
        self.assertIn("Adjustment details:", summary_str)

        diag_df = adjusted.diagnostics()
        self.assertIsInstance(diag_df, pd.DataFrame)
        self.assertEqual(diag_df.columns.tolist(), ["metric", "val", "var"])
        self.assertGreater(len(diag_df), 0)

        csv_str = adjusted.to_csv()
        self.assertIsInstance(csv_str, str)
        assert csv_str is not None
        roundtrip = pd.read_csv(io.StringIO(csv_str))
        # df (flat) does not include "source" column
        self.assertNotIn("source", roundtrip.columns)

    def test_null_method_adjustment(self) -> None:
        """Null method adjustment leaves weights unchanged."""
        adjusted = self.bf.adjust(method="null")

        self.assertTrue(adjusted.is_adjusted)
        model = adjusted.model
        self.assertIsNotNone(model)
        assert model is not None
        self.assertEqual(model["method"], "null_adjustment")

        orig_weights = self.resp_sf.df_weights.iloc[:, 0].tolist()
        adj_weights = adjusted.responders.df_weights.iloc[:, 0].tolist()
        for orig, adj in zip(orig_weights, adj_weights):
            self.assertAlmostEqual(orig, adj, places=8)

    def test_to_csv_unadjusted_is_flat(self) -> None:
        """to_csv() on an unadjusted BalanceFrame exports flat responders only."""
        csv_str = self.bf.to_csv()
        self.assertIsInstance(csv_str, str)
        assert csv_str is not None
        roundtrip = pd.read_csv(io.StringIO(csv_str))
        self.assertNotIn("source", roundtrip.columns)
        # Row count matches responders (this fixture has 4 rows)
        self.assertEqual(len(roundtrip), len(self.bf._sf_sample))

    def test_keep_only_some_rows_columns_expression_matches_no_rows(self) -> None:
        """Filter expression that matches no rows produces a 0-row BalanceFrame."""
        filtered = self.bf.keep_only_some_rows_columns(rows_to_keep="x > 9999")
        self.assertEqual(len(filtered.responders._df), 0)
        assert filtered.target is not None
        self.assertEqual(len(filtered.target._df), 0)
        self.assertEqual(len(self.bf.responders._df), 4)


class TestBalanceFrameEndToEnd(BalanceTestCase):
    """End-to-end integration test verifying full BalanceFrame workflow produces
    numerically equivalent results to the old Sample API for all 4 weighting
    methods: ipw, cbps, rake, poststratify.

    Exercises: load_data -> SampleFrame.from_frame -> BalanceFrame -> adjust ->
    covars().mean(), covars().asmd(), weights().summary(), outcomes().mean(),
    summary(), diagnostics(), weights().design_effect(), to_csv().
    """

    def _assert_numeric_df_equal(
        self,
        old_df: pd.DataFrame,
        new_df: pd.DataFrame,
        places: int = 5,
        msg: str = "",
    ) -> None:
        """Assert numeric columns of two DataFrames are almost equal."""
        old_numeric = old_df.select_dtypes(include=[np.number])
        new_numeric = new_df.select_dtypes(include=[np.number])
        for col in old_numeric.columns:
            if col in new_numeric.columns:
                old_vals = old_numeric[col].dropna()
                new_vals = new_numeric[col].dropna()
                self.assertEqual(
                    len(old_vals),
                    len(new_vals),
                    f"{msg} column '{col}' length mismatch",
                )
                for old_val, new_val in zip(old_vals, new_vals):
                    self.assertAlmostEqual(
                        old_val,
                        new_val,
                        places=places,
                        msg=f"{msg} column '{col}'",
                    )

    def _run_equivalence_for_method(
        self,
        method: Literal["cbps", "ipw", "null", "poststratify", "rake"],
    ) -> None:
        """Run full workflow through both APIs and verify numerical equivalence
        for a given adjustment method."""
        target_df, sample_df = load_data()
        assert target_df is not None and sample_df is not None
        # Poststratify needs all cell combos present; use more rows for it.
        # NOTE: we compare method against the literal in a bool variable so
        # that Pyre doesn't narrow (and then widen to LiteralString) the
        # Literal-typed ``method`` parameter.
        use_full_data = method == "poststratify"
        if not use_full_data:
            sample_df = sample_df.head(50)
            target_df = target_df.head(100)

        # --- Old Sample API ---
        old_sample = Sample.from_frame(sample_df, outcome_columns=["happiness"])
        old_target = Sample.from_frame(target_df, outcome_columns=["happiness"])
        old_adjusted = old_sample.set_target(old_target).adjust(method=method)

        # --- New BalanceFrame API ---
        new_resp = SampleFrame.from_frame(sample_df, outcome_columns=["happiness"])
        new_tgt = SampleFrame.from_frame(target_df, outcome_columns=["happiness"])
        new_bf = BalanceFrame(sample=new_resp, sf_target=new_tgt)
        new_adjusted = new_bf.adjust(method=method)

        # --- covars().mean() ---
        old_covars_mean = old_adjusted.covars().mean()
        new_covars_mean = new_adjusted.covars().mean()
        self._assert_numeric_df_equal(
            old_covars_mean, new_covars_mean, msg=f"{method}: covars().mean()"
        )

        # --- covars().asmd() ---
        old_covars_asmd = old_adjusted.covars().asmd()
        new_covars_asmd = new_adjusted.covars().asmd()
        self._assert_numeric_df_equal(
            old_covars_asmd, new_covars_asmd, msg=f"{method}: covars().asmd()"
        )

        # --- weights().summary() ---
        old_weights_summary = old_adjusted.weights().summary()
        new_weights_summary = new_adjusted.weights().summary()
        self._assert_numeric_df_equal(
            old_weights_summary,
            new_weights_summary,
            msg=f"{method}: weights().summary()",
        )

        # --- weights().design_effect() ---
        old_de = float(old_adjusted.weights().design_effect())
        new_de = float(new_adjusted.weights().design_effect())
        self.assertAlmostEqual(
            old_de, new_de, places=5, msg=f"{method}: weights().design_effect()"
        )

        # --- outcomes().mean() ---
        old_outcomes = old_adjusted.outcomes()
        new_outcomes = new_adjusted.outcomes()
        assert old_outcomes is not None
        assert new_outcomes is not None
        self._assert_numeric_df_equal(
            old_outcomes.mean(),
            new_outcomes.mean(),
            msg=f"{method}: outcomes().mean()",
        )

        # --- summary() ---
        old_summary = old_adjusted.summary()
        new_summary = new_adjusted.summary()
        self.assertIsInstance(old_summary, str)
        self.assertIsInstance(new_summary, str)
        # Both should contain the same key sections
        for section in ["Covariate diagnostics:", "Adjustment details:"]:
            self.assertIn(section, new_summary, f"{method}: missing '{section}'")

        # --- to_csv() ---
        csv_output = new_adjusted.to_csv()
        self.assertIsInstance(csv_output, str)
        assert csv_output is not None
        roundtrip = pd.read_csv(io.StringIO(csv_output))
        # df (flat, user-facing) does not have a "source" column
        self.assertNotIn("source", roundtrip.columns)

    def test_ipw_end_to_end_equivalence(self) -> None:
        """Full workflow equivalence for IPW (inverse propensity weighting)."""
        self._run_equivalence_for_method("ipw")

    def test_cbps_end_to_end_equivalence(self) -> None:
        """Full workflow equivalence for CBPS (covariate balancing propensity score)."""
        self._run_equivalence_for_method("cbps")

    def test_rake_end_to_end_equivalence(self) -> None:
        """Full workflow equivalence for raking."""
        self._run_equivalence_for_method("rake")

    def test_poststratify_end_to_end_equivalence(self) -> None:
        """Full workflow equivalence for post-stratification."""
        self._run_equivalence_for_method("poststratify")

    def test_unadjusted_covars_mean_sources(self) -> None:
        """Verify unadjusted BalanceFrame covars().mean() has self+target only."""
        resp = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": [str(i) for i in range(5)],
                    "x": [1.0, 2, 3, 4, 5],
                    "weight": [1.0] * 5,
                }
            ),
        )
        tgt = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": [str(i) for i in range(5, 10)],
                    "x": [2.0, 3, 4, 5, 6],
                    "weight": [1.0] * 5,
                }
            ),
        )
        bf = BalanceFrame(sample=resp, sf_target=tgt)

        mean_df = bf.covars().mean()
        self.assertEqual(mean_df.index.name, "source")
        self.assertIn("self", mean_df.index)
        self.assertIn("target", mean_df.index)
        self.assertNotIn("unadjusted", mean_df.index)

    def test_adjusted_covars_mean_sources(self) -> None:
        """Verify adjusted BalanceFrame covars().mean() has self+target+unadjusted."""
        resp = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": [str(i) for i in range(5)],
                    "x": [1.0, 2, 3, 4, 5],
                    "weight": [1.0] * 5,
                }
            ),
        )
        tgt = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": [str(i) for i in range(5, 10)],
                    "x": [2.0, 3, 4, 5, 6],
                    "weight": [1.0] * 5,
                }
            ),
        )
        bf = BalanceFrame(sample=resp, sf_target=tgt)
        adjusted = bf.adjust(method="null")

        mean_df = adjusted.covars().mean()
        self.assertEqual(mean_df.index.name, "source")
        self.assertIn("self", mean_df.index)
        self.assertIn("target", mean_df.index)
        self.assertIn("unadjusted", mean_df.index)

    def test_immutability_across_methods(self) -> None:
        """Verify adjust() does not mutate the original BalanceFrame."""
        resp = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": [str(i) for i in range(5)],
                    "x": [1.0, 2, 3, 4, 5],
                    "weight": [1.0] * 5,
                }
            ),
        )
        tgt = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": [str(i) for i in range(5, 10)],
                    "x": [2.0, 3, 4, 5, 6],
                    "weight": [1.0] * 5,
                }
            ),
        )
        bf = BalanceFrame(sample=resp, sf_target=tgt)

        original_weights = bf.responders.df_weights.copy()

        adjusted_null1 = bf.adjust(method="null")
        adjusted_null2 = bf.adjust(method="null")

        # Original is unchanged
        self.assertFalse(bf.is_adjusted)
        pd.testing.assert_frame_equal(bf.responders.df_weights, original_weights)

        # Each adjusted BF is independent
        self.assertTrue(adjusted_null1.is_adjusted)
        self.assertTrue(adjusted_null2.is_adjusted)
        self.assertIsNot(adjusted_null1, adjusted_null2)

    def test_diagnostics_equivalence(self) -> None:
        """Verify diagnostics() produces consistent output between APIs."""
        target_df, sample_df = load_data()
        assert target_df is not None and sample_df is not None
        sample_df = sample_df.head(50)
        target_df = target_df.head(100)

        old_sample = Sample.from_frame(sample_df, outcome_columns=["happiness"])
        old_target = Sample.from_frame(target_df, outcome_columns=["happiness"])
        old_adjusted = old_sample.set_target(old_target).adjust(method="ipw")

        resp = SampleFrame.from_frame(sample_df, outcome_columns=["happiness"])
        tgt = SampleFrame.from_frame(target_df, outcome_columns=["happiness"])
        new_adjusted = BalanceFrame(sample=resp, sf_target=tgt).adjust(method="ipw")

        old_diag = old_adjusted.diagnostics()
        new_diag = new_adjusted.diagnostics()

        self.assertEqual(old_diag.shape, new_diag.shape)
        self.assertEqual(old_diag["metric"].tolist(), new_diag["metric"].tolist())

    def test_covars_mean_equivalence(self) -> None:
        """Verify covars().mean() matches between old and new APIs."""
        target_df, sample_df = load_data()
        assert target_df is not None and sample_df is not None
        sample_df = sample_df.head(50)
        target_df = target_df.head(100)

        old_sample = Sample.from_frame(sample_df, outcome_columns=["happiness"])
        old_target = Sample.from_frame(target_df, outcome_columns=["happiness"])
        old_adjusted = old_sample.set_target(old_target).adjust(method="ipw")

        resp = SampleFrame.from_frame(sample_df, outcome_columns=["happiness"])
        tgt = SampleFrame.from_frame(target_df, outcome_columns=["happiness"])
        new_adjusted = BalanceFrame(sample=resp, sf_target=tgt).adjust(method="ipw")

        old_means = old_adjusted.covars().mean()
        new_means = new_adjusted.covars().mean()
        self._assert_numeric_df_equal(old_means, new_means, msg="covars().mean()")


class TestBalanceFrameFromSample(BalanceTestCase):
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
        self.sample = Sample.from_frame(self.resp_df)
        self.target = Sample.from_frame(self.tgt_df)

    def test_from_sample_unadjusted(self) -> None:
        sample_with_target = self.sample.set_target(self.target)
        bf = BalanceFrame.from_sample(sample_with_target)
        self.assertFalse(bf.is_adjusted)
        self.assertIsNone(bf.unadjusted)
        self.assertEqual(len(bf.responders._df), 3)
        self.assertIsNotNone(bf.target)
        assert bf.target is not None
        self.assertEqual(len(bf.target._df), 3)

    def test_from_sample_adjusted(self) -> None:
        adjusted = self.sample.set_target(self.target).adjust(method="null")
        bf = BalanceFrame.from_sample(adjusted)
        self.assertTrue(bf.is_adjusted)
        self.assertIsNotNone(bf.unadjusted)
        self.assertIsNotNone(bf.model)

    def test_from_sample_covars_preserved(self) -> None:
        sample_with_target = self.sample.set_target(self.target)
        bf = BalanceFrame.from_sample(sample_with_target)
        self.assertEqual(
            sorted(bf.responders._column_roles["covars"]),
            sorted(self.sample._covar_columns_names()),
        )

    def test_from_sample_no_target_raises(self) -> None:
        with self.assertRaises(ValueError):
            BalanceFrame.from_sample(self.sample)

    def test_from_sample_type_error(self) -> None:
        with self.assertRaises(TypeError):
            BalanceFrame.from_sample("not a sample")  # pyre-ignore[6]

    def test_from_sample_with_outcomes(self) -> None:
        resp_df = pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "x1": [10.0, 20.0, 30.0],
                "y": [0.5, 0.8, 0.3],
                "weight": [1.0, 1.0, 1.0],
            }
        )
        tgt_df = pd.DataFrame(
            {
                "id": ["4", "5", "6"],
                "x1": [15.0, 25.0, 35.0],
                "weight": [1.0, 1.0, 1.0],
            }
        )
        sample = Sample.from_frame(resp_df, outcome_columns=["y"])
        target = Sample.from_frame(tgt_df)
        bf = BalanceFrame.from_sample(sample.set_target(target))
        self.assertEqual(bf.responders._column_roles["outcomes"], ["y"])

    def test_from_sample_roundtrip_equivalence(self) -> None:
        """Verify that converting Sample->BalanceFrame produces equivalent results."""
        target_df, sample_df = load_data()
        assert target_df is not None and sample_df is not None
        sample_df = sample_df.head(50)
        target_df = target_df.head(100)

        old_sample = Sample.from_frame(sample_df, outcome_columns=["happiness"])
        old_target = Sample.from_frame(target_df, outcome_columns=["happiness"])
        old_adjusted = old_sample.set_target(old_target).adjust(method="ipw")

        bf = BalanceFrame.from_sample(old_adjusted)
        self.assertTrue(bf.is_adjusted)

        # Compare covars mean numerically
        old_mean = old_adjusted.covars().mean()
        new_mean = bf.covars().mean()
        old_num_cols = old_mean.select_dtypes(include=[np.number])
        new_num_cols = new_mean.select_dtypes(include=[np.number])
        for col in old_num_cols.columns:
            if col in new_num_cols.columns:
                for o_val, n_val in zip(
                    old_num_cols[col].dropna(), new_num_cols[col].dropna()
                ):
                    self.assertAlmostEqual(o_val, n_val, places=5)


class TestBalanceFrameToSample(BalanceTestCase):
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
        self.bf = BalanceFrame(sample=self.resp_sf, sf_target=self.tgt_sf)

    def test_to_sample_has_target(self) -> None:
        s = self.bf.to_sample()
        self.assertTrue(s.has_target())

    def test_to_sample_not_adjusted(self) -> None:
        s = self.bf.to_sample()
        self.assertFalse(s.is_adjusted())

    def test_to_sample_covars_preserved(self) -> None:
        s = self.bf.to_sample()
        self.assertEqual(sorted(s._covar_columns_names()), ["x1", "x2"])

    def test_to_sample_weight_values(self) -> None:
        s = self.bf.to_sample()
        self.assertEqual(s.weight_series.tolist(), [1.0, 1.0, 1.0])

    def test_to_sample_id_values(self) -> None:
        s = self.bf.to_sample()
        self.assertEqual(s.id_column.tolist(), ["1", "2", "3"])

    def test_to_sample_target_data(self) -> None:
        s = self.bf.to_sample()
        target = s._links["target"]
        self.assertEqual(target.id_column.tolist(), ["4", "5", "6"])
        self.assertEqual(sorted(target._covar_columns_names()), ["x1", "x2"])

    def test_to_sample_adjusted(self) -> None:
        adjusted_bf = self.bf.adjust(method="null")
        s = adjusted_bf.to_sample()
        self.assertTrue(s.is_adjusted())
        self.assertTrue(s.has_target())
        self.assertIsNotNone(s.model)

    def test_to_sample_adjusted_weight_column(self) -> None:
        adjusted_bf = self.bf.adjust(method="null")
        s = adjusted_bf.to_sample()
        self.assertEqual(s.weight_series.name, "weight")

    def test_to_sample_with_outcomes(self) -> None:
        resp_df = pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "x1": [10.0, 20.0, 30.0],
                "y": [0.5, 0.8, 0.3],
                "weight": [1.0, 1.0, 1.0],
            }
        )
        resp_sf = SampleFrame.from_frame(resp_df, outcome_columns=["y"])
        bf = BalanceFrame(sample=resp_sf, sf_target=self.tgt_sf)
        s = bf.to_sample()
        self.assertIsNotNone(s._outcome_columns)
        assert s._outcome_columns is not None
        self.assertEqual(s._outcome_columns.columns.tolist(), ["y"])

    def test_to_sample_roundtrip_sample_bf_sample(self) -> None:
        """Sample -> BalanceFrame -> Sample round-trip preserves data."""
        original = Sample.from_frame(self.resp_df)
        target = Sample.from_frame(self.tgt_df)
        original_with_target = original.set_target(target)

        bf = BalanceFrame.from_sample(original_with_target)
        roundtrip = bf.to_sample()

        self.assertTrue(roundtrip.has_target())
        self.assertFalse(roundtrip.is_adjusted())
        self.assertEqual(
            sorted(roundtrip._covar_columns_names()),
            sorted(original._covar_columns_names()),
        )
        for o_val, r_val in zip(
            _assert_type(original.weight_series).tolist(),
            _assert_type(roundtrip.weight_series).tolist(),
        ):
            self.assertAlmostEqual(o_val, r_val, places=10)

    def test_to_sample_roundtrip_adjusted(self) -> None:
        """Sample -> adjust -> BalanceFrame -> Sample round-trip for adjusted."""
        original = Sample.from_frame(self.resp_df)
        target = Sample.from_frame(self.tgt_df)
        adjusted = original.set_target(target).adjust(method="null")

        bf = BalanceFrame.from_sample(adjusted)
        roundtrip = bf.to_sample()

        self.assertTrue(roundtrip.is_adjusted())
        self.assertTrue(roundtrip.has_target())
        for o_val, r_val in zip(
            _assert_type(adjusted.weight_series).tolist(),
            _assert_type(roundtrip.weight_series).tolist(),
        ):
            self.assertAlmostEqual(o_val, r_val, places=10)

    def test_to_sample_roundtrip_load_data(self) -> None:
        """Round-trip with load_data ensures real-world equivalence."""
        target_df, sample_df = load_data()
        assert target_df is not None and sample_df is not None
        sample_df = sample_df.head(50)
        target_df = target_df.head(100)

        old_sample = Sample.from_frame(sample_df, outcome_columns=["happiness"])
        old_target = Sample.from_frame(target_df, outcome_columns=["happiness"])
        old_adjusted = old_sample.set_target(old_target).adjust(method="ipw")

        bf = BalanceFrame.from_sample(old_adjusted)
        roundtrip = bf.to_sample()

        self.assertTrue(roundtrip.is_adjusted())
        self.assertTrue(roundtrip.has_target())
        for o_val, r_val in zip(
            _assert_type(old_adjusted.weight_series).tolist(),
            _assert_type(roundtrip.weight_series).tolist(),
        ):
            self.assertAlmostEqual(o_val, r_val, places=5)
        self.assertEqual(
            sorted(old_adjusted._covar_columns_names()),
            sorted(roundtrip._covar_columns_names()),
        )


class TestBalanceFrameSetWeights(BalanceTestCase):
    """Verify set_weights preserves is_adjusted when unadjusted."""

    def test_set_weights_unadjusted_preserves_is_adjusted(self) -> None:
        """set_weights on an unadjusted BalanceFrame should keep is_adjusted False."""
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["1", "2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]}),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        tgt_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["3", "4"], "x": [1.5, 2.5], "weight": [1.0, 1.0]}),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        bf = BalanceFrame(sample=resp_sf, sf_target=tgt_sf)
        self.assertFalse(bf.is_adjusted)
        bf.set_weights(2.0)
        self.assertFalse(bf.is_adjusted)
        # Verify the weights actually changed
        self.assertEqual(_assert_type(bf.weight_series).tolist(), [2.0, 2.0])

    def test_set_weights_adjusted_stays_adjusted(self) -> None:
        """set_weights on an adjusted BalanceFrame should keep is_adjusted True."""
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["1", "2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]}),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        tgt_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["3", "4"], "x": [1.5, 2.5], "weight": [1.0, 1.0]}),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        bf = BalanceFrame(sample=resp_sf, sf_target=tgt_sf)
        adjusted = bf.adjust(method="null")
        self.assertTrue(adjusted.is_adjusted)
        adjusted.set_weights(3.0)
        self.assertTrue(adjusted.is_adjusted)
        self.assertEqual(_assert_type(adjusted.weight_series).tolist(), [3.0, 3.0])


class TestBalanceFrameDfSetterRejectsNone(BalanceTestCase):
    """Verify _df setter raises on None instead of silently ignoring."""

    def test_set_df_to_none_raises(self) -> None:
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["1", "2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]}),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        bf = BalanceFrame(sample=resp_sf)
        with self.assertRaises(ValueError):
            bf._df = None  # pyre-ignore[8]


class TestBalanceFrameRIndicator(BalanceTestCase):
    """Verify r_indicator() uses BalanceFrame's links, not SampleFrame._links."""

    def test_r_indicator_uses_balance_frame_links(self) -> None:
        """r_indicator on bf.weights() should resolve the target via links_override."""
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": ["1", "2"],
                    "x": [1.0, 2.0],
                    "weight": [2.0, 4.0],
                }
            ),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        tgt_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": ["10", "11", "12"],
                    "x": [1.0, 2.0, 3.0],
                    "weight": [1.0, 1.0, 1.0],
                }
            ),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        bf = BalanceFrame(sample=resp_sf, sf_target=tgt_sf)
        # This should work — the target is set via BalanceFrame's links.
        # Before the fix, this would raise ValueError because r_indicator
        # was accessing resp_sf._links directly (which has no target).
        result = bf.weights().r_indicator()
        self.assertIsInstance(result, np.floating)
        self.assertGreater(result, 0.0)
        self.assertLessEqual(result, 1.0)


class TestBalanceFrameDiagnosticsNullMethod(BalanceTestCase):
    """Verify diagnostics() works with methods that produce no model_dict."""

    def test_diagnostics_with_null_method(self) -> None:
        """adjust(method='null') followed by diagnostics() should not crash."""
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["1", "2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]}),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        tgt_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["3", "4"], "x": [1.5, 2.5], "weight": [1.0, 1.0]}),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        bf = BalanceFrame(sample=resp_sf, sf_target=tgt_sf)
        adjusted = bf.adjust(method="null")
        # This previously crashed with _assert_type on None model_dict
        diag = adjusted.diagnostics()
        self.assertIsInstance(diag, pd.DataFrame)
        # Should have an adjustment_method row
        method_rows = diag[diag["metric"] == "adjustment_method"]
        self.assertGreater(len(method_rows), 0)
        self.assertEqual(method_rows["var"].iloc[0], "null_adjustment")


class TestBalanceFrameSetTargetValidation(BalanceTestCase):
    """Verify set_target propagates validation errors immediately."""

    def test_set_target_no_shared_covariates_raises(self) -> None:
        """set_target with no shared covariates should raise ValueError."""
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["1", "2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]}),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        # Target has column "y" — no overlap with "x"
        tgt_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["3", "4"], "y": [1.5, 2.5], "weight": [1.0, 1.0]}),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        bf = BalanceFrame(sample=resp_sf)
        # Wrapping target in BalanceFrame to test the BalanceFrame path
        tgt_bf = BalanceFrame(sample=tgt_sf)
        with self.assertRaises(ValueError):
            bf.set_target(tgt_bf)


class TestBalanceFrameEdgeCases(BalanceTestCase):
    """Edge case tests for BalanceFrame: pickle, null adjust, max_de, 0-row."""

    def _make_bf(self) -> BalanceFrame:
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["1", "2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]}),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        tgt_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["3", "4"], "x": [1.5, 2.5], "weight": [1.0, 1.0]}),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        return BalanceFrame(sample=resp_sf, sf_target=tgt_sf)

    def test_pickle_round_trip_unadjusted(self) -> None:
        """Pickle round-trip for unadjusted BalanceFrame."""
        import pickle

        bf = self._make_bf()
        data = pickle.dumps(bf)
        bf2 = pickle.loads(data)
        self.assertFalse(bf2.is_adjusted)
        self.assertTrue(bf2.has_target)
        pd.testing.assert_frame_equal(bf.df, bf2.df)

    def test_pickle_round_trip_adjusted(self) -> None:
        """Pickle round-trip for adjusted BalanceFrame."""
        import pickle

        bf = self._make_bf()
        adjusted = bf.adjust(method="null")
        data = pickle.dumps(adjusted)
        adj2 = pickle.loads(data)
        self.assertTrue(adj2.is_adjusted)
        pd.testing.assert_frame_equal(adjusted.df, adj2.df)

    def test_deepcopy_preserves_adjustment_state(self) -> None:
        bf = self._make_bf()
        adjusted = bf.adjust(method="null")
        bf_copy = copy.deepcopy(adjusted)
        self.assertTrue(bf_copy.is_adjusted)
        self.assertTrue(bf_copy.has_target)

    def test_adjust_null_then_diagnostics(self) -> None:
        """adjust(method='null') followed by diagnostics() should work."""
        bf = self._make_bf()
        adjusted = bf.adjust(method="null")
        diag = adjusted.diagnostics()
        self.assertIsInstance(diag, pd.DataFrame)

    def test_zero_row_balanceframe(self) -> None:
        """BalanceFrame with 0-row data should not crash on construction."""
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": pd.Series([], dtype="str"),
                    "x": pd.Series([], dtype="float64"),
                    "weight": pd.Series([], dtype="float64"),
                }
            ),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        tgt_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": pd.Series([], dtype="str"),
                    "x": pd.Series([], dtype="float64"),
                    "weight": pd.Series([], dtype="float64"),
                }
            ),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        bf = BalanceFrame(sample=resp_sf, sf_target=tgt_sf)
        self.assertEqual(bf.df.shape[0], 0)
