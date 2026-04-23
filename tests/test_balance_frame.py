# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import copy
import io
import logging
import pickle
import unittest
from typing import Any, Literal
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from balance.balance_frame import BalanceFrame
from balance.datasets import load_data
from balance.sample_class import Sample
from balance.sample_frame import SampleFrame
from balance.testutil import _SKLEARN_1_4_AVAILABLE, BalanceTestCase
from balance.util import _assert_type
from balance.weighting_methods.ipw import ipw as ipw_func
from scipy.special import expit


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
        bf = BalanceFrame(sample=self.resp_sf, target=self.tgt_sf)
        self.assertIsInstance(bf, BalanceFrame)
        self.assertIs(bf.responders, self.resp_sf)
        self.assertIs(bf.target, self.tgt_sf)

    def test_is_adjusted_false_on_creation(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf, target=self.tgt_sf)
        self.assertFalse(bf.is_adjusted)

    def test_unadjusted_none_on_creation(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf, target=self.tgt_sf)
        self.assertIsNone(bf.unadjusted)

    def test_model_none_on_creation(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf, target=self.tgt_sf)
        self.assertIsNone(bf.model)

    def test_missing_responders_raises(self) -> None:
        with self.assertRaises(TypeError):
            BalanceFrame(target=self.tgt_sf)

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
                target=self.tgt_sf,
            )

    def test_non_sampleframe_target_raises(self) -> None:
        with self.assertRaises(TypeError):
            BalanceFrame._create(
                sample=self.resp_sf,
                target=pd.DataFrame(),  # pyre-ignore[6]
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
            BalanceFrame(sample=resp_sf, target=tgt_sf)
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
            bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
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
            bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
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
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
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
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
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
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
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
        bf = BalanceFrame._create(sample=resp_sf, target=tgt_sf)
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
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
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
        self.bf = BalanceFrame(sample=self.resp_sf, target=self.tgt_sf)

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
        bf = BalanceFrame(sample=self.resp_sf, target=self.tgt_sf)
        self.assertTrue(bf.has_target())

    def test_set_target_inplace(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf)
        result = bf.set_target(self.tgt_sf)
        self.assertIs(result, bf)
        self.assertTrue(bf.has_target())
        self.assertIs(bf.target, self.tgt_sf)

    def test_set_target_copy(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf)
        new_bf = bf.set_target(self.tgt_sf, inplace=False)
        self.assertIsNot(new_bf, bf)
        self.assertTrue(new_bf.has_target())
        self.assertFalse(bf.has_target())

    def test_set_target_replaces_existing(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf, target=self.tgt_sf)
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
        bf = BalanceFrame(sample=self.resp_sf, target=self.tgt_sf)
        adjusted = bf.adjust(method="null")
        self.assertTrue(adjusted.is_adjusted)
        self.assertIn("unadjusted", adjusted._links)
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
        # For SampleFrame targets, set_target defaults to in-place behavior.
        retargeted = adjusted.set_target(tgt2_sf)
        self.assertIs(retargeted, adjusted)
        self.assertFalse(retargeted.is_adjusted)
        self.assertIsNone(retargeted.model)
        self.assertNotIn("unadjusted", retargeted._links)

    def test_set_target_on_adjusted_logs_reset_warning(self) -> None:
        adjusted = BalanceFrame(sample=self.resp_sf, target=self.tgt_sf).adjust(
            method="null"
        )
        tgt2_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": ["7", "8"],
                    "x1": [50.0, 60.0],
                    "x2": [5.0, 6.0],
                    "weight": [1.0, 1.0],
                }
            )
        )
        with self.assertLogs("balance", level="WARNING") as cm:
            adjusted.set_target(tgt2_sf)
        self.assertTrue(
            any("discards current adjustment results" in line for line in cm.output)
        )

    def test_set_target_on_unadjusted_does_not_log_reset_warning(self) -> None:
        bf = BalanceFrame(sample=self.resp_sf, target=self.tgt_sf)
        tgt2_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": ["7", "8"],
                    "x1": [50.0, 60.0],
                    "x2": [5.0, 6.0],
                    "weight": [1.0, 1.0],
                }
            )
        )
        with patch("balance.balance_frame.logger.warning") as mock_warning:
            bf.set_target(tgt2_sf)
        self.assertFalse(mock_warning.called)

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
        self.bf = BalanceFrame(sample=self.resp_sf, target=self.tgt_sf)

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
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
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
        new_bf = BalanceFrame(sample=new_resp, target=new_tgt)
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
        new_adjusted = BalanceFrame(sample=new_resp, target=new_tgt).adjust(
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
        self.bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
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
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
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
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
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
        self.bf = BalanceFrame(sample=self.resp_sf, target=self.tgt_sf)

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
        bf = BalanceFrame(sample=resp, target=tgt)
        self.assertIsNone(bf.responders.df_ignored)

    def test_id_series_returns_series(self) -> None:
        result = self.bf.id_series
        self.assertIsInstance(result, pd.Series)
        assert result is not None
        self.assertEqual(result.tolist(), ["1", "2", "3"])

    def test_id_series_delegates_to_responders(self) -> None:
        pd.testing.assert_series_equal(self.bf.id_series, self.bf.responders.id_series)


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
        self.bf = BalanceFrame(sample=self.resp_sf, target=self.tgt_sf)
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
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
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
        self.bf = BalanceFrame(sample=self.resp_sf, target=self.tgt_sf)

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
        new_bf = BalanceFrame(sample=new_resp, target=new_tgt)
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
        bf = BalanceFrame(sample=resp, target=tgt)

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
        bf = BalanceFrame(sample=resp, target=tgt)
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
        bf = BalanceFrame(sample=resp, target=tgt)

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
        new_adjusted = BalanceFrame(sample=resp, target=tgt).adjust(method="ipw")

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
        new_adjusted = BalanceFrame(sample=resp, target=tgt).adjust(method="ipw")

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
            BalanceFrame.from_sample("not a sample")

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
        self.bf = BalanceFrame(sample=self.resp_sf, target=self.tgt_sf)

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
        self.assertEqual(s.id_series.tolist(), ["1", "2", "3"])

    def test_to_sample_target_data(self) -> None:
        s = self.bf.to_sample()
        target = s._links["target"]
        self.assertEqual(target.id_series.tolist(), ["4", "5", "6"])
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
        bf = BalanceFrame(sample=resp_sf, target=self.tgt_sf)
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
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
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
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
        adjusted = bf.adjust(method="null")
        self.assertTrue(adjusted.is_adjusted)
        adjusted.set_weights(3.0)
        self.assertTrue(adjusted.is_adjusted)
        self.assertEqual(_assert_type(adjusted.weight_series).tolist(), [3.0, 3.0])

    def test_set_weights_none_sets_one(self) -> None:
        """set_weights(None) should set all weights to 1.0 (delegates to SampleFrame)."""
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["1", "2"], "x": [1.0, 2.0], "weight": [3.0, 4.0]}),
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
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
        bf.set_weights(None)
        self.assertEqual(_assert_type(bf.weight_series).tolist(), [1.0, 1.0])

    def test_set_weights_use_index(self) -> None:
        """set_weights with use_index=True should delegate correctly."""
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
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
        weights = pd.Series([5.0, 6.0], index=bf.df.index)
        bf.set_weights(weights, use_index=True)
        self.assertEqual(_assert_type(bf.weight_series).tolist(), [5.0, 6.0])


class TestBalanceFrameSetAsPreAdjust(BalanceTestCase):
    class _NoDeepcopy:
        def __deepcopy__(self, memo: dict[int, object]) -> object:
            raise RuntimeError("should not be deep-copied")

    def _make_adjusted(self) -> BalanceFrame:
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["1", "2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]})
        )
        tgt_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["3", "4"], "x": [1.5, 2.5], "weight": [1.0, 1.0]})
        )
        return BalanceFrame(sample=resp_sf, target=tgt_sf).adjust(method="null")

    def test_set_as_pre_adjust_returns_copy_by_default(self) -> None:
        adjusted = self._make_adjusted()
        reset = adjusted.set_as_pre_adjust()
        self.assertIsNot(reset, adjusted)
        self.assertFalse(reset.is_adjusted)
        self.assertNotIn("unadjusted", reset._links)
        self.assertIsNone(reset.model)
        # Original object remains adjusted and keeps its model.
        self.assertTrue(adjusted.is_adjusted)
        self.assertIsNotNone(adjusted.model)

    def test_set_as_pre_adjust_inplace(self) -> None:
        adjusted = self._make_adjusted()
        result = adjusted.set_as_pre_adjust(inplace=True)
        self.assertIs(result, adjusted)
        self.assertFalse(adjusted.is_adjusted)
        self.assertNotIn("unadjusted", adjusted._links)
        self.assertIsNone(adjusted.model)

    def test_set_as_pre_adjust_preserves_target_link(self) -> None:
        adjusted = self._make_adjusted()
        reset = adjusted.set_as_pre_adjust()
        self.assertTrue(reset.has_target())

    def test_set_as_pre_adjust_unadjusted_noop_semantics(self) -> None:
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["1", "2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]})
        )
        tgt_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["3", "4"], "x": [1.5, 2.5], "weight": [1.0, 1.0]})
        )
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
        reset = bf.set_as_pre_adjust()
        self.assertFalse(reset.is_adjusted)
        self.assertIsNone(reset.model)

    def test_set_as_pre_adjust_copy_does_not_deepcopy_unadjusted_link(self) -> None:
        adjusted = self._make_adjusted()
        adjusted._links["unadjusted"] = self._NoDeepcopy()
        reset = adjusted.set_as_pre_adjust()
        self.assertFalse(reset.is_adjusted)


class TestBalanceFrameTrim(BalanceTestCase):
    """Verify BalanceFrame.trim() delegates to SampleFrame and returns new BF."""

    def _make_bf(self) -> BalanceFrame:
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": ["1", "2", "3"],
                    "x": [1.0, 2.0, 3.0],
                    "weight": [1.0, 1.0, 100.0],
                }
            ),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        tgt_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {"id": ["4", "5", "6"], "x": [1.5, 2.5, 3.5], "weight": [1.0, 1.0, 1.0]}
            ),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        return BalanceFrame(sample=resp_sf, target=tgt_sf)

    def test_trim_returns_new_bf(self) -> None:
        bf = self._make_bf()
        trimmed = bf.trim(ratio=2)
        self.assertIsNot(trimmed, bf)
        self.assertTrue(_assert_type(trimmed.weight_series).max() < 100.0)
        # Original unchanged
        self.assertEqual(_assert_type(bf.weight_series).tolist(), [1.0, 1.0, 100.0])
        # Weight history present
        self.assertIn("weight_trimmed_1", trimmed._sf_sample._df.columns)

    def test_trim_preserves_target(self) -> None:
        bf = self._make_bf()
        trimmed = bf.trim(ratio=2)
        self.assertTrue(trimmed.has_target())
        self.assertIs(trimmed._sf_target, bf._sf_target)

    def test_trim_preserves_pre_adjust(self) -> None:
        bf = self._make_bf()
        trimmed = bf.trim(ratio=2)
        self.assertIs(trimmed._sf_sample_pre_adjust, bf._sf_sample_pre_adjust)

    def test_trim_inplace(self) -> None:
        bf = self._make_bf()
        result = bf.trim(ratio=2, inplace=True)
        self.assertIs(result, bf)
        self.assertTrue(_assert_type(bf.weight_series).max() < 100.0)

    def test_trim_after_adjust_global_counter(self) -> None:
        """trim after adjust uses global action counter."""
        bf = self._make_bf()
        adjusted = bf.adjust(method="null")
        trimmed = adjusted.trim(ratio=2)
        # adjust created weight_adjusted_1, so trim should be weight_trimmed_2
        self.assertIn("weight_adjusted_1", trimmed._sf_sample._df.columns)
        self.assertIn("weight_trimmed_2", trimmed._sf_sample._df.columns)


class TestTrimInPlaceFalsePreservesFitArtifacts(BalanceTestCase):
    """Verify trim(inplace=False) preserves _adjustment_model."""

    def test_trim_inplace_false_preserves_fit_artifacts(self) -> None:
        """fit -> trim(inplace=False) should preserve the model so predict_weights() works."""
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": ["1", "2", "3"],
                    "x": [1.0, 2.0, 3.0],
                    "weight": [1.0, 1.0, 100.0],
                }
            ),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        tgt_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {"id": ["4", "5", "6"], "x": [1.5, 2.5, 3.5], "weight": [1.0, 1.0, 1.0]}
            ),
            id_column="id",
            weight_column="weight",
            standardize_types=False,
        )
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
        adjusted = bf.adjust(method="null")

        # Ensure model is present after adjust
        self.assertIsNotNone(adjusted._adjustment_model)

        # Default trim (returns new object) should preserve the model
        trimmed = adjusted.trim(ratio=2)
        self.assertIsNotNone(trimmed._adjustment_model)
        self.assertEqual(trimmed._adjustment_model, adjusted._adjustment_model)


class TestSetWeightsDocstringWarnsAboutFitInconsistency(BalanceTestCase):
    """Verify set_weights works and its docstring warns about fit inconsistency."""

    def test_set_weights_docstring_warns_about_fit_inconsistency(self) -> None:
        """set_weights should work and its docstring should mention the inconsistency."""
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
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
        bf.set_weights(2.0)
        self.assertEqual(_assert_type(bf.weight_series).tolist(), [2.0, 2.0])
        # Verify the docstring mentions the warning about re-fitting
        self.assertIn("re-fit", BalanceFrame.set_weights.__doc__ or "")


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
            bf._df = None


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
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
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
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
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
        return BalanceFrame(sample=resp_sf, target=tgt_sf)

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
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
        self.assertEqual(bf.df.shape[0], 0)


class TestBalanceFrameSklearnLikeApi(BalanceTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": [f"s{i}" for i in range(8)],
                    "x": [0.1, 0.3, 0.7, 1.1, 1.4, 1.7, 2.0, 2.2],
                    "z": ["a", "a", "b", "b", "a", "b", "a", "b"],
                    "weight": [1.0] * 8,
                }
            )
        )
        self.target = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": [f"t{i}" for i in range(8)],
                    "x": [0.2, 0.5, 0.8, 1.0, 1.6, 1.9, 2.1, 2.4],
                    "z": ["a", "b", "a", "b", "a", "b", "a", "b"],
                    "weight": [1.0] * 8,
                }
            )
        )
        self.bf = BalanceFrame.from_sample(self.sample.set_target(self.target))

    def test_fit_is_alias_for_adjust(self) -> None:
        adjusted = self.bf.fit(method="ipw")
        self.assertTrue(adjusted.is_adjusted)
        self.assertIsNotNone(adjusted.model)
        self.assertEqual(_assert_type(adjusted.model)["method"], "ipw")

    def test_fit_rejects_positional_forwarding(self) -> None:
        with self.assertRaises(TypeError):
            self.bf.fit("ipw")
        with self.assertRaises(TypeError):
            self.bf.fit(None, "ipw", "unexpected positional")

    def test_fit_callable_ipw_enables_store_fit_matrices(self) -> None:
        adjusted = self.bf.fit(method=ipw_func)
        x_sample = adjusted.design_matrix(on="sample")
        self.assertEqual(x_sample.shape[0], len(self.sample.df))

    def test_fit_inplace_true_mutates_self(self) -> None:
        bf = BalanceFrame.from_sample(self.sample.set_target(self.target))
        result = bf.fit(method="ipw")
        self.assertIs(result, bf)
        self.assertTrue(bf.is_adjusted)
        self.assertIsNotNone(bf.model)

    def test_fit_inplace_false_returns_new_object(self) -> None:
        bf = BalanceFrame.from_sample(self.sample.set_target(self.target))
        adjusted = bf.fit(method="ipw", inplace=False)
        self.assertIsNot(adjusted, bf)
        self.assertTrue(adjusted.is_adjusted)
        self.assertFalse(bf.is_adjusted)

    def test_fit_sampleframe_target_with_inplace_false(self) -> None:
        bf_no_target = BalanceFrame(sample=SampleFrame.from_sample(self.sample))
        target_sf = SampleFrame.from_sample(self.target)
        adjusted = bf_no_target.fit(target=target_sf, method="null", inplace=False)
        self.assertFalse(bf_no_target.has_target)
        self.assertTrue(adjusted.has_target)
        self.assertTrue(adjusted.is_adjusted)

    def test_fit_custom_callable_named_ipw_does_not_inject_ipw_kwargs(self) -> None:
        def ipw(
            sample_df: pd.DataFrame,
            sample_weights: pd.Series,
            target_df: pd.DataFrame,
            target_weights: pd.Series,
        ) -> dict[str, Any]:
            return {"weight": sample_weights, "model": {"method": "custom_ipw"}}

        adjusted = self.bf.fit(method=ipw)
        self.assertTrue(adjusted.is_adjusted)
        self.assertEqual(_assert_type(adjusted.model)["method"], "custom_ipw")

    def test_design_matrix_predict_proba_and_predict_weights(self) -> None:
        adjusted = self.bf.fit(method="ipw")

        x_sample, x_target = adjusted.design_matrix(on="both")
        self.assertEqual(x_sample.shape[1], x_target.shape[1])
        self.assertEqual(x_sample.shape[0], len(self.sample.df))
        self.assertEqual(x_target.shape[0], len(self.target.df))

        p_target = adjusted.predict_proba(on="target", output="probability")
        self.assertEqual(p_target.shape[0], len(self.target.df))
        self.assertTrue(np.all(np.isfinite(p_target.to_numpy())))

        w_sample = adjusted.predict_weights()
        self.assertEqual(w_sample.shape[0], len(self.sample.df))
        self.assertTrue(np.all(w_sample.to_numpy() > 0))
        self.assertEqual(w_sample.name, _assert_type(adjusted.weight_series).name)
        np.testing.assert_allclose(
            w_sample.to_numpy(),
            _assert_type(adjusted.weight_series).to_numpy(),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_predict_proba_on_both_and_link_output(self) -> None:
        adjusted = self.bf.fit(method="ipw")
        link_sample, link_target = adjusted.predict_proba(on="both", output="link")
        self.assertEqual(link_sample.shape[0], len(self.sample.df))
        self.assertEqual(link_target.shape[0], len(self.target.df))
        self.assertTrue(np.all(np.isfinite(link_sample.to_numpy())))
        self.assertTrue(np.all(np.isfinite(link_target.to_numpy())))

    def test_design_matrix_predict_proba_invalid_on_raises(self) -> None:
        adjusted = self.bf.fit(method="ipw")
        with self.assertRaises(ValueError):
            adjusted.design_matrix(on="bad")
        with self.assertRaises(ValueError):
            adjusted.predict_proba(on="bad")
        with self.assertRaises(ValueError):
            adjusted.predict_proba(output="bad")

    def test_predict_proba_design_matrix_raise_for_non_ipw(self) -> None:
        adjusted = self.bf.fit(method="null")
        with self.assertRaises(ValueError):
            adjusted.design_matrix()
        with self.assertRaises(ValueError):
            adjusted.predict_proba()
        with self.assertRaises(ValueError):
            adjusted.predict_weights()

    def test_predict_proba_design_matrix_raise_actionable_error_without_model(
        self,
    ) -> None:
        no_model = BalanceFrame.from_sample(self.sample.set_target(self.target))
        with self.assertRaisesRegex(ValueError, "adjusted model"):
            no_model.design_matrix()
        with self.assertRaisesRegex(ValueError, "adjusted model"):
            no_model.predict_proba()
        with self.assertRaisesRegex(ValueError, "adjusted model"):
            no_model.predict_weights()

    def test_set_fitted_model_requires_adjusted_input(self) -> None:
        no_model = BalanceFrame(sample=SampleFrame.from_sample(self.sample))
        with self.assertRaisesRegex(
            ValueError, "adjusted BalanceFrame with a stored model"
        ):
            self.bf.set_fitted_model(no_model)

    def test_set_fitted_model_requires_matching_covariate_columns(self) -> None:
        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        holdout_sample_df = self.sample.df.iloc[:5].copy().rename(columns={"x": "x2"})
        holdout_target_df = self.target.df.iloc[:5].copy().rename(columns={"x": "x2"})
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(holdout_sample_df),
            target=SampleFrame.from_frame(holdout_target_df),
        )
        fitted_train = train_bf.fit(method="ipw")
        with self.assertRaisesRegex(
            ValueError, "matching sample covariate column names"
        ):
            holdout_bf.set_fitted_model(fitted_train)

    def test_predict_proba_and_predict_weights_actionable_error_without_fit_metadata(
        self,
    ) -> None:
        adjusted = self.bf.adjust(method="ipw")
        model = _assert_type(adjusted.model)
        self.assertNotIn("training_sample_weights", model)
        self.assertNotIn("training_target_weights", model)
        with self.assertRaisesRegex(ValueError, "store_fit_metadata=True"):
            adjusted.predict_proba(on="sample", output="link")
        with self.assertRaisesRegex(ValueError, "store_fit_metadata=True"):
            adjusted.predict_weights()

    def test_design_matrix_error_message_is_actionable_without_fit_matrices(
        self,
    ) -> None:
        adjusted = self.bf.adjust(method="ipw")
        with self.assertRaisesRegex(ValueError, "store_fit_matrices=True"):
            adjusted.design_matrix(on="sample")

    def test_fit_with_store_fit_flags_disabled_keeps_design_matrix_actionable_error(
        self,
    ) -> None:
        adjusted = self.bf.fit(
            method="ipw",
            store_fit_matrices=False,
            store_fit_metadata=False,
        )
        with self.assertRaisesRegex(ValueError, "store_fit_matrices=True"):
            adjusted.design_matrix(on="sample")

    def test_design_matrix_predict_proba_and_predict_weights_handle_duplicate_indices(
        self,
    ) -> None:
        adjusted = self.bf.fit(method="ipw")
        model = _assert_type(adjusted.model)
        model["sample_index"] = pd.Index(["dup"] * len(adjusted._sf_sample.df))
        model["target_index"] = pd.Index(
            ["dup_t"] * len(_assert_type(adjusted._sf_target).df)
        )

        transformed = adjusted.design_matrix(on="sample")
        self.assertEqual(list(transformed.index), list(adjusted._sf_sample.df.index))

        predicted = adjusted.predict_proba(on="sample", output="probability")
        self.assertEqual(list(predicted.index), list(adjusted._sf_sample.df.index))

        predicted_weights = adjusted.predict_weights()
        self.assertEqual(
            list(predicted_weights.index), list(adjusted._sf_sample.df.index)
        )

    def test_design_matrix_handles_dataframe_fit_matrix_with_duplicate_indices(
        self,
    ) -> None:
        from sklearn.naive_bayes import GaussianNB

        sample_df = self.sample.df.iloc[:6].copy()
        target_df = self.target.df.iloc[:6].copy()
        sample_df.index = pd.Index(["a", "a", "b", "b", "c", "c"])
        target_df.index = pd.Index(["ta", "ta", "tb", "tb", "tc", "tc"])

        bf = BalanceFrame(
            sample=SampleFrame.from_frame(sample_df),
            target=SampleFrame.from_frame(target_df),
        )
        adjusted = bf.fit(
            method="ipw",
            model=GaussianNB(),
            variables=["x"],
            use_model_matrix=False,
            transformations=None,
        )

        transformed = adjusted.design_matrix(on="sample")
        self.assertEqual(
            list(transformed.index),
            list(adjusted._sf_sample.df.index),
        )

    def test_fit_on_subset_and_apply_to_holdout(self) -> None:
        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[5:].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[5:].copy()),
        )

        fitted_train = train_bf.fit(method="ipw")
        scored_holdout = holdout_bf.set_fitted_model(fitted_train, inplace=False)
        self.assertIsNotNone(scored_holdout.model)
        holdout_model = scored_holdout._require_ipw_model()
        self.assertIsNot(holdout_model, fitted_train.model)
        self.assertIs(
            holdout_model["fit"],
            _assert_type(fitted_train.model)["fit"],
        )

        transformed = scored_holdout.design_matrix(on="sample")
        propensity = scored_holdout.predict_proba(on="sample", output="probability")
        link = scored_holdout.predict_proba(on="sample", output="link")

        self.assertEqual(transformed.shape[0], len(holdout_bf._sf_sample.df))
        self.assertEqual(propensity.shape[0], len(holdout_bf._sf_sample.df))
        np.testing.assert_allclose(propensity.to_numpy(), expit(link.to_numpy()))

        # set_fitted_model produces a fully adjusted object with computed weights
        adjusted_weights = _assert_type(scored_holdout.weight_series).to_numpy()
        self.assertEqual(len(adjusted_weights), len(holdout_bf._sf_sample.df))
        self.assertTrue(np.all(np.isfinite(adjusted_weights)))
        self.assertTrue(np.all(adjusted_weights > 0))

        from balance.weighting_methods.ipw import weights_from_link

        expected_weights = weights_from_link(
            link=link.to_numpy(),
            balance_classes=bool(holdout_model.get("balance_classes")),
            sample_weights=holdout_bf._sf_sample.df_weights.iloc[:, 0],
            target_weights=_assert_type(holdout_bf._sf_target).df_weights.iloc[:, 0],
            weight_trimming_mean_ratio=holdout_model.get("weight_trimming_mean_ratio"),
            weight_trimming_percentile=holdout_model.get("weight_trimming_percentile"),
        )
        np.testing.assert_allclose(adjusted_weights, expected_weights.to_numpy())

    def test_set_fitted_model_produces_adjusted_object(self) -> None:
        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        fitted_train = train_bf.fit(method="ipw")
        scored_holdout = holdout_bf.set_fitted_model(fitted_train, inplace=False)

        self.assertTrue(scored_holdout.is_adjusted)
        self.assertIsNotNone(scored_holdout.model)
        model = scored_holdout.model
        assert model is not None
        self.assertEqual(model["method"], "ipw")

    def test_fit_rejects_na_action_drop_with_fit_artifact_storage(self) -> None:
        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        with self.assertRaisesRegex(
            ValueError,
            "fit\\(method='ipw', na_action='drop'\\) is incompatible with stored fit artifacts",
        ):
            train_bf.fit(method="ipw", na_action="drop")

    def test_fit_allows_na_action_drop_when_fit_artifact_storage_disabled(self) -> None:
        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        adjusted = train_bf.fit(
            method="ipw",
            na_action="drop",
            store_fit_matrices=False,
            store_fit_metadata=False,
        )
        self.assertTrue(adjusted.is_adjusted)

    def test_same_length_holdout_recomputes_instead_of_silent_reindex(self) -> None:
        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        holdout_sample_df = self.sample.df.iloc[:5].copy()
        holdout_target_df = self.target.df.iloc[:5].copy()
        holdout_sample_df.index = pd.Index([f"h{i}" for i in range(5)])
        holdout_target_df.index = pd.Index([f"ht{i}" for i in range(5)])
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(holdout_sample_df),
            target=SampleFrame.from_frame(holdout_target_df),
        )

        fitted_train = train_bf.fit(method="ipw")
        scored_holdout = holdout_bf.set_fitted_model(fitted_train, inplace=False)

        transformed = scored_holdout.design_matrix(on="sample")
        propensity = scored_holdout.predict_proba(on="sample", output="probability")
        target_propensity = scored_holdout.predict_proba(
            on="target", output="probability"
        )

        model = scored_holdout._require_ipw_model()
        fit_model = _assert_type(model.get("fit"))
        class_index = scored_holdout._ipw_class_index(fit_model)
        sample_matrix, target_matrix = scored_holdout._compute_ipw_matrices(model)
        expected_sample = np.asarray(
            fit_model.predict_proba(sample_matrix)[:, class_index]
        )
        expected_target = np.asarray(
            fit_model.predict_proba(target_matrix)[:, class_index]
        )

        self.assertEqual(list(transformed.index), list(holdout_bf._sf_sample.df.index))
        self.assertEqual(list(propensity.index), list(holdout_bf._sf_sample.df.index))
        np.testing.assert_allclose(propensity.to_numpy(), expected_sample)
        np.testing.assert_allclose(target_propensity.to_numpy(), expected_target)
        self.assertFalse(transformed.isna().all(axis=None))

    def test_holdout_scoring_caches_recomputed_fit_artifacts(self) -> None:
        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        holdout_sample_df = self.sample.df.iloc[:5].copy()
        holdout_target_df = self.target.df.iloc[:5].copy()
        holdout_sample_df.index = pd.Index([f"h{i}" for i in range(5)])
        holdout_target_df.index = pd.Index([f"ht{i}" for i in range(5)])
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(holdout_sample_df),
            target=SampleFrame.from_frame(holdout_target_df),
        )

        fitted_train = train_bf.fit(method="ipw")
        scored_holdout = holdout_bf.set_fitted_model(fitted_train, inplace=False)

        with patch.object(
            scored_holdout,
            "_compute_ipw_matrices",
            wraps=scored_holdout._compute_ipw_matrices,
        ) as compute_mock:
            scored_holdout.design_matrix(on="sample")
            scored_holdout.design_matrix(on="sample")
            self.assertEqual(compute_mock.call_count, 1)

        scored_holdout_for_predict_proba = holdout_bf.set_fitted_model(
            fitted_train, inplace=False
        )
        with patch.object(
            scored_holdout_for_predict_proba,
            "_compute_ipw_matrices",
            wraps=scored_holdout_for_predict_proba._compute_ipw_matrices,
        ) as compute_mock:
            scored_holdout_for_predict_proba.predict_proba(
                on="sample", output="probability"
            )
            scored_holdout_for_predict_proba.predict_proba(
                on="sample", output="probability"
            )
            self.assertEqual(compute_mock.call_count, 1)

    def test_predict_proba_reordered_indices_uses_alignment_without_recompute(
        self,
    ) -> None:
        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        reordered_sample_df = self.sample.df.iloc[:5].copy().iloc[::-1]
        reordered_target_df = self.target.df.iloc[:5].copy().iloc[::-1]
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(reordered_sample_df),
            target=SampleFrame.from_frame(reordered_target_df),
        )
        fitted_train = train_bf.fit(method="ipw")
        scored_holdout = holdout_bf.set_fitted_model(fitted_train, inplace=False)

        with patch.object(
            scored_holdout,
            "_compute_ipw_matrices",
            wraps=scored_holdout._compute_ipw_matrices,
        ) as compute_mock:
            scored_holdout.predict_proba(on="sample", output="probability")
            scored_holdout.predict_proba(on="target", output="probability")
            self.assertEqual(compute_mock.call_count, 0)

    def test_design_matrix_on_target_does_not_require_sample_matrix(self) -> None:
        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        holdout_sample_df = self.sample.df.iloc[:5].copy()
        holdout_target_df = self.target.df.iloc[:5].copy()
        holdout_sample_df.index = pd.Index([f"h{i}" for i in range(5)])
        holdout_target_df.index = pd.Index([f"ht{i}" for i in range(5)])
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(holdout_sample_df),
            target=SampleFrame.from_frame(holdout_target_df),
        )
        fitted_train = train_bf.fit(method="ipw")
        scored_holdout = holdout_bf.set_fitted_model(fitted_train, inplace=False)
        model = scored_holdout._require_ipw_model()
        model.pop("model_matrix_sample", None)

        transformed_target = scored_holdout.design_matrix(on="target")
        self.assertEqual(
            list(transformed_target.index),
            list(_assert_type(holdout_bf._sf_target).df.index),
        )

    def test_predict_proba_on_target_does_not_require_sample_predictions(self) -> None:
        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        holdout_sample_df = self.sample.df.iloc[:5].copy()
        holdout_target_df = self.target.df.iloc[:5].copy()
        holdout_sample_df.index = pd.Index([f"h{i}" for i in range(5)])
        holdout_target_df.index = pd.Index([f"ht{i}" for i in range(5)])
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(holdout_sample_df),
            target=SampleFrame.from_frame(holdout_target_df),
        )
        fitted_train = train_bf.fit(method="ipw")
        scored_holdout = holdout_bf.set_fitted_model(fitted_train, inplace=False)
        model = scored_holdout._require_ipw_model()
        model.pop("sample_probability", None)
        model.pop("sample_link", None)

        probability = scored_holdout.predict_proba(on="target", output="probability")
        link = scored_holdout.predict_proba(on="target", output="link")
        self.assertEqual(
            list(probability.index), list(_assert_type(holdout_bf._sf_target).df.index)
        )
        self.assertEqual(
            list(link.index), list(_assert_type(holdout_bf._sf_target).df.index)
        )

    def test_predict_weights_uses_current_design_weights_for_different_index(
        self,
    ) -> None:
        train_sample_df = self.sample.df.iloc[:5].copy()
        train_target_df = self.target.df.iloc[:5].copy()
        holdout_sample_df = self.sample.df.iloc[:5].copy()
        holdout_target_df = self.target.df.iloc[:5].copy()

        train_sample_df["weight"] = [1.0, 1.0, 1.0, 1.0, 1.0]
        train_target_df["weight"] = [2.0, 2.0, 2.0, 2.0, 2.0]
        holdout_sample_df["weight"] = [7.0, 8.0, 9.0, 10.0, 11.0]
        holdout_target_df["weight"] = [3.0, 4.0, 5.0, 6.0, 7.0]
        holdout_sample_df.index = pd.Index([f"h{i}" for i in range(5)])
        holdout_target_df.index = pd.Index([f"ht{i}" for i in range(5)])

        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(train_sample_df),
            target=SampleFrame.from_frame(train_target_df),
        )
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(holdout_sample_df),
            target=SampleFrame.from_frame(holdout_target_df),
        )
        fitted_train = train_bf.fit(method="ipw")
        scored_holdout = holdout_bf.set_fitted_model(fitted_train, inplace=False)

        # set_fitted_model produces a fully adjusted object; verify the
        # adjusted weights match what weights_from_link would produce
        # using the holdout's original design weights.
        adjusted_weights = _assert_type(scored_holdout.weight_series).to_numpy()
        link = scored_holdout.predict_proba(on="sample", output="link")

        from balance.weighting_methods.ipw import weights_from_link

        expected = weights_from_link(
            link=link.to_numpy(),
            balance_classes=bool(
                scored_holdout._require_ipw_model().get("balance_classes")
            ),
            sample_weights=holdout_bf._sf_sample.df_weights.iloc[:, 0],
            target_weights=_assert_type(holdout_bf._sf_target).df_weights.iloc[:, 0],
            weight_trimming_mean_ratio=scored_holdout._require_ipw_model().get(
                "weight_trimming_mean_ratio"
            ),
            weight_trimming_percentile=scored_holdout._require_ipw_model().get(
                "weight_trimming_percentile"
            ),
        )
        np.testing.assert_allclose(adjusted_weights, expected.to_numpy())

    @pytest.mark.requires_sklearn_1_4  # pyre-ignore[56]
    @unittest.skipUnless(_SKLEARN_1_4_AVAILABLE, "requires scikit-learn >= 1.4")
    def test_use_model_matrix_false_recompute_matches_fit_preprocessing(self) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        sample_df = pd.DataFrame(
            {
                "id": [f"s{i}" for i in range(6)],
                "x": [1.0, np.nan, 2.0, 3.0, np.nan, 4.0],
                "z": [0, 1, 0, 1, 0, 1],
                "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        target_df = pd.DataFrame(
            {
                "id": [f"t{i}" for i in range(6)],
                "x": [1.5, 2.5, np.nan, 3.5, 4.5, np.nan],
                "z": [0, 1, 0, 1, 0, 1],
                "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(sample_df.copy()),
            target=SampleFrame.from_frame(target_df.copy()),
        )
        holdout_sample_df = sample_df.copy()
        holdout_target_df = target_df.copy()
        holdout_sample_df.index = pd.Index(
            [f"h{i}" for i in range(len(holdout_sample_df))]
        )
        holdout_target_df.index = pd.Index(
            [f"ht{i}" for i in range(len(holdout_target_df))]
        )
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(holdout_sample_df),
            target=SampleFrame.from_frame(holdout_target_df),
        )

        fitted_train = train_bf.fit(
            method="ipw",
            use_model_matrix=False,
            model=HistGradientBoostingClassifier(random_state=0),
            na_action="add_indicator",
            transformations=None,
        )
        scored_holdout = holdout_bf.set_fitted_model(fitted_train, inplace=False)

        transformed = scored_holdout.design_matrix(on="sample")
        self.assertListEqual(
            list(transformed.columns),
            list(scored_holdout._require_ipw_model()["X_matrix_columns"]),
        )
        self.assertIn("_is_na_x", transformed.columns)
        self.assertEqual(
            list(transformed.index),
            list(holdout_bf._sf_sample.df.index),
        )

    def test_custom_model_dense_recompute_for_holdout(self) -> None:
        from sklearn.naive_bayes import GaussianNB

        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        holdout_sample_df = self.sample.df.iloc[:5].copy()
        holdout_target_df = self.target.df.iloc[:5].copy()
        holdout_sample_df.index = pd.Index([f"gh{i}" for i in range(5)])
        holdout_target_df.index = pd.Index([f"ght{i}" for i in range(5)])
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(holdout_sample_df),
            target=SampleFrame.from_frame(holdout_target_df),
        )

        fitted_train = train_bf.fit(method="ipw", model=GaussianNB())
        scored_holdout = holdout_bf.set_fitted_model(fitted_train, inplace=False)

        propensity = scored_holdout.predict_proba(on="sample", output="probability")
        self.assertEqual(propensity.shape[0], len(holdout_bf._sf_sample.df))

    def test_holdout_recompute_handles_missing_na_indicator_columns_in_formula(
        self,
    ) -> None:
        train_sample_df = pd.DataFrame(
            {
                "id": [f"s{i}" for i in range(6)],
                "gender": ["f", None, "m", "f", "m", "f"],
                "income": [10, 20, 30, 40, 50, 60],
                "weight": [1.0] * 6,
            }
        )
        train_target_df = pd.DataFrame(
            {
                "id": [f"t{i}" for i in range(6)],
                "gender": ["m", "m", None, "f", "m", "f"],
                "income": [15, 25, 35, 45, 55, 65],
                "weight": [1.0] * 6,
            }
        )
        holdout_sample_df = pd.DataFrame(
            {
                "id": [f"hs{i}" for i in range(4)],
                "gender": ["f", "m", "m", "f"],
                "income": [11, 22, 33, 44],
                "weight": [1.0] * 4,
            }
        )
        holdout_target_df = pd.DataFrame(
            {
                "id": [f"ht{i}" for i in range(4)],
                "gender": ["m", "f", "m", "f"],
                "income": [12, 24, 36, 48],
                "weight": [1.0] * 4,
            }
        )
        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(train_sample_df),
            target=SampleFrame.from_frame(train_target_df),
        )
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(holdout_sample_df),
            target=SampleFrame.from_frame(holdout_target_df),
        )

        fitted_train = train_bf.fit(method="ipw", na_action="add_indicator")
        scored_holdout = holdout_bf.set_fitted_model(fitted_train, inplace=False)

        transformed = scored_holdout.design_matrix(on="sample")
        propensity = scored_holdout.predict_proba(on="sample", output="probability")

        self.assertEqual(transformed.shape[0], len(holdout_sample_df))
        self.assertEqual(propensity.shape[0], len(holdout_sample_df))

    def test_predict_weights_dispatch_rejects_unsupported_method(self) -> None:
        adjusted = self.bf.fit(method="null")
        with self.assertRaisesRegex(
            ValueError, "predict_weights\\(\\) is not yet supported for method"
        ):
            adjusted.predict_weights()

    def test_predict_weights_dispatch_accepts_ipw(self) -> None:
        adjusted = self.bf.fit(method="ipw")
        w = adjusted.predict_weights()
        self.assertEqual(w.shape[0], len(self.sample.df))
        self.assertTrue(np.all(w.to_numpy() > 0))

    def test_pickle_deepcopy_roundtrip_preserves_fit_artifacts(self) -> None:
        from scipy import sparse as sp

        fitted = self.bf.fit(method="ipw")
        fitted_model = _assert_type(fitted.model)

        roundtrip = pickle.loads(pickle.dumps(fitted))
        copied = copy.deepcopy(fitted)

        for candidate in (roundtrip, copied):
            candidate_model = _assert_type(candidate.model)
            self.assertIsNotNone(candidate_model.get("fit"))
            self.assertIn("sample_index", candidate_model)
            self.assertIn("target_index", candidate_model)
            self.assertIn("model_matrix_sample", candidate_model)
            self.assertIn("model_matrix_target", candidate_model)
            self.assertIn("training_sample_weights", candidate_model)
            self.assertIn("training_target_weights", candidate_model)

            sample_index = _assert_type(candidate_model.get("sample_index"))
            target_index = _assert_type(candidate_model.get("target_index"))
            self.assertTrue(isinstance(sample_index, pd.Index))
            self.assertTrue(isinstance(target_index, pd.Index))
            self.assertEqual(len(sample_index), len(candidate._sf_sample.df))
            self.assertEqual(
                len(target_index), len(_assert_type(candidate._sf_target).df)
            )
            self.assertTrue(
                sample_index.equals(_assert_type(fitted_model.get("sample_index")))
            )
            self.assertTrue(
                target_index.equals(_assert_type(fitted_model.get("target_index")))
            )

            sample_matrix = _assert_type(candidate_model.get("model_matrix_sample"))
            target_matrix = _assert_type(candidate_model.get("model_matrix_target"))
            fitted_sample_matrix = _assert_type(fitted_model.get("model_matrix_sample"))
            fitted_target_matrix = _assert_type(fitted_model.get("model_matrix_target"))
            self.assertIsNotNone(sample_matrix)
            self.assertIsNotNone(target_matrix)
            self.assertEqual(type(sample_matrix), type(fitted_sample_matrix))
            self.assertEqual(type(target_matrix), type(fitted_target_matrix))
            self.assertEqual(sample_matrix.shape, fitted_sample_matrix.shape)
            self.assertEqual(target_matrix.shape, fitted_target_matrix.shape)
            if sp.issparse(sample_matrix):
                self.assertEqual(sample_matrix.nnz, fitted_sample_matrix.nnz)
            if sp.issparse(target_matrix):
                self.assertEqual(target_matrix.nnz, fitted_target_matrix.nnz)

            training_sample_weights = _assert_type(
                candidate_model.get("training_sample_weights")
            )
            training_target_weights = _assert_type(
                candidate_model.get("training_target_weights")
            )
            self.assertTrue(isinstance(training_sample_weights, pd.Series))
            self.assertTrue(isinstance(training_target_weights, pd.Series))
            self.assertEqual(
                training_sample_weights.shape,
                _assert_type(fitted_model.get("training_sample_weights")).shape,
            )
            self.assertEqual(
                training_target_weights.shape,
                _assert_type(fitted_model.get("training_target_weights")).shape,
            )
            pd.testing.assert_index_equal(
                training_sample_weights.index,
                _assert_type(fitted_model.get("training_sample_weights")).index,
            )
            pd.testing.assert_index_equal(
                training_target_weights.index,
                _assert_type(fitted_model.get("training_target_weights")).index,
            )
            self.assertEqual(
                list(candidate_model.get("X_matrix_columns", [])),
                list(fitted_model.get("X_matrix_columns", [])),
            )

            w = candidate.predict_weights()
            self.assertEqual(w.shape[0], len(candidate._sf_sample.df))
            self.assertTrue(np.all(w.to_numpy() > 0))

    def test_store_fit_metadata_default_logistic_regression(self) -> None:
        fitted = self.bf.fit(method="ipw")
        model = _assert_type(fitted.model)
        self.assertIn("training_sample_weights", model)
        self.assertIn("training_target_weights", model)
        self.assertTrue(isinstance(model.get("training_sample_weights"), pd.Series))
        self.assertTrue(isinstance(model.get("training_target_weights"), pd.Series))

    @pytest.mark.requires_sklearn_1_4  # pyre-ignore[56]
    def test_store_fit_matrices_use_model_matrix_false(self) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        sample_df = pd.DataFrame(
            {
                "id": [f"s{i}" for i in range(8)],
                "x": [0.1, 0.3, 0.7, 1.1, 1.4, 1.7, 2.0, 2.2],
                "weight": [1.0] * 8,
            }
        )
        target_df = pd.DataFrame(
            {
                "id": [f"t{i}" for i in range(8)],
                "x": [0.2, 0.5, 0.8, 1.0, 1.6, 1.9, 2.1, 2.4],
                "weight": [1.0] * 8,
            }
        )
        bf = BalanceFrame(
            sample=SampleFrame.from_frame(sample_df),
            target=SampleFrame.from_frame(target_df),
        )

        fitted = bf.fit(
            method="ipw",
            model=HistGradientBoostingClassifier(
                random_state=0, categorical_features="from_dtype"
            ),
            use_model_matrix=False,
            transformations=None,
            store_fit_matrices=True,
        )
        model = _assert_type(fitted.model)
        self.assertTrue(isinstance(model.get("model_matrix_sample"), pd.DataFrame))
        self.assertTrue(isinstance(model.get("model_matrix_target"), pd.DataFrame))
        self.assertListEqual(
            list(_assert_type(model.get("model_matrix_sample")).columns),
            list(_assert_type(model.get("X_matrix_columns"))),
        )

    def test_fit_edge_cases_empty_input_raises_and_near_separation_is_stable(
        self,
    ) -> None:
        empty_bf = BalanceFrame(
            sample=SampleFrame.from_frame(
                pd.DataFrame({"id": [], "x": [], "weight": []})
            ),
            target=SampleFrame.from_frame(
                pd.DataFrame({"id": [], "x": [], "weight": []})
            ),
        )
        with self.assertRaisesRegex(ValueError, "more than zero rows"):
            empty_bf.fit(method="ipw")

        # Near-separation should still yield finite, strictly-positive weights.
        sample_df = pd.DataFrame(
            {
                "id": [f"s{i}" for i in range(20)],
                "x": np.linspace(0.0, 0.2, 20),
                "weight": 1.0,
            }
        )
        target_df = pd.DataFrame(
            {
                "id": [f"t{i}" for i in range(20)],
                "x": np.linspace(0.8, 1.0, 20),
                "weight": 1.0,
            }
        )
        bf = BalanceFrame(
            sample=SampleFrame.from_frame(sample_df),
            target=SampleFrame.from_frame(target_df),
        )
        adjusted = bf.fit(method="ipw")
        weights = adjusted.predict_weights()
        self.assertTrue(np.all(np.isfinite(weights.to_numpy())))
        self.assertTrue(np.all(weights.to_numpy() > 0))

    def test_predict_weights_with_data_argument(self) -> None:
        """predict_weights(data=holdout) matches set_fitted_model workflow."""
        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        holdout_sample_df = self.sample.df.iloc[5:].copy()
        holdout_target_df = self.target.df.iloc[5:].copy()
        holdout_sample_df.index = pd.Index(
            [f"h{i}" for i in range(len(holdout_sample_df))]
        )
        holdout_target_df.index = pd.Index(
            [f"ht{i}" for i in range(len(holdout_target_df))]
        )
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(holdout_sample_df),
            target=SampleFrame.from_frame(holdout_target_df),
        )

        fitted = train_bf.fit(method="ipw")

        # data= path
        weights_via_data = fitted.predict_weights(data=holdout_bf)

        # set_fitted_model path — the adjusted weights are on the object directly
        scored = holdout_bf.set_fitted_model(fitted, inplace=False)
        weights_via_sfm = _assert_type(scored.weight_series).to_numpy()

        self.assertEqual(weights_via_data.shape[0], len(holdout_sample_df))
        np.testing.assert_allclose(
            weights_via_data.to_numpy(),
            weights_via_sfm,
            rtol=1e-6,
            atol=1e-8,
        )
        self.assertTrue(np.all(weights_via_data.to_numpy() > 0))

    def test_predict_proba_with_data_argument(self) -> None:
        """predict_proba(data=holdout) matches set_fitted_model workflow."""
        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        holdout_sample_df = self.sample.df.iloc[5:].copy()
        holdout_target_df = self.target.df.iloc[5:].copy()
        holdout_sample_df.index = pd.Index(
            [f"h{i}" for i in range(len(holdout_sample_df))]
        )
        holdout_target_df.index = pd.Index(
            [f"ht{i}" for i in range(len(holdout_target_df))]
        )
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(holdout_sample_df),
            target=SampleFrame.from_frame(holdout_target_df),
        )

        fitted = train_bf.fit(method="ipw")

        # data= path: probability
        prob_via_data = fitted.predict_proba(
            on="sample", output="probability", data=holdout_bf
        )
        # set_fitted_model path
        scored = holdout_bf.set_fitted_model(fitted, inplace=False)
        prob_via_wfm = scored.predict_proba(on="sample", output="probability")

        self.assertEqual(prob_via_data.shape[0], len(holdout_sample_df))
        np.testing.assert_allclose(
            prob_via_data.to_numpy(),
            prob_via_wfm.to_numpy(),
            rtol=1e-6,
            atol=1e-8,
        )

        # data= path: link
        link_via_data = fitted.predict_proba(
            on="sample", output="link", data=holdout_bf
        )
        link_via_wfm = scored.predict_proba(on="sample", output="link")
        np.testing.assert_allclose(
            link_via_data.to_numpy(),
            link_via_wfm.to_numpy(),
            rtol=1e-6,
            atol=1e-8,
        )

        # data= path: target side
        prob_target_via_data = fitted.predict_proba(
            on="target", output="probability", data=holdout_bf
        )
        prob_target_via_wfm = scored.predict_proba(on="target", output="probability")
        np.testing.assert_allclose(
            prob_target_via_data.to_numpy(),
            prob_target_via_wfm.to_numpy(),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_design_matrix_with_data_argument(self) -> None:
        """design_matrix(data=holdout) matches set_fitted_model workflow."""
        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        holdout_sample_df = self.sample.df.iloc[5:].copy()
        holdout_target_df = self.target.df.iloc[5:].copy()
        holdout_sample_df.index = pd.Index(
            [f"h{i}" for i in range(len(holdout_sample_df))]
        )
        holdout_target_df.index = pd.Index(
            [f"ht{i}" for i in range(len(holdout_target_df))]
        )
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(holdout_sample_df),
            target=SampleFrame.from_frame(holdout_target_df),
        )

        fitted = train_bf.fit(method="ipw")

        # data= path
        dm_sample_via_data = fitted.design_matrix(on="sample", data=holdout_bf)
        dm_target_via_data = fitted.design_matrix(on="target", data=holdout_bf)
        dm_both_via_data = fitted.design_matrix(on="both", data=holdout_bf)

        # set_fitted_model path
        scored = holdout_bf.set_fitted_model(fitted, inplace=False)
        dm_sample_via_wfm = scored.design_matrix(on="sample")
        dm_target_via_wfm = scored.design_matrix(on="target")

        self.assertEqual(dm_sample_via_data.shape[0], len(holdout_sample_df))
        self.assertEqual(dm_target_via_data.shape[0], len(holdout_target_df))

        pd.testing.assert_frame_equal(dm_sample_via_data, dm_sample_via_wfm)
        pd.testing.assert_frame_equal(dm_target_via_data, dm_target_via_wfm)

        # on="both" returns a tuple
        self.assertIsInstance(dm_both_via_data, tuple)
        pd.testing.assert_frame_equal(dm_both_via_data[0], dm_sample_via_wfm)
        pd.testing.assert_frame_equal(dm_both_via_data[1], dm_target_via_wfm)

    def test_data_argument_validates_covariate_columns(self) -> None:
        """data= raises ValueError when covariate columns don't match."""
        fitted = self.bf.fit(method="ipw")
        bad_sample_df = self.sample.df.copy().rename(columns={"x": "x2"})
        bad_target_df = self.target.df.copy().rename(columns={"x": "x2"})
        bad_bf = BalanceFrame(
            sample=SampleFrame.from_frame(bad_sample_df),
            target=SampleFrame.from_frame(bad_target_df),
        )
        with self.assertRaisesRegex(
            ValueError, "matching sample covariate column names"
        ):
            fitted.predict_weights(data=bad_bf)
        with self.assertRaisesRegex(
            ValueError, "matching sample covariate column names"
        ):
            fitted.predict_proba(on="sample", data=bad_bf)
        with self.assertRaisesRegex(
            ValueError, "matching sample covariate column names"
        ):
            fitted.design_matrix(on="sample", data=bad_bf)

    def test_data_argument_does_not_cache_results(self) -> None:
        """data= path should not mutate the fitted object's model dict."""
        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        holdout_sample_df = self.sample.df.iloc[5:].copy()
        holdout_target_df = self.target.df.iloc[5:].copy()
        holdout_sample_df.index = pd.Index(
            [f"h{i}" for i in range(len(holdout_sample_df))]
        )
        holdout_target_df.index = pd.Index(
            [f"ht{i}" for i in range(len(holdout_target_df))]
        )
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(holdout_sample_df),
            target=SampleFrame.from_frame(holdout_target_df),
        )

        fitted = train_bf.fit(method="ipw")
        model_before = dict(_assert_type(fitted.model))

        # Call with data= — should not alter the fitted model's cached artifacts
        fitted.predict_weights(data=holdout_bf)
        fitted.predict_proba(on="sample", data=holdout_bf)
        fitted.design_matrix(on="sample", data=holdout_bf)

        model_after = _assert_type(fitted.model)
        # The stored sample/target indices should remain unchanged
        self.assertTrue(
            model_before.get("sample_index", pd.Index([])).equals(
                model_after.get("sample_index", pd.Index([]))
            )
        )

    def test_data_argument_without_target_raises(self) -> None:
        """data= with no target on holdout should raise ValueError."""
        fitted = self.bf.fit(method="ipw")
        no_target_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
        )
        with self.assertRaisesRegex(ValueError, "target"):
            fitted.predict_weights(data=no_target_bf)
        with self.assertRaisesRegex(ValueError, "target"):
            fitted.predict_proba(on="sample", data=no_target_bf)

    def test_predict_proba_on_both_with_data_argument(self) -> None:
        """predict_proba(on='both', data=...) should return both sides."""
        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[5:].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[5:].copy()),
        )
        fitted = train_bf.fit(method="ipw")
        sample_p, target_p = fitted.predict_proba(on="both", data=holdout_bf)
        self.assertEqual(sample_p.shape[0], len(holdout_bf._sf_sample.df))
        self.assertEqual(
            target_p.shape[0],
            len(_assert_type(holdout_bf._sf_target).df),
        )
        self.assertTrue(np.all(np.isfinite(sample_p.to_numpy())))
        self.assertTrue(np.all(np.isfinite(target_p.to_numpy())))

    def test_data_argument_rejects_non_ipw_method(self) -> None:
        """predict_weights(data=...) with non-IPW method should raise."""
        adjusted = self.bf.fit(method="null")
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        with self.assertRaisesRegex(ValueError, "not yet supported"):
            adjusted.predict_weights(data=holdout_bf)

    def test_predict_weights_cbps_fit_matches_weight_series(self) -> None:
        """CBPS fitted model supports predict_weights() reconstruction."""
        adjusted = self.bf.fit(method="cbps", transformations=None)
        predicted_weights = adjusted.predict_weights()
        np.testing.assert_allclose(
            predicted_weights.to_numpy(),
            _assert_type(adjusted.weight_series).to_numpy(),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_predict_weights_cbps_data_argument(self) -> None:
        """CBPS predict_weights(data=...) scores holdout sample rows."""
        train_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[:5].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[:5].copy()),
        )
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(self.sample.df.iloc[5:].copy()),
            target=SampleFrame.from_frame(self.target.df.iloc[5:].copy()),
        )
        fitted = train_bf.fit(method="cbps", transformations=None)
        weights_via_data = fitted.predict_weights(data=holdout_bf)
        self.assertEqual(weights_via_data.shape[0], len(holdout_bf._sf_sample.df))
        self.assertTrue(np.all(np.isfinite(weights_via_data.to_numpy())))
        self.assertTrue(np.all(weights_via_data.to_numpy() >= 0))

    def test_predict_weights_cbps_requires_fit_metadata(self) -> None:
        adjusted = self.bf.adjust(method="cbps", transformations=None)
        with self.assertRaisesRegex(ValueError, "store_fit_metadata=True"):
            adjusted.predict_weights()

    def test_predict_weights_cbps_raises_on_missing_svd_metadata(self) -> None:
        adjusted = self.bf.fit(method="cbps", transformations=None)
        model = _assert_type(adjusted.model)
        model.pop("svd_s", None)
        with self.assertRaisesRegex(ValueError, "missing fit-time metadata"):
            adjusted.predict_weights()

    def test_predict_weights_cbps_raises_on_bad_standardization_shape(self) -> None:
        adjusted = self.bf.fit(method="cbps", transformations=None)
        model = _assert_type(adjusted.model)
        model["model_matrix_mean"] = np.asarray([0.0])
        with self.assertRaisesRegex(ValueError, "incompatible standardization vectors"):
            adjusted.predict_weights(data=self.bf)

    def test_fit_cbps_na_drop_with_explicit_fit_metadata_raises(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "incompatible with stored fit metadata"
        ):
            self.bf.fit(method="cbps", na_action="drop", store_fit_metadata=True)

    def test_fit_cbps_na_drop_defaults_to_warning_and_disables_metadata(self) -> None:
        with self.assertWarnsRegex(UserWarning, "disables store_fit_metadata"):
            adjusted = self.bf.fit(
                method="cbps", na_action="drop", transformations=None
            )
        self.assertTrue(bool(adjusted.is_adjusted))
        model = _assert_type(adjusted.model)
        self.assertNotIn("store_fit_metadata", model)

    def test_fit_cbps_na_drop_without_fit_metadata_allowed(self) -> None:
        adjusted = self.bf.fit(
            method="cbps",
            na_action="drop",
            store_fit_metadata=False,
            transformations=None,
        )
        self.assertTrue(bool(adjusted.is_adjusted))

    def test_predict_weights_cbps_near_collinear_design_reconstructs(self) -> None:
        """CBPS reconstruction remains stable with near-collinear covariates."""
        n = 24
        rng = np.random.default_rng(2026)
        x_sample = np.linspace(0.0, 1.0, n)
        x_target = np.clip(x_sample + rng.normal(0.0, 0.03, size=n), 0.0, 1.0)
        near_col_sample = x_sample + 1e-7 * np.arange(n)
        near_col_target = x_target + 1e-7 * np.arange(n)

        sample_df = pd.DataFrame(
            {
                "id": [f"s{i}" for i in range(n)],
                "x": x_sample,
                "x_near_collinear": near_col_sample,
                "weight": np.ones(n),
            }
        )
        target_df = pd.DataFrame(
            {
                "id": [f"t{i}" for i in range(n)],
                "x": x_target,
                "x_near_collinear": near_col_target,
                "weight": np.ones(n),
            }
        )
        bf = BalanceFrame(
            sample=SampleFrame.from_frame(sample_df),
            target=SampleFrame.from_frame(target_df),
        )
        fitted = bf.fit(method="cbps", transformations=None)
        predicted_weights = fitted.predict_weights()
        np.testing.assert_allclose(
            predicted_weights.to_numpy(),
            _assert_type(fitted.weight_series).to_numpy(),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_predict_weights_cbps_raises_on_missing_training_weights(self) -> None:
        """In-place scoring must not silently fall back to adjusted weights."""
        adjusted = self.bf.fit(method="cbps", transformations=None)
        model = _assert_type(adjusted.model)
        model.pop("training_sample_weights", None)
        with self.assertRaisesRegex(ValueError, "stored training weights"):
            adjusted.predict_weights()

    def test_predict_weights_cbps_zero_sum_weights_raises(self) -> None:
        """Zero-sum holdout weights surface a clear error via public predict_weights()."""
        fitted = self.bf.fit(method="cbps", transformations=None)
        holdout_sample_df = self.sample.df.copy()
        holdout_sample_df["weight"] = 0.0
        holdout_target_df = self.target.df.copy()
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(holdout_sample_df),
            target=SampleFrame.from_frame(holdout_target_df),
        )
        with self.assertRaisesRegex(ValueError, "positive sample and target weight"):
            fitted.predict_weights(data=holdout_bf)


# =====================================================================
# Coverage tests for uncovered lines in balance_frame.py
# =====================================================================


class TestBalanceFrameIdColumnWarning(BalanceTestCase):
    """Cover lines 201-208: id_column property FutureWarning."""

    def test_id_column_raises_future_warning(self) -> None:
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["1", "2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]})
        )
        bf = BalanceFrame(sample=resp_sf)
        with self.assertWarns(FutureWarning):
            result = bf.id_column
        self.assertEqual(result, "id")


class TestBalanceFrameWeightSeriesNone(BalanceTestCase):
    """Cover lines 215-216: weight_series returns None on ValueError."""

    def test_weight_series_returns_none_on_value_error(self) -> None:
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["1", "2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]})
        )
        bf = BalanceFrame(sample=resp_sf)
        with patch.object(
            type(bf._sf_sample),
            "weight_series",
            new_callable=lambda: property(
                lambda self: (_ for _ in ()).throw(ValueError("no weight"))
            ),
        ):
            result = bf.weight_series
        self.assertIsNone(result)


class TestBalanceFrameOutcomeColumnsSetter(BalanceTestCase):
    """Cover line 247: _outcome_columns setter when set to None."""

    def test_set_outcome_columns_to_none_clears_roles(self) -> None:
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": ["1", "2"],
                    "x": [1.0, 2.0],
                    "y": [0.5, 0.6],
                    "weight": [1.0, 1.0],
                }
            ),
            outcome_columns=["y"],
        )
        bf = BalanceFrame(sample=resp_sf)
        self.assertIsNotNone(bf._outcome_columns)
        bf._outcome_columns = None
        self.assertIsNone(bf._outcome_columns)
        self.assertEqual(bf._sf_sample._column_roles["outcomes"], [])

    def test_set_outcome_columns_to_dataframe(self) -> None:
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": ["1", "2"],
                    "x": [1.0, 2.0],
                    "y": [0.5, 0.6],
                    "weight": [1.0, 1.0],
                }
            ),
        )
        bf = BalanceFrame(sample=resp_sf)
        bf._outcome_columns = pd.DataFrame({"y": [0.5, 0.6]})
        self.assertEqual(bf._sf_sample._column_roles["outcomes"], ["y"])


class TestBalanceFrameDfAccessors(BalanceTestCase):
    """Cover lines 418, 423-425, 430: df_responders, df_target, df_responders_unadjusted."""

    def test_df_responders(self) -> None:
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["1", "2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]})
        )
        tgt_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["3", "4"], "x": [1.5, 2.5], "weight": [1.0, 1.0]})
        )
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
        pd.testing.assert_frame_equal(bf.df_responders, resp_sf.df)

    def test_df_target_with_target(self) -> None:
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["1", "2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]})
        )
        tgt_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["3", "4"], "x": [1.5, 2.5], "weight": [1.0, 1.0]})
        )
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
        pd.testing.assert_frame_equal(_assert_type(bf.df_target), tgt_sf.df)

    def test_df_target_without_target(self) -> None:
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["1", "2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]})
        )
        bf = BalanceFrame(sample=resp_sf)
        self.assertIsNone(bf.df_target)

    def test_df_responders_unadjusted(self) -> None:
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["1", "2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]})
        )
        tgt_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["3", "4"], "x": [1.5, 2.5], "weight": [1.0, 1.0]})
        )
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
        pd.testing.assert_frame_equal(bf.df_responders_unadjusted, resp_sf.df)


class TestBalanceFrameGetCovarsNoTarget(BalanceTestCase):
    """Cover line 661: _get_covars raises when no target."""

    def test_get_covars_raises_without_target(self) -> None:
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["1", "2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]})
        )
        bf = BalanceFrame(sample=resp_sf)
        with self.assertRaises(ValueError):
            bf._get_covars()


class TestBalanceFrameSetFittedModelValidation(BalanceTestCase):
    """Cover lines 1148, 1151, 1158, 1169, 1211: set_fitted_model validation."""

    def _make_fitted(self) -> tuple[BalanceFrame, SampleFrame, SampleFrame]:
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": [f"s{i}" for i in range(8)],
                    "x": [0.1, 0.3, 0.7, 1.1, 1.4, 1.7, 2.0, 2.2],
                    "weight": [1.0] * 8,
                }
            )
        )
        tgt_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": [f"t{i}" for i in range(8)],
                    "x": [0.2, 0.5, 0.8, 1.0, 1.6, 1.9, 2.1, 2.4],
                    "weight": [1.0] * 8,
                }
            )
        )
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
        fitted = bf.fit(method="ipw")
        return fitted, resp_sf, tgt_sf

    def test_model_not_dict_raises(self) -> None:
        """Line 1148: fitted model is not a dict."""
        fitted, resp_sf, tgt_sf = self._make_fitted()

        fitted._adjustment_model = "not_a_dict"  # pyre-ignore[8]
        holdout = BalanceFrame(
            sample=SampleFrame.from_frame(resp_sf._df.copy()),
            target=SampleFrame.from_frame(tgt_sf._df.copy()),
        )
        with self.assertRaisesRegex(ValueError, "valid adjustment model dict"):
            holdout.set_fitted_model(fitted)

    def test_method_not_ipw_raises(self) -> None:
        """Line 1151: method is not ipw."""
        fitted, resp_sf, tgt_sf = self._make_fitted()
        _assert_type(fitted._adjustment_model)["method"] = "cbps"
        holdout = BalanceFrame(
            sample=SampleFrame.from_frame(resp_sf._df.copy()),
            target=SampleFrame.from_frame(tgt_sf._df.copy()),
        )
        with self.assertRaisesRegex(ValueError, "only IPW models"):
            holdout.set_fitted_model(fitted)

    def test_missing_fit_raises(self) -> None:
        """Line 1158: model missing fit key."""
        fitted, resp_sf, tgt_sf = self._make_fitted()
        _assert_type(fitted._adjustment_model)["fit"] = None
        holdout = BalanceFrame(
            sample=SampleFrame.from_frame(resp_sf._df.copy()),
            target=SampleFrame.from_frame(tgt_sf._df.copy()),
        )
        with self.assertRaisesRegex(ValueError, "missing estimator"):
            holdout.set_fitted_model(fitted)

    def test_mismatched_target_covariates_raises(self) -> None:
        """Line 1169: target covariates mismatch."""
        fitted, resp_sf, _tgt_sf = self._make_fitted()
        # Create holdout with matching sample covars but different target covars.
        # Use _create to set explicit covar_columns on the target.
        holdout_tgt = SampleFrame._create(
            df=pd.DataFrame(
                {
                    "id": [f"ht{i}" for i in range(8)],
                    "x": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
                    "z": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    "weight": [1.0] * 8,
                }
            ),
            id_column="id",
            covar_columns=["x", "z"],
            weight_columns=["weight"],
        )
        holdout = BalanceFrame(
            sample=SampleFrame.from_frame(resp_sf._df.copy()),
            target=holdout_tgt,
        )
        with self.assertRaisesRegex(ValueError, "matching target covariate"):
            holdout.set_fitted_model(fitted)

    def test_set_fitted_model_inplace_false(self) -> None:
        """Line 1211: inplace=False branch."""
        fitted, resp_sf, tgt_sf = self._make_fitted()
        holdout = BalanceFrame(
            sample=SampleFrame.from_frame(resp_sf._df.copy()),
            target=SampleFrame.from_frame(tgt_sf._df.copy()),
        )
        result = holdout.set_fitted_model(fitted, inplace=False)
        self.assertIsNot(result, holdout)
        self.assertTrue(result.is_adjusted)
        self.assertFalse(holdout.is_adjusted)

    def test_set_fitted_model_inplace_true(self) -> None:
        """Line 1211: inplace=True (default) branch."""
        fitted, resp_sf, tgt_sf = self._make_fitted()
        holdout = BalanceFrame(
            sample=SampleFrame.from_frame(resp_sf._df.copy()),
            target=SampleFrame.from_frame(tgt_sf._df.copy()),
        )
        result = holdout.set_fitted_model(fitted, inplace=True)
        self.assertIs(result, holdout)
        self.assertTrue(holdout.is_adjusted)


class TestBalanceFrameRequireIpwModelAndHelpers(BalanceTestCase):
    """Cover lines 1270, 1282, 1291, 1310, 1317, 1328, 1333."""

    def setUp(self) -> None:
        super().setUp()
        self.resp_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": [f"s{i}" for i in range(8)],
                    "x": [0.1, 0.3, 0.7, 1.1, 1.4, 1.7, 2.0, 2.2],
                    "weight": [1.0] * 8,
                }
            )
        )
        self.tgt_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": [f"t{i}" for i in range(8)],
                    "x": [0.2, 0.5, 0.8, 1.0, 1.6, 1.9, 2.1, 2.4],
                    "weight": [1.0] * 8,
                }
            )
        )
        self.bf = BalanceFrame(sample=self.resp_sf, target=self.tgt_sf)

    def test_require_ipw_model_non_ipw_raises(self) -> None:
        """Line 1270: non-IPW model."""
        adjusted = self.bf.fit(method="null")
        with self.assertRaisesRegex(ValueError, "only IPW models"):
            adjusted._require_ipw_model()

    def test_require_ipw_model_missing_fit_info(self) -> None:
        """Line 1270: IPW method but missing fit info."""
        adjusted = self.bf.fit(method="ipw")
        model = _assert_type(adjusted.model)
        model["fit"] = None
        model["X_matrix_columns"] = None
        with self.assertRaisesRegex(ValueError, "missing fitted model"):
            adjusted._require_ipw_model()

    def test_matrix_to_dataframe_numpy_array(self) -> None:
        """Line 1282: numpy array path."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        idx = pd.Index([0, 1])
        cols = ["a", "b"]
        result = self.bf._matrix_to_dataframe(arr, idx, cols)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(list(result.columns), ["a", "b"])

    def test_matrix_to_dataframe_spmatrix(self) -> None:
        """Line 1291: sparse matrix conversion."""
        from scipy.sparse import csr_matrix

        arr = csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
        idx = pd.Index([0, 1])
        cols = ["a", "b"]
        result = self.bf._matrix_to_dataframe(arr, idx, cols)
        self.assertEqual(result.shape, (2, 2))

    def test_matrix_to_dataframe_unknown_type_raises(self) -> None:
        """Line 1291: unknown matrix type raises ValueError."""
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.bf._matrix_to_dataframe("not_a_matrix", pd.Index([0]), ["a"])

    def test_align_to_index_mismatched_unique(self) -> None:
        """Line 1310: indices are unique, same length, but different values."""
        data = pd.Series([1, 2], index=pd.Index([10, 20]))
        target_idx = pd.Index([30, 40])
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.bf._align_to_index(data, target_idx, "test")

    def test_align_to_index_mismatched_length(self) -> None:
        """Line 1317: non-unique indices with different lengths."""
        # Non-unique indices to bypass the first if-branch
        data = pd.Series([1, 2, 3], index=pd.Index(["a", "a", "b"]))
        target_idx = pd.Index(["a", "a"])
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            self.bf._align_to_index(data, target_idx, "test")

    def test_ipw_class_index_missing_classes(self) -> None:
        """Line 1328: missing classes_ attribute."""

        class FakeModel:
            pass

        with self.assertRaisesRegex(ValueError, "missing classes_"):
            BalanceFrame._ipw_class_index(FakeModel())

    def test_ipw_class_index_missing_class_1(self) -> None:
        """Line 1333: classes_ present but class 1 is missing."""

        class FakeModel:
            classes_ = [0, 2]

        with self.assertRaisesRegex(ValueError, "missing class label 1"):
            BalanceFrame._ipw_class_index(FakeModel())


class TestBalanceFrameComputeIpwMatricesInferType(BalanceTestCase):
    """Cover lines 1375-1381: infer matrix_type from stored artifacts."""

    def test_infer_matrix_type_from_sparse_artifact(self) -> None:
        from scipy.sparse import csr_matrix

        bf = BalanceFrame(
            sample=SampleFrame.from_frame(
                pd.DataFrame(
                    {
                        "id": [f"s{i}" for i in range(8)],
                        "x": [0.1, 0.3, 0.7, 1.1, 1.4, 1.7, 2.0, 2.2],
                        "weight": [1.0] * 8,
                    }
                )
            ),
            target=SampleFrame.from_frame(
                pd.DataFrame(
                    {
                        "id": [f"t{i}" for i in range(8)],
                        "x": [0.2, 0.5, 0.8, 1.0, 1.6, 1.9, 2.1, 2.4],
                        "weight": [1.0] * 8,
                    }
                )
            ),
        )
        fitted = bf.fit(method="ipw")
        model = _assert_type(fitted.model)
        # Remove explicit fit_matrix_type to force inference
        model.pop("fit_matrix_type", None)
        # Store a sparse matrix in model_matrix_sample to trigger inference
        model["model_matrix_sample"] = csr_matrix(np.eye(8))
        # The method should infer matrix_type as "sparse" and not crash
        sample_matrix, target_matrix = fitted._compute_ipw_matrices(model)
        self.assertIsNotNone(sample_matrix)
        self.assertIsNotNone(target_matrix)

    def _make_bf_and_fitted(self) -> tuple[BalanceFrame, BalanceFrame]:
        bf = BalanceFrame(
            sample=SampleFrame.from_frame(
                pd.DataFrame(
                    {
                        "id": [f"s{i}" for i in range(8)],
                        "x": [0.1, 0.3, 0.7, 1.1, 1.4, 1.7, 2.0, 2.2],
                        "weight": [1.0] * 8,
                    }
                )
            ),
            target=SampleFrame.from_frame(
                pd.DataFrame(
                    {
                        "id": [f"t{i}" for i in range(8)],
                        "x": [0.2, 0.5, 0.8, 1.0, 1.6, 1.9, 2.1, 2.4],
                        "weight": [1.0] * 8,
                    }
                )
            ),
        )
        fitted = bf.fit(method="ipw")
        return bf, fitted

    def test_infer_matrix_type_from_ndarray_artifact(self) -> None:
        _bf, fitted = self._make_bf_and_fitted()
        model = _assert_type(fitted.model)
        model.pop("fit_matrix_type", None)
        model["model_matrix_sample"] = np.eye(8)
        sample_matrix, target_matrix = fitted._compute_ipw_matrices(model)
        self.assertIsNotNone(sample_matrix)

    def test_infer_matrix_type_from_dataframe_artifact(self) -> None:
        _bf, fitted = self._make_bf_and_fitted()
        model = _assert_type(fitted.model)
        model.pop("fit_matrix_type", None)
        model["model_matrix_sample"] = pd.DataFrame(np.eye(8))
        sample_matrix, target_matrix = fitted._compute_ipw_matrices(model)
        self.assertIsNotNone(sample_matrix)


class TestBalanceFrameIsArtifactStale(BalanceTestCase):
    """Cover line 1407: _is_artifact_stale returns True when artifact is None."""

    def test_stale_when_artifact_is_none(self) -> None:
        result = BalanceFrame._is_artifact_stale(None, pd.Index([0]), pd.Index([0]))
        self.assertTrue(result)

    def test_stale_when_length_mismatch(self) -> None:
        arr = np.array([1.0, 2.0])
        result = BalanceFrame._is_artifact_stale(arr, pd.Index([0, 1]), pd.Index([0]))
        self.assertTrue(result)

    def test_not_stale_when_matching(self) -> None:
        arr = np.array([1.0, 2.0])
        idx = pd.Index([0, 1])
        result = BalanceFrame._is_artifact_stale(arr, idx, idx)
        self.assertFalse(result)


class TestBalanceFrameValidateDataCovariates(BalanceTestCase):
    """Cover line 1485: mismatched target covariates in _validate_data_covariates."""

    def test_mismatched_target_covariates(self) -> None:
        bf1 = BalanceFrame(
            sample=SampleFrame._create(
                df=pd.DataFrame({"id": ["1"], "x": [1.0], "weight": [1.0]}),
                id_column="id",
                covar_columns=["x"],
                weight_columns=["weight"],
            ),
            target=SampleFrame._create(
                df=pd.DataFrame({"id": ["2"], "x": [2.0], "weight": [1.0]}),
                id_column="id",
                covar_columns=["x"],
                weight_columns=["weight"],
            ),
        )
        # Build data with sample covar "x" matching bf1, but target covar
        # "x" AND "z" so it differs from bf1's target (just "x").
        # Use _create to set explicit covar_columns.
        data_sample = SampleFrame._create(
            df=pd.DataFrame({"id": ["3"], "x": [3.0], "weight": [1.0]}),
            id_column="id",
            covar_columns=["x"],
            weight_columns=["weight"],
        )
        data_target = SampleFrame._create(
            df=pd.DataFrame({"id": ["4"], "x": [4.0], "z": [5.0], "weight": [1.0]}),
            id_column="id",
            covar_columns=["x", "z"],
            weight_columns=["weight"],
        )
        data = BalanceFrame(sample=data_sample, target=data_target)
        with self.assertRaisesRegex(ValueError, "matching target covariate"):
            bf1._validate_data_covariates(data)


class TestBalanceFrameDesignMatrixTargetErrors(BalanceTestCase):
    """Cover lines 1578, 1629: design_matrix errors for target path."""

    def setUp(self) -> None:
        super().setUp()
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": [f"s{i}" for i in range(8)],
                    "x": [0.1, 0.3, 0.7, 1.1, 1.4, 1.7, 2.0, 2.2],
                    "weight": [1.0] * 8,
                }
            )
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": [f"t{i}" for i in range(8)],
                    "x": [0.2, 0.5, 0.8, 1.0, 1.6, 1.9, 2.1, 2.4],
                    "weight": [1.0] * 8,
                }
            )
        )
        self.bf = BalanceFrame.from_sample(sample.set_target(target))
        self.fitted = self.bf.fit(method="ipw")

    def test_design_matrix_data_on_target_no_target(self) -> None:
        """Line 1578: data must have a target set for on='target'."""
        data_no_target = BalanceFrame(
            sample=SampleFrame.from_frame(
                pd.DataFrame(
                    {
                        "id": ["h1", "h2"],
                        "x": [1.0, 2.0],
                        "weight": [1.0, 1.0],
                    }
                )
            ),
        )
        with self.assertRaisesRegex(ValueError, "target"):
            self.fitted.design_matrix(on="target", data=data_no_target)

    def test_design_matrix_on_target_missing_matrix(self) -> None:
        """Line 1629: IPW model missing fit-time target matrix."""
        model = _assert_type(self.fitted.model)
        model.pop("model_matrix_target", None)
        with self.assertRaisesRegex(ValueError, "missing fit-time target matrix"):
            self.fitted.design_matrix(on="target")


class TestBalanceFramePredictProbaTargetErrors(BalanceTestCase):
    """Cover lines 1749, 1778: predict_proba errors for data= target path."""

    def setUp(self) -> None:
        super().setUp()
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": [f"s{i}" for i in range(8)],
                    "x": [0.1, 0.3, 0.7, 1.1, 1.4, 1.7, 2.0, 2.2],
                    "weight": [1.0] * 8,
                }
            )
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": [f"t{i}" for i in range(8)],
                    "x": [0.2, 0.5, 0.8, 1.0, 1.6, 1.9, 2.1, 2.4],
                    "weight": [1.0] * 8,
                }
            )
        )
        self.bf = BalanceFrame.from_sample(sample.set_target(target))
        self.fitted = self.bf.fit(method="ipw")

    def test_predict_proba_data_on_target_no_target(self) -> None:
        """Line 1749: data must have a target set for on='target'."""
        data_no_target = BalanceFrame(
            sample=SampleFrame.from_frame(
                pd.DataFrame(
                    {"id": ["h1", "h2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]}
                )
            ),
        )
        with self.assertRaisesRegex(ValueError, "target"):
            self.fitted.predict_proba(on="target", data=data_no_target)

    def test_predict_proba_link_output_for_target_data(self) -> None:
        """Line 1778: target link transform path in data= mode."""
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(
                pd.DataFrame(
                    {"id": ["h1", "h2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]}
                )
            ),
            target=SampleFrame.from_frame(
                pd.DataFrame(
                    {"id": ["ht1", "ht2"], "x": [1.5, 2.5], "weight": [1.0, 1.0]}
                )
            ),
        )
        result = self.fitted.predict_proba(on="target", output="link", data=holdout_bf)
        self.assertEqual(result.shape[0], 2)
        self.assertTrue(np.all(np.isfinite(result.to_numpy())))


class TestBalanceFramePredictWeightsDataErrors(BalanceTestCase):
    """Cover lines 1926, 1938: predict_weights data= error branches."""

    def setUp(self) -> None:
        super().setUp()
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": [f"s{i}" for i in range(8)],
                    "x": [0.1, 0.3, 0.7, 1.1, 1.4, 1.7, 2.0, 2.2],
                    "weight": [1.0] * 8,
                }
            )
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": [f"t{i}" for i in range(8)],
                    "x": [0.2, 0.5, 0.8, 1.0, 1.6, 1.9, 2.1, 2.4],
                    "weight": [1.0] * 8,
                }
            )
        )
        self.bf = BalanceFrame.from_sample(sample.set_target(target))
        self.fitted = self.bf.fit(method="ipw")

    def test_predict_weights_data_missing_metadata(self) -> None:
        """Line 1926: IPW model metadata missing fitted model info."""
        model = _assert_type(self.fitted.model)
        model["fit"] = None
        model["X_matrix_columns"] = None
        holdout_bf = BalanceFrame(
            sample=SampleFrame.from_frame(
                pd.DataFrame(
                    {"id": ["h1", "h2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]}
                )
            ),
            target=SampleFrame.from_frame(
                pd.DataFrame(
                    {"id": ["ht1", "ht2"], "x": [1.5, 2.5], "weight": [1.0, 1.0]}
                )
            ),
        )
        with self.assertRaisesRegex(ValueError, "missing fitted model"):
            self.fitted.predict_weights(data=holdout_bf)


class TestBalanceFrameResolveTrainingWeights(BalanceTestCase):
    """Cover lines 2049-2053, 2061-2065: _resolve_training_weights fallbacks."""

    def setUp(self) -> None:
        super().setUp()
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": [f"s{i}" for i in range(8)],
                    "x": [0.1, 0.3, 0.7, 1.1, 1.4, 1.7, 2.0, 2.2],
                    "weight": [1.0] * 8,
                }
            )
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": [f"t{i}" for i in range(8)],
                    "x": [0.2, 0.5, 0.8, 1.0, 1.6, 1.9, 2.1, 2.4],
                    "weight": [1.0] * 8,
                }
            )
        )
        self.bf = BalanceFrame.from_sample(sample.set_target(target))
        self.fitted = self.bf.fit(method="ipw")

    def test_sample_weights_fallback(self) -> None:
        """Lines 2049-2053: sample weights fallback when training weights unavailable."""
        model = _assert_type(self.fitted.model)
        model.pop("training_sample_weights", None)
        link = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        with self.assertLogs("balance", level="WARNING") as cm:
            sample_w, target_w = self.fitted._resolve_design_weights(model, link)
        self.assertTrue(any("training_sample_weights" in msg for msg in cm.output))
        self.assertEqual(len(sample_w), 8)

    def test_target_weights_fallback(self) -> None:
        """Lines 2061-2065: target weights fallback when training weights unavailable."""
        model = _assert_type(self.fitted.model)
        model.pop("training_target_weights", None)
        link = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        with self.assertLogs("balance", level="WARNING") as cm:
            sample_w, target_w = self.fitted._resolve_design_weights(model, link)
        self.assertTrue(any("training_target_weights" in msg for msg in cm.output))
        self.assertEqual(len(target_w), 8)


class TestBalanceFramePredictWeightsIpwIndexMismatch(BalanceTestCase):
    """Cover lines 2076, 2095: _predict_weights_ipw sample index mismatch."""

    def test_predict_weights_ipw_missing_fit_info(self) -> None:
        """Line 2076: IPW model metadata missing fitted model info."""
        bf = BalanceFrame(
            sample=SampleFrame.from_frame(
                pd.DataFrame(
                    {
                        "id": [f"s{i}" for i in range(8)],
                        "x": [0.1, 0.3, 0.7, 1.1, 1.4, 1.7, 2.0, 2.2],
                        "weight": [1.0] * 8,
                    }
                )
            ),
            target=SampleFrame.from_frame(
                pd.DataFrame(
                    {
                        "id": [f"t{i}" for i in range(8)],
                        "x": [0.2, 0.5, 0.8, 1.0, 1.6, 1.9, 2.1, 2.4],
                        "weight": [1.0] * 8,
                    }
                )
            ),
        )
        adjusted = bf.fit(method="ipw")
        model = _assert_type(adjusted.model)
        model["fit"] = None
        model["X_matrix_columns"] = None
        with self.assertRaisesRegex(ValueError, "missing fitted model"):
            adjusted._predict_weights_ipw(model)

    def test_predict_weights_ipw_index_fallback(self) -> None:
        """Line 2095: sample_idx fallback when it doesn't match current."""
        bf = BalanceFrame(
            sample=SampleFrame.from_frame(
                pd.DataFrame(
                    {
                        "id": [f"s{i}" for i in range(8)],
                        "x": [0.1, 0.3, 0.7, 1.1, 1.4, 1.7, 2.0, 2.2],
                        "weight": [1.0] * 8,
                    }
                )
            ),
            target=SampleFrame.from_frame(
                pd.DataFrame(
                    {
                        "id": [f"t{i}" for i in range(8)],
                        "x": [0.2, 0.5, 0.8, 1.0, 1.6, 1.9, 2.1, 2.4],
                        "weight": [1.0] * 8,
                    }
                )
            ),
        )
        adjusted = bf.fit(method="ipw")
        model = _assert_type(adjusted.model)
        # Set sample_index to something that doesn't match current
        model["sample_index"] = pd.Index([f"a{i}" for i in range(8)])
        # This should still work by falling back
        w = adjusted._predict_weights_ipw(model)
        self.assertEqual(len(w), 8)


class TestBalanceFrameToSampleNoTarget(BalanceTestCase):
    """Cover line 2218: to_sample without target raises ValueError."""

    def test_to_sample_without_target_raises(self) -> None:
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["1", "2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]})
        )
        bf = BalanceFrame(sample=resp_sf)
        with self.assertRaisesRegex(ValueError, "no target"):
            bf.to_sample()


class TestBalanceFrameEffectiveSampleSizeNone(BalanceTestCase):
    """Cover line 2410: _design_effect_diagnostics returns None,None,None for non-finite de."""

    def test_design_effect_nonfinite_returns_three_nones(self) -> None:
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["1", "2"], "x": [1.0, 2.0], "weight": [1.0, 1.0]})
        )
        tgt_sf = SampleFrame.from_frame(
            pd.DataFrame({"id": ["3", "4"], "x": [1.5, 2.5], "weight": [1.0, 1.0]})
        )
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
        from balance.stats_and_plots import weights_stats

        with patch.object(weights_stats, "design_effect", return_value=float("inf")):
            de, ess, essp = bf._design_effect_diagnostics()
        self.assertIsNone(de)
        self.assertIsNone(ess)
        self.assertIsNone(essp)


class TestBalanceFrameKeepOnlyPreAdjustAndLinks(BalanceTestCase):
    """Cover lines 2777, 2790-2791: keep_only_some_rows_columns with pre_adjust and _links."""

    def test_filter_pre_adjust_when_differs_from_sample(self) -> None:
        """Line 2777: filter pre_adjust when it differs from sample."""
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": ["1", "2", "3"],
                    "x": [10.0, 20.0, 30.0],
                    "weight": [1.0, 1.0, 1.0],
                }
            )
        )
        tgt_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": ["4", "5", "6"],
                    "x": [15.0, 25.0, 35.0],
                    "weight": [1.0, 1.0, 1.0],
                }
            )
        )
        bf = BalanceFrame(sample=resp_sf, target=tgt_sf)
        adjusted = bf.adjust(method="null")
        # adjusted has pre_adjust != sample
        filtered = adjusted.keep_only_some_rows_columns(rows_to_keep="x > 15")
        self.assertEqual(len(filtered.responders._df), 2)
        self.assertIsNotNone(filtered.unadjusted)
        assert filtered.unadjusted is not None
        self.assertEqual(len(filtered.unadjusted._df), 2)

    def test_links_filter_error_is_caught(self) -> None:
        """Lines 2790-2791: catch errors filtering _links."""
        resp_sf = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": ["1", "2"],
                    "x": [1.0, 2.0],
                    "weight": [1.0, 1.0],
                }
            )
        )
        bf = BalanceFrame(sample=resp_sf)
        # Put a BalanceFrame in _links whose keep_only_some_rows_columns will fail
        linked_bf = BalanceFrame(
            sample=SampleFrame.from_frame(
                pd.DataFrame({"id": ["3"], "z": [99.0], "weight": [1.0]})
            )
        )
        bf._links["test_link"] = linked_bf
        # Filter using column "x" which doesn't exist in linked_bf -> raises
        with self.assertLogs("balance", level="WARNING"):
            result = bf.keep_only_some_rows_columns(rows_to_keep="x > 0")
        self.assertIsNotNone(result)


class TestBalanceFrameFilterSfPredicted(BalanceTestCase):
    """Cover line 2859: _filter_sf for predicted column roles."""

    def test_filter_sf_predicted_columns(self) -> None:
        sf = SampleFrame._create(
            df=pd.DataFrame(
                {
                    "id": ["1", "2"],
                    "x": [1.0, 2.0],
                    "pred": [0.5, 0.6],
                    "weight": [1.0, 1.0],
                }
            ),
            id_column="id",
            covar_columns=["x"],
            weight_columns=["weight"],
        )
        sf._column_roles["predicted"] = ["pred"]
        filtered = BalanceFrame._filter_sf(sf, None, ["x"])
        # predicted column should be filtered (pred not in keep set)
        self.assertEqual(filtered._column_roles.get("predicted", []), [])
