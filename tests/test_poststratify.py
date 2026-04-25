# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import (
    absolute_import,
    annotations,
    division,
    print_function,
    unicode_literals,
)

import balance.testutil
import numpy as np
import pandas as pd
from balance import adjustment as balance_adjustment
from balance.sample_class import Sample
from balance.util import _assert_type
from balance.weighting_methods.poststratify import poststratify


class Testpoststratify(
    balance.testutil.BalanceTestCase,
):
    def test_poststratify(self) -> None:
        s = pd.DataFrame(
            {
                "a": (0, 1, 0, 1),
                "c": ("a", "a", "b", "b"),
            },
        )
        s_weights = pd.Series([4, 2, 1, 1])
        t = s
        t_weights = pd.Series([4, 2, 2, 8])
        result = poststratify(
            sample_df=s, sample_weights=s_weights, target_df=t, target_weights=t_weights
        )["weight"]
        self.assertEqual(result, t_weights.astype("float64"))

        # same example when dataframe of elements are all related to weights of one
        s = pd.DataFrame(
            {
                "a": (0, 0, 0, 0, 1, 1, 0, 1),
                "c": ("a", "a", "a", "a", "a", "a", "b", "b"),
            },
        )
        s_weights = pd.Series([1, 1, 1, 1, 1, 1, 1, 1])
        result = poststratify(
            sample_df=s, sample_weights=s_weights, target_df=t, target_weights=t_weights
        )["weight"]
        self.assertEqual(result, pd.Series((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 8.0)))

        # same example with normalized weights
        s = pd.DataFrame(
            {
                "a": (0, 1, 0, 1),
                "c": ("a", "a", "b", "b"),
            },
        )
        s_weights = pd.Series([1 / 2, 1 / 4, 1 / 8, 1 / 8])
        result = poststratify(
            sample_df=s, sample_weights=s_weights, target_df=t, target_weights=t_weights
        )["weight"]
        self.assertEqual(result, t_weights.astype("float64"))

        # test through adjustment
        # TODO: test the previous example through adjustment as well
        sample = Sample.from_frame(
            df=pd.DataFrame(
                {
                    "a": (1, 2, 3, 1),
                    "b": (-42, 8, 2, -42),
                    "o": (7, 8, 9, 10),
                    "c": ("x", "y", "z", "x"),
                    "id": (1, 2, 3, 4),
                    "w": (0.5, 2, 1, 1),
                }
            ),
            id_column="id",
            weight_column="w",
            outcome_columns="o",
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": (1, 2, 3),
                    "b": (-42, 8, 2),
                    "c": ("x", "y", "z"),
                    "id": (1, 2, 3),
                    "w": (2, 0.5, 1),
                }
            ),
            id_column="id",
            weight_column="w",
        )
        result = sample.adjust(target, method="poststratify", transformations=None)
        expected = pd.Series(
            (
                (2 / 1.5 * 0.5),
                (0.5 / 2 * 2),
                (1 / 1 * 1),
                (2 / 1.5 * 1),
            )
        )
        self.assertEqual(expected, result.weights().df.iloc[:, 0].values)

    def test_poststratify_weight_trimming_applied(self) -> None:
        s = pd.DataFrame(
            {
                "a": (0, 1, 0, 1),
                "c": ("a", "a", "b", "b"),
            },
        )
        s_weights = pd.Series([4, 2, 1, 1])
        t = s
        t_weights = pd.Series([4, 2, 2, 8])

        baseline_result = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
        )["weight"]
        assert isinstance(baseline_result, pd.Series)
        baseline = baseline_result

        trimmed_result = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
            weight_trimming_mean_ratio=1.0,
        )["weight"]
        assert isinstance(trimmed_result, pd.Series)
        trimmed = trimmed_result

        expected = balance_adjustment.trim_weights(
            baseline,
            target_sum_weights=baseline.sum(),
            weight_trimming_mean_ratio=1.0,
        ).rename(baseline.name)

        pd.testing.assert_series_equal(trimmed, expected)

    def test_poststratify_percentile_trimming_applied(self) -> None:
        s = pd.DataFrame(
            {
                "a": (0, 1, 0, 1),
                "c": ("a", "a", "b", "b"),
            },
        )
        s_weights = pd.Series([4, 2, 1, 1])
        t = s
        t_weights = pd.Series([4, 2, 2, 8])

        baseline_result = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
        )["weight"]
        assert isinstance(baseline_result, pd.Series)
        baseline = baseline_result

        trimmed_result = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
            weight_trimming_percentile=0.25,
        )["weight"]
        assert isinstance(trimmed_result, pd.Series)
        trimmed = trimmed_result

        expected = balance_adjustment.trim_weights(
            baseline,
            target_sum_weights=baseline.sum(),
            weight_trimming_percentile=0.25,
        ).rename(baseline.name)

        pd.testing.assert_series_equal(trimmed, expected)

    def test_poststratify_stores_fit_metadata_when_enabled(self) -> None:
        sample_df = pd.DataFrame({"a": ["x", "y", "x"], "b": ["u", "u", "v"]})
        target_df = pd.DataFrame({"a": ["x", "y", "x"], "b": ["u", "u", "v"]})
        s_weights = pd.Series([1.0, 1.0, 1.0])
        t_weights = pd.Series([1.0, 1.0, 1.0])

        result = poststratify(
            sample_df=sample_df,
            sample_weights=s_weights,
            target_df=target_df,
            target_weights=t_weights,
            variables=["a", "b"],
            transformations=None,
            store_fit_metadata=True,
        )
        model = result["model"]
        assert isinstance(model, dict)
        self.assertEqual(model["method"], "poststratify")
        self.assertIn("cell_weight_ratio", model)
        self.assertIn("training_sample_weights", model)
        self.assertIn("training_target_weights", model)
        self.assertIn("variables", model)
        self.assertTrue(bool(model.get("store_fit_metadata")))

    def test_poststratify_defaults_to_minimal_model_payload(self) -> None:
        sample_df = pd.DataFrame({"a": ["x", "y", "x"]})
        target_df = pd.DataFrame({"a": ["x", "x", "y"]})
        weights = pd.Series([1.0, 1.0, 1.0])
        result = poststratify(
            sample_df=sample_df,
            sample_weights=weights,
            target_df=target_df,
            target_weights=weights,
            variables=["a"],
            transformations=None,
        )
        model = result["model"]
        assert isinstance(model, dict)
        self.assertEqual(model, {"method": "poststratify"})

    def test_poststratify_can_disable_fit_metadata_storage(self) -> None:
        sample_df = pd.DataFrame({"a": ["x", "y", "x"]})
        target_df = pd.DataFrame({"a": ["x", "x", "y"]})
        s_weights = pd.Series([1.0, 1.0, 1.0])
        t_weights = pd.Series([1.0, 1.0, 1.0])

        result = poststratify(
            sample_df=sample_df,
            sample_weights=s_weights,
            target_df=target_df,
            target_weights=t_weights,
            variables=["a"],
            transformations=None,
            store_fit_metadata=False,
        )
        model = result["model"]
        assert isinstance(model, dict)
        self.assertEqual(model, {"method": "poststratify"})

    def test_poststratify_na_drop_stores_full_training_weight_indices(self) -> None:
        sample_df = pd.DataFrame({"a": ["x", "y", None], "b": ["u", "v", "u"]})
        target_df = pd.DataFrame({"a": ["x", "y", None], "b": ["u", "v", "v"]})
        s_weights = pd.Series([1.0, 2.0, 3.0], index=["s0", "s1", "s2"])
        t_weights = pd.Series([1.0, 1.0, 1.0], index=["t0", "t1", "t2"])
        sample_df.index = s_weights.index
        target_df.index = t_weights.index

        result = poststratify(
            sample_df=sample_df,
            sample_weights=s_weights,
            target_df=target_df,
            target_weights=t_weights,
            variables=["a"],
            transformations=None,
            na_action="drop",
            strict_matching=False,
            store_fit_metadata=True,
        )
        model = result["model"]
        assert isinstance(model, dict)
        stored_sample = _assert_type(model.get("training_sample_weights"))
        stored_target = _assert_type(model.get("training_target_weights"))
        self.assertTrue(stored_sample.index.equals(s_weights.index))
        self.assertTrue(stored_target.index.equals(t_weights.index))

    def test_poststratify_rejects_non_bool_store_fit_metadata(self) -> None:
        sample_df = pd.DataFrame({"a": ["x", "y"]})
        target_df = pd.DataFrame({"a": ["x", "y"]})
        weights = pd.Series([1.0, 1.0])
        with self.assertRaisesRegex(TypeError, "store_fit_metadata"):
            poststratify(
                sample_df=sample_df,
                sample_weights=weights,
                target_df=target_df,
                target_weights=weights,
                variables=["a"],
                transformations=None,
                store_fit_metadata="False",  # type: ignore[arg-type]
            )

    def test_poststratify_rejects_unpickleable_transformations_when_storing_metadata(
        self,
    ) -> None:
        sample_df = pd.DataFrame({"a": ["x", "y", "x"]})
        target_df = pd.DataFrame({"a": ["x", "y", "y"]})
        weights = pd.Series([1.0, 1.0, 1.0])

        with self.assertRaisesRegex(ValueError, "must be pickleable"):
            poststratify(
                sample_df=sample_df,
                sample_weights=weights,
                target_df=target_df,
                target_weights=weights,
                variables=["a"],
                transformations={"a": lambda x: x},
                store_fit_metadata=True,
            )

    def test_poststratify_variables_arg(self) -> None:
        s = pd.DataFrame(
            {
                "a": (0, 1, 0, 1),
                "c": ("a", "a", "b", "b"),
            },
        )
        s_weights = pd.Series([4, 2, 2, 3])
        t = s
        t_weights = pd.Series([4, 2, 2, 8])
        result = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
            variables=["a"],
        )["weight"]
        self.assertEqual(result, pd.Series([4.0, 4.0, 2.0, 6.0]))

    def test_poststratify_transformations(self) -> None:
        # for numeric
        size = 10000
        s = pd.DataFrame({"age": np.random.uniform(0, 1, size)})
        tmp = int(size * 0.2)
        t = pd.DataFrame(
            {
                "age": np.concatenate(
                    (
                        np.random.uniform(0, 0.4, tmp),
                        np.random.uniform(0.4, 1, size - tmp),
                    )
                )
            }
        )
        result = poststratify(
            sample_df=s,
            sample_weights=pd.Series([1] * size),
            target_df=t,
            target_weights=pd.Series([1] * size),
        )["weight"]

        # age>0.4 has 4 times as many people than age <0.4 in the target
        # Check that the weights come out as 0.2 and 0.8
        eps = 0.05
        # pyrefly: ignore [missing-attribute]
        self.assertAlmostEqual(result[s.age < 0.4].sum() / size, 0.2, delta=eps)
        # pyrefly: ignore [missing-attribute]
        self.assertAlmostEqual(result[s.age >= 0.4].sum() / size, 0.8, delta=eps)

        # for strings
        size = 10000
        s = pd.DataFrame(
            {"x": np.random.choice(("a", "b", "c"), size, p=(0.95, 0.035, 0.015))}
        )
        t = pd.DataFrame(
            {"x": np.random.choice(("a", "b", "c"), size, p=(0.95, 0.015, 0.035))}
        )
        result = poststratify(
            sample_df=s,
            sample_weights=pd.Series([1] * size),
            target_df=t,
            target_weights=pd.Series([1] * size),
        )["weight"]

        # Output weights should ignore the difference between values 'b' and 'c'
        # since these are combined in default transformations (into '_lumped_other').
        # Hence their frequency would be as in sample
        eps = 0.05
        # pyrefly: ignore [missing-attribute]
        self.assertAlmostEqual(result[s.x == "a"].sum() / size, 0.95, delta=eps)
        # pyrefly: ignore [missing-attribute]
        self.assertAlmostEqual(result[s.x == "b"].sum() / size, 0.035, delta=eps)
        # pyrefly: ignore [missing-attribute]
        self.assertAlmostEqual(result[s.x == "c"].sum() / size, 0.015, delta=eps)

    def test_poststratify_formula(self) -> None:
        s = pd.DataFrame(
            {
                "a": (0, 1, 0, 1),
                "c": ("a", "a", "b", "b"),
            },
        )
        s_weights = pd.Series([4, 2, 2, 3])
        t = s
        t_weights = pd.Series([4, 2, 2, 8])

        result_formula = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
            formula=["a"],
            transformations=None,
        )["weight"]
        self.assertEqual(result_formula, pd.Series([4.0, 4.0, 2.0, 6.0]))

        sample = Sample.from_frame(
            df=pd.DataFrame(
                {
                    "a": (0, 1, 0, 1),
                    "c": ("a", "a", "b", "b"),
                    "id": (1, 2, 3, 4),
                    "w": s_weights,
                }
            ),
            id_column="id",
            weight_column="w",
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": (0, 1, 0, 1),
                    "c": ("a", "a", "b", "b"),
                    "id": (5, 6, 7, 8),
                    "w": t_weights,
                }
            ),
            id_column="id",
            weight_column="w",
        )
        adjusted = sample.adjust(
            target,
            method="poststratify",
            formula=["a"],
            transformations=None,
        )
        self.assertEqual(
            adjusted.weights().df.iloc[:, 0].reset_index(drop=True),
            pd.Series([4.0, 4.0, 2.0, 6.0], name="w"),
        )

    def test_poststratify_formula_edge_cases(self) -> None:
        s = pd.DataFrame(
            {
                "a": (0, 1, 0, 1),
                "b": ("x", "x", "y", "y"),
                "c": ("u", "v", "u", "v"),
            },
        )
        s_weights = pd.Series([1.0, 1.0, 1.0, 1.0])
        t = s.copy()
        t_weights = pd.Series([1.0, 3.0, 1.0, 3.0])

        # Reference result: joint cells on (a, b) via `variables=`.
        result_with_variables = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
            variables=["a", "b"],
            transformations=None,
        )["weight"]

        # Interaction operator `:` is the canonical poststratify syntax.
        result_with_interaction = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
            formula="a:b",
            transformations=None,
        )["weight"]
        pd.testing.assert_series_equal(result_with_interaction, result_with_variables)

        # Dot expansion with exclusion: all common columns minus `c`.
        result_with_dot = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
            formula=". - c",
            transformations=None,
        )["weight"]
        pd.testing.assert_series_equal(result_with_dot, result_with_variables)

        # Plain `.` picks up every common column jointly.
        result_with_bare_dot = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
            formula=".",
            transformations=None,
        )["weight"]
        result_with_all_vars = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
            variables=["a", "b", "c"],
            transformations=None,
        )["weight"]
        pd.testing.assert_series_equal(result_with_bare_dot, result_with_all_vars)

        # A leading `~` LHS is ignored.
        result_with_tilde = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
            formula="outcome ~ a:b",
            transformations=None,
        )["weight"]
        pd.testing.assert_series_equal(result_with_tilde, result_with_variables)

        # List form joint-cells all items, equivalent to ":"-joining them.
        result_with_list = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
            formula=["a", "b"],
            transformations=None,
        )["weight"]
        pd.testing.assert_series_equal(result_with_list, result_with_variables)

        # Three-way interaction still works.
        result_with_triple_interaction = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
            formula="a:b:c",
            transformations=None,
        )["weight"]
        pd.testing.assert_series_equal(
            result_with_triple_interaction, result_with_all_vars
        )

    def test_poststratify_formula_keeps_positional_argument_compatibility(self) -> None:
        s = pd.DataFrame(
            {
                "a": (0, 1, 0, 1),
                "c": ("a", "a", "b", "b"),
            },
        )
        s_weights = pd.Series([4, 2, 2, 3])
        t = s
        t_weights = pd.Series([4, 2, 2, 8])

        result_positional = poststratify(
            s,
            s_weights,
            t,
            t_weights,
            None,  # variables
            None,  # transformations
            True,  # transformations_drop
            True,  # strict_matching
            "add_indicator",  # na_action
            None,  # weight_trimming_mean_ratio
            None,  # weight_trimming_percentile
            True,  # keep_sum_of_weights
        )["weight"]
        result_keywords = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
            transformations=None,
        )["weight"]
        pd.testing.assert_series_equal(result_positional, result_keywords)

        result_with_formula = poststratify(
            s,
            s_weights,
            t,
            t_weights,
            None,
            None,
            True,
            True,
            "add_indicator",
            None,
            None,
            True,
            formula=["a"],
        )["weight"]
        self.assertEqual(result_with_formula, pd.Series([4.0, 4.0, 2.0, 6.0]))

    def test_poststratify_na_action(self) -> None:
        s = pd.DataFrame(
            {
                "a": (1, np.nan, 2),
                "b": ("x", "x", "y"),
            }
        )
        t = s.copy()
        s_weights = pd.Series([1, 1, 1])
        t_weights = pd.Series([2, 3, 4])

        result_add = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
            na_action="add_indicator",
            transformations=None,
        )["weight"]
        self.assertEqual(result_add, t_weights.astype("float64"))

        result_drop = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
            na_action="drop",
            transformations=None,
        )["weight"]
        expected = t_weights.loc[s.dropna().index].astype("float64")
        self.assertEqual(result_drop, expected)

    def test_poststratify_na_drop_warns(self) -> None:
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": (1, np.nan, 2),
                    "id": (1, 2, 3),
                }
            ),
            id_column="id",
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": (1, 2, np.nan),
                    "id": (1, 2, 3),
                }
            ),
            id_column="id",
        )
        self.assertWarnsRegexp(
            "Dropped 1/3 rows of sample",
            sample.adjust,
            target,
            method="poststratify",
            na_action="drop",
            transformations=None,
        )

    def test_poststratify_dropna_empty(self) -> None:
        s = pd.DataFrame({"a": (np.nan, None), "b": (np.nan, None)})
        s_w = pd.Series((1, 2))
        self.assertRaisesRegex(
            ValueError,
            "Dropping rows led to empty",
            poststratify,
            s,
            s_w,
            s,
            s_w,
            na_action="drop",
            transformations=None,
        )

    def test_poststratify_exceptions(self) -> None:
        # column with name weight
        s = pd.DataFrame(
            {
                "weight": (0, 1, 0, 1),
                "c": ("a", "a", "b", "b"),
            },
        )
        s_weights = pd.Series([4, 2, 1, 1])
        t = pd.DataFrame(
            {
                "a": (0, 1, 0, 1),
                "c": ("a", "a", "b", "b"),
            },
        )
        t_weights = pd.Series([4, 2, 2, 8])
        with self.assertRaisesRegex(
            ValueError,
            "weight can't be a name of a column in sample or target when applying poststratify",
        ):
            poststratify(s, s_weights, t, t_weights)
        with self.assertRaisesRegex(
            ValueError,
            "weight can't be a name of a column in sample or target when applying poststratify",
        ):
            poststratify(t, t_weights, s, s_weights)

        # not all sample cells are in target
        s = pd.DataFrame(
            {
                "a": ("x", "y"),
                "b": (0, 1),
            },
        )
        s_weights = pd.Series([1] * 2)
        t = pd.DataFrame(
            {
                "a": ("x", "x", "y"),
                "b": (0, 1, 0),
            },
        )
        t_weights = pd.Series([2] * 3)
        with self.assertRaisesRegex(
            ValueError, "all combinations of cells in sample_df must be in target_df"
        ):
            poststratify(s, s_weights, t, t_weights)

        # Check that strict_matching=False works
        result = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
            strict_matching=False,
        )["weight"]
        self.assertEqual(result, pd.Series([2.0, 0.0]))

        with self.assertRaisesRegex(
            ValueError, "`na_action` must be 'add_indicator' or 'drop'"
        ):
            poststratify(
                sample_df=s,
                sample_weights=s_weights,
                target_df=t,
                target_weights=t_weights,
                na_action="invalid",
            )

        with self.assertRaisesRegex(
            ValueError, "Specify only one of `variables` or `formula`"
        ):
            poststratify(
                sample_df=s,
                sample_weights=s_weights,
                target_df=t,
                target_weights=t_weights,
                variables=["a"],
                formula=["a"],
            )

        result_with_empty_variables_and_formula = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
            variables=[],
            formula=["a"],
        )["weight"]
        self.assertEqual(
            result_with_empty_variables_and_formula,
            pd.Series([4.0, 2.0]),
        )

        with self.assertRaisesRegex(
            ValueError, "`formula` must contain at least one formula string"
        ):
            poststratify(
                sample_df=s,
                sample_weights=s_weights,
                target_df=t,
                target_weights=t_weights,
                formula=[],
            )

        with self.assertRaisesRegex(
            ValueError, "Each element of `formula` must be a string"
        ):
            poststratify(
                sample_df=s,
                sample_weights=s_weights,
                target_df=t,
                target_weights=t_weights,
                formula=["a", 123],  # type: ignore[list-item]
            )

        with self.assertRaisesRegex(
            ValueError, "`formula` must be a string or list of strings"
        ):
            poststratify(
                sample_df=s,
                sample_weights=s_weights,
                target_df=t,
                target_weights=t_weights,
                formula=123,  # type: ignore[arg-type]
            )

        with self.assertRaisesRegex(
            ValueError, "`formula` must be a string or list of strings"
        ):
            poststratify(
                sample_df=s,
                sample_weights=s_weights,
                target_df=t,
                target_weights=t_weights,
                formula=("a",),  # type: ignore[arg-type]
            )

        with self.assertRaisesRegex(
            ValueError, "Formula items must be non-empty strings"
        ):
            poststratify(
                sample_df=s,
                sample_weights=s_weights,
                target_df=t,
                target_weights=t_weights,
                formula=["   "],
            )

        with self.assertRaisesRegex(
            ValueError, "Variable 'missing_var' from `formula` is not present"
        ):
            poststratify(
                sample_df=s,
                sample_weights=s_weights,
                target_df=t,
                target_weights=t_weights,
                formula="missing_var:a",
            )

        with self.assertRaisesRegex(
            ValueError, "Unsupported poststratify formula term"
        ):
            poststratify(
                sample_df=s,
                sample_weights=s_weights,
                target_df=t,
                target_weights=t_weights,
                formula="np.log(a)",
            )

        # `+` (additive) is rejected because poststratify always joint-cells.
        with self.assertRaisesRegex(
            ValueError,
            r"Poststratify formula operator '\+' is not supported",
        ):
            poststratify(
                sample_df=s,
                sample_weights=s_weights,
                target_df=t,
                target_weights=t_weights,
                formula="a + b",
            )

        # `+` inside a list item is also rejected.
        with self.assertRaisesRegex(
            ValueError,
            r"Poststratify formula operator '\+' is not supported",
        ):
            poststratify(
                sample_df=s,
                sample_weights=s_weights,
                target_df=t,
                target_weights=t_weights,
                formula=["a + b"],
            )

        # `*` (main + interaction) is rejected for the same reason.
        with self.assertRaisesRegex(
            ValueError,
            r"Poststratify formula operator '\*' is not supported",
        ):
            poststratify(
                sample_df=s,
                sample_weights=s_weights,
                target_df=t,
                target_weights=t_weights,
                formula="a * b",
            )

        # `+` on the RHS of a `~` is also caught (LHS is ignored).
        with self.assertRaisesRegex(
            ValueError,
            r"Poststratify formula operator '\+' is not supported",
        ):
            poststratify(
                sample_df=s,
                sample_weights=s_weights,
                target_df=t,
                target_weights=t_weights,
                formula="outcome ~ a + b",
            )

        # Mixed `+` inside a dot-expansion expression is rejected BEFORE
        # our internal dot substitution (which also uses `+`).
        with self.assertRaisesRegex(
            ValueError,
            r"Poststratify formula operator '\+' is not supported",
        ):
            poststratify(
                sample_df=s,
                sample_weights=s_weights,
                target_df=t,
                target_weights=t_weights,
                formula="a + . - c",
            )

    def test_chained_adjustment_ipw_then_poststratify_improves_poststrat_covars(
        self,
    ) -> None:
        sample_df = pd.DataFrame(
            {
                "id": [f"s{i}" for i in range(8)],
                "w": np.ones(8),
                "signal": [0, 0, 0, 1, 1, 1, 1, 1],
                "age_group": [
                    "young",
                    "young",
                    "young",
                    "young",
                    "old",
                    "old",
                    "old",
                    "old",
                ],
            }
        )
        target_df = pd.DataFrame(
            {
                "id": [f"t{i}" for i in range(8)],
                "w": np.ones(8),
                "signal": [0, 1, 0, 1, 0, 1, 0, 1],
                "age_group": ["young", "old", "old", "old", "old", "old", "old", "old"],
            }
        )
        sample = Sample.from_frame(sample_df, id_column="id", weight_column="w")
        target = Sample.from_frame(target_df, id_column="id", weight_column="w")

        stage_1 = sample.adjust(
            target,
            method="ipw",
            variables=["signal"],
            transformations=None,
            num_lambdas=1,
        )
        stage_2 = stage_1.adjust(
            method="poststratify",
            variables=["age_group"],
            transformations=None,
        )

        stage_1_asmd = stage_1.covars().asmd()
        stage_2_asmd = stage_2.covars().asmd()
        self.assertLessEqual(
            float(stage_2_asmd.loc["self", "age_group[young]"]),
            float(stage_1_asmd.loc["self", "age_group[young]"]),
        )
        self.assertAlmostEqual(
            float(stage_2_asmd.loc["self", "age_group[young]"]),
            0.0,
            places=10,
        )

    def test_transformations_identity_only_uses_transformed_columns(self) -> None:
        sample_df = pd.DataFrame(
            {
                "age_group": ["young", "young", "old", "old"],
                "region": ["east", "west", "east", "west"],
            }
        )
        target_df = pd.DataFrame(
            {
                "age_group": ["young", "young", "old", "old"],
                "region": ["east", "west", "east", "west"],
            }
        )
        s_weights = pd.Series([1.0, 1.0, 1.0, 1.0])
        t_weights = pd.Series([1.0, 1.0, 1.0, 1.0])

        transformed_only = poststratify(
            sample_df=sample_df,
            sample_weights=s_weights,
            target_df=target_df,
            target_weights=t_weights,
            transformations={"age_group": lambda x: x},
            store_fit_metadata=False,
        )["weight"]
        via_variables = poststratify(
            sample_df=sample_df,
            sample_weights=s_weights,
            target_df=target_df,
            target_weights=t_weights,
            variables=["age_group"],
            transformations=None,
        )["weight"]

        pd.testing.assert_series_equal(transformed_only, via_variables)

    def test_transformations_drop_false_keeps_untransformed_columns(self) -> None:
        sample_df = pd.DataFrame(
            {
                "age_group": ["young", "young", "old", "old"],
                "region": ["east", "west", "east", "west"],
            }
        )
        target_df = pd.DataFrame(
            {
                "age_group": ["young", "young", "old", "old"],
                "region": ["east", "west", "east", "west"],
            }
        )
        s_weights = pd.Series([1.0, 1.0, 1.0, 1.0])
        t_weights = pd.Series([1.0, 1.0, 1.0, 1.0])

        transformed_keep_rest = poststratify(
            sample_df=sample_df,
            sample_weights=s_weights,
            target_df=target_df,
            target_weights=t_weights,
            transformations={"age_group": lambda x: x},
            transformations_drop=False,
            store_fit_metadata=False,
        )["weight"]
        via_both_variables = poststratify(
            sample_df=sample_df,
            sample_weights=s_weights,
            target_df=target_df,
            target_weights=t_weights,
            variables=["age_group", "region"],
            transformations=None,
        )["weight"]

        pd.testing.assert_series_equal(transformed_keep_rest, via_both_variables)

    def test_variables_and_transformations_prioritize_explicit_variables(self) -> None:
        sample_df = pd.DataFrame(
            {
                "age_group": ["young", "young", "old", "old"],
                "region": ["east", "west", "east", "west"],
            }
        )
        target_df = pd.DataFrame(
            {
                "age_group": ["young", "old", "old", "old"],
                "region": ["west", "east", "east", "east"],
            }
        )
        s_weights = pd.Series([1.0, 1.0, 1.0, 1.0])
        t_weights = pd.Series([1.0, 1.0, 1.0, 1.0])

        mixed_arguments = poststratify(
            sample_df=sample_df,
            sample_weights=s_weights,
            target_df=target_df,
            target_weights=t_weights,
            variables=["region"],
            transformations={"age_group": lambda x: x},
            store_fit_metadata=False,
        )["weight"]
        via_region = poststratify(
            sample_df=sample_df,
            sample_weights=s_weights,
            target_df=target_df,
            target_weights=t_weights,
            variables=["region"],
            transformations=None,
        )["weight"]

        pd.testing.assert_series_equal(mixed_arguments, via_region)

    def test_poststratify_drives_asmd_to_zero_on_poststrat_variables(self) -> None:
        sample_df = pd.DataFrame(
            {
                "id": ["1", "2", "3", "4"],
                "w": [1.0, 1.0, 1.0, 1.0],
                "age_group": ["young", "young", "old", "old"],
            }
        )
        target_df = pd.DataFrame(
            {
                "id": ["5", "6", "7", "8"],
                "w": [1.0, 1.0, 1.0, 1.0],
                "age_group": ["young", "old", "old", "old"],
            }
        )
        sample = Sample.from_frame(sample_df, id_column="id", weight_column="w")
        target = Sample.from_frame(target_df, id_column="id", weight_column="w")
        adjusted = sample.adjust(
            target,
            method="poststratify",
            variables=["age_group"],
            transformations=None,
        )

        asmd = adjusted.covars().asmd()
        self.assertAlmostEqual(
            float(asmd.loc["self", "age_group[young]"]), 0.0, places=10
        )
        self.assertAlmostEqual(
            float(asmd.loc["self", "age_group[old]"]), 0.0, places=10
        )

    def test_poststratify_updates_outcome_distribution_toward_target(self) -> None:
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": ["1", "2", "3", "4"],
                    "w": [1.0, 1.0, 1.0, 1.0],
                    "age_group": ["young", "young", "old", "old"],
                    "y": [1.0, 1.0, 10.0, 10.0],
                }
            ),
            id_column="id",
            weight_column="w",
            outcome_columns=["y"],
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": ["5", "6", "7", "8"],
                    "w": [1.0, 1.0, 1.0, 1.0],
                    "age_group": ["young", "old", "old", "old"],
                    "y": [1.0, 10.0, 10.0, 10.0],
                }
            ),
            id_column="id",
            weight_column="w",
            outcome_columns=["y"],
        )

        adjusted = sample.adjust(
            target,
            method="poststratify",
            variables=["age_group"],
            transformations=None,
        )
        means = adjusted.outcomes().mean()
        self.assertAlmostEqual(
            float(means.loc["self", "y"]), float(means.loc["target", "y"])
        )
        self.assertNotAlmostEqual(
            float(means.loc["self", "y"]),
            float(means.loc["unadjusted", "y"]),
        )

    def test_poststratify_on_continuous_variable_with_default_transformations(
        self,
    ) -> None:
        sample_df = pd.DataFrame(
            {
                "x": [10.0, 20.0, 30.0, 40.0],
            }
        )
        target_df = pd.DataFrame(
            {
                "x": [12.0, 18.0, 33.0, 45.0],
            }
        )
        s_weights = pd.Series([1.0, 1.0, 1.0, 1.0])
        t_weights = pd.Series([1.0, 1.0, 1.0, 1.0])

        with self.assertRaisesRegex(
            ValueError, "Cannot normalise weights because their sum is zero"
        ):
            poststratify(
                sample_df=sample_df,
                sample_weights=s_weights,
                target_df=target_df,
                target_weights=t_weights,
                variables=["x"],
                transformations="default",
                strict_matching=False,
            )
