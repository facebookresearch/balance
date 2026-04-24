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

    # TODO: test chained adjustment (IPW → poststratify) — the two-stage pattern
    #   from balance notebook v03:
    #       sample_with_target = sample.set_target(target)
    #       adjust_stage_1 = sample_with_target.adjust(method="ipw")
    #       adjust_stage_2 = adjust_stage_1.adjust(method="poststratify")
    #   Verify ASMD improves at each stage.

    # TODO: test transformations as a dict with identity lambdas — verify only
    #   the named column is used for cell definition (others are dropped when
    #   transformations_drop=True, the default).
    #   Example from balance notebook v03:
    #       transformations = {"age_group": lambda x: x}
    #       adjusted = ipw_adjusted.adjust(method="poststratify",
    #                                      transformations=transformations)

    # TODO: test transformations_drop=False — verify that columns NOT in the
    #   transformations dict are kept (not dropped) when this flag is set.

    # TODO: test interaction between variables= and transformations= — what
    #   happens when both are provided? Which takes precedence?

    # TODO: test ASMD after poststratify — verify that ASMD on the PS variables
    #   reaches 0 (or near 0) after adjustment.
    #   Example pattern from balance notebook v03:
    #       adjusted.covars().asmd().T

    # TODO: test outcomes after poststratify — verify outcome distributions
    #   shift appropriately when accessed via .outcomes().
    #   Example pattern from balance notebook v03:
    #       adjusted.outcomes().plot()

    # TODO: test poststratify with continuous (non-categorical) variables —
    #   what happens when you PS on a numeric column without binning? The
    #   default transformations handle this via quantize/fct_lump, but explicit
    #   tests for this edge case are missing.
