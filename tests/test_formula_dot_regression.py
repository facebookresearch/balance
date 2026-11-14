# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import pandas as pd
from balance.util import _assert_type
from balance.utils.model_matrix import build_model_matrix, dot_expansion, model_matrix


class FormulaDotRegressionTest(unittest.TestCase):
    def test_malformed_formula_tokens_raise_actionable_value_error(self) -> None:
        for formula in ("I(x * 0.5", "x + (.", "x\n  .\n ."):
            with self.subTest(formula=formula):
                with self.assertRaisesRegex(ValueError, "Invalid formula.*token"):
                    dot_expansion(formula, ["x"])
                with self.assertRaisesRegex(ValueError, "Invalid formula.*token"):
                    build_model_matrix(pd.DataFrame({"x": [1.0, 2.0]}), formula=formula)

    def test_does_not_expand_a_dot_attached_to_a_name(self) -> None:
        for formula in ("a. + b", "y ~ a. + b", "a . + b"):
            with self.subTest(formula=formula):
                self.assertEqual(dot_expansion(formula, ["x", "z"]), formula)
        self.assertEqual(dot_expansion("y ~ . + b", ["x", "z"]), "y ~ (x+z) + b")

    def test_does_not_expand_a_dot_following_an_expression(self) -> None:
        for formula in (
            "(x). + y",
            "x[0]. + y",
            "x(). + y",
            "'x'. + y",
            "1 . + y",
            ". 1 + x",
        ):
            with self.subTest(formula=formula):
                self.assertEqual(dot_expansion(formula, ["x", "y"]), formula)
        self.assertEqual(dot_expansion("(. + y)", ["x"]), "((x) + y)")
        self.assertEqual(dot_expansion("I(x * .5) + .", ["x"]), "I(x * .5) + (x)")

    def test_preserves_decimal_attribute_and_quoted_dots(self) -> None:
        self.assertEqual(dot_expansion("I(x * 0.5)", ["x"]), "I(x * 0.5)")
        self.assertEqual(dot_expansion("np.sqrt(x)", ["x"]), "np.sqrt(x)")
        self.assertEqual(dot_expansion("Q('x.y') + .", ["x"]), "Q('x.y') + (x)")
        frame = pd.DataFrame({"x": [1.0, 4.0]})
        result = build_model_matrix(frame, formula="I(x * 0.5)")
        self.assertEqual(result["model_matrix"].iloc[:, 0].tolist(), [0.5, 2.0])
        result = build_model_matrix(frame, formula="np.sqrt(x)")
        self.assertEqual(result["model_matrix"].iloc[:, 0].tolist(), [1.0, 2.0])

    def test_standalone_dot_without_variables_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "standalone.*without variables"):
            dot_expansion("y ~ .", [])
        with self.assertRaisesRegex(ValueError, "standalone.*without variables"):
            build_model_matrix(pd.DataFrame(index=pd.Index([0, 1])), formula=".")

    def test_empty_variables_keep_explicit_formulas_and_inferred_columns(self) -> None:
        self.assertEqual(dot_expansion("0", []), "0")
        self.assertEqual(dot_expansion("I(x * 0.5)", []), "I(x * 0.5)")
        empty = build_model_matrix(pd.DataFrame(index=pd.Index([0, 1])), formula="0")
        self.assertEqual(empty["model_matrix"].shape, (2, 0))
        result = model_matrix(
            pd.DataFrame({"x": [1.0, 2.0]}), variables=[], formula="."
        )
        self.assertEqual(
            _assert_type(result["sample"], pd.DataFrame).values.tolist(), [[1.0], [2.0]]
        )
