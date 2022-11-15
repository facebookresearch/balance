# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

import logging
import sys

import balance.testutil

import numpy as np
import pandas as pd


class TestTestUtil(
    balance.testutil.BalanceTestCase,
):
    def test_testutil(self):
        # _assert_frame_equal_lazy
        self.assertRaises(
            AssertionError,
            balance.testutil._assert_frame_equal_lazy,
            pd.DataFrame({"a": (1, 2, 3)}),
            pd.DataFrame({"a": (1, 2, 4)}),
        )

        self.assertRaises(
            AssertionError,
            balance.testutil._assert_frame_equal_lazy,
            pd.DataFrame({"a": (1, 2, 3), "b": (1, 2, 3)}),
            pd.DataFrame({"a": (1, 2, 4), "c": (1, 2, 3)}),
        )

        a = pd.DataFrame({"a": (1, 2, 3), "b": (4, 5, 6)})
        b = pd.DataFrame({"a": (1, 2, 3), "b": (4, 5, 6)}, columns=("b", "a"))

        # Doesn't raise an error
        balance.testutil._assert_frame_equal_lazy(a, b)

        self.assertRaises(
            AssertionError, balance.testutil._assert_frame_equal_lazy, a, b, False
        )

        # _assert_index_equal_lazy
        self.assertRaises(
            AssertionError,
            balance.testutil._assert_index_equal_lazy,
            pd.Index([1, 2, 3]),
            pd.Index([1, 2, 4]),
        )

        a = pd.Index([1, 2, 3])
        b = pd.Index([1, 3, 2])
        # Doesn't raise an error
        balance.testutil._assert_index_equal_lazy(a, b)
        self.assertRaises(
            AssertionError, balance.testutil._assert_index_equal_lazy, a, b, False
        )


class TestTestUtil_BalanceTestCase_Equal(
    balance.testutil.BalanceTestCase,
):
    def test_additional_equality_tests_mixin(self):
        # np.array
        self.assertRaises(AssertionError, self.assertEqual, 1, 2)
        self.assertRaises(
            AssertionError, self.assertEqual, np.array((1, 2)), np.array((2, 1))
        )

        # Does not raise
        self.assertEqual(1, 1)
        self.assertEqual(np.array((1, 2)), np.array((1, 2)))

        # pd.DataFrames
        # The default is for non-lazy testing of pandas DataFrames
        a = pd.DataFrame({"a": (1, 2, 3), "b": (4, 5, 6)})
        b = pd.DataFrame({"a": (1, 2, 3), "b": (4, 5, 6)}, columns=("b", "a"))

        # Does raise an error by default or if lazy=False
        self.assertRaises(AssertionError, self.assertEqual, a, b)
        self.assertRaises(AssertionError, self.assertEqual, a, b, lazy=False)

        # Doesn't raise an error
        self.assertEqual(a, b, lazy=True)

        # pd.Series
        self.assertEqual(pd.Series([1, 2]), pd.Series([1, 2]))
        self.assertRaises(
            AssertionError, self.assertEqual, pd.Series([1, 2]), pd.Series([2, 1])
        )

        #  Indices
        self.assertEqual(pd.Index((1, 2)), pd.Index((1, 2)))
        self.assertRaises(
            AssertionError, self.assertEqual, pd.Index((1, 2)), pd.Index((2, 1))
        )
        self.assertRaises(
            AssertionError,
            self.assertEqual,
            pd.Index((1, 2)),
            pd.Index((2, 1)),
            lazy=False,
        )
        self.assertEqual(pd.Index((1, 2)), pd.Index((2, 1)), lazy=True)

        # Other types
        self.assertEqual("a", "a")
        self.assertRaises(AssertionError, self.assertEqual, "a", "b")


class TestTestUtil_BalanceTestCase_Warns(
    balance.testutil.BalanceTestCase,
):
    def test_unit_test_warning_mixin(self):
        logger = logging.getLogger(__package__)

        self.assertWarns(lambda: logger.warning("test"))
        self.assertNotWarns(lambda: "x")

        self.assertWarnsRegexp("abc", lambda: logger.warning("abcde"))
        self.assertRaises(
            AssertionError,
            self.assertWarnsRegexp,
            "abcdef",
            lambda: logger.warning("abcde"),
        )

        self.assertNotWarnsRegexp("abcdef", lambda: logger.warning("abcde"))


class TestTestUtil_BalanceTestCase_Print(
    balance.testutil.BalanceTestCase,
):
    def test_unit_test_print_mixin(self):
        self.assertPrints(lambda: print("x"))
        self.assertNotPrints(lambda: "x")

        self.assertPrintsRegexp("abc", lambda: print("abcde"))
        self.assertRaises(
            AssertionError, self.assertPrintsRegexp, "abcdef", lambda: print("abcde")
        )

        # assertPrintsRegexp() doesn't necessarily work with logging.warning(),
        # as logging handlers can change (e.g. in PyTest)
        self.assertPrintsRegexp("abc", lambda: print("abcde", file=sys.stderr))
