# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

import io
import re
import sys

import unittest
from contextlib import contextmanager
from typing import Any, Union

import numpy as np
import pandas as pd


def _assert_frame_equal_lazy(
    x: pd.DataFrame, y: pd.DataFrame, lazy: bool = True
) -> None:
    """Wrapper around pd.testing.assert_frame_equal, which transforms the
    dataframes to ignore some errors.

    Ignores order of columns

    Args:
        x (pd.DataFrame): DataFrame to compare
        y (pd.DataFrame): DataFrame to compare
        lazy (bool, optional): Should Ignores be applied. Defaults to True.

    Returns:
        None.
    """
    if lazy:
        x = x.sort_index(axis=0).sort_index(axis=1)
        y = y.sort_index(axis=0).sort_index(axis=1)

    return pd.testing.assert_frame_equal(x, y)


def _assert_index_equal_lazy(x: pd.Index, y: pd.Index, lazy: bool = True) -> None:
    """
    Wrapper around pd.testing.assert_index_equal which transforms the
    dataindexs to ignore some errors.

    Ignores:
        - order of entries

    Args:
        x (pd.Index): Index to compare
        y (pd.Index): Index to compare
        lazy (bool, optional): Should Ignores be applied. Defaults to True.
    """
    if lazy:
        x = x.sort_values()
        y = y.sort_values()

    return pd.testing.assert_index_equal(x, y)


@contextmanager
def _capture_output():
    redirect_out, redirect_err = io.StringIO(), io.StringIO()
    original_out, original_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = redirect_out, redirect_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = original_out, original_err


class BalanceTestCase(unittest.TestCase):
    # Some Warns
    def assertWarns(self, callable, *args, **kwargs) -> None:
        with self.assertLogs(level="NOTSET") as cm:
            callable(*args, **kwargs)
            self.assertTrue(len(cm.output) > 0, "No warning produced.")

    def assertNotWarns(self, callable, *args, **kwargs) -> None:
        output = None
        try:
            with self.assertLogs() as cm:
                callable(*args, **kwargs)
                output = cm
        except AssertionError:
            return
        raise AssertionError(f"Warning produced {output.output}.")

    def assertWarnsRegexp(self, regexp, callable, *args, **kwargs) -> None:
        with self.assertLogs(level="NOTSET") as cm:
            callable(*args, **kwargs)
            self.assertTrue(
                any((re.search(regexp, c) is not None) for c in cm.output),
                f"Warning {cm.output} does not match regex {regexp}.",
            )

    def assertNotWarnsRegexp(self, regexp, callable, *args, **kwargs) -> None:
        with self.assertLogs(level="NOTSET") as cm:
            callable(*args, **kwargs)
            self.assertFalse(
                any((re.search(regexp, c) is not None) for c in cm.output),
                f"Warning {cm.output} matches regex {regexp}.",
            )

    # Some Equal
    def assertEqual(
        self,
        first: Union[np.ndarray, pd.DataFrame, pd.Index, pd.Series, Any],
        second: Union[np.ndarray, pd.DataFrame, pd.Index, pd.Series, Any],
        msg: Any = ...,
        **kwargs,
    ) -> None:
        """
        Check if first and second are equal.
        Uses np.testing.assert_array_equal for np.ndarray,
        _assert_frame_equal_lazy for pd.DataFrame,
        assert_series_equal for  pd.DataFrame,
        _assert_index_equal_lazy for pd.Index,
        or unittest.TestCase.assertEqual otherwise.

        Args:
            first (Union[np.ndarray, pd.DataFrame, pd.Index, pd.Series]): first element to compare.
            second (Union[np.ndarray, pd.DataFrame, pd.Index, pd.Series]): second element to compare.
            msg (Any, optional): The error message on failure.
        """
        lazy: bool = kwargs.get("lazy", False)
        if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
            np.testing.assert_array_equal(first, second, **kwargs)
        elif isinstance(first, pd.DataFrame) or isinstance(second, pd.DataFrame):
            _assert_frame_equal_lazy(
                first,
                second,
                lazy,
            )
        elif isinstance(first, pd.Series) or isinstance(second, pd.Series):
            pd.testing.assert_series_equal(first, second)
        elif isinstance(first, pd.Index) or isinstance(second, pd.Index):
            _assert_index_equal_lazy(first, second, lazy)
        else:
            super().assertEqual(first, second, msg=msg, **kwargs)

    # Some Prints
    def assertPrints(self, callable, *args, **kwargs) -> None:
        with _capture_output() as (out, err):
            callable(*args, **kwargs)
            out, err = out.getvalue(), err.getvalue()
            self.assertTrue((len(out) + len(err)) > 0, "No printed output.")

    def assertNotPrints(self, callable, *args, **kwargs) -> None:
        with _capture_output() as (out, err):
            callable(*args, **kwargs)
            out, err = out.getvalue(), err.getvalue()
            self.assertTrue(
                (len(out) + len(err)) == 0,
                f"Printed output is longer than 0: {(out, err)}.",
            )

    def assertPrintsRegexp(self, regexp, callable, *args, **kwargs) -> None:
        with _capture_output() as (out, err):
            callable(*args, **kwargs)
            out, err = out.getvalue(), err.getvalue()
            self.assertTrue(
                any((re.search(regexp, o) is not None) for o in (out, err)),
                f"Printed output {(out, err)} does not match regex {regexp}.",
            )

    def assertNotPrintsRegexp(self, regexp, callable, *args, **kwargs) -> None:
        with _capture_output() as (out, err):
            callable(*args, **kwargs)
            out, err = out.getvalue(), err.getvalue()
            self.assertFalse(
                any((re.search(regexp, o) is not None) for o in (out, err)),
                f"Printed output {(out, err)} matches regex {regexp}.",
            )
