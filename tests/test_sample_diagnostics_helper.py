# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import pandas as pd

from balance.sample_class import _concat_metric_val_var


def test_concat_sequence_lengths_match() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])  # pyre-ignore[6]

    result = _concat_metric_val_var(
        diagnostics,
        "size",
        [1, 2],
        ["sample_obs", "target_obs"],
    )

    expected = pd.DataFrame(
        {
            "metric": ["size", "size"],
            "val": [1, 2],
            "var": ["sample_obs", "target_obs"],
        }
    )

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected, check_dtype=False
    )


def test_concat_single_row() -> None:
    base = pd.DataFrame({"metric": ["existing"], "val": [10], "var": ["foo"]})

    result = _concat_metric_val_var(base, "adjustment_method", [0], ["ipw"])

    expected = pd.DataFrame(
        {
            "metric": ["existing", "adjustment_method"],
            "val": [10, 0],
            "var": ["foo", "ipw"],
        }
    )

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected, check_dtype=False
    )


def test_concat_multiple_rows() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])  # pyre-ignore[6]

    result = _concat_metric_val_var(
        diagnostics,
        "weights_diagnostics",
        [0.5, 0.6],
        ["mean", "median"],
    )

    expected = pd.DataFrame(
        {
            "metric": ["weights_diagnostics", "weights_diagnostics"],
            "val": [0.5, 0.6],
            "var": ["mean", "median"],
        }
    )

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected, check_dtype=False
    )


def test_concat_scalar_pair() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])  # pyre-ignore[6]

    result = _concat_metric_val_var(diagnostics, "model_glance", [0.1], ["lambda"])

    expected = pd.DataFrame(
        {"metric": ["model_glance"], "val": [0.1], "var": ["lambda"]}
    )

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected, check_dtype=False
    )


def test_concat_string_values() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])  # pyre-ignore[6]

    result = _concat_metric_val_var(
        diagnostics, "ipw_solver", ["lbfgs", "saga"], ["solver", "fused"]
    )

    expected = pd.DataFrame(
        {
            "metric": ["ipw_solver", "ipw_solver"],
            "val": ["lbfgs", "saga"],
            "var": ["solver", "fused"],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_concat_empty_list_returns_copy() -> None:
    diagnostics = pd.DataFrame(
        {"metric": ["existing"], "val": [1], "var": ["foo"]},
        index=[5],  # pyre-ignore[6]
    )

    result = _concat_metric_val_var(diagnostics, "size", [], [])

    pd.testing.assert_frame_equal(result, diagnostics)


def test_concat_bytes_values() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])  # pyre-ignore[6]

    result = _concat_metric_val_var(diagnostics, "encoding", [b"abc", b"def"], [1, 2])

    expected = pd.DataFrame(
        {
            "metric": ["encoding", "encoding"],
            "val": [b"abc", b"def"],
            "var": [1, 2],
        }
    )

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_concat_retains_existing_columns() -> None:
    diagnostics = pd.DataFrame(
        {
            "metric": ["existing"],
            "val": [10],
            "var": ["foo"],
            "note": ["keep"],
        }
    )

    result = _concat_metric_val_var(diagnostics, "size", [1], ["bar"])

    expected = pd.DataFrame(
        {
            "metric": ["existing", "size"],
            "val": [10, 1],
            "var": ["foo", "bar"],
            "note": ["keep", np.nan],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_concat_respects_empty_frame_column_order() -> None:
    diagnostics = pd.DataFrame(
        columns=["metric", "val", "var", "note"]
    )  # pyre-ignore[6]

    result = _concat_metric_val_var(diagnostics, "size", [1], ["bar"])

    expected = pd.DataFrame(
        {
            "metric": ["size"],
            "val": [1],
            "var": ["bar"],
            "note": [pd.NA],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_concat_appends_when_no_columns_present() -> None:
    diagnostics = pd.DataFrame()

    result = _concat_metric_val_var(diagnostics, "size", [1, 2], ["bar", "baz"])

    expected = pd.DataFrame(
        {"metric": ["size", "size"], "val": [1, 2], "var": ["bar", "baz"]}
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_concat_accepts_series() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])  # pyre-ignore[6]

    vals = pd.Series([10, 20], index=["x", "y"])
    vars_series = pd.Series(["alpha", "beta"], index=["x", "y"])

    result = _concat_metric_val_var(
        diagnostics, "series_metric", list(vals), list(vars_series)
    )

    expected = pd.DataFrame(
        {
            "metric": ["series_metric", "series_metric"],
            "val": [10, 20],
            "var": ["alpha", "beta"],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_concat_accepts_numpy_arrays() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])  # pyre-ignore[6]

    vals = np.array([1.5, 2.5])
    vars_arr = np.array(["mean", "std"])

    result = _concat_metric_val_var(
        diagnostics, "array_metric", list(vals), list(vars_arr)
    )

    expected = pd.DataFrame(
        {
            "metric": ["array_metric", "array_metric"],
            "val": [1.5, 2.5],
            "var": ["mean", "std"],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_concat_preserves_nonstandard_column_order() -> None:
    diagnostics = pd.DataFrame(
        columns=["metric", "note", "val", "var"]
    )  # pyre-ignore[6]

    result = _concat_metric_val_var(diagnostics, "size", [1, 2], ["bar", "baz"])

    expected = pd.DataFrame(
        {
            "metric": ["size", "size"],
            "note": [pd.NA, pd.NA],
            "val": [1, 2],
            "var": ["bar", "baz"],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_concat_mismatched_lengths_raises_error() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])  # pyre-ignore[6]

    try:
        _concat_metric_val_var(diagnostics, "size", [1, 2, 3], ["a", "b"])
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        assert "vals and vars must have the same length" in str(e)
