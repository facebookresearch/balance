# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import pandas as pd
import pytest

from balance.sample_class import _concat_metric_val_var


def test_concat_sequence_lengths_match() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

    result = _concat_metric_val_var(
        diagnostics, "size", [1, 2], ["sample_obs", "target_obs"]
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


def test_concat_broadcast_scalar_var() -> None:
    base = pd.DataFrame({"metric": ["existing"], "val": [10], "var": ["foo"]})

    result = _concat_metric_val_var(base, "adjustment_method", [0, 1], "ipw")

    expected = pd.DataFrame(
        {
            "metric": ["existing", "adjustment_method", "adjustment_method"],
            "val": [10, 0, 1],
            "var": ["foo", "ipw", "ipw"],
        }
    )

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected, check_dtype=False
    )


def test_concat_broadcast_scalar_val() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

    result = _concat_metric_val_var(
        diagnostics, "weights_diagnostics", 0.5, ["mean", "median"]
    )

    expected = pd.DataFrame(
        {
            "metric": ["weights_diagnostics", "weights_diagnostics"],
            "val": [0.5, 0.5],
            "var": ["mean", "median"],
        }
    )

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected, check_dtype=False
    )


def test_concat_scalars() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

    result = _concat_metric_val_var(diagnostics, "model_glance", 0.1, "lambda")

    expected = pd.DataFrame(
        {"metric": ["model_glance"], "val": [0.1], "var": ["lambda"]}
    )

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected, check_dtype=False
    )


def test_concat_treats_strings_as_scalars() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

    result = _concat_metric_val_var(
        diagnostics, "ipw_solver", "lbfgs", ["solver", "fused"]
    )

    expected = pd.DataFrame(
        {
            "metric": ["ipw_solver", "ipw_solver"],
            "val": ["lbfgs", "lbfgs"],
            "var": ["solver", "fused"],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_concat_mismatched_sequence_lengths() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

    with pytest.raises(
        ValueError, match="val and var must have the same length when sequences"
    ):
        _concat_metric_val_var(diagnostics, "size", [1, 2, 3], ["a", "b"])


def test_concat_preserves_index_on_empty_rows() -> None:
    diagnostics = pd.DataFrame(
        {"metric": ["existing"], "val": [1], "var": ["foo"]}, index=[5]
    )

    result = _concat_metric_val_var(diagnostics, "size", [], [])

    pd.testing.assert_frame_equal(result, diagnostics)


def test_concat_accepts_generators_and_broadcasts() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

    val_generator = (val for val in (1, 2, 3))
    result = _concat_metric_val_var(diagnostics, "size", val_generator, "foo")

    expected = pd.DataFrame(
        {
            "metric": ["size", "size", "size"],
            "val": [1, 2, 3],
            "var": ["foo", "foo", "foo"],
        }
    )

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_concat_treats_bytes_as_scalar() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

    result = _concat_metric_val_var(diagnostics, "encoding", b"abc", [1, 2])

    expected = pd.DataFrame(
        {
            "metric": ["encoding", "encoding"],
            "val": [b"abc", b"abc"],
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
    diagnostics = pd.DataFrame(columns=["metric", "val", "var", "note"])

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


def test_concat_accepts_pandas_series_inputs() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

    vals = pd.Series([10, 20], index=["x", "y"])
    vars = pd.Series(["alpha", "beta"], index=["x", "y"])

    result = _concat_metric_val_var(diagnostics, "series_metric", vals, vars)

    expected = pd.DataFrame(
        {
            "metric": ["series_metric", "series_metric"],
            "val": [10, 20],
            "var": ["alpha", "beta"],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_concat_preserves_nonstandard_column_order() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "note", "val", "var"])

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


def test_concat_accepts_numpy_arrays() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

    vals = np.array([1.5, 2.5])
    vars = np.array(["mean", "std"])

    result = _concat_metric_val_var(diagnostics, "array_metric", vals, vars)

    expected = pd.DataFrame(
        {
            "metric": ["array_metric", "array_metric"],
            "val": [1.5, 2.5],
            "var": ["mean", "std"],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)
