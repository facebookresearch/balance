# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import pandas as pd
from balance.sample_class import Sample
from balance.summary_utils import (
    _build_diagnostics,
    _build_summary,
    _concat_metric_val_var,
    _resolve_adjustment_failure_metadata,
)


def test_concat_sequence_lengths_match() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

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
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

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
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

    result = _concat_metric_val_var(diagnostics, "model_glance", [0.1], ["lambda"])

    expected = pd.DataFrame(
        {"metric": ["model_glance"], "val": [0.1], "var": ["lambda"]}
    )

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected, check_dtype=False
    )


def test_concat_string_values() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

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

    result = result.reset_index(drop=True)
    if "note" in result.columns:
        result["note"] = result["note"].where(~result["note"].isna(), np.nan)
    if "note" in expected.columns:
        expected["note"] = expected["note"].where(~expected["note"].isna(), np.nan)

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
    )


def test_concat_empty_list_returns_copy() -> None:
    diagnostics = pd.DataFrame(
        {"metric": ["existing"], "val": [1], "var": ["foo"]}, index=pd.Index([5])
    )

    result = _concat_metric_val_var(diagnostics, "size", [], [])

    pd.testing.assert_frame_equal(result, diagnostics)


def test_concat_bytes_values() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

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

    result = result.reset_index(drop=True)
    if "note" in result.columns:
        result["note"] = result["note"].where(~result["note"].isna(), np.nan)
    if "note" in expected.columns:
        expected["note"] = expected["note"].where(~expected["note"].isna(), np.nan)

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
    )


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


def test_concat_accepts_series() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

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
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

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


def test_concat_mismatched_lengths_raises_error() -> None:
    diagnostics = pd.DataFrame(columns=["metric", "val", "var"])

    try:
        _concat_metric_val_var(diagnostics, "size", [1, 2, 3], ["a", "b"])
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        assert "vals and vars must have the same length" in str(e)


def test_build_summary_matches_sample_summary() -> None:
    """Verify _build_summary produces the same output as Sample.summary()."""
    # pyrefly: ignore [bad-argument-type]
    np.random.seed(42)
    sample = Sample.from_frame(
        pd.DataFrame(
            {
                "a": np.random.randn(100),
                "id": range(100),
                "w": [1.0] * 100,
            }
        ),
        weight_column="w",
    )
    target = Sample.from_frame(
        pd.DataFrame(
            {
                "a": np.random.randn(100) + 1,
                "id": range(100, 200),
                "w": [1.0] * 100,
            }
        ),
        weight_column="w",
    )
    adjusted = sample.set_target(target).adjust(method="ipw", max_de=1.5)

    # Call Sample.summary() to get the reference output
    expected = adjusted.summary()

    # Reproduce the same inputs and call _build_summary directly
    covars_asmd = adjusted.covars().asmd()
    covars_kld = adjusted.covars().kld(aggregate_by_main_covar=True)
    asmd_improvement_pct = 100 * adjusted.covars().asmd_improvement()
    quick_adj = adjusted._quick_adjustment_details(adjusted._df.shape[0])
    de, ess, essp = adjusted._design_effect_diagnostics(adjusted._df.shape[0])

    result = _build_summary(
        is_adjusted=adjusted.is_adjusted(),
        has_target=adjusted.has_target(),
        covars_asmd=covars_asmd,
        covars_kld=covars_kld,
        asmd_improvement_pct=asmd_improvement_pct,
        quick_adjustment_details=quick_adj,
        design_effect=de,
        effective_sample_size=ess,
        effective_sample_proportion=essp,
        model_dict=adjusted.model,
        outcome_means=None,
    )

    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"


def test_build_diagnostics_matches_sample_diagnostics() -> None:
    """Verify _build_diagnostics produces the same output as Sample.diagnostics()."""
    sample = Sample.from_frame(
        pd.DataFrame({"id": ["1", "2"], "x": [0, 1], "weight": [1.0, 2.0]}),
        id_column="id",
        weight_column="weight",
        standardize_types=False,
    )
    target = Sample.from_frame(
        pd.DataFrame({"id": ["3", "4"], "x": [0, 1], "weight": [1.0, 1.0]}),
        id_column="id",
        weight_column="weight",
        standardize_types=False,
    )
    adjusted = sample.set_target(target).adjust(method="null")

    expected = adjusted.diagnostics()

    result = _build_diagnostics(
        covars_df=adjusted.covars().df,
        # pyrefly: ignore [unsupported-operation]
        target_covars_df=adjusted._links["target"].covars().df,
        weights_summary=adjusted.weights().summary(),
        model_dict=adjusted.model,
        covars_asmd=adjusted.covars().asmd(),
        covars_asmd_main=adjusted.covars().asmd(aggregate_by_main_covar=True),
        outcome_columns=adjusted._outcome_columns,
        weights_impact_on_outcome_method="t_test",
        weights_impact_on_outcome_conf_level=0.95,
        outcome_impact=None,
    )

    pd.testing.assert_frame_equal(result, expected)


def test_build_diagnostics_with_ipw_matches_sample_diagnostics() -> None:
    """Verify _build_diagnostics matches Sample.diagnostics() for IPW adjustment."""
    # pyrefly: ignore [bad-argument-type]
    np.random.seed(42)
    sample = Sample.from_frame(
        pd.DataFrame(
            {
                "a": np.random.randn(100),
                "id": range(100),
                "w": [1.0] * 100,
            }
        ),
        weight_column="w",
    )
    target = Sample.from_frame(
        pd.DataFrame(
            {
                "a": np.random.randn(100) + 1,
                "id": range(100, 200),
                "w": [1.0] * 100,
            }
        ),
        weight_column="w",
    )
    adjusted = sample.set_target(target).adjust(method="ipw", max_de=1.5)

    expected = adjusted.diagnostics()

    result = _build_diagnostics(
        covars_df=adjusted.covars().df,
        # pyrefly: ignore [unsupported-operation]
        target_covars_df=adjusted._links["target"].covars().df,
        weights_summary=adjusted.weights().summary(),
        model_dict=adjusted.model,
        covars_asmd=adjusted.covars().asmd(),
        covars_asmd_main=adjusted.covars().asmd(aggregate_by_main_covar=True),
        outcome_columns=adjusted._outcome_columns,
        weights_impact_on_outcome_method="t_test",
        weights_impact_on_outcome_conf_level=0.95,
        outcome_impact=None,
    )

    pd.testing.assert_frame_equal(result, expected)


def test_build_diagnostics_model_dict_none_shows_unknown_method() -> None:
    """When model_dict is None, _build_diagnostics should include 'unknown' adjustment method."""
    sample = Sample.from_frame(
        pd.DataFrame({"id": ["1", "2"], "x": [0, 1], "weight": [1.0, 2.0]}),
        id_column="id",
        weight_column="weight",
        standardize_types=False,
    )
    target = Sample.from_frame(
        pd.DataFrame({"id": ["3", "4"], "x": [0, 1], "weight": [1.0, 1.0]}),
        id_column="id",
        weight_column="weight",
        standardize_types=False,
    )
    adjusted = sample.set_target(target).adjust(method="null")

    result = _build_diagnostics(
        covars_df=adjusted.covars().df,
        # pyrefly: ignore [unsupported-operation]
        target_covars_df=adjusted._links["target"].covars().df,
        weights_summary=adjusted.weights().summary(),
        model_dict=None,
        covars_asmd=adjusted.covars().asmd(),
        covars_asmd_main=adjusted.covars().asmd(aggregate_by_main_covar=True),
        outcome_columns=adjusted._outcome_columns,
        weights_impact_on_outcome_method="t_test",
        weights_impact_on_outcome_conf_level=0.95,
        outcome_impact=None,
    )

    method_rows = result[result["metric"] == "adjustment_method"]
    assert len(method_rows) == 1
    assert method_rows["var"].iloc[0] == "unknown"


def test_build_diagnostics_uses_model_adjustment_failure_metadata() -> None:
    """_build_diagnostics should propagate adjustment failure metadata from model_dict."""
    sample = Sample.from_frame(
        pd.DataFrame({"id": ["1", "2"], "x": [0, 1], "weight": [1.0, 2.0]}),
        id_column="id",
        weight_column="weight",
        standardize_types=False,
    )
    target = Sample.from_frame(
        pd.DataFrame({"id": ["3", "4"], "x": [0, 1], "weight": [1.0, 1.0]}),
        id_column="id",
        weight_column="weight",
        standardize_types=False,
    )
    adjusted = sample.set_target(target).adjust(method="null")

    model_dict = {
        "method": "null",
        "adjustment_failure": 1,
        "adjustment_failure_reason": "boom",
    }

    result = _build_diagnostics(
        covars_df=adjusted.covars().df,
        # pyrefly: ignore [unsupported-operation]
        target_covars_df=adjusted._links["target"].covars().df,
        weights_summary=adjusted.weights().summary(),
        model_dict=model_dict,
        covars_asmd=adjusted.covars().asmd(),
        covars_asmd_main=adjusted.covars().asmd(aggregate_by_main_covar=True),
        outcome_columns=adjusted._outcome_columns,
        weights_impact_on_outcome_method="t_test",
        weights_impact_on_outcome_conf_level=0.95,
        outcome_impact=None,
    )

    failure_rows = result[result["metric"] == "adjustment_failure"]
    assert len(failure_rows) == 1
    assert int(failure_rows["val"].iloc[0]) == 1

    reason_rows = result[result["metric"] == "adjustment_failure_reason"]
    assert len(reason_rows) == 1
    assert reason_rows["val"].iloc[0] == "boom"


def test_build_diagnostics_omits_reason_when_failure_is_zero() -> None:
    """Failure reason should not be emitted when adjustment_failure resolves to zero."""
    sample = Sample.from_frame(
        pd.DataFrame({"id": ["1", "2"], "x": [0, 1], "weight": [1.0, 2.0]}),
        id_column="id",
        weight_column="weight",
        standardize_types=False,
    )
    target = Sample.from_frame(
        pd.DataFrame({"id": ["3", "4"], "x": [0, 1], "weight": [1.0, 1.0]}),
        id_column="id",
        weight_column="weight",
        standardize_types=False,
    )
    adjusted = sample.set_target(target).adjust(method="null")

    model_dict = {
        "method": "null",
        "adjustment_failure": 0,
        "adjustment_failure_reason": "should-not-appear",
    }

    result = _build_diagnostics(
        covars_df=adjusted.covars().df,
        # pyrefly: ignore [unsupported-operation]
        target_covars_df=adjusted._links["target"].covars().df,
        weights_summary=adjusted.weights().summary(),
        model_dict=model_dict,
        covars_asmd=adjusted.covars().asmd(),
        covars_asmd_main=adjusted.covars().asmd(aggregate_by_main_covar=True),
        outcome_columns=adjusted._outcome_columns,
        weights_impact_on_outcome_method="t_test",
        weights_impact_on_outcome_conf_level=0.95,
        outcome_impact=None,
    )

    failure_rows = result[result["metric"] == "adjustment_failure"]
    assert len(failure_rows) == 1
    assert int(failure_rows["val"].iloc[0]) == 0
    assert "adjustment_failure_reason" not in set(result["metric"].values)


def test_resolve_adjustment_failure_metadata_edge_cases() -> None:
    """Resolve helper should normalize supported edge-case input shapes."""
    assert _resolve_adjustment_failure_metadata(None) == (0, None)
    assert _resolve_adjustment_failure_metadata({}) == (0, None)
    assert _resolve_adjustment_failure_metadata({"adjustment_failure": True}) == (
        1,
        None,
    )
    assert _resolve_adjustment_failure_metadata({"adjustment_failure": "yes"}) == (
        1,
        None,
    )
    assert _resolve_adjustment_failure_metadata(
        {"adjustment_failure": 1.0, "adjustment_failure_reason": " boom "}
    ) == (1, "boom")
    assert _resolve_adjustment_failure_metadata(
        {"adjustment_failure": float("nan")}
    ) == (0, None)
    assert _resolve_adjustment_failure_metadata(
        {"adjustment_failure": 0, "adjustment_failure_reason": "ignored"}
    ) == (0, None)
