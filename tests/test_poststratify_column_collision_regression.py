# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from balance import BalanceFrame, Sample, SampleFrame
from balance.weighting_methods.poststratify import poststratify


@pytest.mark.parametrize("dtype", ["float64", "str", "category"])
def test_design_weight_covariate_preserves_cell_totals(dtype: str) -> None:
    index = pd.Index(["s2", "s0", "s1"], name="respondent")
    sample = pd.DataFrame({"design_weight": [1, 1, 2]}, index=index).astype(dtype)
    target = pd.DataFrame({"design_weight": [1, 2, 2, 2]}).astype(dtype)
    sample_weights = pd.Series([1.0, 2.0, 1.0], index=index, name="survey_weight")
    target_weights = pd.Series([2.0, 1.0, 3.0, 6.0], name="survey_weight")
    original_sample = sample.copy(deep=True)
    original_target = target.copy(deep=True)

    result = poststratify(
        sample,
        sample_weights,
        target,
        target_weights,
        transformations=None,
        store_fit_metadata=True,
    )

    pd.testing.assert_series_equal(
        result["weight"], pd.Series([2 / 3, 4 / 3, 10.0], index=index)
    )
    np.testing.assert_allclose(result["model"]["cell_weight_ratio"], [2 / 3, 10.0])
    actual_totals = (
        result["weight"].groupby(sample["design_weight"], observed=True).sum()
    )
    expected_totals = target_weights.groupby(
        target["design_weight"], observed=True
    ).sum()
    pd.testing.assert_series_equal(actual_totals, expected_totals, check_names=False)
    pd.testing.assert_frame_equal(sample, original_sample)
    pd.testing.assert_frame_equal(target, original_target)


def test_design_weight_temporary_names_do_not_overwrite_joint_covariates() -> None:
    sample = pd.DataFrame(
        {
            "design_weight": [1, 1, 2, 2],
            "_design_weight": ["x", "y", "x", "y"],
            "__design_weight": [0, 1, 2, 3],
        },
        index=pd.Index([4, 4, 2, 2], name="respondent"),
    )
    target = sample.reset_index(drop=True)
    result = poststratify(
        sample,
        pd.Series([1.0, 2.0, 3.0, 4.0], index=sample.index),
        target,
        pd.Series([8.0, 6.0, 4.0, 2.0]),
        transformations=None,
    )

    pd.testing.assert_series_equal(
        result["weight"], pd.Series([8.0, 6.0, 4.0, 2.0], index=sample.index)
    )


@pytest.mark.parametrize("frame_type", [Sample, BalanceFrame])
@pytest.mark.parametrize("na_action", ["add_indicator", "drop"])
def test_design_weight_covariate_fit_and_replay(
    frame_type: type[Sample] | type[BalanceFrame], na_action: str
) -> None:
    sample_df = pd.DataFrame(
        {
            "id": ["s0", "s1", "s2", "s3"],
            "design_weight": ["a", "a", "b", None],
            "survey_weight": [1.0, 2.0, 1.0, 4.0],
        },
        index=pd.Index([20, 10, 30, 40], name="respondent"),
    )
    target_df = pd.DataFrame(
        {
            "id": ["t0", "t1", "t2", "t3", "t4"],
            "design_weight": ["a", "b", "b", "b", None],
            "survey_weight": [2.0, 1.0, 3.0, 6.0, 5.0],
        }
    )
    if frame_type is Sample:
        frame = Sample.from_frame(
            sample_df, id_column="id", weight_column="survey_weight"
        ).set_target(
            Sample.from_frame(target_df, id_column="id", weight_column="survey_weight")
        )
    else:
        frame = BalanceFrame(
            sample=SampleFrame.from_frame(
                sample_df, id_column="id", weight_column="survey_weight"
            ),
            target=SampleFrame.from_frame(
                target_df, id_column="id", weight_column="survey_weight"
            ),
        )

    fitted = frame.fit(
        method="poststratify",
        transformations=None,
        na_action=na_action,
        inplace=False,
    )
    expected = pd.Series(
        [2 / 3, 4 / 3, 10.0, np.nan if na_action == "drop" else 5.0],
        index=sample_df.index,
        name="survey_weight",
    )
    pd.testing.assert_series_equal(fitted.weight_series, expected)
    pd.testing.assert_series_equal(
        fitted.predict_weights(), expected, check_names=False
    )
    pd.testing.assert_series_equal(frame.weight_series, sample_df["survey_weight"])
