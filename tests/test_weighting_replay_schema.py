# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pickle
import unittest

import numpy as np
import pandas as pd
from balance import Sample
from balance.util import _assert_type


def _frames() -> tuple[Sample, Sample]:
    sample_groups = ["a", "b", "c"] + ["a"] * 17 + ["b"] * 9 + ["c"] * 9
    target_groups = ["a", "b", "c"] + ["a"] * 9 + ["b"] * 17 + ["c"] * 9
    sample = Sample.from_frame(
        pd.DataFrame(
            {
                "id": [f"s{i}" for i in range(len(sample_groups))],
                "group": sample_groups,
                "x": np.linspace(0, 1, len(sample_groups)),
            }
        )
    )
    target = Sample.from_frame(
        pd.DataFrame(
            {
                "id": [f"t{i}" for i in range(len(target_groups))],
                "group": target_groups,
                "x": np.linspace(0, 1, len(target_groups)),
            }
        )
    )
    return sample, target


def _subset(sample: Sample, target: Sample, rows: list[int]) -> Sample:
    return Sample.from_frame(sample.df.iloc[rows]).set_target(
        Sample.from_frame(target.df.iloc[rows])
    )


class WeightingReplaySchemaTest(unittest.TestCase):
    def test_ipw_reuses_training_encoding_when_a_category_is_absent(self) -> None:
        sample, target = _frames()
        fitted = sample.set_target(target).fit(
            method="ipw", transformations=None, one_hot_encoding=True, num_lambdas=1
        )
        subset = _subset(sample, target, [0, 1])
        expected = fitted.design_matrix(on="sample").iloc[:2]
        actual = fitted.design_matrix(on="sample", data=subset)

        self.assertEqual(actual.columns.tolist(), expected.columns.tolist())
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy())
        np.testing.assert_allclose(
            fitted.predict_proba(on="sample", data=subset).to_numpy(),
            fitted.predict_proba(on="sample").iloc[:2].to_numpy(),
        )

    def test_ipw_replay_survives_pickle_round_trip(self) -> None:
        sample, target = _frames()
        fitted = sample.set_target(target).fit(
            method="ipw", transformations=None, one_hot_encoding=True, num_lambdas=1
        )
        # @lint-ignore PYTHONPICKLEISBAD - Trusted synthetic model created in this test.
        restored = pickle.loads(pickle.dumps(fitted))
        subset = _subset(sample, target, [0, 1])

        self.assertIsNotNone(_assert_type(restored.model)["preprocessing_state"])
        np.testing.assert_allclose(
            restored.predict_proba(on="sample", data=subset).to_numpy(),
            fitted.predict_proba(on="sample").iloc[:2].to_numpy(),
        )

    def test_cbps_replay_preserves_weights_when_reference_level_is_absent(self) -> None:
        sample, target = _frames()
        fitted = sample.set_target(target).fit(method="cbps", transformations=None)
        subset = _subset(sample, target, [1, 2])
        expected = _assert_type(fitted.weight_series).iloc[[1, 2]]
        expected = (
            expected / expected.sum() * _assert_type(subset.target).weight_series.sum()
        )
        actual = fitted.predict_weights(data=subset)

        self.assertEqual(actual.index.tolist(), expected.index.tolist())
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-5)

    def test_default_ipw_transfer_reports_data_dependent_transformations(self) -> None:
        sample, target = _frames()
        fitted = sample.set_target(target).fit(method="ipw", num_lambdas=1)
        subset = _subset(sample, target, [0, 1])

        with self.assertRaisesRegex(ValueError, "data-dependent.*not replayable"):
            fitted.predict_weights(data=subset)

    def test_default_cbps_transfer_reports_data_dependent_transformations(self) -> None:
        sample, target = _frames()
        fitted = sample.set_target(target).fit(method="cbps")
        subset = _subset(sample, target, [0, 1])

        with self.assertRaisesRegex(ValueError, "data-dependent.*not replayable"):
            fitted.predict_weights(data=subset)

    def test_default_same_data_replay_remains_available(self) -> None:
        sample, target = _frames()
        original = sample.set_target(target)
        fitted = original.fit(method="ipw", num_lambdas=1, inplace=False)

        predicted = fitted.predict_weights(data=original)
        self.assertEqual(len(predicted), len(sample.df))
        np.testing.assert_allclose(
            predicted.to_numpy(), _assert_type(fitted.weight_series).to_numpy()
        )
