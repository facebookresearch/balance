# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pickle
import unittest
from copy import deepcopy
from typing import Any

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


class SameDataReplayContractTest(unittest.TestCase):
    def test_default_replay_accepts_unchanged_original_copy_and_pickle(self) -> None:
        sample, target = _frames()
        original = sample.set_target(target)
        for method in ("ipw", "cbps"):
            with self.subTest(method=method):
                kwargs: dict[str, Any] = {"num_lambdas": 1} if method == "ipw" else {}
                fitted = original.fit(method=method, inplace=False, **kwargs)
                expected = _assert_type(fitted.weight_series).to_numpy()
                inputs = (
                    original,
                    deepcopy(original),
                    pickle.loads(pickle.dumps(original)),
                )
                for data in inputs:
                    np.testing.assert_allclose(
                        fitted.predict_weights(data=data).to_numpy(),
                        expected,
                        rtol=1e-5,
                    )
                models = (deepcopy(fitted), pickle.loads(pickle.dumps(fitted)))
                for model in models:
                    np.testing.assert_allclose(
                        model.predict_weights(data=original).to_numpy(),
                        expected,
                        rtol=1e-5,
                    )

    def test_default_replay_treats_dtype_and_new_labels_as_transfer(self) -> None:
        sample, target = _frames()
        original = sample.set_target(target)
        for method in ("ipw", "cbps"):
            kwargs: dict[str, Any] = {"num_lambdas": 1} if method == "ipw" else {}
            fitted = original.fit(method=method, inplace=False, **kwargs)
            for change in ("dtype", "index"):
                with self.subTest(method=method, change=change):
                    frame = sample.df.copy(deep=True)
                    if change == "dtype":
                        frame = frame.astype({"x": object})
                    else:
                        frame.index = pd.Index(range(100, 100 + len(frame)))
                    data = Sample.from_frame(frame, standardize_types=False).set_target(
                        Sample.from_frame(target.df, standardize_types=False)
                    )
                    np.testing.assert_array_equal(
                        data.covars().df.to_numpy(), original.covars().df.to_numpy()
                    )
                    self.assertFalse(data.covars().df.equals(original.covars().df))
                    with self.assertRaisesRegex(
                        ValueError, "data-dependent.*not replayable"
                    ) as error:
                        fitted.predict_weights(data=data)
                    self.assertIn("deterministic transformations", str(error.exception))
                    self.assertIn("re-fit", str(error.exception))
