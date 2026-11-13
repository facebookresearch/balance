# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from balance.weighting_methods.ipw import ipw


class IPWFullDataFitTest(unittest.TestCase):
    def test_cv_metadata_does_not_change_full_data_weights(self) -> None:
        sample = pd.DataFrame({"x": [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7] * 2})
        target = pd.DataFrame({"x": [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5] * 2})
        without_metadata = ipw(
            sample,
            pd.Series(1.0, index=sample.index),
            target,
            pd.Series(1.0, index=target.index),
            transformations=None,
            max_de=None,
            num_lambdas=2,
            lambda_min=0.01,
            lambda_max=0.1,
            random_seed=2020,
            store_fit_metadata=False,
        )["weight"]
        with_metadata = ipw(
            sample,
            pd.Series(1.0, index=sample.index),
            target,
            pd.Series(1.0, index=target.index),
            transformations=None,
            max_de=None,
            num_lambdas=2,
            lambda_min=0.01,
            lambda_max=0.1,
            random_seed=2020,
            store_fit_metadata=True,
        )["weight"]
        np.testing.assert_allclose(
            with_metadata - without_metadata, [0.0] * 20, rtol=0.0, atol=1e-10
        )
