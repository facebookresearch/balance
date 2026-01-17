# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from balance.sample_class import Sample
from balance.estimators import BalanceWeighting

class TestBalanceWeighting(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        self.sample_df = pd.DataFrame({
            "id": [str(i) for i in range(100)],
            "x": np.random.normal(0, 1, 100),
            "w": np.ones(100)
        })
        self.target_df = pd.DataFrame({
            "id": [str(i) for i in range(100, 200)],
            "x": np.random.normal(0.5, 1, 100),
            "w": np.ones(100)
        })
        
        self.sample = Sample.from_frame(self.sample_df, id_column="id", weight_column="w")
        self.target = Sample.from_frame(self.target_df, id_column="id", weight_column="w")

    def test_fit_predict_transform_ipw(self) -> None:
        # Use transformations=None to avoid 'quantize' binning mismatch between 
        # fit (sample+target) and predict (sample only).
        est = BalanceWeighting(method="ipw", transformations=None)
        
        # Test fit
        est.fit(self.sample, self.target)
        self.assertIsNotNone(est.model_)
        
        # Test predict
        weights = est.predict(self.sample)
        self.assertIsInstance(weights, pd.Series)
        self.assertEqual(len(weights), 100)
        
        # Verify weights are different from 1 (adjustment happened)
        # Note: with continuous x and small sample, adjustment might be subtle but should be present.
        self.assertFalse(np.allclose(weights, 1.0))
        
        # Test transform
        new_sample = est.transform(self.sample)
        self.assertIsInstance(new_sample, Sample)
        self.assertTrue(new_sample.is_adjusted())
        self.assertEqual(new_sample._links["unadjusted"], self.sample)
        self.assertEqual(new_sample._links["target"], self.target)
        
        # Verify weights in new sample match predicted weights
        pd.testing.assert_series_equal(
            new_sample.weight_column.reset_index(drop=True), 
            weights.reset_index(drop=True),
            check_names=False
        )

    def test_fit_without_target_arg(self) -> None:
        # Set target on sample
        sample_with_target = self.sample.set_target(self.target)
        
        est = BalanceWeighting(method="ipw", transformations=None)
        est.fit(sample_with_target) # No target arg
        
        self.assertIsNotNone(est.model_)
        self.assertEqual(est._target, self.target) # Should have picked up the target

    def test_transform_new_data(self) -> None:
        # Fit on original data
        est = BalanceWeighting(method="ipw", transformations=None)
        est.fit(self.sample, self.target)
        
        # Create new data (subset)
        new_df = self.sample_df.iloc[:10].copy()
        new_df["id"] = [str(i+1000) for i in range(10)]
        new_sample = Sample.from_frame(new_df, id_column="id", weight_column="w")
        
        # Predict on new data
        weights = est.predict(new_sample)
        self.assertEqual(len(weights), 10)
        
        # Transform new data
        transformed = est.transform(new_sample)
        self.assertEqual(len(transformed.df), 10)
        self.assertTrue(transformed.is_adjusted())

    def test_sklearn_compliance(self) -> None:
        from sklearn.base import clone
        est = BalanceWeighting(method="ipw", transformations=None)
        est_clone = clone(est)
        self.assertEqual(est_clone.method, "ipw")

if __name__ == "__main__":
    unittest.main()

