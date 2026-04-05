# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""End-to-end tests that reproduce the balance_quickstart tutorial.

Each test mirrors a section of the tutorial notebook and compares output
against known-good expected values captured from a healthy codebase run.

IPW adjustments use a fixed lambda (``num_lambdas=1``) to skip the
cross-validation grid search and keep runtime under a few seconds.
"""

from __future__ import annotations

import io
import re
import sys
import unittest
from typing import Any

import numpy as np
import pandas as pd
import pytest
from balance import load_data, Sample


def _has_sklearn_1_4() -> bool:
    """Return True if scikit-learn >= 1.4 is available."""
    try:
        import sklearn

        return tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 4)
    except Exception:
        return False


_SKLEARN_1_4_AVAILABLE: bool = _has_sklearn_1_4()


# ---------------------------------------------------------------------------
# Fixed lambda from the tutorial run — avoids the slow CV search.
# ---------------------------------------------------------------------------

_FIXED_LAMBDA: float = 0.041158338186664825
_IPW_FAST_KWARGS: dict[str, Any] = {
    "num_lambdas": 1,
    "lambda_min": _FIXED_LAMBDA,
    "lambda_max": _FIXED_LAMBDA,
}


class E2ETutorialQuickstartTest(unittest.TestCase):
    """Reproduce balance_quickstart.ipynb and verify outputs."""

    # shared across all tests — created once
    target_df: pd.DataFrame
    sample_df: pd.DataFrame
    sample: Sample
    target: Sample
    sample_with_target: Sample
    adjusted: Sample

    @classmethod
    def setUpClass(cls) -> None:
        target_df, sample_df = load_data()
        assert target_df is not None and sample_df is not None
        cls.target_df = target_df
        cls.sample_df = sample_df
        cls.sample = Sample.from_frame(cls.sample_df, outcome_columns=["happiness"])
        cls.target = Sample.from_frame(cls.target_df, outcome_columns=["happiness"])
        cls.sample_with_target = cls.sample.set_target(cls.target)
        cls.adjusted = cls.sample_with_target.adjust(**_IPW_FAST_KWARGS)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    def test_load_data_shapes(self) -> None:
        self.assertEqual(self.sample_df.shape, (1000, 5))
        self.assertEqual(self.target_df.shape, (10000, 5))

    def test_target_df_head(self) -> None:
        expected = {
            "id": {0: "100000", 1: "100001", 2: "100002", 3: "100003", 4: "100004"},
            "gender": {0: "Male", 1: "Male", 2: "Male", 3: np.nan, 4: np.nan},
            "age_group": {0: "45+", 1: "45+", 2: "35-44", 3: "45+", 4: "25-34"},
            "income": {0: 10.18, 1: 6.04, 2: 5.23, 3: 5.75, 4: 4.84},
            "happiness": {0: 61.71, 1: 79.12, 2: 44.21, 3: 83.99, 4: 49.34},
        }
        result = self.target_df.head().round(2).to_dict()

        # Compare non-NaN values; NaN == NaN is False, so handle gender separately
        for col in ["id", "age_group", "income", "happiness"]:
            self.assertEqual(result[col], expected[col], f"Mismatch in column {col}")

        gender = result["gender"]
        self.assertEqual(gender[0], "Male")
        self.assertEqual(gender[1], "Male")
        self.assertEqual(gender[2], "Male")
        self.assertTrue(pd.isna(gender[3]))
        self.assertTrue(pd.isna(gender[4]))

    def test_sample_df_head(self) -> None:
        head = self.sample_df.head().round(2)
        self.assertEqual(head["id"].tolist(), ["0", "1", "2", "3", "4"])

        # Compare non-NaN values; NaN == NaN is False, so handle gender separately
        gender = head["gender"].tolist()
        self.assertEqual(gender[0], "Male")
        self.assertEqual(gender[1], "Female")
        self.assertEqual(gender[2], "Male")
        self.assertTrue(pd.isna(gender[3]))
        self.assertTrue(pd.isna(gender[4]))

        self.assertEqual(
            head["age_group"].tolist(),
            ["25-34", "18-24", "18-24", "18-24", "18-24"],
        )
        self.assertEqual(head["income"].tolist(), [6.43, 9.94, 2.67, 10.55, 2.69])
        self.assertEqual(
            head["happiness"].tolist(), [26.04, 66.89, 37.09, 49.39, 72.30]
        )

    # -----------------------------------------------------------------------
    # 2. Sample object creation
    # -----------------------------------------------------------------------
    def test_sample_df_shape_and_columns(self) -> None:
        """sample.df has 1000 rows x 6 columns (id, gender, age_group, income, happiness, weight)."""
        self.assertEqual(self.sample.df.shape, (1000, 6))
        self.assertCountEqual(
            self.sample.df.columns.tolist(),
            ["id", "gender", "age_group", "income", "happiness", "weight"],
        )

    def test_sample_repr(self) -> None:
        r = repr(self.sample)
        self.assertIn("balance Sample object", r)
        self.assertIn("1000 observations x 3 variables", r)
        self.assertIn("gender,age_group,income", r)
        self.assertIn("outcome_columns: happiness", r)

    def test_target_repr(self) -> None:
        r = repr(self.target)
        self.assertIn("balance Sample object", r)
        self.assertIn("10000 observations x 3 variables", r)

    def test_sample_with_target_repr(self) -> None:
        r = repr(self.sample_with_target)
        self.assertIn("balance Sample object with target set", r)
        self.assertIn("3 common variables", r)

    # -----------------------------------------------------------------------
    # 3. Pre-adjustment diagnostics: covars().mean()
    # -----------------------------------------------------------------------
    def test_covars_mean_before_adjustment(self) -> None:
        mean_df = self.sample_with_target.covars().mean().T
        _expected_str = """  # noqa: F841
source                     self     target
_is_na_gender[T.True]  0.088000   0.089800
age_group[T.25-34]     0.300000   0.297400
age_group[T.35-44]     0.156000   0.299200
age_group[T.45+]       0.053000   0.206300
gender[Female]         0.268000   0.455100
gender[Male]           0.644000   0.455100
gender[_NA]            0.088000   0.089800
income                 6.297302  12.737608
"""
        # Verify key values with rounding
        self.assertAlmostEqual(mean_df.loc["income", "self"], 6.297302, places=2)
        self.assertAlmostEqual(mean_df.loc["income", "target"], 12.737608, places=2)
        self.assertAlmostEqual(mean_df.loc["gender[Female]", "self"], 0.268, places=3)
        self.assertAlmostEqual(
            mean_df.loc["gender[Female]", "target"], 0.4551, places=3
        )
        self.assertAlmostEqual(mean_df.loc["age_group[T.45+]", "self"], 0.053, places=3)
        self.assertAlmostEqual(
            mean_df.loc["age_group[T.45+]", "target"], 0.2063, places=3
        )
        self.assertAlmostEqual(
            mean_df.loc["_is_na_gender[T.True]", "self"], 0.088, places=3
        )

    # -----------------------------------------------------------------------
    # 4. Pre-adjustment diagnostics: covars().asmd()
    # -----------------------------------------------------------------------
    def test_covars_asmd_before_adjustment(self) -> None:
        asmd_df = self.sample_with_target.covars().asmd().T
        _expected_str = """  # noqa: F841
source                  self
age_group[T.25-34]  0.005688
age_group[T.35-44]  0.312711
age_group[T.45+]    0.378828
gender[Female]      0.375699
gender[Male]        0.379314
gender[_NA]         0.006296
income              0.494217
mean(asmd)          0.326799
"""
        self.assertAlmostEqual(asmd_df.loc["mean(asmd)", "self"], 0.326799, places=3)
        self.assertAlmostEqual(asmd_df.loc["income", "self"], 0.494217, places=3)
        self.assertAlmostEqual(
            asmd_df.loc["gender[Female]", "self"], 0.375699, places=3
        )

    def test_covars_asmd_aggregated_before_adjustment(self) -> None:
        asmd_agg = self.sample_with_target.covars().asmd(aggregate_by_main_covar=True).T
        _expected_str = """  # noqa: F841
source          self
age_group   0.232409
gender      0.253769
income      0.494217
mean(asmd)  0.326799
"""
        self.assertAlmostEqual(asmd_agg.loc["age_group", "self"], 0.232409, places=3)
        self.assertAlmostEqual(asmd_agg.loc["gender", "self"], 0.253769, places=3)
        self.assertAlmostEqual(asmd_agg.loc["income", "self"], 0.494217, places=3)
        self.assertAlmostEqual(asmd_agg.loc["mean(asmd)", "self"], 0.326799, places=3)

    # -----------------------------------------------------------------------
    # 5. Distribution diagnostics (KLD, EMD, CVMD, KS)
    # -----------------------------------------------------------------------
    def test_covars_kld_before_adjustment(self) -> None:
        kld = self.sample_with_target.covars().kld().T
        _expected_str = """  # noqa: F841
source         self
gender     0.079889
age_group  0.277138
income     0.114895
mean(kld)  0.157307
"""
        self.assertAlmostEqual(kld.loc["mean(kld)", "self"], 0.157307, places=3)
        self.assertAlmostEqual(kld.loc["gender", "self"], 0.079889, places=3)
        self.assertAlmostEqual(kld.loc["age_group", "self"], 0.277138, places=3)
        self.assertAlmostEqual(kld.loc["income", "self"], 0.114895, places=3)

    def test_covars_emd_before_adjustment(self) -> None:
        emd = self.sample_with_target.covars().emd().T
        _expected_str = """  # noqa: F841
source         self
gender     0.188900
age_group  0.743700
income     6.440306
mean(emd)  2.457635
"""
        self.assertAlmostEqual(emd.loc["mean(emd)", "self"], 2.457635, places=2)
        self.assertAlmostEqual(emd.loc["income", "self"], 6.440306, places=2)

    def test_covars_cvmd_before_adjustment(self) -> None:
        cvmd = self.sample_with_target.covars().cvmd().T
        _expected_str = """  # noqa: F841
source          self
gender      0.012658
age_group   0.061326
income      0.029834
mean(cvmd)  0.034606
"""
        self.assertAlmostEqual(cvmd.loc["mean(cvmd)", "self"], 0.034606, places=3)

    def test_covars_ks_before_adjustment(self) -> None:
        ks = self.sample_with_target.covars().ks().T
        _expected_str = """  # noqa: F841
source         self
gender     0.187100
age_group  0.296500
income     0.246400
mean(ks)   0.243333
"""
        self.assertAlmostEqual(ks.loc["mean(ks)", "self"], 0.243333, places=3)

    # -----------------------------------------------------------------------
    # 6. Pre-adjustment plots (no-crash)
    # -----------------------------------------------------------------------
    def test_covars_plot_plotly_no_crash(self) -> None:
        """covars().plot() should not crash (plotly, default)."""
        import matplotlib

        matplotlib.use("Agg")
        # Default library is plotly; this should not raise
        self.sample_with_target.covars().plot()

    def test_covars_plot_seaborn_kde_no_crash(self) -> None:
        """covars().plot(library='seaborn', dist_type='kde') should not crash."""
        import matplotlib

        matplotlib.use("Agg")
        self.sample_with_target.covars().plot(library="seaborn", dist_type="kde")

    # -----------------------------------------------------------------------
    # 7. Adjustment
    # -----------------------------------------------------------------------
    def test_adjusted_repr(self) -> None:
        r = str(self.adjusted)
        _expected_str = """  # noqa: F841
        Adjusted balance Sample object with target set using ipw
        1000 observations x 3 variables: gender,age_group,income
        id_column: id, weight_column: weight,
        outcome_columns: happiness
"""
        self.assertIn("Adjusted balance Sample object", r)
        self.assertIn("ipw", r)
        self.assertIn("1000 observations x 3 variables", r)
        self.assertIn("outcome_columns: happiness", r)

    def test_adjusted_is_adjusted(self) -> None:
        self.assertTrue(self.adjusted.is_adjusted())
        self.assertFalse(self.sample_with_target.is_adjusted())

    # -----------------------------------------------------------------------
    # 8. Post-adjustment diagnostics: summary
    # -----------------------------------------------------------------------
    def test_adjusted_summary_key_metrics(self) -> None:
        summary = self.adjusted.summary()
        _expected_str = """  # noqa: F841
Adjustment details:
    method: ipw
    weight trimming mean ratio: 20
Covariate diagnostics:
    Covar ASMD reduction: 63.4%
    Covar ASMD (7 variables): 0.327 -> 0.120
    Covar mean KLD reduction: 92.3%
    Covar mean KLD (3 variables): 0.157 -> 0.012
Weight diagnostics:
    design effect (Deff): 1.880
    effective sample size proportion (ESSP): 0.532
    effective sample size (ESS): 531.9
Outcome weighted means:
            happiness
source
self           53.295
target         56.278
unadjusted     48.559
Model performance: Model proportion deviance explained: 0.173
"""
        self.assertIn("method: ipw", summary)
        self.assertIn("weight trimming mean ratio: 20", summary)

        # ASMD reduction
        asmd_match = re.search(
            r"Covar ASMD \(7 variables\): ([\d.]+) -> ([\d.]+)", summary
        )
        self.assertIsNotNone(asmd_match)
        assert asmd_match is not None
        self.assertAlmostEqual(float(asmd_match.group(1)), 0.327, places=2)
        self.assertAlmostEqual(float(asmd_match.group(2)), 0.120, places=1)

        # Design effect
        deff_match = re.search(r"design effect \(Deff\): ([\d.]+)", summary)
        self.assertIsNotNone(deff_match)
        assert deff_match is not None
        self.assertAlmostEqual(float(deff_match.group(1)), 1.880, places=1)

        # ESS
        ess_match = re.search(r"effective sample size \(ESS\): ([\d.]+)", summary)
        self.assertIsNotNone(ess_match)
        assert ess_match is not None
        self.assertAlmostEqual(float(ess_match.group(1)), 531.9, places=0)

        # Outcome means
        self.assertIn("happiness", summary)
        self.assertIn("unadjusted", summary)

    # -----------------------------------------------------------------------
    # 9. Post-adjustment diagnostics: covars().mean()
    # -----------------------------------------------------------------------
    def test_adjusted_covars_mean(self) -> None:
        mean_df = self.adjusted.covars().mean().T
        _expected_str = """  # noqa: F841
source                      self     target  unadjusted
_is_na_gender[T.True]   0.086776   0.089800    0.088000
age_group[T.25-34]      0.307355   0.297400    0.300000
age_group[T.35-44]      0.273609   0.299200    0.156000
age_group[T.45+]        0.137581   0.206300    0.053000
gender[Female]          0.406337   0.455100    0.268000
gender[Male]            0.506887   0.455100    0.644000
gender[_NA]             0.086776   0.089800    0.088000
income                 10.060068  12.737608    6.297302
"""
        # Check that "unadjusted" column now appears
        self.assertIn("unadjusted", mean_df.columns)
        self.assertIn("self", mean_df.columns)
        self.assertIn("target", mean_df.columns)

        # Verify key values
        self.assertAlmostEqual(mean_df.loc["income", "self"], 10.060, places=1)
        self.assertAlmostEqual(mean_df.loc["income", "target"], 12.738, places=1)
        self.assertAlmostEqual(mean_df.loc["income", "unadjusted"], 6.297, places=1)
        self.assertAlmostEqual(mean_df.loc["gender[Female]", "self"], 0.406, places=2)
        self.assertAlmostEqual(
            mean_df.loc["age_group[T.35-44]", "self"], 0.274, places=2
        )
        self.assertAlmostEqual(
            mean_df.loc["age_group[T.35-44]", "unadjusted"], 0.156, places=3
        )

    # -----------------------------------------------------------------------
    # 10. Post-adjustment diagnostics: covars().asmd()
    # -----------------------------------------------------------------------
    def test_adjusted_covars_asmd(self) -> None:
        asmd_df = self.adjusted.covars().asmd().T
        _expected_str = """  # noqa: F841
source                  self  unadjusted  unadjusted - self
age_group[T.25-34]  0.021777    0.005688          -0.016090
age_group[T.35-44]  0.055884    0.312711           0.256827
age_group[T.45+]    0.169816    0.378828           0.209013
gender[Female]      0.097916    0.375699           0.277783
gender[Male]        0.103989    0.379314           0.275324
gender[_NA]         0.010578    0.006296          -0.004282
income              0.205469    0.494217           0.288748
mean(asmd)          0.119597    0.326799           0.207202
"""
        # mean(asmd) should decrease from unadjusted to adjusted
        mean_adjusted = asmd_df.loc["mean(asmd)", "self"]
        mean_unadjusted = asmd_df.loc["mean(asmd)", "unadjusted"]
        self.assertAlmostEqual(mean_adjusted, 0.1196, places=2)
        self.assertAlmostEqual(mean_unadjusted, 0.3268, places=2)
        self.assertGreater(mean_unadjusted, mean_adjusted)

        # income ASMD should shrink
        self.assertAlmostEqual(asmd_df.loc["income", "self"], 0.2055, places=2)
        self.assertAlmostEqual(asmd_df.loc["income", "unadjusted"], 0.4942, places=2)

    # -----------------------------------------------------------------------
    # 11. Post-adjustment diagnostics: covars().kld()
    # -----------------------------------------------------------------------
    def test_adjusted_covars_kld(self) -> None:
        kld = self.adjusted.covars().kld().T
        _expected_str = """  # noqa: F841
source         self  unadjusted  unadjusted - self
gender     0.005603    0.079889           0.074286
age_group  0.030191    0.277138           0.246947
income     0.000768    0.114895           0.114127
mean(kld)  0.012187    0.157307           0.145120
"""
        self.assertAlmostEqual(kld.loc["mean(kld)", "self"], 0.0122, places=2)
        self.assertAlmostEqual(kld.loc["mean(kld)", "unadjusted"], 0.1573, places=2)
        self.assertGreater(
            kld.loc["mean(kld)", "unadjusted"], kld.loc["mean(kld)", "self"]
        )

    # -----------------------------------------------------------------------
    # 12. Post-adjustment plots (no-crash)
    # -----------------------------------------------------------------------
    def test_adjusted_covars_plot_plotly_no_crash(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        self.adjusted.covars().plot()

    def test_adjusted_covars_plot_seaborn_kde_no_crash(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        self.adjusted.covars().plot(library="seaborn", dist_type="kde")

    def test_adjusted_covars_plot_ascii(self) -> None:
        """ASCII plot should produce readable text output."""
        import matplotlib

        matplotlib.use("Agg")

        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            self.adjusted.covars().plot(library="balance", bar_width=30)
        finally:
            sys.stdout = old_stdout

        output = buf.getvalue()

        _expected_str = """  # noqa: F841
=== gender (categorical) ===

Category | population  adjusted  sample
         |
Female   | █████████████████████ (50.0%)
         | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (44.5%)
         | ▐▐▐▐▐▐▐▐▐▐▐▐ (29.4%)

Male     | █████████████████████ (50.0%)
         | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (55.5%)
         | ▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐ (70.6%)

Legend: █ population  ▒ adjusted  ▐ sample
Bar lengths are proportional to weighted frequency within each dataset.

=== age_group (categorical) ===

Category | population  adjusted  sample
         |
18-24    | ████████████ (19.7%)
         | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (28.1%)
         | ▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐ (49.1%)

25-34    | ██████████████████ (29.7%)
         | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (30.7%)
         | ▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐ (30.0%)

35-44    | ██████████████████ (29.9%)
         | ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (27.4%)
         | ▐▐▐▐▐▐▐▐▐▐ (15.6%)

45+      | █████████████ (20.6%)
         | ▒▒▒▒▒▒▒▒ (13.8%)
         | ▐▐▐ (5.3%)

Legend: █ population  ▒ adjusted  ▐ sample
Bar lengths are proportional to weighted frequency within each dataset.
"""
        # Verify key structural elements of ASCII plot
        self.assertIn("=== gender (categorical) ===", output)
        self.assertIn("=== age_group (categorical) ===", output)
        self.assertIn("=== income (numeric, comparative) ===", output)
        self.assertIn("Female", output)
        self.assertIn("Male", output)
        self.assertIn("18-24", output)
        self.assertIn("25-34", output)
        self.assertIn("35-44", output)
        self.assertIn("45+", output)
        self.assertIn("population", output)
        self.assertIn("adjusted", output)
        self.assertIn("sample", output)

        # Verify percentages appear (with some tolerance for rounding)
        self.assertIn("50.0%", output)  # gender population split
        self.assertIn("49.1%", output)  # 18-24 sample proportion

    # -----------------------------------------------------------------------
    # 13. Weights diagnostics
    # -----------------------------------------------------------------------
    def test_weights_plot_no_crash(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        self.adjusted.weights().plot()

    def test_weights_summary(self) -> None:
        ws = self.adjusted.weights().summary().round(2)
        _expected_str = """  # noqa: F841
                                var       val
0                     design_effect      1.88
1       effective_sample_proportion      0.53
2             effective_sample_size    531.92
3                               sum  10000.00
4                    describe_count   1000.00
5                     describe_mean      1.00
6                      describe_std      0.94
7                      describe_min      0.30
8                      describe_25%      0.45
9                      describe_50%      0.65
10                     describe_75%      1.17
11                     describe_max     11.36
12                    prop(w < 0.1)      0.00
13                    prop(w < 0.2)      0.00
14                  prop(w < 0.333)      0.11
15                    prop(w < 0.5)      0.32
16                      prop(w < 1)      0.67
17                     prop(w >= 1)      0.33
18                     prop(w >= 2)      0.10
19                     prop(w >= 3)      0.03
20                     prop(w >= 5)      0.01
21                    prop(w >= 10)      0.00
22               nonparametric_skew      0.37
23  weighted_median_breakdown_point      0.21
"""
        # Build a lookup dict from var -> val
        lookup = dict(zip(ws["var"].tolist(), ws["val"].tolist()))
        self.assertAlmostEqual(lookup["design_effect"], 1.88, places=1)
        self.assertAlmostEqual(lookup["effective_sample_proportion"], 0.53, places=1)
        self.assertAlmostEqual(lookup["effective_sample_size"], 531.92, places=0)
        self.assertAlmostEqual(lookup["sum"], 10000.0, places=0)
        self.assertAlmostEqual(lookup["describe_count"], 1000.0, places=0)
        self.assertAlmostEqual(lookup["describe_mean"], 1.0, places=1)
        self.assertAlmostEqual(lookup["describe_min"], 0.30, places=1)
        self.assertAlmostEqual(lookup["describe_max"], 11.36, places=0)
        self.assertAlmostEqual(lookup["nonparametric_skew"], 0.37, places=1)

    # -----------------------------------------------------------------------
    # 14. Outcomes
    # -----------------------------------------------------------------------
    def test_outcomes_summary(self) -> None:
        summary = self.adjusted.outcomes().summary()
        _expected_str = """  # noqa: F841
1 outcomes: ['happiness']
Mean outcomes (with 95% confidence intervals):
source       self  target  unadjusted           self_ci         target_ci     unadjusted_ci
happiness  53.295  56.278      48.559  (52.096, 54.495)  (55.961, 56.595)  (47.669, 49.449)
"""
        self.assertIn("1 outcomes: ['happiness']", summary)
        self.assertIn("happiness", summary)
        self.assertIn("Mean outcomes", summary)

        # Extract numeric values from the mean outcomes line
        # happiness  53.295  56.278      48.559
        happiness_match = re.search(
            r"happiness\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", summary
        )
        self.assertIsNotNone(happiness_match)
        assert happiness_match is not None
        adjusted_mean = float(happiness_match.group(1))
        target_mean = float(happiness_match.group(2))
        unadjusted_mean = float(happiness_match.group(3))

        self.assertAlmostEqual(adjusted_mean, 53.295, places=0)
        self.assertAlmostEqual(target_mean, 56.278, places=0)
        self.assertAlmostEqual(unadjusted_mean, 48.559, places=0)

        # Adjusted should be closer to target than unadjusted
        self.assertLess(
            abs(adjusted_mean - target_mean),
            abs(unadjusted_mean - target_mean),
        )

    def test_outcomes_plot_no_crash(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        self.adjusted.outcomes().plot()

    # -----------------------------------------------------------------------
    # 15. Comparing adjustment methods: HGB
    # -----------------------------------------------------------------------
    @pytest.mark.requires_sklearn_1_4  # pyre-ignore[56]
    @unittest.skipUnless(_SKLEARN_1_4_AVAILABLE, "requires sklearn >= 1.4")
    def test_adjust_hgb_native_categorical(self) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        hgb = HistGradientBoostingClassifier(
            random_state=0, categorical_features="from_dtype"
        )
        adjusted_hgb = self.sample_with_target.adjust(model=hgb, use_model_matrix=False)
        summary = adjusted_hgb.summary()
        _expected_str = """  # noqa: F841
Adjustment details:
    method: ipw
    weight trimming mean ratio: 20
Covariate diagnostics:
    Covar ASMD reduction: 67.7%
    Covar ASMD (7 variables): 0.327 -> 0.106
    Covar mean KLD reduction: 95.7%
    Covar mean KLD (3 variables): 0.157 -> 0.007
Weight diagnostics:
    design effect (Deff): 2.539
    effective sample size proportion (ESSP): 0.394
    effective sample size (ESS): 393.8
Outcome weighted means:
            happiness
source
self           54.326
target         56.278
unadjusted     48.559
Model performance: Model proportion deviance explained: 0.204
"""
        self.assertIn("method: ipw", summary)

        deff_match = re.search(r"design effect \(Deff\): ([\d.]+)", summary)
        self.assertIsNotNone(deff_match)
        assert deff_match is not None
        self.assertAlmostEqual(float(deff_match.group(1)), 2.539, places=1)

        ess_match = re.search(r"effective sample size \(ESS\): ([\d.]+)", summary)
        self.assertIsNotNone(ess_match)
        assert ess_match is not None
        self.assertAlmostEqual(float(ess_match.group(1)), 393.8, places=0)

    def test_adjust_hgb_with_model_matrix(self) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        hgb_mm = HistGradientBoostingClassifier(random_state=0)
        adjusted_hgb_mm = self.sample_with_target.adjust(
            model=hgb_mm, use_model_matrix=True
        )
        summary = adjusted_hgb_mm.summary()
        _expected_str = """  # noqa: F841
Adjustment details:
    method: ipw
    weight trimming mean ratio: 20
Covariate diagnostics:
    Covar ASMD reduction: 68.8%
    Covar ASMD (7 variables): 0.327 -> 0.102
    Covar mean KLD reduction: 95.7%
    Covar mean KLD (3 variables): 0.157 -> 0.007
Weight diagnostics:
    design effect (Deff): 2.699
    effective sample size proportion (ESSP): 0.370
    effective sample size (ESS): 370.5
Outcome weighted means:
            happiness
source
self           54.460
target         56.278
unadjusted     48.559
Model performance: Model proportion deviance explained: 0.206
"""
        self.assertIn("method: ipw", summary)

        deff_match = re.search(r"design effect \(Deff\): ([\d.]+)", summary)
        self.assertIsNotNone(deff_match)
        assert deff_match is not None
        self.assertAlmostEqual(float(deff_match.group(1)), 2.699, places=1)

        ess_match = re.search(r"effective sample size \(ESS\): ([\d.]+)", summary)
        self.assertIsNotNone(ess_match)
        assert ess_match is not None
        self.assertAlmostEqual(float(ess_match.group(1)), 370.5, places=0)


class E2ETutorialRakeTest(unittest.TestCase):
    """Reproduce key sections of balance_quickstart_rake.ipynb."""

    sample_with_target: Sample
    adjusted_rake: Sample

    @classmethod
    def setUpClass(cls) -> None:
        target_df, sample_df = load_data()
        assert target_df is not None and sample_df is not None
        # Tutorial uses only gender + age_group (categorical covariates)
        sample = Sample.from_frame(
            sample_df[["id", "gender", "age_group", "happiness"]],
            outcome_columns=["happiness"],
        )
        target = Sample.from_frame(
            target_df[["id", "gender", "age_group", "happiness"]],
            outcome_columns=["happiness"],
        )
        cls.sample_with_target = sample.set_target(target)
        cls.adjusted_rake = cls.sample_with_target.adjust(method="rake")

    def test_rake_summary(self) -> None:
        summary = self.adjusted_rake.summary()
        _expected_str = """  # noqa: F841
Adjustment details:
    method: rake
Covariate diagnostics:
    Covar ASMD reduction: 100.0%
    Covar ASMD (6 variables): 0.243 -> 0.000
    Covar mean KLD reduction: 100.0%
    Covar mean KLD (2 variables): 0.179 -> 0.000
Weight diagnostics:
    design effect (Deff): 2.103
    effective sample size proportion (ESSP): 0.476
    effective sample size (ESS): 475.6
Outcome weighted means:
            happiness
source
self           55.484
target         56.278
unadjusted     48.559
"""
        self.assertIn("method: rake", summary)
        self.assertIn("Covar ASMD reduction: 100.0%", summary)

        deff_match = re.search(r"design effect \(Deff\): ([\d.]+)", summary)
        self.assertIsNotNone(deff_match)
        assert deff_match is not None
        self.assertAlmostEqual(float(deff_match.group(1)), 2.103, places=1)

        ess_match = re.search(r"effective sample size \(ESS\): ([\d.]+)", summary)
        self.assertIsNotNone(ess_match)
        assert ess_match is not None
        self.assertAlmostEqual(float(ess_match.group(1)), 475.6, places=0)

    def test_rake_asmd_near_zero(self) -> None:
        """Rake should achieve near-perfect balance on categorical covariates."""
        asmd_df = self.adjusted_rake.covars().asmd().T
        _expected_str = """  # noqa: F841
source                      self  unadjusted  unadjusted - self
age_group[T.25-34]  9.805611e-06    0.005688           0.005678
age_group[T.35-44]  1.769428e-05    0.312711           0.312694
age_group[T.45+]    3.015863e-05    0.378828           0.378798
gender[Female]      1.114671e-16    0.375699           0.375699
gender[Male]        3.344013e-16    0.379314           0.379314
gender[_NA]         4.853912e-17    0.006296           0.006296
mean(asmd)          9.609754e-06    0.243089           0.243080
"""
        mean_asmd = asmd_df.loc["mean(asmd)", "self"]
        # Rake achieves near-zero ASMD on categorical marginals
        self.assertLess(mean_asmd, 0.001)
        # Unadjusted should be much higher
        self.assertAlmostEqual(
            asmd_df.loc["mean(asmd)", "unadjusted"], 0.243089, places=3
        )

    def test_rake_covars_mean(self) -> None:
        """Rake covars().mean() should match target exactly."""
        mean_df = self.adjusted_rake.covars().mean().T
        _expected_str = """  # noqa: F841
source                     self  target  unadjusted
_is_na_gender[T.True]  0.089800  0.0898       0.088
age_group[T.25-34]     0.297404  0.2974       0.300
age_group[T.35-44]     0.299208  0.2992       0.156
age_group[T.45+]       0.206288  0.2063       0.053
gender[Female]         0.455100  0.4551       0.268
gender[Male]           0.455100  0.4551       0.644
gender[_NA]            0.089800  0.0898       0.088
"""
        # Gender proportions should match target exactly after raking
        self.assertAlmostEqual(
            mean_df.loc["gender[Female]", "self"],
            mean_df.loc["gender[Female]", "target"],
            places=4,
        )
        self.assertAlmostEqual(
            mean_df.loc["gender[Male]", "self"],
            mean_df.loc["gender[Male]", "target"],
            places=4,
        )
        # Age group proportions should also match target closely
        self.assertAlmostEqual(
            mean_df.loc["age_group[T.45+]", "self"],
            mean_df.loc["age_group[T.45+]", "target"],
            places=3,
        )


class E2ETutorialPoststratifyTest(unittest.TestCase):
    """Reproduce key sections of balance_quickstart_poststratify.ipynb."""

    def test_poststratify_single_variable(self) -> None:
        """Post-stratify on gender matches target counts exactly."""
        from balance.weighting_methods.poststratify import poststratify

        target_df, sample_df = load_data()
        assert target_df is not None and sample_df is not None

        sample_gender = sample_df.dropna(subset=["gender"])
        target_gender = target_df.dropna(subset=["gender"])

        result = poststratify(
            sample_df=sample_gender[["gender"]],
            sample_weights=pd.Series(1, index=sample_gender.index),
            target_df=target_gender[["gender"]],
            target_weights=pd.Series(1, index=target_gender.index),
        )

        weighted = sample_gender.assign(weight=result["weight"])
        weighted_counts = weighted.groupby("gender")["weight"].sum()
        target_counts = target_gender.groupby("gender").size()

        _expected_str = """  # noqa: F841
        weighted_sample  target_population
gender
Female           4551.0               4551
Male             4551.0               4551
"""
        # Weighted sample counts should exactly match target population counts
        self.assertAlmostEqual(weighted_counts["Female"], 4551.0, places=0)
        self.assertAlmostEqual(weighted_counts["Male"], 4551.0, places=0)
        self.assertAlmostEqual(
            weighted_counts["Female"], float(target_counts["Female"]), places=0
        )
        self.assertAlmostEqual(
            weighted_counts["Male"], float(target_counts["Male"]), places=0
        )

    def test_poststratify_joint_distribution(self) -> None:
        """Post-stratify on gender x age_group matches joint target counts."""
        from balance.weighting_methods.poststratify import poststratify

        target_df, sample_df = load_data()
        assert target_df is not None and sample_df is not None

        covariates = ["gender", "age_group"]
        sample_cells = sample_df.dropna(subset=covariates)
        target_cells = target_df.dropna(subset=covariates)

        result = poststratify(
            sample_df=sample_cells[covariates],
            sample_weights=pd.Series(1, index=sample_cells.index),
            target_df=target_cells[covariates],
            target_weights=pd.Series(1, index=target_cells.index),
        )

        weighted = sample_cells.assign(weight=result["weight"])
        weighted_counts = weighted.groupby(covariates)["weight"].sum()
        target_counts = target_cells.groupby(covariates).size()

        _expected_str = """  # noqa: F841
                  weighted_sample  target_population
gender age_group
Female 18-24                876.0                876
       25-34               1360.0               1360
       35-44               1370.0               1370
       45+                  945.0                945
Male   18-24                905.0                905
       25-34               1355.0               1355
       35-44               1347.0               1347
       45+                  944.0                944
"""
        # Each cell's weighted count should match target exactly
        for cell, count in target_counts.items():
            self.assertAlmostEqual(
                weighted_counts[cell],
                float(count),
                places=0,
                msg=f"Cell {cell} mismatch",
            )


if __name__ == "__main__":
    unittest.main()
