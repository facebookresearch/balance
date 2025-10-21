---
title: Balance in motion - v0.12.0 expands Python version support, visualizations, and statistical methods

authors:
  - name: Tal Galili
    title: Machine Learning Engineer
    url: https://www.linkedin.com/in/tal-galili-5993085/
  - name: Wesley Lee
    title: Research Scientist
    url: https://www.linkedin.com/in/wesley-lee/
tags: [python, open-source, survey-statistics, package-update]
hide_table_of_contents: true
---

**tl;dr – balance v0.12.0**

We're excited to announce [**balance v0.12.0**](https://pypi.org/project/balance/)! Since our initial release, [balance](https://import-balance.org/) has evolved into a comprehensive Python package for [adjusting biased samples](https://import-balance.org/blog/2023/01/09/bringing-balance-to-your-data/). This post highlights the most significant improvements from [v0.1.0 (2022-11-20)](https://github.com/facebookresearch/balance/releases/tag/0.1.0) through [v0.12.0 (2025-10-14)](https://github.com/facebookresearch/balance/releases/tag/0.12.0), showcasing how we've made balance easier to use:

- **Expanded compatibility:** Now supports Python 3.9–3.14 on Windows, macOS, and Linux, with smarter dependency management and a switch to the MIT license.
- **Major upgrades:** Improved statistical methods (IPW, raking, poststratification), interactive Plotly visualizations, and new variance/confidence interval tools.
- **Better experience:** Enhanced CLI, bug fixes, and expanded docs/tutorials for easier use and learning.

[![balance_logo_horizontal](https://raw.githubusercontent.com/facebookresearch/balance/main/website/static/img/balance_logo/PNG/Horizontal/balance_Logo_Horizontal_FullColor_RGB.png)](https://import-balance.org/)

<!--truncate-->


## Cutting-Edge Python Compatibility

**balance** now supports all three major OS platforms: **Windows**, **macOS**, and **Linux** - for **Python 3.9 through 3.14**, ensuring you can use the latest Python features without compatibility concerns.

### What's New:
- **Full support for Windows**
- **Full support for Python 3.11, 3.12, 3.13 and 3.14**
- **Smart dependency management** with version-specific constraints for `numpy`, `pandas`, `scipy`, and `scikit-learn` for Python 3.9-3.11.
- **Greater flexibility** for Python 3.12+ users with removed upper version constraints, while eliminating **260+ pandas deprecation warnings** and modernized our code.
- **Python 3.8 deprecated** due to typing incompatibilities
- **License Update** from GPL v2 to the **[MIT license](https://github.com/facebookresearch/balance/blob/main/LICENSE)** for greater flexibility and easier integration into your projects


## Methodological Improvements

### Transition to scikit-learn for IPW

We've migrated from `glmnet` to **scikit-learn's logistic regression** (v0.10.0) for our [inverse propensity weighting (IPW)](https://import-balance.org/docs/docs/statistical_methods/ipw/) method, bringing significant benefits:

**Pros:**
- Windows OS support
- Python 3.11+ compatibility
- Eliminated `glmnet` dependency

**Trade-offs:**
- Uses L2 penalties instead of L1 (slight weight differences)
- 2-5x slower than previous version

### Raking Algorithm: Faster and More Reliable

We've completely refactored our [raking (rake weighting)](https://import-balance.org/docs/docs/statistical_methods/rake/) implementation with an **array-based IPFN algorithm** that delivers:

- **Support for marginal distribution target distributions** with the new `prepare_marginal_dist_for_raking` helper function
- **Better performance** across all Python versions
- **Consistent results** through automatic variable alphabetization


### Flexibility with Poststratification

The [poststratify](https://import-balance.org/docs/docs/statistical_methods/poststratify/) method now includes a **`strict_matching` parameter** (default `True`). When set to `False`, it gracefully handles missing sample cells by issuing warnings and assigning zero weights.


## Visualization and Summarization Enhancements

### Interactive Plotting

All visualization functions now produce **interactive Plotly plots** by default:

- **Customizable layouts** via `kwargs` (control width, height, and more)
- **New `plotly_plot_density`** for interactive kernel density estimation with support for 'kde' plots in `plotly_plot_dist` and `plot_dist`
- **`BalanceWeightsDF.plot`** now uses Plotly instead of static seaborn plots

![](https://import-balance.org/assets/images/fig_09_seaborn_outcome_kde_after-26fa9668164349253b2614335961ade9.png)

All bar plots now support **`ylim` argument** for precise y-axis control:
    s3_null.covars().plot(ylim=(0, 1))

### Statistical Summaries

New variance and confidence interval methods make it easier to assess uncertainty:

- `.var_of_mean()` - Variance of weighted means
- `.ci_of_mean()` - Confidence intervals for weighted means
- `.mean_with_ci()` - Combined mean with confidence intervals
- Enhanced `.summary()` method for `BalanceWeightsDF`

## Developer Experience


### Notable Bug Fixes

- Fixed `rm_mutual_nas` to preserve Series index
- Improved `Sample.from_frame` weight column detection (now recognizes "weights" and "weight")
- Better handling of int8/int16 columns (converts to float16)
- Fixed color assignments in comparison plots
- Resolved various edge cases in `plot_hist_kde` and `plot_bar`

### CLI Improvements

The command-line interface now offers more control:

- **Formula specification** via string arguments
- **Type standardization** controls
- **Original dtype preservation** with `--return_df_with_original_dtypes`
- **Flexible trimming** with `weight_trimming_mean_ratio=None` option
- **Enhanced logging** with dtype change warnings


## Documentation & Tutorials

We've significantly expanded our [documentation](https://import-balance.org/docs/docs/general_framework/) with new tutorials:
- **[Quickstart](https://import-balance.org/docs/tutorials/quickstart/)** - Get started with balance basics
- **[Quickstart with Raking](https://import-balance.org/docs/tutorials/quickstart_rake/)** - Compare raking vs. IPW
- **[Quickstart with CBPS](https://import-balance.org/docs/tutorials/quickstart_cbps/)** - Covariate Balancing Propensity Score method
- **[Transformations and Formulas](https://import-balance.org/docs/tutorials/balance_transformations_and_formulas/)** - Advanced covariate preprocessing
- **[CBPS: R vs. Python Comparison](https://import-balance.org/docs/tutorials/comparing_cbps_in_r_vs_python_using_sim_data/)** - Validation against R's CBPS package

Also added a Link to [conference presentations](https://github.com/facebookresearch/balance/blob/main/website/static/docs/Balancing_biased_data_samples_with_the_balance_Python_package_-_ISA_conference_2023-06-01.pdf) (ISA 2023).

This extends our existing Statistical Methods Documentation:
- [Inverse Propensity Weighting (IPW)](https://import-balance.org/docs/docs/statistical_methods/ipw/)
- [Covariate Balancing Propensity Score (CBPS)](https://import-balance.org/docs/docs/statistical_methods/cbps/)
- [Post-stratification](https://import-balance.org/docs/docs/statistical_methods/poststratify/)
- [Raking](https://import-balance.org/docs/docs/statistical_methods/rake/)


## Community & Contributors

A huge thank you to our contributors: **@talgalili**, **@wesleytlee**, **@SarigT**, **@ahakso**, **@stevemandala**, **@tomwagstaff-opml**, **@zbraiterman**, and **@luca-martial**!

Want to contribute? Check out our [contributing guide](https://github.com/facebookresearch/balance/blob/main/CONTRIBUTING.md).

---

## Get Started Today

Ready to try **balance** or upgrade to v0.12.0?

### Installation:

    python -m pip install balance

### Resources:
- **Website:** https://import-balance.org/
- **GitHub:** https://github.com/facebookresearch/balance
- **Documentation:** https://import-balance.org/docs/docs/general_framework/
- **Tutorials:** https://import-balance.org/docs/tutorials/
- **Blog:** https://import-balance.org/blog/
- **Paper:** [balance – a Python package for balancing biased data samples](https://arxiv.org/abs/2307.06024) (Sarig, Galili, & Eilat, 2023)

### Get Help:
- **Ask questions:** https://github.com/facebookresearch/balance/issues/new?template=support_question.md
- **Report bugs:** https://github.com/facebookresearch/balance/issues/new?template=bug_report.md
- **Request features:** https://github.com/facebookresearch/balance/issues/new?template=feature_request.md

We welcome your feedback, questions, and contributions as we continue making **balance** the go-to tool for survey statistics and bias adjustment in Python!
