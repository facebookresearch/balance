[Unreleased]
==================
### New Features
- Added compatibility for SciPy 1.9.2 and Python 3.11

### Bug Fixes
- Fixed pip install from source on Windows machines

0.2.0 (2023-01-19)
==================
### New Features
- cli improvements:
    - Add an option to set weight_trimming_mean_ratio = None for no trimming.
    - Add an option to set transformations to be None (i.e. no transformations).
- Add an option to adapt the title in:
    - stats_and_plots.weighted_comparison_plots.plot_bar
    - stats_and_plots.weighted_comparison_plots.plot_hist_kde

### Bug Fixes
- Fix (and simplify) balanceDF.plot to organize the order of groups (now unadjusted/self is left, adjusted/self center, and target is on the right)
- Fix plotly functions to use the red color for self when only compared to target (since in that case it is likely unadjusted): balance.stats_and_plots.weighted_comparisons_plots.plotly_plot_qq and balance.stats_and_plots.weighted_comparisons_plots.plotly_plot_bar
- Fix seaborn_plot_dist: output None by default (instead of axis object). Added a return_Axes argument to control this behavior.
- Fix some test_cbps tests that were failing due to non-exact matches (we made the test less sensitive)

### Documentation
- New blog section, with the post: [Bringing "balance" to your data
](https://import-balance.org/blog/2023/01/09/bringing-balance-to-your-data/)
- New tutorial:
    - [**quickstart_cbps**](https://import-balance.org/docs/tutorials/quickstart_cbps/) - like the [**quickstart**](https://import-balance.org/docs/tutorials/quickstart/) tutorial, but shows how to use the CBPS algorithm and compares the results to IPW (logistic regression with LASSO).
    - [**balance_transformations_and_formulas**](https://import-balance.org/docs/tutorials/balance_transformations_and_formulas/) - This tutorial showcases ways in which transformations, formulas and penalty can be included in your pre-processing of the covariates
    before adjusting for them.
- API docs:
    - New: highlighting on codeblocks
    - a bunch of text fixes.
- Update README.md
    - logo
    - with contributors
    - typo fixes (props to @zbraiterman and @luca-martial).
- Added section about "Releasing a new version" to CONTRIBUTING.md
    - Available under ["Docs/Contributing"](https://import-balance.org/docs/docs/contributing/#releasing-a-new-version) section of website

## Misc
- Added automated Github Action package builds & deployment to PyPi on release.
  - See [release.yml](https://github.com/facebookresearch/balance/blob/main/.github/workflows/release.yml)

0.1.0 (2022-11-20)
==================
### Summary
- balance released to the world!
