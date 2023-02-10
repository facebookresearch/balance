0.5.0 (the future)
==================
### New Features
- The `datasets.load_data` function now also supports the input "sim_data_cbps", which loads the simulated data used in the CBPS R vs Python tutorial. It is also used in unit-testing to compare the CBPS weights produced from Python (i.e.: balance) with R (i.e.: the CBPS package). The testing shows how the correlation of the weights from the two implementations (both Pearson and Spearman) produce a correlation of >0.98.

### Documentation
- New tutorial:
    - TBD (BCPS comparison to R)


0.4.0 (2023-02-08)
==================
### New Features
- Added two new flags to the cli:
    - `--standardize_types`: This gives cli users the ability to set the `standardize_types` parameter in Sample.from_frame
        to True or False. To learn more about this parameter, see:
        https://import-balance.org/api_reference/html/balance.sample_class.html#balance.sample_class.Sample.from_frame
    - `--return_df_with_original_dtypes`: the Sample object now stores the dtypes of the original df that was read using Sample.from_frame. This can be used to restore the original dtypes of the file output from the cli. This is relevant in cases in which we want to convert back the dtypes of columns from how they are stored in Sample, to their original types (e.g.: if something was Int32 it would be turned in float32 in balance.Sample, and using the new flag will return that column, when using the cli, to be back in the Int32 type). This feature may not be robust to various edge cases. So use with caution.
- In the logging:
    - Added warnings about dtypes changes. E.g.: if using Sample.from_frame with a column that has Int32, it will be turned into float32 in the internal storage of sample. Now there will be a warning message indicating of this change.
    - Increase the default length of logger printing (from 500 to 2000)


### Bug Fixes
- Fix pandas warning: SettingWithCopyWarning in from_frame (and other places in sample_class.py)
- sample.from_frame has a new argument `use_deepcopy` to decide if changes made to the df inside the sample object would also change the original df that was provided to the sample object. The default is now set to `True` since it's more likely that we'd like to keep the changes inside the sample object to the df contained in it, and not have them spill into the original df.


0.3.1 (2023-02-01)
==================
### Bug Fixes
- Sample.from_frame now also converts int16 and in8 to float16 and float16. Thus helping to avoid `TypeError: Cannot interpret 'Int16Dtype()' as a data type` style errors.

### Documentation
- Added ISSUE_TEMPLATE


0.3.0 (2023-01-30)
==================
### New Features
- Added compatibility for Python 3.11 (by supporting SciPy 1.9.2) (props to @tomwagstaff-opml for flagging this issue).
- Added the `session-info` package as a dependency.

### Bug Fixes
- Fixed pip install from source on Windows machines (props to @tomwagstaff-opml for the bug report).

### Documentation
- Added `session_info.show()` outputs to the end of the three tutorials (at: https://import-balance.org/docs/tutorials/)
- Misc updates to the README.


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
