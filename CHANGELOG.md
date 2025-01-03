0.10.0 (2025-01-03)
==================
## New Features
- Dependency on glmnet has been removed, and the `ipw` method now uses sklearn. This should enable support for newer python versions.
- `ipw` method uses logistic regression with L2-penalties instead of L1-penalties for computational reasons. The transition from glmnet to sklearn and use of L2-penalties will lead to slightly different generated weights compared to previous versions of Balance.
- Unfortunately, the sklearn-based `ipw` method is generally slower than the previous version by 2-5x. Consider using the new arguments `lambda_min`, `lambda_max`, and `num_lambdas` for a more efficient search over the `ipw` penalization space.

## Misc
-- Update to MIT license
-- Updated Python and package compatibility. Balance is now compatible with Python 3.11, but no longer compatible with Python 3.8 due to typing errors. Balance is currently incompatible with Python 3.12 due to the removal of distutils.

## Contributors
@wesleytlee, @talgalili, @SarigT


0.9.1 (2023-07-30)
==================
## Bug Fixes
- Fix E721 flake8 issue (see: https://github.com/facebookresearch/balance/actions/runs/5704381365/job/15457952704)
- Remove support for python 3.11 from release.yml

## Documentation
- Added links to presentation given at ISA 2023.
- Fixed misc typos.


0.9.0 (2023-05-22)
==================
## News
- Remove support for python 3.11 due to new test failures. This will be the case until glmnet will be replaced by sklearn. hopefully before end of year.

## New Features
- All plotly functions: add kwargs to pass arguments to update_layout in all plotly figures. This is useful to control width and height of the plot. For example, when wanting to save a high resolution of the image.
- Add a `summary` methods to `BalanceWeightsDF` (i.e.: `Sample.weights().summary()`) to easily get access to summary statistics of the survey weights. Also, it means that `Sample.diagnostics()` now uses this new summary method in its internal implementation.
- `BalanceWeightsDF.plot` method now relies on the default `BalanceDF.plot` method. This means that instead of a static seaborn kde plot we'll get an interactive plotly version.

## Bug Fixes
- datasets
    - Remove a no-op in `load_data` and accommodate deprecation of pandas syntax by using a list rather than a set when selecting df columns (thanks @ahakso for the PR).
    - Make the outcome variable (`happiness`) be properly displayed in the tutorials (so we can see the benefit of the weighting process). This included fixing the simulation code in the target.
- Fix `Sample.outcomes().summary()` so it will output the ci columns without truncating them.

## Documentation
- Fix text based on updated from version 0.7.0 and 0.8.0.
    - https://import-balance.org/docs/docs/general_framework/adjusting_sample_to_population/
- Fix tutorials to include the outcome in the target.

## Contributors
@talgalili, @SarigT, @ahakso


0.8.0 (2023-04-26)
==================
## New Features
- Add `rake` method to .adjust (currently in beta, given that it doesn't handles marginal target as input).
- Add a new function `prepare_marginal_dist_for_raking` - to take in a dict of marginal proportions and turn them into a pandas DataFrame. This can serve as an input target population for raking.

## Misc
- The `ipw` function now gets max_de=None as default (instead of 1.5). This version is faster, and the user can still choose a threshold as desired.
- Adding hex stickers graphics files

## Documentation
- New section on [raking.](https://import-balance.org/docs/docs/statistical_methods/rake/)
- New notebook (in the tutorial section):
    - [**quickstart_rake**](https://import-balance.org/docs/tutorials/quickstart_rake/) - like the [**quickstart**](https://import-balance.org/docs/tutorials/quickstart/) tutorial, but shows how to use the rake (raking) algorithm and compares the results to IPW (logistic regression with LASSO).

## Contributors
@talgalili, @SarigT


0.7.0 (2023-04-10)
==================
## New Features
- Add `plotly_plot_density` function: Plots interactive density plots of the given variables using kernel density estimation.
- Modified `plotly_plot_dist` and `plot_dist` to also support 'kde' plots. Also, these are now the default options. This automatically percolates to `BalanceDF.plot()` methods.
- `Sample.from_frame` can now guess that a column called "weights" is a weight column (instead of only guessing so if the column is called "weight").

## Bug Fixes
- Fix `rm_mutual_nas`: it now remembers the index of pandas.Series that were used as input. This fixed erroneous plots produced by seaborn functions which uses rm_mutual_nas.
- Fix `plot_hist_kde` to work when dist_type = "ecdf"
- Fix `plot_hist_kde` and `plot_bar` when having an input only with "self" and "target", by fixing `_return_sample_palette`.

## Misc
- All plotting functions moved internally to expect weight column to be called `weight`, instead of `weights`.
- All adjust (ipw, cbps, poststratify, null) functions now export a dict with a key called `weight` instead of `weights`.

## Contributors
@talgalili, @SarigT


0.6.0 (2023-04-05)
==================
## New Features
- Variance of the weighted mean
    - Add the `var_of_weighted_mean` function (from balance.stats_and_plots.weighted_stats import var_of_weighted_mean):
        Computes the variance of the weighted average (pi estimator for ratio-mean) of a list of values and their corresponding weights.
        - Added the `var_of_mean` option to stat in the `descriptive_stats` function (based on `var_of_weighted_mean`)
        - Added the `.var_of_mean()` method to BalanceDF.
    - Add the `ci_of_weighted_mean` function (from balance.stats_and_plots.weighted_stats import ci_of_weighted_mean):
        Computes the confidence intervals of the weighted mean using the (just added) variance of the weighted mean.
        - Added the `ci_of_mean` option to stat in the `descriptive_stats` function (based on `ci_of_weighted_mean`). Also added kwargs support.
        - Added the `.ci_of_mean()` method to BalanceDF.
        - Added the `.mean_with_ci()` method to BalanceDF.
        - Updated `.summary()` methods to include the output of `ci_of_mean`.
- All bar plots now have an added ylim argument to control the limits of the y axis.
    For example use: `plot_dist(dfs1, names=["self", "unadjusted", "target"], ylim = (0,1))`
    Or this: `s3_null.covars().plot(ylim = (0,1))`
- Improve 'choose_variables' function to control the order of the returned variables
    - The return type is now a list (and not a Tuple)
    - The order of the returned list is based on the variables argument. If it is not supplied, it is based on the order of the column names in the DataFrames. The df_for_var_order arg controls which df to use.
- Misc
    - The `_prepare_input_model_matrix` and downstream functions (e.g.: `model_matrix`, `sample.outcomes().mean()`, etc) can now handle DataFrame with special characters in the column names, by replacing special characters with '_' (or '_i', if we end up with columns with duplicate names). It also handles cases in which the column names have duplicates (using the new `_make_df_column_names_unique` function).
    - Improve choose_variables to control the order of the returned variables
        - The return type is now a list (and not a Tuple)
        - The order of the returned list is based on the variables argument. If it is not supplied, it is based on column names in the DataFrames. The df_for_var_order arg controls which df to use.

## Contributors
@talgalili, @SarigT


0.5.0 (2023-03-06)
==================
## New Features
- The `datasets.load_data` function now also supports the input "sim_data_cbps", which loads the simulated data used in the CBPS R vs Python tutorial. It is also used in unit-testing to compare the CBPS weights produced from Python (i.e.: balance) with R (i.e.: the CBPS package). The testing shows how the correlation of the weights from the two implementations (both Pearson and Spearman) produce a correlation of >0.98.
- cli improvements:
    - Add an option to set formula (as string) in the cli.

## Documentation
- New notebook (in the tutorial section):
    - Comparing results of fitting CBPS between R's `CBPS` package and Python's `balance` package (using simulated data). [link](https://import-balance.org/docs/tutorials/comparing_cbps_in_r_vs_python_using_sim_data/)

## Contributors
@stevemandala, @talgalili, @SarigT


0.4.0 (2023-02-08)
==================
## New Features
- Added two new flags to the cli:
    - `--standardize_types`: This gives cli users the ability to set the `standardize_types` parameter in Sample.from_frame
        to True or False. To learn more about this parameter, see:
        https://import-balance.org/api_reference/html/balance.sample_class.html#balance.sample_class.Sample.from_frame
    - `--return_df_with_original_dtypes`: the Sample object now stores the dtypes of the original df that was read using Sample.from_frame. This can be used to restore the original dtypes of the file output from the cli. This is relevant in cases in which we want to convert back the dtypes of columns from how they are stored in Sample, to their original types (e.g.: if something was Int32 it would be turned in float32 in balance.Sample, and using the new flag will return that column, when using the cli, to be back in the Int32 type). This feature may not be robust to various edge cases. So use with caution.
- In the logging:
    - Added warnings about dtypes changes. E.g.: if using Sample.from_frame with a column that has Int32, it will be turned into float32 in the internal storage of sample. Now there will be a warning message indicating of this change.
    - Increase the default length of logger printing (from 500 to 2000)


## Bug Fixes
- Fix pandas warning: SettingWithCopyWarning in from_frame (and other places in sample_class.py)
- sample.from_frame has a new argument `use_deepcopy` to decide if changes made to the df inside the sample object would also change the original df that was provided to the sample object. The default is now set to `True` since it's more likely that we'd like to keep the changes inside the sample object to the df contained in it, and not have them spill into the original df.

## Contributors
@SarigT, @talgalili


0.3.1 (2023-02-01)
==================
## Bug Fixes
- Sample.from_frame now also converts int16 and in8 to float16 and float16. Thus helping to avoid `TypeError: Cannot interpret 'Int16Dtype()' as a data type` style errors.

## Documentation
- Added ISSUE_TEMPLATE

## Contributors
@talgalili, @stevemandala, @SarigT


0.3.0 (2023-01-30)
==================
## New Features
- Added compatibility for Python 3.11 (by supporting SciPy 1.9.2) (props to @tomwagstaff-opml for flagging this issue).
- Added the `session-info` package as a dependency.

## Bug Fixes
- Fixed pip install from source on Windows machines (props to @tomwagstaff-opml for the bug report).

## Documentation
- Added `session_info.show()` outputs to the end of the three tutorials (at: https://import-balance.org/docs/tutorials/)
- Misc updates to the README.

## Contributors
@stevemandala, @SarigT, @talgalili


0.2.0 (2023-01-19)
==================
## New Features
- cli improvements:
    - Add an option to set weight_trimming_mean_ratio = None for no trimming.
    - Add an option to set transformations to be None (i.e. no transformations).
- Add an option to adapt the title in:
    - stats_and_plots.weighted_comparison_plots.plot_bar
    - stats_and_plots.weighted_comparison_plots.plot_hist_kde

## Bug Fixes
- Fix (and simplify) balanceDF.plot to organize the order of groups (now unadjusted/self is left, adjusted/self center, and target is on the right)
- Fix plotly functions to use the red color for self when only compared to target (since in that case it is likely unadjusted): balance.stats_and_plots.weighted_comparisons_plots.plotly_plot_qq and balance.stats_and_plots.weighted_comparisons_plots.plotly_plot_bar
- Fix seaborn_plot_dist: output None by default (instead of axis object). Added a return_Axes argument to control this behavior.
- Fix some test_cbps tests that were failing due to non-exact matches (we made the test less sensitive)

## Documentation
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

## Contributors
@stevemandala, @SarigT, @talgalili


0.1.0 (2022-11-20)
==================
## Summary
- balance released to the world!

## Contributors
@SarigT, @talgalili, @stevemandala
