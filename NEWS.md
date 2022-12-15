0.1.1
==================
### New Features
- cli improvements: Add an option to set weight_trimming_mean_ratio = None for no trimming.
- cli improvements:: Add an option to set transformations to be None (i.e. no transformations).
- stats_and_plots.weighted_comparison_plots.plot_bar: Add an option to adapt the title.
- stats_and_plots.weighted_comparison_plots.plot_hist_kde: Add an option to adapt the title.

### Bug Fixes
- Fix (and simplify) balanceDF.plot to organize the order of groups (now unadjusted/self is left, adjusted/self center, and target is on the right)
- Fix plotly functions to use the red color for self when only compared to target (since in that case it is likely unadjusted): balance.stats_and_plots.weighted_comparisons_plots.plotly_plot_qq and balance.stats_and_plots.weighted_comparisons_plots.plotly_plot_bar
- Fix seaborn_plot_dist: output None by default (instead of axis object). Added a return_Axes argument to control this behavior.


0.1.0 (2022-11-20)
==================
### Summary
- balance released to the world!
