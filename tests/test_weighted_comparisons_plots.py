# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from __future__ import absolute_import, division, print_function, unicode_literals

import balance.testutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.testing

from balance.sample_class import Sample
from balance.stats_and_plots import weighted_comparisons_plots, weighted_stats


s = Sample.from_frame(
    pd.DataFrame(
        {
            "a": (1, 2, 3),
            "b": (-42, 8, 2),
            "c": ("x", "y", "z"),
            "id": (1, 2, 3),
            "w": (0.5, 2, 1),
        }
    ),
    id_column="id",
    weight_column="w",
)

s2 = Sample.from_frame(
    pd.DataFrame(
        {
            "a": (1, 2, 3),
            "b": (17, 9, -3),
            "c": ("x", "y", "z"),
            "id": (1, 2, 3),
            "w": (0.5, 2, 1),
        }
    ),
    id_column="id",
    weight_column="w",
)


# TODO: split out the weighted_stats.relative_frequency_table function.
class Test_weighted_comparisons_plots(
    balance.testutil.BalanceTestCase,
):
    def test_plot_bar(self):
        import matplotlib.pyplot as plt
        from balance.stats_and_plots.weighted_comparisons_plots import plot_bar

        df = pd.DataFrame(
            {
                "group": ("a", "b", "c", "c"),
                "v1": (1, 2, 3, 4),
            }
        )

        plt.figure(1)
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        plot_bar(
            [
                {"df": df, "weights": pd.Series((1, 1, 1, 1))},
                {"df": df, "weights": pd.Series((2, 1, 1, 1))},
            ],
            names=["self", "target"],
            column="group",
            axis=ax,
            weighted=True,
        )

        self.assertIn("barplot of covar ", ax.get_title())

    def test_plot_dist_weighted_kde_error(self):
        df4 = {
            "df": pd.DataFrame(
                {"numeric_5": pd.Series([0, 0, 0, 0, 0, 1, 1, 2, 3, 4])}
            ),
            "weights": pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 5]),
        }

        plot_dist_type = type(
            weighted_comparisons_plots.plot_dist(
                (df4,),
                dist_type="kde",
                numeric_n_values_threshold=0,
                weighted=False,
                library="seaborn",
                return_axes=True,
            )[0]
        )

        # NOTE: There is no AxesSubplot class until one is invoked and created on the fly.
        #     See details here: https://stackoverflow.com/a/11690800/256662
        self.assertTrue(issubclass(plot_dist_type, plt.Axes))
        self.assertTrue(issubclass(plot_dist_type, matplotlib.axes.SubplotBase))

    def test_relative_frequency_table(self):
        df = pd.DataFrame({"a": list("abcd"), "b": list("bbcd")})

        e = pd.DataFrame({"a": list("abcd"), "prop": (0.25,) * 4})
        self.assertEqual(weighted_stats.relative_frequency_table(df, "a"), e)

        t = weighted_stats.relative_frequency_table(df, "a", pd.Series((1, 1, 1, 1)))
        self.assertEqual(t, e)

        e = pd.DataFrame({"b": list("bcd"), "prop": (0.5, 0.25, 0.25)})
        t = weighted_stats.relative_frequency_table(df, "b")
        self.assertEqual(t, e)

        t = weighted_stats.relative_frequency_table(df, "b", pd.Series((1, 1, 1, 1)))
        self.assertEqual(t, e)

        with self.assertRaisesRegex(TypeError, "must be a pandas Series"):
            weighted_stats.relative_frequency_table(df, "a", 1)

        e = pd.DataFrame({"a": list("abcd"), "prop": (0.2, 0.4, 0.2, 0.2)})
        t = weighted_stats.relative_frequency_table(df, "a", pd.Series((1, 2, 1, 1)))
        self.assertEqual(t, e)

        e = pd.DataFrame({"b": list("bcd"), "prop": (1 / 3, 1 / 3, 1 / 3)})
        t = weighted_stats.relative_frequency_table(
            df, "b", pd.Series((0.5, 0.5, 1, 1))
        )
        self.assertEqual(t, e)

        df = pd.DataFrame(
            {
                "group": ("a", "b", "c", "c"),
                "v1": (1, 2, 3, 4),
            }
        )
        t = weighted_stats.relative_frequency_table(
            df,
            "group",
            pd.Series((2, 1, 1, 1)),
        ).to_dict()
        e = {"group": {0: "a", 1: "b", 2: "c"}, "prop": {0: 0.4, 1: 0.2, 2: 0.4}}
        self.assertEqual(t, e)
        # check that using a pd.Series will give the same results
        t_series = weighted_stats.relative_frequency_table(
            df=df["group"],
            w=pd.Series((2, 1, 1, 1)),
        ).to_dict()
        self.assertEqual(t_series, e)

    def test_naming_plot(self):
        self.assertEqual(
            weighted_comparisons_plots.naming_legend(
                "self", ["self", "target", "unadjusted"]
            ),
            "adjusted",
        )
        self.assertEqual(
            weighted_comparisons_plots.naming_legend(
                "unadjusted", ["self", "target", "unadjusted"]
            ),
            "sample",
        )
        self.assertEqual(
            weighted_comparisons_plots.naming_legend("self", ["self", "target"]),
            "sample",
        )
        self.assertEqual(
            weighted_comparisons_plots.naming_legend("other_name", ["self", "target"]),
            "other_name",
        )

    def test_seaborn_plot_dist(self):
        df4 = {
            "df": pd.DataFrame(
                {"numeric_5": pd.Series([0, 0, 0, 0, 0, 1, 1, 2, 3, 4])}
            ),
            "weights": pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 5]),
        }

        # Check that we get a list of matplotlib axes
        out = []
        for dist_type in ("hist", "kde", "qq", "ecdf"):
            out.append(
                type(
                    weighted_comparisons_plots.seaborn_plot_dist(
                        (df4,),
                        names=["test"],
                        dist_type=dist_type,
                        return_axes=True,
                    )[0]
                )
            )

        # NOTE: There is no AxesSubplot class until one is invoked and created on the fly.
        #     See details here: https://stackoverflow.com/a/11690800/256662
        self.assertTrue(issubclass(out[0], matplotlib.axes.SubplotBase))
        self.assertTrue(issubclass(out[1], matplotlib.axes.SubplotBase))
        self.assertTrue(issubclass(out[2], matplotlib.axes.SubplotBase))
        self.assertTrue(issubclass(out[3], matplotlib.axes.SubplotBase))

    def test_plot_dist(self):

        import plotly.graph_objs as go
        from balance.stats_and_plots.weighted_comparisons_plots import plot_dist
        from numpy import random

        random.seed(96483)

        df = pd.DataFrame(
            {
                "v1": random.random_integers(11111, 11114, size=100).astype(str),
                "v2": random.normal(size=100),
                "v3": random.uniform(size=100),
            }
        ).sort_values(by=["v2"])

        dfs1 = [
            {"df": df, "weights": pd.Series(np.ones(100))},
            {"df": df, "weights": pd.Series(np.ones(99).tolist() + [1000])},
            {"df": df, "weights": pd.Series(np.ones(100))},
        ]

        # If plot_it=False and
        self.assertTrue(
            plot_dist(
                dfs1,
                names=["self", "unadjusted", "target"],
                library="plotly",
                plot_it=False,
            )
            is None
        )

        # check the dict of figures returned:
        dict_of_figures = plot_dist(
            dfs1,
            names=["self", "unadjusted", "target"],
            library="plotly",
            plot_it=False,
            return_dict_of_figures=True,
        )

        self.assertEqual(type(dict_of_figures), dict)
        self.assertEqual(sorted(dict_of_figures.keys()), ["v1", "v2", "v3"])
        self.assertEqual(type(dict_of_figures["v1"]), go.Figure)

        with self.assertRaisesRegex(ValueError, "library must be either*"):
            plot_dist(
                dfs1,
                names=["self", "unadjusted", "target"],
                library="ploting_library_which_is_not_plotly_or_seaborn",
            )
