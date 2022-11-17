# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

import logging
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from balance import util as balance_util
from balance.adjustment import trim_weights
from balance.sample_class import Sample
from balance.stats_and_plots import (
    general_stats,
    weighted_comparisons_plots,
    weighted_comparisons_stats,
    weighted_stats,
    weights_stats,
)

from balance.typing import FilePathOrBuffer

from IPython.lib.display import FileLink

logger: "logging.Logger" = logging.getLogger(__package__)


class BalanceDF:
    """
    Wrapper class around a Sample which provides additional balance-specific functionality
    """

    _model_matrix = None

    def __init__(
        self: "BalanceDF",
        df: pd.DataFrame,
        sample: Sample,
        name: Literal["outcomes", "weights", "covars"],
    ) -> None:
        """A basic init method used by BalanceOutcomesDF,BalanceCovarsDF, and BalanceWeightsDF

        Args:
            self (BalanceDF): The object that is initiated.
            df (pd.DataFrame): a df from a sample object.
            sample (Sample): A sample object to be stored as reference.
            name (Literal["outcomes", "weights", "covars"]): The type of object that will be created. In practice, used for "outcomes", "weights" and "covars".
        """
        # NOTE: double underscore helps to add friction so that users do not change these objects.
        #       see details here: https://stackoverflow.com/a/1301369/256662
        # TODO: when refactoring the object class model, re-evaluate if we want to keep such objects protected or not.
        self.__sample = sample
        self.__df = df
        self.__name = name

    def __str__(self: "BalanceDF") -> str:
        name = self.__name
        sample = object.__repr__(self._sample)
        df = self.df.__repr__()
        return f"{name} from {sample}:\n{df}"

    def __repr__(self: "BalanceDF") -> str:
        return (
            f"({self.__class__.__module__}.{self.__class__.__qualname__})\n"
            f"{self.__str__()}"
        )

    #  Private API
    @staticmethod
    def _check_if_not_BalanceDF(
        BalanceDF_class_obj: "BalanceDF", object_name: str = "sample_BalanceDF"
    ) -> None:
        """Check if an object is BalanceDF, if not then it raises ValueError

        Args:
            BalanceDF_class_obj (BalanceDF): Object to check.
            object_name (str, optional): Object name (to use when raising the ValueError). Defaults to "sample_BalanceDF".

        Returns: None.

        Raises:
            ValueError: if BalanceDF_class_obj is not BalanceDF object.
        """
        if not isinstance(BalanceDF_class_obj, BalanceDF):
            raise ValueError(
                f"{object_name} must be balancedf_class.BalanceDF, is {type(BalanceDF_class_obj)}"
            )

    @property
    def _sample(self: "BalanceDF") -> "Sample":
        """Access __sample internal object.

        Args:
            self (BalanceDF): Object

        Returns:
            Sample: __sample
        """
        return self.__sample

    @property
    def _weights(
        self: "BalanceDF",
    ) -> Optional[pd.DataFrame]:
        """Access the weight_column in __sample.

        Args:
            self (BalanceDF): Object

        Returns:
            Optional[pd.DataFrame]: The weights (with no column name)
        """
        w = self._sample.weight_column
        return w.rename(None)

    # NOTE: only in the case of BalanceOutcomesDF can it result in a None value.
    def _BalanceDF_child_from_linked_samples(
        self: "BalanceDF",
    ) -> Dict[
        str,
        Union["BalanceCovarsDF", "BalanceWeightsDF", Union["BalanceOutcomesDF", None]],
    ]:
        """Returns a dict with self and the same type of BalanceDF_child when created from the linked samples.

        For example, if this function is called from a BalanceCovarsDF (originally created using `Sample.covars()`),
        that was invoked by a Sample with a target then the return dict will have the keys 'self' and 'target',
        with the BalanceCovarsDF of the self and that of the target.

        If the object has nothing but self, then it will be a dict with only one key:value pair (of self).

        Args:
            self (BalanceDF): Object (used in practice only with children of BalanceDF).


        Returns:
            Dict[str, Union["BalanceCovarsDF", "BalanceWeightsDF", Union["BalanceOutcomesDF", None]],]:
            A dict mapping the link relationship to the result.
                First item is self, and it just returns it without using method on it.
                The other items are based on the objects in _links. E.g.: it can be 'target'
                and 'unadjusted', and it will return them after running the same BalanceDF child creation method on them.
        Examples:
            ::
                from balance.sample_class import Sample
                import pandas as pd

                s1 = Sample.from_frame(
                    pd.DataFrame(
                        {
                            "a": (1, 2, 3, 1),
                            "b": (-42, 8, 2, -42),
                            "o": (7, 8, 9, 10),
                            "c": ("x", "y", "z", "v"),
                            "id": (1, 2, 3, 4),
                            "w": (0.5, 2, 1, 1),
                        }
                    ),
                    id_column="id",
                    weight_column="w",
                    outcome_columns="o",
                )

                s2 = Sample.from_frame(
                    pd.DataFrame(
                        {
                            "a": (1, 2, 3),
                            "b": (4, 6, 8),
                            "id": (1, 2, 3),
                            "w": (0.5, 1, 2),
                            "c": ("x", "y", "z"),
                        }
                    ),
                    id_column="id",
                    weight_column="w",
                )

                s3 = s1.set_target(s2)
                s3_null = s3.adjust(method="null")


                # keys depends on which samples are linked to the object:
                list(s1.covars()._BalanceDF_child_from_linked_samples().keys())  # ['self']
                list(s3.covars()._BalanceDF_child_from_linked_samples().keys())  # ['self', 'target']
                list(s3_null.covars()._BalanceDF_child_from_linked_samples().keys())  # ['self', 'target', 'unadjusted']

                # Indeed, all are of the same BalanceDF child type:
                s3.covars()._BalanceDF_child_from_linked_samples()
                # {'self': (balance.balancedf_class.BalanceCovarsDF)
                # covars from <balance.sample_class.Sample object at 0x7f4392ea61c0>:
                #     a   b  c
                # 0  1 -42  x
                # 1  2   8  y
                # 2  3   2  z
                # 3  1 -42  v,
                # 'target': (balance.balancedf_class.BalanceCovarsDF)
                # covars from <balance.sample_class.Sample object at 0x7f43958fbd90>:
                #     a  b  c
                # 0  1  4  x
                # 1  2  6  y
                # 2  3  8  z}

                s3_null.covars()._BalanceDF_child_from_linked_samples()
                # {'self': (balance.balancedf_class.BalanceCovarsDF)
                # covars from <balance.sample_class.Sample object at 0x7f4392ea60d0>:
                #     a   b  c
                # 0  1 -42  x
                # 1  2   8  y
                # 2  3   2  z
                # 3  1 -42  v,
                # 'target': (balance.balancedf_class.BalanceCovarsDF)
                # covars from <balance.sample_class.Sample object at 0x7f43958fbd90>:
                #     a  b  c
                # 0  1  4  x
                # 1  2  6  y
                # 2  3  8  z,
                # 'unadjusted': (balance.balancedf_class.BalanceCovarsDF)
                # covars from <balance.sample_class.Sample object at 0x7f4392ea61c0>:
                #     a   b  c
                # 0  1 -42  x
                # 1  2   8  y
                # 2  3   2  z
                # 3  1 -42  v}

                the_dict = s3_null.covars()._BalanceDF_child_from_linked_samples()
                [v.__class__ for (k,v) in the_dict.items()]
                [balance.balancedf_class.BalanceCovarsDF,
                balance.balancedf_class.BalanceCovarsDF,
                balance.balancedf_class.BalanceCovarsDF]


                # This also works for outcomes (returns None if there is none):
                s3.outcomes()._BalanceDF_child_from_linked_samples()
                # {'self': (balance.balancedf_class.BalanceOutcomesDF)
                #  outcomes from <balance.sample_class.Sample object at 0x7f4392ea61c0>:
                #      o
                #  0   7
                #  1   8
                #  2   9
                #  3  10,
                #  'target': None}

                # And also works for weights:
                s3.weights()._BalanceDF_child_from_linked_samples()
                # {'self': (balance.balancedf_class.BalanceWeightsDF)
                #  weights from <balance.sample_class.Sample object at 0x7f4392ea61c0>:
                #       w
                #  0  0.5
                #  1  2.0
                #  2  1.0
                #  3  1.0,
                #  'target': (balance.balancedf_class.BalanceWeightsDF)
                #  weights from <balance.sample_class.Sample object at 0x7f43958fbd90>:
                #       w
                #  0  0.5
                #  1  1.0
                #  2  2.0}
        """
        # NOTE: this assumes that the .__name is the same as the creation method (i.e.: .covars(), .weights(), .outcomes())
        BalanceDF_child_method = self.__name
        d = {"self": self}
        d.update(
            {
                k: getattr(v, BalanceDF_child_method)()
                for k, v in self._sample._links.items()
            }
        )
        return d  # pyre-fixme[7]: this returns what's declared `Dict[str, Union[BalanceCovarsDF, BalanceOutcomesDF, BalanceWeightsDF]]` but got `Dict[str, BalanceDF]`

    def _call_on_linked(
        self: "BalanceDF",
        method: str,
        exclude: Union[Tuple[str], Tuple] = (),
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """Call a given method on the linked DFs of the BalanceDF object.
        Returns the result as a pandas DataFrame where the source column
        indicates where the result came from

        Args:
            self (BalanceDF): Object.
            method (str): A name of a method to call (e.g.: "mean", "std", etc.).
                Can also be a name of an attribute that is a DataFrame (e.g.: 'df')
            exclude (Tuple[str], optional): A tuple of strings which indicates which source should be excluded from the output. Defaults to ().
                E.g.: "self", "target".

        Returns:
            pd.DataFrame: A pandas DataFrame where the source column
                indicates where the result came from. E.g.: 'self', 'target', 'unadjusted'.
                And the columns are based on the method called. E.g.: using mean will give the
                per column mean, after applying `model_matrix` to the df from each object.

        Examples:
            ::

                from balance.sample_class import Sample
                import pandas as pd

                s1 = Sample.from_frame(
                    pd.DataFrame(
                        {
                            "a": (1, 2, 3, 1),
                            "b": (-42, 8, 2, -42),
                            "o": (7, 8, 9, 10),
                            "c": ("x", "y", "z", "v"),
                            "id": (1, 2, 3, 4),
                            "w": (0.5, 2, 1, 1),
                        }
                    ),
                    id_column="id",
                    weight_column="w",
                    outcome_columns="o",
                )

                s2 = Sample.from_frame(
                    pd.DataFrame(
                        {
                            "a": (1, 2, 3),
                            "b": (4, 6, 8),
                            "id": (1, 2, 3),
                            "w": (0.5, 1, 2),
                            "c": ("x", "y", "z"),
                        }
                    ),
                    id_column="id",
                    weight_column="w",
                )

                s3 = s1.set_target(s2)

                print(s3.covars()._call_on_linked("mean").round(3))
                    #             a       b   c[v]   c[x]   c[y]   c[z]
                    # source
                    # self    1.889 -10.000  0.222  0.111  0.444  0.222
                    # target  2.429   6.857    NaN  0.143  0.286  0.571

                print(s3.covars()._call_on_linked("df").round(3))
                    #         a   b  c
                    # source
                    # self    1 -42  x
                    # self    2   8  y
                    # self    3   2  z
                    # self    1 -42  v
                    # target  1   4  x
                    # target  2   6  y
                    # target  3   8  z
        """
        output = []
        for k, v in self._BalanceDF_child_from_linked_samples().items():
            if v is not None and k not in exclude:
                v_att_method = getattr(v, method)
                if callable(v_att_method):
                    v_att_method = v_att_method(
                        on_linked_samples=False, *args, **kwargs
                    )
                output.append(v_att_method.assign(source=k).set_index("source"))

        return pd.concat(output)

        # return pd.concat(
        #     getattr(v, method)(on_linked_samples=False, *args, **kwargs)
        #     .assign(source=k)
        #     .set_index("source")
        #     if callable(getattr(v, method))
        #     else getattr(v, method)(on_linked_samples=False, *args, **kwargs)
        #     .assign(source=k)
        #     .set_index("source")
        #     for k, v in self._BalanceDF_child_from_linked_samples().items()
        #     if v is not None and k not in exclude
        # )

    # TODO: add the ability to pass formula argument to model_matrix
    #       but in which case - notice that we'd want the ability to track
    #       which object is stored in _model_matrix (and to run it over)
    #       Also, the output may sometimes no longer only be pd.DataFrame
    #       so such work will require update the type hinting here.
    def model_matrix(self: "BalanceDF") -> pd.DataFrame:
        """Return a model_matrix version of the df inside the BalanceDF object using balance_util.model_matrix

        This can be used to turn all character columns into a one hot encoding columns.

        Args:
            self (BalanceDF): Object

        Returns:
            pd.DataFrame: The output from :func:`balance_util.model_matrix`

        Examples:
            ::

                import pandas as pd
                from balance.sample_class import Sample

                s1 = Sample.from_frame(
                    pd.DataFrame(
                        {
                            "a": (1, 2, 3, 1),
                            "b": (-42, 8, 2, -42),
                            "o": (7, 8, 9, 10),
                            "c": ("x", "y", "z", "v"),
                            "id": (1, 2, 3, 4),
                            "w": (0.5, 2, 1, 1),
                        }
                    ),
                    id_column="id",
                    weight_column="w",
                    outcome_columns="o",
                )

                print(s1.covars().df)
                    # a   b  c
                    # 0  1 -42  x
                    # 1  2   8  y
                    # 2  3   2  z
                    # 3  1 -42  v

                print(s1.covars().model_matrix())
                    #      a     b  c[v]  c[x]  c[y]  c[z]
                    # 0  1.0 -42.0   0.0   1.0   0.0   0.0
                    # 1  2.0   8.0   0.0   0.0   1.0   0.0
                    # 2  3.0   2.0   0.0   0.0   0.0   1.0
                    # 3  1.0 -42.0   1.0   0.0   0.0   0.0
        """
        if not hasattr(self, "_model_matrix") or self._model_matrix is None:
            self._model_matrix = balance_util.model_matrix(
                self.df, add_na=True, return_type="one"
            )["model_matrix"]
        return self._model_matrix

    def _descriptive_stats(
        self: "BalanceDF",
        stat: Literal["mean", "std", "..."] = "mean",
        weighted: bool = True,
        numeric_only: bool = False,
        add_na: bool = True,
    ) -> pd.DataFrame:
        """
        Calls a given method from :func:`weighted_stats.descriptive_stats` on 'self'.
        This function knows how to extract the df and the weights from a BalanceDF object.

        Args:
            self (BalanceDF): An object to run stats on.
            stat (Literal["mean", "std", "..."], optional): Defaults to "mean".
            weighted (bool, optional): Defaults to True.
            numeric_only (bool, optional): Defaults to False.
            add_na (bool, optional): Defaults to True.

        Returns:
            pd.DataFrame: Returns pd.DataFrame of the output (based on stat argument), for each of the columns in df.
        """
        if numeric_only:
            df = self.df.select_dtypes(include=[np.number])
        else:
            df = self.model_matrix()

        weights = (
            self._weights.values if (weighted and self._weights is not None) else None
        )
        wdf = weighted_stats.descriptive_stats(
            df,
            weights,
            stat,
            weighted=weighted,
            # Using numeric_only=True since we know that df is screened in this function
            # To only include numeric variables. So this saves descriptive_stats from
            # running model_matrix again.
            numeric_only=True,
            add_na=add_na,
        )
        return wdf

    def to_download(self: "BalanceDF", tempdir: Optional[str] = None) -> FileLink:
        """Creates a downloadable link of the DataFrame, with ids, of the BalanceDF object.

        File name starts with tmp_balance_out_, and some random file name (using :func:`uuid.uuid4`).

        Args:
            self (BalanceDF): Object.
            tempdir (Optional[str], optional): Defaults to None (which then uses a temporary folder using :func:`tempfile.gettempdir`).

        Returns:
            FileLink: Embedding a local file link in an IPython session, based on path. Using :func:FileLink.
        """
        return balance_util._to_download(self._df_with_ids(), tempdir)

    #  Public API
    #  All these functions operate on multiple samples
    @property
    def df(self: "BalanceDF") -> pd.DataFrame:
        """
        Get the df of the BalanceDF object.

        The df is stored in the BalanceDF.__df object, that is set during the __init__ of the object.

        Args:
            self (BalanceDF): The object.

        Returns:
            pd.DataFrame: The df (this is __df, with no weights) from the BalanceDF object.
        """
        return self.__df

    def names(self: "BalanceDF") -> List:
        """Returns the column names of the DataFrame (df) inside a BalanceDF object.

        Args:
            self (BalanceDF): The object.

        Returns:
            List: Of column names.

        Examples:
            ::

                s1 = Sample.from_frame(
                    pd.DataFrame(
                        {
                            "a": (1, 2, 3, 1),
                            "b": (-42, 8, 2, -42),
                            "o": (7, 8, 9, 10),
                            "c": ("x", "y", "z", "v"),
                            "id": (1, 2, 3, 4),
                            "w": (0.5, 2, 1, 1),
                        }
                    ),
                    id_column="id",
                    weight_column="w",
                    outcome_columns="o",
                )

                s1.covars().names()
                # ['a', 'b', 'c']
                s1.weights().names()
                # ['w']
                s1.outcomes().names()
                # ['o']
        """
        return list(self.df.columns.values)

    def plot(
        self: "BalanceDF", on_linked_samples: bool = True, **kwargs
    ) -> Union[Union[List, np.ndarray], Dict[str, go.Figure]]:
        """Plots the variables in the df of the BalanceDF object.

        See :func:`weighted_comparisons_plots.plot_dist` for details of various arguments that can be passed.
        The default plotting engine is plotly, but seaborn can be used for static plots.

        This function is inherited as is when invoking BalanceCovarsDF.plot, but some modifications are made when
        preparing the data for BalanceOutcomesDF.plot and BalanceWeightsDF.plot.

        Args:
            self (BalanceDF): Object (used in the plots as "sample" or "self")
            on_linked_samples (bool, optional): Determines if the linked samples should be included in the plot.
                Defaluts to True.

        Returns:
            Union[Union[List, np.ndarray], Dict[str, go.Figure]]:
                If library="plotly" then returns a dictionary containing plots if return_dict_of_figures is True. None otherwise.
                If library="seaborn" then returns either a list or an np.array of matplotlib axis.

        Examples:
            ::

                import numpy as np
                import pandas as pd
                from numpy import random
                from balance.sample_class import Sample

                random.seed(96483)

                df = pd.DataFrame({
                    "id": range(100),
                    'v1': random.random_integers(11111, 11114, size=100).astype(str),
                    'v2': random.normal(size = 100),
                    'v3': random.uniform(size = 100),
                    "w": pd.Series(np.ones(99).tolist() + [1000]),
                }).sort_values(by=['v2'])

                s1 = Sample.from_frame(df,
                    id_column="id",
                    weight_column="w",
                )

                s2 = Sample.from_frame(
                    df.assign(w = pd.Series(np.ones(100))),
                    id_column="id",
                    weight_column="w",
                )

                s3 = s1.set_target(s2)
                s3_null = s3.adjust(method="null")
                s3_null.set_weights(random.random(size = 100) + 0.5)

                s3_null.covars().plot()
                s3_null.covars().plot(library = "seaborn")
        """
        if on_linked_samples:
            dfs_to_add = self._BalanceDF_child_from_linked_samples()
        else:
            dfs_to_add = {"self": self}
        dfs = [
            {"df": v.df, "weights": v._weights}
            for k, v in dfs_to_add.items()
            if (v is not None) and (k != "target")
        ]
        names = [k for k in dfs_to_add.keys() if k != "target"]

        # NOTE: "target", if exists, is placed at the end of the dict so that comparative plotting functions,
        # specifically :func:`plot_qq`, will deal with it properly.
        self_target = dfs_to_add.get("target")
        if self_target is not None:
            dfs.append({"df": self_target.df, "weights": self_target._weights})
            names.append("target")

        # pyre-ignore[6]: there is no concern that dfs can have None instead of a DataFrame
        #                 the only worry is if target is not available, but the function deals
        #                 with it when dfs is first define by ignoring "target", and then only adding it if
        #                 target is available (dfs_to_add.get("target")).
        return weighted_comparisons_plots.plot_dist(dfs, names=names, **kwargs)

    #  NOTE: The following functions use the _call_on_linked method
    #        to return information about the characteristics of linked Samples
    def mean(
        self: "BalanceDF", on_linked_samples: bool = True, **kwargs
    ) -> pd.DataFrame:
        """Calculates a weighted mean on the df of the BalanceDF object.

        Args:
            self (BalanceDF): Object.
            on_linked_samples (bool, optional): Should the calculation be on self AND the linked samples objects? Defaults to True.
                If True, then uses :func:`_call_on_linked` with method "mean".
                If False, then uses :func:`_descriptive_stats` with method "mean".

        Returns:
            pd.DataFrame:
                With row per object: self if on_linked_samples=False, and self and others (e.g.: target and unadjusted) if True.
                Columns are for each of the columns in the relevant df (after applying :func:`model_matrix`)

        Examples:
            ::

                import pandas as pd
                from balance.sample_class import Sample

                s1 = Sample.from_frame(
                    pd.DataFrame(
                        {
                            "a": (1, 2, 3, 1),
                            "b": (-42, 8, 2, -42),
                            "o": (7, 8, 9, 10),
                            "c": ("x", "y", "z", "v"),
                            "id": (1, 2, 3, 4),
                            "w": (0.5, 2, 1, 1),
                        }
                    ),
                    id_column="id",
                    weight_column="w",
                    outcome_columns="o",
                )

                s2 = Sample.from_frame(
                    pd.DataFrame(
                        {
                            "a": (1, 2, 3),
                            "b": (4, 6, 8),
                            "id": (1, 2, 3),
                            "w": (0.5, 1, 2),
                            "c": ("x", "y", "z"),
                        }
                    ),
                    id_column="id",
                    weight_column="w",
                )

                s3 = s1.set_target(s2)
                s3_null = s3.adjust(method="null")

                print(s3_null.covars().mean())

                #                 a          b      c[v]      c[x]      c[y]      c[z]
                # source
                # self        1.888889 -10.000000  0.222222  0.111111  0.444444  0.222222
                # target      2.428571   6.857143       NaN  0.142857  0.285714  0.571429
                # unadjusted  1.888889 -10.000000  0.222222  0.111111  0.444444  0.222222
        """
        if on_linked_samples:
            return self._call_on_linked("mean", **kwargs)
        else:
            return self._descriptive_stats("mean", **kwargs)

    def std(
        self: "BalanceDF", on_linked_samples: bool = True, **kwargs
    ) -> pd.DataFrame:
        """Calculates a weighted std on the df of the BalanceDF object.

        Args:
            self (BalanceDF): Object.
            on_linked_samples (bool, optional): Should the calculation be on self AND the linked samples objects? Defaults to True.
                If True, then uses :func:`_call_on_linked` with method "std".
                If False, then uses :func:`_descriptive_stats` with method "std".

        Returns:
            pd.DataFrame:
                With row per object: self if on_linked_samples=False, and self and others (e.g.: target and unadjusted) if True.
                Columns are for each of the columns in the relevant df (after applying :func:`model_matrix`)

        Examples:
            ::

                import pandas as pd
                from balance.sample_class import Sample

                s1 = Sample.from_frame(
                    pd.DataFrame(
                        {
                            "a": (1, 2, 3, 1),
                            "b": (-42, 8, 2, -42),
                            "o": (7, 8, 9, 10),
                            "c": ("x", "y", "z", "v"),
                            "id": (1, 2, 3, 4),
                            "w": (0.5, 2, 1, 1),
                        }
                    ),
                    id_column="id",
                    weight_column="w",
                    outcome_columns="o",
                )

                s2 = Sample.from_frame(
                    pd.DataFrame(
                        {
                            "a": (1, 2, 3),
                            "b": (4, 6, 8),
                            "id": (1, 2, 3),
                            "w": (0.5, 1, 2),
                            "c": ("x", "y", "z"),
                        }
                    ),
                    id_column="id",
                    weight_column="w",
                )

                s3 = s1.set_target(s2)
                s3_null = s3.adjust(method="null")

                print(s3_null.covars().std())

                    #                 a          b  c[v]      c[x]      c[y]      c[z]
                    # source
                    # self        0.886405  27.354812   0.5  0.377964  0.597614  0.500000
                    # target      0.963624   1.927248   NaN  0.462910  0.597614  0.654654
                    # unadjusted  0.886405  27.354812   0.5  0.377964  0.597614  0.500000
        """
        if on_linked_samples:
            return self._call_on_linked("std", **kwargs)
        else:
            return self._descriptive_stats("std", **kwargs)

    # NOTE: Summary could return also an str in case it is overridden in other children's methods.
    def summary(
        self: "BalanceDF", on_linked_samples: bool = True
    ) -> Union[pd.DataFrame, str]:
        """Returns a summary of BalanceDF class.

        Currently just uses :func:`BalanceDF.mean`. In the future this may be extended.

        Args:
            self (BalanceDF): Object.
            on_linked_samples (bool, optional): Passed to :func:`BalanceDF.mean`. Defaults to True.

        Returns:
            Union[pd.DataFrame, str]: :func:`BalanceDF.mean`.
        """
        # TODO model matrix means to include categorical columns, fix model_matrix to accept DataFrame
        # TODO: inlcude min/max/std/etc. show min/mean/max if there's a single column, just means if multiple (covars and outcomes)
        #       Doing so would either require to implement a min/max etc methods in BalanceDF and use them with _call_on_linked.
        #       Or, update _call_on_linked to deal with non functions, get 'df' from it, and apply the needed functions on it.
        # TODO add outcome variance ratio
        return self.mean(on_linked_samples)

    def _get_df_and_weights(
        self: "BalanceDF",
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Extract covars df (after using model_matrix) and weights from a BalanceDF object.

        Args:
            self (BalanceDF): Object

        Returns:
            Tuple[pd.DataFrame, Optional[np.ndarray]]:
                A pd.DataFrame output from running :func:`model_metrix`, and
                A np.ndarray of weights from :func:`_weights`, or just None (if there are no weights).
        """
        # get df values (like in BalanceDF._descriptive_stats)
        df_model_metrix = self.model_matrix()
        # get weights (like in BalanceDF._descriptive_stats)
        weights = self._weights.values if (self._weights is not None) else None
        return df_model_metrix, weights

    @staticmethod
    def _asmd_BalanceDF(
        sample_BalanceDF: "BalanceDF",
        target_BalanceDF: "BalanceDF",
        aggregate_by_main_covar: bool = False,
    ) -> pd.Series:
        """Run asmd on two BalanceDF objects

        Prepares the BalanceDF objects by passing them through :func:`_get_df_and_weights`, and
        then pass the df and weights from the two objects into :func:`weighted_comparisons_stats.asmd`.

        Note that this will works on the result of model_matrix (default behavior, no formula supplied),
        which is different than just the raw covars. E.g.: in case there are nulls (will produce an indicator column of that),
        as well as if there are categorical variables (transforming them using one hot encoding).

        Args:
            sample_df (BalanceDF): Object
            target_df (BalanceDF): Object
            aggregate_by_main_covar (bool, optional): See :func:`weighted_comparisons_stats.asmd`. Defaults to False.

        Returns:
            pd.Series: See :func:`weighted_comparisons_stats.asmd`

        Examples:
            ::

                from balance.balancedf_class import BalanceDF

                BalanceDF._asmd_BalanceDF(
                    Sample.from_frame(
                        pd.DataFrame(
                            {"id": (1, 2), "a": (1, 2), "b": (-1, 12), "weight": (1, 2)}
                        )
                    ).covars(),
                    Sample.from_frame(
                        pd.DataFrame(
                            {"id": (1, 2), "a": (3, 4), "b": (0, 42), "weight": (1, 2)}
                        )
                    ).covars(),
                )

                    # a             2.828427
                    # b             0.684659
                    # mean(asmd)    1.756543
                    # dtype: float64
        """
        BalanceDF._check_if_not_BalanceDF(sample_BalanceDF, "sample_BalanceDF")
        BalanceDF._check_if_not_BalanceDF(sample_BalanceDF, "target_BalanceDF")

        sample_df_values, sample_weights = sample_BalanceDF._get_df_and_weights()
        target_df_values, target_weights = target_BalanceDF._get_df_and_weights()

        return weighted_comparisons_stats.asmd(
            sample_df_values,
            target_df_values,
            sample_weights,
            target_weights,
            std_type="target",
            aggregate_by_main_covar=aggregate_by_main_covar,
        )

    def asmd(
        self: "BalanceDF",
        on_linked_samples: bool = True,
        target: Optional["BalanceDF"] = None,
        aggregate_by_main_covar: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """ASMD is the absolute difference of the means of two groups (say, P and T), divided by some standard deviation (std).
        It can be std of P or of T, or of P and T.
        These are all variations on the absolute value of cohen's d (see: https://en.wikipedia.org/wiki/Effect_size#Cohen's_d).

        We can use asmd to compares multiple Samples (with and without adjustment) to a target population.

        Args:
            self (BalanceDF): Object from sample (with/without adjustment, but it needs some target)
            on_linked_samples (bool, optional): If to comapre also to linked sample objects (specifically: unadjusted), or not. Defaults to True.
            target (Optional["BalanceDF"], optional): A BalanceDF (of the same type as the one used in self) to compare against.
                If None then it looks for a target in the self linked objects. Defaults to None.
            aggregate_by_main_covar (bool, optional): Defaults to False.
                If True, it will make sure to return the asmd DataFrame after averaging
                all the columns from using the one-hot encoding for categorical variables.
                See ::_aggregate_asmd_by_main_covar:: for more details.

        Raises:
            ValueError:
                If self has no target and no target is supplied.

        Returns:
            pd.DataFrame:
                If on_linked_samples is False, then only one row (index name depends on BalanceDF type, e.g.: covars), with asmd of self vs the target (depending if it's covars, or something else).
                If on_linked_samples is True, then two rows per source (self, unadjusted), each with the asmd compared to target, and a third row for the difference (self-unadjusted).

        Examples:
            ::

                import pandas as pd
                from balance.sample_class import Sample

                from copy import deepcopy


                s1 = Sample.from_frame(
                    pd.DataFrame(
                        {
                            "a": (1, 2, 3, 1),
                            "b": (-42, 8, 2, -42),
                            "o": (7, 8, 9, 10),
                            "c": ("x", "y", "z", "v"),
                            "id": (1, 2, 3, 4),
                            "w": (0.5, 2, 1, 1),
                        }
                    ),
                    id_column="id",
                    weight_column="w",
                    outcome_columns="o",
                )

                s2 = Sample.from_frame(
                    pd.DataFrame(
                        {
                            "a": (1, 2, 3),
                            "b": (4, 6, 8),
                            "id": (1, 2, 3),
                            "w": (0.5, 1, 2),
                            "c": ("x", "y", "z"),
                        }
                    ),
                    id_column="id",
                    weight_column="w",
                )

                s3 = s1.set_target(s2)
                s3_null = s3.adjust(method="null")

                s3_null_madeup_weights = deepcopy(s3_null)
                s3_null_madeup_weights.set_weights((1, 2, 3, 1))

                print(s3_null.covars().asmd().round(3))
                    #                     a      b  c[v]   c[x]   c[y]   c[z]  mean(asmd)
                    # source
                    # self               0.56  8.747   NaN  0.069  0.266  0.533       3.175
                    # unadjusted         0.56  8.747   NaN  0.069  0.266  0.533       3.175
                    # unadjusted - self  0.00  0.000   NaN  0.000  0.000  0.000       0.000

                # show that on_linked_samples = False works:
                print(s3_null.covars().asmd(on_linked_samples = False).round(3))
                    #            a      b  c[v]   c[x]   c[y]   c[z]  mean(asmd)
                    # index
                    # covars  0.56  8.747   NaN  0.069  0.266  0.533       3.175

                # verify this also works when we have some weights
                print(s3_null_madeup_weights.covars().asmd())
                    #                           a         b  c[v]  ...      c[y]      c[z]  mean(asmd)
                    # source                                       ...
                    # self               0.296500  8.153742   NaN  ...  0.000000  0.218218    2.834932
                    # unadjusted         0.560055  8.746742   NaN  ...  0.265606  0.533422    3.174566
                    # unadjusted - self  0.263555  0.592999   NaN  ...  0.265606  0.315204    0.33963
        """
        target_from_self = self._BalanceDF_child_from_linked_samples().get("target")

        if target is None:
            target = target_from_self

        if target is None:
            raise ValueError(
                f"Sample {object.__str__(self._sample)} has no target set, or target has no {self.__name} to compare against."
            )
        elif on_linked_samples:
            return balance_util.row_pairwise_diffs(
                self._call_on_linked(
                    "asmd",
                    exclude=("target",),
                    target=target,
                    aggregate_by_main_covar=aggregate_by_main_covar,
                    **kwargs,
                )
            )
        else:
            out = (
                pd.DataFrame(
                    self._asmd_BalanceDF(self, target, aggregate_by_main_covar)
                )
                .transpose()
                .assign(index=(self.__name,))
                .set_index("index")
            )
            return out

    def asmd_improvement(
        self: "BalanceDF",
        unadjusted: Optional["BalanceDF"] = None,
        target: Optional["BalanceDF"] = None,
    ) -> np.float64:
        """Calculates the improvement in mean(asmd) from before to after applying some weight adjustment.

        See :func:`weighted_comparisons_stats.asmd_improvement` for details.

        Args:
            self (BalanceDF): BalanceDF (e.g.: of self after adjustment)
            unadjusted (Optional["BalanceDF"], optional): BalanceDF (e.g.: of self before adjustment). Defaults to None.
            target (Optional["BalanceDF"], optional): To compare against. Defaults to None.

        Raises:
            ValueError: If target is not linked in self and also not provided to the function.
            ValueError: If unadjusted is not linked in self and also not provided to the function.

        Returns:
            np.float64: The improvement is taking the (before_mean_asmd-after_mean_asmd)/before_mean_asmd.
                The asmd is calculated using :func:`asmd`.

        Examples:
            ::

                import pandas as pd
                from balance.sample_class import Sample

                from copy import deepcopy


                s1 = Sample.from_frame(
                    pd.DataFrame(
                        {
                            "a": (1, 2, 3, 1),
                            "b": (-42, 8, 2, -42),
                            "o": (7, 8, 9, 10),
                            "c": ("x", "y", "z", "v"),
                            "id": (1, 2, 3, 4),
                            "w": (0.5, 2, 1, 1),
                        }
                    ),
                    id_column="id",
                    weight_column="w",
                    outcome_columns="o",
                )

                s2 = Sample.from_frame(
                    pd.DataFrame(
                        {
                            "a": (1, 2, 3),
                            "b": (4, 6, 8),
                            "id": (1, 2, 3),
                            "w": (0.5, 1, 2),
                            "c": ("x", "y", "z"),
                        }
                    ),
                    id_column="id",
                    weight_column="w",
                )

                s3 = s1.set_target(s2)
                s3_null = s3.adjust(method="null")

                s3_null_madeup_weights = deepcopy(s3_null)
                s3_null_madeup_weights.set_weights((1, 2, 3, 1))

                s3_null.covars().asmd_improvement() # 0. since unadjusted is just a copy of self
                s3_null_madeup_weights.covars().asmd_improvement() # 0.10698596233975825

                asmd_df = s3_null_madeup_weights.covars().asmd()
                print(asmd_df["mean(asmd)"])
                    # source
                    # self                 2.834932
                    # unadjusted           3.174566
                    # unadjusted - self    0.339634
                    # Name: mean(asmd), dtype: float64
                (asmd_df["mean(asmd)"][1] - asmd_df["mean(asmd)"][0]) / asmd_df["mean(asmd)"][1]  # 0.10698596233975825
                # just like asmd_improvement
        """
        if unadjusted is None:
            unadjusted = self._BalanceDF_child_from_linked_samples().get("unadjusted")
        if unadjusted is None:
            raise ValueError(
                f"Sample {object.__repr__(self._sample)} has no unadjusted set or unadjusted has no {self.__name}."
            )

        if target is None:
            target = self._BalanceDF_child_from_linked_samples().get("target")
        if target is None:
            raise ValueError(
                f"Sample {object.__repr__(self._sample)} has no target set or target has no {self.__name}."
            )

        sample_before_df, sample_before_weights = unadjusted._get_df_and_weights()
        sample_after_df, sample_after_weights = self._get_df_and_weights()
        target_df, target_weights = target._get_df_and_weights()

        return weighted_comparisons_stats.asmd_improvement(
            sample_before=sample_before_df,
            sample_after=sample_after_df,
            target=target_df,
            sample_before_weights=sample_before_weights,
            sample_after_weights=sample_after_weights,
            target_weights=target_weights,
        )

    # TODO: implement the following methods (probably first in balance.stats_and_plots.weighted_comparisons_stats)
    # def emd(self):
    #     return NotImplementedError()

    # def cvmd(self):
    #     return NotImplementedError()

    # def ks(self):
    #     return NotImplementedError()

    def _df_with_ids(self: "BalanceDF") -> pd.DataFrame:
        """Creates a DataFrame of the BalanceDF, with ids.

        Args:
            self (BalanceDF): Object.

        Returns:
            pd.DataFrame: DataFrame with id_column and then the df.
        """
        return pd.concat((self._sample.id_column, self.df), axis=1)

    def to_csv(
        self: "BalanceDF",
        path_or_buf: Optional[FilePathOrBuffer] = None,
        *args,
        **kwargs,
    ) -> Optional[str]:
        """Write df with ids from BalanceDF to a comma-separated values (csv) file.

        Uses :func:`pd.DataFrame.to_csv`.

        If an 'index' argument is not provided then it defaults to False.

        Args:
            self (BalanceDF): Object.
            path_or_buf (Optional[FilePathOrBuffer], optional): location where to save the csv.

        Returns:
            Optional[str]: If path_or_buf is None, returns the resulting csv format as a string. Otherwise returns None.
        """
        if "index" not in kwargs:
            kwargs["index"] = False
        return self._df_with_ids().to_csv(path_or_buf=path_or_buf, *args, **kwargs)


class BalanceOutcomesDF(BalanceDF):
    def __init__(self: "BalanceOutcomesDF", sample: Sample) -> None:
        """A factory function to create BalanceOutcomesDF

        This is used through :func:`Sample.outcomes`.
        It initates a BalanceOutcomesDF object by passing the relevant arguments to
        :func:`BalanceDF.__init__`.

        Args:
            self (BalanceOutcomesDF): Object that is initiated.
            sample (Sample): Object
        """
        super().__init__(sample._outcome_columns, sample, name="outcomes")

    # TODO: add the `relative_to` argument (with options 'self' and 'target')
    #       this will also require to update _relative_response_rates a bit.
    def relative_response_rates(
        self: "BalanceOutcomesDF",
        target: Union[bool, pd.DataFrame] = False,
        per_column: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Produces a summary table of number of responses and proportion of completed responses.

        See :func:`general_stats.relative_response_rates`.

        Args:
            self (BalanceOutcomesDF): Object
            target (Union[bool, pd.DataFrame], optional): Defaults to False.
                Determines what is passed to df_target in :func:`general_stats.relative_response_rates`
                If False: passes None.
                If True: passes the df from the target of sample (notice, it's the df of target, NOT target.outcome().df).
                    So it means it will count only rows that are all notnull rows (so if the target has covars and outcomes,
                        both will need to be notnull to be counted).
                    If you want to control this in a more specific way, pass pd.DataFrame instead.
                If pd.DataFrame: passes it as is.
            per_column (bool, optional): Default is False. See :func:`general_stats.relative_response_rates`.

        Returns:
            Optional[pd.DataFrame]: A column per outcome, and two rows.
                One row with number of non-null observations, and
                A second row with the proportion of non-null observations.

                If 'target' is set to True but there is no target, the function returns None.

        Examples:
            ::

                import numpy as np
                import pandas as pd

                from balance.sample_class import Sample


                s_o = Sample.from_frame(
                    pd.DataFrame({"o1": (7, 8, 9, 10), "o2": (7, 8, 9, np.nan), "id": (1, 2, 3, 4)}),
                    id_column="id",
                    outcome_columns=("o1", "o2"),
                )

                print(s_o.outcomes().relative_response_rates())
                    #       o1    o2
                    # n    4.0   3.0
                    # %  100.0  75.0

                s_o.outcomes().relative_response_rates(target = True)
                # None

                # compared with a larget target

                t_o = Sample.from_frame(
                    pd.DataFrame(
                        {
                            "o1": (7, 8, 9, 10, 11, 12, 13, 14),
                            "o2": (7, 8, 9, np.nan, np.nan, 12, 13, 14),
                            "id": (1, 2, 3, 4, 5, 6, 7, 8),
                        }
                    ),
                    id_column="id",
                    outcome_columns=("o1", "o2"),
                )
                s_o2 = s_o.set_target(t_o)

                print(s_o2.outcomes().relative_response_rates(True, per_column = True))
                    #     o1    o2
                    # n   4.0   3.0
                    # %  50.0  50.0

                df_target = pd.DataFrame(
                        {
                            "o1": (7, 8, 9, 10, 11, 12, 13, 14),
                            "o2": (7, 8, 9, np.nan, np.nan, 12, 13, 14),
                        }
                    )

                print(s_o2.outcomes().relative_response_rates(target = df_target, per_column = True))
                    #     o1    o2
                    # n   4.0   3.0
                    # %  50.0  50.0
        """
        if type(target) is bool:
            # Then: get target from self:
            if target:
                self_target = self._BalanceDF_child_from_linked_samples().get("target")
                if self_target is None:
                    logger.warning("Sample does not have target set")
                    return None
                else:
                    df_target = self_target.df
            else:
                df_target = None
        else:
            df_target = target

        return general_stats.relative_response_rates(
            self.df, df_target, per_column=per_column
        )

    def target_response_rates(self: "BalanceOutcomesDF") -> Optional[pd.DataFrame]:
        """Caclulates relative_response_rates for the target in a Sample object.

        See :func:`general_stats.relative_response_rates`.

        Args:
            self (BalanceOutcomesDF): Object (with/without a target set)

        Returns:
            Optional[pd.DataFrame]: None if the object doesn't have a target.
                If the object has a target, it returns the output of :func:`general_stats.relative_response_rates`.

        Examples:
            ::

                import numpy as np
                import pandas as pd

                from balance.sample_class import Sample


                s_o = Sample.from_frame(
                    pd.DataFrame({"o1": (7, 8, 9, 10), "o2": (7, 8, 9, np.nan), "id": (1, 2, 3, 4)}),
                    id_column="id",
                    outcome_columns=("o1", "o2"),
                )

                t_o = Sample.from_frame(
                    pd.DataFrame(
                        {
                            "o1": (7, 8, 9, 10, 11, 12, 13, 14),
                            "o2": (7, 8, 9, np.nan, 11, 12, 13, 14),
                            "id": (1, 2, 3, 4, 5, 6, 7, 8),
                        }
                    ),
                    id_column="id",
                    outcome_columns=("o1", "o2"),
                )
                s_o = s_o.set_target(t_o)

                print(s_o.outcomes().target_response_rates())
                    #       o1    o2
                    # n    8.0   7.0
                    # %  100.0  87.5
        """
        self_target = self._BalanceDF_child_from_linked_samples().get("target")
        if self_target is None:
            logger.warning("Sample does not have target set")
            return None
        else:
            return general_stats.relative_response_rates(self_target.df)

    # TODO: it's a question if summary should produce a printable output or a DataFrame.
    #       The BalanceDF.summary method only returns a DataFrame. So it's a question
    #       what is the best way to structure this more generally.
    def summary(
        self: "BalanceOutcomesDF", on_linked_samples: Optional[bool] = None
    ) -> str:
        """Produces summary printable string of a BalanceOutcomesDF object.

        Args:
            self (BalanceOutcomesDF): Object.
            on_linked_samples (Optional[bool]): Ignored. Only here since summary overrides BalanceDF.summary.

        Returns:
            str: A printable string, with mean of outcome variables and reponse rates.

        Examples:
            ::

                import numpy as np
                import pandas as pd

                from balance.sample_class import Sample

                s_o = Sample.from_frame(
                    pd.DataFrame({"o1": (7, 8, 9, 10), "o2": (7, 8, 9, np.nan), "id": (1, 2, 3, 4)}),
                    id_column="id",
                    outcome_columns=("o1", "o2"),
                )

                t_o = Sample.from_frame(
                    pd.DataFrame(
                        {
                            "o1": (7, 8, 9, 10, 11, 12, 13, 14),
                            "o2": (7, 8, 9, np.nan, np.nan, 12, 13, 14),
                            "id": (1, 2, 3, 4, 5, 6, 7, 8),
                        }
                    ),
                    id_column="id",
                    outcome_columns=("o1", "o2"),
                )
                s_o2 = s_o.set_target(t_o)

                print(s_o.outcomes().summary())

                # 2 outcomes: ['o1' 'o2']
                # Mean outcomes:
                #         _is_na_o2[False]  _is_na_o2[True]   o1   o2
                # source
                # self                0.75             0.25  8.5  6.0

                # Response rates (relative to number of respondents in sample):
                #       o1    o2
                # n    4.0   3.0
                # %  100.0  75.0


                print(s_o2.outcomes().summary())

                # 2 outcomes: ['o1' 'o2']
                # Mean outcomes:
                #         _is_na_o2[False]  _is_na_o2[True]    o1     o2
                # source
                # self                0.75             0.25   8.5  6.000
                # target              0.75             0.25  10.5  7.875

                # Response rates (relative to number of respondents in sample):
                #       o1    o2
                # n    4.0   3.0
                # %  100.0  75.0
                # Response rates (relative to notnull rows in the target):
                #            o1    o2
                # n   4.000000   3.0
                # %  66.666667  50.0
                # Response rates (in the target):
                #        o1    o2
                # n    8.0   6.0
                # %  100.0  75.0
        """
        mean_outcomes = self.mean()
        relative_response_rates = self.relative_response_rates()
        target_response_rates = self.target_response_rates()
        if target_response_rates is None:
            target_clause = ""
            relative_to_target_clause = ""
        else:
            relative_to_target_response_rates = self.relative_response_rates(
                target=True, per_column=False
            )
            relative_to_target_clause = f"Response rates (relative to notnull rows in the target):\n {relative_to_target_response_rates}"

            target_clause = f"Response rates (in the target):\n {target_response_rates}"

        n_outcomes = self.df.shape[1]
        list_outcomes = self.df.columns.values
        mean_outcomes = mean_outcomes
        relative_response_rates = relative_response_rates
        target_clause = target_clause

        out = (
            f"{n_outcomes} outcomes: {list_outcomes}\n"
            "Mean outcomes:\n"
            f"{mean_outcomes}\n\n"
            "Response rates (relative to number of respondents in sample):\n"
            f"{relative_response_rates}\n"
            f"{relative_to_target_clause}\n"
            f"{target_clause}\n"
        )

        return out

    # TODO: once we have hist/kde methods for plotly, consider changing the defaults here to use plotly instead of seaborn.
    def plot(
        self: "BalanceOutcomesDF", on_linked_samples: bool = True, **kwargs
    ) -> Union[Union[List, np.ndarray], Dict[str, go.Figure]]:
        """Plots histogram of covariates in a BalanceOutcomesDF object using seaborn (as default).

        It's possible to use other plots using dist_type with arguments such as "hist" (default), "kde", "qq", and "ecdf".
        Look at :func:`plot_dist` for more details.

        Args:
            self (BalanceOutcomesDF): a BalanceOutcomesDF object, with a set of variables.
            on_linked_samples (bool, optional): Determines if the linked samples should be included in the plot.
                Defaluts to True.

        Returns:
            Union[Union[List, np.ndarray], Dict[str, go.Figure]]:
                If library="plotly" then returns a dictionary containing plots if return_dict_of_figures is True. None otherwise.
                If library="seaborn" then returns either a list or an np.array of matplotlib axis.

        Examples:
            ::

                import numpy as np
                import pandas as pd
                from numpy import random
                from balance.sample_class import Sample

                random.seed(96483)

                df = pd.DataFrame({
                    "id": range(100),
                    'v1': random.random_integers(11111, 11114, size=100).astype(str),
                    'v2': random.normal(size = 100),
                    'v3': random.uniform(size = 100),
                    "w": pd.Series(np.ones(99).tolist() + [1000]),
                }).sort_values(by=['v2'])

                s1 = Sample.from_frame(df,
                    id_column="id",
                    weight_column="w",
                    outcome_columns=["v1", "v2"],
                )

                s2 = Sample.from_frame(
                    df.assign(w = pd.Series(np.ones(100))),
                    id_column="id",
                    weight_column="w",
                    outcome_columns=["v1", "v2"],
                )

                s3 = s1.set_target(s2)
                s3_null = s3.adjust(method="null")
                s3_null.set_weights(random.random(size = 100) + 0.5)

                # default: seaborn with dist_type = "hist"
                s3_null.outcomes().plot()

                # using dist_type = "kde"
                s3_null.outcomes().plot(dist_type = "kde")

                # using plotly
                s3_null.outcomes().plot(library = "plotly")
        """
        default_kwargs = {
            "library": "seaborn",
            "dist_type": "kde",
        }
        default_kwargs.update(kwargs)
        return super().plot(on_linked_samples=on_linked_samples, **default_kwargs)


class BalanceCovarsDF(BalanceDF):
    def __init__(self: "BalanceCovarsDF", sample: Sample) -> None:
        """A factory function to create BalanceCovarsDF

        This is used through :func:`Sample.covars`.
        It initates a BalanceCovarsDF object by passing the relevant arguments to
        :func:`BalanceDF.__init__`.

        Args:
            self (BalanceCovarsDF): Object that is initiated.
            sample (Sample): Object
        """
        super().__init__(sample._covar_columns(), sample, name="covars")

    def from_frame(
        self: "BalanceCovarsDF",
        df: pd.DataFrame,
        weights=Optional[pd.Series],
    ) -> "BalanceCovarsDF":
        """A factory function to create a BalanceCovarsDF from a df.

        Although generally the main way the object is created is through the __init__ method.

        Args:
            self (BalanceCovarsDF): Object
            df (pd.DataFrame): A df.
            weights (Optional[pd.Series], optional): _description_. Defaults to None.

        Returns:
            BalanceCovarsDF: Object.
        """
        df = df.reset_index()
        df = pd.concat(
            (df, pd.Series(np.arange(0, df.shape[0]), name="id"), weights), axis=1
        )
        return Sample.from_frame(df, id_column="id").covars()


class BalanceWeightsDF(BalanceDF):
    def __init__(self: "BalanceWeightsDF", sample: Sample) -> None:
        """A factory function to create BalanceWeightsDF

        This is used through :func:`Sample.weights`.
        It initates a BalanceWeightsDF object by passing the relevant arguments to
        :func:`BalanceDF.__init__`.

        Args:
            self (BalanceWeightsDF): Object that is initiated.
            sample (Sample): Object
        """
        super().__init__(sample.weight_column.to_frame(), sample, name="weights")

    # TODO: maybe add better control if there are no weights for unadjusted or target (the current default shows them in the legend, but not in the figure)
    def plot(
        self: "BalanceWeightsDF", on_linked_samples: bool = True, **kwargs
    ) -> Union[Union[List, np.ndarray], Dict[str, go.Figure]]:
        """Plots kde (kernal density estimation) of the weights in a BalanceWeightsDF object using seaborn (as default).

        It's possible to use other plots using dist_type with arguments such as "hist" (default), "kde", "qq", and "ecdf".
        Look at :func:`plot_dist` for more details.

        Args:
            self (BalanceWeightsDF): a BalanceOutcomesDF object, with a set of variables.
            on_linked_samples (bool, optional): Determines if the linked samples should be included in the plot.
                Defaluts to True.

        Returns:
            Union[Union[List, np.ndarray], Dict[str, go.Figure]]:
                If library="plotly" then returns a dictionary containing plots if return_dict_of_figures is True. None otherwise.
                If library="seaborn" then returns either a list or an np.array of matplotlib axis.

        Examples:
            ::

                import numpy as np
                import pandas as pd
                from numpy import random
                from balance.sample_class import Sample

                random.seed(96483)

                df = pd.DataFrame({
                    "id": range(100),
                    'v1': random.random_integers(11111, 11114, size=100).astype(str),
                    'v2': random.normal(size = 100),
                    'v3': random.uniform(size = 100),
                    "w": pd.Series(np.ones(99).tolist() + [1000]),
                }).sort_values(by=['v2'])

                s1 = Sample.from_frame(df,
                    id_column="id",
                    weight_column="w",
                    outcome_columns=["v1", "v2"],
                )

                s2 = Sample.from_frame(
                    df.assign(w = pd.Series(np.ones(100))),
                    id_column="id",
                    weight_column="w",
                    outcome_columns=["v1", "v2"],
                )

                s3 = s1.set_target(s2)
                s3_null = s3.adjust(method="null")
                s3_null.set_weights(random.random(size = 100) + 0.5)

                # default: seaborn with dist_type = "kde"
                s3_null.weights().plot()
        """
        default_kwargs = {
            "weighted": False,
            "library": "seaborn",
            "dist_type": "kde",
            "numeric_n_values_threshold": -1,
        }
        default_kwargs.update(kwargs)
        return super().plot(on_linked_samples=on_linked_samples, **default_kwargs)

    def design_effect(self: "BalanceWeightsDF") -> np.float64:
        """Calculates Kish's design effect (deff) on the BalanceWeightsDF weights.

        Extract the first column to get a pd.Series of the weights.

        See :func:`weights_stats.design_effect` for details.

        Args:
            self (BalanceWeightsDF): Object.

        Returns:
            np.float64: Deff.
        """
        return weights_stats.design_effect(self.df.iloc[:, 0])

    # TODO: in the future, consider if this type of overriding is the best solution.
    #       to reconsider as part of a larger code refactoring.
    @property
    def _weights(self: "BalanceWeightsDF") -> None:
        """A BalanceWeightsDF has no weights (its df is that of the weights.)

        Args:
            self (BalanceWeightsDF): Object.

        Returns:
            NoneType: None.
        """
        return None

    def trim(
        self: "BalanceWeightsDF",
        ratio: Optional[Union[float, int]] = None,
        percentile: Optional[float] = None,
        keep_sum_of_weights: bool = True,
    ) -> None:
        """Trim weights in the sample object.

        Uses :func:`adjustments.trim_weights` for the weights trimming.

        Args:
            self (BalanceWeightsDF): Object.
            ratio (Optional[Union[float, int]], optional): Maps to weight_trimming_mean_ratio. Defaults to None.
            percentile (Optional[float], optional): Maps to weight_trimming_percentile. Defaults to None.
            keep_sum_of_weights (bool, optional): Maps to weight_trimming_percentile. Defaults to True.

        Returns:
            None. This function updates the :func:`_sample` using :func:`set_weights`
        """
        # TODO: verify which object exactly gets updated - and explain it here.
        self._sample.set_weights(
            trim_weights(
                self.df.iloc[:, 0],
                weight_trimming_mean_ratio=ratio,
                weight_trimming_percentile=percentile,
                keep_sum_of_weights=keep_sum_of_weights,
            )
        )
