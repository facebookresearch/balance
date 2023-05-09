# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

import pathlib
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

# TODO: move the functions from datasets/__init__.py to some other file (e.g.: datasets/loading_data.py),
#       and then import the functions from that file in the init file (so the behavior would remain the same)


# TODO: add tests
def load_sim_data(
    version: str = "01",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load simulated data for target and sample of interest

    Version 01 returns two dataframes containing the columns gender ("Male", "Female" and nan),
    age_group ("18-24", "25-34", "35-44", "45+"), income (some numbers from a normal distribution), and id.
    The sample_df also has a column called happiness with a value from 0 to 100 that depends on the covariates.

    The target_df DataFrame has 10000 rows, and sample_df has 1000 rows.

    The sample_df is imbalanced when compared to target_df, as is demonstrated in the examples/tutorials.

    If you want to see how this works, you can import balance and run this code:
        import inspect
        import balance
        print(inspect.getsource(balance.datasets.load_sim_data))

    Args:
        version (str, optional): The version of simulated data. Currently available is only "01". Defaults to "01".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing simulated data for the target and sample of interest.
    """

    if version == "01":
        np.random.seed(2022 - 11 - 8)  # for reproducibility
        n_target = 10000
        target_df = pd.DataFrame(
            {
                "id": (np.array(range(n_target)) + 100000).astype(str),
                "gender": np.random.choice(["Male", "Female"], size=n_target, replace=True, p=[0.5, 0.5]),
                "age_group": np.random.choice(
                    ["18-24", "25-34", "35-44", "45+"],
                    size=n_target,
                    replace=True,
                    p=[0.20, 0.30, 0.30, 0.20],
                ),
                "income": np.random.normal(3, 2, size=n_target) ** 2,
                # "unrelated_variable": np.random.uniform(size = n_target),
                # "weight": np.random.uniform(size = n_target) + 0.5,
            }
        )
        # We also have missing values in gender
        target_df.loc[3:900, "gender"] = np.nan

        n_sample = 1000
        sample_df = pd.DataFrame(
            {
                "id": (np.array(range(n_sample))).astype(str),
                "gender": np.random.choice(["Male", "Female"], size=n_sample, replace=True, p=[0.7, 0.3]),
                "age_group": np.random.choice(
                    ["18-24", "25-34", "35-44", "45+"],
                    size=n_sample,
                    replace=True,
                    p=[0.50, 0.30, 0.15, 0.05],
                ),
                "income": np.random.normal(2, 1.5, size=n_sample) ** 2,
                # "unrelated_variable": np.random.uniform(size = n_sample),
                # "weight": np.random.uniform(size = n_sample) + 0.5,
            }
        )
        # females are happier
        # older people are happier
        # people with higher income are happeir
        sample_df["happiness"] = (
            np.random.normal(40, 10, size=n_sample)
            + np.where(sample_df.gender == "Female", 1, 0) * np.random.normal(20, 1, size=n_sample)
            + np.where(sample_df.age_group == "35-44", 1, 0) * np.random.normal(5, 1, size=n_sample)
            + np.where(sample_df.age_group == "45+", 1, 0) * np.random.normal(20, 1, size=n_sample)
            + np.random.normal((np.random.normal(3, 2, size=n_sample) ** 2) / 20, 1, size=n_sample)
        )
        # Truncate for max to be 100
        sample_df["happiness"] = np.where(sample_df["happiness"] < 100, sample_df["happiness"], 100)

        # We also have missing values in gender
        sample_df.loc[3:90, "gender"] = np.nan

        return target_df, sample_df

    return None, None


# TODO: add tests
def load_cbps_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load simulated data for CBPS comparison with R

    The code in balance that implements CBPS attempts to mimic the code from the R package CBPS (https://cran.r-project.org/web/packages/CBPS/).

    In the help page of the CBPS function in R (i.e.: `?CBPS::CBPS`) there is a simulated dataset that is used to showcase the CBPS function.
    The output of that simulated dataset is saved in balance in order to allow for comparison of `balance` (Python) with `CBPS` (R).

    You can view the structure of the simulated dataset by looking at the example below.

    In the original simulation dataset (available in sim_data_cbps.csv), when the `treat` variable is 0, the row belongs to sample.
    And when the `treat` variable is 1, the row belongs to target.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]: Two DataFrames containing simulated data for the target and sample of interest.

    Example:
        ::
            import balance
            target_df, sample_df = balance.datasets.load_data("sim_data_cbps")
            print(target_df.head())
            #           X1         X2        X3          X4  cbps_weights           y  id
            # 1   0.723769   9.911956  0.189488  383.759778      0.003937  199.817495   2
            # 3   0.347071   9.907768  0.096706  399.366071      0.003937  174.685348   4
            # 11  0.691174  10.725262  0.214618  398.313184      0.003937  189.578368  12
            # 12  0.779949   9.562130  0.181408  370.178863      0.003937  208.178724  13
            # 13  0.818348   9.801834  0.210592  434.453795      0.003937  214.277306  14

            print(target_df.info())
            # <class 'pandas.core.frame.DataFrame'>
            # Int64Index: 254 entries, 1 to 498
            # Data columns (total 7 columns):
            #  #   Column        Non-Null Count  Dtype
            # ---  ------        --------------  -----
            #  0   X1            254 non-null    float64
            #  1   X2            254 non-null    float64
            #  2   X3            254 non-null    float64
            #  3   X4            254 non-null    float64
            #  4   cbps_weights  254 non-null    float64
            #  5   y             254 non-null    float64
            #  6   id            254 non-null    int64
            # dtypes: float64(6), int64(1)
            # memory usage: 15.9 KB
    """
    # NOTE: the reason we use __file__ and not importlib.resources is because the later one changed API in Python 3.11.
    #       so in order to be compliant with 3.7-3.10 and also 3.11, using __file__ is the safer option.
    df_all = pd.read_csv(pathlib.Path(__file__).parent.joinpath("sim_data_cbps.csv"))
    target_df = df_all[df_all.treat == 1].drop(["treat"], 1)
    sample_df = df_all[df_all.treat == 0].drop(["treat"], 1)

    return (target_df, sample_df)


def load_data(
    source: Literal["sim_data_01", "sim_data_cbps"] = "sim_data_01",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Returns a tuple of two DataFrames containing simulated data.

    To learn more about each dataset, please refer to their help pages:
    - sim_data_01: :func:`balance.datasets.load_sim_data`.
    - sim_data_cbps: :func:`balance.datasets.load_cbps_data`.

    Args:
        source (Literal["sim_data_01", "sim_data_cbps"]): The name of the data to return. Defaults to "sim_data_01".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The first dataframe contains simulated data of the "target" and the second dataframe contains simulated data of the "sample".
    """

    if source == "sim_data_01":
        target_df, sample_df = load_sim_data("01")
        return (target_df, sample_df)
    elif source == "sim_data_cbps":
        target_df, sample_df = load_cbps_data()
        return (target_df, sample_df)

    return None, None
