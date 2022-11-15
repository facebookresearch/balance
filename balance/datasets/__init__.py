# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from typing import Optional, Tuple

import numpy as np
import pandas as pd


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
        Tuple[pd.DataFrame, pd.DataFrame]: Two dataframes containing simulated data for the target and sample of interest.
    """

    if version == "01":
        np.random.seed(2022 - 11 - 8)  # for reproducibility
        n_target = 10000
        target_df = pd.DataFrame(
            {
                "id": (np.array(range(n_target)) + 100000).astype(str),
                "gender": np.random.choice(
                    ["Male", "Female"], size=n_target, replace=True, p=[0.5, 0.5]
                ),
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
        target_df.happiness = np.random.normal(50, 10, size=n_target) + np.where(
            target_df.gender == "Female", 1, 0
        ) * np.random.normal(5, 2, size=n_target)
        # We also have missing values in gender
        target_df.loc[3:900, "gender"] = np.nan

        n_sample = 1000
        sample_df = pd.DataFrame(
            {
                "id": (np.array(range(n_sample))).astype(str),
                "gender": np.random.choice(
                    ["Male", "Female"], size=n_sample, replace=True, p=[0.7, 0.3]
                ),
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
            + np.where(sample_df.gender == "Female", 1, 0)
            * np.random.normal(20, 1, size=n_sample)
            + np.where(sample_df.age_group == "35-44", 1, 0)
            * np.random.normal(5, 1, size=n_sample)
            + np.where(sample_df.age_group == "45+", 1, 0)
            * np.random.normal(20, 1, size=n_sample)
            + np.random.normal(
                (np.random.normal(3, 2, size=n_sample) ** 2) / 20, 1, size=n_sample
            )
        )
        # Truncate for max to be 100
        sample_df["happiness"] = np.where(
            sample_df["happiness"] < 100, sample_df["happiness"], 100
        )

        # We also have missing values in gender
        sample_df.loc[3:90, "gender"] = np.nan

        return target_df, sample_df

    return None, None


def load_data(
    source: str = "sim_data_01",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Returns a tuple of two dataframes containing simulated data.

    Args:
        source (str, optional): The name of the data to return. Defaults to "sim_data_01".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The first dataframe contains simulated data of the "target" and the second dataframe contains simulated data of the "sample".
    """

    if source == "sim_data_01":
        target_df, sample_df = load_sim_data("01")
        return (target_df, sample_df)

    return None, None
