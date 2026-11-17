# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

# This module provides data loading utilities for simulated datasets used in the
# balance package. It supports loading simulation data with reproducible random
# seeds for testing and examples.


def load_sim_data(
    version: str = "01",
) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load simulated data for target and sample of interest.

    This function generates reproducible simulated datasets using fixed random seeds
    to ensure consistent results across multiple calls and different environments.

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

    def _create_outcome_happiness(df: pd.DataFrame, n: int) -> npt.NDArray[np.floating]:
        # females are happier
        # older people are happier
        # people with higher income are happier
        out = (
            np.random.normal(40, 10, size=n)
            + np.where(df.gender == "Female", 1, 0) * np.random.normal(20, 1, size=n)
            + np.where(df.age_group == "35-44", 1, 0) * np.random.normal(5, 1, size=n)
            + np.where(df.age_group == "45+", 1, 0) * np.random.normal(20, 1, size=n)
            + np.random.normal((np.random.normal(3, 2, size=n) ** 2) / 20, 1, size=n)
        )
        # Truncate for max to be 100
        out = np.where(out < 100, out, 100)
        return out

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
            }
        )
        target_df["happiness"] = _create_outcome_happiness(target_df, n_target)
        # We also have missing values in gender
        target_df.loc[3:900, "gender"] = np.nan

        np.random.seed(2023 - 5 - 14)  # for reproducibility
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
            }
        )
        sample_df["happiness"] = _create_outcome_happiness(sample_df, n_sample)

        # We also have missing values in gender
        sample_df.loc[3:90, "gender"] = np.nan

        return target_df, sample_df

    return None, None


def load_data(
    source: Literal["sim_data_01"] = "sim_data_01",
) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Return a tuple of two DataFrames containing simulated data.

    To learn more about each dataset, please refer to their help pages:
    - sim_data_01: :func:`balance.datasets.load_sim_data`.

    Args:
        source (Literal["sim_data_01"]): The name of the data to return. Defaults to "sim_data_01".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The first dataframe contains simulated data of the "target" and the second dataframe contains simulated data of the "sample".
    """

    if source == "sim_data_01":
        return load_sim_data("01")

    return None, None
