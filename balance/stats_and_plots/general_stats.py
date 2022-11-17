# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from typing import Optional

import pandas as pd


def relative_response_rates(
    df: pd.DataFrame,
    df_target: Optional[pd.DataFrame] = None,
    per_column: bool = True,
) -> pd.DataFrame:
    """Produces a summary table of number of responses and proportion of completed responses.

    Args:
        df (pd.DataFrame): A DataFrame to calculate aggregated response rates for.
        df_target (Optional[pd.DataFrame], optional): Defaults to None.
            Determines what is the denominator from which notnull are a fraction of.
            If None - it's the number of rows in df.
            If some df is provided - then it is assumed that df is a subset of df_target, and the
            response rate is calculated as the fraction of notnull values in df from (divided by)
            the number of notnull values in df_target.
        per_column (bool, optional): Default is True.
            The per_column argument is relevant only if df_target is other than None (i.e.: trying to compare df to some df_target).
            If per_column is True (default) - it indicates that the relative response rates of columns in df will be
                by comparing each column in df to the same column in target.
                If this is True, the columns in df and df_target must be identical.
            If per_column is False then df is compared to the overall number of nonnull rows in the target df.

    Returns:
        pd.DataFrame: A column per column in the original df, and two rows:
            One row with number of non-null observations, and
            A second row with the proportion of non-null observations.

    Examples:
        ::

            import numpy as np
            import pandas as pd
            from balance.stats_and_plots.general_stats import relative_response_rates

            df = pd.DataFrame({"o1": (7, 8, 9, 10), "o2": (7, 8, 9, np.nan), "id": (1, 2, 3, 4)})

            relative_response_rates(df).to_dict()

                # {'o1': {'n': 4.0, '%': 100.0},
                # 'o2': {'n': 3.0, '%': 75.0},
                # 'id': {'n': 4.0, '%': 100.0}}

            df_target = pd.concat([df, df])
            relative_response_rates(df, df_target).to_dict()

                # {'o1': {'n': 4.0, '%': 50.0},
                #  'o2': {'n': 3.0, '%': 50.0},
                #  'id': {'n': 4.0, '%': 50.0}}


            # Dividing by number of total notnull rows in df_rarget
            df_target.notnull().all(axis=1).sum()  # == 6
            relative_response_rates(df, df_target, False).to_dict()

                # {'o1': {'n': 4.0, '%': 66.66666666666666},
                # 'o2': {'n': 3.0, '%': 50.0},
                # 'id': {'n': 4.0, '%': 66.66666666666666}}

    """
    df_n_notnull_rows = df.notnull().sum()

    if df_target is None:
        target_n_notnull_rows = df.shape[0]
    elif per_column:  # number of notnull rows, *per column*, in df_target

        # verify that the columns of df and df_target are identical:
        if (len(df.columns) != len(df_target.columns)) or (
            df.columns.tolist() != df_target.columns.tolist()
        ):
            raise ValueError(
                f"""
                df and df_target must have the exact same columns.
                Instead, thes column names are, (df, df_target) = ({df.columns.tolist()}, {df_target.columns.tolist()})
                """
            )

        # If they are, we can proceed forward:
        target_n_notnull_rows = df_target.notnull().sum()
    else:  # number of notnull *rows* (i.e.: complete rows) in df_target
        target_n_notnull_rows = df_target.notnull().all(axis=1).sum()

    if any(df_n_notnull_rows > target_n_notnull_rows):
        raise ValueError(
            f"""
            The number of (notnull) rows in df MUST be smaller or equal to the number of rows in df_target.
            These were, (df_n_notnull_rows, target_n_notnull_rows) = ({df_n_notnull_rows}, {target_n_notnull_rows})
            """
        )

    return pd.DataFrame(
        {
            "n": df_n_notnull_rows,
            "%": 100 * (df_n_notnull_rows / target_n_notnull_rows),
        }
    ).transpose()
