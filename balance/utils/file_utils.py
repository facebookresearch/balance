# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import tempfile
import uuid

import pandas as pd
from IPython.lib.display import FileLink


def _to_download(
    df: pd.DataFrame,
    tempdir: str | None = None,
) -> FileLink:
    """Creates a downloadable link of the DataFrame (df).

    File name starts with tmp_balance_out_, and some random file name (using :func:`uuid.uuid4`).

    Args:
        self (BalanceDF): Object.
        tempdir (str | None, optional): Defaults to None (which then uses a temporary folder using :func:`tempfile.gettempdir`).

    Returns:
        FileLink: Embedding a local file link in an IPython session, based on path. Using :func:FileLink.
    """
    if tempdir is None:
        tempdir = tempfile.gettempdir()
    path = f"{tempdir}/tmp_balance_out_{uuid.uuid4()}.csv"

    df.to_csv(path_or_buf=path, index=False)
    return FileLink(path, result_html_prefix="Click here to download: ")
