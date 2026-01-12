# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from balance.typing import FilePathOrBuffer


def _apply_csv_defaults(kwargs: dict[str, Any]) -> None:
    """Set default CSV arguments shared across BalanceDF and Sample.

    The defaults keep CSV output consistent across platforms by:
    - Disabling index writing when the caller did not request it.
    - Normalizing line endings to ``"\n"`` to avoid platform-dependent newlines.
    """

    if "index" not in kwargs:
        kwargs["index"] = False
    if "lineterminator" not in kwargs and "line_terminator" not in kwargs:
        kwargs["lineterminator"] = "\n"


def to_csv_with_defaults(
    df: pd.DataFrame, path_or_buf: FilePathOrBuffer | None, *args: Any, **kwargs: Any
) -> str | None:
    """Write a DataFrame to CSV with consistent defaults and newline handling.

    When the destination is a string or ``PathLike``, the file is opened with
    ``newline=""`` so pandas does not translate line endings on Windows, keeping
    the on-disk output aligned with the in-memory string returned when
    ``path_or_buf`` is ``None``.

    Examples:
    .. code-block:: python

        import pandas as pd
        from balance.csv_utils import to_csv_with_defaults
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        csv_text = to_csv_with_defaults(df, None)
        "a,b" in csv_text
        # True
    """

    _apply_csv_defaults(kwargs)

    if isinstance(path_or_buf, (str, os.PathLike)):
        with open(path_or_buf, "w", newline="", encoding="utf-8") as buffer:
            return df.to_csv(*args, path_or_buf=buffer, **kwargs)

    return df.to_csv(*args, path_or_buf=path_or_buf, **kwargs)
