# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
from typing import Any


def _truncate_text(s: str, length: int) -> str:
    """Truncate string s to be of length 'length'. If the length of s is larger than 'length', then the
    function will add '...' at the end of the truncated text.

    Args:
        s (str):
        length (int):

    Returns:
        str:
    """

    return s[:length] + "..." * (len(s) > length)


class TruncationFormatter(logging.Formatter):
    """
    Logging formatter which truncates the logged message to 500 characters.

    This is useful in the cases where the logging message includes objects
    --- like DataFrames --- whose string representation is very long.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(TruncationFormatter, self).__init__(*args, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        result = super(TruncationFormatter, self).format(record)
        return _truncate_text(result, 2000)
