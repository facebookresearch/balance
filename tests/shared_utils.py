# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Shared test utilities."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from typing import ContextManager


@contextmanager
def tempfile_path() -> ContextManager[str]:
    """Yield a cross-platform temporary file path and remove it afterwards."""

    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        tmp.close()
        yield tmp.name
    finally:
        try:
            os.unlink(tmp.name)
        except FileNotFoundError:
            pass
