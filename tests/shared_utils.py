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


@contextmanager
def tempfile_path():
    """Yield a cross-platform temporary file path and remove it afterwards."""

    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        tmp.close()
        yield tmp.name
    finally:
        os.unlink(tmp.name)
