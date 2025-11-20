# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import AnyStr, IO

FilePathOrBuffer = str | Path | IO[AnyStr]
