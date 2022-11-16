# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import AnyStr, IO, Union

FilePathOrBuffer = Union[str, Path, IO[AnyStr]]
