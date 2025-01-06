# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import AnyStr, IO, Union

FilePathOrBuffer = Union[str, Path, IO[AnyStr]]
