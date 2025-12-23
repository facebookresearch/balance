# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from pathlib import Path
from typing import AnyStr, IO, Iterable, Union

# NOTE: Type aliases must use Union instead of | syntax for Python 3.9 compatibility.
# While `from __future__ import annotations` enables PEP 604 syntax (|) in function
# and variable annotations, it does NOT work for type alias assignments because these
# are evaluated at runtime. The | operator for unions only works at runtime in Python 3.10+.
#
# Example of what works vs what doesn't in Python 3.9:
#   ✓ def foo(x: str | int) -> str | None:  # Works with __future__ import
#   ✓ variable: str | None = None           # Works with __future__ import
#   ✗ TypeAlias = str | int                 # Runtime error in Python 3.9
#   ✓ TypeAlias = Union[str, int]           # Works in all versions
FilePathOrBuffer = Union[str, Path, IO[AnyStr]]

# Scalar types for diagnostics val/var columns.
# These are atomic values that should not be iterated over character-by-character.
DiagnosticScalar = Union[str, int, float, bytes, None]

# Input type for val/var parameters in _concat_metric_val_var and related helpers.
# Accepts either a single scalar or an iterable of scalars (list, tuple, Series, etc.).
# Note: str and bytes are technically Iterable but are treated as scalars at runtime.
DiagnosticInput = Union[DiagnosticScalar, Iterable[DiagnosticScalar]]
