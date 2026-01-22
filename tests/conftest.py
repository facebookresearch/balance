# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Test configuration shared across pytest runs"""

import os

import matplotlib
from balance import testutil
from typing import Optional

# Force a non-interactive backend so tests do not require a Tk installation.
matplotlib.use("Agg", force=True)


def pytest_runtest_setup(item: object) -> None:
    """Reset high-cardinality global state before each test."""
    original_env = os.environ.get("BALANCE_HIGH_CARDINALITY_RATIO_THRESHOLD")
    setattr(item, "_high_cardinality_env", original_env)
    testutil._reset_high_cardinality_threshold_state(original_env)


def pytest_runtest_teardown(item: object, nextitem: Optional[object]) -> None:
    """Restore high-cardinality global state after each test."""
    original_env = getattr(item, "_high_cardinality_env", None)
    testutil._reset_high_cardinality_threshold_state(original_env)
    if hasattr(item, "_high_cardinality_env"):
        delattr(item, "_high_cardinality_env")
