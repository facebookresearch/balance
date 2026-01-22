# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Test configuration shared across pytest runs"""

import os

import matplotlib
import pytest

from balance import testutil

# Force a non-interactive backend so tests do not require a Tk installation.
matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def reset_high_cardinality_threshold() -> None:
    """Reset high-cardinality global state between tests to avoid pollution."""
    original_env = os.environ.get("BALANCE_HIGH_CARDINALITY_RATIO_THRESHOLD")
    testutil._reset_high_cardinality_threshold_state(original_env)

    yield

    testutil._reset_high_cardinality_threshold_state(original_env)
