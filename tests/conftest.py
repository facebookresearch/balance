# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Test configuration shared across pytest runs"""

import os
from typing import Iterator

import matplotlib

# Force a non-interactive backend so tests do not require a Tk installation.
matplotlib.use("Agg", force=True)

import balance.utils.pandas_utils as pandas_utils  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def reset_high_cardinality_threshold() -> Iterator[None]:
    """Reset high-cardinality threshold global state before each test.

    This prevents test pollution when tests modify the global threshold
    configuration via set_high_cardinality_ratio_threshold() or the
    BALANCE_HIGH_CARDINALITY_RATIO_THRESHOLD environment variable.
    """
    env_key = "BALANCE_HIGH_CARDINALITY_RATIO_THRESHOLD"
    original_env = os.environ.get(env_key)

    # Reset to clean state before test
    pandas_utils.set_high_cardinality_ratio_threshold(None)
    pandas_utils._warned_invalid_high_cardinality_env = False
    os.environ.pop(env_key, None)

    yield

    # Restore original state after test
    pandas_utils.set_high_cardinality_ratio_threshold(None)
    pandas_utils._warned_invalid_high_cardinality_env = False
    if original_env is None:
        os.environ.pop(env_key, None)
    else:
        os.environ[env_key] = original_env
