# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Test configuration shared across pytest runs"""

import os

import matplotlib
import pytest

from balance.utils import pandas_utils

# Force a non-interactive backend so tests do not require a Tk installation.
matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def reset_high_cardinality_threshold() -> None:
    """Reset high-cardinality global state between tests to avoid pollution."""
    env_key = "BALANCE_HIGH_CARDINALITY_RATIO_THRESHOLD"
    original_env = os.environ.get(env_key)

    pandas_utils.set_high_cardinality_ratio_threshold(None)
    pandas_utils._warned_invalid_high_cardinality_env = False
    if original_env is None:
        os.environ.pop(env_key, None)
    else:
        os.environ[env_key] = original_env

    yield

    pandas_utils.set_high_cardinality_ratio_threshold(None)
    pandas_utils._warned_invalid_high_cardinality_env = False
    if original_env is None:
        os.environ.pop(env_key, None)
    else:
        os.environ[env_key] = original_env
