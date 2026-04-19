# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Test configuration shared across pytest runs"""

from __future__ import annotations

import matplotlib

# Force a non-interactive backend so tests do not require a Tk installation.
matplotlib.use("Agg", force=True)

import plotly.io as pio  # noqa: E402
import pytest  # noqa: E402
from balance.testutil import _SKLEARN_1_4_AVAILABLE  # noqa: E402

# Force plotly to use a non-interactive renderer so tests don't open a browser.
pio.renderers.default = "json"


# ---------------------------------------------------------------------------
# Deselect (rather than skip) tests that require scikit-learn >= 1.4.
#
# Using ``@unittest.skipIf`` for version-gated tests emits noisy "SKIPPED"
# warnings in CI output.  By deselecting at collection time the tests simply
# disappear from the run — no warnings, no noise.
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Deselect tests marked ``requires_sklearn_1_4`` when sklearn < 1.4."""
    if _SKLEARN_1_4_AVAILABLE:
        return  # nothing to deselect

    keep: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        if item.get_closest_marker("requires_sklearn_1_4"):
            deselected.append(item)
        else:
            keep.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = keep
