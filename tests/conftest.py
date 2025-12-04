# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import matplotlib

# Force a non-interactive backend so tests do not require a Tk installation.
matplotlib.use("Agg", force=True)
