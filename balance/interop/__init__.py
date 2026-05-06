# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""balance.interop — adapters between balance and adjacent libraries.

Empty re-export surface by design. Sub-modules (e.g.
``balance.interop.diff_diff``) are imported explicitly by users so that
optional heavy dependencies stay lazy:

    from balance.interop import diff_diff as bd
"""

from __future__ import annotations
