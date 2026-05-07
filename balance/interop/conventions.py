# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Recommended column names for balance interop handoffs.

Both :mod:`balance.interop.diff_diff` and the future
:mod:`balance.interop.svy` default to the names declared here. The names
match svy's ``RepWeights.columns_from_data()`` auto-detect (per the
balance/svy plan) and the column conventions documented in
``svy.core.design``.

Adapters never enforce these names — they consume them as defaults when
the user does not pass an explicit column mapping. Locking them down in
one place keeps both adapters in sync as svy evolves.

The naming policy itself (why both adapters share these names) is
documented inline in this module's symbols and in
``balance/interop/diff_diff.py``'s module docstring -- there is no
separate external policy document to track.
"""

from __future__ import annotations

#: Convention default name for the active weight column when
#: ``Sample.from_frame`` is called WITHOUT a user-supplied
#: ``weight_column``. NOTE: ``BalanceFrame.adjust()`` preserves whatever
#: column name was active on the input Sample (see
#: ``balance_frame.py:820-822``) -- a Sample created with
#: ``weight_column="w"`` keeps ``weight_column == "w"`` after adjustment;
#: it does NOT canonicalise to ``"weight"``. This default is only used by
#: adapters when the caller has not configured a custom weight column.
WEIGHT_COLUMN: str = "weight"

#: Canonical stratum column name (svy / diff-diff design).
STRATUM_COLUMN: str = "stratum"

#: Canonical primary-sampling-unit column name.
PSU_COLUMN: str = "psu"

#: Canonical secondary-sampling-unit column name. Reserved for the future
#: ``balance.interop.svy`` adapter — diff-diff's ``SurveyDesign`` does not
#: currently expose an ``ssu`` field, so the diff-diff adapter does not
#: forward this name (no entry in ``DEFAULT_DESIGN_COLUMNS`` below).
SSU_COLUMN: str = "ssu"

#: Canonical finite-population-correction column name.
FPC_COLUMN: str = "fpc"

#: Prefix for replicate-weight columns (``repweight_1``, ``repweight_2``,
#: …). Used by svy's ``RepWeights.columns_from_data()`` auto-detect.
REPWEIGHT_PREFIX: str = "repweight_"

#: Default weight type. ``"pweight"`` is the only choice compatible with
#: the full diff-diff estimator family (CallawaySantAnna, StackedDiD,
#: HAD-continuous, ImputationDiD, TwoStageDiD, EfficientDiD, WooldridgeDiD,
#: TROP, dCDH) — see ``staggered.py:1500-1506`` and
#: ``survey.py:1051-1085`` in diff-diff for the rejection guard.
WEIGHT_TYPE_DEFAULT: str = "pweight"

#: Default mapping of design-column NAMES to forward into a
#: ``diff_diff.SurveyDesign`` (or, in the future, an ``svy.Design``). Both
#: adapters merge user-supplied overrides on top of this dict; the
#: ``weights`` slot is always overridden with the active balance weight
#: column name (the source of truth for which column is "live").
DEFAULT_DESIGN_COLUMNS: dict[str, str] = {
    "weights": WEIGHT_COLUMN,
    "strata": STRATUM_COLUMN,
    "psu": PSU_COLUMN,
    "fpc": FPC_COLUMN,
    "weight_type": WEIGHT_TYPE_DEFAULT,
}
