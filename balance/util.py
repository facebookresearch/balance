# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging

from balance.utils.data_transformation import (
    add_na_indicator,
    auto_aggregate,
    auto_spread,
    drop_na_rows,
    fct_lump,
    fct_lump_by,
    qcut,
    quantize,
    row_pairwise_diffs,
)
from balance.utils.file_utils import _to_download
from balance.utils.input_validation import (
    _check_weighting_methods_input,
    _float_or_none,
    _is_arraylike,
    _isinstance_sample,
    _true_false_str_to_bool,
    _assert_type,
    choose_variables,
    find_items_index_in_list,
    get_items_from_list_via_indices,
    guess_id_column,
    rm_mutual_nas,
)
from balance.utils.logging_utils import _truncate_text, TruncationFormatter
from balance.utils.model_matrix import (
    _prepare_input_model_matrix,
    build_model_matrix,
    dot_expansion,
    formula_generator,
    model_matrix,
    one_hot_encoding_greater_2,
    process_formula,
)
from balance.utils.pandas_utils import (
    _are_dtypes_equal,
    _astype_in_df_from_dtypes,
    _coerce_scalar,
    _compute_cardinality_metrics,
    _detect_high_cardinality_features,
    _dict_intersect,
    _is_categorical_dtype,
    _make_df_column_names_unique,
    _pd_convert_all_types,
    _process_series_for_missing_mask,
    _safe_divide_with_zero_handling,
    _safe_fillna_and_infer,
    _safe_groupby_apply,
    _safe_replace_and_infer,
    _safe_show_legend,
    _warn_of_df_dtypes_change,
    HIGH_CARDINALITY_RATIO_THRESHOLD,
    HighCardinalityFeature,
)

logger: logging.Logger = logging.getLogger(__package__)

__all__ = [
    "HIGH_CARDINALITY_RATIO_THRESHOLD",
    "HighCardinalityFeature",
    "TruncationFormatter",
    "_are_dtypes_equal",
    "_astype_in_df_from_dtypes",
    "_check_weighting_methods_input",
    "_coerce_scalar",
    "_compute_cardinality_metrics",
    "_detect_high_cardinality_features",
    "_dict_intersect",
    "_float_or_none",
    "_is_arraylike",
    "_is_categorical_dtype",
    "_isinstance_sample",
    "_make_df_column_names_unique",
    "_pd_convert_all_types",
    "_prepare_input_model_matrix",
    "_process_series_for_missing_mask",
    "_safe_divide_with_zero_handling",
    "_safe_fillna_and_infer",
    "_safe_groupby_apply",
    "_safe_replace_and_infer",
    "_safe_show_legend",
    "_to_download",
    "_truncate_text",
    "_true_false_str_to_bool",
    "_assert_type",
    "_warn_of_df_dtypes_change",
    "add_na_indicator",
    "auto_aggregate",
    "auto_spread",
    "build_model_matrix",
    "choose_variables",
    "dot_expansion",
    "drop_na_rows",
    "fct_lump",
    "fct_lump_by",
    "find_items_index_in_list",
    "formula_generator",
    "get_items_from_list_via_indices",
    "guess_id_column",
    "model_matrix",
    "one_hot_encoding_greater_2",
    "process_formula",
    "qcut",
    "quantize",
    "rm_mutual_nas",
    "row_pairwise_diffs",
]
