# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# TODO: replace with "from __future__ import annotations" and remove dead Python 2 imports
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import pickle
import re
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
from balance import adjustment as balance_adjustment, util as balance_util
from balance.util import _safe_fillna_and_infer
from patsy.highlevel import ModelDesc

logger: logging.Logger = logging.getLogger(__package__)


def poststratify(
    sample_df: pd.DataFrame,
    sample_weights: pd.Series,
    target_df: pd.DataFrame,
    target_weights: pd.Series,
    variables: Optional[List[str]] = None,
    transformations: Union[Dict[str, Callable[..., Any]], str, None] = "default",
    transformations_drop: bool = True,
    strict_matching: bool = True,
    na_action: str = "add_indicator",
    weight_trimming_mean_ratio: Union[float, int, None] = None,
    weight_trimming_percentile: Union[float, None] = None,
    keep_sum_of_weights: bool = True,
    *args: Any,
    formula: Optional[Union[str, List[str]]] = None,
    store_fit_metadata: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Perform cell-based post-stratification to adjust sample weights so that the sample matches the joint distribution of, one or more, specified variables in the target population.

    This method computes one weight per *cell* - a unique combination of the supplied variables - so that the weighted sample reproduces the cell distribution observed in the target population.
    When more than one variable is supplied, the function operates on cells from the joint distribution (as opposed to raking, which operates on the marginals distribution).

    Reference: https://docs.wfp.org/api/documents/WFP-0000121326/download/

    Args:
        sample_df (pd.DataFrame): DataFrame representing the sample.
        sample_weights (pd.Series): Design weights for the sample.
        target_df (pd.DataFrame): DataFrame representing the target population.
        target_weights (pd.Series): Design weights for the target.
        variables (Optional[List[str]], optional): List of variables to define post-stratification cells. If None, uses the intersection of columns in sample_df and target_df.
        transformations (Dict[str, Callable[..., Any]] | str | None, optional):
            Transformations to apply to data before fitting the model.
            Accepts the same forms as
            :func:`balance.adjustment.apply_transformations`. Defaults to
            ``"default"``.
        transformations_drop (bool, optional): If True, drops variables not affected by transformations. Default is True.
        strict_matching (bool, optional): If True, requires all sample cells to be present in the target. If False, cells missing in the target are assigned weight 0 (and a warning is raised). Default is True.
        na_action (str, optional): How to handle missing values. Use
            ``"add_indicator"`` to treat missing values as their own category, or
            ``"drop"`` to remove rows with missing values from both sample and target.
            Defaults to ``"add_indicator"``.
        weight_trimming_mean_ratio (Union[float, int, None], optional): Forwarded to
            :func:`balance.adjustment.trim_weights` to clip weights at a multiple of the mean.
        weight_trimming_percentile (Union[float, None], optional): Percentile limit(s) for
            winsorisation, passed to :func:`balance.adjustment.trim_weights`.
        keep_sum_of_weights (bool, optional): Preserve the sum of weights during trimming before
            the final normalisation to the target total. Defaults to True.
        formula (Optional[Union[str, List[str]]], optional): Formula-like
            specification to select post-stratification variables, as an
            alternative to ``variables``. Supported operators are ``:``
            (interaction), ``.`` (all common columns of sample and target),
            ``-`` (exclude a variable), and an optional leading ``~``
            (the LHS is ignored). Examples: ``"a:b:c"``, ``"."``,
            ``". - c"``, ``"y ~ a:b"``, ``["a", "b"]`` (list form
            joint-cells all items).

            Additive operators ``+`` and ``*`` are **not** supported and
            will raise ``ValueError``. Post-stratification defines cells
            by the *joint* distribution of the selected variables — every
            variable added only refines the cell grid — so ``a + b``,
            ``a * b`` and ``a:b`` would all produce identical cells.
            Rejecting ``+``/``*`` prevents users from silently writing a
            formula that looks like a main-effects model but is treated
            as a joint interaction. (Note: raking, unlike
            post-stratification, operates on marginals and will support
            additive formulas when it gains a ``formula=`` argument.)

            Parsing uses patsy operators for variable extraction only;
            general patsy transforms/functions (e.g., ``np.log(a)``) are
            not supported. Mutually exclusive with non-empty
            ``variables``.
        *args: Additional positional arguments (currently unused).
        store_fit_metadata (bool, optional): Whether to include fit-time
            artifacts in the returned model dictionary so
            ``BalanceFrame.predict_weights()`` can reconstruct
            poststratification weights. Defaults to ``False``.
        **kwargs: Reserved for backward compatibility. Unknown keys raise
            ``TypeError`` to avoid silently ignoring typos.

    Returns:
        dict:
            weight (pd.Series): Final weights for the sample, summing to the target's total weight.
            model (dict): Description of the adjustment method used, with
                optional fit metadata when ``store_fit_metadata=True``.

    Raises:
        ValueError: If strict_matching is True and some sample cells are missing in the target.

    Notes:
        * The function expects that every combination of ``variables`` present
          in ``sample_df`` is also present in ``target_df``. Set
          ``strict_matching=False`` to keep rows whose cell is missing in the
          target and assign them weight 0.
        * When no ``variables`` are provided, the intersection of columns in
          ``sample_df`` and ``target_df`` is used. In practice you will
          usually provide a small number of categorical variables (often one
          or two) describing the post-stratification cells.

    Examples:
        Post-stratifying on a single categorical variable:

            >>> import pandas as pd
            >>> sample_df = pd.DataFrame({"gender": ["Female", "Male", "Female"]})
            >>> target_df = pd.DataFrame({"gender": ["Female", "Female", "Male", "Male"]})
            >>> design = pd.Series(1, index=sample_df.index)
            >>> target_design = pd.Series(1, index=target_df.index)
            >>> weights = poststratify(
            ...     sample_df=sample_df,
            ...     sample_weights=design,
            ...     target_df=target_df,
            ...     target_weights=target_design,
            ...     variables=["gender"],
            ... )["weight"]
            >>> weights.tolist()
            [1.0, 2.0, 1.0]

        Post-stratifying on the joint distribution of two variables (the
        resulting weights depend on the combination of both columns rather
        than their marginals):

            >>> sample_df = pd.DataFrame(
            ...     {
            ...         "gender": ["Female", "Female", "Male", "Male"],
            ...         "age_group": ["18-34", "35+", "18-34", "35+"],
            ...     }
            ... )
            >>> target_df = pd.DataFrame(
            ...     {
            ...         "gender": ["Female", "Female", "Female", "Male", "Male", "Male"],
            ...         "age_group": ["18-34", "18-34", "35+", "18-34", "35+", "35+"],
            ...     }
            ... )
            >>> design = pd.Series(1, index=sample_df.index)
            >>> target_design = pd.Series(1, index=target_df.index)
            >>> weights = poststratify(
            ...     sample_df=sample_df,
            ...     sample_weights=design,
            ...     target_df=target_df,
            ...     target_weights=target_design,
            ...     variables=["gender", "age_group"],
            ... )["weight"]
            >>> weights.tolist()
            [2.0, 1.0, 1.0, 2.0]
    """
    balance_util._check_weighting_methods_input(sample_df, sample_weights, "sample")
    balance_util._check_weighting_methods_input(target_df, target_weights, "target")

    if ("weight" in sample_df.columns.values) or ("weight" in target_df.columns.values):
        raise ValueError(
            "weight can't be a name of a column in sample or target when applying poststratify"
        )

    if variables is not None and len(variables) == 0:
        variables = None
    explicit_cell_selection = formula is not None or variables is not None

    if formula is not None and variables is not None:
        raise ValueError("Specify only one of `variables` or `formula`.")

    if not isinstance(store_fit_metadata, bool):
        raise TypeError("`store_fit_metadata` must be a bool.")
    if kwargs:
        unknown = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected keyword arguments: {unknown}")

    original_sample_weights: Optional[pd.Series] = (
        sample_weights.copy() if store_fit_metadata else None
    )
    original_target_weights: Optional[pd.Series] = (
        target_weights.copy() if store_fit_metadata else None
    )

    if formula is not None:
        variables = _variables_from_formula(sample_df, target_df, formula)

    variables = balance_util.choose_variables(sample_df, target_df, variables=variables)
    variables_before_transformations = list(variables)
    logger.debug(f"Join variables for sample and target: {variables}")

    transformations_to_apply = transformations
    # When cell-definition variables are explicitly set (via `variables` or
    # `formula`), ensure cell-definition precedence: only transformations on
    # selected variables are applied. Transformations for out-of-scope keys are
    # ignored so they cannot be treated as additions.
    if explicit_cell_selection and isinstance(transformations_to_apply, dict):
        selected = set(variables)
        filtered = {k: v for k, v in transformations_to_apply.items() if k in selected}
        transformations_to_apply = filtered if filtered else None

    sample_df = sample_df.loc[:, variables]
    target_df = target_df.loc[:, variables]

    if na_action == "drop":
        (sample_df, sample_weights) = balance_util.drop_na_rows(
            sample_df, sample_weights, "sample"
        )
        (target_df, target_weights) = balance_util.drop_na_rows(
            target_df, target_weights, "target"
        )
    elif na_action == "add_indicator":
        sample_df = pd.DataFrame(_safe_fillna_and_infer(sample_df, "__NaN__"))
        target_df = pd.DataFrame(_safe_fillna_and_infer(target_df, "__NaN__"))
    else:
        raise ValueError("`na_action` must be 'add_indicator' or 'drop'")

    sample_df, target_df = balance_adjustment.apply_transformations(
        (sample_df, target_df),
        transformations=transformations_to_apply,
        drop=transformations_drop,
    )
    variables = list(sample_df.columns)
    logger.debug(f"Final variables in the model after transformations: {variables}")

    target_df = target_df.assign(weight=target_weights)
    target_cell_props = target_df.groupby(list(variables))["weight"].sum()

    sample_df = sample_df.assign(design_weight=sample_weights)
    sample_cell_props = sample_df.groupby(list(variables))["design_weight"].sum()

    combined = pd.merge(
        target_cell_props,
        sample_cell_props,
        right_index=True,
        left_index=True,
        how="outer",
    )

    # check that all combinations of cells in sample_df are also in target_df
    if any(combined["weight"].isna()):
        if strict_matching:
            raise ValueError(
                "all combinations of cells in sample_df must be in target_df. Set strict_matching=False to continue."
            )
        else:
            logger.warning(
                "Detected some cells in sample_df that are not in target_df. "
                "Samples in cells not covered by the target are given weight 0."
            )
            combined["weight"] = _safe_fillna_and_infer(combined["weight"], 0)

    combined["weight"] = combined["weight"] / combined["design_weight"]
    sample_df = sample_df.join(combined["weight"], on=variables)
    raw_weights = sample_df.weight * sample_df.design_weight
    target_total = raw_weights.sum()
    w = balance_adjustment.trim_weights(
        raw_weights,
        target_sum_weights=target_total,
        weight_trimming_mean_ratio=weight_trimming_mean_ratio,
        weight_trimming_percentile=weight_trimming_percentile,
        keep_sum_of_weights=keep_sum_of_weights,
    ).rename(raw_weights.name)

    model: Dict[str, Any] = {"method": "poststratify"}
    if store_fit_metadata:
        if original_sample_weights is None or original_target_weights is None:
            raise RuntimeError("Unexpected missing stored training weights.")
        # Persisting non-pickleable callables (e.g., lambdas/closures) breaks
        # serialization workflows for fitted objects. Require picklable
        # transformations when fit metadata storage is enabled.
        try:
            pickle.dumps(transformations_to_apply)
        except Exception as exc:
            raise ValueError(
                "`transformations` must be pickleable when "
                "store_fit_metadata=True. Pass store_fit_metadata=False to "
                "disable fit-artifact persistence for this run."
            ) from exc
        model.update(
            {
                "variables": variables,
                "variables_before_transformations": variables_before_transformations,
                "na_action": na_action,
                "strict_matching": strict_matching,
                "transformations": transformations_to_apply,
                "transformations_drop": transformations_drop,
                "weight_trimming_mean_ratio": weight_trimming_mean_ratio,
                "weight_trimming_percentile": weight_trimming_percentile,
                "keep_sum_of_weights": keep_sum_of_weights,
                "cell_weight_ratio": combined["weight"].copy(),
                "training_sample_weights": original_sample_weights,
                "training_target_weights": original_target_weights,
                "sample_index": sample_df.index.copy(),
                "target_index": target_df.index.copy(),
                "store_fit_metadata": True,
            }
        )

    return {
        "weight": w,
        "model": model,
    }


def _variables_from_formula(
    sample_df: pd.DataFrame,
    target_df: pd.DataFrame,
    formula: Union[str, List[str]],
) -> List[str]:
    """Resolve a post-stratification variable list from formula snippets."""
    if not isinstance(formula, (str, list)):
        raise ValueError("`formula` must be a string or list of strings.")

    formulas: List[str] = [formula] if isinstance(formula, str) else formula
    if len(formulas) == 0:
        raise ValueError("`formula` must contain at least one formula string.")

    common_columns = set(sample_df.columns).intersection(set(target_df.columns))
    common_columns_ordered = [c for c in sample_df.columns if c in common_columns]
    variables: List[str] = []
    dot_pattern = re.compile(r"(?<![A-Za-z0-9_])\.(?![A-Za-z0-9_])")

    for formula_item in formulas:
        if not isinstance(formula_item, str):
            raise ValueError("Each element of `formula` must be a string.")

        item = formula_item.strip()
        if len(item) == 0:
            raise ValueError("Formula items must be non-empty strings.")

        # Poststratify defines cells by the joint distribution of all selected
        # variables. Additive ('+') and mixed ('*') operators would therefore
        # behave identically to ':' and are rejected to prevent silent
        # semantic confusion. Raking, unlike poststratify, operates on
        # marginals and should accept '+' and '*' when it gains `formula=`.
        # We check the user-provided RHS (post `~`, pre `.` expansion) so that
        # our own '+' injection during dot expansion is not flagged.
        rhs_for_check = item.split("~", 1)[1] if "~" in item else item
        for forbidden_op in ("+", "*"):
            if forbidden_op in rhs_for_check:
                raise ValueError(
                    f"Poststratify formula operator '{forbidden_op}' is not "
                    "supported. Poststratify defines cells by the joint "
                    "distribution of the specified variables, so '+' "
                    "(additive) and '*' (main + interaction) would be "
                    "semantically identical to ':' here. Use ':' (e.g., "
                    "'a:b:c'), '.' (all common columns), '-' (exclude), or "
                    "pass `variables=[...]`. Note: raking, which operates "
                    "on marginals, will support '+' and '*' when it gains "
                    "a `formula=` argument."
                )

        if item == ".":
            for col in common_columns_ordered:
                if col not in variables:
                    variables.append(col)
            continue

        if dot_pattern.search(item):
            if len(common_columns_ordered) == 0:
                raise ValueError(
                    "Cannot expand '.' in `formula` because sample and target share no columns."
                )
            expanded = " + ".join(common_columns_ordered)
            item = dot_pattern.sub(f"({expanded})", item)

        formula_for_parse = item if "~" in item else f"~ {item}"
        model_desc = ModelDesc.from_formula(formula_for_parse)
        for term in model_desc.rhs_termlist:
            for factor in term.factors:
                code = getattr(factor, "code", str(factor)).strip()
                if code in common_columns:
                    if code not in variables:
                        variables.append(code)
                    continue

                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", code):
                    raise ValueError(
                        f"Variable '{code}' from `formula` is not present in both sample and target."
                    )

                raise ValueError(
                    "Unsupported poststratify formula term "
                    f"'{code}'. Use raw column names joined by ':' "
                    "(e.g., 'a:b') or pass `variables=[...]`."
                )

    if len(variables) == 0:
        raise ValueError(
            "Could not resolve poststratification variables from `formula`."
        )

    return variables
