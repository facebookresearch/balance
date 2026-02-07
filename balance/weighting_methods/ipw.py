# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import copy
import logging
from typing import Any, cast, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from balance import adjustment as balance_adjustment, util as balance_util
from balance.stats_and_plots.weighted_comparisons_stats import (
    asmd,
    asmd_improvement as compute_asmd_improvement,
)
from balance.stats_and_plots.weights_stats import design_effect
from balance.util import _assert_type
from scipy.sparse import csc_matrix, csr_matrix, issparse
from sklearn.base import ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler


logger: logging.Logger = logging.getLogger(__package__)


# TODO: Add tests for model_coefs()
# TODO: Improve interpretability of model coefficients, as variables are no longer zero-centered.
def model_coefs(
    model: ClassifierMixin,
    feature_names: list[str] | None = None,
) -> Dict[str, Any]:
    """Extract coefficient-like information from sklearn classifiers.

    For linear models such as :class:`~sklearn.linear_model.LogisticRegression`,
    this returns the fitted coefficients (and intercept when available).  For
    classifiers that do not expose a ``coef_`` attribute (e.g. tree ensembles),
    an empty :class:`pandas.Series` is returned so downstream diagnostics can
    handle the absence of coefficients gracefully.

    Args:
        model (ClassifierMixin): Fitted sklearn classifier.
        feature_names (Optional[list], optional): Feature names associated with
            the model matrix columns. When provided and the model exposes a
            one-dimensional ``coef_`` array, the returned Series is indexed by
            ``["intercept"] + feature_names``.

    Returns:
        Dict[str, Any]: Dictionary containing a ``coefs`` key with a
        :class:`pandas.Series` of coefficients (which may be empty when the
        model does not expose linear coefficients).

    Examples:
    .. code-block:: python

        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from balance.weighting_methods.ipw import model_coefs
        X = np.array([[0.0], [1.0]])
        y = np.array([0, 1])
        model = LogisticRegression().fit(X, y)
        "coefs" in model_coefs(model)
        # True
    """

    if not hasattr(model, "coef_"):
        return {"coefs": pd.Series(dtype=float)}
    coefs = np.asarray(cast(Any, model).coef_)
    intercept = getattr(model, "intercept_", None)

    if coefs.ndim > 1 and coefs.shape[0] == 1:
        coefs = coefs.reshape(-1)

    if feature_names is not None and coefs.ndim == 1:
        index: List[str] = list(feature_names[: len(coefs)])
        values = coefs
        if intercept is not None:
            intercept_array = np.asarray(intercept).ravel()
            if intercept_array.size == 1:
                index = ["intercept"] + index
                values = np.hstack((intercept_array, coefs))
        coefs_series = pd.Series(data=values, index=index)
    else:
        coefs_series = pd.Series(data=np.ravel(coefs))

    return {
        "coefs": coefs_series,
    }


# TODO: Add tests for link_transform()
def link_transform(pred: np.ndarray) -> np.ndarray:
    """Transforms probabilities into log odds (link function).

    Args:
        pred (np.ndarray): LogisticRegression probability predictions from sklearn.

    Returns:
        np.ndarray: Array of log odds.

    Examples:
    .. code-block:: python

        import numpy as np
        from balance.weighting_methods.ipw import link_transform
        float(link_transform(np.array([0.5]))[0])
        # 0.0
    """
    pred = np.asarray(pred, dtype=float)
    # Clip probabilities to avoid dividing by zero or taking log of zero
    eps = np.finfo(float).eps
    pred = np.clip(pred, eps, 1 - eps)
    return np.log(pred / (1 - pred))


def _compute_deviance(
    y: np.ndarray,
    pred: np.ndarray,
    model_weights: np.ndarray,
    labels: list[int] | None = None,
) -> float:
    """Compute deviance (2 * log loss).

    Used multiple times throughout ipw() and calc_dev() functions.

    Args:
        y (np.ndarray): True labels.
        pred (np.ndarray): Predicted probabilities.
        model_weights (np.ndarray): Sample weights.
        labels (Optional[List[int]], optional): Label specification. Defaults to None.

    Returns:
        float: Deviance value.
    """
    if labels is not None:
        return 2 * log_loss(y, pred, sample_weight=model_weights, labels=labels)
    return 2 * log_loss(y, pred, sample_weight=model_weights)


def _compute_proportion_deviance(dev: float, null_dev: float) -> float:
    """Compute proportion of null deviance explained.

    Used multiple times in ipw() for model evaluation.

    Args:
        dev (float): Model deviance.
        null_dev (float): Null model deviance.

    Returns:
        float: Proportion of deviance explained (1 - dev/null_dev).
    """
    return 1 - dev / null_dev


def _convert_to_dense_array(
    X_matrix: Union[csc_matrix, csr_matrix, np.ndarray, pd.DataFrame],
) -> np.ndarray:
    """Convert sparse matrix or DataFrame to dense numpy array.

    If the input is a CSC matrix, first convert to CSR for efficiency,
    then convert to dense array. If already a dense numpy array or DataFrame,
    return as-is (note: DataFrames will be returned unchanged and may need
    explicit conversion to numpy array elsewhere if needed).

    Args:
        X_matrix: Input matrix - can be a sparse matrix (csc_matrix, csr_matrix),
            dense numpy array, or pandas DataFrame.

    Returns:
        np.ndarray: Dense numpy array (or DataFrame if input was DataFrame).
    """
    if isinstance(X_matrix, csc_matrix):
        X_matrix = X_matrix.tocsr()

    if issparse(X_matrix):
        X_matrix = X_matrix.toarray()

    return X_matrix


# TODO: Add tests for calc_dev()
def calc_dev(
    X_matrix: csr_matrix,
    y: np.ndarray,
    model: ClassifierMixin,
    model_weights: np.ndarray,
    foldids: np.ndarray,
) -> Tuple[float, float]:
    """10 fold cross validation to calculate holdout deviance.

    Args:
        X_matrix (csr_matrix): Model matrix,
        y (np.ndarray): Vector of sample inclusion (1=sample, 0=target),
        model (_type_): LogisticRegression object from sklearn,
        model_weights (np.ndarray): Vector of sample and target weights,
        foldids (np.ndarray): Vector of cross-validation fold indices.

    Returns:
        float, float: mean and standard deviance of holdout deviance.

    Examples:
    .. code-block:: python

        import numpy as np
        from scipy.sparse import csr_matrix
        from sklearn.linear_model import LogisticRegression
        from balance.weighting_methods.ipw import calc_dev
        X = csr_matrix(np.arange(10).reshape(10, 1))
        y = np.array([0, 1] * 5)
        model = LogisticRegression(max_iter=1000)
        weights = np.ones(10)
        foldids = np.arange(10)
        dev_mean, dev_sd = calc_dev(X, y, model, weights, foldids)
        dev_mean > 0
        # True
    """
    cv_dev = [0.0 for _ in range(10)]

    for i in range(10):
        X_train = X_matrix[foldids != i, :]
        X_test = X_matrix[foldids == i, :]
        y_train = y[foldids != i]
        y_test = y[foldids == i]
        model_weights_train = model_weights[foldids != i]
        model_weights_test = model_weights[foldids == i]
        # pyre-ignore[16]: ClassifierMixin has fit method at runtime
        model_fit = model.fit(X_train, y_train, sample_weight=model_weights_train)
        pred_test = model_fit.predict_proba(X_test)[:, 1]
        cv_dev[i] = _compute_deviance(
            y_test, pred_test, model_weights_test, labels=[0, 1]
        )

    logger.debug(
        f"dev_mean: {np.mean(cv_dev)}, dev_sd: {np.std(cv_dev, ddof=1) / np.sqrt(10)}"
    )
    return np.mean(cv_dev), np.std(cv_dev, ddof=1) / np.sqrt(10)


# TODO: consider add option to normalize weights to sample size
def weights_from_link(
    link: Any,
    balance_classes: bool,
    sample_weights: pd.Series,
    target_weights: pd.Series,
    weight_trimming_mean_ratio: None | float | int = None,
    weight_trimming_percentile: float | None = None,
    keep_sum_of_weights: bool = True,
) -> pd.Series:
    """Transform link predictions into weights, by exponentiating them, and optionally balancing the classes and trimming
    the weights, then normalize the weights to have sum equal to the sum of the target weights.

    Args:
        link (Any): link predictions
        balance_classes (bool): whether balance_classes used
        sample_weights (pd.Series): vector of sample weights
        target_weights (pd.Series): vector of sample weights
        weight_trimming_mean_ratio (Union[None, float, int], optional): to be used in :func:`trim_weights`. Defaults to None.
        weight_trimming_percentile (Optional[float], optional): to be used in :func:`trim_weights`. Defaults to None.
        keep_sum_of_weights (bool, optional): to be used in :func:`trim_weights`. Defaults to True.

    Returns:
        pd.Series: A vecotr of normalized weights (for sum of target weights)

    Examples:
    .. code-block:: python

        import numpy as np
        import pandas as pd
        from balance.weighting_methods.ipw import weights_from_link
        link = np.zeros(2)
        sample_weights = pd.Series([1.0, 1.0])
        target_weights = pd.Series([1.0, 1.0])
        weights_from_link(link, False, sample_weights, target_weights).tolist()
        # [1.0, 1.0]
    """
    link = link.reshape((link.shape[0],))
    if balance_classes:
        odds = np.sum(sample_weights) / np.sum(target_weights)
        link = link + np.log(odds)
    weights = sample_weights / np.exp(link)
    weights = balance_adjustment.trim_weights(
        weights,
        weight_trimming_mean_ratio,
        weight_trimming_percentile,
        keep_sum_of_weights=keep_sum_of_weights,
    )
    # Normalize weights such that the sum will be the sum of the weights of target
    weights = weights * np.sum(target_weights) / np.sum(weights)
    return weights


# TODO: Update choose_regularization function to be based on mse (instead of grid search)
def choose_regularization(
    links: List[Any],
    lambdas: np.ndarray,
    sample_df: pd.DataFrame,
    target_df: pd.DataFrame,
    sample_weights: pd.Series,
    target_weights: pd.Series,
    balance_classes: bool,
    max_de: float = 1.5,
    trim_options: Tuple[
        int, int, int, float, float, float, float, float, float, float
    ] = (20, 10, 5, 2.5, 1.25, 0.5, 0.25, 0.125, 0.05, 0.01),
    n_asmd_candidates: int = 10,
) -> Dict[str, Any]:
    """Searches through the regularisation parameters of the model and weight
    trimming levels to find the combination with the highest covariate
    ASMD reduction (in sample_df and target_df, NOT in the model matrix used for modeling
    the response) subject to the design effect being lower than max_de (deafults to 1.5).
    The function preforms a grid search over the n_asmd_candidates (deafults to 10) models
    with highest DE lower than max_de (assuming higher DE means more bias reduction).

    Args:
        links (Links[Any]): list of link predictions from sklearn
        lambdas (np.ndarray): the lambda values for regularization
        sample_df (pd.DataFrame): a dataframe representing the sample
        target_df (pd.DataFrame): a dataframe representing the target
        sample_weights (pd.Series): design weights for sample
        target_weights (pd.Series): design weights for target
        balance_classes (bool): whether balance_classes used
        max_de (float, optional): upper bound for the design effect of the computed weights.
            Used for choosing the model regularization and trimming.
            If set to None, then it uses 'lambda_1se'. Defaults to 1.5.
        trim_options (Tuple[ int, int, int, float, float, float, float, float, float, float ], optional):
            options for weight_trimming_mean_ratio. Defaults to (20, 10, 5, 2.5, 1.25, 0.5, 0.25, 0.125, 0.05, 0.01).
        n_asmd_candidates (int, optional): number of candidates for grid search.. Defaults to 10.

    Returns:
        Dict[str, Any]: Dict of the value of the chosen lambda, the value of trimming, model description.
            Shape is
                {
                    "best": {"s": best.s.values, "trim": best.trim.values[0]},
                    "perf": all_perf,
                }

    Examples:
    .. code-block:: python

        import numpy as np
        import pandas as pd
        from balance.weighting_methods.ipw import choose_regularization
        links = [np.zeros(2)]
        lambdas = np.array([1.0])
        sample_df = pd.DataFrame({"x": [0, 1]})
        target_df = pd.DataFrame({"x": [0, 1]})
        sample_weights = pd.Series([1.0, 1.0])
        target_weights = pd.Series([1.0, 1.0])
        result = choose_regularization(
            links,
            lambdas,
            sample_df,
            target_df,
            sample_weights,
            target_weights,
            False,
            max_de=2.0,
            trim_options=(1.0,),
            n_asmd_candidates=1,
        )
        sorted(result.keys())
        # ['best', 'perf']
    """

    logger.info("Starting choosing regularisation parameters")
    # get all non-null links
    links = [link for link in links if link is not None]

    # Grid search over regularisation parameter and weight trimming
    # First calculates design effects for all combinations, because this is cheap
    all_perf = []
    for wr in trim_options:
        for i in range(len(links)):
            s = lambdas[i]
            link = links[i]
            weights = weights_from_link(
                link,
                balance_classes,
                sample_weights,
                target_weights,
                weight_trimming_mean_ratio=wr,
            )

            deff = design_effect(weights)
            all_perf.append(
                {
                    "s": s,
                    "s_index": i,
                    "trim": wr,
                    "design_effect": deff,
                }
            )
    all_perf = pd.DataFrame(all_perf)
    best = (
        all_perf[all_perf.design_effect < max_de]
        .sort_values("design_effect")
        .tail(n_asmd_candidates)
    )
    logger.debug(f"Regularisation with design effect below {max_de}: \n {best}")

    # Calculate ASMDS for best 10 candidates (assuming that higher DE means
    #  more bias reduction)
    all_perf = []
    for _, r in best.iterrows():
        wr = r.trim
        s_index = int(r.s_index)
        s = lambdas[s_index]
        link = links[s_index]

        weights = weights_from_link(
            link,
            balance_classes,
            sample_weights,
            target_weights,
            weight_trimming_mean_ratio=wr,
        )
        adjusted_df = sample_df[sample_df.index.isin(weights.index)]

        asmd_after = asmd(
            sample_df=adjusted_df,
            target_df=target_df,
            sample_weights=weights,
            target_weights=target_weights,
        )
        asmd_impr = compute_asmd_improvement(
            sample_before=sample_df,
            sample_after=adjusted_df,
            target=target_df,
            sample_before_weights=sample_weights,
            sample_after_weights=weights,
            target_weights=target_weights,
        )
        deff = design_effect(weights)
        all_perf.append(
            {
                "s": s,
                "s_index": s_index,
                "trim": wr,
                "design_effect": deff,
                "asmd_improvement": asmd_impr,
                "asmd": asmd_after.loc["mean(asmd)"],
            }
        )

    all_perf = pd.DataFrame(all_perf)
    best = (
        all_perf[all_perf.design_effect < max_de]
        .sort_values("asmd_improvement")
        .tail(1)
    )
    logger.info(f"Best regularisation: \n {best}")
    solution = {
        "best": {"s_index": best.s_index.values[0], "trim": best.trim.values[0]},
        "perf": all_perf,
    }
    return solution


# Lambda regularization parameters can be used to speedup the IPW algorithm,
# counteracting the slow computational speed of sklearn.
def ipw(
    sample_df: pd.DataFrame,
    sample_weights: pd.Series,
    target_df: pd.DataFrame,
    target_weights: pd.Series,
    variables: list[str] | None = None,
    model: str | ClassifierMixin | None = "sklearn",
    weight_trimming_mean_ratio: int | float | None = 20,
    weight_trimming_percentile: float | None = None,
    balance_classes: bool = True,
    transformations: str | None = "default",
    na_action: str = "add_indicator",
    max_de: float | None = None,
    lambda_min: float = 1e-05,
    lambda_max: float = 10,
    num_lambdas: int = 250,
    formula: str | list[str] | None = None,
    penalty_factor: list[float] | None = None,
    # TODO: This is set to be false in order to keep reproducibility of works that uses balance.
    # The best practice is for this to be true.
    one_hot_encoding: bool = False,
    use_model_matrix: bool = True,
    random_seed: int = 2020,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Fit an ipw (inverse propensity score weighting) for the sample using the target.

    Args:
        sample_df (pd.DataFrame): a dataframe representing the sample
        sample_weights (pd.Series): design weights for sample
        target_df (pd.DataFrame): a dataframe representing the target
        target_weights (pd.Series): design weights for target
        variables (Optional[List[str]], optional): list of variables to include in the model.
            If None all joint variables of sample_df and target_df are used. Defaults to None.
        model (Union[str, ClassifierMixin, None], optional): Model used for modeling the
            propensity scores. Provide ``"sklearn"`` (default) or ``None`` to use logistic
            regression, or pass an sklearn classifier implementing ``fit`` and
            ``predict_proba``. Common choices include scikit-learn estimators such as
            :class:`sklearn.linear_model.LogisticRegression`,
            :class:`sklearn.ensemble.RandomForestClassifier`,
            :class:`sklearn.ensemble.GradientBoostingClassifier`,
            :class:`sklearn.ensemble.HistGradientBoostingClassifier`, and
            :class:`sklearn.linear_model.SGDClassifier` configured with
            ``loss="log_loss"``. To customize the built-in logistic regression
            settings, pass a configured :class:`sklearn.linear_model.LogisticRegression`
            instance as ``model``. Custom classifiers should expose a
            ``predict_proba`` method returning class probabilities.
        weight_trimming_mean_ratio (Optional[Union[int, float]], optional): indicating the ratio from above according to which
            the weights are trimmed by mean(weights) * ratio.
            Defaults to 20.
        weight_trimming_percentile (Optional[float], optional): if weight_trimming_percentile is not none, winsorization is applied.
            if None then trimming is applied. Defaults to None.
        balance_classes (bool, optional): whether to balance the sample and target size for running the model.
            True is preferable for imbalanced cases.
            It shouldn't have an effect on the final weights as this is factored
            into the computation of the weights. TODO: add ref. Defaults to True.
        transformations (str, optional): what transformations to apply to data before fitting the model.
            See apply_transformations function. Defaults to "default".
        na_action (str, optional): what to do with NAs.
            See add_na_indicator function. Defaults to "add_indicator".
        max_de (Optional[float], optional): upper bound for the design effect of the computed weights.
            Used for choosing the model regularization and trimming.
            If set to None, then it uses 'lambda_1se'. Defaults to 1.5.
        formula (Union[str, List[str], None], optional): The formula according to which build the model.
            In case of list of formula, the model matrix will be built in steps and
            concatenated together. Defaults to None.
        penalty_factor (Optional[List[float]], optional): the penalty factors used in ipw. The penalty
            should have the same length as the formula list (and applies to each element of formula).
            Smaller penalty on some formula will lead to elements in that formula to get more adjusted, i.e. to have a higher chance to get into the model (and not zero out). A penalty of 0 will make sure the element is included in the model.
            If not provided, assume the same penalty (1) for all variables. Defaults to None.
        one_hot_encoding (bool, optional): whether to encode all factor variables in the model matrix with
            almost_one_hot_encoding. This is recomended in case of using
            LASSO on the data (Default: False).
            one_hot_encoding_greater_3 creates one-hot-encoding for all
            categorical variables with more than 2 categories (i.e. the
            number of columns will be equal to the number of categories),
            and only 1 column for variables with 2 levels (treatment contrast). Defaults to False.
        use_model_matrix (bool, optional): whether to build the model matrix using
            :func:`model_matrix`. When set to False, the model is fit on the raw
            covariate data after applying transformations and adding NA indicators,
            and categorical columns are encoded as integer codes (i.e., an ordinal
            / label encoding based on the category codes in the input data). For
            scikit-learn tree-based models, these integer-coded features are treated
            as ordered numeric variables with threshold-based splits, which is not
            equivalent to true unordered categorical handling and can change results
            depending on how categories are coded. In settings where invariance to
            category coding is important (e.g., most linear models, or when using
            trees/forests without native categorical support), prefer
            ``use_model_matrix=True`` with appropriate one-hot encoding, or use a
            model that natively supports categorical features. The built-in
            logistic-regression path (i.e., when ``model`` is ``None`` or the
            default) currently requires ``use_model_matrix=True`` and will raise if
            ``use_model_matrix=False``. Defaults to True.
        random_seed (int, optional): Random seed to use. Defaults to 2020.

    Examples:
        Example 1: Using RandomForestClassifier

        .. code-block:: python

            import pandas as pd
            from balance.datasets import load_sim_data
            from balance.weighting_methods.ipw import ipw
            from sklearn.ensemble import RandomForestClassifier

            # Load simulated data
            target_df, sample_df = load_sim_data()

            # Assign weights
            sample_weights = pd.Series(1, index=sample_df.index)
            target_weights = pd.Series(1, index=target_df.index)

            # Define model
            rf = RandomForestClassifier(n_estimators=200, random_state=0)

            # Run IPW with sklearn model
            result_rf = ipw(
                sample_df,
                sample_weights,
                target_df,
                target_weights,
                variables=["gender", "age_group", "income"],
                model=rf,
                use_model_matrix=False,
            )
            print("RandomForestClassifier result:")
            print(result_rf)

        Output::

            RandomForestClassifier result:
            {'weight': 0      6.727622
            1      6.811271
            2      1.861028
            3      5.023780
            4      8.212948
                     ...
            995    8.302406
            996    2.229000
            997    8.344916
            998    9.747695
            999    6.018009
            Length: 1000, dtype: float64, 'model': {'method': 'ipw', ...}}

        Example 2: Using default sklearn model (LogisticRegression)

        .. code-block:: python

            # Run IPW with default sklearn model
            result_sklearn = ipw(
                sample_df,
                sample_weights,
                target_df,
                target_weights,
                variables=["gender", "age_group", "income"],
                model="sklearn",
            )
            print("Default sklearn model result:")
            print(result_sklearn)

        Output::

            Default sklearn model result:
            {'weight': 0       6.531728
            1       9.617159
            2       3.562973
            3       6.952117
            4       5.129230
                     ...
            995     9.353052
            996     3.973554
            997     7.095483
            998    11.331144
            999     7.913133
            Length: 1000, dtype: float64, 'model': {'method': 'ipw', ...}}

        Example 3: Comparing weights from different models

        .. code-block:: python

            import pandas as pd
            # Combine weights into a DataFrame
            comparison_df = pd.DataFrame({
                'RandomForest': result_rf['weight'],
                'Sklearn': result_sklearn['weight']
            })
            # Calculate difference
            comparison_df['Difference'] = comparison_df['RandomForest'] - comparison_df['Sklearn']
            # Print summary statistics
            print("\nMean difference:", comparison_df['Difference'].mean())
            print("Std difference:", comparison_df['Difference'].std())

        Output::

            Mean difference: -2.9416469260468146e-15
            Std difference: 8.043311719328154

    Raises:
        Exception: f"Sample indicator only has value {_n_unique}. This can happen when your sample or target are empty from unknown reason"
        NotImplementedError: If ``model`` is a string other than "sklearn" (the
            built-in logistic regression option) or the deprecated "glmnet".
        TypeError: If ``model`` is neither a supported string nor an sklearn
            classifier exposing ``predict_proba``.

    Returns:
        Dict[str, Any]: A dictionary includes:
            "weight" --- The weights for the sample.
            "model" --- parameters of the model:fit, performance, X_matrix_columns, lambda,
                        weight_trimming_mean_ratio
            Shape of the Dict:
            {
                "weight": weights,
                "model": {
                    "method": "ipw",
                    "X_matrix_columns": X_matrix_columns_names,
                    "fit": fit,
                    "perf": performance,
                    "lambda": best_s,
                    "weight_trimming_mean_ratio": weight_trimming_mean_ratio,
                },
            }
    """
    custom_model: ClassifierMixin | None = None
    model_name: str | None = None

    if isinstance(model, ClassifierMixin):
        custom_model = model
        model_name = "sklearn"
    elif model is None:
        model_name = "sklearn"
    elif isinstance(model, str):
        model_name = model
    else:
        raise TypeError(
            "model must be 'sklearn', an sklearn classifier implementing predict_proba, or None"
        )

    if model_name == "glmnet":
        raise NotImplementedError("glmnet is no longer supported")
    elif model_name != "sklearn":
        raise NotImplementedError(
            f"Model '{model_name}' is not supported. Only 'sklearn' is currently implemented."
        )

    logger.info("Starting ipw function")
    np.random.seed(
        random_seed
    )  # setting random seed for cases of variations in sklearn

    balance_util._check_weighting_methods_input(sample_df, sample_weights, "sample")
    balance_util._check_weighting_methods_input(target_df, target_weights, "target")

    variables = balance_util.choose_variables(sample_df, target_df, variables=variables)
    logger.debug(f"Join variables for sample and target: {variables}")

    sample_df = sample_df.loc[:, variables]
    target_df = target_df.loc[:, variables]

    if na_action == "drop":
        (sample_df, sample_weights) = balance_util.drop_na_rows(
            sample_df, sample_weights, "sample"
        )
        (target_df, target_weights) = balance_util.drop_na_rows(
            target_df, target_weights, "target"
        )
    sample_n = sample_df.shape[0]
    target_n = target_df.shape[0]

    # Applying transformations
    # Important! Variables that don't need transformations
    # should be transformed with the *identity function*,
    # otherwise will be dropped from the model
    sample_df, target_df = balance_adjustment.apply_transformations(
        (sample_df, target_df), transformations=transformations
    )
    variables = list(sample_df.columns)
    logger.debug(f"Final variables in the model: {variables}")

    if use_model_matrix:
        logger.info("Building model matrix")
        # Convert formula to List[str] if it's a single string
        formula_list: list[str] | None = (
            [formula] if isinstance(formula, str) else formula
        )
        model_matrix_output = balance_util.model_matrix(
            sample_df,
            target_df,
            variables,
            add_na=(na_action == "add_indicator"),
            return_type="one",
            return_var_type="sparse",
            formula=formula_list,
            penalty_factor=penalty_factor,
            one_hot_encoding=one_hot_encoding,
        )
        X_matrix = cast(
            Union[pd.DataFrame, np.ndarray, csc_matrix],
            model_matrix_output["model_matrix"],
        )
        X_matrix_columns_names = cast(
            List[str], model_matrix_output["model_matrix_columns_names"]
        )
        penalty_factor_expanded = cast(
            List[float], model_matrix_output["penalty_factor"]
        )
        logger.info(
            f"The formula used to build the model matrix: {model_matrix_output['formula']}"
        )
        logger.info(f"The number of columns in the model matrix: {X_matrix.shape[1]}")
        logger.info(f"The number of rows in the model matrix: {X_matrix.shape[0]}")
    else:
        if custom_model is None:
            raise ValueError(
                "use_model_matrix=False is only supported when providing a custom "
                "sklearn classifier (e.g., RandomForestClassifier)."
            )
        if formula is not None:
            logger.warning(
                "Argument 'formula' is ignored because use_model_matrix=False; "
                "no model matrix will be constructed."
            )
        if one_hot_encoding:
            logger.warning(
                "Argument 'one_hot_encoding' is ignored because use_model_matrix=False; "
                "no model matrix will be constructed."
            )
        if penalty_factor is not None:
            logger.warning(
                "Argument 'penalty_factor' is ignored because use_model_matrix=False "
                "with a custom sklearn model; this argument only applies when "
                "use_model_matrix=True with the built-in logistic regression model."
            )

        combined = pd.concat([sample_df, target_df], axis=0)
        if na_action == "add_indicator":
            existing_indicator_cols = [
                col
                for col in combined.columns
                if isinstance(col, str) and col.startswith("_is_na_")
            ]
            if not existing_indicator_cols:
                combined = balance_util.add_na_indicator(combined)
            else:
                base_cols = [
                    col
                    for col in combined.columns
                    if col not in existing_indicator_cols
                ]
                combined_base = balance_util.add_na_indicator(combined[base_cols])
                combined_base = combined_base.drop(
                    columns=existing_indicator_cols, errors="ignore"
                )
                combined = pd.concat(
                    [combined_base, combined[existing_indicator_cols]],
                    axis=1,
                )

        categorical_cols = combined.select_dtypes(
            ["category", "string", "boolean", "object"]
        ).columns
        for col in categorical_cols:
            combined[col] = pd.Categorical(combined[col]).codes

        X_matrix = combined.to_numpy()
        X_matrix_columns_names = combined.columns.tolist()
        penalty_factor_expanded = [1.0] * combined.shape[1]
        logger.info("Fitting model on raw covariates without model matrix encoding.")
        logger.info(f"The number of columns in the raw matrix: {X_matrix.shape[1]}")
        logger.info(f"The number of rows in the raw matrix: {X_matrix.shape[0]}")

    y = np.concatenate((np.ones(sample_n), np.zeros(target_n)))
    _n_unique = np.unique(y.reshape(y.shape[0]))
    if len(_n_unique) == 1:
        raise Exception(
            f"Sample indicator only has value {_n_unique}. This can happen when your "
            "sample or target are empty from unknown reason"
        )

    if balance_classes:
        odds = np.sum(sample_weights) / np.sum(target_weights)
    else:
        odds = 1
    logger.debug(f"odds for balancing classes: {odds}")

    model_weights = np.concatenate((sample_weights, target_weights * odds))

    logger.debug(f"X_matrix shape: {X_matrix.shape}")
    logger.debug(f"y input shape: {y.shape}")
    logger.debug(
        f"penalty_factor frequency table {pd.crosstab(index=penalty_factor_expanded, columns='count')}"
    )

    foldids = np.resize(range(10), y.shape[0])
    np.random.shuffle(
        foldids
    )  # shuffels the values of foldid - note that we set the seed in the beginning of the function, so this order is fixed
    logger.debug(
        f"foldid frequency table {pd.crosstab(index=foldids, columns='count')}"
    )
    logger.debug(f"first 10 elements of foldids: {foldids[0:9]}")

    logger.debug("Fitting propensity model")

    null_dev = _compute_deviance(
        y,
        np.full(len(y), np.sum(model_weights * y) / np.sum(model_weights)),
        model_weights,
    )

    if custom_model is None:  # using_default_logistic
        # Standardize columns of the X matrix and penalize the columns of the X matrix according to the penalty_factor.
        # Workaround for sklearn, which doesn't allow for covariate specific penalty terms.
        # Note that penalty = 0 is not truly supported, and large differences in penalty factors
        # may affect convergence speed.

        scaler = StandardScaler(with_mean=False, copy=False)

        # TODO: add test to verify expected behavior from model_weights
        X_matrix = scaler.fit_transform(X_matrix, sample_weight=model_weights)

        if penalty_factor is not None:
            penalties_skl = [
                # TODO: fix 'magic numbers' that are not explained in the code
                1 / pf if pf > 0.1 else 10
                for pf in penalty_factor_expanded
            ]
            for i in range(len(penalties_skl)):
                X_matrix[:, i] *= penalties_skl[i]

        X_matrix = csr_matrix(X_matrix)

        lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), num_lambdas)

        # Using L2 regression since L1 is too slow. Observed "lbfgs" was the most computationally efficient solver.
        # NOTE: default uses "penalty": "l2", for sklearn < 1.18
        #       and "l1_ratio": -.0, for sklearn >= 1.18
        #       see: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        lr_kwargs: Dict[str, Any] = {
            "solver": "lbfgs",
            "tol": 1e-4,
            "max_iter": 5000,
            "warm_start": True,
        }

        lr = LogisticRegression(**lr_kwargs)
        fits: list[ClassifierMixin | None] = [None for _ in range(len(lambdas))]
        links: list[np.ndarray | None] = [None for _ in range(len(lambdas))]
        prop_dev = [np.nan for _ in range(len(lambdas))]
        dev = [np.nan for _ in range(len(lambdas))]
        cv_dev_mean = [np.nan for _ in range(len(lambdas))]
        cv_dev_sd = [np.nan for _ in range(len(lambdas))]

        prev_prop_dev = None

        for i in range(len(lambdas)):
            # Conversion between glmnet lambda penalty parameter and sklearn 'C' parameter referenced
            # from https://stats.stackexchange.com/questions/203816/logistic-regression-scikit-learn-vs-glmnet
            lr.C = 1 / (sum(model_weights) * lambdas[i])

            model = lr.fit(X_matrix, y, sample_weight=model_weights)
            pred = model.predict_proba(X_matrix)[:, 1]
            dev[i] = _compute_deviance(y, pred, model_weights)
            prop_dev[i] = _compute_proportion_deviance(dev[i], null_dev)

            # Early stopping criteria: improvement in prop_dev is less than 1e-5 (mirrors glmnet)
            if (
                np.sum(np.abs(model.coef_)) > 0
                and prev_prop_dev is not None
                and prop_dev[i] - prev_prop_dev < 1e-5
            ):
                break

            # Cross-validation procedure is only used for choosing best lambda if max_de is None
            # Previously, cross validation was run even when max_de is not None,
            # but the results weren't used for model selection.
            if max_de is not None:
                logger.debug(
                    f"iter {i}: lambda: {lambdas[i]}, dev: {dev[i]}, prop_dev: {prop_dev[i]}"
                )
            elif num_lambdas > 1:
                dev_mean, dev_sd = calc_dev(X_matrix, y, lr, model_weights, foldids)
                logger.debug(
                    f"iter {i}: lambda: {lambdas[i]}, cv_dev: {dev_mean}, dev_diff: {dev_mean - dev[i]}, prop_dev: {prop_dev[i]}"
                )
                cv_dev_mean[i] = dev_mean
                cv_dev_sd[i] = dev_sd

            prev_prop_dev = prop_dev[i]
            links[i] = link_transform(pred)[:sample_n,]
            fits[i] = copy.deepcopy(model)

    else:
        if penalty_factor is not None:
            logger.warning("penalty_factor is ignored when using a custom model.")

        cloned_model = clone(custom_model)
        if not hasattr(cloned_model, "predict_proba"):
            raise ValueError(
                "The provided custom model must implement predict_proba for propensity estimation."
            )

        X_matrix = _convert_to_dense_array(X_matrix)

        lambdas = np.array([np.nan])
        fits: list[ClassifierMixin | None] = [None]
        links: list[np.ndarray | None] = [None]
        prop_dev = [np.nan]
        dev = [np.nan]
        cv_dev_mean = [np.nan]
        cv_dev_sd = [np.nan]

        model = cloned_model.fit(X_matrix, y, sample_weight=model_weights)
        probas = model.predict_proba(X_matrix)
        if probas.ndim != 2 or probas.shape[1] < 2:
            raise ValueError(
                "The provided custom model predict_proba must return probability estimates for both classes."
            )
        try:
            class_index = list(model.classes_).index(1)
        except ValueError as error:
            raise ValueError(
                "The provided custom model must be trained on the binary labels {0, 1}."
            ) from error
        pred = probas[:, class_index]
        dev[0] = _compute_deviance(y, pred, model_weights)
        prop_dev[0] = _compute_proportion_deviance(dev[0], null_dev)
        links[0] = link_transform(pred)[:sample_n,]
        fits[0] = copy.deepcopy(model)

    logger.info("Done with sklearn")

    logger.info(f"max_de: {max_de}")

    best_s_index = 0
    regularisation_perf = None
    min_s_index = 0

    if max_de is not None:
        regularisation_perf = choose_regularization(
            links,
            lambdas,
            sample_df,
            target_df,
            sample_weights,
            target_weights,
            balance_classes,
            max_de,
        )
        best_s_index = regularisation_perf["best"]["s_index"]
        weight_trimming_mean_ratio = regularisation_perf["best"]["trim"]
        weight_trimming_percentile = None
    elif num_lambdas > 1 and custom_model is None:  # using_default_logistic
        # Cross-validation procedure
        logger.info("Starting model selection")

        min_s_index = np.nanargmin(cv_dev_mean)
        min_dev_mean = cv_dev_mean[min_s_index]
        min_dev_sd = cv_dev_sd[min_s_index]

        # Mirrors 'lambda.1se' from glmnet:
        # 'the most regularized model such that the cross-validated error is within one standard error of the minimum.'
        best_s_index = np.argmax(
            [
                (
                    l
                    if (loss is not np.nan) and (loss < min_dev_mean + min_dev_sd)
                    else 0
                )
                for loss, l in zip(cv_dev_mean, lambdas)
            ]
        )

    best_model = fits[best_s_index]
    link = links[best_s_index]
    best_s = lambdas[best_s_index]

    logger.debug("Predicting")
    weights = weights_from_link(
        link,
        balance_classes,
        sample_weights,
        target_weights,
        weight_trimming_mean_ratio,
        weight_trimming_percentile,
    )

    logger.info(f"Chosen lambda: {best_s}")
    best_model = _assert_type(best_model)
    performance = model_coefs(
        best_model,
        feature_names=list(X_matrix_columns_names),
    )
    performance["null_deviance"] = null_dev
    performance["deviance"] = dev[best_s_index]
    performance["prop_dev_explained"] = prop_dev[best_s_index]
    if (
        max_de is None and num_lambdas > 1 and custom_model is None
    ):  # using_default_logistic
        performance["cv_dev_mean"] = cv_dev_mean[best_s_index]
        performance["lambda_min"] = lambdas[min_s_index]
        performance["min_cv_dev_mean"] = cv_dev_mean[min_s_index]
        performance["min_cv_dev_sd"] = cv_dev_sd[min_s_index]

    dev = performance["prop_dev_explained"]
    logger.info(f"Proportion null deviance explained {dev}")

    if (np.max(weights) - np.min(weights)) / np.mean(
        weights
    ) < 1e-04:  # All weights are (essentially) the same
        logger.warning("All weights are identical. The estimates will not be adjusted")

    if dev < 0.10:
        logger.warning(
            "The propensity model has low fraction null deviance explained "
            f"({dev}). Results may not be accurate"
        )

    out = {
        "weight": weights,
        "model": {
            "method": "ipw",
            "X_matrix_columns": X_matrix_columns_names,
            "fit": fits[best_s_index],
            "perf": performance,
            "lambda": best_s,
            "weight_trimming_mean_ratio": weight_trimming_mean_ratio,
            "regularisation_perf": regularisation_perf,
        },
    }

    logger.debug("Done ipw function")

    return out
