# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from balance.sample_class import Sample
from balance.adjustment import _find_adjustment_method
from balance.weighting_methods.ipw import weights_from_link, link_transform
from balance.util import model_matrix
from balance.adjustment import apply_transformations

logger: logging.Logger = logging.getLogger(__package__)


class BalanceWeighting(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible estimator for calculating sample weights using various methods
    (IPW, CBPS, Rake, Poststratify, Null).

    Attributes:
        method (str): The weighting method to use ("ipw", "cbps", "rake", "poststratify", "null").
        kwargs (dict): Additional arguments passed to the weighting method.
        model_ (dict): The fitted model/results from the adjustment.
    """

    def __init__(
        self,
        method: Literal["cbps", "ipw", "null", "poststratify", "rake"] = "ipw",
        **kwargs: Any,
    ) -> None:
        self.method = method
        self.kwargs = kwargs
        self.model_: Dict[str, Any] | None = None
        self._target: Sample | None = None

    def fit(self, X: Sample, target: Sample | None = None) -> "BalanceWeighting":
        """
        Fit the weighting model using the sample X and the target.

        Args:
            X (Sample): The source sample to be weighted.
            target (Sample, optional): The target sample to match. If None, X must have a target set.

        Returns:
            self: The fitted estimator.
        """
        if target is None:
            if not X.has_target():
                raise ValueError("Target must be provided or set in the Sample object.")
            target = X._links["target"]
        
        self._target = target

        # We use the Sample.adjust logic but we want to capture the model
        # Sample.adjust calls adjustment_function(...)
        # We will replicate the call to the adjustment function to get the model directly.
        
        adjustment_function = _find_adjustment_method(self.method)
        
        # Prepare arguments for the adjustment function
        # This mirrors what Sample.adjust does
        sample_covars_df = X.covars().df
        target_covars_df = target.covars().df
        
        # Perform a lightweight high-cardinality check similar in spirit to Sample.adjust.
        # We warn (rather than error) to avoid changing behavior while still surfacing UX issues.
        def _check_high_cardinality(df: pd.DataFrame, df_name: str) -> None:
            """
            Emit a warning if any object/categorical covariate has very high cardinality.
            """
            high_card_cols: list[tuple[str, int]] = []
            for col in df.columns:
                series = df[col]
                # Focus on columns where high cardinality is most problematic for modeling.
                if series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype):
                    n_levels = int(series.nunique(dropna=True))
                    if n_levels > 100:
                        high_card_cols.append((str(col), n_levels))
            if high_card_cols:
                details = ", ".join(f"{name} ({n_levels} levels)" for name, n_levels in high_card_cols)
                logger.warning(
                    "High-cardinality covariates detected in %s: %s. "
                    "This may lead to large model matrices and slow or unstable fitting. "
                    "Consider binning or excluding some of these variables.",
                    df_name,
                    details,
                )

        _check_high_cardinality(sample_covars_df, "sample_covars_df")
        _check_high_cardinality(target_covars_df, "target_covars_df")

        # Call the adjustment function
        # Note: Sample.adjust performs additional checks; here we include a basic
        # high-cardinality warning while keeping the estimator behavior unchanged.
        
        result = adjustment_function(
            sample_df=sample_covars_df,
            sample_weights=X.weight_column,
            target_df=target_covars_df,
            target_weights=target.weight_column,
            **self.kwargs
        )
        
        self.model_ = result["model"]
        
        # For methods that don't return a predictive model (like raking might not in a portable way),
        # we might be limited in predict/transform on NEW data, but fit works.
        
        return self

    def predict(self, X: Sample) -> pd.Series:
        """
        Predict weights for the sample X using the fitted model.

        Args:
            X (Sample): The sample to predict weights for.

        Returns:
            pd.Series: The predicted weights.
        """
        if not isinstance(X, Sample):
            raise TypeError(f"Expected X to be a Sample object, got {type(X).__name__}")

        if self.model_ is None:
            raise ValueError("This BalanceWeighting instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        if self.method == "ipw":
            return self._predict_ipw(X)
        elif self.method == "null":
            return X.weight_column
        else:
            # For other methods, if they support prediction, implement here.
            # Rake and Poststratify often just return weights for the passed sample.
            # To predict on NEW data, we need the stored marginals/strata.
            # Current implementations might not expose this easily for prediction on new data
            # without refactoring.
            raise NotImplementedError(f"Prediction for method '{self.method}' on new data is not yet fully supported.")

    def transform(self, X: Sample) -> Sample:
        """
        Apply the weights to the sample X.

        Args:
            X (Sample): The sample to transform.

        Returns:
            Sample: A new Sample object with updated weights.

        Example:
            Fit a :class:`BalanceWeighting` estimator and transform a sample::

                >>> from balance.estimators import BalanceWeighting
                >>> from balance.sample_class import Sample
                >>> # ... load data ...
                >>> # sample = Sample.from_pandas(df, weight_col="weight")
                >>> bw = BalanceWeighting(method="ipw")
                >>> _ = bw.fit(sample)
                >>> adjusted = bw.transform(sample)
                >>> isinstance(adjusted, Sample)
                True
        """
        weights = self.predict(X)
        
        # Return a new sample with these weights
        from copy import deepcopy
        new_sample = deepcopy(X)
        new_sample.set_weights(weights)
        
        # Attach the model to the new sample so it looks like Sample.adjust output
        new_sample._adjustment_model = self.model_
        
        # Set links to make it look like a proper adjusted sample
        if self._target:
            new_sample._links["target"] = self._target
            
        # Set unadjusted link to the input X
        new_sample._links["unadjusted"] = X
        
        return new_sample

    def _predict_ipw(self, X: Sample) -> pd.Series:
        """
        Helper to predict weights for IPW method.
        """
        model_info = self.model_
        sklearn_model = model_info["fit"]
        
        # We need to reconstruct the model matrix for X
        # This requires applying transformations and formula
        
        # 1. Transformations
        # Note: We don't have the fitted transformations! 
        # This is a limitation. We assume 'default' or provided transformations 
        # behave deterministically or X is already transformed if needed.
        # Ideally, we should store the transformation logic.
        
        # For now, we use the same kwargs/defaults as passed to fit (via self.kwargs)
        transformations = self.kwargs.get("transformations", "default")
        
        def _contains_stateful_transformation(obj: Any) -> bool:
            """
            Detect whether the provided `transformations` spec contains any
            transformations that are known to depend on the joint
            sample+target distribution (e.g., 'quantize').
            This is a best-effort static check based on names only; it does
            not execute any user code.
            """
            # Common case: a single string name
            if isinstance(obj, str):
                return "quantize" in obj.lower() or "quantile" in obj.lower()
            
            # Collections: inspect elements/values recursively
            if isinstance(obj, dict):
                return any(
                    _contains_stateful_transformation(v) for v in obj.values()
                )
            if isinstance(obj, (list, tuple, set)):
                return any(_contains_stateful_transformation(v) for v in obj)
                
            # Check callables (e.g. functions)
            if hasattr(obj, "__name__"):
                name = obj.__name__.lower()
                if "quantize" in name or "quantile" in name:
                    return True

            return False

        # Guard against transformations that cannot be safely recomputed on
        # new data because they depend on the full sample+target used at fit
        # time (e.g., 'quantize').
        if _contains_stateful_transformation(transformations):
            msg = (
                "Prediction with stateful transformations such as 'quantize' is "
                "not supported in _predict_ipw, because the fit-time "
                "transformation state (computed on the combined sample+target) "
                "is not stored. To obtain valid predictions, either:\n"
                "  * remove or replace stateful transformations (e.g., set "
                "`transformations=None` or use only stateless transforms), or\n"
                "  * apply your own stable preprocessing pipeline to both the "
                "training and prediction data, and pass `transformations=None` "
                "to the estimator.\n"
                "This explicit error avoids silently producing incorrect "
                "weights for new data."
            )
            logger.error(msg)
            raise ValueError(msg)

        # Apply transformations
        # apply_transformations expects a tuple
        dfs = apply_transformations((X.covars().df,), transformations=transformations)
        transformed_df = dfs[0]
        
        # 2. Model Matrix
        # We need to use the same columns as the fitted model
        model_cols = model_info.get("X_matrix_columns")
        
        # We can pass the formula if we have it?
        # IPW stores formula? 'model' dict doesn't seem to have 'formula' explicitly in the return 
        # based on ipw.py code, but 'model_matrix_output' has it.
        # ipw.py logs it.
        
        # Check if we can just build matrix with variables
        variables = list(transformed_df.columns)
        
        # We need to ensure columns match. 
        # balance.util.model_matrix can help.
        
        # Warning: This re-derives the matrix. If one-hot encoding happened, 
        # we need to ensure levels match.
        # This is the "fct_lump" / categorical levels issue.
        
        # For this implementation, we attempt to build the matrix.
        mm_res = model_matrix(
            sample=transformed_df,
            variables=variables,
            add_na=self.kwargs.get("na_action", "add_indicator") == "add_indicator",
            return_type="one",
            return_var_type="matrix", # IPW usually expects dense or sparse
            formula=self.kwargs.get("formula", None),
            one_hot_encoding=self.kwargs.get("one_hot_encoding", False)
        )
        
        X_matrix = mm_res["model_matrix"]
        current_cols = mm_res["model_matrix_columns_names"]
        
        # Align columns with model_cols
        # This is crucial for sklearn prediction
        if model_cols:
            # We need to reindex/pad X_matrix to match model_cols
            # Since X_matrix is numpy/scipy, we can't just .reindex
            
            # Convert to DataFrame for alignment if not already
            if not isinstance(X_matrix, pd.DataFrame):
                X_df = pd.DataFrame(X_matrix, columns=current_cols)
            else:
                X_df = X_matrix
                
            # Add missing cols with 0
            for col in model_cols:
                if col not in X_df.columns:
                    X_df[col] = 0.0
            
            # Drop extra cols
            X_df = X_df[model_cols]
            
            # Back to array/matrix
            X_final = X_df.values
        else:
            X_final = X_matrix
            
        # 3. Predict Proba
        # Check if model supports predict_proba
        if hasattr(sklearn_model, "predict_proba"):
            # We might need to handle sample_weights for prediction? No, usually not.
            pred = sklearn_model.predict_proba(X_final)[:, 1]
        else:
             raise ValueError("Fitted model does not support predict_proba")
             
        # 4. Link Transform & Weights
        # We need sample weights to calculate initial weights
        sample_weights = X.weight_column
        
        # We need target weights sum for normalization?
        # fit() stored self._target.
        if self._target:
             target_weights = self._target.weight_column
        else:
             # If target not available, maybe we can't normalize to target sum?
             target_weights = pd.Series([1.0]) # Dummy
        
        links = link_transform(pred)
        
        # We need the lambda (best_s) ?
        # ipw function already chose the best model and returned it.
        # The 'fit' object IS the best model.
        
        # Calculate weights
        # We reuse weights_from_link
        
        balance_classes = self.kwargs.get("balance_classes", True)
        trim_ratio = model_info.get("weight_trimming_mean_ratio")
        
        weights = weights_from_link(
            link=links,
            balance_classes=balance_classes,
            sample_weights=sample_weights,
            target_weights=target_weights, # Used for odds and normalization
            weight_trimming_mean_ratio=trim_ratio
        )
        
        return weights


