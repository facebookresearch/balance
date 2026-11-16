# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# ---------------------------------------------------------------------------
# External R oracle for balance's outcome-model estimate  mu_OM.
#
# balance's outcome model implements the g-computation / regression-adjustment
# TRANSPORT estimator of a single target-population mean:
#
#     mu_OM = weighted_mean_over_target( g_hat(X_T) ),   g_hat(X) = E[Y|X] on responders
#
# This is NOT the GREG (model-assisted) estimator and NOT an ATE. So the correct
# R oracle is plain regression standardization over the target, i.e.
#     lm() -> predict(newdata = target) -> weighted.mean(..., w = target)
# and the library equivalent marginaleffects::avg_predictions(fit, newdata=target,
# wts=w). (mase::greg / survey::calibrate would add a design residual-correction
# term and therefore would NOT match mu_OM.)
#
# NOTE (deliberately different from the ATE/DR pipeline): we do NOT Poisson-sample
# and then force-pad/truncate to a fixed row count. Tampering with the realized
# sample severs it from its inclusion probabilities and makes an "exact" fixture
# meaningless. We just generate two clean, fixed-size frames.
#
# We use NUMERIC-only covariates so that R's lm predictions and balance's linear
# learner (one-hot + StandardScaler, which is prediction-invariant) agree to
# machine precision -- giving an EXACT cross-language fixture, not an approximate
# correlation check.
#
# Usage:
#   Rscript generate_outcome_model_fixture.R
#   # then move the two CSVs into  core_stats/balance/datasets/
# ---------------------------------------------------------------------------

set.seed(2026)

gen_covars <- function(n, x1_mean, x1_sd, x2_mean) {
  data.frame(
    x1 = rnorm(n, mean = x1_mean, sd = x1_sd),
    x2 = rnorm(n, mean = x2_mean, sd = 1.0),
    x3 = runif(n, min = -2.0, max = 2.0)
  )
}

n_responders <- 400L
n_target <- 250L

# Responders: observed covariates + outcome + a fit weight.
responders <- gen_covars(n_responders, x1_mean = 50, x1_sd = 10, x2_mean = 0.0)
responders$y <- 3 + 1.5 * responders$x1 - 2 * responders$x2 +
  0.7 * responders$x3 + rnorm(n_responders, 0, 2)
responders$w <- runif(n_responders, 0.5, 2.0)

# Target: covariates + averaging weight only (outcome is unobserved).
target <- gen_covars(n_target, x1_mean = 55, x1_sd = 12, x2_mean = 0.3)
target$w <- runif(n_target, 0.5, 2.0)

# --- mu_OM, unweighted outcome fit (OLS) ------------------------------------
fit_unweighted <- lm(y ~ x1 + x2 + x3, data = responders)
preds_unweighted <- predict(fit_unweighted, newdata = target)
mu_unweighted <- weighted.mean(preds_unweighted, w = target$w)

# --- mu_OM, weighted outcome fit (WLS) --------------------------------------
fit_weighted <- lm(y ~ x1 + x2 + x3, data = responders, weights = responders$w)
preds_weighted <- predict(fit_weighted, newdata = target)
mu_weighted <- weighted.mean(preds_weighted, w = target$w)

# --- Library cross-check: marginaleffects::avg_predictions ------------------
# Same estimand, independent implementation; must agree with the hand oracle.
if (requireNamespace("marginaleffects", quietly = TRUE)) {
  mu_u_me <- marginaleffects::avg_predictions(
    fit_unweighted, newdata = target, wts = target$w
  )$estimate
  stopifnot(isTRUE(all.equal(mu_unweighted, mu_u_me)))
  cat(sprintf("marginaleffects agrees (unweighted): %.10f\n", mu_u_me))
} else {
  cat("marginaleffects not installed; skipping library cross-check.\n")
}

cat(sprintf("mu_OM (unweighted fit): %.10f\n", mu_unweighted))
cat(sprintf("mu_OM (weighted fit):   %.10f\n", mu_weighted))

# --- Export the fixture -----------------------------------------------------
# One stacked CSV (is_target flag; y is NA for target rows), plus a tiny CSV of
# the expected scalars. Mirrors the sim_data_cbps.csv precedent.
responders_out <- data.frame(
  is_target = 0L,
  x1 = responders$x1, x2 = responders$x2, x3 = responders$x3,
  y = responders$y, w = responders$w
)
target_out <- data.frame(
  is_target = 1L,
  x1 = target$x1, x2 = target$x2, x3 = target$x3,
  y = NA_real_, w = target$w
)
write.csv(rbind(responders_out, target_out),
  "sim_data_outcome_model.csv", row.names = FALSE)

write.csv(
  data.frame(
    estimator = c("mu_OM_unweighted_fit", "mu_OM_weighted_fit"),
    mu_OM = c(mu_unweighted, mu_weighted)
  ),
  "sim_data_outcome_model_expected.csv", row.names = FALSE
)

cat("Wrote sim_data_outcome_model.csv and sim_data_outcome_model_expected.csv\n")
