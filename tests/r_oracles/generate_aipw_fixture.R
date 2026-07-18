# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# ---------------------------------------------------------------------------
# External R oracle for balance's doubly-robust (AIPW) estimate  mu_DR.
#
# Estimand: a single target-population mean, mu = E_T[Y], from a reweighted
# responder sample. balance's one-sample AIPW / augmented estimator is
#
#     mu_DR = weighted.mean( g_hat(X_T), w_T )
#             + weighted.mean( Y - g_hat(X_S), w )        # responders, observed Y
#
# where g_hat is the outcome model fit on the responders and w are the balance
# weights. It is doubly robust: consistent if EITHER g_hat is correct OR w
# correctly reweights the responders to the target. Equivalently it is a GREG
# (model-assisted) estimator with the balance weights as the design weights.
#
# Method attribution (for copyright / credit): the augmented IPW / doubly-robust
# estimator is due to Robins, Rotnitzky & Zhao (1994, JASA 89:846-866); the
# model-assisted GREG form is Sarndal, Swensson & Wretman (1992, "Model Assisted
# Survey Sampling"). This script uses only base R (stats::lm, stats::predict,
# stats::weighted.mean), which is distributed under the GNU GPL (>= 2) by the R
# Foundation; no third-party R package code is copied here.
#
# We use NUMERIC-only covariates and fit g_hat UNWEIGHTED (matching balance's
# default fit_outcome_model), while the balance weights w are NON-UNIFORM, so the
# augmentation term is non-zero and mu_DR differs from mu_OM. base-R lm and
# balance's linear learner (one-hot + StandardScaler, prediction-invariant) agree
# to machine precision, giving an EXACT cross-language fixture.
#
# Usage:
#   Rscript generate_aipw_fixture.R
#   # then move the two CSVs into  core_stats/balance/datasets/
# ---------------------------------------------------------------------------

set.seed(2026)

gen_covars <- function(n, x1_mean, x1_sd, x2_mean) {
  data.frame(
    x1 = rnorm(n, mean = x1_mean, sd = x1_sd),
    x2 = rnorm(n, mean = x2_mean, sd = 1.0)
  )
}

n_responders <- 300L
n_target <- 200L

responders <- gen_covars(n_responders, x1_mean = 50, x1_sd = 10, x2_mean = 0.0)
responders$y <- 3 + 1.5 * responders$x1 - 2 * responders$x2 +
  rnorm(n_responders, 0, 2)
responders$w <- runif(n_responders, 0.5, 3.0) # non-uniform balance weights

target <- gen_covars(n_target, x1_mean = 55, x1_sd = 12, x2_mean = 0.3)
target$w <- runif(n_target, 0.5, 3.0)

# g_hat: UNWEIGHTED outcome model on the responders.
fit <- lm(y ~ x1 + x2, data = responders)
ghat_sample <- predict(fit, newdata = responders)
ghat_target <- predict(fit, newdata = target)

mu_ipw <- weighted.mean(responders$y, w = responders$w)
mu_om_target <- weighted.mean(ghat_target, w = target$w)
augmentation <- weighted.mean(responders$y - ghat_sample, w = responders$w)
mu_dr <- mu_om_target + augmentation

# Sanity: the algebraic identity mu_DR == mu_IPW + (mu_OM_target - mu_OM_sample,w)
mu_om_sample_w <- weighted.mean(ghat_sample, w = responders$w)
stopifnot(isTRUE(all.equal(mu_dr, mu_ipw + (mu_om_target - mu_om_sample_w))))

cat(sprintf("mu_IPW:        %.10f\n", mu_ipw))
cat(sprintf("mu_OM_target:  %.10f\n", mu_om_target))
cat(sprintf("mu_DR:         %.10f\n", mu_dr))

responders_out <- data.frame(
  is_target = 0L,
  x1 = responders$x1, x2 = responders$x2,
  y = responders$y, w = responders$w
)
target_out <- data.frame(
  is_target = 1L,
  x1 = target$x1, x2 = target$x2,
  y = NA_real_, w = target$w
)
write.csv(rbind(responders_out, target_out),
  "sim_data_aipw.csv", row.names = FALSE)

write.csv(
  data.frame(
    estimator = c("mu_IPW", "mu_OM_target", "mu_DR"),
    value = c(mu_ipw, mu_om_target, mu_dr)
  ),
  "sim_data_aipw_expected.csv", row.names = FALSE
)

cat("Wrote sim_data_aipw.csv and sim_data_aipw_expected.csv\n")
