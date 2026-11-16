# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# R reference program for balance weighted diagnostics (asmd / ks / emd / cvmd /
# kld / relative_frequency_table / weighted_var) on upstream R package data.
#
# Sources (fbsource vendored copies, read-only; versions from each DESCRIPTION):
#   MatchIt 4.4.0 (GPL (>= 2))  third-party/R/MatchIt     data/lalonde.tab,
#       inst/doc/assessing-balance.{R,html} (vignette code + upstream-rendered output)
#   survey 4.1-1 (GPL-2 | GPL-3) third-party/R/survey     data/api.rda,
#       man/svycdf.Rd example (apistrat design vs apipop population ECDF)
#   Hmisc 4.7-0 (GPL (>= 2))     third-party/R/Hmisc/R/wtd.stats.s, R/Misc.s
#       (wtd.var / wtd.table functions evaluated verbatim from source; Hmisc's
#       full dependency tree is not installed)
#
# Usage:
#   R_LIBS=/tmp/balance-diag-refs-eb8be413/lib \
#     Rscript balance_diagnostics_r_reference.R <fbsource_root> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
fbsource <- if (length(args) >= 1) args[[1]] else "/data/atlas/boxes/full-fbsource"
out_dir <- if (length(args) >= 2) args[[2]] else "/tmp/balance-diag-refs-eb8be413/fixtures"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
rdir <- file.path(fbsource, "third-party", "R")

suppressPackageStartupMessages({
  library(MatchIt)
  library(survey)
})
stopifnot(packageVersion("MatchIt") == "4.4.0", packageVersion("survey") == "4.1-1")

fmt17 <- function(df) {
  for (nm in names(df)) if (is.double(df[[nm]])) df[[nm]] <- sprintf("%.17g", df[[nm]])
  df
}
write_fixture <- function(df, name) {
  write.csv(fmt17(df), file.path(out_dir, name), row.names = FALSE, quote = TRUE)
}

# ---- Hmisc functions, evaluated verbatim from the vendored source -------------
hmisc_env <- new.env()
load_named_functions <- function(path, names, env) {
  exprs <- parse(path, keep.source = FALSE)
  for (e in exprs) {
    if (is.call(e) && identical(e[[1]], as.name("<-")) &&
        as.character(e[[2]]) %in% names) eval(e, env)
  }
}
load_named_functions(file.path(rdir, "Hmisc/R/wtd.stats.s"),
                     c("wtd.mean", "wtd.var", "wtd.table"), hmisc_env)
load_named_functions(file.path(rdir, "Hmisc/R/Misc.s"),
                     c("all.is.numeric", "testDateTime"), hmisc_env)
stopifnot(all(c("wtd.var", "wtd.table", "all.is.numeric", "testDateTime") %in% ls(hmisc_env)))

# =============================================================================
# Part A. MatchIt lalonde: assessing-balance vignette (summary.matchit tables)
# =============================================================================
data("lalonde", package = "MatchIt")
lalonde_md5 <- unname(tools::md5sum(file.path(rdir, "MatchIt/data/lalonde.tab")))
stopifnot(lalonde_md5 == "0aa041b2d13d2489b60afea0103660f7", nrow(lalonde) == 614)

fml <- treat ~ age + educ + race + married + nodegree + re74 + re75
addl <- ~ I(age^2) + I(re74 == 0) + I(re75 == 0) + educ:race
m.out <- matchit(fml, data = lalonde, replace = TRUE)               # vignette chunk 1
s.out <- matchit(fml, data = lalonde, method = "subclass", subclass = 4)  # vignette chunk 5
stopifnot(identical(m.out$estimand, "ATT"), identical(s.out$estimand, "ATT"))
# ATT: every treated unit keeps weight 1, so a target-group SD computed from the
# matching weights equals MatchIt's treated SD computed without them.
stopifnot(all(m.out$weights[lalonde$treat == 1] == 1),
          all(s.out$weights[lalonde$treat == 1] == 1))

sum_nn <- summary(m.out, addlvariables = addl)
sum_sub <- summary(s.out)
sum_sub_by <- summary(s.out, subclass = TRUE, un = FALSE)

# Covariate matrix in exactly MatchIt's column coding (all factor levels kept).
X <- MatchIt:::get.covs.matrix(data = m.out$X)
Xa <- MatchIt:::get.covs.matrix(addl, data = m.out$X)
X <- cbind(X, Xa[, setdiff(colnames(Xa), colnames(X)), drop = FALSE])
stopifnot(identical(colnames(X), rownames(sum_nn$sum.all)[-1]))
safe <- function(x) gsub("[^A-Za-z0-9]+", "_", gsub("I\\((.*)\\)", "\\1", x))
lal <- data.frame(id = rownames(lalonde), treat = lalonde$treat,
                  distance = m.out$distance, check.names = FALSE)
stopifnot(isTRUE(all.equal(m.out$distance, s.out$distance)))
covs <- as.data.frame(X)
names(covs) <- safe(colnames(X))
lal <- cbind(lal, covs, w_nn_replace = m.out$weights,
             w_subclass4 = s.out$weights, subclass4 = as.integer(s.out$subclass))
write_fixture(lal, "lalonde_matchit_weights.csv")

n_treated <- sum(lalonde$treat == 1)
summary_rows <- function(tab, table_name, method) {
  vars <- rownames(tab)
  bin <- c(FALSE, apply(X, 2, function(x) all(x == 0 | x == 1)))[match(vars, c("distance", colnames(X)))]
  smd <- tab[, "Std. Mean Diff."]
  data.frame(
    method = method, table = table_name, variable = vars, column = safe(vars),
    is_binary = bin,
    means_treated = tab[, "Means Treated"], means_control = tab[, "Means Control"],
    matchit_std_mean_diff = smd,
    matchit_var_ratio = tab[, "Var. Ratio"],
    matchit_ecdf_mean = tab[, "eCDF Mean"],
    matchit_ecdf_max = tab[, "eCDF Max"],
    # balance asmd(std_type="target"): binary SD uses the n-1 (reliability) form
    # sqrt(p(1-p) n/(n-1)); MatchIt uses sqrt(p(1-p)). Exact algebraic conversion:
    expected_balance_asmd_target = abs(smd) * ifelse(bin, sqrt((n_treated - 1) / n_treated), 1),
    expected_balance_ks = tab[, "eCDF Max"],
    stringsAsFactors = FALSE, row.names = NULL)
}
lal_expected <- rbind(
  summary_rows(sum_nn$sum.all, "all", "unadjusted"),
  summary_rows(sum_nn$sum.matched, "matched", "nearest_replace"),
  summary_rows(sum_sub$sum.all, "all", "unadjusted_subclass_summary"),
  summary_rows(sum_sub$sum.across, "across_subclasses", "subclass4"))
write_fixture(lal_expected, "lalonde_matchit_expected.csv")

# Within-subclass eCDF Max (qoi.subclass -> unweighted qqsum inside each subclass).
by_rows <- do.call(rbind, lapply(seq_along(sum_sub_by$sum.subclass), function(k) {
  tab <- sum_sub_by$sum.subclass[[k]]
  data.frame(subclass = k, variable = rownames(tab), column = safe(rownames(tab)),
             matchit_ecdf_max = tab[, "eCDF Max"], expected_balance_ks = tab[, "eCDF Max"],
             row.names = NULL)
}))
write_fixture(by_rows, "lalonde_subclass_within_ks_expected.csv")

# Upstream-recorded output (inst/doc/assessing-balance.html, printed at 3 dp by
# the MatchIt maintainers' vignette build). Checks that this local run reproduces it.
recorded_nn_matched <- read.table(header = TRUE, text = "
variable smd vr ecdf_mean ecdf_max
distance 0.004 0.991 0.003 0.049
age 0.239 0.556 0.077 0.341
educ -0.016 0.576 0.022 0.059
raceblack 0.015 NA 0.005 0.005
racehispan -0.023 NA 0.005 0.005
racewhite 0.000 NA 0.000 0.000
married 0.152 NA 0.059 0.059
nodegree 0.012 NA 0.005 0.005
re74 -0.049 1.035 0.041 0.216
re75 0.009 2.126 0.068 0.238")
recorded_sub_across <- read.table(header = TRUE, text = "
variable smd vr ecdf_mean ecdf_max
distance 0.173 0.678 0.062 0.126
age 0.118 0.465 0.085 0.297
educ -0.043 0.582 0.023 0.061
raceblack 0.210 NA 0.076 0.076
racehispan 0.076 NA 0.018 0.018
racewhite -0.318 NA 0.094 0.094
married -0.017 NA 0.007 0.007
nodegree 0.113 NA 0.051 0.051
re74 -0.095 0.968 0.048 0.264
re75 0.013 1.505 0.035 0.146")
recorded_all <- read.table(header = TRUE, text = "
variable smd vr ecdf_mean ecdf_max
distance 1.794 0.921 0.377 0.644
age -0.309 0.440 0.081 0.158
educ 0.055 0.496 0.035 0.111
raceblack 1.762 NA 0.640 0.640
racehispan -0.350 NA 0.083 0.083
racewhite -1.882 NA 0.558 0.558
married -0.826 NA 0.324 0.324
nodegree 0.245 NA 0.111 0.111
re74 -0.721 0.518 0.225 0.447
re75 -0.290 0.956 0.134 0.288")
check_recorded <- function(rec, tab, label) {
  t <- tab[rec$variable, ]
  d <- c(abs(round(t[, "Std. Mean Diff."], 3) - rec$smd),
         abs(round(t[, "Var. Ratio"], 3) - rec$vr),
         abs(round(t[, "eCDF Mean"], 3) - rec$ecdf_mean),
         abs(round(t[, "eCDF Max"], 3) - rec$ecdf_max))
  stopifnot(max(d, na.rm = TRUE) < 1e-9)
  cat("recorded vignette table reproduced:", label, "\n")
}
check_recorded(recorded_all, sum_nn$sum.all, "all data")
check_recorded(recorded_nn_matched, sum_nn$sum.matched, "NN with replacement")
check_recorded(recorded_sub_across, sum_sub$sum.across, "subclass=4 across")
stopifnot(abs(round(sum_sub$nn["Matched (ESS)", "Control"], 1) - 102.3) < 1e-9,
          abs(round(sum_nn$nn["Matched (ESS)", "Control"], 2) - 43.49) < 1e-9)

# =============================================================================
# Part B. survey api: svycdf.Rd example (stratified sample vs population)
# =============================================================================
data(api)
dstrat <- svydesign(id = ~1, strata = ~stype, weights = ~pw, data = apistrat, fpc = ~fpc)
num_vars <- c("enroll", "api00", "api99", "meals", "ell")
fac_vars <- c("stype", "awards", "sch.wide", "comp.imp", "both")

write_fixture(data.frame(apistrat[, c("cds", num_vars, fac_vars, "pw")]), "api_apistrat.csv")
write_fixture(data.frame(apipop[, c("cds", num_vars, fac_vars)]), "api_apipop.csv")

ecdf_rows <- list()
for (v in num_vars) {
  # one variable per svycdf call: svycdf drops rows NA in ANY formula variable.
  F_s <- svycdf(as.formula(paste0("~", v)), dstrat)[[1]]   # weights(design, "sampling") = pw
  xpop <- apipop[[v]][!is.na(apipop[[v]])]
  F_t <- ecdf(xpop)
  xs <- apistrat[[v]][!is.na(apistrat[[v]])]
  ws <- apistrat$pw[!is.na(apistrat[[v]])]
  pts <- sort(unique(c(xs, xpop)))
  diffs <- F_s(pts) - F_t(pts)
  ks <- max(abs(diffs))
  emd <- sum(diff(pts) * abs(diffs[-length(diffs)]))   # integral of |F_s - F_t|
  # balance cvmd definition: combined pmf = (sample pmf + target pmf) / 2
  pm_s <- tapply(ws / sum(ws), factor(xs, levels = pts), sum); pm_s[is.na(pm_s)] <- 0
  pm_t <- tapply(rep(1 / length(xpop), length(xpop)), factor(xpop, levels = pts), sum); pm_t[is.na(pm_t)] <- 0
  cvmd <- sum(diffs^2 * (pm_s + pm_t) / 2)
  m_s <- unname(coef(svymean(as.formula(paste0("~", v)), dstrat, na.rm = TRUE)))
  m_t <- mean(xpop)
  sd_t <- sd(xpop)
  sd_s <- sqrt(hmisc_env$wtd.var(xs, ws, normwt = TRUE))   # reliability weights
  sd_s_cov_wt <- sqrt(cov.wt(cbind(xs), ws)$cov[1, 1])
  stopifnot(isTRUE(all.equal(sd_s, sd_s_cov_wt, tolerance = 1e-14)))
  svyvar_pt <- unname(coef(svyvar(as.formula(paste0("~", v)), dstrat, na.rm = TRUE)))[1]
  ecdf_rows[[v]] <- data.frame(
    variable = v, n_sample = length(xs), n_target = length(xpop),
    sample_weighted_mean_svymean = m_s, target_mean = m_t,
    sample_weighted_sd_reliability = sd_s, target_sd = sd_t,
    survey_svyvar_point_sd = sqrt(svyvar_pt),
    expected_asmd_target = abs(m_s - m_t) / sd_t,
    expected_asmd_sample = abs(m_s - m_t) / sd_s,
    expected_asmd_pooled = abs(m_s - m_t) / sqrt((sd_s^2 + sd_t^2) / 2),
    expected_ks = ks, expected_emd = emd, expected_cvmd_balance_definition = cvmd,
    row.names = NULL)
}
api_num <- do.call(rbind, ecdf_rows)
write_fixture(api_num, "api_numeric_expected.csv")

fac_rows <- list()
for (v in fac_vars) {
  p_svy <- prop.table(svytable(as.formula(paste0("~", v)), dstrat))
  p_mean <- coef(svymean(as.formula(paste0("~", v)), dstrat))
  wt <- hmisc_env$wtd.table(apistrat[[v]], apistrat$pw, type = "table")
  p_hmisc <- wt / sum(wt)
  q <- prop.table(table(apipop[[v]]))
  lev <- names(q)
  stopifnot(isTRUE(all.equal(as.numeric(p_svy[lev]), as.numeric(p_mean[paste0(v, lev)]), tolerance = 1e-14)),
            isTRUE(all.equal(as.numeric(p_svy[lev]), as.numeric(p_hmisc[lev]), tolerance = 1e-14)))
  p <- as.numeric(p_svy[lev]); qq <- as.numeric(q[lev])
  eps <- 1e-12
  kl_factor <- sum(pmax(p, eps) * log(pmax(p, eps) / pmax(qq, eps)))
  kl_indicator <- sapply(seq_along(lev), function(i) {
    a <- c(1 - p[i], p[i]); b <- c(1 - qq[i], qq[i])
    sum(pmax(a, eps) * log(pmax(a, eps) / pmax(b, eps)))
  })
  # ordered-category ECDF over sorted level labels (balance sorts the labels)
  ks_factor <- max(abs(cumsum(p) - cumsum(qq)))
  fac_rows[[v]] <- data.frame(
    variable = v, level = lev, sample_weighted_prop = p, target_prop = qq,
    expected_kld_factor_pmf = kl_factor, expected_kld_level_indicator = kl_indicator,
    expected_ks_level_indicator = abs(p - qq), expected_ks_factor_ordered = ks_factor,
    expected_asmd_target_level_indicator = abs(p - qq) / sqrt(qq * (1 - qq) * length(apipop[[v]]) / (length(apipop[[v]]) - 1)),
    row.names = NULL)
}
api_fac <- do.call(rbind, fac_rows)
write_fixture(api_fac, "api_factor_expected.csv")

meta <- data.frame(
  key = c("R.version", "MatchIt", "survey", "Hmisc_source", "lalonde_tab_md5",
          "api_rda_md5", "assessing_balance_html_md5"),
  value = c(R.version.string, as.character(packageVersion("MatchIt")),
            as.character(packageVersion("survey")), "4.7-0 wtd.stats.s/Misc.s (sourced)",
            lalonde_md5,
            unname(tools::md5sum(file.path(rdir, "survey/data/api.rda"))),
            unname(tools::md5sum(file.path(rdir, "MatchIt/inst/doc/assessing-balance.html")))))
write.csv(meta, file.path(out_dir, "source_metadata.csv"), row.names = FALSE)

options(digits = 10)
print(lal_expected[lal_expected$table != "all" | lal_expected$method == "unadjusted",
                   c("method", "variable", "matchit_std_mean_diff", "expected_balance_asmd_target",
                     "expected_balance_ks", "matchit_var_ratio")])
print(api_num)
print(api_fac)
cat("fixtures written to", out_dir, "\n")
