# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Fixture inputs reproduced from survey 4.1-1 tests/kalton.R, tests/badcal.R,
# and its supplied api dataset. survey declares GPL-2 | GPL-3; its source
# notice is copyright 2002-2014 Thomas Lumley, except as stated in inst/COPYING.
# Kalton example attribution: Kalton & Flores-Cervantes (2003), Weighting methods,
# Journal of Official Statistics 19(2), 81-97 (survey tests/README).
# badcal.R credits Takahiro Tsuchiya's bug report.
# This generator calls the installed R package; it does not port its algorithms.
# Base-R model.matrix is the oracle used by recipes 0.2.0 test_interact.R.
# Fixture construction there is MIT + file LICENSE, copyright 2021 recipes authors.
# The same declared-license/notice information accompanies the exported fixtures.

args <- commandArgs(trailingOnly=TRUE)
root <- args[[1]]
out <- args[[2]]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
suppressPackageStartupMessages(library(survey))
stopifnot(as.character(packageVersion('survey')) == '4.1.1')
fmt <- function(d) {
  for (n in names(d)) if (is.double(d[[n]])) d[[n]] <- sprintf('%.17g', d[[n]])
  d
}
write17 <- function(d, name) write.csv(fmt(d), file.path(out, name), row.names=FALSE)

ab <- expand.grid(a=factor(1:4), b=factor(1:3))
ns <- c(20,50,100,30,40,140,50,100,40,310,50,70)
nt <- c(80,60,170,55,40,150,60,165,55,340,200,125)
sample <- ab[rep(1:12, ns), ]
pop <- ab[rep(1:12, nt), ]
des <- svydesign(id=~1, data=sample)
ps <- postStratify(des, ~a+b, xtabs(~a+b, pop))
rr <- rake(des, list(~a,~b), list(xtabs(~a,pop),xtabs(~b,pop)), control=list(epsilon=1e-12,maxit=10000))
ps_tot <- as.vector(svytable(~a+b,ps))
r_tot <- as.vector(svytable(~a+b,rr))
ratios <- r_tot/ns
recorded <- c(1.81,1.08,2.20,1.83,1.45,.87,1.76,1.47,2.02,1.21,2.45,2.04)
stopifnot(isTRUE(all.equal(round(ratios,2),recorded)), isTRUE(all.equal(ps_tot,nt)))
write17(data.frame(a=as.character(ab$a),b=as.character(ab$b),sample_count=ns,target_count=nt,
                  poststratify_total=ps_tot,rake_total=r_tot,rake_multiplier=ratios,
                  recorded_rake_multiplier=recorded), 'survey_kalton.csv')

bad <- data.frame(x=c(1,1,1,1,2,2,2,2,2,2),w=rep(10,10))
bd <- svydesign(ids=~1,weights=~w,data=bad)
bc <- calibrate(bd,~factor(x),c(10000,5000))
br <- calibrate(bd,~factor(x),c(10000,5000),calfun='raking',epsilon=1e-12,maxit=10000)
stopifnot(isTRUE(all.equal(svytotal(~factor(x),bc),svytotal(~factor(x),br))))
write17(data.frame(x=bad$x,design_weight=bad$w,expected_weight=as.numeric(weights(br))), 'survey_badcal.csv')

# The exact arithmetic train/test frames used by recipes' interaction tests.
tr <- data.frame(x1=1:10,x2=(1:10)+1,x3=(1:10)+2,x4=(1:10)+3,x5=(1:10)+4,
                 y=-(101:110),z=factor(rep_len(letters[1:3],10)))
te <- data.frame(x1=21:30,x2=(21:30)+1,x3=(21:30)+2,x4=(21:30)+3,x5=(21:30)+3,
                 y=-(201:210),z=factor(rep_len(letters[3:1],10)))
f <- ~x1+x2+x3+x4+x5+x1:x2+x3:x4:x5
numeric_matrix <- model.matrix(f,te)[,-1,drop=FALSE]
dummy_matrix <- model.matrix(~x1+x2+x3+x4+x5+x1*z,te)[,-1,drop=FALSE]
write17(tr,'recipes_interaction_train.csv')
write17(te,'recipes_interaction_test.csv')
write17(as.data.frame(numeric_matrix,check.names=FALSE),'recipes_interaction_numeric_expected.csv')
write17(as.data.frame(dummy_matrix,check.names=FALSE),'recipes_interaction_dummy_expected.csv')

# Actual package data and an independently implemented calibrated GREG estimator.
data(api)
vars <- c('api99','meals','ell')
stopifnot(!anyNA(apistrat[,c(vars,'api00','pw')]), !anyNA(apipop[,vars]))
original <- svydesign(id=~1,strata=~stype,weights=~pw,data=apistrat,fpc=~fpc)
N <- nrow(apipop)
base <- calibrate(original,~1,population=N)
w <- as.numeric(weights(base))
fml <- api00 ~ api99+meals+ell
fit_u <- lm(fml, data=apistrat)
fit_w <- svyglm(fml, design=base, family=gaussian())
p_s_u <- as.numeric(predict(fit_u, newdata=apistrat))
p_t_u <- as.numeric(predict(fit_u, newdata=apipop))
p_s_w <- as.numeric(predict(fit_w, newdata=apistrat, se.fit=FALSE))
p_t_w <- as.numeric(predict(fit_w, newdata=apipop, se.fit=FALSE))
xf <- ~api99+meals+ell
pop_totals <- colSums(model.matrix(xf,apipop))
# Per-unit calibration variance equal to design weight makes the GREG slope OLS.
# This produces the transport AIPW point estimate with a NONZERO augmentation.
greg <- calibrate(base,xf,population=pop_totals,variance=w)
mu_greg <- as.numeric(coef(svymean(~api00,greg)))
mu_ipw <- as.numeric(coef(svymean(~api00,base)))
mu_om <- mean(p_t_u)
augmentation <- as.numeric(coef(svymean(I(apistrat$api00-p_s_u),base)))
stopifnot(abs(mu_greg-(mu_om+augmentation))<1e-8,abs(augmentation)>1e-6)
write17(data.frame(row_id=seq_len(nrow(apistrat)),apistrat[,c(vars,'api00')],weight=w,
                  expected_pred_unweighted=p_s_u,expected_pred_weighted=p_s_w), 'survey_api_outcome_sample.csv')
write17(data.frame(row_id=seq_len(N),apipop[,vars],expected_pred_unweighted=p_t_u,
                  expected_pred_weighted=p_t_w), 'survey_api_outcome_target.csv')
write17(data.frame(mu_ipw=mu_ipw,mu_om_unweighted=mu_om,mu_om_weighted=mean(p_t_w),
                  mu_dr_greg=mu_greg,augmentation=augmentation), 'survey_api_outcome_expected.csv')

# Paired t-test on actual LaLonde outcomes and MatchIt's own matching weights.
suppressPackageStartupMessages(library(MatchIt))
data(lalonde,package='MatchIt')
matched <- matchit(treat~age+educ+race+married+nodegree+re74+re75,data=lalonde,replace=TRUE)
idx <- which(lalonde$treat==0)
y <- lalonde$re78[idx]
w0 <- rep(1,length(idx))
w1 <- matched$weights[idx]
yw0 <- y*w0/mean(w0)
yw1 <- y*w1/mean(w1)
comparison <- t.test(yw1,yw0,paired=TRUE,conf.level=.95)
write17(data.frame(row_id=idx,y=y,w0=w0,w1=w1), 'matchit_lalonde_weight_impact_inputs.csv')
write17(data.frame(mean_yw0=mean(yw0),mean_yw1=mean(yw1),mean_diff=as.numeric(comparison$estimate),
                  diff_ci_lower=comparison$conf.int[1],diff_ci_upper=comparison$conf.int[2],
                  t_stat=as.numeric(comparison$statistic),p_value=comparison$p.value,n=length(idx)),
        'matchit_lalonde_weight_impact_expected.csv')

write17(data.frame(r_version=R.version.string,survey_version=as.character(packageVersion('survey')),
                  matchit_version=as.character(packageVersion('MatchIt')),
                  kalton_md5=unname(tools::md5sum(file.path(root,'third-party/R/survey/tests/kalton.R'))),
                  api_md5=unname(tools::md5sum(file.path(root,'third-party/R/survey/data/api.rda')))),
        'reference_metadata.csv')
cat('Wrote independent package outputs to',out,'\n')
