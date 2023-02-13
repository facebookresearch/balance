# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

# R code to re-create the simulated data from the CBPS R package.
# The code here is kept for reference, and is not run directly from the balance package.

# install.packages("CBPS")
library("CBPS")

set.seed(123456)
n <- 500
X <- mvrnorm(n, mu = rep(0, 4), Sigma = diag(4))
prop <- 1 / (1 + exp(X[,1] - 0.5 * X[,2] +
                       0.25*X[,3] + 0.1 * X[,4]))
treat <- rbinom(n, 1, prop)
y <- 210 + 27.4*X[,1] + 13.7*X[,2] + 13.7*X[,3] + 13.7*X[,4] + rnorm(n)

##Estimate CBPS with a misspecified model
X.mis <- cbind(exp(X[,1]/2), X[,2]*(1+exp(X[,1]))^(-1)+10,
               (X[,1]*X[,3]/25+.6)^3, (X[,2]+X[,4]+20)^2)
fit1 <- CBPS(treat ~ X.mis, ATT = 1)  # we treat 1 as the "target population"
summary(fit1)
# Call:
# CBPS(formula = treat ~ X.mis, ATT = 1)

# Coefficients:
#             Estimate Std. Error z value Pr(>|z|)
# (Intercept) -4.34    1.51       -2.86   0.00419  **
# X.mis1      -1.6     0.125      -12.8   0.000    ***
# X.mis2      0.548    0.144      3.82    0.000136 ***
# X.mis3      1.99     0.0425     46.9    0.000    ***
# X.mis4      0.000537 0.163      0.0033  0.997
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# J - statistic:    0.005098105
# Log-Likelihood:  -299.7309

df <- data.frame(treat, X.mis, cbps_weights = fit1$weights, y)
head(df)
# > head(df)
#   treat        X1        X2         X3       X4 cbps_weights        y
# 1     0 1.0744852 10.320361 0.21255104 463.2808  0.005368889 227.5325
# 2     1 0.7237691  9.911956 0.18948750 383.7598  0.003937008 199.8175
# 3     0 0.6909134 10.645240 0.21737631 424.2880  0.011736818 196.8860
# 4     1 0.3470712  9.907768 0.09670587 399.3661  0.003937008 174.6853
# 5     0 0.5016829  9.594918 0.23255891 472.8546  0.009463430 191.2977
# 6     0 1.5231081 10.031016 0.32572077 438.3759  0.002758207 280.4517
with(df, boxplot(X1~treat))
with(df, boxplot(X2~treat))
with(df, boxplot(X3~treat))
with(df, boxplot(X4~treat))
with(df, boxplot(cbps_weights~treat)) # Showing that the target "treat==1" all have the same weight

with(df[df$treat == 1,], sum(cbps_weights * y) / sum(cbps_weights)) # 199.5444
with(df[df$treat == 0,], sum(cbps_weights * y) / sum(cbps_weights)) # 206.8441
tapply(df[,"treat"], df[,"treat"], mean)
#        0        1
# 220.6768 199.5444

df$id <- 1:nrow(df)

write.csv(df, "~/Downloads/simulated_cbps_data.csv", row.names = FALSE)


sessionInfo()
# R version 3.6.1 (2019-07-05)
# Platform: x86_64-apple-darwin15.6.0 (64-bit)
# Running under: macOS  10.16

# Matrix products: default
# LAPACK: /Library/Frameworks/R.framework/Versions/3.6/Resources/lib/libRlapack.dylib

# locale:
# [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8

# attached base packages:
# [1] stats     graphics  grDevices utils     datasets  methods   base

# other attached packages:
# [1] CBPS_0.23           glmnet_4.1-3        Matrix_1.2-17       numDeriv_2016.8-1.1
# [5] nnet_7.3-12         MatchIt_4.5.0       MASS_7.3-51.4

# loaded via a namespace (and not attached):
#  [1] Rcpp_1.0.8       lattice_0.20-38  codetools_0.2-16 foreach_1.4.4    grid_3.6.1
#  [6] backports_1.4.1  splines_3.6.1    iterators_1.0.13 tools_3.6.1      yaml_2.2.0
# [11] survival_3.1-11  compiler_3.6.1   shape_1.4.6
