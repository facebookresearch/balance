# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Exact fixture generation from matrixStats 0.62.0 tests/rowWeightedMeans.R.
# Source package declares Artistic-2.0; author/holder Henrik Bengtsson.
# stats::cov.wt is used as a matching reliability-variance oracle, not
# matrixStats::weightedVar's different frequency-weight convention.
args <- commandArgs(trailingOnly=TRUE)
root <- args[[1]]; out <- args[[2]]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
suppressPackageStartupMessages(library(matrixStats))
stopifnot(as.character(packageVersion('matrixStats'))=='0.62.0')
fmt <- function(d) {for(n in names(d))if(is.double(d[[n]]))d[[n]]<-sprintf('%.17g',d[[n]]);d}
write17 <- function(d,n)write.csv(fmt(d),file.path(out,n),row.names=FALSE)
# Execute all original assertions before capturing the same fixture inputs.
source(file.path(root,'third-party/R/matrixStats/tests/rowWeightedMeans.R'),local=new.env())
set.seed(1)
x <- matrix(rnorm(20),nrow=5,ncol=4)
w <- 1:4
m <- rowWeightedMeans(x,w=w,useNames=FALSE)
stopifnot(isTRUE(all.equal(m,apply(x,1,weighted.mean,w=w))))
zero <- rowWeightedMeans(x,w=c(1,1,0,1),useNames=FALSE)
xt <- t(x); colnames(xt)<-letters[1:5]
write17(as.data.frame(xt),'matrixstats_values.csv')
x[sample(length(x),size=.3*length(x))]<-NA
missing <- rowWeightedMeans(x,w=w,na.rm=TRUE,useNames=FALSE)
xt<-t(x);colnames(xt)<-letters[1:5]
write17(as.data.frame(xt),'matrixstats_missing_values.csv')
write17(data.frame(column=letters[1:5],weighted_mean=m,zero_weight_mean=zero,missing_mean=missing),'matrixstats_expected.csv')
# Exact frequency fixture from matrixStats weightedVar.R:93-104.
y<-c(2,2,4,5,5,5);v<-unique(y);ww<-as.numeric(table(y))
frequency<-weightedVar(v,w=ww)
stopifnot(isTRUE(all.equal(frequency,var(y))))
reliability<-cov.wt(cbind(v),wt=ww,method='unbiased')
write17(data.frame(value=v,weight=ww),'matrixstats_variance_inputs.csv')
write17(data.frame(mean=reliability$center,variance=reliability$cov[1,1],frequency_variance=frequency),'matrixstats_variance_expected.csv')
# Exact hardhat metadata-only novel-level fixture; stats model.matrix provides
# independently executed numeric expectations for its four fitted indicators.
tr<-data.frame(y=1:4,f=factor(letters[1:4]))
ne<-data.frame(y=1:5,f=factor(letters[1:5]));ne<-ne[1:4,]
mm<-model.matrix(~f+0,ne,xlev=list(f=levels(tr$f)))
stopifnot(identical(colnames(mm),c('fa','fb','fc','fd')))
write17(as.data.frame(mm),'hardhat_factor_expected.csv')
write17(data.frame(y=tr$y,f=as.character(tr$f)),'hardhat_factor_train.csv')
write17(data.frame(y=ne$y,f=as.character(ne$f)),'hardhat_factor_test.csv')
write17(data.frame(r_version=R.version.string,matrixStats=as.character(packageVersion('matrixStats')),
                  source_md5=unname(tools::md5sum(file.path(root,'third-party/R/matrixStats/tests/rowWeightedMeans.R')))),
        'basic_reference_metadata.csv')
cat('Captured exact matrixStats fixtures and stats/hardhat matrix references\n')
