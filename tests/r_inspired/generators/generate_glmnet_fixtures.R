# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Oracle built only from vendored glmnet 4.1-4's stored BinomialExample.
# Source metadata declares GPL-2; authors/notices preserved in accompanying
# provenance files. Fixture is upstream-generated synthetic data, not a new simulation.
args <- commandArgs(trailingOnly=TRUE)
root <- args[[1]]
out <- args[[2]]
if (length(args) >= 3) .libPaths(c(args[[3]], .libPaths()))
dir.create(out, recursive=TRUE, showWarnings=FALSE)
if (!file.exists(file.path(out, 'foldids.csv'))) {
  stop('Captured foldids.csv is required; refusing to omit CV oracle generation.')
}
options(digits=17)
write17 <- function(df,path) {
  df <- as.data.frame(df)
  df[] <- lapply(df,function(v) if(is.numeric(v)) sprintf('%.17g',v) else v)
  write.table(df,path,sep=',',row.names=FALSE,col.names=TRUE,quote=FALSE,na='NA')
}
e <- new.env(parent=emptyenv())
load(file.path(root,'data/BinomialExample.rda'),envir=e)
x <- e$BinomialExample$x
y <- e$BinomialExample$y
stopifnot(identical(dim(x),c(100L,30L)),sum(y==1)==56,sum(y==0)==44)
colnames(x) <- sprintf('x%02d',seq_len(ncol(x)))
write17(data.frame(row=seq_len(nrow(x)),y=y,x),file.path(out,'inputs.csv'))
if (!requireNamespace('glmnet',quietly=TRUE)) {
  cat('Exported exact input; glmnet unavailable.\n')
  quit(status=2)
}
cat('R:',R.version.string,' glmnet:',as.character(packageVersion('glmnet')),'\n')
stopifnot(packageVersion('glmnet')==numeric_version('4.1-4'))
for (balanced in c(FALSE,TRUE)) {
  odds <- if(balanced) sum(y==1)/sum(y==0) else 1
  mw <- ifelse(y==1,1,odds)
  mu <- colSums(x*mw)/sum(mw)
  sd <- sqrt(colSums(sweep(x,2,mu)^2*mw)/sum(mw))
  write17(data.frame(column=colnames(x),mean=mu,scale=sd),file.path(out,paste0('scaler_balanced_',balanced,'.csv')))
  for (lambda in c(1,0.1,0.01)) {
    fit <- glmnet::glmnet(x,y,family='binomial',alpha=0,lambda=lambda,
       weights=mw,standardize=TRUE,intercept=TRUE,thresh=1e-14,maxit=10000000)
    stopifnot(fit$jerr==0,length(fit$lambda)==1,fit$lambda==lambda)
    p <- as.numeric(predict(fit,newx=x,type='response'))
    link <- as.numeric(predict(fit,newx=x,type='link'))
    b <- as.numeric(coef(fit))
    # glmnet coefficients returned on original x scale; centered standardized
    # coefficients are b[-1]*sd. Intercept is not penalized.
    residual <- mw*(p-y)/sum(mw)
    grad <- c(sum(residual),colSums(sweep(x,2,sd,'/')*residual)+lambda*b[-1]*sd)
    objective <- -sum(mw*(y*log(p)+(1-y)*log1p(-p)))/sum(mw) + lambda*sum((b[-1]*sd)^2)/2
    raw <- exp(-link[y==1])/odds
    w <- rep(NA_real_,length(y))
    w[y==1] <- raw*sum(y==0)/sum(raw)
    key <- paste0('lambda_',lambda,'_balanced_',balanced)
    write17(data.frame(row=seq_len(nrow(x)),y=y,probability=p,link=link,weight=w),file.path(out,paste0(key,'.csv')))
    write17(data.frame(term=c('(Intercept)',colnames(x)),coef=b),file.path(out,paste0(key,'_coef.csv')))
    write17(data.frame(lambda=lambda,balanced=balanced,sum_model_weights=sum(mw),C=1/(sum(mw)*lambda),deviance=-2*sum(mw*(y*log(p)+(1-y)*log1p(-p)))/sum(mw),objective=objective,max_abs_gradient=max(abs(grad)),npasses=fit$npasses),file.path(out,paste0(key,'_metrics.csv')))
    foldfile <- file.path(out,'foldids.csv')
    if (file.exists(foldfile)) {
      folds <- read.csv(foldfile)
      order <- folds$row
      xs <- sweep(x,2,sd,'/')
      cv <- numeric(10)
      fold_lambdas <- numeric(10)
      for (i in 0:9) {
        train <- order[folds$fold != i]
        test <- order[folds$fold == i]
        # Balance keeps full-data C fixed in every fold, and globally scales x.
        # This is a glmnet replication of calc_dev, NOT default cv.glmnet.
        fold_lambdas[i+1] <- lambda*sum(mw)/sum(mw[train])
        fcv <- glmnet::glmnet(xs[train,,drop=FALSE],y[train],family='binomial',
          alpha=0,lambda=fold_lambdas[i+1],weights=mw[train],standardize=FALSE,
          intercept=TRUE,thresh=1e-14,maxit=10000000)
        pcv <- as.numeric(predict(fcv,newx=xs[test,,drop=FALSE],type='response'))
        cv[i+1] <- -2*sum(mw[test]*(y[test]*log(pcv)+(1-y[test])*log1p(-pcv)))/sum(mw[test])
      }
      write17(data.frame(fold=0:9,lambda=fold_lambdas,deviance=cv),file.path(out,paste0(key,'_cv_folds.csv')))
      write17(data.frame(mean=mean(cv),se=sd(cv)/sqrt(10)),file.path(out,paste0(key,'_cv_metrics.csv')))
    }
    cat(key,'C=',1/(sum(mw)*lambda),' max_abs_gradient=',max(abs(grad)), 'weight_range=',range(w,na.rm=TRUE),'\n')
  }
}
writeLines(capture.output(sessionInfo()),file.path(out,'R_sessionInfo.txt'))
