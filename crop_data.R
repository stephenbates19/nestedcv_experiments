#takes about 2000 cpu minutes

args <- commandArgs(trailingOnly = T)
print(args)
if(length(args) >= 2) {
  n_cores <- as.integer(args[1])
  n_sim <- as.integer(args[2])
} else {
  n_cores <- 1
  n_sim <- NA
}
print(paste("Detected", n_cores, "cores and ", n_sim, "repetitions from command line arguments."))

library(glmnet)
library(nestedcv)
library(doParallel)
library(foreach)

source("data_wrapper.R")

##############################################
#problem setting
##############################################

#crops data from https://archive.ics.uci.edu/ml/datasets/Crop+mapping+using+fused+optical-radar+data+set
crop_data <- read.csv("data/WinnipegDataset.txt", header=T)
unique(crop_data[, 1])
dim(crop_data)

#extract predictors
set.seed(9)
X <- as.matrix(crop_data[, 2:175])
Y <- crop_data[, 1]
X <- X[Y %in% c(1,5), ]
Y <- Y[Y %in% c(1,5)]
Y[Y == 5] <- 0
train_idx <- sample(1:nrow(X))

#corruptions
mixed <- rnorm(length(Y)) > 1
Y[mixed] <- sample(Y[mixed], replace = F)

#simulation parameters
ns <- c(50, 100)
n_folds <- 10

##############################################

##############################################
#helpers for logstic lasso
##############################################
misclass_loss <- function(y1, y2, funcs_params = NA) {
  y1 != y2
} 

fitter_glmnet_logistic <- function(X, Y, idx = NA, funcs_params = NA) {
  if(sum(is.na(idx)) > 0) {idx <- 1:nrow(X)}
  fit <- glmnet(X[idx, ], Y[idx], family = "binomial", lambda = funcs_params$lambdas) 
  fit
}

predictor_glmnet_logistic <- function(fit, X_new, funcs_params = NA) {
  beta_hat <- fit$beta[, funcs_params$best_lam] 
  a0_hat <- fit$a0[funcs_params$best_lam]
  preds <- (X_new %*% beta_hat + a0_hat > 0)
  
  preds
} 

logistic_lasso_funs <- list(fitter = fitter_glmnet_logistic,
                            predictor = predictor_glmnet_logistic,
                            loss = misclass_loss,
                            name = "logistic_lasso")
##############################################

##############################################
#run the sims
##############################################

for(n in ns[2]) {
  print(paste0("Starting run: ", n))

  #set lasso reg level
  Xt <- X[train_idx[1:n], ]
  Yt <- Y[train_idx[1:n]]
  fit <- cv.glmnet(Xt, Yt, family="binomial", foldid = (1:n %% n_folds + 1))
  lambdas <- fit$lambda
  best_lam <- match(fit$lambda.min, lambdas)
  print(lambdas)
  print(best_lam)
  best_lam
  lambdas[best_lam]
  lambdas <- lambdas[1:best_lam]

  out <- ncv_simulator(X, Y, fun_list = list(logistic_lasso_funs), n = n, n_folds = n_folds,
                       double_cv_reps = 200, n_cores = n_cores, n_sim = n_sim, tag = "crops",
                       funcs_params = list("lambdas" = lambdas, "best_lam" = best_lam),
                       do_ncv = T, do_boot632 = T)

  save(out, file = paste0("data/crops_sparse_log_n-", n,"_632.RData"))
  print(paste0("Results saved to disk."))
}

n_ds_sims <- 2000
for(n in ns) {
  print(paste0("Starting run: ", n))
  
  #single fit to find regularization level
  #set lasso reg level
  Xt <- X[train_idx[1:n], ]
  Yt <- Y[train_idx[1:n]]
  fit <- cv.glmnet(Xt, Yt, family="binomial", foldid = (1:n %% n_folds + 1))
  lambdas <- fit$lambda
  best_lam <- match(fit$lambda.min, lambdas)
  print(lambdas)
  print(best_lam)
  best_lam
  lambdas[best_lam]
  lambdas <- lambdas[1:best_lam]
  
  set.seed(1)
  X2 <- X
  Y2 <- Y
  ds_sims <- ncv_simulator(X2, Y2, fun_list = list(logistic_lasso_funs), n = n, n_folds = n_folds, 
                           double_cv_reps = 200, n_cores = n_cores, n_sim = n_ds_sims / 10, tag = "crops",
                           funcs_params = list("lambdas" = lambdas, "best_lam" = best_lam),
                           do_ncv = F, do_cv = F, do_boot632 = F)
  save(ds_sims, file = paste0("data/crops_sparse_log_n-", n,"_ds.RData"))
  print(paste0("Results saved to disk."))
}

#check compute times
for(n in ns) {
  print(paste0("Starting run: ", n))
  
  #single fit to find regularization level
  #set lasso reg level
  Xt <- X[train_idx[1:n], ]
  Yt <- Y[train_idx[1:n]]
  fit <- cv.glmnet(Xt, Yt, family="binomial", foldid = (1:n %% n_folds + 1))
  lambdas <- fit$lambda
  best_lam <- match(fit$lambda.min, lambdas)
  print(lambdas)
  print(best_lam)
  best_lam
  lambdas[best_lam]
  lambdas <- lambdas[1:best_lam]
  
  set.seed(1)
  ds_sims <- ncv_simulator(X, Y, fun_list = list(logistic_lasso_funs), n = n, n_folds = n_folds, 
                           double_cv_reps = 200, n_cores = n_cores, n_sim = 1, tag = "crops",
                           funcs_params = list("lambdas" = lambdas, "best_lam" = best_lam),
                           do_ncv = T, do_cv = T, do_boot632 = F)
}

##############################################



##############################################
#look at results
##############################################

quit() # don't execute the following code for batch jobs

print("Entering result analysis")
library(ggplot2)

#
n <- 100
load(file = paste0("data/crops_sparse_log_n-", n,"_632.RData"))}
load(file = paste0("data/crops_sparse_log_n-", n,"_ds.RData"))

alpha <- .1
qv <- qnorm(1-alpha/2)
vst <- function(x) {asin(sqrt(max(0, x)))}

out$parameters
length(out$ncv_results)

mean_err <- mean(sapply(out$cv_results, function(x) {x$ho_err}))
mean_err

#width
mean(sapply(out$ncv_results, function(x) {x$ci_hi - x$ci_lo}), na.rm = T) / mean(sapply(out$cv_results, function(x) {x$ci_hi - x$ci_lo}))
mean(sapply(ds_sims$ds_results, function(x) {x$se_hat * 2 * qv})) / mean(sapply(out$cv_results, function(x) {x$ci_hi - x$ci_lo}))

#point estimates
mean_err
mean(sapply(out$cv_results, function(x) {x$err_hat}))
mean(sapply(out$ncv_results, function(x) {x$err_hat}))
mean(sapply(ds_sims$ds_results, function(x) {x$err_hat}))

#cv coverage
mean(sapply(out$cv_results, function(x) {vst(x$ho_err) < vst(x$err_hat) - qv * sqrt(1/(4*n))}))
mean(sapply(out$cv_results, function(x) {vst(x$ho_err) > vst(x$err_hat) + qv * sqrt(1/(4*n))}))

mean(sapply(out$cv_results, function(x) {vst(mean_err) < vst(x$err_hat) - qv * sqrt(1/(4*n))}))
mean(sapply(out$cv_results, function(x) {vst(mean_err) > vst(x$err_hat) + qv * sqrt(1/(4*n))}))

#ncv coverage
mean(sapply(out$ncv_results, function(x) {vst(x$ho_err) < vst(x$err_hat) - qv * x$sd_infl * sqrt(1/(4*n))}), na.rm = T)
mean(sapply(out$ncv_results, function(x) {vst(x$ho_err) > vst(x$err_hat) + qv * x$sd_infl * sqrt(1/(4*n))}), na.rm = T)

mean(sapply(out$ncv_results, function(x) {vst(mean_err) < vst(x$err_hat) - qv * x$sd_infl * sqrt(1/(4*n))}), na.rm = T)
mean(sapply(out$ncv_results, function(x) {vst(mean_err) > vst(x$err_hat) + qv * x$sd_infl * sqrt(1/(4*n))}), na.rm = T)

#data splitting with refitting
mean(sapply(ds_sims$ds_results, function(x) {vst(x$ho_err) < vst(x$err_hat) - qv * sqrt(5/(4*n))}))
mean(sapply(ds_sims$ds_results, function(x) {vst(x$ho_err) > vst(x$err_hat) + qv * sqrt(5/(4*n))}))

mean(sapply(ds_sims$ds_results, function(x) {vst(mean_err) < vst(x$err_hat) - qv * sqrt(5/(4*n))}))
mean(sapply(ds_sims$ds_results, function(x) {vst(mean_err) > vst(x$err_hat) + qv * sqrt(5/(4*n))}))


## Boot results

#boot width
mean(pmin(1,unlist(sapply(out$boot632_results, function(x){x$ci_hi - x$ci_lo}))), na.rm = T) / mean(sapply(out$cv_results, function(x){x$ci_hi - x$ci_lo}))
mean(pmin(1,unlist(sapply(out$boot632_results, function(x){(x$ci_hi - x$ci_lo) * x$raw_mean / x$err_hat}))), na.rm = T) / mean(sapply(out$cv_results, function(x){x$ci_hi - x$ci_lo}))

#boot means
mean(unlist(sapply(out$boot632_results, function(x){x$err_hat})))
mean(unlist(sapply(out$boot632_results, function(x){x$raw_mean})))

#boot632 coverage
mean(unlist(sapply(out$boot632_results, function(x) {x$ho_err < x$ci_lo})), na.rm = T)
mean(unlist(sapply(out$boot632_results, function(x) {x$ho_err > x$ci_hi})), na.rm = T)

mean(unlist(sapply(out$boot632_results, function(x) {mean_err < x$ci_lo})), na.rm = T)
mean(unlist(sapply(out$boot632_results, function(x) {mean_err > x$ci_hi})), na.rm = T)

#boot OOB coverage
mean(unlist(sapply(out$boot632_results, function(x) {x$ho_err < x$raw_mean - qv * x$se_est2 * x$raw_mean / x$err_hat})), na.rm = T)
mean(unlist(sapply(out$boot632_results, function(x) {x$ho_err > x$raw_mean + qv * x$se_est2 * x$raw_mean / x$err_hat})), na.rm = T)

mean(unlist(sapply(out$boot632_results, function(x) {mean_err < x$raw_mean - qv * x$se_est2 * x$raw_mean / x$err_hat})), na.rm = T)
mean(unlist(sapply(out$boot632_results, function(x) {mean_err > x$raw_mean + qv * x$se_est2 * x$raw_mean / x$err_hat})), na.rm = T)

