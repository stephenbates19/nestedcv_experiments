args <- commandArgs(trailingOnly = T)
print(args)
if(length(args) >= 2) {
  n_cores <- as.integer(args[1])
  n_sim <- as.integer(args[2])
} else {
  n_cores <- 1
  n_sim <- 10
}
print(paste("Detected", n_cores, "cores and ", n_sim, "repetitions from command line arguments."))


library(glmnet)
library(nestedcv)
library(parallel)
library(doParallel)
library(foreach)
library(doRNG)
source("data_wrapper.R")

################################################
#### Harrel setup (hihg-dimensional logistic regression)
################################################

### Problem setting
n <- 90
p <- 1000
k <- 4 #number of nonzeros
alpha <- .1 #nominal error rate, total across both tails.

# create the design matrix 
set.seed(1)
X <- matrix(rnorm(n = n * p), nrow = n)

#sample Y from a logistic model
strength <- 1 #signal strength
beta = strength * c(rep(1, k), rep(0, p - k))

#determine bayes error with this beta vector
set.seed(555)
n_holdout <- 10000
X_holdout <- matrix(rnorm(n_holdout * p), nrow = n_holdout)
p_holdout <- 1 / (1 + exp(-X_holdout %*% beta))
Y_holdout <- (runif(n_holdout) < p_holdout) * 1.0
bayes_error <- (rowSums(X_holdout[, 1:k]) < 0) * (Y_holdout == 1) + (rowSums(X_holdout[, 1:k]) > 0) * (Y_holdout == 0)

# Bayes error rate
error_rate <- mean(bayes_error)
print(error_rate)
################################################


##############################################
#helpers for logistic lasso
##############################################
misclass_loss <- function(y1, y2, funcs_params = NA) {
  y1 != y2
} 

fitter_glmnet_logistic <- function(X, Y, idx = NA, funcs_params = NA) {
  if(sum(is.na(idx)) > 0) {idx <- 1:nrow(X)}
  fit <- glmnet(X[idx, ], Y[idx], family = "binomial", lambda = funcs_params$lambdas) #assumes lambda is in global env
  
  fit
}

predictor_glmnet_logistic <- function(fit, X_new, funcs_params = NA) {
  beta_hat <- fit$beta[, funcs_params$best_lam] #assumes best_lam is in global env
  a0_hat <- fit$a0[funcs_params$best_lam]
  preds <- (X_new %*% beta_hat + a0_hat > 0)
  
  preds
} 

logistic_lasso_funs <- list(fitter = fitter_glmnet_logistic,
                            predictor = predictor_glmnet_logistic,
                            loss = misclass_loss,
                            name = "logistic_lasso")

##############################################


if(n_cores > 1) {
  cl <- makeForkCluster(n_cores)
  registerDoParallel(cl)
}

print("Starting sims")
print(Sys.time())

#sim settings
dcv_reps <- 200
n_folds <- 10
ns <- c(90)
n_sim <- 1000

#Fit one model to find a good lambda. This lambda will be fixed in future simulations.
fit <- cv.glmnet(X_holdout[1:n, ], Y_holdout[1:n], family = "binomial", foldid = (1:n %% n_folds + 1))
lambdas <- fit$lambda
best_lam <- match(fit$lambda.min, lambdas)
best_lam
print(paste("best lambda: ", lambdas[best_lam]))

#different regularization levels
lindices = c(min(length(lambdas), best_lam + 30), max(1, best_lam - 5), best_lam)

for(lindex in 1:3) {
  print(paste0("Starting run: ", lindex))
  print(lambdas[lindices[lindex]])
  
  #run the simulation
  set.seed(1)
  out <- ncv_simulator(X_holdout, Y_holdout, fun_list = list(logistic_lasso_funs), n = n, n_folds = n_folds,
                       double_cv_reps = dcv_reps, nboot = 1000, n_cores = n_cores, n_sim = n_sim, tag = "sparse_logistic",
                       funcs_params = list("lambdas" = lambdas, "best_lam" = lindices[lindex]), 
                       do_ncv = T, do_boot632 = T)
  save(out, file = paste0("data/sparse_log_n-", n, "_lambda-", lindex, "_632full.RData"))
  print(paste0("Results saved to disk."))
}

print("exiting")
quit()

##################################
##### Analysis
##############################
n <- 90
load(file = paste0("data/sparse_log_n-", n,"_632full.RData"))
out$boot632_results[[1]]
out$parameters
length(out$boot632_results)

alpha <- .1
q_val <- qnorm(1 - alpha/2)
vst <- function(x) {asin(sqrt(x))}

mean_err <- mean(sapply(out$cv_results, function(x){x$ho_err}))
mean_err

#width
mean(sapply(out$ncv_results, function(x){x$ci_hi - x$ci_lo})) / mean(sapply(out$cv_results, function(x){x$ci_hi - x$ci_lo}))
mean(sapply(out$boot632_results, function(x){x$ci_hi - x$ci_lo})) / mean(sapply(out$cv_results, function(x){x$ci_hi - x$ci_lo}))
mean(sapply(out$boot632_results, function(x){(x$ci_hi - x$ci_lo) * x$raw_mean / x$err_hat})) / mean(sapply(out$cv_results, function(x){x$ci_hi - x$ci_lo}))

#point estimates
mean(sapply(out$cv_results, function(x){x$err_hat}))
mean(sapply(out$ncv_results, function(x){x$err_hat}))
mean(sapply(out$boot632_results, function(x){x$err_hat}))
mean(sapply(out$boot632_results, function(x){x$raw_mean}))

#cv coverage
mean(sapply(out$cv_results, function(x) {vst(x$ho_err) < vst(x$err_hat) - qv * sqrt(1/(4*n))}))
mean(sapply(out$cv_results, function(x) {vst(x$ho_err) > vst(x$err_hat) + qv * sqrt(1/(4*n))}))

mean(sapply(out$cv_results, function(x) {vst(mean_err) < vst(x$err_hat) - qv * sqrt(1/(4*n))}))
mean(sapply(out$cv_results, function(x) {vst(mean_err) > vst(x$err_hat) + qv * sqrt(1/(4*n))}))

#ncv coverage
mean(sapply(out$ncv_results, function(x) {vst(x$ho_err) < vst(x$err_hat) - qv * x$sd_infl * sqrt(1/(4*n))}))
mean(sapply(out$ncv_results, function(x) {vst(x$ho_err) > vst(x$err_hat) + qv * x$sd_infl * sqrt(1/(4*n))}))

mean(sapply(out$ncv_results, function(x) {vst(mean_err) < vst(x$err_hat) - qv * x$sd_infl * sqrt(1/(4*n))}))
mean(sapply(out$ncv_results, function(x) {vst(mean_err) > vst(x$err_hat) + qv * x$sd_infl * sqrt(1/(4*n))}))

#boot632 coverage
mean(sapply(out$boot632_results, function(x) {x$ho_err < x$ci_lo}))
mean(sapply(out$boot632_results, function(x) {x$ho_err > x$ci_hi}))

mean(sapply(out$boot632_results, function(x) {mean_err < x$ci_lo}))
mean(sapply(out$boot632_results, function(x) {mean_err > x$ci_hi}))

#boot OOB coverage
mean(sapply(out$boot632_results, function(x) {x$ho_err < x$raw_mean - qv * x$se_est2 * x$raw_mean / x$err_hat}))
mean(sapply(out$boot632_results, function(x) {x$ho_err > x$raw_mean + qv * x$se_est2 * x$raw_mean / x$err_hat}))

mean(sapply(out$boot632_results, function(x) {mean_err < x$raw_mean - qv * x$se_est2 * x$raw_mean / x$err_hat}))
mean(sapply(out$boot632_results, function(x) {mean_err > x$raw_mean + qv * x$se_est2 * x$raw_mean / x$err_hat}))

