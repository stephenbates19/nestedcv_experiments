#takes about 1000 cpu minutes

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
# problem setting
##############################################

#problem settings
ns <- c(50, 100)
p <- 500
n_sig <- 4 #number of nonzeros
alpha <- .1 #nominal error rate, total across both tails.

# create the design matrix 
get_X <- function(n, p, rho = 0) {
  X <- sqrt(1 - rho^2) * matrix(rnorm(n = n * p), nrow = n)
  for(i in 1:p) {
    X[, i] <- X[, i] + rho * rnorm(n)
  }
  
  X
} 
set.seed(1)

#sample Y from a sparse linear model
strength <- 1 #signal strength
beta = strength * c(rep(1, n_sig), rep(0, p - n_sig))
get_Y <- function(X, beta) {
  X %*% beta + rnorm(nrow(X))
}

#determine bayes error with this beta vector
set.seed(555)
n_holdout <- 20000
X_holdout <- get_X(n_holdout, p)
Y_holdout <- get_Y(X_holdout, beta)
bayes_error <- 1 #by construction

# Bayes error rate
error_rate <- mean(bayes_error)
print(error_rate)
snr <- var(X_holdout %*% beta)
print(snr)

##############################################
#helpers for linear lasso
##############################################
se_loss <- function(y1, y2, funcs_params = NA) {
  (y1 - y2)^2
} 

fitter_glmnet_lin <- function(X, Y, idx = NA, funcs_params = NA) {
  if(sum(is.na(idx)) > 0) {idx <- 1:nrow(X)}
  fit <- glmnet(X[idx, ], Y[idx], lambda = funcs_params$lambdas) 
  
  fit
}

predictor_glmnet_lin <- function(fit, X_new, funcs_params = NA) {
  beta_hat <- fit$beta[, funcs_params$best_lam] 
  a0_hat <- fit$a0[funcs_params$best_lam]
  preds <- X_new %*% beta_hat + a0_hat
  
  preds
} 

lasso_funs <- list(fitter = fitter_glmnet_lin,
                   predictor = predictor_glmnet_lin,
                   loss = se_loss,
                   name = "lasso")

get_test_err <- function(fit, funcs, funcs_params = NA) {
  preds <- funcs$predictor(fit, X_holdout, funcs_params = funcs_params)
  mean(funcs$loss(preds, Y_holdout, funcs_params = funcs_params))
}
##############################################


##############################################
#run the simulation
##############################################

#simulation setting
n_folds <- 10
dcv_reps <- 200
n_sim <- 1000

# if(n_cores > 1) {
#   print("Starting cluster")
#   cl <- makeForkCluster(nnodes = n_cores)
#   registerDoParallel(cl, cores = n_cores)
#   clusterSetRNGStream(cl = cl, 123)
# }

for(n in ns) {
  print(paste0("Starting run: ", n))
  print(Sys.time())
  
  #Fit one model to find a good lambda. This lambda will be fixed in future simulations.
  fit <- cv.glmnet(X_holdout[1:n, ], Y_holdout[1:n], foldid = (1:n %% n_folds + 1))
  lambdas <- fit$lambda
  best_lam <- match(fit$lambda.min, lambdas)
  best_lam
  lambdas[best_lam]
  lambdas <- lambdas[1:best_lam]
  
  #run the simulation
  out <- ncv_simulator(X_holdout, Y_holdout, fun_list = list(lasso_funs), n = n, n_folds = n_folds, 
                       double_cv_reps = dcv_reps, n_cores = n_cores, n_sim = n_sim, tag = "sparse_linear",
                       funcs_params = list("lambdas" = lambdas, "best_lam" = best_lam),
                       trans = list(identity, sqrt, log))
  save(out, file = paste0("data/sparse_lin_n-", n,".RData"))
  print(paste0("Results saved to disk."))
}

n_ds_sims <- 2000
for(n in ns) {
  #data splitting sims (fast, about 5 minutes)
  fit <- cv.glmnet(X_holdout[1:n, ], Y_holdout[1:n], foldid = (1:n %% n_folds + 1))
  lambdas <- fit$lambda
  best_lam <- match(fit$lambda.min, lambdas)
  best_lam
  lambdas[best_lam]
  lambdas <- lambdas[1:best_lam]
  
  set.seed(1)
  ds_sims <- ncv_simulator(X_holdout, Y_holdout, fun_list = list(lasso_funs), n = n, n_folds = 10, 
                           n_cores = n_cores, n_sim = n_ds_sims / 10, tag = "sparse_linear",
                           funcs_params = list("lambdas" = lambdas, "best_lam" = best_lam), 
                           do_cv = F, do_ncv = F, do_boot632 = F)
  save(ds_sims, file = paste0("data/sparse_lin_n-", n,"_ds.RData"))
  print(paste0("Results saved to disk."))
}


#check compute times
for(n in ns) {
  fit <- cv.glmnet(X_holdout[1:n, ], Y_holdout[1:n], foldid = (1:n %% n_folds + 1))
  lambdas <- fit$lambda
  best_lam <- match(fit$lambda.min, lambdas)
  best_lam
  lambdas[best_lam]
  lambdas <- lambdas[1:best_lam]
  
  set.seed(1)
  ds_sims <- ncv_simulator(X_holdout, Y_holdout, fun_list = list(lasso_funs), n = n, n_folds = 10, 
                           n_cores = n_cores, n_sim = 1, tag = "sparse_linear",
                           funcs_params = list("lambdas" = lambdas, "best_lam" = best_lam), 
                           do_cv = T, do_ncv = T, do_boot632 = F)
}

# if(n_cores > 1) {
#   stopCluster(cl)
# }
##############################################



##############################################
#look at results
##############################################
quit() # don't execute the following code for batch jobs

print("Entering result analysis")
library(ggplot2)

n <- 100
load(file = paste0("data/sparse_lin_n-", n,".RData"))
load(file = paste0("data/sparse_lin_n-", n,"_ds.RData"))

alpha <- .1
qv <- qnorm(1-alpha/2)

out$parameters
length(out$ncv_results)

mean_err <- mean(sapply(out$cv_results, function(x) {x$ho_err}))
mean_err

#width
mean(sapply(out$ncv_results, function(x) {x$ci_hi - x$ci_lo})) / mean(sapply(out$cv_results, function(x) {x$ci_hi - x$ci_lo}))
mean(sapply(ds_sims$ds_results, function(x) {x$se_hat * 2 * qv})) / mean(sapply(out$cv_results, function(x) {x$ci_hi - x$ci_lo}))

#point estimates
mean_err
mean(sapply(out$cv_results, function(x) {x$err_hat}))
mean(sapply(out$ncv_results, function(x) {x$err_hat}))
mean(sapply(ds_sims$ds_results, function(x) {x$err_hat}))

#cv coverage
mean(sapply(out$cv_results, function(x){x$ho_err < x$ci_lo}))
mean(sapply(out$cv_results, function(x){x$ho_err > x$ci_hi}))

mean(sapply(out$cv_results, function(x){mean_err < x$ci_lo}))
mean(sapply(out$cv_results, function(x){mean_err > x$ci_hi}))

#ncv coverage
mean(sapply(out$ncv_results, function(x){x$ho_err < x$err_hat + x$bias_est - qv * x$sd * x$sd_infl / sqrt(n)}))
mean(sapply(out$ncv_results, function(x){x$ho_err > x$err_hat + x$bias_est + qv * x$sd * x$sd_infl / sqrt(n)}))

mean(sapply(out$ncv_results, function(x){mean_err < x$err_hat + x$bias_est - qv * x$sd * x$sd_infl / sqrt(n)}))
mean(sapply(out$ncv_results, function(x){mean_err > x$err_hat + x$bias_est + qv * x$sd * x$sd_infl / sqrt(n)}))

#data splitting with refitting
mean(sapply(ds_sims$ds_results, function(x){x$ho_err < x$err_hat - qv*x$se_hat}))
mean(sapply(ds_sims$ds_results, function(x){x$ho_err > x$err_hat + qv*x$se_hat}))

mean_err2 <- mean(sapply(ds_sims$ds_results, function(x){x$ho_err}))
mean(sapply(ds_sims$ds_results, function(x){mean_err2 < x$err_hat - qv*x$se_hat}))
mean(sapply(ds_sims$ds_results, function(x){mean_err2 > x$err_hat + qv*x$se_hat}))

#plot histograms
for(n in ns) {
  load(file = paste0("data/sparse_lin_n-", n,".RData"))
  
  my_hist <- qplot(sapply(out$ncv_results, function(x){x$sd_infl}), bins = 10) +
    labs(x = "NCV width") +
    theme_bw() +
    theme(aspect.ratio = 1)
  
  ggsave(my_hist, file = paste0("figures/highd_linear_inf_hist_", n, ".pdf"), height = 2, width = 2.5)
}



