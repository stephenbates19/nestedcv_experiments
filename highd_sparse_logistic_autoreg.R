args <- commandArgs(trailingOnly = T)
print(args)
if(length(args) >= 2) {
  rho <- .5 #default value
  n_cores <- as.integer(args[1])
  n_sim <- as.integer(args[2])
} else {
  rho <- .5
  n_cores <- 1
  n_sim <- NA
} 
if(length(args) >= 3) {
  rho <- as.numeric(args[3])
}
print(paste("Detected", n_cores, "cores and ", n_sim, "repetitions from command line arguments."))
print(paste0("rho: ", rho))

library(glmnet)
library(nestedcv)
library(doParallel)
library(foreach)

source("data_wrapper.R")


##############################################
#problem setting for logistic lasso
##############################################

#sim settings
#sim settings
dcv_reps <- 200
n_folds <- 10
ns <- c(90)


n <- 90
p <- 1000
k <- 4 #number of nonzeros
alpha <- .1 #nominal error rate, total across both tails.

#sample Y from a logistic model
strength <- 2 #signal strength
set.seed(1)
beta = sample(strength * c(rep(1, k), rep(0, p - k)))

#determine bayes error with this beta vector
set.seed(555)
n_holdout <- 10000
X_holdout <- matrix(rnorm(n_holdout * p), nrow = n_holdout)

#autoregressive design matrix
for(j in 2:p) {
  X_holdout[, j] <- rho * X_holdout[, j - 1] + sqrt(1-rho^2) * X_holdout[, j]
}
p_holdout <- 1 / (1 + exp(-X_holdout %*% beta))
Y_holdout <- (runif(n_holdout) < p_holdout) * 1.0
bayes_error <- (rowSums(X_holdout[, beta != 0]) < 0) * (Y_holdout == 1) + (rowSums(X_holdout[, beta != 0]) > 0) * (Y_holdout == 0)

# Bayes error rate
error_rate <- mean(bayes_error)
print(error_rate)

##############################################
#helpers for logistic lasso
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
# Run the sim
##############################################

if(n_cores > 1) {
  print("Starting cluster")
  if(exists("cl")) {
    print("Warning, object cl exits! Printing")
    print(cl)
  }
  cl <- makeForkCluster(nnodes = n_cores)
  registerDoParallel(cl, cores = n_cores)
  clusterSetRNGStream(cl = cl, 123)
}

for(n in ns) {
  print(paste0("Starting run: ", n))
  
  #Fit one model to find a good lambda. This lambda will be fixed in future simulations.
  fit <- cv.glmnet(X_holdout[1:n, ], Y_holdout[1:n], family = "binomial")
  lambdas <- fit$lambda
  best_lam <- match(fit$lambda.min, lambdas)
  best_lam
  lambdas[best_lam]
  lambdas <- lambdas[1:best_lam]
  
  #run the simulation
  out <- ncv_simulator(X_holdout, Y_holdout, fun_list = list(logistic_lasso_funs), n = n, n_folds = n_folds, 
                       double_cv_reps = dcv_reps, n_cores = n_cores, n_sim = n_sim, tag = "sparse_logistic",
                       funcs_params = list("lambdas" = lambdas, "best_lam" = best_lam))
  save(out, file = paste0("data/sparse_log_n-", n, 
                          "_rho-", rho*100,
                          ".RData"))
  print(paste0("Results saved to disk."))
}

if(exists("cl")) {stopCluster(cl)}


#data splitting sims
n_ds_sims <- 5000
for(n in c(90)) {
  #data splitting sims (fast, about 5 minutes)
  fit <- cv.glmnet(X_holdout[1:n, ], Y_holdout[1:n], family = "binomial", foldid = (1:n %% n_folds + 1))
  lambdas <- fit$lambda
  best_lam <- match(fit$lambda.min, lambdas)
  best_lam
  lambdas[best_lam]
  lambdas <- lambdas[1:best_lam]
  
  set.seed(1)
  ds_sims <- ncv_simulator(X_holdout, Y_holdout, fun_list = list(logistic_lasso_funs), n = n, n_folds = 5, 
                           n_cores = n_cores, n_sim = n_ds_sims / 10, tag = "sparse_logistic",
                           funcs_params = list("lambdas" = lambdas, "best_lam" = best_lam), do_cv = F, do_ncv = F, do_boot632 = F)
  save(ds_sims, file = paste0("data/sparse_log_n-", n, 
                              "_rho-", rho*100,
                              "_datasplit.RData"))
  print(paste0("Results saved to disk."))
}

#check compute times
for(n in c(90)) {
  fit <- cv.glmnet(X_holdout[1:n, ], Y_holdout[1:n], family = "binomial", foldid = (1:n %% n_folds + 1))
  lambdas <- fit$lambda
  best_lam <- match(fit$lambda.min, lambdas)
  best_lam
  lambdas[best_lam]
  lambdas <- lambdas[1:best_lam]
  
  set.seed(1)
  temp <- ncv_simulator(X_holdout, Y_holdout, fun_list = list(logistic_lasso_funs), n = n, n_folds = 10, 
                        n_cores = n_cores, n_sim = 1, tag = "sparse_logistic",
                        funcs_params = list("lambdas" = lambdas, "best_lam" = best_lam), do_cv = T, do_ncv = T, do_boot632 = F)
}

##############################################


##############################################
#look at results
##############################################


quit() # don't execute the following code for batch jobs

print("Entering result analysis")
library(ggplot2)

n <- 90
rho <- .5
load(file = paste0("data/sparse_log_n-", n, 
                   "_rho-", rho*100,
                   ".RData"))
load(file = paste0("data/sparse_log_n-", n, 
                   "_rho-", rho*100,
                   "_datasplit.RData"))

alpha <- .1
qv <- qnorm(1-alpha/2)
vst <- function(x) {asin(sqrt(x))}

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
mean(sapply(out$cv_results, function(x) {vst(x$ho_err) < vst(x$err_hat) - qv * sqrt(1/(4*n))}))
mean(sapply(out$cv_results, function(x) {vst(x$ho_err) > vst(x$err_hat) + qv * sqrt(1/(4*n))}))

mean(sapply(out$cv_results, function(x) {vst(mean_err) < vst(x$err_hat) - qv * sqrt(1/(4*n))}))
mean(sapply(out$cv_results, function(x) {vst(mean_err) > vst(x$err_hat) + qv * sqrt(1/(4*n))}))

#ncv coverage
mean(sapply(out$ncv_results, function(x) {vst(x$ho_err) < vst(x$err_hat) - qv * x$sd_infl * sqrt(1/(4*n))}))
mean(sapply(out$ncv_results, function(x) {vst(x$ho_err) > vst(x$err_hat) + qv * x$sd_infl * sqrt(1/(4*n))}))

mean(sapply(out$ncv_results, function(x) {vst(mean_err) < vst(x$err_hat) - qv * x$sd_infl * sqrt(1/(4*n))}))
mean(sapply(out$ncv_results, function(x) {vst(mean_err) > vst(x$err_hat) + qv * x$sd_infl * sqrt(1/(4*n))}))

#data splitting with refitting
mean(sapply(ds_sims$ds_results, function(x) {vst(x$ho_err) < vst(x$err_hat) - qv * sqrt(5/(4*n))}))
mean(sapply(ds_sims$ds_results, function(x) {vst(x$ho_err) > vst(x$err_hat) + qv * sqrt(5/(4*n))}))

mean(sapply(ds_sims$ds_results, function(x) {vst(mean_err) < vst(x$err_hat) - qv * sqrt(5/(4*n))}))
mean(sapply(ds_sims$ds_results, function(x) {vst(mean_err) > vst(x$err_hat) + qv * sqrt(5/(4*n))}))


# inflation histograms
####################
my_hist <- qplot(sapply(out$ncv_results, function(x){x$sd_infl}), bins = 10) +
  labs(x = "NCV width") +
  theme_bw() +
  theme(aspect.ratio = 1)
my_hist

ggsave(my_hist, file = paste0("figures/high_logistic_inf_hist_", n,"_", 10*rho, ".pdf"), height = 2, width = 2.5)



