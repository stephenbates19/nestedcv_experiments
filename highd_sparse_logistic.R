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
#problem setting for logistic lasso
##############################################

#sim settings
dcv_reps <- 200
n_folds <- 10
ns <- c(90, 200)

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


##############################################
# Run the sim
##############################################

# if(n_cores > 1) {
#   print("Starting cluster")
#   if(exists("cl")) {
#     print("Warning, object cl exits! Printing")
#     print(cl)
#   }
#   cl <- makeForkCluster(nnodes = n_cores)
#   registerDoParallel(cl, cores = n_cores)
#   clusterSetRNGStream(cl = cl, 123)
# }

# for(n in ns) {
#   print(paste0("Starting run: ", n))
#   
#   #Fit one model to find a good lambda. This lambda will be fixed in future simulations.
#   fit <- cv.glmnet(X_holdout[1:n, ], Y_holdout[1:n], family = "binomial", foldid = (1:n %% n_folds + 1))
#   lambdas <- fit$lambda
#   best_lam <- match(fit$lambda.min, lambdas)
#   best_lam
#   lambdas[best_lam]
#   lambdas <- lambdas[1:best_lam]
#   
#   #run the simulation
#   set.seed(1)
#   out <- ncv_simulator(X_holdout, Y_holdout, fun_list = list(logistic_lasso_funs), n = n, n_folds = n_folds, 
#                        double_cv_reps = dcv_reps, n_cores = n_cores, n_sim = n_sim, tag = "sparse_logistic",
#                        funcs_params = list("lambdas" = lambdas, "best_lam" = best_lam))
#   save(out, file = paste0("data/sparse_log_n-", n,".RData"))
#   print(paste0("Results saved to disk."))
# }

# if(exists("cl")) {stopCluster(cl)}

#data splitting sims (fast, about 5 minutes)
n_ds_sims <- 5000
for(n in ns) {
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
  save(ds_sims, file = paste0("data/sparse_log_n-", n,"_datasplit.RData"))
  print(paste0("Results saved to disk."))
}



#check compute times
for(n in ns) {
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
quit() # don't execute the following code for batch jobs

##############################################
#look at results
##############################################


print("Entering result analysis")
library(ggplot2)

n <- 90
load(file = paste0("data/sparse_log_n-", n,".RData"))
load(file = paste0("data/sparse_log_n-", n,"_datasplit.RData"))

out$parameters
length(out$ncv_results)




# teaser plot
##############################################

true_error <- sapply(out$cv_results, function(x){x$ho_err})
mean_error <- sapply(out$cv_results, function(x){x$err_hat})
mean_sd <- sapply(out$cv_results, function(x){x$sd}) / sqrt(n)

cor_plot <- ggplot() +
  geom_point(aes(x = true_error, y = mean_error), alpha = .04) +
  geom_smooth(aes(x = true_error, y = mean_error, color = "CI midpoint"), method = lm, se = FALSE, size = .5) +
  geom_smooth(aes(x = true_error, y = mean_error + 1.68 * mean_sd, color = "CI endpoints"), method = lm, se = FALSE, size = .5) +
  geom_smooth(aes(x = true_error, y = mean_error - 1.68 * mean_sd, color = "CI endpoints"), method = lm, se = FALSE, size = .5) +
  geom_quantile(aes(x = true_error, y = mean_error, color = "5% & 95% quantiles"), 
                quantiles = c(.05, .95), size = .5) +
  labs(x = "true error", y = "CV estimated error", color = "quantity") + 
  theme_bw() +
  guides(color=guide_legend(override.aes=list(fill=NA))) +
  theme(aspect.ratio = 1) 

cor_plot

ggsave(cor_plot, file = "figures/teaser.pdf", height = 2.5, width = 6)


#coverage calculations
##############################################

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
for(n in ns) {
  load(file = paste0("data/sparse_log_n-", n,".RData"))
  if(n == 90){load(file = paste0("data/sparse_log_n-", n," 1.RData"))}
  
  my_hist <- qplot(sapply(out$ncv_results, function(x){x$sd_infl}), bins = 10) +
    labs(x = "NCV width") +
    theme_bw() +
    theme(aspect.ratio = 1)
  
  ggsave(my_hist, file = paste0("figures/high_logistic_inf_hist_", n, ".pdf"), height = 2, width = 2.5)
}

# rough numbers for section 4.1
var(sapply(out$cv_results, function(x) {x$err_hat})) / mean(sapply(out$cv_results, function(x) {(x$ci_hi - x$ci_lo) / (2*qv)}))^2








