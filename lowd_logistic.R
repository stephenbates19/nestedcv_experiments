library(nestedcv)
library(ggplot2)
library(tidyverse)
library(doParallel)
library(foreach)
source("data_wrapper.R")

n <- 100
p <- 10

beta <- c(1, rep(0, p - 1))
strength <- 1
beta <- beta * strength

# create the design matrix 
set.seed(111)
X <- matrix(rnorm(n = n * p), nrow = n)
probs <- 1 / (1 + exp(-X %*% beta))
Y <- (runif(n) < probs) * 1.0

#create a large holdout set and comput SNR
set.seed(555)
n_holdout <- 20000
X_holdout <- matrix(rnorm(n_holdout * p), nrow = n_holdout)
probs_holdout <- 1 / (1 + exp(-X_holdout %*% beta))
Y_holdout <- (runif(n_holdout) < probs_holdout) * 1.0

snr <- var(probs_holdout) / (var(Y_holdout) - var(probs_holdout))
print(paste0("The SNR is : ", snr))
print(paste0("The Bayes error rate is:", 1 - mean((probs_holdout > .5) == (Y_holdout == 1))))

##############################################
#subroutines for logistic regression
##############################################
misclass_loss <- function(y_hat, y, funcs_params = NULL) {
  y_hat <- round(y_hat) #convert to 0-1 predictions
  y_hat != y
} 

fitter_logistic <- function(X, Y, idx = NA, funcs_params = NULL) {
  if(sum(is.na(idx)) > 0) {idx <- 1:nrow(X)}
  fit <- list("coefficients" = rep(0, p + 1))
  tryCatch(
    fit <- glm(Y[idx] ~ X[idx, ], family = binomial(link="logit")),
    warning = function(w) {
      print(w)
      #upon warning, use intercept-only model
      fit$coefficients <- rep(0, p + 1)
      fit$coefficients[1] <- -log(1/mean(Y) - 1)
    },
    error = function(w) {
      print(w)
      #upon warning, use intercept-only model
      fit$coefficients <- rep(0, p + 1)
      fit$coefficients[1] <- -log(1/mean(Y) - 1)
    }
  )
  
  fit
}

predictor_logistic <- function(fit, X_new, funcs_params = NULL) {
  probs <- 1/(1 + exp(-X_new %*% fit$coefficients[-1] - fit$coefficients[1]))
  
  probs > .5
} 

logistic_funs <- list(fitter = fitter_logistic,
                 predictor = predictor_logistic,
                 loss = misclass_loss)
################################################

n_sim <- 1000

#experiment parameters
ns <- c(100)
strengths <- c(1, 2, 4)
n_cores <- 1

# loop across n and k
for(strength in strengths[1:2]) {
  print(paste0("Starting run: ", strength))
  beta <- c(1, rep(0, p - 1))
  beta <- beta * strength
  
  probs_holdout <- 1 / (1 + exp(-X_holdout %*% beta))
  Y_holdout <- (runif(n_holdout) < probs_holdout) * 1.0
  
  
  set.seed(100)
  out <- ncv_simulator(X_holdout, Y_holdout, fun_list = list(logistic_funs), n = n, n_folds = 10, 
                       double_cv_reps = 200, n_cores = n_cores, n_sim = n_sim, tag = "log_reg",
                       do_cv = T, do_ncv = T, do_boot632 = F)
  save(out, file = paste0("data/lowd_logistic_s-", strength,"_ds.RData"))
  print(paste0("Results saved to disk."))
}
save(results, file ="data/lowd_logistic.RData")


# check compute times
for(strength in strengths[1:2]) {
  print(paste0("Starting run: ", strength))
  beta <- c(1, rep(0, p - 1))
  beta <- beta * strength
  
  probs_holdout <- 1 / (1 + exp(-X_holdout %*% beta))
  Y_holdout <- (runif(n_holdout) < probs_holdout) * 1.0
  
  
  set.seed(100)
  out <- ncv_simulator(X_holdout, Y_holdout, fun_list = list(logistic_funs), n = n, n_folds = 10, 
                       double_cv_reps = 200, n_cores = n_cores, n_sim = 1, tag = "log_reg",
                       do_cv = T, do_ncv = T, do_boot632 = F)
}


##############################################
#analysis + plotting
##############################################
quit() #dont run this for batch jobs


strength <- 2
load(file = paste0("data/lowd_logistic_s-", strength,"_ds 1.RData"))
alpha <- .1
qv <- qnorm(1 - alpha / 2)

mean_err <- mean(sapply(out$cv_results, function(x) {x$ho_err}))
mean_err

#width
mean(sapply(out$ncv_results, function(x) {x$ci_hi - x$ci_lo})) / mean(sapply(out$cv_results, function(x) {x$ci_hi - x$ci_lo}))
mean(sapply(out$ds_results, function(x) {x$se_hat * 2 * qv})) / mean(sapply(out$cv_results, function(x) {x$ci_hi - x$ci_lo}))

#point estimates
mean_err
mean(sapply(out$cv_results, function(x) {x$err_hat}))
mean(sapply(out$ncv_results, function(x) {x$err_hat}))
mean(sapply(out$ds_results, function(x) {x$err_hat}))

#cv coverage
mean(sapply(out$cv_results, function(x) {vst(x$ho_err) < vst(x$err_hat) - qv * sqrt(1/400)}))
mean(sapply(out$cv_results, function(x) {vst(x$ho_err) > vst(x$err_hat) + qv * sqrt(1/400)}))

mean(sapply(out$cv_results, function(x) {vst(mean_err) < vst(x$err_hat) - qv * sqrt(1/400)}))
mean(sapply(out$cv_results, function(x) {vst(mean_err) > vst(x$err_hat) + qv * sqrt(1/400)}))


#ncv coverage
mean(sapply(out$ncv_results, function(x) {vst(x$ho_err) < vst(x$err_hat) - qv * x$sd_infl * sqrt(1/400)}))
mean(sapply(out$ncv_results, function(x) {vst(x$ho_err) > vst(x$err_hat) + qv * x$sd_infl * sqrt(1/400)}))

mean(sapply(out$ncv_results, function(x) {vst(mean_err) < vst(x$err_hat) - qv * x$sd_infl * sqrt(1/400)}))
mean(sapply(out$ncv_results, function(x) {vst(mean_err) > vst(x$err_hat) + qv * x$sd_infl * sqrt(1/400)}))


#data splitting with refitting
mean(sapply(out$ds_results, function(x) {vst(x$ho_err) < vst(x$err_hat) - qv * sqrt(1/80)}))
mean(sapply(out$ds_results, function(x) {vst(x$ho_err) > vst(x$err_hat) + qv * sqrt(1/80)}))

mean(sapply(out$ds_results, function(x) {vst(mean_err) < vst(x$err_hat) - qv * sqrt(1/80)}))
mean(sapply(out$ds_results, function(x) {vst(mean_err) > vst(x$err_hat) + qv * sqrt(1/80)}))



# size histograms
######################3
for(strength in c(1,2)) {
  load(file = paste0("data/lowd_logistic_s-", strength,"_ds 1.RData"))
  
  my_hist <- qplot(sapply(out$ncv_results, function(x){x$sd_infl}), bins = 10) +
    labs(x = "NCV width") +
    theme_bw() +
    theme(aspect.ratio = 1)
  
  ggsave(my_hist, file = paste0("figures/lowd_logistic_inf_hist_", strength, ".pdf"), height = 2, width = 2.5)
}




