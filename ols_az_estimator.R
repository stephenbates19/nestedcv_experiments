#takes about 2000 cpu minutes

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

library(nestedcv)
library(doParallel)
library(foreach)
library(tidyverse)

source("data_wrapper.R")

##############################################
#problem setting
##############################################

p <- 20
k <- 4 #number of nonzeros
alpha <- .1 #nominal error rate, total across both tails.
qv <- qnorm(1 - alpha / 2) #unadjusted sd multiplier from gaussian density
n_folds <- 10
ns <- c(40, 100, 200, 400, 1600)


#sample Y from a linear model
strength <- 0  #signal strength
beta = strength * c(rep(1, k), rep(0, p - k))

#determine bayes error with this beta vector
set.seed(555)
n_holdout <- 20000
X_holdout <- matrix(rnorm(n_holdout * p), nrow = n_holdout)
Y_holdout <- rnorm(n_holdout)
snr <- var(X_holdout %*% beta) / (var(Y_holdout) - var(X_holdout %*% beta)) #SNR doesn't matter for ols

print(snr)
##############################################


##############################################
#subroutines for OLS
##############################################
se_loss <- function(y1, y2, funcs_params = NA) {
  (y1 - y2)^2
} 

fitter_ols <- function(X, Y, idx = NA, funcs_params = NA) {
  if(sum(is.na(idx)) > 0) {idx <- 1:nrow(X)}
  fit <- lm(Y[idx] ~ X[idx, ])
  
  fit
}

predictor_ols <- function(fit, X_new, funcs_params = NA) {
  X_new %*% fit$coefficients[-1] + fit$coefficients[1]
} 

ols_funs <- list(fitter = fitter_ols,
                 predictor = predictor_ols,
                 loss = se_loss,
                 name = "ols")
##############################################


##############################################
#Austern & Zhou estimator
##############################################
X <- X_holdout[1:100, ]
Y <- Y_holdout[1:100]
fold_id <- (1:(n/2)) %% (10) + 1
fold_id <- c(fold_id, fold_id)
print(fold_id)

naive_cv(X, Y, ols_funs, fold_id = fold_id)

az_estimator <- function(X, Y, funcs, K = 10) {
  n <- nrow(X)
  fold_id <- (1:(n/2)) %% (K) + 1
  fold_id <- c(fold_id, fold_id)
  
  diffs <- rep(0, n / 2)
  for(i in 1:(n/2)) {
    r <- naive_cv(X[1:(n/2), ], Y[1:(n/2)], funcs = funcs, fold_id = fold_id[1:(n/2)])$raw_mean
    
    idx <- (1:(n/2))
    idx[i] <- i + n/2
    rprime <- naive_cv(X[idx, ], Y[idx], funcs = funcs, fold_id = fold_id[idx])$raw_mean
    
    diffs[i] <- (r - rprime)^2
  }
  
  # print(diffs)
  return(sqrt(sum(diffs) / 2))
}

az_estimator(X, Y, ols_funs, K = 10)

#############################################




###


reps <- 200
out <- c()
#n <- 400
for(n in c(40, 100, 200, 400, 800)) {

  print(paste0("n: ", n))
  
  for(i in 1:reps) {
    if(i %% 10 == 0) {print(i)}
    X <- matrix(rnorm(n * p), nrow = n)
    Y <- rnorm(n)
    
    fit <- lm(Y ~ X)
    ho_err <- mean((X_holdout %*% fit$coefficients[-1] + fit$coefficients[1] - Y_holdout)^2)
    
    az <- az_estimator(X, Y, ols_funs, K = 10)
    cv <- naive_cv(X, Y, ols_funs)
    out <- rbind(out, c(ho_err, cv$raw_mean, cv$sd / sqrt(n), az, n, p))
  }

}
out <- as.data.frame(out)
colnames(out) <- c("ho_err", "cv_est", "cv_se", "az_se", "n", "p")

# temp <- out
# temp2 <- rbind(temp, out)
# save(temp2, file = "data/ols_az_est.RData")
load(file = "data/ols_az_est.RData")
dim(temp2)

out2 <- temp2 %>% group_by(n, p) %>%
  summarize(err = mean(ho_err),
            cv_mean = mean(cv_est),
            cv_se_hat = mean(cv_se),
            cv_se = sd(cv_est),
            az_se_hat = mean(az_se))
out2

az_plot <- ggplot(out2, aes(x = n, y = az_se_hat / cv_se_hat, color = factor(p))) +
  geom_point(aes(shape = factor(p))) + 
  geom_line() +
  # lims(y = c(0, 3)) +
  scale_x_sqrt(breaks = c(40, 100, 200, 400, 800)) + 
  geom_hline(yintercept = 1, color = "dark grey") +
  labs(y = "A-Z SE / Naive SE", color = "dimension", shape = "dimension") + 
  theme_bw() + 
  theme(aspect.ratio = 1)
az_plot

ggsave(az_plot, filename = "figures/ols_infl_az.pdf", height = 2.75, width = 3.5)



colMeans(out)
sd(out[, 2])



