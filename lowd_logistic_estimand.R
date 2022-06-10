library(glmnet)
library(nestedcv)
library(tidyverse)
library(magrittr)
library(latex2exp)

#sim settings
n_folds <- 10

n <- 300
p <- 20
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

p_x <- 1 / (1 + exp(-X %*% beta))
Y <- (runif(n) < p_x) * 1.0

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

n_sim <- 2000
# n <- 100
# p <- 15

set.seed(1)
out <- c()
for(n in c(100, 200, 400)) {
for(p in c(n * .15, n / 10, n / 25)) {
print(paste0("starting: ", n, ", ", p))

#large holdout data set
strength <- 1 #signal strength
beta = strength * c(rep(1, k), rep(0, p - k))
n_holdout <- 10000
X_holdout <- matrix(rnorm(n_holdout * p), nrow = n_holdout)
p_holdout <- 1 / (1 + exp(-X_holdout %*% beta))
Y_holdout <- (runif(n_holdout) < p_holdout) * 1.0

for(i in 1:n_sim) {
  if(i %% 100 == 0){print(i)}
  
  X <- matrix(rnorm(n = n * p), nrow = n)
  p_x <- 1 / (1 + exp(-X %*% beta))
  Y <- (runif(n) < p_x) * 1.0
  
  cv <- naive_cv(X, Y, funcs = logistic_funs)
  fit <- fitter_logistic(X, Y)
  preds <- predictor_logistic(fit, X_holdout)
  ho_err = mean((preds > .5) != Y_holdout)
  out <- rbind(out, c(cv$raw_mean, ho_err, n, p))
}
}
}
out
colnames(out) <- c("cv_est", "ho_err", "n", "p")
save(out, file = "data/lowd_logistic_estimand.RData")

idx <- which(out[, 3] == 400 & out[, 4] == 16)

cor(out[idx, 1], out[idx, 2])
mean(abs(out[idx, 1] - out[idx, 2]))
mean(abs(out[idx, 1] - mean(out[idx, 2])))

out2 <- as.data.frame(out) %>%
  group_by(n, p) %>%
  summarise(cv_mean = mean(cv_est),
            err = mean(ho_err),
            corr = cor(cv_est, ho_err),
            mab = mean(abs(cv_est - err)),
            mab_xy = mean(abs(cv_est - ho_err)))
out2

cor_plot <- ggplot(out2, aes(x = n, y = corr, color = as.factor(round(n / p, 2)))) +
  geom_point(aes(shape = as.factor(round(n / p, 2)))) +
  geom_line() + 
  lims(y = c(0, .2)) +
  scale_x_sqrt() + 
  geom_hline(yintercept = 0, color = "dark grey") + 
  theme_bw() +
  labs(color = "n / p", shape = "n / p", y = TeX("cor( \ $\\widehat{Err}$, \ $Err_{XY}$ \ )")) +
  theme(aspect.ratio = 1)
cor_plot

ggsave(cor_plot, filename = "figures/lowd_logistic_cor.pdf", height = 2.5, width = 3)

prec_plot <- ggplot(out2, aes(x = n, y = (mab_xy - mab) / mab, color = as.factor(round(n / p, 2)))) +
  geom_point(aes(shape = as.factor(round(n / p, 2)))) +
  geom_line() +
  lims(y = c(0, .06)) +
  geom_hline(yintercept = 0, color = "dark grey") + 
  scale_x_sqrt() + 
  theme_bw() +
  labs(color = "n / p", shape = "n / p", y = "precision gap (%)") +
  theme(aspect.ratio = 1)
prec_plot
ggsave(prec_plot, filename = "figures/lowd_logistic_prec.pdf", height = 2.5, width = 3)


