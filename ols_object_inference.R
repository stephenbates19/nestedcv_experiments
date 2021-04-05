library(nestedcv)
library(ggplot2)
library(tidyverse)
library(ggpubr)
library(latex2exp)

n <- 100
p <- 20
k <- 10

beta <- rep(0, p)  #note, strength doesn't matter with OLS

#create a large holdout set and comput SNR
set.seed(555)
n_holdout <- 20000
X_holdout <- matrix(rnorm(n = n_holdout * p), nrow = n_holdout)
Y_holdout <- X_holdout %*% beta + rnorm(n_holdout)
snr <- var(X_holdout %*% beta) / (var(Y_holdout) - var(X_holdout %*% beta))
print(paste0("The SNR is : ", snr))


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
                 loss = se_loss)
################################################

n_x <- 1000 #number of design matrices
n_reps <- 20 #number of reps

#experiment parameters
results <- list() #store results

# loop across n and k
set.seed(100)
for(i in 1:n_x) {
  print(paste0("starting design: ", i))
  X <- matrix(rnorm(n = n * p), nrow = n)

  for(j in 1:n_reps) {
    Y <- X %*% beta + rnorm(nrow(X))
    cv_fit <- naive_cv(X, Y, ols_funs, n_folds = k)
    fit <- lm(Y ~ X)
    ho_error <- mean((X_holdout %*% fit$coefficients[-1] + fit$coefficients[1] - Y_holdout)^2)
    cv_fit[["ho_error"]] <- ho_error
    cv_fit[["design"]] <- i
    
    results[[length(results) + 1]] <- cv_fit
  }
}

mean(sapply(results, function(x){x$ho_error > x$ci_hi}))

out <- data.frame(idx = 1:length(results))
for(mycol in c("err_hat", "ci_lo", "ci_hi", "raw_mean", "sd", "ho_error", "design")) {
  out[[mycol]] <- sapply(results, function(x){x[[mycol]]})
}

save(out, file ="data/ols_cover_choice.RData")
load(file ="data/ols_cover_choice.RData")


X <- matrix(rnorm(n = n * p), nrow = n)
Y <- X %*% beta + rnorm(nrow(X))
naive_cv(X, Y, ols_funs, n_folds = 10)$err_hat


temp <- out %>% 
  mutate(avg_err = mean(ho_error)) %>%
  group_by(design) %>%
  summarize("err" = mean(avg_err),
            "err_x" = mean(ho_error),
            "mse" = mean((err_hat - err)^2),
            "mse_x" = mean((err_hat - err_x)^2),
            "mse_xy" = mean((err_hat - ho_error)^2),
            "err_hat" = mean(raw_mean),
            "mis_lo" = mean(ho_error < ci_lo),
            "mis_hi" = mean(ho_error > ci_hi),
            "mis" = mis_lo + mis_hi,
            "ho_error" = mean(ho_error),
            "mis_lo_x" = mean(ho_error < ci_lo),
            "mis_hi_x" = mean(ho_error > ci_hi),
            "mis_x" = mis_lo_x + mis_hi_x,
            "mis_lo_pop" = mean(err < ci_lo),
            "mis_hi_pop" = mean(err > ci_hi),
            "mis_pop" = mis_lo_pop + mis_hi_pop)
colMeans(temp)          

mean((out$err_hat - mean(out$ho_error))^2)
mean((out$err_hat - out$ho_error)^2)



#plotting

triple_box <- ggplot(temp %>% pivot_longer(c("mis", "mis_x", "mis_pop"))) +
  geom_boxplot(aes(x = name, y = value)) +
  geom_line(aes(x  = name, y = value, group = design), color = "grey", alpha = .4) +
  geom_point(aes(x = name, y = value)) +
  labs(x = "target of inference", y = "miscoverage") +
  scale_x_discrete(limits=c("mis_pop", "mis_x", "mis"), labels = c("Err", "Err_x", "Err_xy")) +
  theme_bw() +
  theme(aspect.ratio = 1)
triple_box
ggsave(triple_box, filename = "figures/ols_cov_box.pdf", height = 2, width = 2.15)

triple_box_mse <- ggplot(temp %>% pivot_longer(c("mse_xy", "mse_x", "mse"))) +
  geom_boxplot(aes(x = name, y = value)) +
  geom_line(aes(x  = name, y = value, group = design), color = "grey", alpha = .4) +
  geom_point(aes(x = name, y = value)) +
  labs(x = "target of inference", y = "MSE") +
  scale_x_discrete(limits=c("mse", "mse_x", "mse_xy"), labels = c("Err", "Err_x", "Err_xy")) +
  theme_bw() +
  theme(aspect.ratio = 1)
triple_box_mse
ggsave(triple_box_mse, filename = "figures/ols_mse_box.pdf", height = 2, width = 2.15)

cor(temp$err_x, temp$err_hat)

scatter2 <- ggplot(out %>% 
                    filter(design %in% c(3)), 
                   aes(x = ho_error, y = raw_mean)) +
  geom_point(alpha = .1) +
  geom_smooth(method='lm', formula= y~x) +
  labs(x = "Err_XY", y ="CV estimate") +
  theme_bw() + 
  theme(aspect.ratio = 1)
scatter2

ggsave(scatter2, filename = "figures/ols_target3.pdf", height = 2, width = 2.15)


