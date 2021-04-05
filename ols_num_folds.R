library(nestedcv)
library(ggplot2)

n <- 100
p <- 20

beta <- c(1, rep(0, p - 1))
strength <- 1
beta <- beta * strength #note, strength doesn't matter with OLS

# create the design matrix 
set.seed(111)
X <- matrix(rnorm(n = n * p), nrow = n)
Y <- X %*% beta + rnorm(nrow(X))

#create a large holdout set and comput SNR
set.seed(555)
n_holdout <- 20000
X_holdout <- matrix(rnorm(n_holdout * p), nrow = n_holdout)
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

n_reps <- 1000

#experiment parameters
ns <- c(50, 100, 200, 400)
ks <- c(5, 10, 25)
results <- c() #store results

# loop across n and k
set.seed(100)
for(n in ns) {
  for(k in c(ks,n)) {
    print(c(n, k))

    z_vals <- c()
    cvg <- 0
    wid <- c()
    for(i in 1:n_reps) {
      X <- matrix(rnorm(n = n * p), nrow = n)
      Y <- X %*% beta + rnorm(nrow(X))
      cv_fit <- naive_cv(X, Y, ols_funs, n_folds = k)
      fit <- lm(Y ~ X)
      ho_error <- mean((X_holdout %*% fit$coefficients[-1] + fit$coefficients[1] - Y_holdout)^2)
      
      z_val <- (ho_error - cv_fit$raw_mean) / cv_fit$sd * sqrt(n)
      z_vals <- c(z_vals, z_val)
      
      if((ho_error < cv_fit$ci_hi) & (ho_error > cv_fit$ci_lo)) {
        cvg <- cvg + 1
      }
      wid <- c(wid, cv_fit$sd / sqrt(n))
    }
    print(sd(z_vals))
    print(cvg / n_reps)
    
    results <- rbind(results, c(n, k, sd(z_vals), cvg / n_reps, mean(wid)))
  }
}
colnames(results) <- c("n", "k", "inflation", "coverage", "width")

save(results, file ="data/ols_num_folds.RData")
load(file ="data/ols_num_folds.RData")

results <- data.frame(results)
results$k[results$k == results$n] <- "LOO"

#plotting
infl_plot <- ggplot(as.data.frame(results), aes(x = n, y = inflation, color = as.factor(k))) +
  geom_point() +
  geom_line() +
  geom_hline(yintercept = 1, color = "dark grey") + 
  scale_x_sqrt() + 
  labs(x = "n", y = "CV inflation", color = "num. folds") +
  theme_bw() + 
  theme(aspect.ratio = 1)

infl_plot

covg_plot <- ggplot(as.data.frame(results), aes(x = n, y = 1 - coverage, color = as.factor(k))) +
  geom_point() +
  geom_line() +
  geom_hline(yintercept = .1, color = "dark grey") + 
  lims(y = c(0, .5)) + 
  scale_x_sqrt() + 
  labs(x = "n", y = "miscoverage", color = "num. folds") +
  theme_bw() + 
  theme(aspect.ratio = 1)

covg_plot

width_plot <- ggplot(as.data.frame(results), aes(x = n, y = width, color = as.factor(k))) +
  geom_point() +
  geom_line() +
  lims(y = c(0, .5)) + 
  scale_x_sqrt() + 
  labs(x = "n", y = "CI width", color = "num. folds") +
  theme_bw() + 
  theme(aspect.ratio = 1)
width_plot

#combine into one figure
library(ggpubr)
combined_plot <- ggarrange(infl_plot, covg_plot, width_plot, ncol = 3, align = "v", 
                           common.legend = TRUE, legend="bottom")
combined_plot

ggsave(combined_plot, filename = "figures/choice_k.pdf", width = 6.5, height = 2.5)
