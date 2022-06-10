library(glmnet)

#sample Y from a linear model
p <- 20
k <- 4 #number of nonzeros
strength <- 0  #signal strength
beta = strength * c(rep(1, k), rep(0, p - k))

#determine bayes error with this beta vector
set.seed(555)
n_holdout <- 20000
X_holdout <- matrix(rnorm(n_holdout * p), nrow = n_holdout)
Y_holdout <- rnorm(n_holdout)
snr <- var(X_holdout %*% beta) / (var(Y_holdout) - var(X_holdout %*% beta)) #SNR doesn't matter for ols

print(snr)

#one instance of the problem
n <- 100
X <- X_holdout[1:n, ]
Y <- Y_holdout[1:n]

fit <- cv.glmnet(X, Y, intercept = F)
fit$cvm
lambdas <- c(fit$lambda)
mean((Y_holdout - predict.glmnet(fit$glmnet.fit, X_holdout))^2)

#norm of ridge beta / norm of ols beta at different lambdas
colSums(fit$glmnet.fit$beta^2)
fit2 <- lm(Y ~ X)
sum(coef(fit2)^2)
ref_grid <- colSums(fit$glmnet.fit$beta^2) / sum(coef(fit2)^2) 
sqrt(ref_grid)



#cv fits
n_sim <- 4000
fits <- list()
for(i in 1:n_sim) {
  X <- matrix(rnorm(n * p), nrow = n)
  Y <- rnorm(n)
  
  fits[[i]] <- cv.glmnet(X, Y, lambda = lambdas, intercept = F)
}

cors <- c()
for(i in 1:length(lambdas)) {
  errhats <- unlist(lapply(fits, function(x){x$cvm[i]}))
  pred_acc <- unlist(lapply(fits, function(x){
    beta <- x$glmnet.fit$beta[, i]
    1 + sum(beta^2)
  }))
  
  cors <- c(cors, cor(errhats, pred_acc))
}
cors

dat <- data.frame("shrinkage" = 1 - ref_grid, "cor" = cors)
cor_plot <- ggplot(dat, aes(x = shrinkage, y = cor)) +
  geom_point() + 
  labs(x = "shrinkage (%)", y = TeX("cor( \ $\\widehat{Err}$, \ $Err_{XY}$ \ )")) + 
  theme_bw() + 
  theme(aspect.ratio = 1)
cor_plot
ggsave(cor_plot, filename = "figures/highd_linear_cor.pdf", height = 3, width = 4)
