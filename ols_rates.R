library(tidyverse)
library(ggplot2)

gamma <- 5
n_reps <- 5000
ns <-c(50, 100, 140, 200, 280)
ns <- c(50, 100, 140, 200, 280, 400, 560, 800, 1120)
ho_frac <- .2

cv_helper <- function(X, Y, n_folds = 10) {
  fold_id <- sort(1:nrow(X) %% n_folds + 1)
  #print(fold_id)
  
  temp <- c()
  for(k in 1:n_folds) {
    Xt <- X[fold_id != k, ]
    Yt <- Y[fold_id != k]
    
    betat <- solve((t(Xt) %*% Xt), t(Xt) %*% Yt)
    errs <- (X[fold_id == k, ] %*% betat - Y[fold_id == k])^2
    #print(length(errs))
    temp <- c(temp, errs)
  }
  
  #print(length(temp))
  c(mean(temp), sd(temp) / sqrt(nrow(X)))
}
#cv_helper(X,Y)

results <- c()
for(n in ns) {
  print(n)
  print(Sys.time())
  p <- round(n / gamma)

  train_idx <- 1:((1-ho_frac)*n)
  out_idx <- setdiff(1:n, train_idx)
  
  set.seed(100)
  for(i in 1:n_reps){
    X <- matrix(rnorm(n * p), nrow = n)
    Xt <- X[train_idx, ]
    Y <- rnorm(n)
    Yt <- Y[train_idx]
    beta <- solve((t(X) %*% X), t(X) %*% Y)
    betat <- solve((t(Xt) %*% Xt), t(Xt) %*% Yt)
    
    pe_est <- mean((X[out_idx, ] %*% betat - Y[out_idx])^2)
    sd_est <- sd((X[out_idx, ] %*% betat - Y[out_idx])^2)
    
    cv_est <- cv_helper(X, Y)
    #cv_est <- c(0,0)
    
    err_xy <-  1 + sum(beta^2)
    err_x <- 1 + sum(diag(solve(t(X) %*% X)))
    
    results <- rbind(results, c(n, err_xy, err_x, pe_est, sd_est, cv_est))
  }
}
colnames(results) <- c("n", "err_xy", "err_x", "ho_est", "ho_sd_est", "cv_est", "cv_se_est")

save(results, file = "data/ols_rates.RData")
load(file = "data/ols_rates.RData")

temp <- results %>% 
  as.data.frame() %>%
  group_by(n) %>%
  mutate(err = mean(err_x)) %>%
  ungroup()

#rates plot
temp2 <- temp %>% 
  group_by(n) %>%
  summarize(err = mean(err),
            err_xy_gap = mean(abs(err_xy - err)), 
            err_x_gap = mean(abs(err_x - err)), 
            bias = mean(cv_est - err),
            ds_gap = mean(abs(ho_est - err_xy)),
            ds_gap_deb = mean(abs(ho_est - bias - err_xy)),
            ds_se = sqrt(mean(ho_sd_est^2 / (n*ho_frac))),
            cv_gap = mean(abs(cv_est - err)),
            cv_gap_xy = mean(abs(cv_est - err_xy)),
            cv_se = sqrt(mean(cv_se_est^2)),
            cor = cor(err_xy, cv_est))
  
library(latex2exp)
l1 <- TeX(r'(| \widehat{Err} - Err |)')
plot(l1)
l2 <- TeX(r'(| Err_{XY} - Err |)')
plot(l2)
l3 <- TeX(r'(| Err_X - Err |)')
plot(l3)

mylabs <- unname(TeX(c("Err",r'($|$ $\widehat{Err}$ - Err $|$)', 
                       r'($|$ $Err_{XY} - Err$ $|$)', r'($|$ $Err_{X} - Err$ $|$)')))

rates_plot <- ggplot(temp2, aes(x = n)) +
  geom_line(aes(y= err_xy_gap, color = "3")) +
  geom_point(aes(y= err_xy_gap, color = "3", shape = "3")) +
  geom_line(aes(y= err_x_gap, color = "4")) +
  geom_point(aes(y= err_x_gap, color = "4", shape = "4")) +
  geom_line(aes(y= cv_gap, color = "2")) +
  geom_point(aes(y= cv_gap, color = "2", shape = "2")) +
  # geom_line(aes(y= cv_gap_xy, color = "|Errhat_cv - Err_xy|")) +
  # geom_point(aes(y= cv_gap_xy, color = "|Errhat_cv - Err_xy|", shape = "|Errhat_cv - Err_xy|")) +
  geom_line(aes(y= err, color = "1")) +
  geom_point(aes(y= err, color = "1", shape = "1")) +
  # geom_line(aes(y= bias, color = "sample-size bias")) +
  # geom_point(aes(y= bias, color = "sample-size bias")) +
  # geom_line(aes(y= ds_gap_deb, color = "|err_hat - err_xy| debiased")) +
  # geom_point(aes(y= ds_gap_db, color = "|err_hat - err_xy| debiased")) +
  # geom_line(aes(y= ds_sd, color = "data split est")) +
  # geom_point(aes(y= ds_sd, color = "data split est")) +
  scale_y_log10() +
  scale_x_log10() +
  scale_color_discrete(name = "quantity", labels = mylabs) +
  scale_shape_discrete(name = "quantity", labels = mylabs) + 
  labs(y = "") + 
  theme_bw() +
  theme(aspect.ratio = 1)
rates_plot
ggsave(rates_plot, file = "figures/highd_rates.pdf", height = 2.5, width = 4.5)

mylabs2 <- unname(TeX(c(r'($Err_{XY}$)', "Err")))
rates_plot2 <- ggplot(temp2, aes(x = n)) +
  geom_line(aes(y= cv_gap, color = "2")) +
  geom_point(aes(y= cv_gap, color = "2", shape = "2")) +
  geom_line(aes(y= cv_gap_xy, color = "1")) +
  geom_point(aes(y= cv_gap_xy, color = "1", shape = "1")) +
  # scale_x_log10(limits = c(100,1200)) +
  # scale_y_log10(limits = c(.055,.2)) +
  scale_x_log10() +
  scale_y_log10() +
  scale_color_discrete(name = "target", labels = mylabs2) +
  scale_shape_discrete(name = "target", labels = mylabs2) + 
  labs(y = "mean absolute deviation") + 
  theme_bw() +
  theme(aspect.ratio = 1)
rates_plot2
ggsave(rates_plot2, file = "figures/highd_rates2.pdf", height = 2.5, width = 3.5)

rates_plot3 <- ggplot(temp2, aes(x = n)) +
  geom_line(aes(y= cor)) +
  geom_point(aes(y= cor)) +
  scale_x_log10() +
  geom_hline(yintercept = 0, color = "dark grey") + 
  lims(y = c(-.2, .2)) + 
  labs(y = "correlation", color = "quantity", shape = "quantity") + 
  theme_bw() +
  theme(aspect.ratio = 1)
rates_plot3
ggsave(rates_plot3, file = "figures/highd_rates3.pdf", height = 2.5, width = 3)

# slopes
lm(log(temp2$err_xy_gap) ~ log(temp2$n))
lm(log(temp2$err_x_gap) ~ log(temp2$n))
lm(log(temp2$cv_gap) ~ log(temp2$n))
lm(log(temp2$err) ~ log(temp2$n))


#coverage plot
alpha <- .1
qv <- qnorm(1-alpha/2)

temp3 <- temp %>% 
  group_by(n) %>%
  summarize(bias = mean(cv_est - err),
            covg_xy = mean(abs(cv_est - err_xy) > cv_se_est * qv), 
            covg = mean(abs(cv_est - err) > cv_se_est  * qv),
            covg_xy_deb = mean(abs(cv_est - bias- err_xy) > cv_se_est  * qv), 
            covg_deb = mean(abs(cv_est - bias - err) > cv_se_est  * qv),
            covg_deb_lo = mean((cv_est - bias - err_xy) > cv_se_est  * qv),
            covg_deb_hi = mean((cv_est - bias - err_xy) < -cv_se_est  * qv)
            )

covg_plot <- ggplot(temp3, aes(x = n)) +
  geom_line(aes(y = covg_xy, color = "miscoverage err_xy")) +
  geom_point(aes(y = covg_xy, color = "miscoverage err_xy", shape = "miscoverage err_xy")) +
  geom_line(aes(y = covg, color = "miscoverage err")) +
  geom_point(aes(y = covg, color = "miscoverage err", shape = "miscoverage err")) +
  geom_line(aes(y = covg_xy_deb, color = "miscoverage err_xy, debiased")) +
  geom_point(aes(y = covg_xy_deb, color = "miscoverage err_xy, debiased", shape = "miscoverage err_xy, debiased")) +
  geom_line(aes(y = covg_deb, color = "miscoverage err, debiased")) +
  geom_point(aes(y = covg_deb, color = "miscoverage err, debiased", shape = "miscoverage err, debiased")) +
  geom_hline(yintercept = alpha) + 
  labs(y = "miscoverage", color = "quantity", shape = "quantity") +
  scale_x_log10() + 
  ylim(c(0,.3)) +
  theme_bw() +
  theme(aspect.ratio = 1)
covg_plot

covg_plot2 <- ggplot(temp3, aes(x = n)) +
  geom_line(aes(y = covg_deb_hi, color = "miscoverage above")) +
  geom_point(aes(y = covg_deb_hi, color = "miscoverage above", shape = "miscoverage above")) +
  geom_line(aes(y = covg_deb_lo, color = "miscoverage below")) +
  geom_point(aes(y = covg_deb_lo, color = "miscoverage below", shape = "miscoverage below")) +
  geom_hline(yintercept = alpha / 2) + 
  labs(y = "miscoverage", color = "quantity", shape = "quantity") +
  scale_x_log10() + 
  ylim(c(0,.2)) +
  theme_bw() +
  theme(aspect.ratio = 1)
covg_plot2

ggsave(covg_plot, filename = "figures/cv_ols_covg1.pdf", height = 2.5, width = 5)
ggsave(covg_plot2, filename = "figures/cv_ols_covg2.pdf", height = 2.5, width = 5)

temp %>% group_by(n) %>% summarize(var(sqrt(n) * err_xy))


temp4 <- temp %>% 
  group_by(n) %>%
  summarize(err = mean(err),
            bias = mean(cv_est - err),
            vari = var(cv_est - err))

rates_plot_bv <- ggplot(temp4, aes(x = n)) +
  geom_line(aes(y= bias^2, color = "bias^2")) +
  geom_point(aes(y= bias^2, color = "bias^2", shape = "bias^2")) +
  geom_line(aes(y= (vari), color = "variance")) +
  geom_point(aes(y= (vari), color = "variance", shape = "variance")) +
  scale_y_log10() + 
  scale_x_log10() +
  labs(y = "", shape = "", color = "") + 
  theme_bw() +
  theme(aspect.ratio = 1)
rates_plot_bv
ggsave(rates_plot_bv)


########################
##### Data splitting
########################

n_reps <- 5000
ns <- c(50, 100, 140, 200, 280, 400, 560, 800, 1120, 1600, 2240, 3200)

results2 <- c()
for(n in ns) {
  print(n)
  print(Sys.time())
  p <- round(n / gamma)
  
  train_idx <- 1:((1-ho_frac)*n)
  out_idx <- setdiff(1:n, train_idx)
  
  set.seed(100)
  for(i in 1:n_reps){
    X <- matrix(rnorm(n * p), nrow = n)
    Xt <- X[train_idx, ]
    Y <- rnorm(n)
    Yt <- Y[train_idx]
    beta <- solve((t(X) %*% X), t(X) %*% Y)
    betat <- solve((t(Xt) %*% Xt), t(Xt) %*% Yt)
    
    pe_est <- mean((X[out_idx, ] %*% betat - Y[out_idx])^2)
    sd_est <- sd((X[out_idx, ] %*% betat - Y[out_idx])^2)
    
    err_xy <-  1 + sum(beta^2)
    
    results2 <- rbind(results2, c(n, err_xy, pe_est, sd_est))
  }
}
colnames(results2) <- c("n", "err_xy", "ho_est", "ho_sd_est")

save(results2, file = "data/ols_rates2.RData")
#load(file = "data/ols_rates2.RData")

temp4 <- results2 %>% 
  as.data.frame() %>%
  group_by(n) %>%
  summarize(err = mean(err_xy),
            bias = mean(ho_est - err),
            covg_xy = mean(abs(ho_est - err_xy) > ho_sd_est * qv / sqrt(n * ho_frac)), 
            covg = mean(abs(ho_est - err) > ho_sd_est  * qv / sqrt(n * ho_frac)),
            covg_xy_deb = mean(abs(ho_est - bias- err_xy) > ho_sd_est  * qv / sqrt(n * ho_frac)), 
            covg_deb = mean(abs(ho_est - bias - err) > ho_sd_est  * qv / sqrt(n * ho_frac)))

ds_cvg_plot <- ggplot(temp4, aes(x = n)) +
  geom_line(aes(y = covg_xy, color = "miscovg of Err_xy")) +
  geom_point(aes(y = covg_xy, color = "miscovg of Err_xy", shape = "miscovg of Err_xy",)) +
  geom_line(aes(y = covg, color = "miscovg of Err")) +
  geom_point(aes(y = covg, color = "miscovg of Err", shape = "miscovg of Err")) +
  geom_line(aes(y = covg_xy_deb, color = "miscovg of Err_xy, debiased")) +
  geom_point(aes(y = covg_xy_deb, color = "miscovg of Err_xy, debiased", shape = "miscovg of Err_xy, debiased")) +
  geom_line(aes(y = covg_deb, color = "miscovg of Err, debiased")) +
  geom_point(aes(y = covg_deb, color = "miscovg of Err, debiased", shape = "miscovg of Err, debiased")) +
  geom_hline(yintercept = alpha) + 
  labs(y = "miscoverage", color = "quantity", shape = "quantity") +
  ylim(c(0,.4)) +
  scale_x_log10() + 
  theme_bw() + 
  theme(aspect.ratio = 1)

ds_cvg_plot
ggsave(ds_cvg_plot, file = "figures/ds_covg_ols.pdf", height = 2.5, width = 5)
