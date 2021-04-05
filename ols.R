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
# Run the sim
##############################################
n_sim <- 1000

for(n in ns) {
  print(paste0("Starting run: ", n))
  
  out <- ncv_simulator(X_holdout, Y_holdout, fun_list = list(ols_funs), n = n, n_folds = n_folds, 
                       double_cv_reps = 200, n_cores = n_cores, n_sim = n_sim, tag = "ols",
                       trans = list(identity), do_cv = F, do_ncv = F, do_boot632 = F)
  save(out, file = paste0("data/ols_n-", n,".RData"))
  print(paste0("Results saved to disk."))
}

n_ds_sims <- 5000
for(n in ns) {
  print(paste0("Starting run: ", n))
  set.seed(1)
  ds_sims <- ncv_simulator(X_holdout, Y_holdout, fun_list = list(ols_funs), n = n, n_folds = 5, 
                           n_cores = n_cores, n_sim = n_ds_sims / 10, tag = "ols",
                           funcs_params = NA, 
                           do_cv = F, do_ncv = F, do_boot632 = F)
  save(ds_sims, file = paste0("data/ols_n-", n,"_ds.RData"))
  print(paste0("Results saved to disk."))
}

#check compute times
for(n in ns) {
  print(paste0("Starting run: ", n))
  set.seed(1)
  ds_sims <- ncv_simulator(X_holdout, Y_holdout, fun_list = list(ols_funs), n = n, n_folds = 10, 
                           n_cores = n_cores, n_sim = 1, tag = "ols",
                           funcs_params = NA, 
                           do_cv = T, do_ncv = T, do_boot632 = F)
}

##############################################

quit()

library(tidyverse)
library(ggplot2)

#load results
all_res <- list()
for(n in c(40, 100, 200, 400, 1600)) {
  load(file = paste0("data/ols_n-", n,".RData"))
  print(length(out))
  load(file = paste0("data/ols_n-", n,"_ds.RData"))
  out[["ds_results"]] <- ds_sims$ds_results
  all_res[[as.character(n)]] <- out
}

qv <- qnorm(1-alpha/2)

#compile results to data frame
dat <- data.frame()
for(out in all_res) {
  temp <- data.frame(method = rep("ncv", 1000))
  for(cname in c("ho_err", "err_hat", "bias_est", "sd", "sd_infl")) {
    temp[[cname]] <- sapply(out$ncv_results, function(x){x[[cname]]})
  }
  temp[["method"]] <- "NCV"
  temp[["n"]] <- out$parameters$n
  dat <- rbind(dat, temp)
  
  temp <- data.frame(method = rep("cv", 10000))
  for(cname in c("ho_err", "err_hat", "sd")) {
    temp[[cname]] <- sapply(out$cv_results, function(x){x[[cname]]})
  }
  temp$sd_infl <- 1
  temp$bias_est <- 0
  temp[["method"]] <- "CV"
  temp[["n"]] <- out$parameters$n
  dat <- rbind(dat, temp)
  
  temp <- data.frame(method = rep("ds", length(out$ds_results)))
  for(cname in c("ho_err", "err_hat")) {
    temp[[cname]] <- sapply(out$ds_results, function(x){x[[cname]]})
  }
  temp[["sd"]] <- sapply(out$ds_results, function(x){x[["se_hat"]]}) * sqrt(out$parameters$n)
  temp[["method"]] <- "DS"
  temp[["n"]] <- out$parameters$n
  temp[["bias_est"]] <- 0
  temp[["sd_infl"]] <- 1
  dat <- rbind(dat, temp)
}

sum_dat <- dat %>%
  group_by(n, method) %>%
  summarise(
    mis_lo = mean(ho_err < err_hat + bias_est - qv * sd * sd_infl / sqrt(n)),
    mis_hi = mean(ho_err > err_hat + bias_est + qv * sd * sd_infl / sqrt(n)),
    mis_hi_deb = mean(ho_err < err_hat - qv * sd * sd_infl / sqrt(n)),
    mis_lo_deb = mean(ho_err > err_hat + qv * sd * sd_infl / sqrt(n)),
    ho_err = mean(ho_err),
    sd_infl = mean(sd_infl)
  )
sum_dat



#plot miscoverage vs n
ols_plot1 <- ggplot(sum_dat, aes(x = n, y = mis_lo_deb + mis_hi_deb, color = method, shape = method)) +
  geom_point() +
  geom_line() +
  geom_hline(yintercept = alpha, color = "darkgrey") +
  labs(y = "total miscoverage") +
  scale_x_log10(breaks = c(40, 100, 400, 1600)) +
  ylim(c(0,NA))+
  theme_bw() +
  theme(aspect.ratio = 1)
ols_plot1

ols_plot2 <- ggplot(sum_dat, aes(x = n, y = mis_lo_deb, color = method, shape = method)) +
  geom_point() +
  geom_line() +
  geom_hline(yintercept = alpha / 2, color = "darkgrey") +
  labs(y = "miscover below") +
  scale_x_log10(breaks = c(40, 100, 400, 1600)) +
  ylim(c(0,NA))+
  theme_bw() +
  theme(aspect.ratio = 1)
ols_plot2

ols_plot3 <- ggplot(sum_dat, aes(x = n, y = mis_hi_deb, color = method, shape = method)) +
  geom_point() +
  geom_line() +
  geom_hline(yintercept = alpha / 2, color = "darkgrey") +
  labs(y = "miscover above") +
  scale_x_log10(breaks = c(40, 100, 400, 1600)) +
  ylim(c(0,NA))+
  theme_bw() +
  theme(aspect.ratio = 1)
ols_plot3

library(gridExtra)
library(cowplot)

g_legend<-function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)}
mylegend<- g_legend(ols_plot1 + theme(legend.position = "bottom"))

combined_plot <- plot_grid(ols_plot1 + theme(legend.position="none"), 
                           ols_plot2 + theme(legend.position="none"), 
                           ols_plot3 + theme(legend.position="none"),
                           align = "v", ncol = 3)
combined_plot <- grid.arrange(combined_plot + theme(plot.margin = unit(c(0, 0, 0, 0), "cm")), mylegend, nrow= 2,heights=c(3, 1))
#combined_plot <- plot_grid(combined_plot, mylegend + theme(plot.margin = unit(c(0, 10, 0, 0), "cm")), ncol = 1, rel_heights = c(20, 1))
combined_plot

ggsave(combined_plot, file = "figures/ols_coverage.pdf", height = 2.5, width = 6.5)


#inflation plot
infl_plot <- ggplot(dat %>% filter(method == "NCV"), aes(x = as.factor(n), y = sd_infl)) +
  geom_violin() +
  stat_summary(fun=mean, geom="point") +
  #scale_x_log10(breaks = c(40, 100, 400, 1600)) +
  geom_hline(yintercept = 1, color = "dark grey") +
  labs(y = "NCV width",  x = "n") +
  theme_bw() +
  theme(aspect.ratio = 1)
infl_plot  
ggsave(infl_plot, file = "figures/ols_infl_violins.pdf", height = 2, width = 2.5)
