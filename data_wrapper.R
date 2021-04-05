#harness for dispatching CV and NCV results

# parameters
# X: matrix or data frame of features (should contain many examples)
# Y: vector of responses
# fun_list: list of list of functions to run CV and NCV. functions should be named as in those functions
# n: number of random rows of X to use for training
# n_folds: number of folds in CV and NCV
# double_cv_reps: number of random splits to use for double CV
# n_cores: number of cores available for parallelization
# n_sim: number of replicates
# tag: a string to include in the output results
# export_vars: a character vector of environment variables to pass to workders
# do_ncv: whether or not to do nested CV
# do_632: whether or not to do bootstrap

# rerturn
# ncv_results: list of results of calls to nested_cv function
# cv_restuls: list of results of calls to cv function
# boot632_restuls: list of results of calls to boot632 function
# ds_ restuls: list of results of calls to data splittin function
# parameters: summary of parameters in call to the harness
ncv_simulator <- function(X, Y, fun_list, n = 100, n_folds = 10, double_cv_reps = 200, nboot = 1000, n_cores = 1, n_sim = 1000, tag = NA,
                          export_vars = NULL, funcs_params = NULL, trans = list(identity),
                          do_ncv = T, do_boot632 = T, do_cv = T) {

  #naive CV coverage simulator
  cv_fun <- function(n, n_folds, funs, funcs_params = NULL) {
    train_idx <- sample(1:nrow(X), n, replace = F)
    test_idx = setdiff(1:nrow(X), train_idx)
    X_train <- X[train_idx, ]
    Y_train <- Y[train_idx]
    
    out <- nestedcv::naive_cv(X_train, Y_train, funs, n_folds = n_folds, 
                              funcs_params = funcs_params, trans = trans)
    fit <- funs$fitter(X_train, Y_train, funcs_params = funcs_params)
    ho_err <- mean(funs$loss(funs$predictor(fit, X[test_idx, ], funcs_params = funcs_params), 
                             Y[test_idx], funcs_params = funcs_params)) 
    out[["ho_err"]] <- ho_err
    out[["type"]] <- "cv"
    out[["alg"]] <- funs$name
    
    out
  }
  
  #data splitting coverage simulator
  ds_fun <- function(n, ho_frac = .2, funs, funcs_params = NULL) {
    train_idx <- sample(1:nrow(X), n, replace = F)
    test_idx = setdiff(1:nrow(X), train_idx)
    X_train <- X[train_idx, ]
    Y_train <- Y[train_idx]
    
    #model on subset
    train_idx2 <- train_idx[1:ceiling((1-ho_frac)*n)]
    test_idx2 <- setdiff(train_idx, train_idx2)
    fit_ho <- funs$fitter(X[train_idx2, ], Y[train_idx2], funcs_params = funcs_params)
    es <- funs$loss(funs$predictor(fit_ho, X[test_idx2, ], funcs_params = funcs_params), 
                             Y[test_idx2], funcs_params = funcs_params)
    #true error, no refit
    ho_err2 <- mean(funs$loss(funs$predictor(fit_ho, X[test_idx, ], funcs_params = funcs_params), 
                              Y[test_idx], funcs_params = funcs_params)) 
    
    out <- list()
    out[["err_hat"]] <- mean(es)
    out[["se_hat"]] <- sd(es) / sqrt(length(es))
    out[["ho_err_norefit"]] <- ho_err2
    
    #model on full data
    fit <- funs$fitter(X_train, Y_train, funcs_params = funcs_params)
    ho_err <- mean(funs$loss(funs$predictor(fit, X[test_idx, ], funcs_params = funcs_params), 
                             Y[test_idx], funcs_params = funcs_params)) 
    out[["ho_err"]] <- ho_err
    out[["type"]] <- "ds"
    out[["alg"]] <- funs$name
    
    out
  }

  #nested CV coverage simulator
  nested_cv_fun <- function(n, n_folds, funs, funcs_params = NULL) {
    train_idx <- sample(1:nrow(X), n, replace = F)
    test_idx = setdiff(1:nrow(X), train_idx)
    X_train <- X[train_idx, ]
    Y_train <- Y[train_idx]
    
    out <- nestedcv::nested_cv(X_train, Y_train, funs, n_folds = n_folds, reps = double_cv_reps,
                               funcs_params = funcs_params, trans = trans, n_cores = n_cores)
    fit <- funs$fitter(X_train, Y_train, funcs_params = funcs_params)
    out[["ho_err"]] <- mean(funs$loss(funs$predictor(fit, X[test_idx, ], funcs_params = funcs_params),
                                      Y[test_idx],
                                      funcs_params = funcs_params))
    out[["type"]] <- "ncv"
    out[["alg"]] <- funs$name

    out
  }
  
  #632 coverage simulator
  boot632_fun <- function(n, funs, funcs_params = NULL) {
    train_idx <- sample(1:nrow(X), n, replace = F)
    test_idx = setdiff(1:nrow(X), train_idx)
    X_train <- X[train_idx, ]
    Y_train <- Y[train_idx]
    
    out <- list("Error")
    temp <- tryCatch({
        out <- nestedcv::boot632(X_train, Y_train, funs, nboot = nboot, funcs_params = funcs_params)
        fit <- funs$fitter(X_train, Y_train, funcs_params = funcs_params)
        out[["ho_err"]] <- mean(funs$loss(funs$predictor(fit, X[test_idx, ], funcs_params = funcs_params),
                                          Y[test_idx],
                                          funcs_params = funcs_params))
        out[["type"]] <- "boot632"
        out[["alg"]] <- funs$name
      }, error = function(err) {
        
      }
    )
    
    out
  }

  # if(n_cores > 1) {
  #   print("Starting cluster")
  #   cl <- makeForkCluster(nnodes = n_cores)
  #   registerDoParallel(cl, cores = n_cores)
  #   clusterSetRNGStream(cl = cl, 123)
  # }
  
  results_ds <- list()
  results_cv <- list()
  results_ncv <- list()
  results_boot632 <- list()
  
  for(funs in fun_list) {
    print(paste0("Starting algorithm: ", funs$name))
    
    restults_ds <- NULL
    print("Starting DS sim")
    n_reps2 <- n_sim * 10 #more reps for DS
    t1 <- Sys.time()
    results2 <- foreach(i=1:n_reps2, .export = export_vars) %dopar% {
      ds_fun(n, funs = funs, funcs_params = funcs_params)
    }
    t2 <- Sys.time()
    print(t2 - t1)
    print(Sys.time())
    results_ds <- c(results_ds, results2)
    
    results_cv <- NULL
    if(do_cv) {
      print("Starting CV sim")
      n_reps2 <- n_sim * 10 #more reps for CV
      t1 <- Sys.time()
      results2 <- foreach(i=1:n_reps2, .export = export_vars) %dopar% {
        cv_fun(n, n_folds, funs, funcs_params = funcs_params)
      }
      t2 <- Sys.time()
      print(t2 - t1)
      print(Sys.time())
      results_cv <- c(results_cv, results2)
    }
    
    results_ncv <- NULL
    if(do_ncv) {
      #nested CV coverage
      print("Starting nested CV sim")
      print(n)
      n_reps <- n_sim
      t1 <- Sys.time()
      results <- foreach(i=1:n_reps, .export = export_vars, .verbose = T, .errorhandling = "stop") %dopar% {
        nested_cv_fun(n, n_folds, funs, funcs_params = funcs_params)
      }
      t2 <- Sys.time()
      print(t2 - t1)
      print(Sys.time())
      results_ncv <- c(results_ncv, results)
    }
    
    results_boot632 <- NULL
    if(do_boot632) {
      #nested CV coverage
      print("Starting boot632 sim")
      print(n)
      n_reps <- n_sim
      t1 <- Sys.time()
      results <- foreach(i=1:n_reps, .export = export_vars, .verbose = T, .errorhandling = "stop") %dopar% {
        boot632_fun(n, funs, funcs_params = funcs_params)
      }
      t2 <- Sys.time()
      print(t2 - t1)
      print(Sys.time())
      results_boot632 <- c(results_boot632, results)
    }
  }
  
  # if(n_cores > 1) {
  #   stopCluster(cl)
  # }
  
  list("ncv_results" = results_ncv, 
       "ds_results" = results_ds,
       "cv_results" = results_cv,
       "boot632_results" = results_boot632,
       "parameters" = list("n" = n, "n_folds" = n_folds, "ncv_reps" = double_cv_reps, 
                           "n_sim" = n_sim, "tag" = tag)) 
}
