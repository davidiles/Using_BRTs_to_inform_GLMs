# Motivation

We want to test if Machine Learning methods and Bayesian models can be
used in combination to produce better estimates of species
distributions. Our approach is to:

1.  Fit Boosted Regression Trees (BRTs) to a dataset, which identify the
    most important covariates for predicting species distributions, then

2.  Use the most important covariates in a Bayesian species distribution
    model that allows us to propagage uncertainty, and simultaneously
    accounts for spatial autocorrelation, random effects, and
    informative priors.

This approach may be an efficient way to “let the data speak for itself”
in identifying the most important covariates through machine learning,
after which we can use Bayesian models to properly account for spatial
autocorrelation and error propagation.

Our main concern is whether this is a form of “double-dipping” from the
data that will lead to over-fit models that reduce performance.

# Background

Boosted regression tree (BRT) approaches, and machine learning more
generally, are extremely good at identifying important covariates for
predicting species distributions. They naturally accommodate complex,
non-linear, and interacting response functions among multiple
covariates, and do not suffer from problems with variable collinearity.

However, BRTs cannot include spatial covariation (when the response
variable at a location is similar to the response variable at nearby
locations, after accounting for covariates). Additionally, BRTs are not
well-suited to the inclusion of random effects (e.g., repeated-measures
data), and cannot include integrated models (e.g., where multiple
response variables with different error distributions are affected by a
shared process). It can also be difficult to properly propagate
uncertainty with BRTs, which is critical for species status and trend
assessments.

In contrast, Bayesian models are excellent for describing and
propagating multiple sources of uncertainty. They can also account for
spatial autocorrelation, and can include ‘informative priors’ to help
improve model outputs. However, Bayesian models typically cannot
accommodate large numbers of covariates, and can suffer from lack of
parameter identifiability when multiple covariates are correlated with
each other.

# Methods

We used simulations to examine whether a two-stage approach to fitting
models (BRTs to identify covariates, followed by a Bayesian model to
properly account for uncertainty) improves species distribution
estimates, compared to just fitting a BRT.

## Simulation example

``` r

# ----------------------------------------
# Simulate landscape with 25 spatially autocorrelated covariates
# ----------------------------------------
  
  n_var <- 25
  simdat <- expand.grid(Lat = seq(0,100), Lon = seq(0,100))
  
  for (i in 1:n_var){
    grid <- list(x = seq(0,100), y = seq(0,100))
    obj1 <- matern.image.cov(grid=grid, aRange = runif(1,1,10) , smoothness = runif(1,0.5,1), setup=TRUE)
    simcov <- sim.rf(obj1) %>% reshape2::melt()
    simdat <- cbind(simdat,simcov$value)
  }
  
  colnames(simdat)[3:ncol(simdat)] <- paste0("Cov_",1:n_var)
  
  # ----------------------------------------
  # Simulate response variable, which depends on first 10 covariates
  # ----------------------------------------
  
  # Covariate effects
  coefs <- rep(0,n_var)
  coefs[1:10] <- runif(10,-1,1) # Beta coefficients for first 10 covariates
  coefs <- matrix(coefs,ncol=1)
  
  # Response variable
  y <- as.matrix(simdat[,3:ncol(simdat)]) %*% coefs
  
  simdat <- simdat %>% mutate(y = y) %>% relocate(Lat,Lon,y)
  
  # Map of response variable
  ggplot(simdat)+
    geom_raster(aes(x = Lon, y = Lat, fill = y))+
    scale_fill_gradientn(colors = viridis(10))+
    theme_bw()
```

![](README_files/figure-markdown_github/simulation_example-1.png)

``` r
  
  # Assume we cannot measure three covariates (drop them from dataframe)
  simdat <- simdat %>% dplyr::select(-Cov_1,-Cov_2,-Cov_3)
  
  # ----------------------------------------
  # Select 500 survey locations
  # ----------------------------------------
  
  n_survey <- 500
  
  dat <- sample_n(simdat, n_survey, replace = TRUE)
  
  # Add observation error
  dat$y <- dat$y + rnorm(nrow(dat),0,0.1)
  
  ggplot(simdat)+
    geom_raster(aes(x = Lon, y = Lat, fill = y))+
    scale_fill_gradientn(colors = viridis(10))+
    geom_jitter(data = dat, aes(x = Lon, y = Lat))+
    theme_bw()
```

![](README_files/figure-markdown_github/simulation_example-2.png)

``` r
  
  # ---------------------------------------
  # Fit BRT and generate landscape predictions
  # ---------------------------------------
  
  brt <- gbm.step(data=dat, gbm.x = 4:ncol(dat), gbm.y = 3,
                  family = "gaussian", tree.complexity = 5,
                  learning.rate = 0.01, bag.fraction = 0.5)
#> 
#>  
#>  GBM STEP - version 2.9 
#>  
#> Performing cross-validation optimisation of a boosted regression tree model 
#> for y and using a family of gaussian 
#> Using 500 observations and 22 predictors 
#> creating 10 initial models of 50 trees 
#> 
#>  folds are unstratified 
#> total mean deviance =  3.8086 
#> tolerance is fixed at  0.0038 
#> ntrees resid. dev. 
#> 50    2.6081 
#> now adding trees... 
#> 100   1.8813 
#> 150   1.4249 
#> 200   1.1131 
#> 250   0.8991 
#> 300   0.7528 
#> 350   0.6426 
#> 400   0.5605 
#> 450   0.5006 
#> 500   0.4557 
#> 550   0.4212 
#> 600   0.3972 
#> 650   0.3762 
#> 700   0.3605 
#> 750   0.3488 
#> 800   0.3402 
#> 850   0.3332 
#> 900   0.3271 
#> 950   0.3217 
#> 1000   0.3179 
#> 1050   0.3142 
#> 1100   0.3114 
#> 1150   0.3088 
#> 1200   0.3061 
#> 1250   0.3042 
#> 1300   0.3026 
#> 1350   0.3015 
#> 1400   0.2993 
#> 1450   0.2977 
#> 1500   0.2962 
#> 1550   0.2945 
#> 1600   0.2933 
#> 1650   0.2922 
#> 1700   0.2916 
#> 1750   0.2911 
#> 1800   0.2905 
#> 1850   0.2896 
#> 1900   0.2891 
#> 1950   0.2886 
#> 2000   0.2885 
#> 2050   0.2875 
#> 2100   0.2872 
#> 2150   0.2866 
#> 2200   0.2863 
#> 2250   0.2859 
#> 2300   0.2852 
#> 2350   0.2851 
#> 2400   0.2846 
#> 2450   0.284 
#> 2500   0.2838 
#> 2550   0.2835 
#> 2600   0.2831 
#> 2650   0.2827 
#> 2700   0.2822 
#> 2750   0.2821 
#> 2800   0.2823 
#> 2850   0.2821
```

![](README_files/figure-markdown_github/simulation_example-3.png)

    #> 
    #> mean total deviance = 3.809 
    #> mean residual deviance = 0.012 
    #>  
    #> estimated cv deviance = 0.282 ; se = 0.02 
    #>  
    #> training data correlation = 0.998 
    #> cv correlation =  0.964 ; se = 0.003 
    #>  
    #> elapsed time -  0.25 minutes
      
      # predictions from brt across landscape
      pred_brt <- predict(brt, simdat,n.trees=brt$gbm.call$best.trees, type="response")
      
      # variable importance
      var_imp <- summary(brt)

![](README_files/figure-markdown_github/simulation_example-4.png)

``` r
  
  # ---------------------------------------
  # Fit model using INLA
  # ---------------------------------------
  
  # USE TOP 5 MOST IMPORTANT VARIABLES FROM BRT
  top_vars <- var_imp$var[1:5]
  
  # covert data to spatial object
  simdat_sf <- st_as_sf(simdat, coords = c("Lon","Lat"),remove = FALSE)
  dat_sf <- st_as_sf(dat, coords = c("Lon","Lat"),remove = FALSE)
  
  # make a two extension hulls and mesh for spatial model
  hull <- fm_extensions(
    simdat_sf
  )
  
  # Spatial mesh
  mesh_spatial <- fm_mesh_2d_inla(
    boundary = hull, 
    max.edge = c(5, 10),
    cutoff = 2
  )
  
  # Controls the 'residual spatial field'.  This can be adjusted to create smoother surfaces.
  prior_range <- c(1, 0.1)   # 10% chance range is smaller than 1
  prior_sigma <- c(1,0.1)    # 10% chance sd is larger than 1
  matern_coarse <- inla.spde2.pcmatern(mesh_spatial,
                                       prior.range = prior_range, 
                                       prior.sigma = prior_sigma
  )
  
  # How much shrinkage should be applied to covariate effects?
  sd_linear <- 0.1  
  prec_linear <-  c(1/sd_linear^2,1/(sd_linear/2)^2)
  
  # Model formula
  model_components = as.formula(paste0('~
            Intercept(1)+
            spde_coarse(main = geometry, model = matern_coarse)+',
            paste0("Beta1_",top_vars,'(1,model="linear", mean.linear = 0, prec.linear = ', prec_linear[1],')', collapse = " + ")))
  
  model_formula= as.formula(paste0('y ~
                  Intercept +
                  spde_coarse +',
                  paste0("Beta1_",top_vars,'*',top_vars, collapse = " + ")))
  
  fit_INLA <- NULL
  while(is.null(fit_INLA)){
    
    fit_model <- function(){
      tryCatch(expr = {bru(components = model_components,
                           like(family = "gaussian",
                                formula = model_formula,
                                data = dat_sf),
                           
                           options = list(control.compute = list(waic = FALSE, cpo = FALSE),
                                          bru_verbose = 4))},
               error = function(e){NULL})
    }
    fit_INLA <- fit_model()
    
    if ("try-error" %in% class(fit_INLA)) fit_INLA <- NULL
  }
  
  # Prediction
  pred_formula = as.formula(paste0(' ~
                  Intercept +
                  spde_coarse +',paste0("Beta1_",top_vars,'*',top_vars, collapse = " + ")))
  
  # Note that predictions are initially on log scale
  pred_inla <- generate(fit_INLA,
                        simdat_sf,
                        formula =  pred_formula,
                        n.samples = 1000)
  
  pred_mean_inla <- apply(pred_inla,2,mean)
  
  pred_inla <- apply(pred_inla,1,mean)
  
  
```

``` r
results <- data.frame()

for (rep in 1:150){
  
  # Load if file already exists
  if (file.exists("output/results.RDS")) results <- readRDS("output/results.RDS")
  
  if (nrow(results)>=150) break
  
  # ----------------------------------------
  # Simulate landscape with 25 spatially autocorrelated covariates
  # ----------------------------------------
  
  n_var <- 25
  simdat <- expand.grid(Lat = seq(0,100), Lon = seq(0,100))
  
  for (i in 1:n_var){
    grid <- list(x = seq(0,100), y = seq(0,100))
    obj1 <- matern.image.cov(grid=grid, aRange = runif(1,1,10) , smoothness = runif(1,0.5,1), setup=TRUE)
    simcov <- sim.rf(obj1) %>% reshape2::melt()
    simdat <- cbind(simdat,simcov$value)
  }
  
  colnames(simdat)[3:ncol(simdat)] <- paste0("Cov_",1:n_var)
  
  # ----------------------------------------
  # Simulate response variable, which depends on first 10 covariates
  # ----------------------------------------
  
  # Covariate effects
  coefs <- rep(0,n_var)
  coefs[1:10] <- runif(10,-1,1) # Beta coefficients for first 10 covariates
  coefs <- matrix(coefs,ncol=1)
  
  # Response variable
  y <- as.matrix(simdat[,3:ncol(simdat)]) %*% coefs
  
  simdat <- simdat %>% mutate(y = y) %>% relocate(Lat,Lon,y)
  
  # Map of response variable
  ggplot(simdat)+
    geom_raster(aes(x = Lon, y = Lat, fill = y))+
    scale_fill_gradientn(colors = viridis(10))+
    theme_bw()
  
  # Assume we cannot measure three covariates (drop them from dataframe)
  simdat <- simdat %>% dplyr::select(-Cov_1,-Cov_2,-Cov_3)
  
  # ----------------------------------------
  # Select 500 survey locations
  # ----------------------------------------
  
  n_survey <- 500
  
  dat <- sample_n(simdat, n_survey, replace = TRUE)
  
  # Add observation error
  dat$y <- dat$y + rnorm(nrow(dat),0,0.1)
  
  ggplot(simdat)+
    geom_raster(aes(x = Lon, y = Lat, fill = y))+
    scale_fill_gradientn(colors = viridis(10))+
    geom_jitter(data = dat, aes(x = Lon, y = Lat))+
    theme_bw()
  
  # ---------------------------------------
  # Fit BRT and generate landscape predictions
  # ---------------------------------------
  
  brt <- gbm.step(data=dat, gbm.x = 4:ncol(dat), gbm.y = 3,
                  family = "gaussian", tree.complexity = 5,
                  learning.rate = 0.01, bag.fraction = 0.5)
  
  # predictions from brt across landscape
  pred_brt <- predict(brt, simdat,n.trees=brt$gbm.call$best.trees, type="response")
  
  # variable importance
  var_imp <- summary(brt)
  
  # ---------------------------------------
  # Fit model using INLA
  # ---------------------------------------
  
  # USE TOP 5 MOST IMPORTANT VARIABLES FROM BRT
  top_vars <- var_imp$var[1:5]
  
  # covert data to spatial object
  simdat_sf <- st_as_sf(simdat, coords = c("Lon","Lat"),remove = FALSE)
  dat_sf <- st_as_sf(dat, coords = c("Lon","Lat"),remove = FALSE)
  
  # make a two extension hulls and mesh for spatial model
  hull <- fm_extensions(
    simdat_sf
  )
  
  # Spatial mesh
  mesh_spatial <- fm_mesh_2d_inla(
    boundary = hull, 
    max.edge = c(5, 10),
    cutoff = 2
  )
  
  # Controls the 'residual spatial field'.  This can be adjusted to create smoother surfaces.
  prior_range <- c(1, 0.1)   # 10% chance range is smaller than 1
  prior_sigma <- c(1,0.1)    # 10% chance sd is larger than 1
  matern_coarse <- inla.spde2.pcmatern(mesh_spatial,
                                       prior.range = prior_range, 
                                       prior.sigma = prior_sigma
  )
  
  # How much shrinkage should be applied to covariate effects?
  sd_linear <- 0.1  
  prec_linear <-  c(1/sd_linear^2,1/(sd_linear/2)^2)
  
  # Model formula
  model_components = as.formula(paste0('~
            Intercept(1)+
            spde_coarse(main = geometry, model = matern_coarse)+',
            paste0("Beta1_",top_vars,'(1,model="linear", mean.linear = 0, prec.linear = ', prec_linear[1],')', collapse = " + ")))
  
  model_formula= as.formula(paste0('y ~
                  Intercept +
                  spde_coarse +',
                  paste0("Beta1_",top_vars,'*',top_vars, collapse = " + ")))
  
  fit_INLA <- NULL
  while(is.null(fit_INLA)){
    
    fit_model <- function(){
      tryCatch(expr = {bru(components = model_components,
                           like(family = "gaussian",
                                formula = model_formula,
                                data = dat_sf),
                           
                           options = list(control.compute = list(waic = FALSE, cpo = FALSE),
                                          bru_verbose = 4))},
               error = function(e){NULL})
    }
    fit_INLA <- fit_model()
    
    if ("try-error" %in% class(fit_INLA)) fit_INLA <- NULL
  }
  
  # Prediction
  pred_formula = as.formula(paste0(' ~
                  Intercept +
                  spde_coarse +',paste0("Beta1_",top_vars,'*',top_vars, collapse = " + ")))
  
  # Note that predictions are initially on log scale
  pred_inla <- generate(fit_INLA,
                        simdat_sf,
                        formula =  pred_formula,
                        n.samples = 1000)
  
  pred_mean_inla <- apply(pred_inla,2,mean)
  
  pred_inla <- apply(pred_inla,1,mean)
  
  # ---------------------------------------
  # Evaluate quality of model fits; save in results dataframe
  # ---------------------------------------
  
  RMSE_brt <- sqrt(mean((pred_brt - simdat$y)^2))
  RMSE_inla <- sqrt(mean((pred_inla - simdat$y)^2))
  
  results <- rbind(results,data.frame(RMSE_brt = RMSE_brt,
                                      RMSE_inla = RMSE_inla,
                                      cor_brt = cor(pred_brt,simdat$y),
                                      cor_inla = cor(pred_inla,simdat$y)
  ))
  
  # Save results
  saveRDS(results,"output/results.RDS")
  
}

# ---------------------------------------
# Load results
# ---------------------------------------

results <- readRDS("output/results.RDS")

# ---------------------------------------
# Summarize results; Are GLM predictions better than BRTs?
# ---------------------------------------
results$simulation_number <- 1:nrow(results)

# How much does GAM reduce RMSE?
results$percent_reduction_RMSE <- 100*(results$RMSE_inla - results$RMSE_brt)/results$RMSE_brt
  
RMSE_plot <- ggplot(data = results, aes(x = percent_reduction_RMSE))+
  geom_histogram(fill = "dodgerblue")+
  geom_vline(xintercept = 0, linetype = 2)+
  ylab("Frequency\n(number of simulations)")+
  xlab("Percent reduction in Root Mean Squared Error\n(When fitting INLA after a BRT)")+
  ggtitle("Does INLA reduce Root Mean Squared Error, compared to BRT?")+
  theme_bw()
RMSE_plot
```

![](README_files/figure-markdown_github/conduct_repeated_simulations-1.png)

``` r

cor_plot <- ggplot(data = results, aes(x = cor_inla - cor_brt))+
  geom_histogram(fill = "dodgerblue")+
  geom_vline(xintercept = 0, linetype = 2)+
  ylab("Frequency\n(number of simulations)")+
  xlab("Improvement in Correlation\n(When fitting INLA after a BRT)")+
  ggtitle("Does INLA improve correlation with 'true' density, compared to BRT?")+
  theme_bw()
cor_plot
```

![](README_files/figure-markdown_github/conduct_repeated_simulations-2.png)

``` r

# Percent of simulations where INLA resulted in a lower RMSE:
mean(results$RMSE_inla < results$RMSE_brt) # 0.97
#> [1] 0.9733333

# Percent of simulations where INLA resulted in a higher correlation:
mean(results$cor_inla > results$cor_brt) # 0.97
#> [1] 0.9733333
```
