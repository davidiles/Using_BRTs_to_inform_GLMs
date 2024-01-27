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
models improves species distribution estimates, compared to just fitting
a BRT.

## Simulation example

Since almost all habitat and environmental covariates are spatially
autocorrelated, we simulated 25 spatially autocorrelated random
variables, each with different properties. Some covariates are very
noisy, some are smooth, some are highly spatially autocorrelated, and
others have little spatial autocorrelation.

The plots below illustrate each of the covariates.

In our simulations, we allow the first 10 covariates to influence the
species distribution. We draw a random ‘effect size coefficient’ for
each variable, then create the species distribution by multiplying each
spatial covariate by its effect size, and adding them together. The
resulting species distribution is shown below:

## Repeated simulations

Since the result above might have been a weird and/or unrepresentative
result, we need to re-run the simulation above many times to see how
consistently a Bayesian model improves the model fit.

``` r

simdat <- simdat %>% mutate(y = y) %>% relocate(Lat,Lon,y)

# Map of response variable
ggplot(simdat)+
  geom_raster(aes(x = Lon, y = Lat, fill = y))+
  scale_fill_gradientn(colors = viridis(10))+
  theme_bw()
```

![](README_files/figure-markdown_github/unnamed-chunk-1-1.png)

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

![](README_files/figure-markdown_github/unnamed-chunk-1-2.png)

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
#> total mean deviance =  3.2545 
#> tolerance is fixed at  0.0033 
#> ntrees resid. dev. 
#> 50    2.547 
#> now adding trees... 
#> 100   2.1223 
#> 150   1.849 
#> 200   1.6613 
#> 250   1.5265 
#> 300   1.4245 
#> 350   1.3469 
#> 400   1.2897 
#> 450   1.2425 
#> 500   1.2091 
#> 550   1.1821 
#> 600   1.1647 
#> 650   1.1527 
#> 700   1.1403 
#> 750   1.1283 
#> 800   1.1188 
#> 850   1.1114 
#> 900   1.1056 
#> 950   1.1016 
#> 1000   1.0986 
#> 1050   1.0943 
#> 1100   1.0922 
#> 1150   1.088 
#> 1200   1.0848 
#> 1250   1.0828 
#> 1300   1.0798 
#> 1350   1.0789 
#> 1400   1.0773 
#> 1450   1.0768 
#> 1500   1.0764 
#> 1550   1.0758 
#> 1600   1.0755 
#> 1650   1.0751 
#> 1700   1.073 
#> 1750   1.0706 
#> 1800   1.0682 
#> 1850   1.0677 
#> 1900   1.068 
#> 1950   1.067 
#> 2000   1.0666 
#> 2050   1.0669 
#> 2100   1.0668 
#> 2150   1.0677 
#> 2200   1.0676 
#> 2250   1.0673 
#> 2300   1.0688 
#> 2350   1.0692 
#> 2400   1.0691 
#> 2450   1.0685 
#> 2500   1.0679
```

![](README_files/figure-markdown_github/unnamed-chunk-1-3.png)

    #> 
    #> mean total deviance = 3.254 
    #> mean residual deviance = 0.124 
    #>  
    #> estimated cv deviance = 1.067 ; se = 0.083 
    #>  
    #> training data correlation = 0.983 
    #> cv correlation =  0.823 ; se = 0.011 
    #>  
    #> elapsed time -  0.21 minutes

    # predictions from brt across landscape
    pred_brt <- predict(brt, simdat,n.trees=brt$gbm.call$best.trees, type="response")

    # variable importance
    var_imp <- summary(brt)

![](README_files/figure-markdown_github/unnamed-chunk-1-4.png)

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
