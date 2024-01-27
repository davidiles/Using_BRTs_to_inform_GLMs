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
autocorrelation and error propagation. On the other hand, this may be a
form of “double-dipping” from the data that will lead to over-fit models
that reduce performance.

We therefore conducted simulations to evaluate if this is a reasonable
approach for modeling.

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

![](README_files/figure-markdown_github/simulate_landscapes-1.png)

In our simulations, we allow the first 10 covariates to influence the
species distribution. We draw a random ‘effect size coefficient’ for
each variable, then create the species distribution by multiplying each
spatial covariate by its effect size, and adding them together. The
resulting species distribution is shown below:

![](README_files/figure-markdown_github/species_distribution-1.png)

We then simulate a survey of the species across the landscape. We assume
there were 500 surveys conducted. At each survey location, the response
variable is recorded, along with covariate information.

*However*, we assume that the surveyor does not record the first 3
covariates. Thus, there are several spatially autocorrelated covariates
that are affecting the species distribution but which cannot be (or have
not been) measured. Below, we illustrate the survey locations across the
landscape, and the table shows the first several rows of data available
for analysis:

![](README_files/figure-markdown_github/analysis_data_example-1.png)

    #>   Lat Lon          y        Cov_4      Cov_5      Cov_6       Cov_7      Cov_8
    #> 1   2  66  2.0495142 -0.901953694 -1.4985131  0.2341379 -0.03985908 -0.4821705
    #> 2  48  47 -0.5083888  0.004285082  0.8172039  0.2998487  0.78643462 -0.4229501
    #> 3  53   9  3.0750485  0.648549641 -0.0804429  0.5816619  0.53370855 -1.5089465
    #> 4  71   9 -2.5447837 -0.563256878  1.2399145 -0.2715009 -0.50794293  0.3869722
    #> 5  96  69  0.9069443  0.590791108 -0.5961984 -1.1733251 -0.64520841  0.4809468
    #> 6  85  50  0.7054334  0.502205997  0.2020600 -0.1868524 -0.71287402  0.8358009
    #>        Cov_9     Cov_10     Cov_11     Cov_12     Cov_13     Cov_14     Cov_15
    #> 1 -1.1860667 -0.3303689 -2.3267907  1.6707094  0.1581712  1.1997851 -1.3162588
    #> 2 -0.4837238  0.4692777  0.7606698 -1.4944769 -0.5985132  0.1899987  0.2776687
    #> 3  0.3458464  0.3804128 -0.9161369  0.3545537 -0.4932136 -1.4562203  0.7007146
    #> 4 -0.5331179 -1.0022800 -0.2475285  0.2923376 -1.6176345  2.0783722  0.4759553
    #> 5 -1.4586465 -0.7328391  0.3889186  2.0949696  1.2264297  0.3911868  0.9489299
    #> 6 -0.9678242 -0.2898641 -0.6905166  0.3528309  1.3029215  0.5209788  0.7689862
    #>       Cov_16     Cov_17     Cov_18      Cov_19        Cov_20     Cov_21
    #> 1  0.2811616  1.4789564  0.1886540  0.09012279  0.0240304369 -0.2294860
    #> 2  1.7296375 -0.8177915 -1.1860870  1.65244258 -0.0323000238 -0.7454279
    #> 3 -1.0365798  1.5916521  0.1645963 -0.32312816 -0.5002041507 -0.2718537
    #> 4 -0.3975367 -1.0308267 -0.5202737 -1.41655229 -0.7373499325 -0.9779456
    #> 5  0.6245485 -0.3242176  0.2187872  0.79536795 -0.0008460594  0.2086748
    #> 6  0.4789907  0.4195511 -1.0187512 -1.68531919 -2.1020883008  0.1844929
    #>        Cov_22      Cov_23      Cov_24     Cov_25
    #> 1 -0.88275848 -0.02194099  0.02022157 -0.1894621
    #> 2 -0.51343002 -0.87629320 -0.40628747  0.5247128
    #> 3 -0.08751978 -0.55407738 -1.06988316 -0.4459445
    #> 4 -0.16506843 -1.88513974 -0.27011804 -0.2635481
    #> 5  2.04258900  1.61786152  0.77887162 -0.2015537
    #> 6  0.69133618  0.66022416  0.57504003 -0.7379946

We first analyze the dataset with boosted regression trees, using the
following code. We also plot variable importance.

``` r

# ---------------------------------------
# Fit BRT and generate landscape predictions
# ---------------------------------------

brt <- gbm.step(data=dat, gbm.x = 4:ncol(dat), gbm.y = 3,
                family = "gaussian", tree.complexity = 5,
                learning.rate = 0.01, bag.fraction = 0.5,
                verbose = FALSE,
                plot.main = FALSE)

# generate predictions from brt across landscape
pred_brt <- predict(brt, simdat,n.trees=brt$gbm.call$best.trees, type="response")

# variable importance
var_imp <- summary(brt)
```

![](README_files/figure-markdown_github/fit_brt-1.png)

The top 5 most important variables (in this case Cov_8, Cov_4, Cov_6,
Cov_10, Cov_7) were then included in a Bayesian species distribution
model, fit using the `inlabru` package in R.

The model includes a spatially autocorreled random field to account for
spatial autocorrelation that is not attributable to the measured
covariates.

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
hull <- fm_extensions(simdat_sf)

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

pred_inla <- generate(fit_INLA,
                      simdat_sf,
                      formula =  pred_formula,
                      n.samples = 1000)

# Predicted response for every pixel on the landscape
pred_inla <- apply(pred_inla,1,mean)
```

Below, we compare the ‘true’ response surface (y) with the predicted
response surfaces estimated from either Boosted Regression Trees (BRTs)
or with the Bayesian model (INLA).

![](README_files/figure-markdown_github/compare_BRT_INLA-1.png)

To evaluate which prediction surface is a better representation of the
‘true’ response surface on the left, we calculate two metrics: 1) the
root mean squared error between the prediction and the true surface, and
2) the correlation between the prediction surface and the true surface.

``` r
RMSE_brt <- sqrt(mean((simdat$pred_brt - simdat$y)^2))
RMSE_inla <- sqrt(mean((simdat$pred_inla - simdat$y)^2))
cor_brt <- cor(simdat$pred_brt,simdat$y)
cor_inla <- cor(simdat$pred_inla,simdat$y)
```

The RMSE for the BRT surface is 0.61, while the RMSE for the INLA
surface is 0.54. Thus, in this simulation, the RMSE from the Bayesian
model was 10.42% lower than the model fit using BRT.

Additionally, the correlation between the true surface and the
prediction from BRT was 0.94. The correlation for the Bayesian model
prediction was 0.95.

These results (as well as visual inspection of the surfaces above)
illustrate that the Bayesian model resulted in much better predictions
of the species distribution than the boosted regression tree.

## Repeated simulations

Since the result above might have been weird and/or unrepresentative, we
need to re-run the simulation many times to see how consistently a
Bayesian model improves the model fit.

We conducted 150 simulations. For each simulation, we stored the Root
Mean Squared Error and correlation between model predictions and the
‘true’ response surface (y).

Results are illustrated below.

![](README_files/figure-markdown_github/conduct_repeated_simulations-1.png)![](README_files/figure-markdown_github/conduct_repeated_simulations-2.png)

The Bayesian model resulted in lower RMSE for 97% of simulations.

Additionally, predictions from the Bayesian model had a higher
correlation with the true response surface in 97% of simulations.

# Conclusions

This analysis suggests that a Bayesian spatial model outperforms a
machine learning model for predicting spatial response surfaces.

In this case, we used the machine learning model to identify the best
covariates to consider including in the Bayesian model. Then, we used
the Bayesian model to capture residual spatial autocorrelation in the
response surface, beyond that which is explained by explicit covariates.

This approach holds promise for improving species distribution
estimates.
