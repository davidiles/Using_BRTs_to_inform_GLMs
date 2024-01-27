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

    #>   Lat Lon          y       Cov_4       Cov_5      Cov_6      Cov_7      Cov_8
    #> 1  73  69 -0.2538863 -0.16914296 -0.24950457 -0.1436170  0.1655882  0.9537875
    #> 2  29  63 -0.7192301 -0.65447284 -1.06110775  0.9393136 -0.2473942 -0.2342259
    #> 3  11  33  1.4424011 -0.68305989 -0.84693783  1.5997349 -0.1197359 -0.0679212
    #> 4  47  53  0.4941388 -0.74372393 -0.38376979 -1.2076389 -0.4795987 -1.0962470
    #> 5   5  82 -0.7544612  0.06209212 -0.19755768 -1.4226318 -0.5171540 -1.0441971
    #> 6  48  53  0.4410513  0.60893739  0.05354572 -1.1762807 -0.3918984 -1.0045922
    #>        Cov_9     Cov_10    Cov_11     Cov_12     Cov_13     Cov_14     Cov_15
    #> 1 -1.5864093 -0.6549646 0.3557094 -0.6160459 -0.5829733 -1.0356796  0.9305570
    #> 2 -0.8161743 -0.1396460 0.3443138 -1.6036078  0.5640454 -1.5094987  0.5614001
    #> 3  2.2023648 -1.2545822 0.4948392  0.4111116 -1.3935140  0.3401511 -0.3151036
    #> 4 -0.9056128  0.8629965 1.7771041 -1.0740311  1.1662876  0.3654567  0.9313229
    #> 5 -0.9300381  0.5048646 1.8763404  0.9675379 -1.0407352 -0.1986740  1.1925720
    #> 6 -0.5598764  1.3588330 0.9326299 -0.8688453  0.7598520  0.3707100  0.9351050
    #>        Cov_16      Cov_17    Cov_18     Cov_19      Cov_20       Cov_21
    #> 1  0.53485940 -2.25936368 0.2426735  0.1745034 -0.10922768 -0.349926456
    #> 2 -0.32956232 -1.38952870 0.6642444 -0.7896025 -1.98265542 -0.007420441
    #> 3 -0.91740167 -0.43759328 1.5239995 -1.1706657  0.26876350 -0.955006740
    #> 4 -0.01998660 -0.05852607 0.2778502  1.2257487  0.06207714 -0.693116442
    #> 5 -0.09006953  1.31329887 0.1270773 -1.8047022 -1.23145002  1.558844709
    #> 6  0.38650839  0.38150338 0.5912538  0.9770911  0.30068619 -0.832365081
    #>       Cov_22     Cov_23     Cov_24     Cov_25
    #> 1 -0.3982590 -0.7933425  0.3207482 -0.8365926
    #> 2  1.0083800  0.3332268 -1.1810154  2.4431228
    #> 3 -1.0243096 -1.9324267 -1.2661103 -0.5739098
    #> 4  0.4767564 -0.4557411  0.2325089 -0.4571904
    #> 5  0.5874248 -1.2551049 -0.7659406  0.1937810
    #> 6  0.3881231 -0.7095361  0.5857132 -0.1842713

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

The top 5 most important variables (in this case Cov_9, Cov_8, Cov_10,
Cov_5, Cov_7) were then included in a Bayesian species distribution
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

The RMSE for the BRT surface is 0.55, while the RMSE for the INLA
surface is 0.33. Thus, in this simulation, the RMSE from the Bayesian
model was -40.87% lower than the model fit using BRT.

Additionally, the correlation between the true surface and the
prediction from BRT was 0.91. The correlation for the Bayesian model
prediction was 0.97.

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

    #> [1] 0.9733333
    #> [1] 0.9733333

The Bayesian model resulted in lower RMSE for 0.97% of simulations.

Additionally, predictions from the Bayesian model had a higher
correlation with the true response surface in 0.97% of simulations.

# Conclusions

This analysis suggests that a Bayesian spatial model outperforms a
machine learning model
