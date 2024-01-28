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
there were 250 surveys conducted. At each survey location, the response
variable is recorded, along with covariate information.

However, we assume that the surveyor does not record the first 3
covariates. Thus, there are several spatially autocorrelated covariates
that are affecting the species distribution but which cannot be (or have
not been) measured.

Below, we illustrate the survey locations across the landscape, and the
table shows the first several rows of data available for analysis:

![](README_files/figure-markdown_github/analysis_data_example-1.png)

    #>   Lat Lon            y       Cov_4      Cov_5      Cov_6      Cov_7      Cov_8
    #> 1  78  28 -1.654586111 -0.07434980 -0.4975008  0.5504506 -0.4761887  1.3343854
    #> 2   9  63  0.005181213  0.01980214  1.4102656 -1.1320113  0.9367655 -0.2116192
    #> 3  70  73  2.042104225 -1.09899390 -0.1180165 -0.8052948 -0.6945609 -0.4532229
    #> 4  16  68  1.040877953 -0.58928616  0.7220514 -0.9468051  0.1529217 -0.9904452
    #> 5  74  82 -1.371809638  0.60752793  1.4989495  0.3724967 -1.4892006  0.5367851
    #> 6  21  84  2.394133271 -0.60508809  0.5708039 -0.4035054 -0.5551169 -0.6799614
    #>          Cov_9      Cov_10       Cov_11     Cov_12     Cov_13     Cov_14
    #> 1 -1.364426879  0.04935452 -2.774279753 -1.3545507 -0.2302716 -0.4333240
    #> 2  0.638660606 -1.14063753  0.050317082 -1.2801992  0.3068397 -0.5861858
    #> 3 -0.818230034  1.38491836 -1.277603493  0.7270649  0.9784152 -0.0942090
    #> 4  0.007989822 -0.45985580 -0.356434827 -0.5282500  0.8294289  0.9441035
    #> 5 -0.655736053 -0.50861375 -0.003802338  0.7092838  0.6989671 -1.1231526
    #> 6  0.024731323  2.34740312 -0.770474678  1.6167136  2.1085738 -1.1440075
    #>         Cov_15      Cov_16      Cov_17     Cov_18     Cov_19      Cov_20
    #> 1 -0.583525750  1.34917471 -1.66396834  0.5847973 -0.1539739 -0.58154192
    #> 2 -0.107763348  0.25818662 -0.74353456  0.5365030  2.6746989  0.83873559
    #> 3 -0.001002234 -0.09315291 -0.39629246 -0.8545024 -0.3295951  1.02685304
    #> 4  0.067222462 -1.01429545  0.02836137 -1.9855292  1.2048311 -0.26715148
    #> 5  0.414624593 -1.01539276  0.49874307 -1.0336384 -0.3230452  1.02823266
    #> 6 -0.379919262  2.23204570  0.48803214 -1.0408896  1.5432933 -0.03339307
    #>        Cov_21      Cov_22     Cov_23     Cov_24     Cov_25
    #> 1  0.48740298 -0.83718607  1.5265977 -0.2681775 -0.4440738
    #> 2 -0.05881958  0.42159171 -1.3917077  1.9436308  0.3226499
    #> 3  0.57570665 -0.74510828  0.7199989 -1.6670762 -0.2112773
    #> 4  0.92978991 -0.08458897  0.1437982  0.3166882 -1.1691602
    #> 5 -0.17726519 -0.50255247  0.6737630  0.6103002 -0.2298185
    #> 6 -0.78846273 -0.19770191  0.5202787  0.6699337 -0.8201905

We first analyze the dataset with boosted regression trees, using the
following code.

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

Variable importance scores can be extracted from boosted regression tree
analyses using the `summary` command. A plot of relative variable
importance is shown below:

![](README_files/figure-markdown_github/var_importance-1.png) The top 5
most important variables (in this case Cov_5, Cov_7, Cov_8, Cov_6,
Cov_9) were then included in a Bayesian species distribution model, fit
using the `inlabru` package in R.

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

# Controls the 'residual spatial field'
prior_range <- c(1, 0.1)   # 10% chance range is smaller than 1
prior_sigma <- c(1,0.1)    # 10% chance sd is larger than 1
matern_coarse <- inla.spde2.pcmatern(mesh_spatial,
                                     prior.range = prior_range, 
                                     prior.sigma = prior_sigma
)

# Priors/shrinkage on covariate effects
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

# Mean predicted response for every pixel on the landscape
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

The RMSE for the BRT surface is 0.63, while the RMSE for the INLA
surface is 0.42. Thus, in this simulation, the RMSE from the Bayesian
model was 33.25% lower than the model fit using BRT.

Additionally, the correlation between the true surface and the
prediction from BRT was 0.96. The correlation for the Bayesian model
prediction was 0.98.

These results (as well as visual inspection of the surfaces above)
illustrate that the Bayesian model resulted in much better predictions
of the species distribution than the boosted regression tree.

## Repeated simulations

Since the result above might have been weird and/or unrepresentative, we
need to re-run the simulation many times to see how consistently a
Bayesian model improves the model fit.

We conducted 500 repeated simulations. For each simulation, we stored
the Root Mean Squared Error and correlation between model predictions
and the ‘true’ response surface (y). We stored these values separately
based on either BRT models, or Bayesian models that incorporated the
best covariates identified by BRTs.

Results are illustrated below.

![](README_files/figure-markdown_github/conduct_repeated_simulations-1.png)![](README_files/figure-markdown_github/conduct_repeated_simulations-2.png)

The Bayesian model resulted in lower RMSE for 99% of simulations.

Additionally, predictions from the Bayesian model had a higher
correlation with the true response surface in 99% of simulations.

# Conclusions

This analysis suggests that a Bayesian spatial model outperforms a
machine learning model for predicting spatial response surfaces.

In this case, we used the machine learning model to identify the best
covariates to consider including in the Bayesian model. Then, we used
the Bayesian model to capture residual spatial autocorrelation in the
response surface, beyond that which is explained by explicit covariates.

This approach holds promise for improving species distribution
estimates.
