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

    #>   Lat Lon          y      Cov_4       Cov_5       Cov_6       Cov_7      Cov_8
    #> 1  37  63  0.5013765  0.0484735  0.91672228 -0.19278552  1.28318749  0.3451192
    #> 2  95  39  3.8836887  0.3211503  0.98034182  0.71627686 -0.42615957  1.9318356
    #> 3  37  58 -0.4299123  1.8819539 -0.03973159  0.55899162  1.06866656 -0.3795945
    #> 4  49  81  0.1747588  1.0542864  0.66102782 -1.68547570  1.25568176  0.4802736
    #> 5  11  78  0.2713122 -0.4961796 -0.19733051  0.08498255  0.07833561  0.9647041
    #> 6  23  24 -1.5573140  1.6301771 -0.19742650  1.18198604  1.52050155 -0.9919384
    #>        Cov_9     Cov_10     Cov_11      Cov_12     Cov_13     Cov_14
    #> 1  0.6689385 -0.2520693 -1.3880632 -0.66988743 -2.4338229 -0.6196217
    #> 2 -1.2107203 -0.8733225 -0.5119556  1.58502343 -0.1682873 -0.6361183
    #> 3  0.5868350  0.4863558 -1.2933657 -0.32220352 -0.1204812  0.8270296
    #> 4  1.1623480 -0.7279297  1.2199893 -0.77293112  0.4553692  0.6689714
    #> 5  0.7729302  0.1640673 -0.1231039  0.06361641  1.1639337  0.7103155
    #> 6  0.9097158 -1.1592423 -1.8998907 -1.91608930  1.3870908  1.0318110
    #>        Cov_15     Cov_16     Cov_17      Cov_18     Cov_19      Cov_20
    #> 1 -1.01421525 -1.6036304  0.3018177  0.15861885  0.9469076  0.55148793
    #> 2 -0.52843938  0.9769187  0.8495705  0.82415212 -0.8791933  1.34989805
    #> 3 -0.49255969 -1.9903224 -0.3468766  1.36533404  1.1645482 -0.32265523
    #> 4 -1.31861794 -0.3795740 -1.1232444 -0.03694234  1.1533796 -0.76158481
    #> 5  0.33750643  0.3566984  0.4331472  1.19847849 -1.3513725  0.16903213
    #> 6 -0.06930553  0.3841566  1.1760243  0.64999980  0.3755167 -0.04599694
    #>        Cov_21     Cov_22     Cov_23      Cov_24        Cov_25
    #> 1 -0.15567536  0.8030043 1.23901072 -0.05506202 -0.3642275008
    #> 2  1.10941288  0.1033714 0.07454803 -1.45522370 -1.4872738755
    #> 3  0.86553615  1.1610863 1.03154894 -0.55585866 -0.2862231471
    #> 4  1.20154215  2.6673588 1.39874736  0.48103810 -0.8724692399
    #> 5 -0.45419981  0.8162057 2.20369481 -2.67218827 -0.3548902816
    #> 6 -0.08299207 -0.1941776 1.18069725  0.94158401 -0.0007718658

We first analyze the dataset with boosted regression trees, using the
following code. We also plot variable importance.

![](README_files/figure-markdown_github/fit_brt-1.png)

The top 5 most important variables (in this case Cov_5, Cov_7, Cov_8,
Cov_6, Cov_9) were then included in a Bayesian species distribution
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

We conducted 150 simulations. For each simulation, we stored the Root
Mean Squared Error and correlation between model predictions and the
‘true’ response surface (y).

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
