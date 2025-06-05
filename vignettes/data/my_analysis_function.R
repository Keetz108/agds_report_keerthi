library(caret)
library(dplyr)
library(tidyr)
library(rsample)
library(recipes)
library(ggplot2)

#Comparison of the linear regression and KNN models
#1

daily_fluxes <- read.csv("data/FLX_CH-Dav_FLUXNET2015_FULLSET_DD_1997-2014_1-3.csv")   
  
# Data cleaning: looks ok, no obviously bad data
# no long tail, therefore no further target engineering
daily_fluxes |> 
  ggplot(aes(x = GPP_NT_VUT_REF, y = ..count..)) + 
  geom_histogram()

# Data splitting
set.seed(1982)  # for reproducibility
split <- rsample::initial_split(daily_fluxes, prop = 0.7, strata = "VPD_F")
daily_fluxes_train <- rsample::training(split)
daily_fluxes_test <- rsample::testing(split)

# Model and pre-processing formulation, use all variables but LW_IN_F
pp <- recipes::recipe(GPP_NT_VUT_REF ~ SW_IN_F + VPD_F + TA_F, 
                      data = daily_fluxes_train |> drop_na()) |> 
  recipes::step_BoxCox(recipes::all_predictors()) |> 
  recipes::step_center(recipes::all_numeric(), -recipes::all_outcomes()) |>
  recipes::step_scale(recipes::all_numeric(), -recipes::all_outcomes())

# Fit linear regression model
mod_lm <- caret::train(
  pp, 
  data = daily_fluxes_train |> drop_na(), 
  method = "lm",
  trControl = caret::trainControl(method = "none"),
  metric = "RMSE"
)

# Fit KNN model
mod_knn <- caret::train(
  pp, 
  data = daily_fluxes_train |> drop_na(), 
  method = "knn",
  trControl = caret::trainControl(method = "none"),
  tuneGrid = data.frame(k = 8),
  metric = "RMSE"
)

# make model evaluation into a function to reuse code
eval_model <- function(mod, df_train, df_test){
  
  # add predictions to the data frames
  df_train <- df_train |> 
    drop_na()
  df_train$fitted <- predict(mod, newdata = df_train)
  
  df_test <- df_test |> 
    drop_na()
  df_test$fitted <- predict(mod, newdata = df_test)
  
  # get metrics tables
  metrics_train <- df_train |> 
    yardstick::metrics(GPP_NT_VUT_REF, fitted)
  
  metrics_test <- df_test |> 
    yardstick::metrics(GPP_NT_VUT_REF, fitted)
  
  # extract values from metrics tables
  rmse_train <- metrics_train |> 
    filter(.metric == "rmse") |> 
    pull(.estimate)
  rsq_train <- metrics_train |> 
    filter(.metric == "rsq") |> 
    pull(.estimate)
  
  rmse_test <- metrics_test |> 
    filter(.metric == "rmse") |> 
    pull(.estimate)
  rsq_test <- metrics_test |> 
    filter(.metric == "rsq") |> 
    pull(.estimate)
  
  # visualise as a scatterplot
  # adding information of metrics as sub-titles
  plot_1 <- ggplot(data = df_train, aes(GPP_NT_VUT_REF, fitted)) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "lm", se = FALSE, color = "red") +
    geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
    labs(subtitle = bquote( italic(R)^2 == .(format(rsq_train, digits = 2)) ~~
                              RMSE == .(format(rmse_train, digits = 3))),
         title = "Training set") +
    theme_classic()
  
  plot_2 <- ggplot(data = df_test, aes(GPP_NT_VUT_REF, fitted)) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "lm", se = FALSE, color = "red") +
    geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
    labs(subtitle = bquote( italic(R)^2 == .(format(rsq_test, digits = 2)) ~~
                              RMSE == .(format(rmse_test, digits = 3))),
         title = "Test set") +
    theme_classic()
  
  out <- cowplot::plot_grid(plot_1, plot_2)
  
  return(out)
}

# linear regression model
eval_model(mod = mod_lm, df_train = daily_fluxes_train, df_test = daily_fluxes_test)


# KNN
eval_model(mod = mod_knn, df_train = daily_fluxes_train, df_test = daily_fluxes_test)


