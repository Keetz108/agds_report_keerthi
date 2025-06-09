# 1. Load the dataset
daily_fluxes <- readr::read_csv("data/FLX_CH-Dav_FLUXNET2015_FULLSET_DD_1997-2014_1-3.csv") |>  
  

  # select only the variables we are interested in
  dplyr::select(TIMESTAMP,
                GPP_NT_VUT_REF,    # the target
                ends_with("_QC"),  # quality control info
                ends_with("_F"),   # includes all all meteorological covariates
                -contains("JSB")   # weird useless variable
  ) |>
  
  # convert to a nice date object
  dplyr::mutate(TIMESTAMP = lubridate::ymd(TIMESTAMP)) |>
  
  # set all -9999 to NA
  dplyr::mutate(across(where(is.numeric), ~na_if(., -9999))) |> 
  
  
  # retain only data based on >=80% good-quality measurements
  # overwrite bad data with NA (not dropping rows)
  dplyr::mutate(GPP_NT_VUT_REF = ifelse(NEE_VUT_REF_QC < 0.8, NA, GPP_NT_VUT_REF),
                TA_F           = ifelse(TA_F_QC        < 0.8, NA, TA_F),
                SW_IN_F        = ifelse(SW_IN_F_QC     < 0.8, NA, SW_IN_F),
                LW_IN_F        = ifelse(LW_IN_F_QC     < 0.8, NA, LW_IN_F),
                VPD_F          = ifelse(VPD_F_QC       < 0.8, NA, VPD_F),
                PA_F           = ifelse(PA_F_QC        < 0.8, NA, PA_F),
                P_F            = ifelse(P_F_QC         < 0.8, NA, P_F),
                WS_F           = ifelse(WS_F_QC        < 0.8, NA, WS_F)) |> 
  
  # drop QC variables (no longer needed)
  dplyr::select(-ends_with("_QC"))

# Data splitting
set.seed(123)  # for reproducibility
split <- rsample::initial_split(daily_fluxes, prop = 0.7, strata = "VPD_F")
daily_fluxes_train <- rsample::training(split)
daily_fluxes_test <- rsample::testing(split)

# The same model formulation is in the previous chapter
pp <- recipes::recipe(GPP_NT_VUT_REF ~ SW_IN_F + VPD_F + TA_F, 
                      data = daily_fluxes_train) |> 
  recipes::step_center(recipes::all_numeric(), -recipes::all_outcomes()) |>
  recipes::step_scale(recipes::all_numeric(), -recipes::all_outcomes())


# make model evaluation into a function to reuse code
eval_model <- function(mod, df_train, df_test)
  
  # Remove NAs and add predictions
  df_train <- df_train |> drop_na()
  df_train$fitted <- predict(mod, newdata = df_train)
  
  df_test <- df_test |> drop_na()
  df_test$fitted <- predict(mod, newdata = df_test)
  
  # Get metrics
  metrics_train <- df_train |> 
    yardstick::metrics(GPP_NT_VUT_REF, fitted)
  
  metrics_test <- df_test |> 
    yardstick::metrics(GPP_NT_VUT_REF, fitted)
  
  # Extract MAE and R-squared
  mae_train <- metrics_train |> 
    filter(.metric == "mae") |> 
    pull(.estimate)
  rsq_train <- metrics_train |> 
    filter(.metric == "rsq") |> 
    pull(.estimate)
  
  mae_test <- metrics_test |> 
    filter(.metric == "mae") |> 
    pull(.estimate)
  rsq_test <- metrics_test |> 
    filter(.metric == "rsq") |> 
    pull(.estimate)
  
  # Visualize with scatter plots
  plot_1 <- ggplot(data = df_train, aes(GPP_NT_VUT_REF, fitted)) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "lm", se = FALSE, color = "red") +
    geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
    labs(subtitle = bquote( italic(R)^2 == .(format(rsq_train, digits = 2)) ~~
                              MAE == .(format(mae_train, digits = 3))),
         title = "Training set") +
    theme_classic()
  
  plot_2 <- ggplot(data = df_test, aes(GPP_NT_VUT_REF, fitted)) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "lm", se = FALSE, color = "red") +
    geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
    labs(subtitle = bquote( italic(R)^2 == .(format(rsq_test, digits = 2)) ~~
                              MAE == .(format(mae_test, digits = 3))),
         title = "Test set") +
    theme_classic()
  
  # Combine plots
  out <- cowplot::plot_grid(plot_1, plot_2)
  



# Assuming `mod` is your trained model
eval_model(mod, daily_fluxes_train, daily_fluxes_test)



# KNN
eval_model(mod = mod_knn, df_train = daily_fluxes_train, df_test = daily_fluxes_test)


set.seed(1982)
mod_cv <- caret::train(pp, 
                       data = daily_fluxes_train |> drop_na(), 
                       method = "knn",
                       trControl = caret::trainControl(method = "cv", number = 10),
                       tuneGrid = data.frame(k = c(2, 5, 10, 15, 20, 25, 30, 35, 40, 60, 100)),
                       metric = "MAE")



# generic plot of the caret model object
ggplot(mod_cv)



# k evaluation
df_k <- data.frame(k = c(2, 5, 10, 15, 20, 25, 30, 35, 40, 60, 100, 200, 300)) |> 
  mutate(idx = 1:n())


# model training for the specified set of K
list_mod_knn <- purrr::map(
  df_k$k,
  ~caret::train(pp, 
                data = daily_fluxes_train |> drop_na(), 
                method = "knn",
                trControl = caret::trainControl(method = "none"),
                tuneGrid = data.frame(k = .),   # '.' evaulates k
                metric = "RMSE"))

list_metrics <- purrr::map(
  list_mod_knn,
  ~eval_model(., 
              daily_fluxes_train |> drop_na(), 
              daily_fluxes_test, 
              return_metrics = TRUE))


# extract metrics on training data
list_metrics_train <- purrr::map(
  list_metrics,
  "train") |> 
  # add K to the data frame
  bind_rows(.id = "idx") |> 
  mutate(idx = as.numeric(idx)) |> 
  left_join(df_k, by = "idx")

# extract metrics on testing data
list_metrics_test <- purrr::map(
  list_metrics,
  "test") |> 
  # add K to the data frame
  bind_rows(.id = "idx") |> 
  mutate(idx = as.numeric(idx)) |> 
  left_join(df_k, by = "idx")



# 7. overfitting and underfitting regions in the graph from 6

get_test_metrics <- function(k_value) {
  set.seed(1982)
  
  # Create pre-processing recipe
  pp <- recipe(GPP_NT_VUT_REF ~ SW_IN_F + VPD_F + TA_F, data = daily_fluxes_train) |>
    step_BoxCox(all_predictors()) |>
    step_center(all_numeric_predictors()) |>
    step_scale(all_numeric_predictors())
  
  # Fit KNN model with given k
  mod <- train(
    pp,
    data = daily_fluxes_train |> drop_na(),
    method = "knn",
    tuneGrid = data.frame(k = k_value),
    trControl = trainControl(method = "none")  # no CV, just fit once
  )
  
  # Predict on test set
  preds <- predict(mod, newdata = daily_fluxes_test |> drop_na())
  
  # True values on test set
  actual <- daily_fluxes_test |> drop_na() |> pull(GPP_NT_VUT_REF)
  
  # Compute MAE and R-squared
  mae_val <- yardstick::mae_vec(truth = actual, estimate = preds)
  rsq_val <- yardstick::rsq_vec(truth = actual, estimate = preds)
  
  # Return as named vector
  c(MAE = mae_val, Rsquared = rsq_val)
}


# Define your k values to test
k_values <- c(2, 5, 10, 15, 20, 25, 30, 35, 40, 60, 100)

# Compute metrics for each k
test_metrics <- t(sapply(k_values, get_test_metrics)) |> as.data.frame()
test_metrics$k <- k_values

# Reshape for ggplot
results_test_long <- test_metrics |> 
  pivot_longer(cols = c("MAE", "Rsquared"), 
               names_to = "Metric", 
               values_to = "Value")

# Plot MAE and R-squared vs k for test data
ggplot(results_test_long, aes(x = k, y = Value, color = Metric)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  scale_color_manual(values = c("MAE" = "red", "Rsquared" = "blue")) +
  labs(title = "Test Set Performance: MAE and R-squared vs. Number of Neighbors (k)",
       x = "Number of Neighbors (k)",
       y = "Metric Value",
       color = "Metric") +
  theme_minimal()



# Step 8: Evaluate k = 40 on the test set (both MAE and R²)

# Refit model with k = 40 on training data
final_model_k40 <- train(
  pp,
  data = daily_fluxes_train |> drop_na(),
  method = "knn",
  tuneGrid = data.frame(k = 40),
  trControl = trainControl(method = "none")
)

# Predict on the test set
test_data <- daily_fluxes_test |> drop_na()
preds_k40 <- predict(final_model_k40, newdata = test_data)

# Actual values
actual_k40 <- test_data$GPP_NT_VUT_REF

# Compute metrics
mae_k40 <- yardstick::mae_vec(truth = actual_k40, estimate = preds_k40)
rsq_k40 <- yardstick::rsq_vec(truth = actual_k40, estimate = preds_k40)

# Print the results
cat("Test MAE for k = 40:", round(mae_k40, 3), "\n")
cat("Test R² for k = 40:", round(rsq_k40, 3), "\n")

ggplot(test_data, aes(x = GPP_NT_VUT_REF, y = preds_k40)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(
    title = "Observed vs. Predicted GPP (Test Set, k = 40)",
    subtitle = bquote(italic(R)^2 == .(format(rsq_k40, digits = 2)) ~~
                        MAE == .(format(mae_k40, digits = 3))),
    x = "Observed GPP",
    y = "Predicted GPP"
  ) +
  theme_minimal()



get_test_metrics <- function(k_value) {
  set.seed(1982)
  
  # Preprocessing recipe same as training
  pp <- recipe(GPP_NT_VUT_REF ~ SW_IN_F + VPD_F + TA_F, data = daily_fluxes_train) |>
    step_BoxCox(all_predictors()) |>
    step_center(all_numeric_predictors()) |>
    step_scale(all_numeric_predictors())
  
  # Train model with this k
  mod <- train(
    pp,
    data = daily_fluxes_train |> drop_na(),
    method = "knn",
    tuneGrid = data.frame(k = k_value),
    trControl = trainControl(method = "none")  # no CV, fit once
  )
  
  # Predict on test set
  preds <- predict(mod, newdata = daily_fluxes_test |> drop_na())
  
  # Actual values
  actual <- daily_fluxes_test |> drop_na() |> pull(GPP_NT_VUT_REF)
  
  # Compute metrics
  mae_val <- yardstick::mae_vec(truth = actual, estimate = preds)
  rsq_val <- yardstick::rsq_vec(truth = actual, estimate = preds)
  
  # Return as named vector
  c(MAE = mae_val, Rsquared = rsq_val)
  
  
  
  
  ggplot(results_test_long, aes(x = k, y = Value, color = Metric)) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    scale_color_manual(values = c("MAE" = "red", "Rsquared" = "blue")) +
    labs(title = "Test Set Performance (k) vs. MAE and R-squared",
         x = "Number of Neighbors (k)",
         y = "Metric Value",
         color = "Metric") +
    theme_minimal()
  
