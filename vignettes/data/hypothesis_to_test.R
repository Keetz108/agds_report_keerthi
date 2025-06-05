
# 1. Load the dataset
daily_fluxes <- readr::read_csv("data/FLX_CH-Dav_FLUXNET2015_FULLSET_DD_1997-2014_1-3.csv")  
  
  
# 2. Split the data into training and test sets
set.seed(1982)  # For reproducibility
split <- initial_split(daily_fluxes, prop = 0.7, strata = "VPD_F")
daily_fluxes_train <- training(split)
daily_fluxes_test <- testing(split)


# 3. Preprocessing recipe: using key predictors
pp <- recipe(GPP_NT_VUT_REF ~ SW_IN_F + VPD_F + TA_F, 
             data = daily_fluxes_train |> drop_na()) |> 
  step_BoxCox(all_predictors()) |> 
  step_center(all_numeric_predictors()) |>
  step_scale(all_numeric_predictors())


# 4. Train KNN model using 10-fold cross-validation
set.seed(1982)
mod_cv <- train(pp, 
                data = daily_fluxes_train |> drop_na(), 
                method = "knn",
                trControl = trainControl(method = "cv", number = 10),
                tuneGrid = data.frame(k = c(2, 5, 10, 15, 20, 25, 30, 35, 40, 60, 100)),
                metric = "MAE")  # Evaluation metric, but Rsquared also reported


print(mod_cv)

# 5. Extract and reshape results for plotting
results_df <- mod_cv$results %>%
  select(k, MAE, Rsquared) %>%
  pivot_longer(cols = c("MAE", "Rsquared"),
               names_to = "Metric",
               values_to = "Value")


# 6. Plot MAE and R² vs k
ggplot(results_df, aes(x = k, y = Value, color = Metric)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  scale_color_manual(values = c("MAE" = "red", "Rsquared" = "blue")) +
  labs(title = "Model Complexity (k) vs. MAE and R-squared",
       x = "Number of Neighbors (k)",
       y = "Metric Value",
       color = "Metric") +
  theme_minimal()


# 7. overfitting and underfitting regions in the graph from 6

# Function to compute test set MAE for a given k
get_test_mae <- function(k_value) {
  set.seed(1982)
  
  # Create pre-processing recipe
  pp <- recipe(GPP_NT_VUT_REF ~ SW_IN_F + VPD_F + TA_F, data = daily_fluxes_train) |>
    step_BoxCox(all_predictors()) |>
    step_center(all_numeric_predictors()) |>
    step_scale(all_numeric_predictors())
  
  # Fit KNN model
  mod <- train(
    pp,
    data = daily_fluxes_train |> drop_na(),
    method = "knn",
    tuneGrid = data.frame(k = k_value),
    trControl = trainControl(method = "none")  # no CV, we want to evaluate on test set
  )
  
  # Predict on test set
  preds <- predict(mod, newdata = daily_fluxes_test |> drop_na())
  
  # True values
  actual <- daily_fluxes_test |> drop_na() |> pull(GPP_NT_VUT_REF)
  
  # Compute MAE
  mae <- yardstick::mae_vec(truth = actual, estimate = preds)
  
  return(mae)
}


# 8. Example usage

get_test_mae(5)
get_test_mae(40)

k_values <- c(2, 5, 10, 20, 40, 60, 100)
test_maes <- sapply(k_values, get_test_mae)

plot_df <- data.frame(k = k_values, Test_MAE = test_maes)

ggplot(plot_df, aes(x = k, y = Test_MAE)) +
  geom_line() +
  geom_point() +
  labs(title = "Test MAE vs. k",
       x = "k (number of neighbors)",
       y = "Test MAE")



#

k_grid <- c(2, 5, 10, 15, 20, 25, 30, 35, 40, 60, 100)

k_values <- k_grid
test_maes <- sapply(k_values, get_test_mae)

plot_df <- data.frame(k = k_values, Test_MAE = test_maes)


ggplot(plot_df, aes(x = k, y = Test_MAE)) +
  geom_line(color = "steelblue", size = 1) +
  geom_point(color = "darkred", size = 2) +
  labs(
    title = "Test MAE across different k values",
    x = "Number of Neighbors (k)",
    y = "Test MAE"
  ) +
  theme_minimal()
