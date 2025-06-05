library(dplyr)
library(tidyr)
library(caret)
library(tidyverse)

daily_fluxes <- read.csv("data/FLX_CH-Dav_FLUXNET2015_FULLSET_DD_1997-2014_1-3.csv")   

# Function to evaluate KNN for a given k
calculate_mae_for_k <- function(k, train_data, test_data, predictors, response) {
  model <- train(
    x = train_data[, predictors],
    y = train_data[[response]],
    method = "knn",
    tuneGrid = data.frame(k = k),
    trControl = trainControl(method = "none")
  )
  preds <- predict(model, newdata = test_data[, predictors])
  mae <- mean(abs(preds - test_data[[response]]))
  return(mae)
}

# Prepare data
set.seed(123)
# Assume 'daily_fluxes' is your dataset with GPP as target
df <- daily_fluxes %>% drop_na()
response <- "GPP_NT_VUT_REF"
predictors <- setdiff(names(df), c(response, "timestamp"))

# Train/test split
train_index <- createDataPartition(df[[response]], p = 0.8, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# Evaluate for multiple k values
k_values <- 1:40
mae_test <- sapply(k_values, calculate_mae_for_k, 
                   train_data = train_data,
                   test_data = test_data,
                   predictors = predictors,
                   response = response)

# Optional: also evaluate training MAE
calculate_training_mae <- function(k) {
  model <- train(
    x = train_data[, predictors],
    y = train_data[[response]],
    method = "knn",
    tuneGrid = data.frame(k = k),
    trControl = trainControl(method = "none")
  )
  preds <- predict(model, newdata = train_data[, predictors])
  mean(abs(preds - train_data[[response]]))
}

mae_train <- sapply(k_values, calculate_training_mae)

# Plotting MAE vs. k
mae_df <- tibble(k = k_values, MAE_test = mae_test, MAE_train = mae_train)

ggplot(mae_df, aes(x = k)) +
  geom_line(aes(y = MAE_test, color = "Test MAE")) +
  geom_line(aes(y = MAE_train, color = "Train MAE")) +
  labs(title = "Model Complexity vs. MAE",
       x = "Number of Neighbors (k)",
       y = "Mean Absolute Error",
       color = "Data Split") +
  theme_minimal()


