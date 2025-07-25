# modeling/model_selection.R
# model selection and evaluation

# Load libraries
library(tidyverse)
library(rsample)     
library(yardstick)   
library(xgboost)     
library(pROC)
library(PRROC)
library(caret)       
library(Matrix)
library(MLmetrics)
library(themis)
library(Metrics)

# Load feature engineering function and generate data
source("utilities/load_fg_data.R")
source("utilities/helpers.R")
source("utilities/feature_engineering.R")

# load data from feature engineering script
data <- prepare_fg_features(season_start = 2013, season_end = 2024)

# summarize dummy variables ... how many are zeros
dummy_vars_summary <- data %>%
  summarise(fg_made = sum(fg_made, na.rm = TRUE) / n(),
            Indoor = sum(Indoor, na.rm = TRUE) / n(),
            Turf = sum(Turf, na.rm = TRUE) / n(),
            Altitude = sum(Altitude, na.rm = TRUE) / n(),
            Precip = sum(Precip, na.rm = TRUE) / n(),
            night_game = sum(night_game, na.rm = TRUE) / n(),
            high_leverage = sum(high_leverage, na.rm = TRUE) / n(),
            cold = sum(cold, na.rm = TRUE) / n(),
            hot = sum(hot, na.rm = TRUE) / n(),
            humid_game = sum(humid_game, na.rm = TRUE) / n(),
            moderate_wind = sum(moderate_wind, na.rm = TRUE) / n(),
            high_wind = sum(high_wind, na.rm = TRUE) / n())


# remove leverage stats since we will not be using
fg_df <- data %>%
  select(-high_leverage, -playoffs)

#select variables to use in model testing
model_data <- fg_df %>%
  select(play_id, game_id, season, posteam, fg_made, kick_distance, windy_day, Indoor:away_game, cold:humid_game, wind_speed_x_kick_distance, cold_x_wind)

# using data from 2008 to 2023 to build our model
model_data <- model_data %>%
  filter(season >= 2008 & season <= 2023)

model_data_clean <- model_data %>%
  drop_na() %>%
  mutate(fg_made = factor(fg_made, levels = c(0, 1), labels = c("no", "yes")))

# train and split data
set.seed(42)
split <- initial_split(model_data_clean, prop = 0.8, strata = fg_made)
train_data <- training(split)
test_data  <- testing(split)

train_x <- train_data %>% select(-fg_made, -play_id, -game_id, -posteam, -season)
train_y <- train_data$fg_made
test_x  <- test_data %>% select(-fg_made, -play_id, -game_id, -posteam)
test_y  <- test_data$fg_made
train_y <- factor(train_y, levels = c("no", "yes"))
test_y  <- factor(test_y, levels = c("no", "yes"))


# Control and Grid Setup
control <- trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = TRUE,
  allowParallel = TRUE
)

xgb_grid <- expand.grid(
  nrounds = 100,
  max_depth = c(3, 5),
  eta = 0.1,
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = c(1, 5),
  subsample = 0.8
)

# Train Models
# Logistic Regression (no SMOTE)
log_model <- train(
  x = train_x,
  y = train_y,
  method = "glm",
  family = "binomial",
  metric = "ROC",
  trControl = control
)

# Logistic Regression (with SMOTE)
control_smote <- control
control_smote$sampling <- "smote"

log_model_smote <- train(
  x = train_x,
  y = train_y,
  method = "glm",
  family = "binomial",
  metric = "ROC",
  trControl = control_smote
)

# XGBoost WITH scale_pos_weight
neg <- sum(train_y == "no")
pos <- sum(train_y == "yes")
scale_weight <- neg / pos

xgb_model_weighted <- train(
  x = train_x,
  y = train_y,
  method = "xgbTree",
  trControl = control,
  tuneGrid = xgb_grid,
  metric = "ROC",
  verbose = 0,
  scale_pos_weight = scale_weight
)

# XGBoost WITHOUT scale_pos_weight
xgb_model_unweighted <- train(
  x = train_x,
  y = train_y,
  method = "xgbTree",
  trControl = control,
  tuneGrid = xgb_grid,
  metric = "ROC",
  verbose = 0
)

# Predict Probabilities
log_prob <- predict(log_model, newdata = test_x, type = "prob")[, "yes"]
log_prob_smote <- predict(log_model_smote, newdata = test_x, type = "prob")[, "yes"]
xgb_prob_weighted <- predict(xgb_model_weighted, newdata = test_x, type = "prob")[, "yes"]
xgb_prob_unweighted <- predict(xgb_model_unweighted, newdata = test_x, type = "prob")[, "yes"]

# Evaluate at default threshold
evaluate_model(log_model, log_prob, test_y, "Logistic (No SMOTE)", threshold = 0.5)
evaluate_model(log_model_smote, log_prob_smote, test_y, "Logistic (SMOTE)", threshold = 0.5)
evaluate_model(xgb_model_weighted, xgb_prob_weighted, test_y, "XGBoost (weighted)", threshold = 0.5)
evaluate_model(xgb_model_unweighted, xgb_prob_unweighted, test_y, "XGBoost (unweighted)", threshold = 0.5)

# Tune thresholds
log_thresh_results <- tune_thresholds(log_prob_smote, test_y)
xgb_weighted_thresh_results <- tune_thresholds(xgb_prob_weighted, test_y)
xgb_unweighted_thresh_results <- tune_thresholds(xgb_prob_unweighted, test_y)

cat("\nTop Logistic (SMOTE) Thresholds:\n")
print(log_thresh_results %>% arrange(desc(balanced_accuracy)) %>% head(5))

cat("\nTop XGBoost (weighted) Thresholds:\n")
print(xgb_weighted_thresh_results %>% arrange(desc(balanced_accuracy)) %>% head(5))

cat("\nTop XGBoost (unweighted) Thresholds:\n")
print(xgb_unweighted_thresh_results %>% arrange(desc(balanced_accuracy)) %>% head(5))

evaluate_model(log_model_smote, log_prob_smote, test_y, "Logistic (SMOTE)", threshold = 0.50)
evaluate_model(xgb_model_weighted, xgb_prob_weighted, test_y, "XGBoost (weighted)", threshold = 0.85)

# feature of logistic regression smote model
summary(log_model_smote)
