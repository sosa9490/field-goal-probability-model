# Load libraries
library(tidyverse)
library(rsample)
library(caret)
library(pROC)
library(MLmetrics)
library(themis)

# Load custom functions
source("utilities/load_fg_data.R")
source("utilities/feature_engineering.R")
source("utilities/helpers.R")  # Must include `evaluate_model()` if used

# Load and prepare data
data <- prepare_fg_features(season_start = 2013, season_end = 2024) %>%
  filter(season >= 2008 & season <= 2023) %>%
  mutate(
    fg_made = factor(fg_made, levels = c(0, 1), labels = c("no", "yes"))
  ) %>%
  select(play_id, game_id, season, posteam, fg_made,
         kick_distance, Turf:away_game, cold:humid_game, windy_day,
         wind_speed_x_kick_distance, cold_x_wind) %>%
  drop_na()

# Split into train and test
set.seed(42)
outer_split <- initial_split(data, prop = 0.8, strata = fg_made)
train_val_data <- training(outer_split)
test_data <- testing(outer_split)

# Split train into training and validation sets
inner_split <- initial_split(train_val_data, prop = 0.8, strata = fg_made)
train_data <- training(inner_split)
val_data <- testing(inner_split)

# Prepare training data
train_x <- train_data %>% select(-fg_made, -play_id, -game_id, -posteam, -season)
train_y <- train_data$fg_made

# Train logistic regression with SMOTE
control_smote <- trainControl(
  method = "cv",
  number = 3,
  sampling = "smote",
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = TRUE
)

log_model_smote <- train(
  x = train_x,
  y = train_y,
  method = "glm",
  family = "binomial",
  metric = "ROC",
  trControl = control_smote
)

# Save trained model
saveRDS(log_model_smote, "models/log_model_smote_final.rds")

# === Calibration: Platt Scaling ===

# Get predictions on validation set
val_x <- val_data %>% select(-fg_made, -play_id, -game_id, -posteam, -season)
val_y <- val_data$fg_made

raw_probs_val <- predict(log_model_smote, newdata = val_x, type = "prob")[, "yes"]

# Fit calibration model
platt_model <- glm(val_y ~ raw_probs_val, family = "binomial")
saveRDS(platt_model, "models/platt_scaling_model.rds")

# # === Predict on Test Set with Calibration ===
# 
# test_x <- test_data %>% select(-fg_made, -play_id, -game_id, -posteam)
# test_y <- test_data$fg_made
# 
# raw_probs_test <- predict(log_model_smote, newdata = test_x, type = "prob")[, "yes"]
# calibrated_probs_test <- predict(platt_model, newdata = data.frame(raw_probs_val = raw_probs_test), type = "response")
# class_preds <- ifelse(calibrated_probs_test > 0.50, "yes", "no")
# 
# # === Evaluate ===
# 
# brier_raw <- mean((ifelse(test_y == "yes", 1, 0) - raw_probs_test)^2)
# brier_cal <- mean((ifelse(test_y == "yes", 1, 0) - calibrated_probs_test)^2)
# 
# cat("Brier Score (Raw):", round(brier_raw, 4), "\n")
# cat("Brier Score (Calibrated):", round(brier_cal, 4), "\n")
# 
# roc_raw <- pROC::auc(pROC::roc(test_y, raw_probs_test))
# roc_cal <- pROC::auc(pROC::roc(test_y, calibrated_probs_test))
# 
# cat("AUC (Raw):", round(roc_raw, 4), "\n")
# cat("AUC (Calibrated):", round(roc_cal, 4), "\n")
# 
# # === Output Final Results ===
# 
# output <- test_data %>%
#   select(play_id, game_id, season, posteam, fg_made) %>%
#   mutate(
#     prob_raw = raw_probs_test,
#     prob_calibrated = calibrated_probs_test,
#     predicted_class = class_preds
#   )