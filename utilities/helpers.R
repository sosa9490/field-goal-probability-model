# Helper functions to clean and evaluate field goal play-by-play data
# Includes tools for summarizing missing data, parsing weather strings,
# evaluating model performance, and tuning decision thresholds.

library(tidyverse)

# summarize_blanks()
# Quick way to see how many blanks or NAs are in each column of a dataset
# Especially useful after pulling messy data like PBP or weather strings.
summarize_blanks <- function(df) {
  data.frame(
    column = names(df),
    na_or_blank = sapply(df, function(col) {
      if (is.character(col)) {
        sum(is.na(col) | col == "")
      } else {
        sum(is.na(col))
      }
    }),
    total = nrow(df)
  ) %>%
    mutate(
      percent_missing = round(100 * na_or_blank / total, 1)
    ) %>%
    arrange(desc(percent_missing))
}

# parse_weather()
# Turns the messy 'weather' string into structured columns like temp, humidity,
# wind direction, and wind speed. Helps normalize inconsistently formatted text.
# Use for filling in missing weather data
parse_weather <- function(weather_str) {
  weather_str <- ifelse(is.na(weather_str), "", weather_str)
  
  tibble(raw = weather_str) %>%
    mutate(
      forecast_parsed = str_extract(raw, "^(.*?)(?=Temp:)") %>% str_trim(),
      temp_parsed = str_extract(raw, "(?<=Temp: )[0-9]+") %>% as.numeric(),
      humidity_parsed = str_extract(raw, "(?<=Humidity: )[0-9]+") %>% as.numeric(),
      
      # Extract wind phrase (e.g., "Wind: from NW 11 mph")
      wind_phrase = str_extract(raw, "Wind: [^,]+"),
      
      # Extract wind direction, removing "from"/"From" prefix that sometimes appears
      wind_direction_parsed = case_when(
        str_detect(wind_phrase, "Wind: (N/A|mph|\\d+ mph)") ~ NA_character_,
        TRUE ~ str_extract(wind_phrase, "(?<=Wind: )([Ff]rom )?[A-Za-z]+") %>%
          str_remove("(?i)from ")  
      ),
      # Extract wind speed as number before 'mph'
      wind_speed_parsed = case_when(
        str_detect(wind_phrase, "mph") ~ str_extract(wind_phrase, "\\d+") %>% as.numeric(),
        TRUE ~ NA_real_
      )
    ) %>%
    select(-raw, -wind_phrase)
}

# evaluate_model()
# Prints performance metrics for a classification model, given probabilities and true labels
# Includes confusion matrix, AUC, Brier Score, and Log Loss.
# Used to compare different models or thresholds
evaluate_model <- function(model, probs, truth, name, threshold = 0.5) {
  # Convert probabilities to predicted labels
  preds <- ifelse(probs > threshold, "yes", "no") %>% factor(levels = c("no", "yes"))
  # Convert truth labels to numeric 0/1 for metrics like Brier and log loss
  y_true <- ifelse(truth == "yes", 1, 0)
  # Bound the probabilities so we’re not feeding log loss a 0 or 1
  safe_probs <- pmin(pmax(probs, 1e-15), 1 - 1e-15)
  # Print all evaluation metrics
  cat("\n----", name, "(Threshold =", threshold, ") ----\n")
  print(confusionMatrix(preds, truth, positive = "yes"))
  cat("AUC:", pROC::auc(pROC::roc(response = truth, predictor = probs)), "\n")
  cat("Brier Score:", mean((y_true - probs)^2), "\n")
  cat("Log Loss:", Metrics::logLoss(truth=="yes", probs), "\n")
}

# tune_thresholds()
# Evaluates model performance across a range of classification thresholds.
# Helps find the sweet spot between sensitivity and specificity.
tune_thresholds <- function(pred_probs, true_labels) {
  thresholds <- seq(0.1, 0.9, by = 0.05)
  results <- data.frame()
  for (t in thresholds) {
    preds <- ifelse(pred_probs > t, "yes", "no") %>% factor(levels = c("no", "yes"))
    cm <- confusionMatrix(preds, true_labels, positive = "yes")
    results <- rbind(results, data.frame(
      threshold = t,
      accuracy = cm$overall["Accuracy"],
      sensitivity = cm$byClass["Sensitivity"],
      specificity = cm$byClass["Specificity"],
      balanced_accuracy = cm$byClass["Balanced Accuracy"],
      F1 = cm$byClass["F1"]
    ))
  }
  return(results)
}