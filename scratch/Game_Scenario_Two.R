# Load necessary libraries
library(tidyverse)
library(nflfastR)
library(nflreadr)
source("utilities/feature_engineering.R")

# load pbp data / we want to identify replacement level kicker then see how FG make percent compares to non-replacement kickers
# in this case we will identify replacement kickers as kickers with less than 20 FGA career attempts from 2013-2024

fg_data_pbp <- load_pbp(2013:2024) %>%
  filter(play_type == "field_goal")

kicker_stats <- fg_data_pbp %>%
  group_by(kicker_player_id) %>%
  summarise(
    fg_made = sum(field_goal_result == "made", na.rm = TRUE),
    fga = n())

kicker_stats <- kicker_stats %>%
  mutate(replacement_level = ifelse(fga <= 20, 1, 0))

replacement_vs_non_df <- kicker_stats %>%
  group_by(replacement_level) %>%
  summarise(
    made = sum(fg_made, na.rm = TRUE),
    att = sum(fga, na.rm = TRUE),
    pct = sum(fg_made, na.rm = TRUE) / sum(fga, na.rm = TRUE))

# the different between replacement and non-replacement kickers
performance_penalty <- replacement_vs_non_df %>%
  filter(replacement_level == 1) %>%
  pull(pct) - 
  replacement_vs_non_df %>%
  filter(replacement_level == 0) %>%
  pull(pct)

# replacement level / kickers signed off the street have a FG make percent that is 11.5% lower than the average kicker

# Determine the FG make probability threshold at which we can be confident an average kicker would make the kick

# Load saved models
log_model <- readRDS("models/log_model_smote_final.rds")
platt_model <- readRDS("models/platt_scaling_model.rds")

kicking_data_2024 <- prepare_fg_features(season_start = 2024, season_end = 2024) %>%
  select(fg_made, kick_distance, Turf:away_game, cold:humid_game, windy_day,
         wind_speed_x_kick_distance, cold_x_wind) %>%
  drop_na()

raw_probs <- predict(log_model, newdata = kicking_data_2024, type = "prob")[, "yes"]

calibrated_probs <- predict(
  platt_model,
  newdata = data.frame(raw_probs_val = raw_probs),  # same variable name used during training
  type = "response"
)
class_preds <- ifelse(calibrated_probs > 0.50, "yes", "no")

results <- kicking_data_2024 %>%
  mutate(
    prob_raw = raw_probs,
    prob_calibrated = calibrated_probs,
    predicted_class = class_preds
  )

confidence_df <- results %>%
  mutate(prob_threshold = case_when(
    prob_calibrated <= 0.50 ~ "Less Than 50%",
    prob_calibrated > 0.50 & prob_calibrated <= 0.75 ~ "50% to 75%",
    prob_calibrated > 0.75 ~ "Greater Than 75%"
  ))

# Calculate the average FG make percentage for each confidence threshold
confidence_summary <- confidence_df %>%
  group_by(prob_threshold) %>%
  summarise(
    fg_pct = sum(fg_made, na.rm = TRUE) / n()
  )

# For average NFL kickers, a predicted make probability >75% is a confident zone — these attempts are made 90% of the time (compared to 73% for 50–75% and 28% for <50%)

# Predict Field Goal probabily under average LA in october (week 7)
# create prediction matrix
fg_weather_data <- prepare_fg_features(season_start = 2022, season_end = 2024)

la_weather_data <- fg_weather_data %>%
  filter(stadium_id == 'LAX01') %>%
  mutate(game_month = lubridate::month(game_date)) %>%
  filter(game_month %in% c(10)) %>%
  summarise(
    turf_avg = mean(Turf),
    Altitude = sum(Altitude) / n(),
    Precip = sum(Precip) / n(),
    temp_avg = mean(temp_parsed),
    humid_avg = mean(humid_game),
    wind_speed = mean(wind_speed_parsed)
  )

# Create input table simulating FG attempts from every yardline under LA condtion
la_wk_seven_fg_inputs <- tibble(
  yardline_100 = 0:50,
  kick_distance = yardline_100 + 17,
  Turf = 1,             
  Altitude = 0,
  Precip = 0,
  night_game = 0,
  away_game = 0,
  cold = 0,
  hot = 0,
  humid_game = 0,
  windy_day = 0,
  wind_speed_x_kick_distance = 0 * (yardline_100 + 17),
  cold_x_wind = 0
)

# Predict field goal probability with calibrated logistic model
raw_probs <- predict(log_model, newdata = la_wk_seven_fg_inputs, type = "prob")[, "yes"]
la_week_seven_fg_predictions <- la_wk_seven_fg_inputs %>%
  mutate(FG_Prob = predict(platt_model, newdata = data.frame(raw_probs_val = raw_probs), type = "response"))

# adjusted probabilities for replacement level kickers
la_week_seven_fg_predictions <- la_week_seven_fg_predictions %>%
  mutate(
    replacement_level_prob = pmax(0, FG_Prob + performance_penalty))

# Find the first yardline where replacement-level probability drops below 75%
target_line <- la_week_seven_fg_predictions %>%
  filter(replacement_level_prob >= 0.75) %>%
  summarise(max_confident_yardline = max(yardline_100)) %>%
  pull(max_confident_yardline)

stretch_line <- la_week_seven_fg_predictions %>%
  filter(replacement_level_prob >= 0.50) %>%
  summarise(max_yardline = max(yardline_100)) %>%
  pull(max_yardline)

# Confidence zones: only High and Low
confidence_zones <- tibble(
  ymin = c(0.75, 0.00),
  ymax = c(1.00, 0.50),
  zone = c("High Confidence", "Low Confidence")
)

ggplot(la_week_seven_fg_predictions, aes(x = yardline_100, y = replacement_level_prob)) +
  
  # Confidence zone background shading
  geom_rect(data = confidence_zones,
            aes(xmin = -Inf, xmax = Inf, ymin = ymin, ymax = ymax, fill = zone),
            inherit.aes = FALSE,
            alpha = 0.15) +
  
  # FG probability line
  geom_line(color = "#1f77b4", size = 1.5) +
  
  # Horizontal cutoff lines
  geom_hline(yintercept = 0.75, linetype = "dashed", color = "#2ca02c", size = 1) +
  geom_hline(yintercept = 0.50, linetype = "dashed", color = "#d62728", size = 1) +
  
  # Vertical decision lines
  geom_vline(xintercept = target_line, linetype = "dashed", color = "#2ca02c", size = 1) +
  geom_vline(xintercept = stretch_line, linetype = "dashed", color = "#d62728", size = 1) +
  
  # Vertical line labels
  geom_label(aes(x = target_line, y = 0.15,
                 label = paste0("Target Line: ", target_line, " yd")),
             color = "white", fill = "#2ca02c",
             fontface = "bold", size = 4, angle = 90,
             label.padding = unit(0.25, "lines")) +
  
  geom_label(aes(x = stretch_line, y = 0.15,
                 label = paste0("Stretch Line: ", stretch_line, " yd")),
             color = "white", fill = "#d62728",
             fontface = "bold", size = 4, angle = 90,
             label.padding = unit(0.25, "lines")) +
  
  # Custom legend
  scale_fill_manual(
    name = "Confidence Zone",
    values = c(
      "High Confidence" = "#2ca02c",  # green
      "Low Confidence" = "#d62728"    # red
    )
  ) +
  
  # Titles and theme
  labs(
    title = "FG Attempt Range for Week 7 Replacement Kicker",
    x = "Yard Line (Distance from End Zone)",
    y = "FG Make Probability"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(color = "black"),
    panel.grid.major = element_line(color = "gray90"),
    legend.position = "top",
    legend.justification = "center",
    legend.title = element_text(face = "bold"),
    plot.margin = margin(10, 20, 10, 10)
  )
