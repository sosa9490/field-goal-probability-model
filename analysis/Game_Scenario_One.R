# Load necessary libraries
library(tidyverse)
library(nflfastR)
library(nflreadr)

# Load feature engineering and data loading functions
source("utilities/feature_engineering.R")
source("utilities/load_fg_data.R")

# Load trained models: SMOTE logistic regression and Platt scaling calibration
log_model <- readRDS("models/log_model_smote_final.rds")
platt_model <- readRDS("models/platt_scaling_model.rds")

# Load play-by-play data for NFL seasons 2013 to 2024
pbp <- nflfastR::load_pbp(2013:2024)

# Compute Go For It Expected Points
# Filter to valid 4th-and-2 run/pass plays
# Exclude special teams, missing data, and edge cases:
# - End of half (last 2 mins)
# - Garbage time (blowouts or <5 min left)
# These situations don't reflect normal strategy
pbp_4th2 <- pbp %>%
  filter(
    down == 4,
    ydstogo == 2,
    play_type %in% c("run", "pass"),
    !is.na(yards_gained), !is.na(ep), !is.na(epa), !is.na(yardline_100),
    !is.na(game_seconds_remaining), !is.na(half_seconds_remaining), !is.na(score_differential),
    half_seconds_remaining > 120,
    game_seconds_remaining > 300 | abs(score_differential) <= 14
  ) %>%
  mutate(result = ifelse(first_down == 1, "success", "failure"),
         next_ep = ep + epa)

# Compute average EP on successful conversions
ep_success <- pbp_4th2 %>%
  filter(result == "success") %>%
  group_by(yardline_100) %>%
  summarize(EP_Success = mean(next_ep, na.rm = TRUE),
            n_success = n(),
            .groups = "drop")

# Compute average EP on failed conversions
ep_failure <- pbp_4th2 %>%
  filter(result == "failure") %>%
  group_by(yardline_100) %>%
  summarize(EP_Failure = mean(next_ep, na.rm = TRUE),
            n_failure = n(),
            .groups = "drop")

# Combine success/failure EPs and compute Go EP as the 50/50 average
go_ep_table <- ep_success %>%
  inner_join(ep_failure, by = "yardline_100") %>%
  mutate(
    Go_EP = 0.5 * EP_Success + 0.5 * EP_Failure,
    n_total = n_success + n_failure
  ) %>%
  filter(yardline_100 >= 0, yardline_100 <= 60) %>%
  arrange(yardline_100) %>%
  select(yardline_100, Go_EP, n_total)

# Filter to yardlines with a reliable sample size
go_ep_filtered <- go_ep_table %>%
  filter(n_total >= 10)

# Predict Field Goal EPs under KC winter conditions
# extract KC-specific game conditions in December/January
fg_data <- prepare_fg_features(season_start = 2022, season_end = 2024)
kc_winter_games <- fg_data %>%
  filter(stadium_id == "KAN00") %>%
  mutate(game_month = lubridate::month(game_date)) %>%
  filter(game_month %in% c(12, 1)) %>%
  summarise(
    turf_avg = mean(Turf),
    Altitude = sum(Altitude) / n(),
    Precip = sum(Precip) / n(),
    temp_avg = mean(temp_parsed),
    humid_avg = mean(humid_game)
  )

# Create input table simulating FG attempts from every yardline under KC conditions
kc_fg_inputs <- tibble(
  yardline_100 = 0:60,
  kick_distance = yardline_100 + 17,
  Turf = 0,             # KC has grass
  Altitude = 0,
  Precip = 0,
  night_game = 1,
  away_game = 1,
  cold = 1,
  hot = 0,
  humid_game = 0,
  windy_day = 1,
  wind_speed_x_kick_distance = 20 * (yardline_100 + 17),
  cold_x_wind = 1
)

# Predict field goal probability with calibrated logistic model
raw_probs <- predict(log_model, newdata = kc_fg_inputs, type = "prob")[, "yes"]
kc_fg_predictions <- kc_fg_inputs %>%
  mutate(FG_Prob = predict(platt_model, newdata = data.frame(raw_probs_val = raw_probs), type = "response"))

# Compute Expected Points if Field Goal Misses

# Build average EP for opponent starting a normal drive from each yardline
miss_spot_ep <- pbp %>%
  filter(
    down == 1,
    ydstogo == 10,
    !is.na(yardline_100),
    !is.na(ep),
    half_seconds_remaining > 120,
    game_seconds_remaining > 300 | abs(score_differential) <= 14
  ) %>%
  mutate(own_yardline_100 = yardline_100) %>%
  filter(own_yardline_100 >= 20, own_yardline_100 <= 93) %>%
  group_by(own_yardline_100) %>%
  summarize(opponent_EP = mean(ep, na.rm = TRUE), .groups = "drop")

# Calculate where opponent would take over after a missed FG
miss_ep_table <- tibble(yardline_100 = 0:60) %>%
  mutate(
    kick_spot = yardline_100 + 7,
    spot_after_miss = if_else(kick_spot < 20, 80, 100 - kick_spot),
    kick_yardline = 100 - kick_spot
  ) %>%
  left_join(miss_spot_ep, by = c("spot_after_miss" = "own_yardline_100")) %>%
  mutate(EP_if_missed = -opponent_EP) %>%
  select(yardline_100, kick_yardline, kick_spot, spot_after_miss, EP_if_missed)

# Merge FG make/miss outcomes into final EP calculation
fg_ep_table <- kc_fg_predictions %>%
  select(yardline_100, FG_Prob) %>%
  left_join(miss_ep_table %>% select(yardline_100, EP_if_missed), by = "yardline_100") %>%
  mutate(FG_EP = FG_Prob * 3 + (1 - FG_Prob) * EP_if_missed)

# Compare Go vs Kick

# Join FG_EP and Go_EP into a single long-format table
ep_compare <- fg_ep_table %>%
  inner_join(go_ep_filtered, by = "yardline_100") %>%
  pivot_longer(cols = c(FG_EP, Go_EP), names_to = "Decision", values_to = "EP") %>%
  mutate(
    Decision = recode(Decision,
                      "FG_EP" = "Field Goal",
                      "Go_EP" = "Go for it")
  )

# Visualization: Expected Points by Yardline
# Plot smoothed EP curves with decision region shading, vertical lines, and labels
ggplot(ep_compare, aes(x = yardline_100, y = EP, color = Decision)) +
  annotate("rect", xmin = 0, xmax = 16, ymin = -Inf, ymax = Inf, alpha = 0.1, fill = "#1f77b4") +
  annotate("rect", xmin = 16, xmax = 31, ymin = -Inf, ymax = Inf, alpha = 0.1, fill = "#d62728") +
  annotate("rect", xmin = 31, xmax = 60, ymin = -Inf, ymax = Inf, alpha = 0.1, fill = "#1f77b4") +
  geom_vline(xintercept = 16, linetype = "dashed", size = 1, color = "black") +
  geom_vline(xintercept = 31, linetype = "dashed", size = 1, color = "black") +
  annotate("text", x = 16, y = min(ep_compare$EP) + 0.25, label = "16 yd", angle = 90, vjust = -0.5, size = 4, fontface = "bold") +
  annotate("text", x = 31, y = min(ep_compare$EP) + 0.25, label = "31 yd", angle = 90, vjust = -0.5, size = 4, fontface = "bold") +
  geom_smooth(se = FALSE, method = "loess", span = 0.3, size = 1.2) +
  annotate("text", x = 8, y = max(ep_compare$EP), label = "Go", color = "#1f77b4", size = 5, fontface = "bold") +
  annotate("text", x = 23.5, y = max(ep_compare$EP), label = "Kick", color = "#d62728", size = 5, fontface = "bold") +
  annotate("text", x = 45, y = max(ep_compare$EP), label = "Go", color = "#1f77b4", size = 5, fontface = "bold") +
  labs(
    title = "Expected Points by Yardline - 4th & 2",
    subtitle = "Week 15: LAC @ KC (20 mph Wind)",
    x = "Yardline (distance from end zone)",
    y = "Expected Points"
  ) +
  scale_color_manual(values = c("Go for it" = "#1f77b4", "Field Goal" = "#d62728")) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    legend.position = "top",
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 11)
  )
             