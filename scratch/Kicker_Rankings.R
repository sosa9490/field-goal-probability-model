# Load necessary libraries
library(tidyverse)
library(nflreadr)
library(nflfastR)

# Load custom data preparation and feature engineering functions
source("utilities/load_fg_data.R")
source("utilities/feature_engineering.R")

# Load the trained logistic regression model and Platt scaling calibration model
log_model <- readRDS("models/log_model_smote_final.rds")
platt_model <- readRDS("models/platt_scaling_model.rds")

# Prepare 2024 kick data and select relevant features for prediction
pred_data_2024 <- prepare_fg_features(season_start = 2024, season_end = 2024) %>%
  select(kicker_player_id, fg_made, kick_distance, Turf:away_game, cold:humid_game, windy_day,
         wind_speed_x_kick_distance, cold_x_wind) %>%
  drop_na()

# Predict raw probabilities from the logistic regression model
raw_probs <- predict(log_model, newdata = pred_data_2024, type = "prob")[, "yes"]

# Apply Platt scaling to calibrate predicted probabilities
calibrated_probs <- predict(
  platt_model,
  newdata = data.frame(raw_probs_val = raw_probs),
  type = "response"
)

# Convert calibrated probabilities into binary class predictions using a 0.50 threshold
class_preds <- ifelse(calibrated_probs > 0.50, "yes", "no")

# Combine predictions with original data and calculate points added + kick range bucket
pred_results_24 <- pred_data_2024 %>%
  mutate(
    prob_raw = raw_probs,
    prob_calibrated = calibrated_probs,
    predicted_class = class_preds,
    points_added = 3 * fg_made - 3 * prob_calibrated,
    kick_range = case_when(
      kick_distance < 30 ~ "0-29",
      kick_distance >= 30 & kick_distance < 40 ~ "30-39",
      kick_distance >= 40 & kick_distance < 50 ~ "40-49",
      kick_distance >= 50 ~ "50+"
    )
  ) %>%
  select(kicker_player_id, fg_made, kick_distance, kick_range, points_added)

# Generate overall stats for each kicker:
# - Total makes, attempts, FG%
# - Points added per attempt
# - Z-score based rating (min 10 attempts)
overall_kicker_stats <- pred_results_24 %>%
  group_by(kicker_player_id) %>%
  summarise(
    longest_fg = {
      made_kicks <- kick_distance[fg_made == 1]
      if (length(made_kicks) > 0) {
        max(made_kicks, na.rm = TRUE)
      } else {
        NA_real_
      }
    },
    fg_m = sum(fg_made),
    fg_attempts = n(),
    fg_pct = 100 * round(sum(fg_made) / n(), 3),
    pts_added_per_attempt = round(sum(points_added) / n(), 2),
    .groups = "drop"
  ) %>%
  filter(fg_attempts >= 10) %>%
  mutate(
    mean_pts = mean(pts_added_per_attempt),
    sd_pts = sd(pts_added_per_attempt),
    z_score = (pts_added_per_attempt - mean_pts) / sd_pts,
    rating = case_when(
      z_score >= 1.0 ~ "Excellent",
      z_score >= 0.5 ~ "Good",
      z_score >= -0.5 ~ "Average",
      z_score >= -1.0 ~ "Below Average",
      TRUE ~ "Poor"
    ),
    M_A = paste0(fg_m, "/", fg_attempts)
  ) %>%
  select(kicker_player_id, M_A, fg_pct, pts_added_per_attempt, rating, longest_fg)

# Summarize each kicker's performance by distance range:
# - Total makes and attempts
# - FG%
# - Points added per attempt
distance_kicker_summary <- pred_results_24 %>%
  group_by(kicker_player_id, kick_range) %>%
  summarise(
    fg_m = sum(fg_made),
    fg_attempts = n(),
    fg_pct = 100 * round(sum(fg_made) / n(), 3),
    pts_added_per_attempt = round(mean(points_added), 2),
    .groups = "drop"
  )

# Compute mean and standard deviation of points added per attempt for each distance range
# Only applied to 40+ yard ranges; short-range buckets use absolute FG% thresholds instead
z_score_ref <- distance_kicker_summary %>%
  filter(!kick_range %in% c("0-29", "30-39"), fg_attempts >= 1) %>%
  group_by(kick_range) %>%
  summarise(
    mean_pts = mean(pts_added_per_attempt),
    sd_pts = sd(pts_added_per_attempt),
    .groups = "drop"
  )

# Assign distance-based ratings to each kicker-range:
# - 0–29 and 30–39 yard ranges use FG% thresholds:
#   • 0–29: thresholds like ≥95% for Excellent due to near-automatic make rate
#   • 30–39: thresholds (93/88/83/78) based on historical NFL averages (league avg ≈ 89–91%)
# - 40+ ranges use z-scores with softened thresholds (0.75, 0.25, etc.)
# - Only kickers with ≥1 attempt in a range are rated; others are NA
distance_kicker_stats <- distance_kicker_summary %>%
  left_join(z_score_ref, by = "kick_range") %>%
  mutate(
    z_score = (pts_added_per_attempt - mean_pts) / sd_pts,
    
    rating = case_when(
      kick_range == "0-29" & fg_attempts >= 1 ~ case_when(
        fg_pct >= 95 ~ "Excellent",
        fg_pct >= 90 ~ "Good",
        fg_pct >= 85 ~ "Average",
        fg_pct >= 80 ~ "Below Average",
        TRUE ~ "Poor"
      ),
      kick_range == "30-39" & fg_attempts >= 1 ~ case_when(
        fg_pct >= 93 ~ "Excellent",
        fg_pct >= 88 ~ "Good",
        fg_pct >= 83 ~ "Average",
        fg_pct >= 78 ~ "Below Average",
        TRUE ~ "Poor"
      ),
      fg_attempts >= 1 ~ case_when(
        z_score >= 0.75 ~ "Excellent",
        z_score >= 0.25 ~ "Good",
        z_score >= -0.25 ~ "Average",
        z_score >= -0.75 ~ "Below Average",
        TRUE ~ "Poor"
      ),
      TRUE ~ NA_character_
    ),
    
    M_A = paste0(fg_m, "/", fg_attempts)
  ) %>%
  select(kicker_player_id, kick_range, M_A, fg_pct, pts_added_per_attempt, rating)

# Pivot to wide format:
# Each kick_range becomes its own group of columns:
# - M_A, fg_pct, points added per attempt, and rating
distance_kicker_stats_wide <- distance_kicker_stats %>%
  pivot_wider(
    names_from = kick_range,
    values_from = c(M_A, fg_pct, pts_added_per_attempt, rating),
    names_glue = "{kick_range}_{.value}"
  )

# Combine overall and range-based kicker stats into final output:
# - Includes overall rating
# - Orders columns by kick range for clarity
kicker_stats_final <- overall_kicker_stats %>%
  left_join(distance_kicker_stats_wide, by = "kicker_player_id") %>%
  arrange(desc(pts_added_per_attempt)) %>%
  select(kicker_player_id:rating, 
         starts_with("0-29"), starts_with("30-39"), starts_with("40-49"), starts_with("50+"), longest_fg)


# get some additional data for each kicker: longest fields, blocked, team played for
pbp_data_2024 <- nflfastR::load_pbp(2024)

#filter for field goals / calculated longest made FG and blocked kicks
pbp_fg_data_2024 <- pbp_data_2024 %>%
  filter(field_goal_attempt == 1) %>%
  group_by(kicker_player_id, posteam) %>%
  summarise(
    blocked_fg = sum(field_goal_result == "blocked", na.rm = TRUE),
    .groups = "drop"
  )

kicker_team_blk_summary <- pbp_fg_data_2024 %>%
  group_by(kicker_player_id, posteam) %>%
  summarise(blocked_fg = sum(blocked_fg), .groups = "drop") %>%
  group_by(kicker_player_id) %>%
  mutate(team_index = paste0("team_", row_number())) %>%
  filter(row_number() <= 5) %>%  # Keep only first 5 teams
  pivot_wider(
    names_from = team_index,
    values_from = posteam
  ) %>%
  summarise(
    kicker_player_id = first(kicker_player_id),
    total_blocks = sum(blocked_fg),
    across(starts_with("team_"), ~ .x[1]),
    .groups = "drop"
  )

# get biographical data for each kicker
roster_data <- load_rosters(2024) %>%
  filter(position == "K") %>%
  select(gsis_id, full_name, birth_date, headshot_url) %>%
  mutate(age= round(as.numeric(difftime(Sys.Date(), as.Date(birth_date), units = "days") / 365.25), 1)) %>%
  select(-birth_date)


# get team data and logo url
team_data <- load_teams(current = TRUE) %>%
  select(team_abbr, team_logo_espn)

# merge all kicker bio data for 2024 season
kicker_bio_data <- roster_data %>%
  left_join(kicker_team_blk_summary, by = c("gsis_id" = "kicker_player_id")) %>%
  left_join(team_data, by = c("team_1" = "team_abbr")) %>%
  left_join(team_data, by = c("team_2" = "team_abbr")) %>%
  left_join(team_data, by = c("team_3" = "team_abbr")) %>%
  rename(
    team_1_logo = team_logo_espn.x,
    team_2_logo = team_logo_espn.y,
    team_3_logo = team_logo_espn
  ) %>%
  select(gsis_id, headshot_url, full_name, age, team_1_logo, team_2_logo, team_3_logo, total_blocks)

# merge with kicker stats
kicker_table <- kicker_bio_data %>%
  left_join(kicker_stats_final, by = c("gsis_id" = "kicker_player_id")) %>%
  select(gsis_id:fg_pct, longest_fg, total_blocks, pts_added_per_attempt:last_col()) %>%
  filter(!is.na(M_A))


  

