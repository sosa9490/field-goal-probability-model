# Load necessary libraries
library(tidyverse)
library(nflreadr)
library(nflfastR)
library(gt)
library(gtExtras)

# Load custom data preparation and feature engineering functions
source("utilities/load_fg_data.R")
source("utilities/feature_engineering.R")

# Load trained models
log_model <- readRDS("models/log_model_smote_final.rds")
platt_model <- readRDS("models/platt_scaling_model.rds")

# Prepare 2024 kick data with relevant features for prediction
pred_data_2024 <- prepare_fg_features(season_start = 2024, season_end = 2024) %>%
  select(kicker_player_id, fg_made, kick_distance, Turf:away_game, cold:humid_game, windy_day,
         wind_speed_x_kick_distance, cold_x_wind) %>%
  drop_na()

# Predict raw probabilities of FG success using the logistic regression model
raw_probs <- predict(log_model, newdata = pred_data_2024, type = "prob")[, "yes"]

# Calibrate probabilities using calibration model
calibrated_probs <- predict(
  platt_model,
  newdata = data.frame(raw_probs_val = raw_probs),
  type = "response"
)

# Generate predictions using a 0.50 probability threshold
class_preds <- ifelse(calibrated_probs > 0.50, "yes", "no")

# Add predictions and calculated metrics to dataset
pred_results_24 <- pred_data_2024 %>%
  mutate(
    prob_raw = raw_probs,
    prob_calibrated = calibrated_probs,
    predicted_class = class_preds,
    points_added = 3 * fg_made - 3 * prob_calibrated,
    kick_range = case_when(
      kick_distance < 30 ~ "zero_29",
      kick_distance < 40 ~ "thirty_39",
      kick_distance < 50 ~ "fourty_49",
      TRUE ~ "fifty_plus"
    )
  ) %>%
  select(kicker_player_id, fg_made, kick_distance, kick_range, points_added)

# overall stats for kickers
overall_kicker_stats <- pred_results_24 %>%
  group_by(kicker_player_id) %>%
  summarise(
    longest_fg = {
      made_kicks <- kick_distance[fg_made == 1]
      if (length(made_kicks) > 0) max(made_kicks, na.rm = TRUE) else NA_real_
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

# kicker performance by kick range
distance_kicker_summary <- pred_results_24 %>%
  group_by(kicker_player_id, kick_range) %>%
  summarise(
    fg_m = sum(fg_made),
    fg_attempts = n(),
    fg_pct = 100 * round(sum(fg_made) / n(), 3),
    pts_added_per_attempt = round(mean(points_added), 2),
    .groups = "drop"
  )

# Compute reference z-scores for 40+ yard ranges
z_score_ref <- distance_kicker_summary %>%
  filter(!kick_range %in% c("zero_29", "thirty_39"), fg_attempts >= 1) %>%
  group_by(kick_range) %>%
  summarise(
    mean_pts = mean(pts_added_per_attempt),
    sd_pts = sd(pts_added_per_attempt),
    .groups = "drop"
  )

# Assign ratings by kick range using FG% (short kicks 0-39) or z-scores (kicks of 40+ yards)
distance_kicker_stats <- distance_kicker_summary %>%
  left_join(z_score_ref, by = "kick_range") %>%
  mutate(
    z_score = (pts_added_per_attempt - mean_pts) / sd_pts,
    rating = case_when(
      kick_range == "zero_29" & fg_attempts >= 1 ~ case_when(
        fg_pct >= 95 ~ "Excellent",
        fg_pct >= 90 ~ "Good",
        fg_pct >= 85 ~ "Average",
        fg_pct >= 80 ~ "Below Average",
        TRUE ~ "Poor"
      ),
      kick_range == "thirty_39" & fg_attempts >= 1 ~ case_when(
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

# Convert distance stats into wide format
distance_kicker_stats_wide <- distance_kicker_stats %>%
  pivot_wider(
    names_from = kick_range,
    values_from = c(M_A, fg_pct, pts_added_per_attempt, rating),
    names_glue = "{kick_range}_{.value}"
  )

# Combine overall and distance stats into one final table
kicker_stats_final <- overall_kicker_stats %>%
  left_join(distance_kicker_stats_wide, by = "kicker_player_id") %>%
  arrange(desc(pts_added_per_attempt)) %>%
  select(kicker_player_id:rating, 
         starts_with("zero_29"), starts_with("thirty_39"), starts_with("fourty_49"), starts_with("fifty_plus"), longest_fg)

# Load 2024 play-by-play data and summarize FG blocks by kicker and team
pbp_data_2024 <- nflfastR::load_pbp(2024)

pbp_fg_data_2024 <- pbp_data_2024 %>%
  filter(field_goal_attempt == 1) %>%
  group_by(kicker_player_id, posteam) %>%
  summarise(
    blocked_fg = sum(field_goal_result == "blocked", na.rm = TRUE),
    .groups = "drop"
  )

# Get kicker team history (up to 5 teams) and total FG blocks
kicker_team_blk_summary <- pbp_fg_data_2024 %>%
  group_by(kicker_player_id, posteam) %>%
  summarise(blocked_fg = sum(blocked_fg), .groups = "drop") %>%
  group_by(kicker_player_id) %>%
  mutate(team_index = paste0("team_", row_number())) %>%
  filter(row_number() <= 5) %>%
  pivot_wider(names_from = team_index, values_from = posteam) %>%
  summarise(
    kicker_player_id = first(kicker_player_id),
    total_blocks = sum(blocked_fg),
    across(starts_with("team_"), ~ .x[1]),
    .groups = "drop"
  )

# Load roster data and calculate kicker ages
roster_data <- load_rosters(2024) %>%
  filter(position == "K") %>%
  select(gsis_id, full_name, birth_date, headshot_url) %>%
  mutate(age = round(as.numeric(difftime(Sys.Date(), as.Date(birth_date), units = "days") / 365.25), 1)) %>%
  select(-birth_date)

# Load team logos
team_data <- load_teams(current = TRUE) %>%
  select(team_abbr, team_logo_espn)

# Merge kicker bio data, team logos, and block info
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

# Merge all data into final kicker table
kicker_table <- kicker_bio_data %>%
  left_join(kicker_stats_final, by = c("gsis_id" = "kicker_player_id")) %>%
  select(gsis_id:fg_pct, longest_fg, total_blocks, pts_added_per_attempt:last_col()) %>%
  filter(!is.na(M_A)) %>%
  select(-gsis_id) %>%
  arrange(desc(pts_added_per_attempt)) %>%
  mutate(Rk = rank(-pts_added_per_attempt, ties.method = "min")) %>%
  select(Rk, headshot_url, full_name, team_1_logo:team_3_logo, age, M_A:fg_pct, longest_fg, total_blocks, pts_added_per_attempt:last_col())

# BUILD OUTPUT

# Define rating color scheme
rating_colors <- list(
  "Excellent" = "#228B22",     # Dark green
  "Good" = "#A8E6A2",          # Light green
  "Average" = "#EAEAEA",       # Light gray
  "Below Average" = "#FFD8A8", # Peach
  "Poor" = "#F8B8B8"           # Light red
)

# Split kicker_table into three parts...won't fit on one page
table_page_1 <- kicker_table[1:15, ]
table_page_2 <- kicker_table[16:30, ]
table_page_3 <- kicker_table[31:nrow(kicker_table), ]

# Define a function to build the table for the kicking report
build_gt_table <- function(data, page_number = 1, total_pages = 2) {
  gt_table <- data %>%
    gt() %>%
    gt_theme_538() %>%
    tab_header(title = "Kicker Ranks by Points Added per Attempt – 2024 NFL Season") %>%
    tab_style(
      style = cell_text(align = "center", size = "25px", weight = "bold"),
      locations = cells_title("title")
    ) %>%
    cols_label(
      Rk = "Rank",
      headshot_url = '',
      full_name = "Name",
      age = "Age",
      team_1_logo = '',
      team_2_logo = '',
      team_3_logo = '',
      M_A = "FGM-FGA",
      fg_pct = "FG%",
      longest_fg = "Long",
      total_blocks = "FG Blocks",
      pts_added_per_attempt = "Pts+/Att",
      rating = "Rating",
      zero_29_M_A = "FGM-FGA",
      zero_29_fg_pct = "FG%",
      zero_29_pts_added_per_attempt = "Pts+/Att",
      zero_29_rating = "Rating",
      thirty_39_M_A = "FGM-FGA",
      thirty_39_fg_pct = "FG%",
      thirty_39_pts_added_per_attempt = "Pts+/Att",
      thirty_39_rating = "Rating",
      fourty_49_M_A = "FGM-FGA",
      fourty_49_fg_pct = "FG%",
      fourty_49_pts_added_per_attempt = "Pts+/Att",
      fourty_49_rating = "Rating",
      fifty_plus_M_A = "FGM-FGA",
      fifty_plus_fg_pct = "FG%",
      fifty_plus_pts_added_per_attempt = "Pts+/Att",
      fifty_plus_rating = "Rating"
    ) %>%
    tab_spanner(label = "Overall", columns = names(kicker_table)[8:13]) %>%
    tab_spanner(label = "0–29 Yards", columns = names(kicker_table)[14:17]) %>%
    tab_spanner(label = "30–39 Yards", columns = names(kicker_table)[18:21]) %>%
    tab_spanner(label = "40–49 Yards", columns = names(kicker_table)[22:25]) %>%
    tab_spanner(label = "50+ Yards", columns = names(kicker_table)[26:29]) %>%
    gt_img_rows(headshot_url, img_source = 'web') %>%
    gt_img_rows(team_1_logo, img_source = 'web') %>%
    gt_img_rows(team_2_logo, img_source = 'web') %>%
    gt_img_rows(team_3_logo, img_source = 'web') %>%
    tab_style(
      style = cell_text(size = "11px", whitespace = "nowrap", weight = "bold", align = "center"),  
      locations = cells_column_labels(everything())
    ) %>%
    tab_style(
      style = cell_text(size = "15px", weight = "bold", align = "center"),
      locations = cells_column_spanners()
    ) %>%
    cols_align(
      align = "center",
      columns = everything()
    ) %>%
    tab_style(
      style = cell_text(whitespace = "nowrap"),
      locations = cells_body(columns = full_name)
    ) %>%
    cols_width(
      age ~ px(95),
      M_A ~ px(95),
      fg_pct ~ px(95),
      longest_fg ~ px(95),
      total_blocks ~ px(95),
      pts_added_per_attempt ~ px(95),
      rating ~ px(95),
      zero_29_M_A ~ px(95),
      zero_29_fg_pct ~ px(95),
      zero_29_pts_added_per_attempt ~ px(95),
      zero_29_rating ~ px(95),
      thirty_39_M_A ~ px(95),
      thirty_39_fg_pct ~ px(95),
      thirty_39_pts_added_per_attempt ~ px(95),
      thirty_39_rating ~ px(95),
      fourty_49_M_A ~ px(95),
      fourty_49_fg_pct ~ px(95),
      fourty_49_pts_added_per_attempt ~ px(95),
      fourty_49_rating ~ px(95),
      fifty_plus_M_A ~ px(95),
      fifty_plus_fg_pct ~ px(95),
      fifty_plus_pts_added_per_attempt ~ px(95),
      fifty_plus_rating ~ px(95)
    ) %>%
    tab_style(
      style = cell_text(size = "11px", whitespace = "nowrap"),
      locations = cells_body()
    ) %>%
    tab_style(
      style = cell_borders(sides = "right", color = "gray60", weight = px(1)),
      locations = list(
        cells_body(columns = c(rating, zero_29_rating, thirty_39_rating, fourty_49_rating)),
        cells_column_labels(columns = c(rating, zero_29_rating, thirty_39_rating, fourty_49_rating))
      )
    ) %>%
    tab_style(
      style = list(
        cell_fill(color = "gray95"),
        cell_text(weight = "bold")
      ),
      locations = list(
        cells_body(columns = pts_added_per_attempt),
        cells_column_labels(columns = pts_added_per_attempt)
      )
    ) %>%
    tab_style(
      style = cell_borders(sides = "bottom", color = "black", weight = px(2)),
      locations = cells_body(rows = nrow(data))
    ) %>%
    tab_source_note(
      source_note = md("**Pts+/Att** estimates how many points a kicker adds per attempt compared to a league-average kicker, adjusted for distance and conditions.")
    ) %>%
    tab_source_note(
      source_note = md(paste0("**Page ", page_number, " of ", total_pages, "**"))
    )
  
# Apply conditional formatting for ratings
  rating_columns <- c("rating", "zero_29_rating", "thirty_39_rating", "fourty_49_rating", "fifty_plus_rating")
  for (col in rating_columns) {
    for (label in names(rating_colors)) {
      gt_table <- gt_table %>%
        tab_style(
          style = list(
            cell_fill(color = rating_colors[[label]]),
            cell_text(weight = "bold")
          ),
          locations = cells_body(
            columns = all_of(col),
            rows = .data[[col]] == label
          )
        )
    }
  }
  
  return(gt_table)
}

# Build pages
gt_page_1 <- build_gt_table(table_page_1, page_number = 1, total_pages = 3)
gt_page_2 <- build_gt_table(table_page_2, page_number = 2, total_pages = 3)
gt_page_3 <- build_gt_table(table_page_3, page_number = 3, total_pages = 3)

gt_page_1
gt_page_2
gt_page_3

