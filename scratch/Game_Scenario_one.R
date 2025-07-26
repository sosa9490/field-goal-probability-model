library(tidyverse)
library(nflfastR)
library(nflreadr)

source("utilities/feature_engineering.R")
source("utilities/load_fg_data.R")
source("utilities/feature_engineering.R")

# Load saved model
log_model <- readRDS("models/log_model_smote_final.rds")
platt_model <- readRDS("models/platt_scaling_model.rds")

pbp <- nflfastR::load_pbp(2013:2024)

# Filter to valid 4th-and-2 go-for-it attempts
# - Keep only 4th-and-2 run/pass plays (exclude punts/FGs)
# - Ensure needed fields (yards_gained, ep, epa, yardline) are not missing
# - Filter out edge-case situations like end of halves and garbage time games

pbp_4th2 <- pbp %>%
  filter(
    down == 4,
    ydstogo == 2,
    play_type %in% c("run", "pass"),
    !is.na(yards_gained),
    !is.na(ep), !is.na(epa),
    !is.na(yardline_100),
    !is.na(game_seconds_remaining), !is.na(half_seconds_remaining), !is.na(score_differential)
  ) %>%
  filter(
    half_seconds_remaining > 120,                              # Exclude last 2 mins of half
    game_seconds_remaining > 300 | abs(score_differential) <= 14  # Exclude blowouts/garbage time
  ) %>%
  mutate(result = ifelse(first_down == 1, "success", "failure"),
         next_ep = ep + epa)
#Compute average next_ep for successful 4th-and-2 plays by yardline

ep_success <- pbp_4th2 %>%
  filter(result == "success") %>%
  group_by(yardline_100) %>%
  summarize(EP_Success = mean(next_ep, na.rm = TRUE),
            n_success = n(),
            .groups = "drop")

# Compute average next_ep for failed 4th-and-2 plays by yardline
# No need to flip — the EPA already accounts for the negative swing
ep_failure <- pbp_4th2 %>%
  filter(result == "failure") %>%
  group_by(yardline_100) %>%
  summarize(EP_Failure = mean(next_ep, na.rm = TRUE), 
            n_failure = n(),
            .groups = "drop")

# STEP 5: Join success and failure EPs, and calculate Go_EP
# Go_EP = average EP from going for it with 50% success probability
# Limit to decision zone: yardline_100 between  0 and 60
go_ep_table <- ep_success %>%
  inner_join(ep_failure, by = "yardline_100") %>%
  mutate(
    Go_EP = 0.5 * EP_Success + 0.5 * EP_Failure,
    n_total = n_success + n_failure
  ) %>%
  filter(yardline_100 >= 0, yardline_100 <= 60) %>%
  arrange(yardline_100) %>%
  select(yardline_100, Go_EP, n_total)

# Filter to yardlines with at least 10 total 4th-and-2 plays
go_ep_filtered <- go_ep_table %>%
  filter(n_total >= 10)

# Plot Go_EP by yardline with LOESS smoothing
# ggplot(go_ep_filtered, aes(x = yardline_100, y = Go_EP)) +
#   geom_point(aes(size = n_total), alpha = 0.8) +                # show actual data points
#   geom_smooth(method = "loess", span = 0.5, se = FALSE, color = "blue", linewidth = 1.2) +  # smoothed curve
#   scale_x_reverse() +                                           # closer to end zone on right
#   labs(
#     title = "Smoothed Expected Points for Going for It on 4th-and-2",
#     subtitle = "Filtered for Yardlines with ≥10 Plays",
#     x = "Yards from End Zone (yardline_100)",
#     y = "Go_EP (Expected Points)",
#     size = "Number of Plays"
#   ) +
#   theme_minimal()

# get EP for field goal attempts

# Create basic input data across yardlines (0 to 60)
# Kick distance = LOS + 17 (10 end zone + 7 snap)
# load feature engineered field goal data

# Load 2024 Kansas City field goal data
fg_data <- prepare_fg_features(season_start = 2022, season_end = 2024)
  
# Filter to games played in December or January
kc_winter_games <- fg_data %>%
  filter(stadium_id == "KAN00") %>%
  mutate(game_month = lubridate::month(game_date)) %>%
  filter(game_month %in% c(12, 1)) %>%
  summarise(turf_avg = mean(Turf),
            Altitude = sum(Altitude) / n(),
            Precip = sum(Precip) / n(),
            temp_avg = mean(temp_parsed),
            humid_avg = mean(humid_game))

kc_fg_inputs <- tibble(
  # Yardline (distance from opponent end zone)
  yardline_100 = 0:60,
  # Kick distance = yardline + 17 yards (accounts for snap and end zone depth)
  kick_distance = yardline_100 + 17,
  # KC is a grass stadium
  Turf = 0,
  # Altitude flag: 0 
  Altitude = 0,
  # Precipitation flag: 0 = assume no rain/snow
  Precip = 0,
  # SNF
  night_game = 1,
  # Away game
  away_game = 1,
  # Cold weather flag / winter game, ~32°F
  cold = 1,
  hot = 0,
  # KC average winter humidity is < 80%
  humid_game = 0,
  # Windy day flag: 1 = true (20 mph wind gusts)
  windy_day = 1,
  # Interaction term: wind speed × kick distance
  wind_speed_x_kick_distance = 20 * (yardline_100 + 17),
  # Interaction term: cold × windy_day = 1 × 1 = 1
  cold_x_wind = 1
)

# predict field goal probabilities using model
# Step 1: Predict raw probabilities using your logistic regression model
raw_probs <- predict(log_model, newdata = kc_fg_inputs, type = "prob")[, "yes"]

kc_fg_predictions <- kc_fg_inputs %>%
  mutate(
    FG_Prob = predict(platt_model, newdata = data.frame(raw_probs_val = raw_probs), type = "response")
  )

# Build EP reference table for the *opposing team* starting a drive
# - Only consider normal 1st-and-10 plays
# - Filter out end-of-half and garbage time scenarios
# - Restrict to reasonable yardlines (0 to 60 yards from opponent’s end zone)
# Step 1: Create a cleaned table of average EP for 1st & 10 drives
# This is from the *opponent's perspective*, from yardlines 20–93 (realistic takeover range)
miss_spot_ep <- pbp %>%
  filter(
    down == 1,
    ydstogo == 10,
    !is.na(yardline_100),
    !is.na(ep),
    half_seconds_remaining > 120,  # Remove final 2 mins of half
    game_seconds_remaining > 300 | abs(score_differential) <= 14  # Remove blowouts
  ) %>%
  mutate(own_yardline_100 = yardline_100) %>%
  filter(own_yardline_100 >= 7, own_yardline_100 <= 80) %>%  # Opponent takes over between own 20–93
  group_by(own_yardline_100) %>%
  summarize(opponent_EP = mean(ep, na.rm = TRUE), .groups = "drop")

# Step 2: For kicks from yardline_100 = 0 to 60, calculate takeover spot if missed
miss_ep_table <- tibble(yardline_100 = 0:60) %>%
  mutate(
    kick_spot = yardline_100 + 7,
    spot_after_miss = pmax(20, 100 - kick_spot),  # Opponent gets ball at this yardline_100
    kick_yardline = 100 - kick_spot               # This is LOS before kick (offensive yardline)
  )


# Step 1: Build a lookup table of average EP for opponent after taking over at each field position
# Only include 1st & 10 plays, non-garbage time, with valid EP values
miss_spot_ep <- pbp %>%
  filter(
    down == 1,
    ydstogo == 10,
    !is.na(yardline_100),
    !is.na(ep),
    half_seconds_remaining > 120,  # Exclude final 2 mins of half
    game_seconds_remaining > 300 | abs(score_differential) <= 14  # Exclude garbage time
  ) %>%
  mutate(own_yardline_100 = yardline_100) %>%
  filter(own_yardline_100 >= 20, own_yardline_100 <= 93) %>%  # Opponent field positions
  group_by(own_yardline_100) %>%
  summarize(opponent_EP = mean(ep, na.rm = TRUE), .groups = "drop")

# Step 2: For each FG attempt yardline (0–60), calculate:
# - Kick spot (snap + 7)
# - Where the opponent would get the ball if missed
# - The actual LOS (kick_yardline) for presentation
miss_ep_table <- tibble(yardline_100 = 0:60) %>%
  mutate(
    kick_spot = yardline_100 + 7,
    # If the kick is from inside the opponent’s 13 (kick_spot < 20), opponent starts at their 20
    spot_after_miss = if_else(kick_spot < 20, 80, 100 - kick_spot),
    kick_yardline = 100 - kick_spot  # LOS from offense's perspective
  )

# Step 3: Merge with average opponent EP and flip sign (from kicker's POV)
miss_ep_table <- miss_ep_table %>%
  left_join(miss_spot_ep, by = c("spot_after_miss" = "own_yardline_100")) %>%
  mutate(
    EP_if_missed = -opponent_EP
  ) %>%
  select(yardline_100, kick_yardline, kick_spot, spot_after_miss, EP_if_missed)


# Join FG Probabilities with EP_if_missed
fg_ep_table <- kc_fg_predictions %>%
  select(yardline_100, FG_Prob) %>%
  left_join(miss_ep_table %>% select(yardline_100, EP_if_missed), by = "yardline_100") %>%
  mutate(
    FG_EP = FG_Prob * 3 + (1 - FG_Prob) * EP_if_missed
  )

# Merge FG and Go EP tables
ep_compare <- fg_ep_table %>%
  inner_join(go_ep_filtered, by = "yardline_100") %>%
  pivot_longer(cols = c(FG_EP, Go_EP), names_to = "Decision", values_to = "EP") %>%
  mutate(
    Decision = recode(Decision,
                      "FG_EP" = "Field Goal",
                      "Go_EP" = "Go for it")
  )

# Plot with shaded decision regions, vertical lines, and labels
ggplot(ep_compare, aes(x = yardline_100, y = EP, color = Decision)) +
  
  # Shade 0–16: Go for it
  annotate("rect", xmin = 0, xmax = 16, ymin = -Inf, ymax = Inf, alpha = 0.1, fill = "#1f77b4") +
  # Shade 16–31: Kick
  annotate("rect", xmin = 16, xmax = 31, ymin = -Inf, ymax = Inf, alpha = 0.1, fill = "#d62728") +
  # Shade 31+: Go for it
  annotate("rect", xmin = 31, xmax = 60, ymin = -Inf, ymax = Inf, alpha = 0.1, fill = "#1f77b4") +
  
  # Vertical dashed lines at decision thresholds
  geom_vline(xintercept = 16, linetype = "dashed", size = 1, color = "black") +
  geom_vline(xintercept = 31, linetype = "dashed", size = 1, color = "black") +
  
  # Yardline labels
  annotate("text", x = 16, y = min(ep_compare$EP) + 0.25, label = "16 yd", angle = 90, vjust = -0.5, size = 4, fontface = "bold") +
  annotate("text", x = 31, y = min(ep_compare$EP) + 0.25, label = "31 yd", angle = 90, vjust = -0.5, size = 4, fontface = "bold") +
  
  # Smooth EP curves
  geom_smooth(se = FALSE, method = "loess", span = 0.3, size = 1.2) +
  
  # Strategy zone labels
  annotate("text", x = 8, y = max(ep_compare$EP), label = "Go", color = "#1f77b4", size = 5, fontface = "bold") +
  annotate("text", x = 23.5, y = max(ep_compare$EP), label = "Kick", color = "#d62728", size = 5, fontface = "bold") +
  annotate("text", x = 45, y = max(ep_compare$EP), label = "Go", color = "#1f77b4", size = 5, fontface = "bold") +
  
  # Labels and theme
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
    legend.position = "top",   # Add the legend
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 11)
  )