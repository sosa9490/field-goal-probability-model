# Field Goal EDA and Feature Engineering Script
# Purpose: Prepare features to model FG success

# Load required libraries
library(tidyverse)
library(janitor)
library(lubridate)
library(broom)

# load the utility function to load FG data
source("utilities/load_fg_data.R")
source("utilities/helpers.R")

options(scipen = 999)

# load the raw data
fg_data <- load_fg_data(2013, 2023)

# create a binary outcome variable for FG success
fg_df <- fg_data %>%
  mutate(fg_made = ifelse(field_goal_result == 'made', 1, 0)) %>%
  select(-field_goal_result)

# some feature engineering for variables we definitely want: Indoor stadium, grass or turf, high altitude (Den / Mexico)
# add a precipitation variable placeholder

fg_df <- fg_df %>%
  mutate(
    Indoor = ifelse(roof %in% c('dome', 'closed'), 1, 0),
    Turf = ifelse(surface_type == 'grass', 0, 1),
    Altitude = ifelse(stadium_id %in% c("DEN00", "MEX00"), 1, 0)
  )

# fixed weather of domes
# for all indoor stadiums (dome and closed roofs), we will set temp to 68, humidity to 70, and wind to 0
dome_df <- fg_df %>%
  filter(Indoor == 1) %>%
  mutate(
    temp_parsed = 68,
    humidity_parsed = 70,
    wind_speed_parsed = 0,
    Precip = 0
  )

#dome_summary <- summarize_blanks(dome_df)

# for all outdoor stadium we need to create variable `Precip` that flags games where it actually rained
#or snowed, based on the `forecast_parsed` field
# Only flag rain/snow if it actually occurred — not if it's just a forecast.
# Phrases like “30% chance of rain” or “chance of flurries” should NOT be flagged.
# Match terms like "rain", "drizzle", "snow", "flurries", "showers", etc.
# Exclude if terms like "chance", "%", or "likely" are also present

outdoor_df <- fg_df %>%
  filter(Indoor == 0) %>%
  mutate(
    Precip = case_when(
      str_detect(str_to_lower(forecast_parsed), "rain|drizzle|shower") &
        !str_detect(str_to_lower(forecast_parsed), "chance|%|likely") ~ 1,
      
      str_detect(str_to_lower(forecast_parsed), "snow|flurries|blizzard") &
        !str_detect(str_to_lower(forecast_parsed), "chance|%|likely") ~ 1,
      
      TRUE ~ 0
    )
  )

# if temp_parsed is na or blank but temp is not then use temp value to fill temp_parsed...same concept for wind_speed_parsed

# outdoor data has some missing values related to retractable roof stadium we will fill those in with our standard dome estimates
outdoor_df <- outdoor_df %>%
  mutate(
    temp_parsed = ifelse(is.na(temp_parsed) | temp_parsed == "", temp, temp_parsed),
    wind_speed_parsed = ifelse(is.na(wind_speed_parsed) | wind_speed_parsed == "", wind, wind_speed_parsed)
  ) %>% mutate(
    temp_parsed = ifelse(is.na(temp_parsed) | temp_parsed == "", 68, temp_parsed),
    humidity_parsed = ifelse(is.na(humidity_parsed) | humidity_parsed == "", 70, humidity_parsed),
    wind_speed_parsed = ifelse(is.na(wind_speed_parsed) | wind_speed_parsed == "", 0, wind_speed_parsed)
  )

# combine the two datasets
# create column to flag as night game if kickoff hour is 6 PM or later
fg_df <- bind_rows(dome_df, outdoor_df)

fg_df <- fg_df %>%
  mutate(
    start_time_clean = str_replace(start_time, ",", ""),
    start_datetime = parse_date_time(start_time_clean, orders = "mdy HMS"),
    # Flag as night game if kickoff hour is 6 PM or later
    night_game = if_else(hour(start_datetime) >= 18, 1, 0)
  ) %>% select(-start_time_clean, -start_datetime)

# create a column detailing high leverage situations: score differential between 4 and -3 with 6 minutes or less remaining in the game...arbitrary for now but captures a lot of the high pressure situations and should be useful for testing...all overtime in high leverage situations

fg_df <- fg_df %>%
  mutate(
    high_leverage = case_when(
      (score_differential >= -3 & score_differential <= 4) & qtr == 4 & game_seconds_remaining <= 360 ~ 1,
      qtr >= 5 ~ 1,
      TRUE ~ 0))

# create a column for home / away
fg_df <- fg_df %>%
  mutate(away_game = ifelse(posteam_type == "away", 1, 0))

# create a column for playoffs
fg_df <- fg_df %>%
  mutate(playoffs = ifelse(season_type == "POST", 1, 0))
# create a column for home / away

#### WEATHER FEATURE ENGINEERING ####

# Visualize how field goal success rate varies with temperature
# This smoothed plot helps identify natural inflection points

fg_df %>%
  filter(!is.na(temp_parsed)) %>%
  ggplot(aes(x = temp_parsed, y = fg_made)) +
  geom_smooth(method = "loess", se = FALSE, color = "steelblue") +
  labs(
    title = "Smoothed FG Success Rate by Temperature",
    x = "Temperature (°F)",
    y = "Probability FG is Made"
  ) +
  theme_minimal()

# Analyze sample size and performance by temperature bucket
# Used to validate that bucket boundaries are meaningful and well-supported

temp_sample_size <- fg_df %>%
  filter(!is.na(temp_parsed)) %>%
  mutate(temp_bucket = case_when(
    temp_parsed < 10 ~ "Extreme Cold",                    
    temp_parsed >= 10 & temp_parsed < 40 ~ "Cold",       
    temp_parsed >= 40 & temp_parsed < 70 ~ "Moderate",     
    temp_parsed >= 70 ~ "Hot"                              
  )) %>%
  group_by(temp_bucket) %>%
  summarise(
    Attempts = n(),                         
    Makes = sum(fg_made),                   
    SuccessRate = round(Makes / Attempts, 3)) %>%
  arrange(temp_bucket)

# Create temperature buckets based on observed success rate trends and logical cutoffs:
# - Cold: <50°F — performance dips noticeably here
# - Moderate: 50–79°F — most consistent performance, includes climate-controlled indoor games
# - Hot: 80°F+ — performance drops again, likely due to extreme outdoor heat

fg_df <- fg_df %>%
  mutate(temp_bucket = case_when(
    temp_parsed < 50 ~ "Cold",
    temp_parsed >= 50 & temp_parsed < 80 ~ "Moderate",
    temp_parsed >= 80 ~ "Hot",
    TRUE ~ NA_character_
  ))
# create two new columns for temperature buckets
fg_df <- fg_df %>%
  mutate(
    cold = ifelse(temp_bucket == "Cold", 1, 0),
    hot = ifelse(temp_bucket == "Hot", 1, 0)
  )

# Visualize field goal success rate across humidity levels
# Use a smoothed curve to identify inflection points in performance

fg_df %>%
  filter(!is.na(humidity_parsed)) %>%
  ggplot(aes(x = humidity_parsed, y = fg_made)) +
  geom_smooth(method = "loess", se = FALSE, color = "darkgreen") +
  labs(
    title = "Smoothed FG Success Rate by Humidity",
    x = "Humidity (%)",
    y = "Probability FG is Made"
  ) +
  theme_minimal()

# Calculate sample size and success rate for proposed humidity buckets
# This helps validate that the visual trends are supported by data

humidity_sample_size <- fg_df %>%
  filter(!is.na(humidity_parsed)) %>%
  mutate(humidity_bucket = case_when(
    humidity_parsed < 40 ~ "Dry",
    humidity_parsed >= 40 & humidity_parsed < 70 ~ "Moderate",
    humidity_parsed >= 70 & humidity_parsed < 90 ~ "Humid",
    humidity_parsed >= 90 ~ "Very Humid"
  )) %>%
  group_by(humidity_bucket) %>%
  summarise(
    Attempts = n(),
    Makes = sum(fg_made),
    SuccessRate = round(Makes / Attempts, 3)
  ) %>%
  arrange(humidity_bucket)

# Create binary humidity variable
# Use 80% as the cutoff to avoid misclassifying indoor dome games
# and to isolate true high-humidity outdoor environments

fg_df <- fg_df %>%
  mutate(humid_game = ifelse(humidity_parsed >= 80, 1, 0))

fg_df %>%
  filter(!is.na(wind_speed_parsed)) %>%
  ggplot(aes(x = wind_speed_parsed, y = fg_made)) +
  geom_smooth(method = "loess", se = FALSE, color = "steelblue") +
  labs(
    title = "Smoothed FG Success Rate by Wind Speed",
    x = "Wind Speed (mph)",
    y = "Probability FG is Made"
  ) +
  theme_minimal()

# Calculate sample size and FG success rate by wind bucket
# This helps validate performance patterns and ensure each bin has sufficient data

wind_sample_size <- fg_df %>%
  filter(!is.na(wind_speed_parsed)) %>%
  mutate(wind_bucket = case_when(
    wind_speed_parsed <= 5 ~ "Low",
    wind_speed_parsed > 5 & wind_speed_parsed <= 15 ~ "Moderate",
    wind_speed_parsed > 15 ~ "High"
  )) %>%
  group_by(wind_bucket) %>%
  summarise(
    Attempts = n(),
    Makes = sum(fg_made),
    SuccessRate = round(Makes / Attempts, 3)
  ) %>%
  arrange(wind_bucket)

# Create three wind speed buckets based on performance patterns
# - Low: 0–5 mph (highest success rate, stable conditions)
# - Moderate: 6–15 mph (slight performance drop)
# - High: 16+ mph (lower success rate, potentially volatile)

fg_df <- fg_df %>%
  mutate(wind_bucket = case_when(
    wind_speed_parsed <= 5 ~ "Low",
    wind_speed_parsed > 5 & wind_speed_parsed <= 15 ~ "Moderate",
    wind_speed_parsed > 15 ~ "High",
    TRUE ~ NA_character_
  ))

fg_df <- fg_df %>%
  mutate(moderate_wind = ifelse(wind_bucket == "Moderate", 1, 0),
         high_wind = ifelse(wind_bucket == "High", 1, 0))

# create interaction variables
# wind_speed x kick_distance...wind speed is probably a bigger factor on longer kicks
fg_df <- fg_df %>%
  mutate(wind_speed_x_kick_distance = wind_speed_parsed * kick_distance)

fg_df <- fg_df %>%
  select(play_id:stadium_id, fg_made:playoffs,cold:humid_game,moderate_wind,high_wind, wind_speed_x_kick_distance)

write.csv(fg_df, "data/processed/fg_data_cleaned.csv")

# test for missing data: test_blanks <- summarize_blanks(fg_df)

# write this to a csv file for now to brainstorm next part





######## testing
# 1. Fit logistic regression model with high_leverage as predictor
model <- glm(fg_made ~ kick_distance + cold + Precip + wind_speed_parsed + Indoor, data = test_df, family = "binomial")

# 2. Summarize model output to check p-value and coefficient
summary(model)

# 3. Optional: Clean output using broom for easier reading
tidy_model <- tidy(model)
print(tidy_model)













 