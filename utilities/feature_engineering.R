# Prepares field goal data for modeling by engineering features, normalizing weather values, and flagging game context (e.g., indoor, high leverage).
# Run this before training any models — gives a clean, ready-to-use dataset

prepare_fg_features <- function(season_start = 2013, season_end = 2024, save_csv = FALSE) {
  
  # Load required libraries
  library(tidyverse)
  library(janitor)
  library(lubridate)
  library(broom)
  library(nflreadr)
  library(nflfastR)
  library(nflplotR)
  
  # Pull in custom helper functions
  source("utilities/load_fg_data.R")
  source("utilities/helpers.R")
  # Avoid scientific notation in outputs
  options(scipen = 999)
  
  # Pull cleaned field goal attempts from play-by-play data
  fg_data <- load_fg_data(season_start, season_end)
  
  # Create outcome variable for whether the field goal was made
  fg_df <- fg_data %>%
    mutate(fg_made = ifelse(field_goal_result == 'made', 1, 0)) %>%
    select(-field_goal_result)
  
  # Flag stadium conditions and altitude
  fg_df <- fg_df %>%
    mutate(
      Indoor = ifelse(roof %in% c('dome', 'closed'), 1, 0),
      Turf = ifelse(surface_type == 'grass', 0, 1),
      Altitude = ifelse(stadium_id %in% c("DEN00", "MEX00"), 1, 0)
    )
  
  # Assign default weather conditions for indoor games (climate-controlled)
  dome_df <- fg_df %>%
    filter(Indoor == 1) %>%
    mutate(
      temp_parsed = 68,
      humidity_parsed = 70,
      wind_speed_parsed = 0,
      Precip = 0
    )
  
  # Parse real weather conditions for outdoor games 
  # (only set Precip flag if rain/snow is actually reported — not just "chance of")
  # Also backfill missing weather values using available or default values
  outdoor_df <- fg_df %>%
    filter(Indoor == 0) %>%
    mutate(
      Precip = case_when(
        str_detect(str_to_lower(forecast_parsed), "rain|drizzle|shower") &
          !str_detect(str_to_lower(forecast_parsed), "chance|%|likely") ~ 1,
        str_detect(str_to_lower(forecast_parsed), "snow|flurries|blizzard") &
          !str_detect(str_to_lower(forecast_parsed), "chance|%|likely") ~ 1,
        TRUE ~ 0
      ),
      temp_parsed = ifelse(is.na(temp_parsed) | temp_parsed == "", temp, temp_parsed),
      wind_speed_parsed = ifelse(is.na(wind_speed_parsed) | wind_speed_parsed == "", wind, wind_speed_parsed),
      temp_parsed = ifelse(is.na(temp_parsed) | temp_parsed == "", 68, temp_parsed),
      humidity_parsed = ifelse(is.na(humidity_parsed) | humidity_parsed == "", 70, humidity_parsed),
      wind_speed_parsed = ifelse(is.na(wind_speed_parsed) | wind_speed_parsed == "", 0, wind_speed_parsed)
    )
  
  # Merge indoor and outdoor data into one cleaned dataframe
  fg_df <- bind_rows(dome_df, outdoor_df)
  
  # Parse kickoff time and flag games played at night (6 PM or later)
  fg_df <- fg_df %>%
    mutate(
      start_time_clean = str_replace(start_time, ",", ""),
      start_datetime = parse_date_time(start_time_clean, orders = "mdy HMS"),
      night_game = if_else(hour(start_datetime) >= 18, 1, 0)
    ) %>%
    select(-start_time_clean, -start_datetime)
  
  # Flag high-leverage kicks (close games late in 4th or any OT)
  fg_df <- fg_df %>%
    mutate(
      high_leverage = case_when(
        (score_differential >= -3 & score_differential <= 4) & qtr == 4 & game_seconds_remaining <= 360 ~ 1,
        qtr >= 5 ~ 1,
        TRUE ~ 0
      )
    )
  
  # Flag away games
  fg_df <- fg_df %>%
    mutate(away_game = ifelse(posteam_type == "away", 1, 0))
  
  # Flag playoff games
  fg_df <- fg_df %>%
    mutate(playoffs = ifelse(season_type == "POST", 1, 0))
  
  # Bucket temperatures into Cold (<50), Moderate (50–79), and Hot (80+)
  fg_df <- fg_df %>%
    mutate(
      temp_bucket = case_when(
        temp_parsed < 50 ~ "Cold",
        temp_parsed >= 50 & temp_parsed < 80 ~ "Moderate",
        temp_parsed >= 80 ~ "Hot",
        TRUE ~ NA_character_
      ),
      cold = ifelse(temp_bucket == "Cold", 1, 0),
      hot = ifelse(temp_bucket == "Hot", 1, 0)
    )
  
  # Flag games with high humidity (80%+)
  fg_df <- fg_df %>%
    mutate(humid_game = ifelse(humidity_parsed >= 80, 1, 0))
  
  # Bucket wind speeds into Low (≤5), Moderate (6–15), and High (16+)
  fg_df <- fg_df %>%
    mutate(
      wind_bucket = case_when(
        wind_speed_parsed <= 5 ~ "Low",
        wind_speed_parsed > 5 & wind_speed_parsed <= 15 ~ "Moderate",
        wind_speed_parsed > 15 ~ "High",
        TRUE ~ NA_character_
      ),
      moderate_wind = ifelse(wind_bucket == "Moderate", 1, 0),
      high_wind = ifelse(wind_bucket == "High", 1, 0)
    )
  
  # Create a simpler 'windy day' variable to help the model pick up broader trends
  fg_df <- fg_df %>%
    mutate(windy_day = ifelse(wind_speed_parsed >= 10, 1, 0))
  
  # Create interaction variables that may help the model learn weather-distance effects
  fg_df <- fg_df %>%
    mutate(
      wind_speed_x_kick_distance = wind_speed_parsed * kick_distance,
      cold_x_wind = cold * windy_day)
  
  # Save cleaned data to CSV if requested
  if (save_csv) {
    write.csv(fg_df, "data/fg_data_cleaned.csv", row.names = FALSE)
  }
  return(fg_df)
}
