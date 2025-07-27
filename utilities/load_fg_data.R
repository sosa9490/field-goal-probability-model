# Pull, clean, and adds variables to NFL field goal play-by-play data for a given season range

# Filters for FG attempts, adds weather and surface info, and keeps only tagged variables

# Example usage:
# source("utilities/load_fg_data.R")
# pbp_fg_data <- load_fg_data(2010, 2024)

load_fg_data <- function(start_year, end_year) {
  
  # Load necessary libraries
  library(tidyverse)
  library(nflreadr)
  library(nflfastR)
  library(nflplotR)
  library(stringr)
  
  # Load helper functions for weather parsing and summarizing blanks
  source("utilities/helpers.R")
  
  # Load the manually tagged column list and keep only the ones marked for modeling
  pbp_column_tags <- read_csv("data/field_descriptions_tagged_kicking.csv") %>%
    filter(Include == 1)
  
  selected_columns <- pbp_column_tags$Field
  
  # Pull PBP data for the selected seasons and keep only the selected columns
  pbp_all <- load_pbp(start_year:end_year) %>%
    select(all_of(selected_columns))
  
  # Filter for just field goal attempts and trim any whitespace in surface column
  pbp_fg <- pbp_all %>%
    filter(play_type == "field_goal") %>%
    mutate(surface = str_trim(surface))
  
  # Parse the free-text weather string into structured columns (e.g., wind, temp, etc.)
  # This help fill in missing weather-related values that we know we will want later
  weather_data <- parse_weather(pbp_fg$weather)
  
  # Join parsed weather columns back onto the FG data
  pbp_fg_weather <- bind_cols(pbp_fg, weather_data)
  
  # Pull schedule data to get official surface info per game
  schedule_data <- load_schedules(start_year:end_year)
  
  # Build surface reference table for domestic games with known surfaces
  nfl_stadium_surface <- schedule_data %>%
    filter(location != "Neutral", surface != "") %>%
    distinct(season, stadium_id, surface)
  
  # Do the same for international games with complete surface info
  intl_stadium_surface_complete_data <- schedule_data %>%
    filter(game_type == "REG", location == "Neutral", surface != "") %>%
    distinct(season, stadium_id, surface)
  
  # Fill in missing international surface types manually based on known stadium info
  # This was based on looking up the stadium and year online
  intl_stadium_surface_missing_data <- schedule_data %>%
    filter(game_type == "REG", location == "Neutral", surface == "") %>%
    distinct(season, stadium_id) %>%
    mutate(surface = case_when(
      stadium_id == "MEX00" ~ "grass",   # Estadio Azteca
      stadium_id == "SAO00" ~ "grass",   # Corinthians Arena
      stadium_id == "GER00" ~ "grass",   # Allianz Arena
      stadium_id == "LON00" ~ "turf",    # London 
      stadium_id == "LON02" ~ "turf",    # London
      TRUE ~ NA_character_
    ))
  
  # Combine all international surface data (complete + filled)
  intl_stadium_surface <- bind_rows(intl_stadium_surface_complete_data, intl_stadium_surface_missing_data) %>%
    distinct(season, stadium_id, surface)
  
  # Merge domestic and international surfaces into a master reference table
  # Remove any duplicate season/stadium combinations by keeping the first one
  stadium_surface_reference <- bind_rows(nfl_stadium_surface, intl_stadium_surface) %>%
    rename(surface_type = surface) %>%
    group_by(season, stadium_id) %>%
    slice_head(n=1) %>%
    ungroup()
  
  # Fill in any missing surface info using the reference table
  pbp_fg_final <- pbp_fg_weather %>%
    left_join(stadium_surface_reference, by = c("season", "stadium_id")) %>%
    mutate(surface_type = ifelse(is.na(surface) | surface == "", surface_type, surface)) %>%
    select(-surface)  
  
  # Final column selection — keep only what's useful for EDA, feature work, and modeling
  pbp_fg_final <- pbp_fg_final %>%
    select(
      play_id, game_id, season, season_type, start_time, posteam, posteam_type, game_date, qtr, game_seconds_remaining,
      play_type, field_goal_result, kick_distance, score_differential, ep, kicker_player_id, kicker_player_name,
      stadium_id, weather, roof, temp, wind,
      forecast_parsed, temp_parsed, humidity_parsed, wind_direction_parsed, wind_speed_parsed,
      surface_type
    )
  
  return(pbp_fg_final)
}

# Example usage:
#pbp_fg_data <- load_fg_data(2010, 2024)
#write.csv(pbp_fg_data, "data/pbp_fg_data.csv")