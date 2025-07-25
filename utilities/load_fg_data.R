# This function pulls, cleans, and enriches NFL field goal play-by-play data for a given range of seasons.
# It keeps only tagged columns, filters for field goal attempts, parses weather strings, and fills in missing stadium surface info.
# example usage: 
#source("utilities/load_fg_data.R")
#pbp_fg_data <- load_fg_data(2010, 2024)

load_fg_data <- function(start_year, end_year) {
  
  # Load required libraries
  library(tidyverse)
  library(nflreadr)
  library(nflfastR)
  library(nflplotR)
  library(stringr)
  
  # Load helper functions for weather parsing and blank summaries
  source("utilities/helpers.R")
  
  # Load file that contains tagged metadata for PBP columns
  # This file was created manually to tag which columns we actually want to keep
  pbp_column_tags <- read_csv("data/processed/field_descriptions_tagged_kicking.csv") %>%
    filter(Include == 1)
  
  selected_columns <- pbp_column_tags$Field
  
  # Pull play-by-play data for the given seasons
  # Only keep columns that were marked as Include = 1 in the metadata
  pbp_all <- load_pbp(start_year:end_year) %>%
    select(all_of(selected_columns))
  
  # Filter for field goal attempts only
  # Also remove any leading/trailing spaces in the surface column
  pbp_fg <- pbp_all %>%
    filter(play_type == "field_goal") %>%
    mutate(surface = str_trim(surface))
  
  # Parse the weather string into structured columns (temp, humidity, wind, etc.)
  # This helps us fill in missing weather-related values that we know we'll want later
  weather_data <- parse_weather(pbp_fg$weather)
  
  # Join parsed weather columns to the main field goal dataset
  pbp_fg_weather <- bind_cols(pbp_fg, weather_data)
  
  # Load schedule data so we can reference surface type for each stadium and game
  schedule_data <- load_schedules(start_year:end_year)
  
  # Create surface reference table for domestic games that already have surface info
  nfl_stadium_surface <- schedule_data %>%
    filter(location != "Neutral", surface != "") %>%
    distinct(season, stadium_id, surface)
  
  # Do the same for international games that already have surface info
  intl_stadium_surface_complete_data <- schedule_data %>%
    filter(game_type == "REG", location == "Neutral", surface != "") %>%
    distinct(season, stadium_id, surface)
  
  # For international games with missing surface info, we manually assign the correct surface
  # This was based on looking up the stadium and year online
  intl_stadium_surface_missing_data <- schedule_data %>%
    filter(game_type == "REG", location == "Neutral", surface == "") %>%
    distinct(season, stadium_id) %>%
    mutate(surface = case_when(
      stadium_id == "MEX00" ~ "grass",   # Estadio Azteca
      stadium_id == "SAO00" ~ "grass",   # Corinthians Arena
      stadium_id == "GER00" ~ "grass",   # Allianz Arena
      stadium_id == "LON00" ~ "turf",    # Tottenham Hotspur
      stadium_id == "LON02" ~ "turf",    # Wembley
      TRUE ~ NA_character_
    ))
  
  # Combine international surface data (complete + missing that we just filled)
  intl_stadium_surface <- bind_rows(intl_stadium_surface_complete_data, intl_stadium_surface_missing_data) %>%
    distinct(season, stadium_id, surface)
  
  # Combine domestic and international surface records into one reference table
  # Remove any duplicate season/stadium combinations by keeping the first one
  stadium_surface_reference <- bind_rows(nfl_stadium_surface, intl_stadium_surface) %>%
    rename(surface_type = surface) %>%
    group_by(season, stadium_id) %>%
    slice_head(n=1) %>%
    ungroup()
  
  # Merge stadium surface type into the main field goal data
  # If surface is missing in the original PBP data, use the one from our reference table
  pbp_fg_final <- pbp_fg_weather %>%
    left_join(stadium_surface_reference, by = c("season", "stadium_id")) %>%
    mutate(surface_type = ifelse(is.na(surface) | surface == "", surface_type, surface)) %>%
    select(-surface)  # drop original surface column
  
  # Keep only columns we know we want for EDA, feature engineering, and modeling
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

#pbp_fg_data <- load_fg_data(2010, 2024)
#write.csv(pbp_fg_data, "data/processed/pbp_fg_data.csv")