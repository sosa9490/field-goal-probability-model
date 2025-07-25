# Load, clean, and enrich field goal play-by-play data for analysis

# Load necessary packages
library(tidyverse)
library(nflreadr)
library(nflfastR)
library(nflplotR)
library(stringr)

# Load reusable helper functions for parsing weather and summarizing blanks
source("utilities/helpers.R")

# Define the seasons for which we want to pull data
season_start <- 2010
season_end <- 2024


# Load file that contains tagged metadata for PBP columns
# This file was created manually to indicate which columns are useful for modeling
# We use only columns where Include == 1

pbp_column_tags <- read_csv("data/processed/field_descriptions_tagged_kicking.csv") %>%
  filter(Include == 1)

selected_columns <- pbp_column_tags$Field

# Pull play-by-play (PBP) data for the specified seasons
# Then filter to include only the selected columns from our metadata file
pbp_all <- nflfastR::load_pbp(season_start:season_end) %>%
  select(all_of(selected_columns))

# Filter down to only field goal attempts
# eliminate white space from 'surface' column
pbp_fg <- pbp_all %>%
  filter(play_type == "field_goal") %>%
  mutate(surface = str_trim(surface))


# Parse the unstructured weather string into structured columns
# This helps fill in blanks for weather-related fields like temperature, humidity, wind direction/speed
# parse_weather() is a helper function designed to handle common edge cases and malformed strings

weather_data <- parse_weather(pbp_fg$weather)

# Append parsed weather columns to our field goal PBP data
pbp_fg_weather <- bind_cols(pbp_fg, weather_data)

# Stadium surface is missing for a few games
# Since surface type is a key feature, we’ll clean and merge it in now
# using game-level schedule data from nflreadr to build a stadium reference table

schedule_data <- load_schedules(season_start:season_end)

# Create reference table for domestic games with valid surface info
nfl_stadium_surface <- schedule_data %>%
  filter(location != "Neutral", surface != "") %>%
  distinct(season, stadium_id, surface)

# Create reference table for international/neutral-site games that are missing surface info
# We manually code the surface type based on public information for these stadiums
intl_stadium_surface_complete_data <- schedule_data %>%
  filter(game_type == "REG", location == "Neutral", surface != "") %>%
  distinct(season, stadium_id, surface)

intl_stadium_surface_missing_data <- schedule_data %>%
  filter(game_type == "REG", location == "Neutral", surface == "") %>%
  distinct(season, stadium_id, surface) %>%
  mutate(surface = case_when(
    stadium_id == "MEX00" ~ "grass",   
    stadium_id == "SAO00" ~ "grass",   
    stadium_id == "GER00" ~ "grass",   
    stadium_id == "LON00" ~ "turf",    
    stadium_id == "LON02" ~ "turf",    
    TRUE ~ NA_character_
  ))
# combine both complete and missing international stadium surface data
intl_stadium_surface <- bind_rows(intl_stadium_surface_complete_data, intl_stadium_surface_missing_data) %>%
  distinct(season, stadium_id, surface)

# Combine both domestic and international stadium surface records / eliminate duplicates / Keep only one record per season/stadium
stadium_surface_reference <- bind_rows(nfl_stadium_surface, intl_stadium_surface) %>%
  rename(surface_type = surface) %>%
  group_by(season, stadium_id) %>%
  slice(1) %>%
  ungroup()

# Merge cleaned stadium surface data into the main field goal dataset
# create one surface variable...if surace is NA or blank, use the stadium surface reference
pbp_fg_final <- pbp_fg_weather %>%
  left_join(stadium_surface_reference, by = c("season", "stadium_id")) %>%
  mutate(surface_type = ifelse(is.na(surface) | surface == "", surface_type, surface)) %>%
  select(-surface)

#select only columns we want to keep for EDA / feature engineering and eventually modeling
pbp_fg_final <- pbp_fg_final %>%
  select(play_id, game_id, season_type, posteam, posteam_type, game_date, game_seconds_remaining, play_type, field_goal_result, kick_distance, score_differential, ep, kicker_player_id, stadium_id, weather, roof, temp, wind, forecast_parsed, temp_parsed, humidity_parsed, wind_direction_parsed, wind_speed_parsed, surface_type)

