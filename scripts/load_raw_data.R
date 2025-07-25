library(tidyverse)
library(nflreadr)
library(nflfastR)
library(nflplotR)

season_start <- 2022
season_end <- 2024

# load file that contains pbp columns that I want to keep in pbp raw data
pbp_column_selection_data <- read_csv("data/processed/field_descriptions_tagged_kicking.csv") %>%
  filter(Include == 1)
keep_columns <- pbp_column_selection_data$Field

# pull pbp data for all field goal attempts
pbp_raw_data <- nflfastR::load_pbp(season_start:season_end) %>%
  select(all_of(keep_columns))
pbp_fg_data_raw <- pbp_raw_data %>%
  filter(play_type == "field_goal")

# blank and NA summary function - figure out what columns have missing or blank data
summarize_blanks <- function(df) {
  data.frame(
    column = names(df),
    na_or_blank = sapply(df, function(col) {
      if (is.character(col)) {
        sum(is.na(col) | col == "")
      } else {
        sum(is.na(col))
      }
    }),
    total = nrow(df)
  ) %>%
    mutate(
      percent_missing = round(100 * na_or_blank / total, 1)
    ) %>%
    arrange(desc(percent_missing))
}

# blank_summary_pbpfgraw <- summarize_blanks(pbp_fg_data_raw)

# from the blank and na summary we see that temp, wind, surface, and some weather data is missing ... we know that we will definitely want these features in our model so we will do some data cleaning now

# first let's create a function that parses out the weather string...this will be used to create new columns for weather data and eliminate some blanks / built to handle most edge cases will address other in feature engineering

parse_weather <- function(weather_str) {
  weather_str <- ifelse(is.na(weather_str), "", weather_str)
  
  tibble(raw = weather_str) %>%
    mutate(
      forecast_parsed = str_extract(raw, "^(.*?)(?=Temp:)") %>% str_trim(),
      temp_parsed = str_extract(raw, "(?<=Temp: )[0-9]+") %>% as.numeric(),
      humidity_parsed = str_extract(raw, "(?<=Humidity: )[0-9]+") %>% as.numeric(),
      
      # Extract wind phrase like "Wind: South 9 mph"
      wind_phrase = str_extract(raw, "Wind: [^,]+"),
      
      # Handle known broken formats
      wind_direction_parsed = case_when(
        str_detect(wind_phrase, "Wind: (N/A|mph|\\d+ mph)") ~ NA_character_,
        TRUE ~ str_extract(wind_phrase, "(?<=Wind: )[A-Za-z]+")
      ),
      
      wind_speed_parsed = case_when(
        str_detect(wind_phrase, "mph") ~ str_extract(wind_phrase, "\\d+") %>% as.numeric(),
        TRUE ~ NA_real_
      )
    ) %>%
    select(-raw, -wind_phrase)
}

weather_parsed <- parse_weather(pbp_fg_data_raw$weather)

pbp_fg_data_raw_two <- bind_cols(pbp_fg_data_raw, weather_parsed)

test <- pbp_fg_data_raw_two %>%
  select(weather, forecast_parsed, temp_parsed, humidity_parsed, wind_direction_parsed, wind_speed_parsed) %>%
  filter(is.na(wind_speed_parsed))

blk_two <- summarize_blanks(pbp_fg_data_raw_two)

# stadium surface is missing for a couple of field so we will create a reference table (stadium_id, season, surface) then merge it into the working dataset

# Deal with international stadiums / manually code the missing ones...brazil: grass, allianz arenaL grass:, azteca: grass, london games have been using turf in 2022 and 2023 per reports

stadium_data <- load_schedules(season_start:season_end)

# some stadium don't have a surface for a game so we will use the surface type that the stadium has used in other games for that current season
nfl_stadium_data <- stadium_data %>%
  filter(location != "Neutral" & surface != "") %>%
  distinct(season, stadium_id, surface)

# some international stadiums don't have a surface type listed in the stadium data, so we will create a reference table for those stadiums. We inferenced the surface type for that game from avaliable information online
# add surface directly to the intl stadium dataframe
intl_stadium_data <- stadium_data %>%
  filter(game_type == 'REG' & location == "Neutral" & surface =="") %>%
  distinct(season, stadium_id, surface) %>%
  mutate(surface = case_when(
    stadium_id == "MEX00" ~ "grass",
    stadium_id == "SAO00" ~ "grass",
    stadium_id == "GER00" ~ "grass",
    stadium_id == "LON00" ~ "turf",
    stadium_id == "LON02" ~ "turf",
    TRUE ~ NA_character_
  ))

stadium_surface_data <- bind_rows(nfl_stadium_data, intl_stadium_data) %>% rename(surface_type = surface)

# merge the stadium surface data into the pbp data
pbp_fg_data_raw_three <- pbp_fg_data_raw_two %>%
  left_join(stadium_surface_data, by = c("stadium_id", "season"))




  
