# -----------------------------------------------
# Script: build_pbp_field_descriptions.R
# Purpose: Export nflfastR field descriptions with a blank category column for manual tagging and model variable selection.
# -----------------------------------------------

library(tidyverse)
library(nflfastR)

# Pull field descriptions from nflfastR package reference
fields_df <- nflfastR::field_descriptions

# Add a blank column to manually tag each field by category 
# (e.g., include, exclude) to streamline variable selection 
# for modeling and analysis.
fields_df <- fields_df %>%
  mutate(Include = '')

# Save as a CSV for review and annotation
write_csv(fields_df, "data/processed/field_descriptions.csv")