# Script: build_pbp_field_descriptions.R
 
# Pulls the list of available play-by-play fields from the nflfastR package and adds a blank column to manually tag which fields should be included in the modeling process. This helps narrow down variables based on context, usefulness, and data quality before building models.

# Output: Saves a CSV file to /data that can be manually edited and used in later scripts.

library(tidyverse)
library(nflfastR)

# Pull field descriptions from nflfastR package reference
fields_df <- nflfastR::field_descriptions

# Add a blank column to manually tag each field by category 
#(e.g., include, exclude) to streamline variable selection 
# for modeling and analysis.
fields_df <- fields_df %>%
  mutate(Include = '')

# Save as a CSV for review and annotation
write_csv(fields_df, "data/field_descriptions.csv")