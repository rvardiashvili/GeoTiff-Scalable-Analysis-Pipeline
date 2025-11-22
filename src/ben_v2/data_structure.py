# src/ben_v2/data_structure.py

# This file defines the directory structure for Sentinel-1 and Sentinel-2 data.
# The paths are defined as glob patterns relative to the tile folder
# provided to the main script.

# Example tile folder: /home/rati/bsc_thesis/sentinel-2/T32UMD_20230608T102601

# Pattern to find Sentinel-1 band files.
# {band_name} will be replaced with 'VV' or 'VH'.
S1_BAND_PATTERN = "S1*.SAFE/measurement/*{band_name}*.tiff"

# Pattern to find Sentinel-2 band files.
# {band_name} will be replaced with 'B01', 'B02', etc.
#S2_BAND_PATTERN = "S2*.SAFE/GRANULE/*/IMG_DATA/*{band_name}*.jp2"

# Pattern to find Sentinel-2 band files.
# {band_name} will be replaced with 'B01', 'B02', etc.
S2_BAND_PATTERN = "*{band_name}*.jp2"