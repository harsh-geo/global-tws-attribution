#!/bin/bash
# preprocess_temperature.sh
# Regrids ERA5 temperature data to 0.5x0.5 degree using CDO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
RAW_DIR="${PROJECT_ROOT}/data/raw"
PROCESSED_DIR="${PROJECT_ROOT}/data/processed"

mkdir -p "${PROCESSED_DIR}"

INPUT_FILE="${RAW_DIR}/era5_t2m_2002_2019.nc"
OUTPUT_FILE="${PROCESSED_DIR}/era5_t2m_05deg.nc"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found. Did the download script finish?"
    exit 1
fi

echo "Regridding ERA5 2m Temperature to 0.5x0.5 degree grid..."
# r720x360 is CDO's built-in 0.5 deg global grid
cdo -s remapbil,r720x360 "$INPUT_FILE" "$OUTPUT_FILE"

if [ -f "$OUTPUT_FILE" ]; then
    echo "Successfully created $OUTPUT_FILE"
else
    echo "Failed to preprocess temperature data!"
    exit 1
fi
