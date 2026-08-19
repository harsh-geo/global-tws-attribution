#!/bin/bash
# download_oni.sh
# Downloads the Oceanic Niño Index (ONI) from NOAA

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
RAW_DIR="${PROJECT_ROOT}/data/raw"

mkdir -p "${RAW_DIR}"
cd "${RAW_DIR}" || exit 1

# Download the ONI data
echo "Downloading ONI data from NOAA..."
curl -sO https://psl.noaa.gov/data/correlation/oni.data

if [ -f "oni.data" ]; then
    echo "Successfully downloaded oni.data to ${RAW_DIR}"
else
    echo "Failed to download ONI data!"
    exit 1
fi
