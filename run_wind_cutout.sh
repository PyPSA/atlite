#!/bin/bash

# Ensure the script is called with the required arguments
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <lat1> <lon1> <lat2> <lon2> <start_date> <end_date>"
    exit 1
fi

# Capture arguments
LAT1=$1
LON1=$2
LAT2=$3
LON2=$4
START_DATE=$5
END_DATE=$6

# Navigate to the appropriate directory
cd cd_atlite || { echo "Error: Could not change to directory 'cd_atlite'"; exit 1; }

# Activate the virtual environment
source venv/bin/activate || { echo "Error: Could not activate virtual environment"; exit 1; }

# Run the Python script with the provided arguments
python3 wind_cutout.py "$LAT1" "$LON1" "$LAT2" "$LON2" "$START_DATE" "$END_DATE" || {
    echo "Error: Failed to run wind_cutout.py"
    exit 1
}

# Print success message
echo "Wind cutout process completed successfully."
