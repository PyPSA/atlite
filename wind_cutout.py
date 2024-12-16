import atlite
from math import cos, radians
import logging
import xarray as xr
from datetime import datetime
import subprocess

logging.basicConfig(level=logging.INFO)

import sys

# Check if the correct number of arguments are provided
if len(sys.argv) != 8:
    raise ValueError("Please provide exactly 7 arguments: lat1, lon1, lat2, lon2, startdate, enddate")

# Assign command line arguments to variables
lat1 = float(sys.argv[1])
lon1 = float(sys.argv[2])
lat2 = float(sys.argv[3])
lon2 = float(sys.argv[4])
startdate = sys.argv[5]
enddate = sys.argv[6]

# Calculate min and max latitudes and longitudes
min_lat = min(lat1, lat2)
max_lat = max(lat1, lat2)
min_lon = min(lon1, lon2)
max_lon = max(lon1, lon2)

nc_filename = f"{lat1}_{lon1}_{lat2}_{lon2}_{startdate}_{enddate}.nc"

cutout = atlite.Cutout(
    path=nc_filename,
    module="era5",
    x=slice(min_lon + 180, max_lon + 180),  # convert to 360 here, avoid differences between fetching & existing .nc paths
    y=slice(min_lat, max_lat),
    time=slice(startdate, enddate),
    dt="h",
)
cutout.prepare(show_progress=True)

# Print out the current time
current_time = datetime.now()
print(f"Post prepare time: {current_time}")

# Construct the shell command
gcloud_command = [
    "gcloud", "storage", "cp", nc_filename, "gs://era5_wind_cutouts"
]

# Run the command
try:
    result = subprocess.run(gcloud_command, capture_output=True, text=True)
    
    # Check if the command was successful
    if result.returncode == 0:
        print("File successfully uploaded to Google Cloud Storage.")
    else:
        print(f"Error: {result.stderr}")
except Exception as e:
    print(f"An error occurred while uploading the file: {e}")


