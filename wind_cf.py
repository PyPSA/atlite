import atlite
from math import cos, radians
import logging
import xarray as xr
from datetime import datetime

logging.basicConfig(level=logging.INFO)

limon_co_lat = 39.26719230055635
limon_co_lon = -103.69300728924804

# much windier spot...
ne_wind_lat = 42.0
ne_wind_lon = -102.7

center_lon = limon_co_lon
center_lat = limon_co_lat

center_lat = ne_wind_lat
center_lon = ne_wind_lon

# Define a roughly 30x30km box around the MISO coordinates
# 1 degree of latitude is approximately 111 km
# 1 degree of longitude varies, but at this latitude it's about 95 km
lat_offset = 100 / 111 / 2  # Half of 30km in degrees latitude
lon_offset = (
    100 / (95 * cos(radians(center_lat))) / 2
)  # Half of 30km in degrees longitude

min_lat = center_lat - lat_offset
max_lat = center_lat + lat_offset
min_lon = center_lon - lon_offset
max_lon = center_lon + lon_offset

print(f"Bounding box coordinates:")
print(f"Min Latitude: {min_lat:.6f}")
print(f"Max Latitude: {max_lat:.6f}")
print(f"Min Longitude: {min_lon:.6f}")
print(f"Max Longitude: {max_lon:.6f}")

cutout = atlite.Cutout(
    # path="limon_co_wind.nc",
    path="ne_wind.nc",
    module="era5",
    x=slice(min_lon + 180, max_lon + 180),  # convert to 360 here, avoid differences between fetching & existing .nc paths
    y=slice(min_lat, max_lat),
    time="2023",
    dt="h",
)
cutout.prepare(show_progress=True)

# Print out the current time
current_time = datetime.now()
print(f"Post prepare time: {current_time}")

cap_factors = cutout.wind(
    turbine="NREL_ReferenceTurbine_2020ATB_5.5MW", capacity_factor_timeseries=True
)

print(cap_factors)

# Convert the DataArray to a DataFrame
df = cap_factors.to_dataframe().reset_index()

# Preview the DataFrame
print(df.head())

# get just a single time series from the grid area 
df_filtered = df[(df['lon'] == 76.75) & (df['lat'] == 41.75)]
print(df_filtered.head())

# Save the DataFrame to a CSV file
df_filtered.to_csv("capacity_factors.csv", index=False)

# Print out the current time
current_time = datetime.now()
print(f"Post model time: {current_time}")
