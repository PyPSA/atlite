import atlite
from math import cos, radians
import logging
import xarray as xr
from datetime import datetime

logging.basicConfig(level=logging.INFO)

limon_co_lat = 39.26719230055635
limon_co_lon = -103.69300728924804

center_lon = limon_co_lon
center_lat = limon_co_lat

# center_lat = miso_lat
# center_lon = miso_lon

# Define a roughly 30x30km box around the MISO coordinates
# 1 degree of latitude is approximately 111 km
# 1 degree of longitude varies, but at this latitude it's about 95 km
lat_offset = 250 / 111 / 2  # Half of 30km in degrees latitude
lon_offset = (
    250 / (95 * cos(radians(center_lat))) / 2
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
    path="limon_co_2023.nc",
    module="era5",
    x=slice(min_lon, max_lon),
    y=slice(min_lat, max_lat),
    time="2023-01",
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

# Save the DataFrame to a CSV file
df.to_csv("capacity_factors.csv", index=False)

# Print out the current time
current_time = datetime.now()
print(f"Post model time: {current_time}")
