# In this example we assume you have
# * Installed the Copernicus Climate Data Store 'cdsapi' package
# * Registered and setup your cdsapi key as described here https://cds.climate.copernicus.eu/api-how-to
# 
# while atlite does support other data sources, using ERA5 from the CDS
# is good for a quick and uncomplicated start.

# Register a logger
import logging
logging.basicConfig(level=logging.DEBUG)

import atlite

# Define the cutout; this will not yet trigger any major operations
cutout = atlite.Cutout("europe-2011-01.nc",
                       module="era5",
                       x=slice(-12.18798349, 41.56244222),
                       y=slice(33.56459975, 71.65648314),
                       time="2011-01"
                       )

# This is where all the work happens (this can take some time).
cutout.prepare()
