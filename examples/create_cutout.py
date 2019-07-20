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

# Setup atlite to use the local directory for
# storing the cutout reanalysis data
atlite.config.cutout_dir = "./"

# Define the cutout; this will not yet trigger any major operations
cutout = atlite.Cutout(name="europe-2011-01",
                       module="era5",
                       xs=slice(-12.18798349, 41.56244222),
                       ys=slice(71.65648314, 33.56459975),
                       years=slice(2011, 2011),
                       months=slice(1,1))

# This is where all the work happens
cutout.prepare()
