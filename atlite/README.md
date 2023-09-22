# atlite-cerra
An attempt to introduce CERRA data to PyPSA.

The atlite repository was cloned from the atlite github. 
Following 'era5.py', a new "cerra" module was created in the dataset of atlite 
which is also included at '__init__.py'. The 'cerra.py' script is a copy of 'era.py'.
The new variables only for wind are introduced and all of the rest technology related code is commented out. 

- "wdir10", wind direction at 10m
- "si10", wind spead at 10m       
- "sr", surface roughness

and everything else in the code has been left intact. These variables are used to create the 
features, the same way this is done in 'era.py', keeping the names of the features and variables the same.

Regarding PyPSA, the new CERRA raw data were manually downloaded from [CERRA sub-daily regional reanalysis data for Europe on single levels from 1984 to present](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-cerra-single-levels?tab=form). Only the first week of March-2018 was downloaded
for time saving. Here is the [documentation](https://confluence.ecmwf.int/display/CKB/Copernicus+European+Regional+ReAnalysis+%28CERRA%29%3A+product+user+guide) 
of CERRA. The raw data named ***"europe-03-2018-cerra.nc"*** is stored at the 'cutout' folder of pypsa-eur repository.

> [!NOTE]
> Ignore all of the code lines referring to _wave technology_. They are irrelevant to the current task.
