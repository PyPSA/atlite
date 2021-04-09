---
title: 'atlite: A Lightweight Python Package for Calculating Renewable Power Potentials and Time-Series'
tags:
  - Python
  - energy systems
authors:
  - name: Fabian Hofmann
    orcid: 0000-0002-6604-5450
    affiliation: "1" # (Multiple affiliations must be quoted)
    # add your names here
affiliations:
 - name: Frankfurt Institute for Advanced Studies
   index: 1
 - name: Institution Name
   index: 2
 - name: Institution Name
   index: 3
date: 01 March 2021
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx #<- update this with the DOI from AAS once you know it.
aas-journal: xxxx #<- The name of the AAS journal.
---

<!-- See https://joss.readthedocs.io/en/latest/submitting.html for all details -->
<!-- compile with 

pandoc --citeproc -s paper.md -o paper.pdf  

 -->

# Summary
<!-- Change whatever you want -->

Renewable energy sources build the backbone of the future global energy system. For a sucessful energy transition it is crucial to rigorously analyse the weather-dependent energy outputs of eligible and existent renewable resources. atlite is an open python software package for retrieving reanalysis weather data and converting it to potentials and time series for renewable energy systems. Based on detailed mathematical models, it simulates the power output of wind turbines, solar photo-voltaic panels, solar-thermal collectors, run-of-river power plants and hydro-electrical dams. It further provides weather-dependant projections on the demand side like heating demand degree days and heat pump coefficients of performance.


# Statement of need


<!-- context of atlite -->
<!-- FABIAN NEUMANN -->
atlite was initially build as a light-weight alternative to the Danish REatlas [@andresen_validation_2015]. Downsides of REatlas.  What other packages exist?  What's their up/downside? Why is atlite necessary? 

<!-- software/packages and implementation -->
<!-- JOHANNES -->
atlite highly relies on the python packages Xarray [@hoyer_xarray_2017], [Dask](https://docs.dask.org/en/latest/) and [Rasterio](https://rasterio.readthedocs.io/en/latest/) which allows for parallelized and memory-efficient processing of large data sets. Modularity etc.


<!-- # Basic Concept -->


<!-- The first step is to create a cutout which is a container for subsets of raw weather data. The second step is to prepare the created `atlite.Cutout` that is retrieving the raw weather data and storing it to a . The second step is to convert the weather data of a `atlite.Cutout` into renewables time-series and/or static potentials using explicit conversion functions. -->





## Basic Concept -- the Cutout Class
<!-- FABIAN H -->


The `atlite.Cutout` class builds the starting point for most atlite functionalities. It serves as a container for a spatio-temporal subset of one or more weather datasets. 
A typical workflow with atlite consists of three steps, illustrated in the figure below. These consist are cutout creation, cutout preparation and cutout conversion. 

![Typical working steps with `atlite`.](figures/workflow.png)


When initializing a new Cutout, geographical and temporal bounds, the path of the created `netcdf` file as well as the source for the weather data have to be defined. Optionally, temporal and spatial resolution may be adjusted, the default is set to 1 hour and 0.25$^\circ$ latitude times 0.25$^\circ$ longitude. So far, atlite supports data handling of three source: 

1. [ECMWF Reanalysis v5 (ERA5)](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) provides various weather-related variables in an hourly resolution from 1950 onward on a spatial grid with a 0.25$^\circ$ x 0.25$^\circ$ resolution. Atlite automatically retrieves the raw data using the [Climate Data Store (CDS) API](https://cds.climate.copernicus.eu/#!/home) which has to be properly set up by the user. When the requested data points diverge from the original grid, the API retrieves averaged values based on the original grid data **(double check)**. 

2. [Heliosat (SARAH-2)](https://wui.cmsaf.eu/safira/action/viewDoiDetails?acronym=SARAH_V002) provides satellite-based solar data in a 30 min resolution from 1983 to 2015 on a spatial grid from -65° to +65$^\circ$ longitude and latitude in a 0.05$^\circ$ x 0.05$^\circ$ resolution. The dataset must be downloaded by the user beforehand. By using a regridding function provided by atlite, the data may be projected on arbitrary resolutions. 

3. [GEBCO](https://www.gebco.net/data_and_products/gridded_bathymetry_data/) is a bathymetric data set covering terrain heights on a 15 arc-second resolved spatial grid. The dataset has to be downloaded by the used beforehand. 

When initializing a Cutout, the grid cells and the coordinate system on which the data will lay are created. Only when the preparation of the cutout is executed, atlite retrieves/loads data levels, adds them to the Cutout and finally writes the Cutout out to a netcdf file. 
Atlite groups weather variables into *features*, which can be used as front-end keys for retrieving a subset of the available weather variables. The following table shows the variable groups for all datasets.


| feature     | ERA5 variables                                     | Sarah variables | Gebco variables  |
|:------------|:--------------------------------------------------|:------------|:-------------------|
| height      | height                                            |             | height             |
| wind        | wnd100m, roughness                                |             |                    |
| influx      | influx\_toa, influx\_direct, influx\_diffuse, albedo | influx\_direct,  influx\_diffuse |                    |
| temperature | temperature, soil temperature                     |             |                    |
| runoff      | runoff                                            |             |                    |


A Cutout may combine features from different sources, e.g. 'height' from GEBCO and 'runoff' from ERA5. Future versions of atlite will likely introduce the possibility to retrieve explicit weather variables from the CDS API. Further, the climate projection dataset [CORDEX](https://rcmes.jpl.nasa.gov/content/cordex) which was removed in v0.2 due to compatibility issues, is likely to be reintroduced. 


## Conversion Functions
<!-- JOHANNES -->



* solar PV: 
  * panel config [@kalogirou_solar_2009] 
  * clearsky model from [@reindl_diffuse_1990] to split downward radiation into direct and diffuse contributions (future implementations could implement more recent models like in [@lauret_bayesian_2013,@ridley_modelling_2010])
  * optimal latitude was taken from fixed values reported by [Solartilt](http://www.solarpaneltilt.com/#fixed)
  * surface orientation based on formulation in [@sproul_derivation_2007]
  * solar panel model, two models provided by [@huld_mapping_2010] and [@beyer_robust_2004], the code is highly aligned to the implementation in [RenewablesNinja](https://github.com/renewables-ninja/gsee/blob/master/gsee/pv.py)
  * Solar position (azimuth and altitude) is based on [@michalsky_astronomical_1988,@sproul_derivation_2007,@kalogirou_solar_2009] 
* solar thermal based on [@henning_comprehensive_2014]
* wind turbines [@andresen_validation_2015], different turbine models with power curves
* runoff, has no reference, directly taken from runoff data, optional weighting by height, smoothing in time and normalizing the yearly energy output to reported energy productions like [EIA](https://www.eia.gov/international/data/world).  
* hydro reservoir and dams based on [@liu_validated_2019] and [@lehner_global_2013]. Data is available at
    the [hydrosheds website](www.hydrosheds.org)
* Heat demand 

## Land-Use Availability
<!-- FABIAN HOFMANN -->

In the real world, renewable infrastructure is often limited by land-use restrictions. Wind turbines can only be placed in eligible places which, according to the country specific policy, has to fulfill certain criteria, e.g. non-protected areas, enough distance to residential areas etc. For this reason atlite provides a performant, parallelized implementation to calculate land-use availabilities within all weather cells of a cutout. The user can exclude geometric shapes and/or geographic raster of arbitrary projection, like the [Corine Land Cover (CLC)](https://land.copernicus.eu/pan-european/corine-land-cover), from the eligible area. The implementation is inspired by the python package [GLAES](https://zenodo.org/record/1122558#.YGVygGhCRTY) **(proper cite)** which is however no longer maintained and causes extensive compatibility problems with recent versions of the underlying [GDAL](https://gdal.org/index.html) software. 

    


# Related Research 
<!-- FABIAN NEUMANN -->


atlite is used b several research projects and groups. The open-source PyPSA-EUR workflow [**cite**] is a   




# Availability
<!-- WHO EVER WANTS -->

The atlite package is available via the open source package management systems *Python Package Index* and *Conda* [@anaconda] for Linux, Mac and Windows systems. atlite is released and licensed under the GPLv3. The source code at can be found on [Github](https://github.com/PyPSA/atlite). The repository has an [Continuous Intergration](https://travis-ci.org/github/PyPSA/atlite) supported by Travis. A full package [documentation](https://atlite.readthedocs.io/en/master/?badge=master) is regularly updated on Read the Docs.  


# Acknowledgements
<!-- WHO EVER WANTS -->


# References