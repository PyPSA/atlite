---
title: 'Atlite: A Lightweight Python Package for Calculating Renewable Power Potentials and Time-Series'
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

Renewable energy sources build the backbone for future energy systems. The urgent need to reduce green house gas emissions and the improving cost-efficiency of renewable technologies, pave the way for the governments and societies to invest into a fast and ongoing decarbonization. Yet, a renewable energy system is highly dependent on the weather, on the production side with technologies like solar PV or wind turbines, as well as on the demand side through temperature regulation. Atlite is an open python software package for retrieving reanalysis weather data and converting it to time series and potentials for energy systems. Based on detailed mathematical models, it simulates the power output of wind turbines, solar photo-voltaic panels, solar-thermal collectors, run-of-river power plants and hydro-electrical dams as well as heating demand degree days and heat pump coefficients of performance.


# Statement of need


<!-- context of atlite -->
<!-- FABIAN NEUMANN -->
Atlite was initially build as a light-weight alternative to the Danish REatlas [@andresen_validation_2015]. Downsides of REatlas.  What other packages exist?  What's their up/downside? Why is atlite necessary? 

<!-- software/packages and implementation -->
<!-- JOHANNES -->
Atlite highly relies on the python packages Xarray [@hoyer_xarray_2017], [Dask](https://docs.dask.org/en/latest/) and [Rasterio](https://rasterio.readthedocs.io/en/latest/) which allows for parallelized and memory-efficient processing of large data sets. Modularity etc.


# Basic Concept

<!-- Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->



## The Cutout Class
<!-- FABIAN H -->

The Cutout class builds the backbone for most atlite functionalities. 
It is associated with a temporal and spatial subset of a weather dataset and a `netcdf` file where the corresponding subset is stored. 
When initializing a new cutout, geographical and temporal bounds, `netcdf` file as well as the source for the weather data have to be defined. ...


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

## Land Use Availability
<!-- FABIAN HOFMANN -->


# Related Research 
<!-- FABIAN NEUMANN -->

pypsa-eur, pypsa-eur-sec, others? 




# Availability
<!-- WHO EVER WANTS -->

The Atlite package is available via the open source package management systems *Python Package Index* and *Conda* [@anaconda], available and tested for Linux, Mac and Windows systems. Atlite is released and licensed under the GPLv3. Source code at [Github](https://github.com/PyPSA/atlite). CI, docs ...


# Acknowledgements
<!-- WHO EVER WANTS -->


# References