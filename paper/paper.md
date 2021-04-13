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
  - name: Johannes Hampp
    orcid: 0000-0002-1776-116X
    affiliation: "2"
  - name: Jonas Hörsch
    orcid: 0000-0001-9438-767X
    affiliation: "3" # (Multiple affiliations must be quoted)
affiliations:
 - name: Frankfurt Institute for Advanced Studies
   index: 1
 - name: Center for international Development and Environmental Research (ZEU), Justus-Liebig University Giessen
   index: 2
 - name: Climate Analytics gGmbH, Berlin
   index: 3
date: 01 March 2021
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx #<- update this with the DOI from AAS once you know it.
aas-journal: xxxx #<- The name of the AAS journal.
---

<!-- See https://joss.readthedocs.io/en/latest/submitting.html for all details -->

<!-- Compile with:  pandoc --citeproc -s paper.md -o paper.pdf   -->

# Summary

Renewable energy sources build the backbone of the future global energy system. One important key to a successful energy transition is to rigorously analyse weather-dependent energy outputs of existent and eligible renewable resources. `atlite` is an open python software package for retrieving reanalysis weather data and converting it to potentials and time-series for renewable energy systems. Based on detailed mathematical models, it simulates the power output of wind turbines, solar photo-voltaic panels, solar-thermal collectors, run-of-river power plants and hydro-electrical dams. It further provides weather-dependant projections on the demand side like heating demand degree days and heat pump coefficients of performance.


# Statement of need



Deriving weather-based time-series and maximum capacity potentials for renewables over large regions is a common problem in energy system modelling.
Websites with exposed open APIs such as [renewables.ninja](https://www.renewables.ninja) [@pfenninger_long-term_2016,@staffell_using_2016] exist for such purpose but are difficult to use for local execution in e.g. cluster environments.
Further they expose, by design, neither the underlying datasets nor methods for deriving time-series, here referred to as conversion functions/methods. This makes them unsuited for utilizing different weather datasets or exploring alternative conversion functions.
The [pvlib](https://github.com/pvlib/pvlib-python) [@holmgren_pvlib_2018] is suited for local execution and allows exchangeable input data but is specialized to PV systems only and intended for single location modelling.
Other packages like the Danish REatlas [@andresen_validation_2015] reveal usage obstacles like proprietary code, missing documentation and a restricted flexibility regarding the input data.


The purpose of `atlite` is to fill this gap and provide an open, community-driven library. `atlite` was initially built as a light-weight alternative to REatlas, but, up to now, has evolved much further and comprises multiple additional features. `atlite` is designed with extensibility for new renewable technologies or different conversion methods in mind.
An abstraction layer for weather datasets enables flexibility for exchange of the underlying datasets.
By leveraging the Python packages [xarray](https://xarray.pydata.org/en/stable/) [@hoyer_xarray_2017],
[dask](https://docs.dask.org/en/latest/) [@dask_development_team_dask_2016] and [rasterio](https://rasterio.readthedocs.io/en/latest/), `atlite` makes use of parallelization
and memory efficient backends thus performing well even on large datasets.

# Basic Concept


The starting point of most `atlite` functionalities  is the `atlite.Cutout` class. It serves as a container for a spatio-temporal subset of one or more weather datasets. As illustrated in Figure \ref{fig:cutout}, a typical workflow consists of three steps: Cutout creation, Cutout preparation and Cutout conversion. 

![Typical working steps with `atlite`. \label{fig:cutout}](figures/workflow.png)


## Cutout Creation and Preparation


The Cutout creation requires specifications of the geographical and temporal bounds, the path of the associated `netcdf` file (which will be created) as well as the source for the weather data, referred to as the module. Optionally, temporal and spatial resolution may be adjusted, the default is set to 1 hour and 0.25$^\circ$ latitude times 0.25$^\circ$ longitude. So far, `atlite` supports three weather data sources: 

1. [ECMWF Reanalysis v5 (ERA5)](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) provides various weather-related variables in an hourly resolution from 1950 onward on a spatial grid with a 0.25$^\circ$ x 0.25$^\circ$ resolution, most of which is reanalysis data. `atlite` automatically retrieves the raw data using the [Climate Data Store (CDS) API](https://cds.climate.copernicus.eu/#!/home) which has to be properly set up by the user. When the requested data points diverge from the original grid, the API retrieves interpolated values based on the original grid data. 

2. [Heliosat (SARAH-2)](https://wui.cmsaf.eu/safira/action/viewDoiDetails?acronym=SARAH_V002) provides satellite-based solar data in a 30 min resolution from 1983 to 2015 on a spatial grid ranging from -65° to +65$^\circ$ longitude/latitude with a resolution of 0.05$^\circ$ x 0.05$^\circ$. In case of a diverging Cutout grid, a resampling function provided by `atlite` projects the data accordingly. The dataset must be downloaded by the user beforehand. 

3. [GEBCO](https://www.gebco.net/data_and_products/gridded_bathymetry_data/) is a bathymetric dataset covering terrain heights on a 15 arc-second resolved spatial grid. Using an averaging resampling method, the data is projected to the Cutout resolution. The dataset has to be downloaded by the user beforehand.

Creating a Cutout triggers the program to initialize the grid cells and the coordinate system on which the data will lay. As indicated in Figure \ref{fig:cutout}, the shapes of the weather cells are created such that their coordinates are centered in the middle. As soon as the preparation of the cutout is executed, `atlite` retrieves/loads data variables, adds them to the Cutout and finally writes the Cutout out to the associated netcdf file. 
`atlite` groups weather variables into *features*, which can be used as front-end keys for preparing a subset of the available weather variables. The following table shows the variable groups for all datasets.

+---------------+----------------------------------+-------------------------+--------------------+
|   feature     |         ERA5                     |        SARAH-2          | GEBCO              |
|               |         variables                |        variables        | variables          |
+===============+==================================+=========================+====================+
| height        | height                           |                         | height             |
+---------------+----------------------------------+-------------------------+--------------------+
| wind          | wnd100m, roughness               |                         |                    |
+---------------+----------------------------------+-------------------------+--------------------+
| influx        | influx\_toa, influx\_direct,     | influx\_direct,         |                    |
|               | influx\_diffuse, albedo          | influx\_diffuse         |                    |
+---------------+----------------------------------+-------------------------+--------------------+
| temperature   | temperature, soil\_temperature   |                         |                    |
+---------------+----------------------------------+-------------------------+--------------------+
| runoff        | runoff                           |                         |                    |
+---------------+----------------------------------+-------------------------+--------------------+


A Cutout may combine features from different sources, e.g. 'height' from GEBCO and 'runoff' from ERA5. Future versions of atlite will likely introduce the possibility to retrieve explicit weather variables from the CDS API. Further, the climate projection dataset [CORDEX](https://rcmes.jpl.nasa.gov/content/cordex) which was removed in v0.2 due to compatibility issues, is likely to be reintroduced. 


## Conversion Functions


`atlite` currently offers conversion functions for deriving time-series and static potentials from Cutouts for the following types of renewables:

* **Solar photovoltaic** --
Two alternative solar panel models are provided based on [@huld_mapping_2010] and [@beyer_robust_2004], both of which
use the clearsky model from [@reindl_diffuse_1990] and a
solar azimuth and altitude position tracking based on [@michalsky_astronomical_1988,@sproul_derivation_2007,@kalogirou_solar_2009] combined with a surface orientation algorithm following
[@sproul_derivation_2007]. Optionally, optimal latitude heuristics from [@charles_r_landau_optimum_2017] are supported.

* **Solar thermal collector** --
Low temperature heat for space or district heating are implemented based on the formulation in [@henning_comprehensive_2014] which combines average global radiation with storage losses dependent on the current outside temperature.

* **Wind turbine** -- 
The wind turbine power output is calculated from down-scaled wind speeds at hub height using either a custom power curve or on one of 16 predefined wind turbine configurations. Optionally, convolution with a Gaussian kernel for region specific calibration given real-world reference data as presented by [@andresen_validation_2015] is supported.

* **Hydro run-off power** --
A heuristic approach uses runoff weather data which is normalized to match reported energy production figures by the [EIA](https://www.eia.gov/international/data/world).
The resulting time-series are optionally weighted by height of the runoff location and time-series may be smoothed for a more realistic representation.

* **Hydro reservoir and dam power** --
Following [@liu_validated_2019] and [@lehner_global_2013] run-off data is aggregated to and collected in basins which are obtained and estimated in their size with the help of the [HydroSHEDS](https:// hydrosheds.org/) dataset.

* **Heating demand** --
Space heating demand is obtained with a simple degree-day approximation where
the difference between outside ground-level temperature and a reference temperature scaled by a linear factor yields the desired estimate.



The conversion functions are highly flexible and allow the user to calculate different types of outputs, which arise from the set of input arguments. In energy system models, network nodes are often associated with geographical regions which serve as catchment areas for electric loads, renewable energy potentials etc. As indicated in third step of Figure \ref{fig:cutout}, `atlite`'s conversion functions allow to project renewable time-series on a set of bus regions. Therefore, `atlite` internally computes the Indicator Matrix $\textbf{I}$ with values $I_{r,x,y}$ representing the per-unit overlap between bus region $r$ and the weather cell at $(x,y)$. Then, the resulting time-series $\varphi_r(t)$ for region $r$ is given by 

$$\varphi_r(t) = \sum_{x,y} I_{r,x,y} \, \varphi_{x,y}(t)$$ 


where $\varphi_{x,y}(t)$ is the converted time-series of weather cell $(x,y)$. Further, the user may define custom weightings $\lambda_{x,y}$ of the weather cells, referred to as layout, representing for instance installed capacities, which modifies the above equation to 

$$\varphi_r(t) = \sum_{x,y} I_{r,x,y} \, \lambda_{x,y} \, \varphi_{x,y}(t).$$

The conversion functions may optionally return the per-unit time-series $\tilde{\varphi}_r(t) = \varphi_r(t) / c_r$ where $c_r$ is the installed capacity per region given by 

$$c_r = \sum_{x,y} I_{r,x,y} \, \lambda_{x,y}$$ 

which may be returned as an output as well.


## Land-Use Restrictions

In the real world, renewable infrastructure is often limited by land-use restrictions. For example, wind turbines can only be placed in eligible places which have to fulfill criteria in accord with the country specific policy, e.g. non-protected areas, enough distance to residential areas etc. 

`atlite` provides a performant, parallelized implementation to calculate land-use availabilities within all weather cells of a Cutout. As illustrated in Figure \ref{fig:land-use}, the entries $A_{r,x,y}$ of the Availability Matrix $\textbf{A}$ indicate the overlap of the eligible area of region $r$ with weather cell at $(x,y)$. Note that this is analogue to the Indicator Matrix $\textbf{I}$ but with reduced area. The user can exclude geometric shapes and/or geographic rasters of arbitrary projection, like specific codes of the [Corine Land Cover (CLC)](https://land.copernicus.eu/pan-european/corine-land-cover).
The implementation is inspired by the [GLAES](https://github.com/FZJ-IEK3-VSA/glaes) [@ryberg_evaluating_2018]
software package which itself is no longer maintained and incompatible with newer versions of the underlying [GDAL](https://gdal.org/index.html) software.



![Example of a land-use restrictions calculated with `atlite`. The left side shows a highly-resolved raster with available areas in green. Excluded areas, which in this example are set to all urban and forest-like sites, are drawn in white. The right side visualizes exemplary entries per region $r$ of the resulting Availability Matrix $\textbf{A}$.  \label{fig:land-use}](figures/land-use-availability.png)



# Related Research 

So far, `atlite` is used by several research projects and groups. The [PyPSA-EUR workflow](https://github.com/PyPSA/pypsa-eur) [@horsch_pypsa-eur_2018] is an open model dataset of the European power system which exploits the full potential of `atlite`  including Cutout preparation and conversion to wind, solar and hydro reservoir time-series with restricted land-use availabilities. The sector-coupled extension [PyPSA-EUR-sec](https://github.com/PyPSA/pypsa-eur-sec) presented in [@brown_synergies_2018] calculates heat-demand profiles as well as heat pump coefficients. The [Euro Calliope](https://github.com/calliope-project/euro-calliope) studied in [@trondle_trade-offs_2020] calculates hydro reservoir time-series with `atlite`. 




# Availability

Stable versions of the `atlite` package are available for Linux, MacOS and Windows via
`pip` in the [Python Package Index (PyPI)](https://pypi.org/project/atlite/) and
for `conda` on [conda-forge](https://anaconda.org/conda-forge/atlite).
Upstream versions and development branches are available in the projects [GitHub repository](https://github.com/PyPSA/atlite).
Documentation including examples are available on [Read the Docs](https://atlite.readthedocs.io/en/latest/).
The `atlite` package is released under [GPLv3](https://github.com/PyPSA/atlite/blob/master/LICENSES/GPL-3.0-or-later.txt) and welcomes contributions via the project's [GitHub repository](https://github.com/PyPSA/atlite).


# Acknowledgements
<!-- WHO EVER WANTS -->
We thank all [contributors](https://github.com/PyPSA/atlite/graphs/contributors) which helped to develop `atlite`. The ongoing development of the package as well as this paper is realized in the frame of the EnergySysAI project funded by the German Federal Ministry for Education and Research. 

# References
