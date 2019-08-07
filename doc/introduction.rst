##########################################
Introduction
##########################################

Who is Atlite for
=================

Atlite is for energy system modellers and other people interested
in converting weather data into (power) generation or capacity factors
from e.g. wind or solar.
Atlite does not provide and graphical user interface (GUI) and relies
on prior knowledge on working with Python commands.

What Atlite does
================

Atlite take weather datasets as input and convertes them into
(electric) power generation time-series for e.g. wind turbines
or photovoltaic to be used in energy system models.

Possible conversions from weather data to energy systems data are:

* Wind power generation: Using predefined or custom turbine properties
  and smoothing options for modelling more realistic results.
  *New:* Turbines can also be imported from the 
  `Open Energy Database <https://openenergy-platform.org/dataedit/view/supply/turbine_library>`_.
* Solar (PV) power generation: Using predefined or custom panel properties.
* Solar (thermal) heat generation from solar collectors.
* Hydro (run-off) power generation.
* Heating demand (based on degree-day approx.).

Atlite takes as input weather datasets, e.g. from reanalysis or from climate
forecasts.
The standard data source we currently employ is ECMWF's ERA5 dataset
(reanalysis weather data in a ca. 30 km x 30 km and hourly resolution).
This dataset is easily available at no additional costs and requires only
minimal setup from the user in comparison to other datasets.

Previously and in the future other datasets where and (hopefully) will 
again be usable, including

* Other reanalysis datasets.
* Satellite based radiation observations, e.g. SARAH-2.
* Weather data forecasts from climate models.

What Atlite not does
====================

Atlite does not provide exact prediction of the time-series generation
at high resolution in a future point in time.
The spatial resolution of the results is limited by the input data used.
The accuracy of the results is in parts limited by the methodologies used
for translating weather data into generation and the underlying assumptions.
With the current assumptions Atlite is not suited for predicting the output
of single wind turbines or solar panels.

As the results of Atlite are theoretical and are not validated per se,
and while usually a good approximation, can deviate significantly from
reality in some cases.
While in the past and also at the moment datasets generate by packages similar
to Atlite where commonly used without a comparison and validation with
reality, there is currently a trend validate the datasets before using them
to make sure that results are atleast plausible.
The Atlite team is planning to include in the future auxiliary functions which
help to validate generated datasets.
