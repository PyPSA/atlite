#########################
Basics to get you running
#########################


TODO
====

It can process the following weather data fields:

* Temperature
* Downward short-wave radiation
* Upward short-wave radiation
* Wind 
* Runoff
* Surface roughness
* Height maps
* Soil temperature

The following power-system relevant time series can be produced for
all possible spatial distributions of assets:

* Wind power generation for a given turbine type
* Solar PV power generation for a given panel type
* Solar thermal collector heat output
* Hydroelectric inflow (simplified)
* Heating demand (based on the degree-day approximation)

Setting up datasources
======================

ERA5
----
ERA5 is downloaded automatically on-demand after the 
`ECMWF ADS API <https://cds.climate.copernicus.eu/api-how-to>`_
(European Centre for Medium-Range Weather Forecasts Climate Data Store
Application Program Interface) client is properly installed. See separate,
linked installation guide for details, especially for correctly setting up
your CDS API key.

Other data sources
------------------

Atlite was originally designed to be modular, so that it can work with
other weather datasets.

.. note::
    The current version v0.2 of Atlite uses the ERA5 dataset and
    can download the required data automatically.

Previous versions supported
* NCEP Climate Forecast System
* EURO-CORDEX Cliamte Change Projection
* CMSAF SARAH-2

Their support is currently on hold (time limitation on developer side).
If you need to process these (or other) data sources, feel free to
file an issue on our `GitHub <https://github.com/PyPSA/atlite>`_ or 
(even better) create a pull request!
