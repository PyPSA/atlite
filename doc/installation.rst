##########################################
Getting started
##########################################

Installation
============

There are three possibilities to install atlite:

* conda::

    conda install -c conda-forge atlite


* pypi::

    pip install atlite

* or directly from GitHub for the most recent version::

    pip install git+https://github.com/PyPSA/atlite/

Get the Datasets
================

ERA5 is downloaded automatically on-demand after the `ECMWF ADS API <https://cds.climate.copernicus.eu/api-how-to>`_
(European Centre for Medium-Range Weather Forecasts Climate Data Store
Application Program Interface) client is properly installed. See separate,
linked installation guide for details, especially for correctly setting up
your CDS API key.

Other data sources
==================

Atlite was originally designed to be modular, so that it can work with
other weather datasets.
The current version v0.2 of Atlite uses the ERA5 dataset and
can download the required data automatically.

Previous versions supported also: *NCEP Climate Forecast System*,
*EURO-CORDEX Cliamte Change Projection*, *CMSAF SARAH-2*.
Their support however is currently on hold (time limitation on developer
side).

If you need to process these (or other) data sources, feel free to
file an issue on our `GitHub <https://github.com/PyPSA/atlite>`_ or (even better) create a pull request!
