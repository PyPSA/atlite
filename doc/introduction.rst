##########################################
Atlite
##########################################

Atlite is a `free software
<http://www.gnu.org/philosophy/free-sw.en.html>`_, `xarray
<http://xarray.pydata.org/en/stable/>`_-based Python library for
converting weather data (such as wind speeds, solar radiation,
temperature and runoff) into power systems data (such as wind
power, solar power, hydro power and heating demand time series).
It is designed to work with big datasets as is common with weather
reanalysis, while maintaining low computational requirements.

The spatial and time resolution of the obtainable power time series
depends on the resolutions of the original weather reanalysis dataset.
E.g. using our recommended dataset `ERA5 <https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5>`_
we can obtain time-series with hourly resolution on a 30 km x 30 km
grid.

The time series derived with Atlite are used in energy system models
like e.g. `PyPSA-EUR <https://github.com/PyPSA/pypsa-eur>`_
or `model.energy <https://model.energy/>`_.

Maintainers
===========

Atlite is currently maintained by volunteers from different institutions
with no dedicated funding for developing this package.
(TODO check!)

Atlite was initially developed by the `Renewable Energy Group
<https://fias.uni-frankfurt.de/physics/schramm/renewable-energy-system-and-network-analysis/>`_
at `FIAS <https://fias.uni-frankfurt.de/>`_ to carry out simulations
for the `CoNDyNet project <http://condynet.de/>`_, financed by the
`German Federal Ministry for Education and Research (BMBF)
<https://www.bmbf.de/en/index.html>`_ as part of the `Stromnetze
Research Initiative
<http://forschung-stromnetze.info/projekte/grundlagen-und-konzepte-fuer-effiziente-dezentrale-stromnetze/>`_.

Origin
======

Atlite was originally conceived as a light-weight version of the Aarhus
University RE Atlas (`doi:10.1016/j.energy.2015.09.071 <http://dx.doi.org/10.1016/j.energy.2015.09.071>`_).
It has since been extended to use weather datasets simulated with projected
climate change and to compute other time series, such as hydro power,
solar thermal collectors and heating demand.

License
=======

Atlite is released and licensed under the 
`GPLv3 <http://www.gnu.org/licenses/gpl-3.0.en.html>`_.

Contributions and Copyrights
============================

+--------------------+----------------------+----------------------+
| Copyright years    | Name                 | Affiliation          |
+====================+======================+======================+
| 2016-2019          | Jonas Hörsch         | * FIAS Frankfurt     |
|                    |                      | * KIT Karlsruhe      |
|                    |                      | * RLI Berlin         |
+--------------------+----------------------+----------------------+
| 2016-2019          | Tom Brown            | * FIAS Frankfurt     |
|                    |                      | * KIT Karlsruhe      |
+--------------------+----------------------+----------------------+
| 2019               | Johannes Hampp       | University Giessen   |
+--------------------+----------------------+----------------------+
| 2016-2017          | Gorm Andresen        | Aarhus University    |
+--------------------+----------------------+----------------------+
| 2016-2017          | David Schlachtberger | FIAS Frankfurt       |
+--------------------+----------------------+----------------------+
| 2016-2017          | Markus Schlott       | FIAS Frankfurt       |
+--------------------+----------------------+----------------------+
