Atlite: Convert weather data to energy systems data
===================================================

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
or projects like `model.energy <https://model.energy/>`_.

Maintainers
===========

Atlite is currently maintained by volunteers from different institutions
with no dedicated funding for developing this package.

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

Documentation
=============

**Getting Started**

* :doc:`introduction`
* :doc:`installation`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   introduction
   installation
   configuration

**References**
   
* :doc:`release_notes`
* :doc:`contributing`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: References

   release_notes
   contributing
   license

License
=======

Atlite is released and licensed under the 
`GPLv3 <http://www.gnu.org/licenses/gpl-3.0.en.html>`_.
