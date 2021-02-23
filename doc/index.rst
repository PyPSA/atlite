..
  SPDX-FileCopyrightText: 2016-2019 The Atlite Authors

  SPDX-License-Identifier: CC-BY-4.0

Atlite: Convert weather data to energy systems data
===================================================

.. image:: https://img.shields.io/pypi/v/atlite.svg
    :target: https://pypi.python.org/pypi/atlite
    :alt: PyPI version

.. image:: https://img.shields.io/conda/vn/conda-forge/atlite.svg
    :target: https://anaconda.org/conda-forge/atlite
    :alt: Conda version

.. .. image:: https://img.shields.io/travis/PyPSA/atlite/master.svg
..     :target: https://travis-ci.org/PyPSA/atlite
..     :alt: Build status on Linux

.. image:: https://readthedocs.org/projects/atlite/badge/?version=latest
    :target: https://atlite.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/l/atlite.svg
    :target: License

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
University RE Atlas (`original publication doi:10.1016/j.energy.2015.09.071 <http://dx.doi.org/10.1016/j.energy.2015.09.071>`_).
It has since been extended to use weather datasets simulated with projected
climate change and to compute other time series, such as hydro power,
solar thermal collectors and heating demand.


.. Documentation
.. =============

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   introduction
   installation

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples

   examples/create_cutout.ipynb
   examples/create_cutout_SARAH.ipynb
   examples/historic-comparison-germany.ipynb
   examples/landuse-availability.ipynb
   examples/using_gebco_heightmap.ipynb
   examples/plotting_with_atlite.ipynb
   examples/logfiles_and_messages.ipynb
   examples/more_examples.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: References

   ref_api
   release_notes
   contributing
   license

License
=======

Atlite is released and licensed under the 
`GPLv3 <http://www.gnu.org/licenses/gpl-3.0.en.html>`_.
