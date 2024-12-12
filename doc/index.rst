..
  SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>

  SPDX-License-Identifier: CC-BY-4.0

atlite: Convert weather data to energy systems data
===================================================

.. image:: https://img.shields.io/pypi/v/atlite.svg
    :target: https://pypi.python.org/pypi/atlite
    :alt: PyPI version

.. image:: https://img.shields.io/conda/vn/conda-forge/atlite.svg
    :target: https://anaconda.org/conda-forge/atlite
    :alt: Conda version

.. image:: https://github.com/PyPSA/atlite/actions/workflows/test.yaml/badge.svg
   :target: https://github.com/PyPSA/atlite/actions/workflows/test.yaml

.. image:: https://codecov.io/gh/PyPSA/atlite/branch/master/graph/badge.svg?token=TEJ16CMIHJ
   :target: https://codecov.io/gh/PyPSA/atlite

.. image:: https://readthedocs.org/projects/atlite/badge/?version=master
    :target: https://atlite.readthedocs.io/en/latest/?badge=master
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/l/atlite.svg
    :target: License

.. image:: https://api.reuse.software/badge/github.com/pypsa/atlite
    :target: https://api.reuse.software/info/github.com/pypsa/atlite

.. image:: https://joss.theoj.org/papers/10.21105/joss.03294/status.svg
    :target: https://doi.org/10.21105/joss.03294

.. image:: https://img.shields.io/discord/911692131440148490?logo=discord
    :target: https://discord.gg/AnuJBk23FU

.. image:: https://img.shields.io/stackexchange/stackoverflow/t/pypsa
   :target: https://stackoverflow.com/questions/tagged/pypsa
   :alt: Stackoverflow

atlite is a `free software
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

The time series derived with atlite are used in energy system models
like e.g. `PyPSA-EUR <https://github.com/PyPSA/pypsa-eur>`_
or projects like `model.energy <https://model.energy/>`_.

Maintainers
===========

atlite is currently maintained by volunteers from different institutions
with no dedicated funding for developing this package.

atlite was initially developed by the `Renewable Energy Group
<https://fias.uni-frankfurt.de/physics/schramm/renewable-energy-system-and-network-analysis/>`_
at `FIAS <https://fias.uni-frankfurt.de/>`_ to carry out simulations
for the `CoNDyNet project <http://condynet.de/>`_, financed by the
`German Federal Ministry for Education and Research (BMBF)
<https://www.bmbf.de/en/index.html>`_ as part of the `Stromnetze
Research Initiative
<http://forschung-stromnetze.info/projekte/grundlagen-und-konzepte-fuer-effiziente-dezentrale-stromnetze/>`_.

Origin
======

atlite was originally conceived as a light-weight version of the Aarhus
University RE Atlas (`original publication doi:10.1016/j.energy.2015.09.071 <http://dx.doi.org/10.1016/j.energy.2015.09.071>`_).
It has since been extended to use weather datasets simulated with projected
climate change and to compute other time series, such as hydro power,
solar thermal collectors and heating demand.


Citing atlite
=============

If you would like to cite the atlite software, please refer to `this paper <https://doi.org/10.21105/joss.03294>`_ published in `JOSS <https://joss.theoj.org/>`_.



.. Documentation
.. =============

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   introduction
   installation
   conventions

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples

   examples/create_cutout.ipynb
   examples/create_cutout_SARAH.ipynb
   examples/historic-comparison-germany.ipynb
   examples/building_stock_weather_aggregation.ipynb
   examples/landuse-availability.ipynb
   examples/using_gebco_heightmap.ipynb
   examples/plotting_with_atlite.ipynb
   examples/logfiles_and_messages.ipynb
   examples/solarpv_tracking_options.ipynb
   examples/working-with-csp.ipynb
   examples/more_examples.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: References

   ref_api
   release_notes
   contributing
   support
   license

License
=======

atlite is released and licensed under the
`MIT license <https://github.com/PyPSA/atlite/blob/mit-license/LICENSES/MIT.txt>`_.
