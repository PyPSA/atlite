  .. SPDX-FileCopyrightText: 2016-2021 The Atlite Authors

  .. SPDX-License-Identifier: CC-BY-4.0

======
Atlite
======

|PyPI version| |Conda version| |Documentation Status| |travis| |standard-readme compliant|

Atlite is a `free software`_, `xarray`_-based Python library for
converting weather data (like wind speeds, solar influx) into energy systems data.
It has a  lightweight design and works with big weather datasets
while keeping the resource requirements especially on CPU and RAM
resources low.


.. Atlite is designed to be modular, so that it can work with any weather
.. datasets. It currently has modules for the following datasets: 

.. * `NCEP Climate Forecast System <http://rda.ucar.edu/datasets/ds094.1/>`_ hourly
..   historical reanalysis weather data available on a 0.2 x 0.2 degree global grid
.. * `ECMWF ERA5
..   <https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation>`_ hourly
..   historical reanalysis weather data on an approximately 0.25 x 0.25 deg global
..   grid
.. * `EURO-CORDEX Climate Change Projection <http://www.euro-cordex.net/>`_
..   three-hourly up until 2100, available on a 0.11 x 0.11 degree grid for Europe
.. * `CMSAF SARAH-2
..   <https://wui.cmsaf.eu/safira/action/viewDoiDetails?acronym=SARAH_V002>`_
..   half-hourly historical surface radiation on a 0.05 x 0.05 deg grid available
..   for Europe and Africa (automatically interpolated to a 0.2 deg grid and
..   combined with ERA5 temperature).


Atlite can process the following weather data fields and can convert them into following power-system relevant time series for any subsets of a full weather data base.

.. image:: doc/workflow_chart.png

.. * Temperature
.. * Downward short-wave radiation
.. * Upward short-wave radiation
.. * Wind 
.. * Runoff
.. * Surface roughness
.. * Height maps
.. * Soil temperature


.. * Wind power generation for a given turbine type
.. * Solar PV power generation for a given panel type
.. * Solar thermal collector heat output
.. * Hydroelectric inflow (simplified)
.. * Heating demand (based on the degree-day approximation)


Atlite was initially developed by the `Renewable Energy Group
<https://fias.uni-frankfurt.de/physics/schramm/renewable-energy-system-and-network-analysis/>`_
at `FIAS <https://fias.uni-frankfurt.de/>`_ to carry out simulations
for the `CoNDyNet project <http://condynet.de/>`_, financed by the
`German Federal Ministry for Education and Research (BMBF)
<https://www.bmbf.de/en/index.html>`_ as part of the `Stromnetze
Research Initiative
<http://forschung-stromnetze.info/projekte/grundlagen-und-konzepte-fuer-effiziente-dezentrale-stromnetze/>`_.


Installation
============

To install you need a working installation running Python 3.6 or above
and we strongly recommend using either miniconda or anaconda for package
management.

To install the current stable version:

with ``conda`` from `conda-forge`_

.. code:: shell

       conda install -c conda-forge atlite

with ``pip`` from `pypi`_

.. code:: shell

       pip install atlite

to install the most recent upstream version from `GitHub`_

.. code:: shell

       pip install git+https://github.com/pypsa/atlite.git


Documentation
===============
.. * Install atlite from conda-forge or pypi.
.. * Download one of the weather datasets listed above (ERA5 is downloaded
..   automatically on-demand after the ECMWF
..   `cdsapi<https://cds.climate.copernicus.eu/api-how-to>` client is 
..   properly installed)
.. * Create a cutout, i.e. a geographical rectangle and a selection of
..   times, e.g. all hours in 2011 and 2012, to narrow down the scope -
..   see `examples/create_cutout.py <examples/create_cutout.py>`_
.. * Select a sparse matrix of the geographical points inside the cutout
..   you want to aggregate for your time series, and pass it to the
..   appropriate converter function - see `examples/ <examples/>`_


Please check the `documentation <https://atlite.readthedocs.io/en/latest>`_.

Contributing
============

If you have any ideas, suggestions or encounter problems, feel invited
to file issues or make pull requests.

Authors and Copyright
---------------------

Copyright (C) 2016-2021 The Atlite Authors.

See the `AUTHORS`_ for details.

Licence
=======

|GPL-3-or-later-image|

This work is licensed under multiple licences:

-  All original source code is licensed under `GPL-3.0-or-later`_.
-  Auxiliary code from SPHINX is licensed under `BSD-2-Clause`_.
-  The documentation is licensed under `CC-BY-4.0`_.
-  Configuration and data files are mostly licensed under `CC0-1.0`_.

See the individual files for license details.

.. _free software: http://www.gnu.org/philosophy/free-sw.en.html
.. _xarray: http://xarray.pydata.org/en/stable/

.. _conda-forge: https://anaconda.org/conda-forge/atlite
.. _pypi: https://pypi.org/project/atlite/%3E
.. _GitHub: https://github.com/pypsa/atlite

.. _documentation on getting started: https://atlite.readthedocs.io/en/latest/getting-started.html

.. _AUTHORS: AUTHORS.rst

.. _GPL-3.0-or-later: LICENSES/GPL-3.0-or-later.txt
.. _BSD-2-Clause: LICENSES/BSD-2-Clause.txt
.. _CC-BY-4.0: LICENSES/CC-BY-4.0.txt
.. _CC0-1.0: LICENSES/CC0-1.0.txt

.. |PyPI version| image:: https://img.shields.io/pypi/v/atlite.svg
   :target: https://pypi.python.org/pypi/atlite
.. |Conda version| image:: https://img.shields.io/conda/vn/conda-forge/atlite.svg
   :target: https://anaconda.org/conda-forge/atlite
.. |Documentation Status| image:: https://readthedocs.org/projects/atlite/badge/?version=latest
   :target: https://atlite.readthedocs.io/en/latest/?badge=latest
.. |standard-readme compliant| image:: https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat
   :target: https://github.com/RichardLitt/standard-readme
.. |GPL-3-or-later-image| image:: https://img.shields.io/pypi/l/atlite.svg
   :target: LICENSES/GPL-3.0-or-later.txt
.. |travis| image:: https://img.shields.io/travis/PyPSA/atlite/master.svg
    :target: https://travis-ci.org/PyPSA/atlite
    :alt: Build status
