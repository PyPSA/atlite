  .. SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>

  .. SPDX-License-Identifier: CC-BY-4.0

======
atlite
======

|PyPI version| |Conda version| |Documentation Status| |ci| |codecov| |standard-readme compliant| |MIT-image| |reuse| |black| |pre-commit.ci| |joss| |discord| |stackoverflow|

atlite is a `free software`_, `xarray`_-based Python library for
converting weather data (like wind speeds, solar influx) into energy systems data.
It is designed to be lightweight, keeping computing resource requirements (CPU, RAM) usage low.
It is therefore well suited to be used with big weather datasets.

atlite can process the following weather data fields and can convert them into following power-system relevant time series for any subsets of a full weather database.

.. image:: doc/workflow_chart.png

atlite was initially developed by the `Renewable Energy Group
<https://fias.uni-frankfurt.de/physics/schramm/renewable-energy-system-and-network-analysis/>`_
at `FIAS <https://fias.uni-frankfurt.de/>`_ to carry out simulations
for the `CoNDyNet project <http://condynet.de/>`_, financed by the
`German Federal Ministry for Education and Research (BMBF)
<https://www.bmbf.de/en/index.html>`_ as part of the `Stromnetze
Research Initiative
<http://forschung-stromnetze.info/projekte/grundlagen-und-konzepte-fuer-effiziente-dezentrale-stromnetze/>`_.


With `atlite` we want to provide an interface between the meteorological and energy systems modelling communities.

Traditionally the MET and ESM communities have not been interacting much.
The outputs and learning of one community were only slowly adapted into the other community.

With `atlite` we want bridge between the communities:
We want to make it easy to use and integrate outputs of the MET communities into energy system models, by offering standardized ways of accessing weather/climate datasets and converting them to weather-dependent inputs for ESMs.

For major next development goals, consult our `vision and roadmap project <https://github.com/orgs/PyPSA/projects/12/views/1>`_ or check our list of possible `enhancements <https://github.com/PyPSA/atlite/issues/?q=is%3Aissue%20state%3Aopen%20label%3A%22type%3A%20enhancement%22>`_.

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

to install the most recent upstream version from GitHub

.. code:: shell

       pip install git+https://github.com/pypsa/atlite.git


Documentation
===============

Please check the `documentation <https://atlite.readthedocs.io/en/latest>`_.


Support & Contributing
======================
* In case of code-related **questions**, please post on `stack overflow <https://stackoverflow.com/questions/tagged/pypsa>`_.
* For non-programming related and more general questions please refer to the `pypsa mailing list <https://groups.google.com/group/pypsa>`_.
* To **discuss** with other PyPSA and atlite users, organise projects, share news, and get in touch with the community you can use the `discord server <https://discord.gg/JTdvaEBb>`_.
* For **bugs and feature requests**, please use the `issue tracker <https://github.com/PyPSA/atlite/issues>`_.
* We strongly welcome anyone interested in providing **contributions** to this project. If you have any ideas, suggestions or encounter problems, feel invited to file issues or make pull requests on the `Github repository <https://github.com/PyPSA/atlite>`_.

Authors and Copyright
---------------------

Copyright (C) Contributors to atlite <https://github.com/pypsa/atlite>

See the `AUTHORS`_ for details.

Licence
=======

|MIT-image|

This work is licensed under multiple licences:

-  All original source code is licensed under `MIT`_
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

.. _MIT: LICENSES/MIT.txt
.. _BSD-2-Clause: LICENSES/BSD-2-Clause.txt
.. _CC-BY-4.0: LICENSES/CC-BY-4.0.txt
.. _CC0-1.0: LICENSES/CC0-1.0.txt

.. |PyPI version| image:: https://img.shields.io/pypi/v/atlite.svg
   :target: https://pypi.python.org/pypi/atlite
.. |Conda version| image:: https://img.shields.io/conda/vn/conda-forge/atlite.svg
   :target: https://anaconda.org/conda-forge/atlite
.. |Documentation Status| image:: https://readthedocs.org/projects/atlite/badge/?version=master
   :target: https://atlite.readthedocs.io/en/master/?badge=master
.. |standard-readme compliant| image:: https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat
   :target: https://github.com/RichardLitt/standard-readme
.. |MIT-image| image:: https://img.shields.io/pypi/l/atlite.svg
   :target: LICENSES/MIT.txt
.. |codecov| image:: https://codecov.io/gh/PyPSA/atlite/branch/master/graph/badge.svg?token=TEJ16CMIHJ
   :target: https://codecov.io/gh/PyPSA/atlite
.. |ci| image:: https://github.com/PyPSA/atlite/actions/workflows/test.yaml/badge.svg
   :target: https://github.com/PyPSA/atlite/actions/workflows/test.yaml
.. |reuse| image:: https://api.reuse.software/badge/github.com/pypsa/atlite
   :target: https://api.reuse.software/info/github.com/pypsa/atlite
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black
.. |pre-commit.ci| image:: https://results.pre-commit.ci/badge/github/PyPSA/atlite/master.svg
   :target: https://results.pre-commit.ci/latest/github/PyPSA/atlite/master
   :alt: pre-commit.ci status
.. |joss| image:: https://joss.theoj.org/papers/10.21105/joss.03294/status.svg
   :target: https://doi.org/10.21105/joss.03294
.. |discord| image:: https://img.shields.io/discord/911692131440148490?logo=discord
   :target: https://discord.gg/AnuJBk23FU
.. |stackoverflow| image:: https://img.shields.io/stackexchange/stackoverflow/t/pypsa
   :target: https://stackoverflow.com/questions/tagged/pypsa
   :alt: Stackoverflow
