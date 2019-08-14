..
  SPDX-FileCopyrightText: 2016-2019 The Atlite Authors

  SPDX-License-Identifier: CC-BY-4.0

======
Atlite
======

|PyPI version| |Conda version| |Documentation Status| |standard-readme compliant|

   Atlite is a `free software`_, `xarray`_-based Python library for
   converting weather data (like wind speeds) into energy systems data.
   It is designed to by lightweight and work with big weather datasets
   while keeping the resource requirements especially on CPU and RAM
   resources low.

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

Getting started
===============

Please check the `documentation on getting started`_.

Contributing
============

If you have any ideas, suggestions or encounter problems, feel invited
to file issues or make pull requests.

Authors and Copyright
---------------------

Copyright (C) 2016-2019 The Atlite Authors.

.. include:: AUTHORS.rst

License
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
