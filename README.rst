========
 Atlite
========

.. image:: https://img.shields.io/pypi/v/atlite.svg
    :target: https://pypi.python.org/pypi/atlite
    :alt: PyPI version

.. image:: https://img.shields.io/conda/vn/conda-forge/atlite.svg
    :target: https://anaconda.org/conda-forge/atlite
    :alt: Conda version

.. image:: https://readthedocs.org/projects/atlite/badge/?version=latest
     :target: https://atlite.readthedocs.io/en/latest/?badge=latest
     :alt: Documentation Status

.. image:: https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square
     :target: https://github.com/RichardLitt/standard-readme
     :alt: standard-readme compliant

.. image:: https://img.shields.io/pypi/l/atlite.svg
    :target: License

Atlite is a `free software
<http://www.gnu.org/philosophy/free-sw.en.html>`_, 
`xarray <http://xarray.pydata.org/en/stable/>`_-based Python library for
converting weather data (like wind speeds) into energy systems data.
It is designed to by lightweight and work with big weather datasets while
keeping the resource requirements especially on CPU and RAM resources low.

Install
=======

To install you need a working installation running Python 3.6 or above
and we strongly recommend using either miniconda or anaconda for package
management.

To install the current stable version:

with ``conda`` from ``conda-forge <https://anaconda.org/conda-forge/atlite>``_
.. code-block:: shell
    conda install -c conda-forge atlite

with ``pip`` from ``pypi <https://pypi.org/project/atlite/>``_
.. code-block:: shell
    pip install atlite

to install the most recent upstream version from GitHub
.. code-block:: shell
    pip install git+https://github.com/pypsa/atlite.git

Getting started
===============
Please check the `documentation on getting started 
<https://atlite.readthedocs.io/en/latest/getting-started.html>`_.

Contributing
============

If you have any ideas, suggestions or encounter problems,
feel invited to file issues or make pull requests.

License
=======

Copyright (C) 2016-2019 Jonas Hörsch (FIAS/KIT/RLI), Tom Brown (FIAS/KIT)
Copyright (C) 2019 Johannes Hampp (JLU Giessen)
Copyright (C) 2016-2017 Gorm Andresen (Aarhus University),
David Schlachtberger (FIAS), Markus Schlott (FIAS)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either 
`version 3 of the License <LICENSE.txt>`_, or (at your option) any 
later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
`GNU General Public License <LICENSE.txt>`_
along with this program.  If not, see <https://www.gnu.org/licenses/>.
